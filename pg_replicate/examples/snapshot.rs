use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    hash::{Hash, Hasher},
    io::{stdin, BufRead},
};

use arrow_array::{
    builder::{ArrayBuilder, GenericByteBuilder, Int32Builder, TimestampMicrosecondBuilder},
    types::Utf8Type,
    Int32Array, RecordBatch, TimestampMicrosecondArray,
};
use chrono::{DateTime, NaiveDateTime, Utc};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use pg_replicate::{Attribute, Table, TableSchema};
use serde::Deserialize;
use serde_json::Value;
use tokio_postgres::types::Type;

#[derive(Deserialize, Debug)]
pub struct Event {
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub relation_id: u32,
    pub data: Value,
}

struct Row<'a> {
    schema: &'a TableSchema,
    data: &'a Value,
}

impl<'a> Hash for Row<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for attr in &self.schema.attributes {
            if attr.identity {
                if let Some(value) = self.data.get(&attr.name) {
                    HashedValue { value }.hash(state)
                }
            }
        }
    }
}

impl<'a> PartialEq for Row<'a> {
    fn eq(&self, other: &Self) -> bool {
        for attr in &self.schema.attributes {
            if attr.identity && self.data.get(&attr.name) != other.data.get(&attr.name) {
                return false;
            }
        }
        true
    }
}

impl<'a> Eq for Row<'a> {}

#[derive(Eq)]
struct HashedValue<'a> {
    value: &'a Value,
}

impl<'a> Hash for HashedValue<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.value {
            Value::Null => {}
            Value::Bool(b) => b.hash(state),
            Value::Number(n) => n.hash(state),
            Value::String(s) => s.hash(state),
            Value::Array(a) => {
                for i in a {
                    HashedValue { value: i }.hash(state)
                }
            }
            Value::Object(o) => {
                for (k, v) in o {
                    k.hash(state);
                    HashedValue { value: v }.hash(state)
                }
            }
        }
    }
}

impl<'a> PartialEq for HashedValue<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self.value, other.value) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(b1), Value::Bool(b2)) => b1 == b2,
            (Value::Number(n1), Value::Number(n2)) => n1 == n2,
            (Value::String(s1), Value::String(s2)) => s1 == s2,
            (Value::Array(a1), Value::Array(a2)) => a1 == a2,
            (Value::Object(o1), Value::Object(o2)) => o1 == o2,
            _ => false,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let stdin = stdin();

    let mut schemas = Vec::new();
    let mut events = Vec::new();
    const PROCESS_EVENTS: usize = 204;

    for (i, line) in stdin.lock().lines().enumerate() {
        let line = &line?;
        let v: Event = serde_json::from_str(line)?;
        if v.event_type == "schema" {
            let schema = event_data_to_schema(&v.data, v.relation_id);
            schemas.push(schema);
        } else {
            events.push(v);
        }
        if i + 1 >= PROCESS_EVENTS {
            break;
        }
    }

    let mut relation_id_to_table_schema = HashMap::new();
    for schema in &schemas {
        relation_id_to_table_schema.insert(schema.relation_id, schema);
    }

    let mut column_builders = create_column_builders(&schemas);
    let mut rows = HashMap::new();

    for event in &events {
        if event.event_type == "insert" || event.event_type == "update" {
            let relation_id = event.relation_id;
            let schema = relation_id_to_table_schema
                .get(&relation_id)
                .expect("missing schema for relation_id");
            let row_key = Row {
                schema,
                data: &event.data,
            };
            let row_value = Row {
                schema,
                data: &event.data,
            };
            rows.insert(row_key, row_value);
        } else if event.event_type == "delete" {
            let relation_id = event.relation_id;
            let schema = relation_id_to_table_schema
                .get(&relation_id)
                .expect("missing schema for relation_id");
            let row_key = Row {
                schema,
                data: &event.data,
            };
            rows.remove(&row_key);
        }
    }

    for row in rows.values() {
        let builders = column_builders
            .get_mut(&row.schema.table)
            .expect("no builder found");
        insert_in_col(&row.schema.attributes, builders, row.data);
    }

    for (table, builders) in column_builders {
        let batch = create_record_batch(builders);
        write_parquet(batch, &format!("./{}.{}.parquet", table.schema, table.name));
    }

    Ok(())
}

fn event_data_to_schema(val: &Value, relation_id: u32) -> TableSchema {
    let obj = val.as_object().expect("expected schema to be an object");
    let schema = obj
        .get("schema")
        .expect("missing schema key")
        .as_str()
        .expect("schema is not str")
        .to_string();
    let name = obj
        .get("table")
        .expect("missing table key")
        .as_str()
        .expect("table is not str")
        .to_string();
    let table = Table { schema, name };

    let attrs = obj
        .get("attrs")
        .expect("missing attrs key")
        .as_array()
        .expect("attrs is not array");
    let attributes = attrs
        .iter()
        .map(|attr| {
            let obj = attr.as_object().expect("expected attr to be an object");
            let name = obj
                .get("name")
                .expect("missing name key")
                .as_str()
                .expect("name is not str")
                .to_string();
            let oid = obj
                .get("type_oid")
                .expect("missing type_oid key")
                .as_number()
                .expect("type_oid is not number")
                .as_u64()
                .expect("type_oid is not u64") as u32;
            let typ = Type::from_oid(oid).expect("invalid oid");
            let type_modifier = obj
                .get("type_modifier")
                .expect("missing type_modifier key")
                .as_number()
                .expect("type_modifier is not number")
                .as_i64()
                .expect("type_modifier is not i64") as i32;
            let nullable = obj
                .get("nullable")
                .expect("missing nullable key")
                .as_bool()
                .expect("nullable is not bool");
            let identity = obj
                .get("identity")
                .expect("missing identity key")
                .as_bool()
                .expect("identity is not bool");
            Attribute {
                name,
                typ,
                nullable,
                type_modifier,
                identity,
            }
        })
        .collect();
    TableSchema {
        table,
        relation_id,
        attributes,
    }
}

fn create_record_batch(builders: Vec<(String, Box<dyn ArrayBuilder>)>) -> RecordBatch {
    let mut cols = vec![];
    for (name, mut builder) in builders {
        cols.push((name, builder.finish()))
    }

    RecordBatch::try_from_iter(cols).expect("failed to create record batch")
}

fn write_parquet(batch: RecordBatch, file_name: &str) {
    let file = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(file_name)
        .unwrap();

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();

    writer.write(&batch).expect("Writing batch");

    writer.close().unwrap();
}

fn insert_in_col(
    attributes: &[Attribute],
    builders: &mut [(String, Box<dyn ArrayBuilder>)],
    data: &Value,
) {
    for (i, attr) in attributes.iter().enumerate() {
        match attr.typ {
            Type::INT4 => {
                let col_builder = builders[i]
                    .1
                    .as_any_mut()
                    .downcast_mut::<Int32Builder>()
                    .expect("builder of incorrect type");

                let val = if attr.nullable {
                    data.get(&attr.name)
                } else {
                    Some(data.get(&attr.name).expect("missing attribute"))
                }
                .map(|d| d.as_i64().expect("attribute not i64") as i32);

                col_builder.append_option(val);
            }
            Type::VARCHAR => {
                let col_builder = builders[i]
                    .1
                    .as_any_mut()
                    .downcast_mut::<GenericByteBuilder<Utf8Type>>()
                    .expect("builder of incorrect type");
                let val = if attr.nullable {
                    data.get(&attr.name)
                } else {
                    Some(data.get(&attr.name).expect("missing attribute"))
                }
                .map(|d| d.as_str().expect("attribute not str"));
                col_builder.append_option(val);
            }
            Type::TIMESTAMP => {
                let col_builder = builders[i]
                    .1
                    .as_any_mut()
                    .downcast_mut::<TimestampMicrosecondBuilder>()
                    .expect("builder of incorrect type");
                let val = if attr.nullable {
                    data.get(&attr.name)
                } else {
                    Some(data.get(&attr.name).expect("missing attribute"))
                }
                .map(|d| {
                    let d = d.as_str().expect("attribute not str");
                    let d: NaiveDateTime = d.parse().expect("failed to parse datetime");
                    d.and_utc().timestamp_micros()
                });
                col_builder.append_option(val);
            }
            ref t => panic!("type {t:?} not yet supported"),
        };
    }
}

type ColumnBuilders = HashMap<Table, Vec<(String, Box<dyn ArrayBuilder>)>>;

fn create_column_builders(schemas: &[TableSchema]) -> ColumnBuilders {
    let mut table_to_col_builders = HashMap::new();

    for schema in schemas {
        table_to_col_builders.insert(
            schema.table.clone(),
            create_column_builders_for_table(&schema.attributes),
        );
    }

    table_to_col_builders
}

fn create_column_builders_for_table(
    attributes: &[Attribute],
) -> Vec<(String, Box<dyn ArrayBuilder>)> {
    let mut col_builders: Vec<(String, Box<dyn ArrayBuilder>)> = Vec::new();
    for attr in attributes {
        match attr.typ {
            Type::INT4 => {
                col_builders.push((attr.name.clone(), Box::new(Int32Array::builder(128))))
            }
            Type::VARCHAR => col_builders.push((
                attr.name.clone(),
                Box::new(GenericByteBuilder::<Utf8Type>::new()),
            )),
            Type::TIMESTAMP => col_builders.push((
                attr.name.clone(),
                Box::new(TimestampMicrosecondArray::builder(128)),
            )),
            ref t => panic!("type {t:?} not yet supported"),
        }
    }
    col_builders
}
