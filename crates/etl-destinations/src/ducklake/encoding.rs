use std::sync::Arc;

use chrono::{NaiveDate, NaiveTime, Timelike};
use duckdb::{
    arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array,
            Int16Array, Int32Array, Int64Array, StringArray, Time64MicrosecondArray,
            TimestampMicrosecondArray, UInt64Array,
        },
        datatypes::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit},
        record_batch::RecordBatch,
    },
    types::{TimeUnit, Value},
};
use etl::{
    data::{ArrayCell, Cell, TableRow},
    error::{ErrorKind, EtlResult},
    etl_error,
    schema::{ColumnSchema, ReplicatedTableSchema, Type, is_array_type},
};
use pg_escape::quote_literal;

use crate::ducklake::DUCKLAKE_COLUMN_NAME_MAPPING;

/// Prepared row payload reused across retry attempts.
pub(super) enum PreparedRows {
    Appender(Vec<Vec<Value>>),
    ArrowRecordBatch(RecordBatch),
    SqlLiterals(Vec<String>),
}

/// Converts table-copy rows into a retryable payload for DuckDB writes.
pub(super) fn prepare_copy_rows(
    replicated_table_schema: &ReplicatedTableSchema,
    table_rows: Vec<TableRow>,
) -> EtlResult<PreparedRows> {
    let Some(column_kinds) = arrow_column_kinds(replicated_table_schema) else {
        return Ok(prepare_rows(table_rows));
    };

    if !copy_rows_match_arrow_kinds(&column_kinds, &table_rows) {
        return Ok(prepare_rows(table_rows));
    }

    Ok(PreparedRows::ArrowRecordBatch(copy_rows_to_arrow_record_batch(
        replicated_table_schema,
        column_kinds,
        table_rows,
    )?))
}

/// Converts table rows into a retryable payload for DuckDB writes.
pub(super) fn prepare_rows(table_rows: Vec<TableRow>) -> PreparedRows {
    if table_rows.iter().any(|row| row.values().iter().any(cell_requires_sql_literals)) {
        return PreparedRows::SqlLiterals(
            table_rows.into_iter().map(table_row_to_sql_literal).collect(),
        );
    }

    PreparedRows::Appender(
        table_rows
            .into_iter()
            .map(|row| row.into_values().into_iter().map(cell_to_value).collect())
            .collect(),
    )
}

/// Arrow column shape used for COPY staging.
#[derive(Clone, Copy)]
enum ArrowColumnKind {
    Bool,
    I16,
    I32,
    I64,
    U64,
    F32,
    F64,
    Utf8,
    Date32,
    Time64Microsecond,
    TimestampMicrosecond,
    TimestampTzMicrosecond,
    Binary,
}

impl ArrowColumnKind {
    /// Returns the Arrow type represented by this column kind.
    fn data_type(self) -> DataType {
        match self {
            Self::Bool => DataType::Boolean,
            Self::I16 => DataType::Int16,
            Self::I32 => DataType::Int32,
            Self::I64 => DataType::Int64,
            Self::U64 => DataType::UInt64,
            Self::F32 => DataType::Float32,
            Self::F64 => DataType::Float64,
            Self::Utf8 => DataType::Utf8,
            Self::Date32 => DataType::Date32,
            Self::Time64Microsecond => DataType::Time64(ArrowTimeUnit::Microsecond),
            Self::TimestampMicrosecond => DataType::Timestamp(ArrowTimeUnit::Microsecond, None),
            Self::TimestampTzMicrosecond => {
                DataType::Timestamp(ArrowTimeUnit::Microsecond, Some("+00:00".into()))
            }
            Self::Binary => DataType::Binary,
        }
    }
}

/// Mutable Arrow column data accumulated from copied rows.
enum ArrowColumnValues {
    Bool(Vec<Option<bool>>),
    I16(Vec<Option<i16>>),
    I32(Vec<Option<i32>>),
    I64(Vec<Option<i64>>),
    U64(Vec<Option<u64>>),
    F32(Vec<Option<f32>>),
    F64(Vec<Option<f64>>),
    Utf8(Vec<Option<String>>),
    Date32(Vec<Option<i32>>),
    Time64Microsecond(Vec<Option<i64>>),
    TimestampMicrosecond(Vec<Option<i64>>),
    TimestampTzMicrosecond(Vec<Option<i64>>),
    Binary(Vec<Option<Vec<u8>>>),
}

impl ArrowColumnValues {
    /// Creates an empty column value accumulator for one Arrow kind.
    fn with_capacity(kind: ArrowColumnKind, capacity: usize) -> Self {
        match kind {
            ArrowColumnKind::Bool => Self::Bool(Vec::with_capacity(capacity)),
            ArrowColumnKind::I16 => Self::I16(Vec::with_capacity(capacity)),
            ArrowColumnKind::I32 => Self::I32(Vec::with_capacity(capacity)),
            ArrowColumnKind::I64 => Self::I64(Vec::with_capacity(capacity)),
            ArrowColumnKind::U64 => Self::U64(Vec::with_capacity(capacity)),
            ArrowColumnKind::F32 => Self::F32(Vec::with_capacity(capacity)),
            ArrowColumnKind::F64 => Self::F64(Vec::with_capacity(capacity)),
            ArrowColumnKind::Utf8 => Self::Utf8(Vec::with_capacity(capacity)),
            ArrowColumnKind::Date32 => Self::Date32(Vec::with_capacity(capacity)),
            ArrowColumnKind::Time64Microsecond => {
                Self::Time64Microsecond(Vec::with_capacity(capacity))
            }
            ArrowColumnKind::TimestampMicrosecond => {
                Self::TimestampMicrosecond(Vec::with_capacity(capacity))
            }
            ArrowColumnKind::TimestampTzMicrosecond => {
                Self::TimestampTzMicrosecond(Vec::with_capacity(capacity))
            }
            ArrowColumnKind::Binary => Self::Binary(Vec::with_capacity(capacity)),
        }
    }

    /// Pushes one cell into the accumulator.
    fn push_cell(&mut self, column_name: &str, cell: Cell) -> EtlResult<()> {
        match (self, cell) {
            (Self::Bool(values), Cell::Null) => values.push(None),
            (Self::Bool(values), Cell::Bool(value)) => values.push(Some(value)),
            (Self::I16(values), Cell::Null) => values.push(None),
            (Self::I16(values), Cell::I16(value)) => values.push(Some(value)),
            (Self::I32(values), Cell::Null) => values.push(None),
            (Self::I32(values), Cell::I32(value)) => values.push(Some(value)),
            (Self::I64(values), Cell::Null) => values.push(None),
            (Self::I64(values), Cell::I64(value)) => values.push(Some(value)),
            (Self::U64(values), Cell::Null) => values.push(None),
            (Self::U64(values), Cell::U32(value)) => values.push(Some(u64::from(value))),
            (Self::F32(values), Cell::Null) => values.push(None),
            (Self::F32(values), Cell::F32(value)) => values.push(Some(value)),
            (Self::F64(values), Cell::Null) => values.push(None),
            (Self::F64(values), Cell::F64(value)) => values.push(Some(value)),
            (Self::Utf8(values), Cell::Null) => values.push(None),
            (Self::Utf8(values), Cell::String(value)) => values.push(Some(value)),
            (Self::Utf8(values), Cell::Numeric(value)) => values.push(Some(value.to_string())),
            (Self::Utf8(values), Cell::TimeTz(value)) => values.push(Some(value.to_string())),
            (Self::Utf8(values), Cell::Uuid(value)) => values.push(Some(value.to_string())),
            (Self::Utf8(values), Cell::Json(value)) => values.push(Some(value.to_string())),
            (Self::Date32(values), Cell::Null) => values.push(None),
            (Self::Date32(values), Cell::Date(value)) => {
                values.push(Some(date_days_since_epoch(value, column_name)?));
            }
            (Self::Time64Microsecond(values), Cell::Null) => values.push(None),
            (Self::Time64Microsecond(values), Cell::Time(value)) => {
                values.push(Some(time_micros_since_midnight(value)));
            }
            (Self::TimestampMicrosecond(values), Cell::Null) => values.push(None),
            (Self::TimestampMicrosecond(values), Cell::Timestamp(value)) => {
                values.push(Some(value.and_utc().timestamp_micros()));
            }
            (Self::TimestampTzMicrosecond(values), Cell::Null) => values.push(None),
            (Self::TimestampTzMicrosecond(values), Cell::TimestampTz(value)) => {
                values.push(Some(value.timestamp_micros()));
            }
            (Self::Binary(values), Cell::Null) => values.push(None),
            (Self::Binary(values), Cell::Bytes(value)) => values.push(Some(value)),
            _ => {
                return Err(etl_error!(
                    ErrorKind::ConversionError,
                    "DuckLake Arrow copy conversion failed",
                    format!("Column '{column_name}' has a value incompatible with its Arrow kind")
                ));
            }
        }
        Ok(())
    }

    /// Converts the accumulated data into an Arrow array.
    fn into_array(self) -> ArrayRef {
        match self {
            Self::Bool(values) => Arc::new(BooleanArray::from(values)),
            Self::I16(values) => Arc::new(Int16Array::from(values)),
            Self::I32(values) => Arc::new(Int32Array::from(values)),
            Self::I64(values) => Arc::new(Int64Array::from(values)),
            Self::U64(values) => Arc::new(UInt64Array::from(values)),
            Self::F32(values) => Arc::new(Float32Array::from(values)),
            Self::F64(values) => Arc::new(Float64Array::from(values)),
            Self::Utf8(values) => Arc::new(StringArray::from(values)),
            Self::Date32(values) => Arc::new(Date32Array::from(values)),
            Self::Time64Microsecond(values) => Arc::new(Time64MicrosecondArray::from(values)),
            Self::TimestampMicrosecond(values) => Arc::new(TimestampMicrosecondArray::from(values)),
            Self::TimestampTzMicrosecond(values) => {
                Arc::new(TimestampMicrosecondArray::from(values).with_timezone_utc())
            }
            Self::Binary(values) => {
                let values: Vec<Option<&[u8]>> = values.iter().map(Option::as_deref).collect();
                Arc::new(BinaryArray::from_opt_vec(values))
            }
        }
    }
}

/// Returns Arrow column kinds for schemas that can use Arrow COPY staging.
fn arrow_column_kinds(
    replicated_table_schema: &ReplicatedTableSchema,
) -> Option<Vec<ArrowColumnKind>> {
    replicated_table_schema.column_schemas().map(arrow_column_kind).collect()
}

/// Returns the Arrow column kind for one replicated column.
fn arrow_column_kind(column_schema: &ColumnSchema) -> Option<ArrowColumnKind> {
    if is_array_type(&column_schema.typ) {
        return None;
    }

    match &column_schema.typ {
        &Type::BOOL => Some(ArrowColumnKind::Bool),
        &Type::INT2 => Some(ArrowColumnKind::I16),
        &Type::INT4 => Some(ArrowColumnKind::I32),
        &Type::INT8 => Some(ArrowColumnKind::I64),
        &Type::OID => Some(ArrowColumnKind::U64),
        &Type::FLOAT4 => Some(ArrowColumnKind::F32),
        &Type::FLOAT8 => Some(ArrowColumnKind::F64),
        &Type::DATE => Some(ArrowColumnKind::Date32),
        &Type::TIME => Some(ArrowColumnKind::Time64Microsecond),
        &Type::TIMESTAMP => Some(ArrowColumnKind::TimestampMicrosecond),
        &Type::TIMESTAMPTZ => Some(ArrowColumnKind::TimestampTzMicrosecond),
        &Type::BYTEA => Some(ArrowColumnKind::Binary),
        &Type::UUID | &Type::JSON | &Type::JSONB => None,
        _ => Some(ArrowColumnKind::Utf8),
    }
}

/// Returns whether all copied rows can be represented by the Arrow kinds.
fn copy_rows_match_arrow_kinds(column_kinds: &[ArrowColumnKind], table_rows: &[TableRow]) -> bool {
    table_rows.iter().all(|row| {
        row.values().len() == column_kinds.len()
            && row
                .values()
                .iter()
                .zip(column_kinds.iter())
                .all(|(cell, kind)| cell_matches_arrow_kind(cell, *kind))
    })
}

/// Returns whether one cell can be pushed into an Arrow column kind.
fn cell_matches_arrow_kind(cell: &Cell, kind: ArrowColumnKind) -> bool {
    if matches!(cell, Cell::Null) {
        return true;
    }

    matches!(
        (kind, cell),
        (ArrowColumnKind::Bool, Cell::Bool(_))
            | (ArrowColumnKind::I16, Cell::I16(_))
            | (ArrowColumnKind::I32, Cell::I32(_))
            | (ArrowColumnKind::I64, Cell::I64(_))
            | (ArrowColumnKind::U64, Cell::U32(_))
            | (ArrowColumnKind::F32, Cell::F32(_))
            | (ArrowColumnKind::F64, Cell::F64(_))
            | (
                ArrowColumnKind::Utf8,
                Cell::String(_)
                    | Cell::Numeric(_)
                    | Cell::TimeTz(_)
                    | Cell::Uuid(_)
                    | Cell::Json(_)
            )
            | (ArrowColumnKind::Date32, Cell::Date(_))
            | (ArrowColumnKind::Time64Microsecond, Cell::Time(_))
            | (ArrowColumnKind::TimestampMicrosecond, Cell::Timestamp(_))
            | (ArrowColumnKind::TimestampTzMicrosecond, Cell::TimestampTz(_))
            | (ArrowColumnKind::Binary, Cell::Bytes(_))
    )
}

/// Converts rows into a DuckDB-compatible Arrow record batch.
fn copy_rows_to_arrow_record_batch(
    replicated_table_schema: &ReplicatedTableSchema,
    column_kinds: Vec<ArrowColumnKind>,
    table_rows: Vec<TableRow>,
) -> EtlResult<RecordBatch> {
    let row_count = table_rows.len();
    let column_schemas = replicated_table_schema.column_schemas().collect::<Vec<_>>();
    let fields = column_schemas
        .iter()
        .zip(column_kinds.iter())
        .map(|(column_schema, kind)| {
            Field::new(
                DUCKLAKE_COLUMN_NAME_MAPPING.map_name(&column_schema.name),
                kind.data_type(),
                column_schema.nullable,
            )
        })
        .collect::<Vec<_>>();
    let mut column_values = column_kinds
        .iter()
        .copied()
        .map(|kind| ArrowColumnValues::with_capacity(kind, row_count))
        .collect::<Vec<_>>();

    for row in table_rows {
        for ((column_value, field), cell) in
            column_values.iter_mut().zip(fields.iter()).zip(row.into_values())
        {
            column_value.push_cell(field.name(), cell)?;
        }
    }

    let arrays = column_values.into_iter().map(ArrowColumnValues::into_array).collect::<Vec<_>>();

    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(|error| {
        etl_error!(
            ErrorKind::ConversionError,
            "DuckLake Arrow copy conversion failed",
            source: error
        )
    })
}

/// Converts a date to DuckDB's Arrow date representation.
fn date_days_since_epoch(value: NaiveDate, column_name: &str) -> EtlResult<i32> {
    let epoch_date = NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch date should be valid");
    i32::try_from(value.signed_duration_since(epoch_date).num_days()).map_err(|_| {
        etl_error!(
            ErrorKind::ConversionError,
            "DuckLake Arrow copy conversion failed",
            format!("Column '{column_name}' contains a date outside DuckDB's supported range")
        )
    })
}

/// Converts a time to DuckDB's microseconds-since-midnight representation.
fn time_micros_since_midnight(value: NaiveTime) -> i64 {
    i64::from(value.num_seconds_from_midnight()) * 1_000_000 + i64::from(value.nanosecond() / 1_000)
}

/// Serializes a borrowed row into a SQL `VALUES (...)` tuple.
pub(super) fn table_row_to_sql_literal_ref(row: &TableRow) -> String {
    format!("({})", row.values().iter().map(cell_to_sql_literal_ref).collect::<Vec<_>>().join(", "))
}

/// Serializes a borrowed cell into a DuckDB SQL literal expression.
pub(super) fn cell_to_sql_literal_ref(cell: &Cell) -> String {
    cell_to_sql_literal(cell_to_owned(cell))
}

/// Returns whether a cell must bypass the DuckDB appender path.
fn cell_requires_sql_literals(cell: &Cell) -> bool {
    matches!(cell, Cell::Array(_))
}

/// Serializes a row into a SQL `VALUES (...)` tuple.
fn table_row_to_sql_literal(row: TableRow) -> String {
    table_row_to_sql_literal_ref(&row)
}

/// Converts a [`Cell`] into a DuckDB SQL literal expression.
fn cell_to_sql_literal(cell: Cell) -> String {
    match cell {
        Cell::Null => "NULL".to_owned(),
        Cell::Bool(b) => {
            if b {
                "TRUE".to_owned()
            } else {
                "FALSE".to_owned()
            }
        }
        Cell::String(s) => quote_literal(&s),
        Cell::I16(i) => i.to_string(),
        Cell::I32(i) => i.to_string(),
        Cell::U32(u) => u.to_string(),
        Cell::I64(i) => i.to_string(),
        Cell::F32(f) => float_literal(f as f64, false),
        Cell::F64(f) => float_literal(f, true),
        Cell::Numeric(n) => quote_literal(&n.to_string()),
        Cell::Date(d) => format!("DATE '{}'", d.format("%Y-%m-%d")),
        Cell::Time(t) => format!("TIME '{}'", t.format("%H:%M:%S%.6f")),
        Cell::TimeTz(t) => quote_literal(&t.to_string()),
        Cell::Timestamp(dt) => {
            format!("TIMESTAMP '{}'", dt.format("%Y-%m-%d %H:%M:%S%.6f"))
        }
        Cell::TimestampTz(dt) => {
            format!("TIMESTAMPTZ '{}'", dt.format("%Y-%m-%d %H:%M:%S%.6f%:z"))
        }
        Cell::Uuid(u) => format!("CAST({} AS UUID)", quote_literal(&u.to_string())),
        Cell::Json(j) => format!("CAST({} AS JSON)", quote_literal(&j.to_string())),
        Cell::Bytes(b) => format!("from_hex('{}')", encode_hex(&b)),
        Cell::Array(arr) => array_cell_to_sql_literal(arr),
    }
}

/// Clones a [`Cell`] from a borrowed row reference.
fn cell_to_owned(cell: &Cell) -> Cell {
    match cell {
        Cell::Null => Cell::Null,
        Cell::Bool(value) => Cell::Bool(*value),
        Cell::String(value) => Cell::String(value.clone()),
        Cell::I16(value) => Cell::I16(*value),
        Cell::I32(value) => Cell::I32(*value),
        Cell::U32(value) => Cell::U32(*value),
        Cell::I64(value) => Cell::I64(*value),
        Cell::F32(value) => Cell::F32(*value),
        Cell::F64(value) => Cell::F64(*value),
        Cell::Numeric(value) => Cell::Numeric(value.clone()),
        Cell::Date(value) => Cell::Date(*value),
        Cell::Time(value) => Cell::Time(*value),
        Cell::TimeTz(value) => Cell::TimeTz(*value),
        Cell::Timestamp(value) => Cell::Timestamp(*value),
        Cell::TimestampTz(value) => Cell::TimestampTz(*value),
        Cell::Uuid(value) => Cell::Uuid(*value),
        Cell::Json(value) => Cell::Json(value.clone()),
        Cell::Bytes(value) => Cell::Bytes(value.clone()),
        Cell::Array(value) => Cell::Array(array_cell_to_owned(value)),
    }
}

/// Clones an [`ArrayCell`] from a borrowed row reference.
fn array_cell_to_owned(cell: &ArrayCell) -> ArrayCell {
    match cell {
        ArrayCell::Bool(values) => ArrayCell::Bool(values.clone()),
        ArrayCell::String(values) => ArrayCell::String(values.clone()),
        ArrayCell::I16(values) => ArrayCell::I16(values.clone()),
        ArrayCell::I32(values) => ArrayCell::I32(values.clone()),
        ArrayCell::U32(values) => ArrayCell::U32(values.clone()),
        ArrayCell::I64(values) => ArrayCell::I64(values.clone()),
        ArrayCell::F32(values) => ArrayCell::F32(values.clone()),
        ArrayCell::F64(values) => ArrayCell::F64(values.clone()),
        ArrayCell::Numeric(values) => ArrayCell::Numeric(values.clone()),
        ArrayCell::Date(values) => ArrayCell::Date(values.clone()),
        ArrayCell::Time(values) => ArrayCell::Time(values.clone()),
        ArrayCell::TimeTz(values) => ArrayCell::TimeTz(values.clone()),
        ArrayCell::Timestamp(values) => ArrayCell::Timestamp(values.clone()),
        ArrayCell::TimestampTz(values) => ArrayCell::TimestampTz(values.clone()),
        ArrayCell::Uuid(values) => ArrayCell::Uuid(values.clone()),
        ArrayCell::Json(values) => ArrayCell::Json(values.clone()),
        ArrayCell::Bytes(values) => ArrayCell::Bytes(values.clone()),
    }
}

/// Converts an [`ArrayCell`] into a DuckDB list literal expression.
fn array_cell_to_sql_literal(arr: ArrayCell) -> String {
    let values: Vec<String> = match arr {
        ArrayCell::Bool(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| if value { "TRUE" } else { "FALSE" }.to_owned(),
                )
            })
            .collect(),
        ArrayCell::String(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| quote_literal(&value)))
            .collect(),
        ArrayCell::I16(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| value.to_string()))
            .collect(),
        ArrayCell::I32(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| value.to_string()))
            .collect(),
        ArrayCell::U32(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| value.to_string()))
            .collect(),
        ArrayCell::I64(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| value.to_string()))
            .collect(),
        ArrayCell::F32(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(|| "NULL".to_owned(), |value| float_literal(value as f64, false))
            })
            .collect(),
        ArrayCell::F64(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| float_literal(value, true)))
            .collect(),
        ArrayCell::Numeric(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| quote_literal(&value.to_string())))
            .collect(),
        ArrayCell::Date(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("DATE '{}'", value.format("%Y-%m-%d")),
                )
            })
            .collect(),
        ArrayCell::Time(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("TIME '{}'", value.format("%H:%M:%S%.6f")),
                )
            })
            .collect(),
        ArrayCell::TimeTz(v) => v
            .into_iter()
            .map(|o| o.map_or_else(|| "NULL".to_owned(), |value| quote_literal(&value.to_string())))
            .collect(),
        ArrayCell::Timestamp(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("TIMESTAMP '{}'", value.format("%Y-%m-%d %H:%M:%S%.6f")),
                )
            })
            .collect(),
        ArrayCell::TimestampTz(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("TIMESTAMPTZ '{}'", value.format("%Y-%m-%d %H:%M:%S%.6f%:z")),
                )
            })
            .collect(),
        ArrayCell::Uuid(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("CAST({} AS UUID)", quote_literal(&value.to_string())),
                )
            })
            .collect(),
        ArrayCell::Json(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("CAST({} AS JSON)", quote_literal(&value.to_string())),
                )
            })
            .collect(),
        ArrayCell::Bytes(v) => v
            .into_iter()
            .map(|o| {
                o.map_or_else(
                    || "NULL".to_owned(),
                    |value| format!("from_hex('{}')", encode_hex(&value)),
                )
            })
            .collect(),
    };

    format!("[{}]", values.join(", "))
}

/// Returns a DuckDB SQL literal for a floating-point value.
fn float_literal(value: f64, is_double: bool) -> String {
    if value.is_nan() {
        return if is_double {
            "CAST('NaN' AS DOUBLE)".to_owned()
        } else {
            "CAST('NaN' AS FLOAT)".to_owned()
        };
    }
    if value == f64::INFINITY {
        return if is_double {
            "CAST('Infinity' AS DOUBLE)".to_owned()
        } else {
            "CAST('Infinity' AS FLOAT)".to_owned()
        };
    }
    if value == f64::NEG_INFINITY {
        return if is_double {
            "CAST('-Infinity' AS DOUBLE)".to_owned()
        } else {
            "CAST('-Infinity' AS FLOAT)".to_owned()
        };
    }

    let typ = if is_double { "DOUBLE" } else { "FLOAT" };
    format!("CAST('{value:e}' AS {typ})")
}

/// Encodes bytes as uppercase hexadecimal for DuckDB's `from_hex`.
fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02X}")).collect()
}

/// Converts a [`Cell`] to a [`duckdb::types::Value`] for use with parameterized
/// INSERT statements.
fn cell_to_value(cell: Cell) -> Value {
    let epoch_date = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();

    match cell {
        Cell::Null => Value::Null,
        Cell::Bool(b) => Value::Boolean(b),
        Cell::String(s) => Value::Text(s),
        Cell::I16(i) => Value::SmallInt(i),
        Cell::I32(i) => Value::Int(i),
        Cell::U32(u) => Value::UInt(u),
        Cell::I64(i) => Value::BigInt(i),
        Cell::F32(f) => Value::Float(f),
        Cell::F64(f) => Value::Double(f),
        // NUMERIC stored as VARCHAR to avoid precision loss.
        Cell::Numeric(n) => Value::Text(n.to_string()),
        Cell::Date(d) => Value::Date32(d.signed_duration_since(epoch_date).num_days() as i32),
        Cell::Time(t) => Value::Time64(TimeUnit::Microsecond, time_micros_since_midnight(t)),
        Cell::TimeTz(t) => Value::Text(t.to_string()),
        Cell::Timestamp(dt) => {
            Value::Timestamp(TimeUnit::Microsecond, dt.and_utc().timestamp_micros())
        }
        Cell::TimestampTz(dt) => Value::Timestamp(TimeUnit::Microsecond, dt.timestamp_micros()),
        // UUID stored as text; DuckDB casts VARCHAR → UUID automatically.
        Cell::Uuid(u) => Value::Text(u.to_string()),
        // JSON serialised as text.
        Cell::Json(j) => Value::Text(j.to_string()),
        Cell::Bytes(b) => Value::Blob(b),
        Cell::Array(arr) => array_cell_to_value(arr),
    }
}

/// Converts an [`ArrayCell`] (with nullable elements) to a `Value::List`.
fn array_cell_to_value(arr: ArrayCell) -> Value {
    let values = match arr {
        ArrayCell::Bool(v) => {
            v.into_iter().map(|o| o.map_or(Value::Null, Value::Boolean)).collect()
        }
        ArrayCell::String(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::Text)).collect(),
        ArrayCell::I16(v) => {
            v.into_iter().map(|o| o.map_or(Value::Null, Value::SmallInt)).collect()
        }
        ArrayCell::I32(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::Int)).collect(),
        ArrayCell::U32(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::UInt)).collect(),
        ArrayCell::I64(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::BigInt)).collect(),
        ArrayCell::F32(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::Float)).collect(),
        ArrayCell::F64(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::Double)).collect(),
        ArrayCell::Numeric(v) => {
            v.into_iter().map(|o| o.map_or(Value::Null, |n| Value::Text(n.to_string()))).collect()
        }
        ArrayCell::Date(v) => {
            let epoch_date = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            v.into_iter()
                .map(|o| {
                    o.map_or(Value::Null, |d| {
                        Value::Date32(d.signed_duration_since(epoch_date).num_days() as i32)
                    })
                })
                .collect()
        }
        ArrayCell::Time(v) => v
            .into_iter()
            .map(|o| {
                o.map_or(Value::Null, |t| {
                    Value::Time64(TimeUnit::Microsecond, time_micros_since_midnight(t))
                })
            })
            .collect(),
        ArrayCell::TimeTz(v) => {
            v.into_iter().map(|o| o.map_or(Value::Null, |t| Value::Text(t.to_string()))).collect()
        }
        ArrayCell::Timestamp(v) => v
            .into_iter()
            .map(|o| {
                o.map_or(Value::Null, |dt| {
                    Value::Timestamp(TimeUnit::Microsecond, dt.and_utc().timestamp_micros())
                })
            })
            .collect(),
        ArrayCell::TimestampTz(v) => v
            .into_iter()
            .map(|o| {
                o.map_or(Value::Null, |dt| {
                    Value::Timestamp(TimeUnit::Microsecond, dt.timestamp_micros())
                })
            })
            .collect(),
        ArrayCell::Uuid(v) => {
            v.into_iter().map(|o| o.map_or(Value::Null, |u| Value::Text(u.to_string()))).collect()
        }
        ArrayCell::Json(v) => {
            v.into_iter().map(|o| o.map_or(Value::Null, |j| Value::Text(j.to_string()))).collect()
        }
        ArrayCell::Bytes(v) => v.into_iter().map(|o| o.map_or(Value::Null, Value::Blob)).collect(),
    };
    Value::List(values)
}

#[cfg(test)]
mod tests {
    use etl::schema::{
        ColumnSchema, ReplicatedTableSchema, TableId, TableName, TableSchema, Type as PgType,
    };

    use super::*;

    fn replicated_schema(columns: Vec<ColumnSchema>) -> ReplicatedTableSchema {
        ReplicatedTableSchema::all(Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            columns,
        )))
    }

    #[test]
    fn sql_float_literals_preserve_bits() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        for value in [-0.0_f64, f64::MIN_POSITIVE, f64::from_bits(1), 1.0000000000000002, f64::MAX]
        {
            let actual: f64 = conn
                .query_row(&format!("select {}", float_literal(value, true)), [], |row| row.get(0))
                .unwrap();
            assert_eq!(actual.to_bits(), value.to_bits());
        }
        for value in [-0.0_f32, f32::MIN_POSITIVE, f32::from_bits(1), 1.0000001, f32::MAX] {
            let actual: f32 = conn
                .query_row(
                    &format!("select {}", float_literal(f64::from(value), false)),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(actual.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn cell_to_value_primitives() {
        assert_eq!(cell_to_value(Cell::Null), Value::Null);
        assert_eq!(cell_to_value(Cell::Bool(true)), Value::Boolean(true));
        assert_eq!(
            cell_to_value(Cell::String("hello".to_owned())),
            Value::Text("hello".to_owned())
        );
        assert_eq!(cell_to_value(Cell::I32(42)), Value::Int(42));
        assert_eq!(cell_to_value(Cell::I64(-1)), Value::BigInt(-1));
        assert_eq!(cell_to_value(Cell::F64(3.46)), Value::Double(3.46));
    }

    #[test]
    fn cell_to_value_preserves_time_microseconds_for_scalars_and_arrays() {
        let time = NaiveTime::from_hms_micro_opt(12, 34, 56, 789_012).unwrap();
        let micros = 45_296_789_012;
        let value = Value::Time64(TimeUnit::Microsecond, micros);

        assert_eq!(cell_to_value(Cell::Time(time)), value);
        assert_eq!(
            array_cell_to_value(ArrayCell::Time(vec![Some(time), None])),
            Value::List(vec![value, Value::Null])
        );
    }

    #[test]
    fn array_cell_to_sql_literal_preserves_nulls() {
        assert_eq!(
            array_cell_to_sql_literal(ArrayCell::I32(vec![Some(1), None, Some(3)])),
            "[1, NULL, 3]"
        );
        assert_eq!(
            array_cell_to_sql_literal(ArrayCell::Json(vec![
                Some(serde_json::json!({"a": 1})),
                None,
            ])),
            "[CAST('{\"a\":1}' AS JSON), NULL]"
        );
    }

    #[test]
    fn prepare_rows_uses_sql_literals_for_arrays() {
        let prepared = prepare_rows(vec![TableRow::new(vec![
            Cell::I32(1),
            Cell::Array(ArrayCell::I32(vec![Some(1), None, Some(3)])),
        ])]);

        match prepared {
            PreparedRows::SqlLiterals(rows) => {
                assert_eq!(rows, vec!["(1, [1, NULL, 3])"]);
            }
            PreparedRows::Appender(_) => panic!("expected sql literal fallback"),
            PreparedRows::ArrowRecordBatch(_) => panic!("expected sql literal fallback"),
        }
    }

    #[test]
    fn prepare_copy_rows_uses_arrow_for_supported_columns() {
        let schema = replicated_schema(vec![
            ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false),
            ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true),
            ColumnSchema::new("created_at".to_owned(), PgType::TIMESTAMP, -1, 3, true),
        ]);
        let rows = vec![
            TableRow::new(vec![
                Cell::I32(1),
                Cell::String("alice".to_owned()),
                Cell::Timestamp(
                    chrono::NaiveDate::from_ymd_opt(2026, 1, 2)
                        .unwrap()
                        .and_hms_opt(3, 4, 5)
                        .unwrap(),
                ),
            ]),
            TableRow::new(vec![Cell::I32(2), Cell::Null, Cell::Null]),
        ];

        let prepared = prepare_copy_rows(&schema, rows).unwrap();

        match prepared {
            PreparedRows::ArrowRecordBatch(record_batch) => {
                assert_eq!(record_batch.num_rows(), 2);
                assert_eq!(record_batch.num_columns(), 3);
                assert_eq!(record_batch.schema().field(0).data_type(), &DataType::Int32);
                assert_eq!(record_batch.schema().field(1).data_type(), &DataType::Utf8);
                assert_eq!(
                    record_batch.schema().field(2).data_type(),
                    &DataType::Timestamp(ArrowTimeUnit::Microsecond, None)
                );
            }
            PreparedRows::Appender(_) | PreparedRows::SqlLiterals(_) => {
                panic!("expected Arrow record batch")
            }
        }
    }

    #[test]
    fn prepare_copy_rows_falls_back_for_array_columns() {
        let schema = replicated_schema(vec![
            ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false),
            ColumnSchema::new("tags".to_owned(), PgType::INT4_ARRAY, -1, 2, true),
        ]);
        let rows = vec![TableRow::new(vec![
            Cell::I32(1),
            Cell::Array(ArrayCell::I32(vec![Some(1), None])),
        ])];

        let prepared = prepare_copy_rows(&schema, rows).unwrap();

        match prepared {
            PreparedRows::SqlLiterals(rows) => {
                assert_eq!(rows, vec!["(1, [1, NULL])"]);
            }
            PreparedRows::Appender(_) | PreparedRows::ArrowRecordBatch(_) => {
                panic!("expected sql literal fallback")
            }
        }
    }

    #[test]
    fn prepare_copy_rows_falls_back_for_cast_sensitive_columns() {
        let schema = replicated_schema(vec![ColumnSchema::new(
            "payload".to_owned(),
            PgType::JSON,
            -1,
            1,
            false,
        )]);
        let rows = vec![TableRow::new(vec![Cell::Json(serde_json::json!({"id": 1}))])];

        let prepared = prepare_copy_rows(&schema, rows).unwrap();

        match prepared {
            PreparedRows::Appender(rows) => {
                assert_eq!(rows.len(), 1);
            }
            PreparedRows::SqlLiterals(_) | PreparedRows::ArrowRecordBatch(_) => {
                panic!("expected row appender fallback")
            }
        }
    }
}
