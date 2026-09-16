//! End-to-end integration tests for DuckLake using a real ETL [`Pipeline`].
//!
//! These tests use a PostgreSQL-backed DuckLake catalog and verify the final
//! table contents by querying DuckLake directly through DuckDB.

use duckdb::Connection;
use etl::{
    config::BatchConfig,
    event::EventType,
    pipeline::PipelineId,
    store::StateStore,
    test_utils::{
        database::{spawn_source_database, test_table_name},
        event::EventCondition,
        notifying_store::NotifyingStore,
        pipeline::{create_pipeline, create_pipeline_with_batch_config},
        test_destination_wrapper::TestDestinationWrapper,
        test_schema::{TableSelection, insert_mock_data, setup_test_database_schema},
    },
};
use etl_destinations::ducklake::{
    DuckLakeDestination, DuckLakeTableName, table_name_to_ducklake_table_name,
};
use etl_postgres::{below_version, tokio::test_utils::TableModification, version::POSTGRES_15};
use etl_telemetry::tracing::init_test_tracing;
use pg_escape::{quote_identifier, quote_literal};
use rand::random;
use rust_decimal::Decimal;
use url::Url;

use crate::support::ducklake::{
    catalog_attach_target, create_test_lake, ducklake_load_sql, open_verification_connection,
};

/// Row size that exceeds the test's preferred 8 MiB batch limit.
const OVERSIZED_COPY_ROW_BYTES: i32 = 9_000_000;

/// Expected row shape after adding columns during replication.
#[derive(Debug, Eq, PartialEq)]
struct SchemaAddRow {
    id: i64,
    name: String,
    age: i32,
    email: Option<String>,
    score: Option<i32>,
}

/// Expected row shape after a rename cycle and same-name replacement.
#[derive(Debug, Eq, PartialEq)]
struct SchemaMultiRow {
    id: i64,
    status: String,
    name: Option<String>,
    age: Option<String>,
}

/// Expected row shape after schema evolution followed by mutations.
#[derive(Debug, Eq, PartialEq)]
struct SchemaMutationRow {
    id: i64,
    full_name: String,
    visits: i32,
    email: Option<String>,
}

/// Expected row shape after simulator-style generated-column rotation.
#[derive(Debug, Eq, PartialEq)]
struct SimulatorDdlRotationRow {
    id: i64,
    text_col: String,
    ddl_col_0_0: Option<String>,
}

/// Expected row shape after simulator-style generated-column type changes.
#[derive(Debug, Eq, PartialEq)]
struct SimulatorDdlTypeRow {
    id: i64,
    text_col: String,
    ddl_col_0_0: Option<String>,
    ddl_col_1_0: Option<i32>,
    ddl_col_2_0: Option<bool>,
    ddl_col_3_0_is_present: bool,
    ddl_col_4_0: Option<Decimal>,
}

/// Opens a DuckLake verification connection using blocking DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn open_lake_conn(catalog: &Url, data: &Url) -> Connection {
    let conn = open_verification_connection();
    let catalog_attach_target = catalog_attach_target(catalog);
    conn.execute_batch(&format!(
        "{} attach {} as {} (data_path {});",
        ducklake_load_sql(),
        quote_literal(&format!("ducklake:{catalog_attach_target}")),
        quote_identifier("lake"),
        quote_literal(data.as_str())
    ))
    .unwrap();

    conn
}

/// Forces DuckLake to checkpoint catalog metadata before cross-connection
/// verification.
///
/// These end-to-end tests shut the destination down and then attach a fresh
/// DuckDB connection to verify the final lake state. Without an explicit
/// checkpoint here, that new connection can observe stale catalog metadata for
/// a short period even though the pipeline shutdown has already completed,
/// which makes the assertions flaky in CI. Running `checkpoint` makes the final
/// durable state visible to the verification connection deterministically.
///
/// Production async code must wrap equivalent blocking DuckDB work in
/// `run_duckdb_blocking`.
fn checkpoint_lake(catalog: &Url, data: &Url) {
    let conn = open_lake_conn(catalog, data);
    conn.execute_batch("checkpoint").unwrap();
}

fn qualified_lake_table_name(table_name: &DuckLakeTableName) -> String {
    format!(
        "{}.{}.{}",
        quote_identifier("lake"),
        quote_identifier(table_name.schema()),
        quote_identifier(table_name.table())
    )
}

/// Counts one applied-marker kind for a destination table.
fn count_applied_batches(
    conn: &Connection,
    table_name: &DuckLakeTableName,
    batch_kind: &str,
) -> i64 {
    let sql = format!(
        "select count(*) from {}.{} where table_name = {} and batch_kind = {}",
        quote_identifier("lake"),
        quote_identifier("__etl_applied_table_batches"),
        quote_literal(&table_name.id()),
        quote_literal(batch_kind),
    );
    conn.query_row(&sql, [], |row| row.get(0)).unwrap()
}

/// Summarizes copied payload rows without loading their contents into the test
/// process.
fn query_payload_summary(conn: &Connection, table_name: &DuckLakeTableName) -> (i64, i64, i64) {
    let sql = format!(
        "select count(*), count(distinct payload), min(length(payload)) from {}",
        qualified_lake_table_name(table_name)
    );
    conn.query_row(&sql, [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap()
}

/// Queries replicated user rows using blocking DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_user_rows(conn: &Connection, table_name: &DuckLakeTableName) -> Vec<(i64, String, i32)> {
    let sql =
        format!("select id, name, age from {} order by id", qualified_lake_table_name(table_name));
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap(), row.get(2).unwrap()));
    }

    result
}

/// Queries replicated order rows using blocking DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_order_rows(conn: &Connection, table_name: &DuckLakeTableName) -> Vec<(i64, String)> {
    let sql = format!(
        "select id, description from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }

    result
}

/// Queries DuckLake table columns using blocking DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_table_columns(conn: &Connection, table_name: &DuckLakeTableName) -> Vec<String> {
    let sql = format!(
        "select column_name from information_schema.columns where table_catalog = {} and \
         table_schema = {} and table_name = {} order by ordinal_position",
        quote_literal("lake"),
        quote_literal(table_name.schema()),
        quote_literal(table_name.table())
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push(row.get(0).unwrap());
    }

    result
}

/// Queries rows written before and after a case-only source rename.
fn query_case_mapped_rows(conn: &Connection, table_name: &DuckLakeTableName) -> Vec<(i64, String)> {
    let sql = format!(
        "select id, display_name from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }

    result
}

/// Queries rows after an add-column schema change using blocking DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_schema_add_rows(conn: &Connection, table_name: &DuckLakeTableName) -> Vec<SchemaAddRow> {
    let sql = format!(
        "select id, name, age, email, score from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push(SchemaAddRow {
            id: row.get(0).unwrap(),
            name: row.get(1).unwrap(),
            age: row.get(2).unwrap(),
            email: row.get(3).unwrap(),
            score: row.get(4).unwrap(),
        });
    }

    result
}

/// Queries rows after an existing source column enters the replication mask.
fn query_publication_add_rows(
    conn: &Connection,
    table_name: &DuckLakeTableName,
) -> Vec<(i64, String, Option<String>)> {
    let sql = format!(
        "select id, name, status from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap(), row.get(2).unwrap()));
    }

    result
}

/// Queries rows after an ordered name-reuse schema change using blocking
/// DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_schema_multi_rows(
    conn: &Connection,
    table_name: &DuckLakeTableName,
) -> Vec<SchemaMultiRow> {
    let sql = format!(
        "select id, status, name, age from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push(SchemaMultiRow {
            id: row.get(0).unwrap(),
            status: row.get(1).unwrap(),
            name: row.get(2).unwrap(),
            age: row.get(3).unwrap(),
        });
    }

    result
}

/// Queries rows after schema evolution and mutations using blocking DuckDB
/// APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_schema_mutation_rows(
    conn: &Connection,
    table_name: &DuckLakeTableName,
) -> Vec<SchemaMutationRow> {
    let sql = format!(
        "select id, full_name, visits, email from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push(SchemaMutationRow {
            id: row.get(0).unwrap(),
            full_name: row.get(1).unwrap(),
            visits: row.get(2).unwrap(),
            email: row.get(3).unwrap(),
        });
    }

    result
}

/// Queries rows after simulator-style generated-column rotation using blocking
/// DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_simulator_ddl_rotation_rows(
    conn: &Connection,
    table_name: &DuckLakeTableName,
) -> Vec<SimulatorDdlRotationRow> {
    let sql = format!(
        "select id, text_col, ddl_col_0_0 from {} order by id",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let mut rows = statement.query([]).unwrap();
    let mut result = Vec::new();

    while let Some(row) = rows.next().unwrap() {
        result.push(SimulatorDdlRotationRow {
            id: row.get(0).unwrap(),
            text_col: row.get(1).unwrap(),
            ddl_col_0_0: row.get(2).unwrap(),
        });
    }

    result
}

/// Queries the final row after simulator-style generated-column type changes
/// using blocking DuckDB APIs.
///
/// Production async code must wrap equivalent DuckDB work in
/// `run_duckdb_blocking`.
fn query_simulator_ddl_type_row(
    conn: &Connection,
    table_name: &DuckLakeTableName,
) -> SimulatorDdlTypeRow {
    let sql = format!(
        "select id, text_col, ddl_col_0_0, ddl_col_1_0, ddl_col_2_0, ddl_col_3_0 is not null, \
         ddl_col_4_0 from {} where id = 5",
        qualified_lake_table_name(table_name)
    );
    let mut statement = conn.prepare(&sql).unwrap();
    let row = statement.query_row([], |row| {
        Ok(SimulatorDdlTypeRow {
            id: row.get(0)?,
            text_col: row.get(1)?,
            ddl_col_0_0: row.get(2)?,
            ddl_col_1_0: row.get(3)?,
            ddl_col_2_0: row.get(4)?,
            ddl_col_3_0_is_present: row.get(5)?,
            ddl_col_4_0: row.get(6)?,
        })
    });

    row.unwrap()
}

async fn build_destination(
    catalog_url: &Url,
    data_url: &Url,
    store: NotifyingStore,
) -> TestDestinationWrapper<DuckLakeDestination<NotifyingStore>> {
    let raw_destination = DuckLakeDestination::new(
        catalog_url.clone(),
        data_url.clone(),
        1,
        None,
        None,
        None,
        None,
        store,
    )
    .await
    .unwrap();

    TestDestinationWrapper::wrap(raw_destination)
}

#[tokio::test(flavor = "multi_thread")]
async fn table_copy_and_streaming_with_restart() {
    init_test_tracing();

    let mut database = spawn_source_database().await;
    let database_schema = setup_test_database_schema(&database, TableSelection::Both).await;
    let lake = create_test_lake("table_copy_and_streaming_with_restart").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();

    let users_table_name =
        table_name_to_ducklake_table_name(&database_schema.users_schema().name).unwrap();
    let orders_table_name =
        table_name_to_ducklake_table_name(&database_schema.orders_schema().name).unwrap();

    insert_mock_data(
        &mut database,
        &database_schema.users_schema().name,
        &database_schema.orders_schema().name,
        1..=2,
        false,
    )
    .await;

    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    let users_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.users_schema().id).await;
    let orders_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.orders_schema().id).await;

    pipeline.start().await.unwrap();

    users_sync_complete_notify.notified().await;
    orders_sync_complete_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_user_rows(&conn, &users_table_name),
        vec![(1, "user_1".to_owned(), 1), (2, "user_2".to_owned(), 2),]
    );
    assert_eq!(
        query_order_rows(&conn, &orders_table_name),
        vec![(1, "description_1".to_owned()), (2, "description_2".to_owned()),]
    );
    assert_eq!(count_applied_batches(&conn, &users_table_name, "copy_complete"), 1);
    assert_eq!(count_applied_batches(&conn, &orders_table_name, "copy_complete"), 1);
    drop(conn);

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    pipeline.start().await.unwrap();

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Insert, database_schema.users_schema().id, 2),
            EventCondition::TableCount(EventType::Insert, database_schema.orders_schema().id, 2),
        ])
        .await;

    insert_mock_data(
        &mut database,
        &database_schema.users_schema().name,
        &database_schema.orders_schema().name,
        3..=4,
        false,
    )
    .await;

    events_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_user_rows(&conn, &users_table_name),
        vec![
            (1, "user_1".to_owned(), 1),
            (2, "user_2".to_owned(), 2),
            (3, "user_3".to_owned(), 3),
            (4, "user_4".to_owned(), 4),
        ]
    );
    assert_eq!(
        query_order_rows(&conn, &orders_table_name),
        vec![
            (1, "description_1".to_owned()),
            (2, "description_2".to_owned()),
            (3, "description_3".to_owned()),
            (4, "description_4".to_owned()),
        ]
    );
}

/// Copy chunks with identical contents must not collapse to one applied
/// marker. Each row exceeds the test's 8 MiB batch limit, forcing the source
/// copy stream to emit two separate batches.
#[tokio::test(flavor = "multi_thread")]
async fn table_copy_preserves_identical_oversized_rows_across_batches() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let source_table_name = test_table_name("ducklake_identical_copy_batches");
    let table_id = database
        .create_table(source_table_name.clone(), false, &[("payload", "text not null")])
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_identical_copy_batches";
    database
        .create_publication(publication_name, std::slice::from_ref(&source_table_name))
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (payload) select repeat('a', {OVERSIZED_COPY_ROW_BYTES}) from \
             generate_series(1, 2)",
            source_table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    let source_summary: (i64, i64, i32) = database
        .client
        .as_ref()
        .unwrap()
        .query_one(
            &format!(
                "select count(*), count(distinct payload), min(octet_length(payload)) from {}",
                source_table_name.as_quoted_identifier()
            ),
            &[],
        )
        .await
        .map(|row| (row.get(0), row.get(1), row.get(2)))
        .unwrap();
    assert_eq!(source_summary, (2, 1, OVERSIZED_COPY_ROW_BYTES));

    let lake =
        create_test_lake("table_copy_preserves_identical_oversized_rows_across_batches").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&source_table_name).unwrap();
    let store = NotifyingStore::new();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline_with_batch_config(
        &database.config,
        random(),
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
        BatchConfig { max_fill_ms: 1000, memory_budget_ratio: 0.2, max_bytes: 8 * 1024 * 1024 },
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_payload_summary(&conn, &ducklake_table_name),
        (2, 1, i64::from(OVERSIZED_COPY_ROW_BYTES))
    );
    assert_eq!(count_applied_batches(&conn, &ducklake_table_name, "copy"), 2);
    assert_eq!(count_applied_batches(&conn, &ducklake_table_name, "copy_complete"), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn table_copy_reset_drops_destination_table_before_recopy() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let database_schema = setup_test_database_schema(&database, TableSelection::UsersOnly).await;
    let lake = create_test_lake("table_copy_reset_drops_destination_table_before_recopy").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();

    let users_schema = database_schema.users_schema();
    let users_table_name = table_name_to_ducklake_table_name(&users_schema.name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();

    database
        .insert_values(users_schema.name.clone(), &["name", "age"], &[&"before_1", &1])
        .await
        .unwrap();
    database
        .insert_values(users_schema.name.clone(), &["name", "age"], &[&"before_2", &2])
        .await
        .unwrap();

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;

    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    let users_sync_complete_notify = store.notify_on_table_sync_complete(users_schema.id).await;

    pipeline.start().await.unwrap();

    users_sync_complete_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();

    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_user_rows(&conn, &users_table_name),
        vec![(1, "before_1".to_owned(), 1), (2, "before_2".to_owned(), 2)]
    );
    drop(conn);
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    database
        .run_sql(&format!("delete from {} where true", users_schema.name.as_quoted_identifier()))
        .await
        .unwrap();
    database
        .insert_values(users_schema.name.clone(), &["name", "age"], &[&"after", &3])
        .await
        .unwrap();
    store.reset_table_state(users_schema.id).await.unwrap();

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;

    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    let users_sync_complete_notify = store.notify_on_table_sync_complete(users_schema.id).await;

    pipeline.start().await.unwrap();

    users_sync_complete_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();

    assert!(destination.was_table_dropped_for_copy(users_schema.id).await);
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(query_user_rows(&conn, &users_table_name), vec![(3, "after".to_owned(), 3)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn table_copy_and_streaming_without_restart() {
    init_test_tracing();

    let mut database = spawn_source_database().await;
    let database_schema = setup_test_database_schema(&database, TableSelection::Both).await;
    let lake = create_test_lake("table_copy_and_streaming_without_restart").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();

    let users_table_name =
        table_name_to_ducklake_table_name(&database_schema.users_schema().name).unwrap();
    let orders_table_name =
        table_name_to_ducklake_table_name(&database_schema.orders_schema().name).unwrap();

    insert_mock_data(
        &mut database,
        &database_schema.users_schema().name,
        &database_schema.orders_schema().name,
        1..=2,
        false,
    )
    .await;

    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    let users_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.users_schema().id).await;
    let orders_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.orders_schema().id).await;

    pipeline.start().await.unwrap();

    users_sync_complete_notify.notified().await;
    orders_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Insert, database_schema.users_schema().id, 2),
            EventCondition::TableCount(EventType::Insert, database_schema.orders_schema().id, 2),
        ])
        .await;

    insert_mock_data(
        &mut database,
        &database_schema.users_schema().name,
        &database_schema.orders_schema().name,
        3..=4,
        false,
    )
    .await;

    events_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_user_rows(&conn, &users_table_name),
        vec![
            (1, "user_1".to_owned(), 1),
            (2, "user_2".to_owned(), 2),
            (3, "user_3".to_owned(), 3),
            (4, "user_4".to_owned(), 4),
        ]
    );
    assert_eq!(
        query_order_rows(&conn, &orders_table_name),
        vec![
            (1, "description_1".to_owned()),
            (2, "description_2".to_owned()),
            (3, "description_3".to_owned()),
            (4, "description_4".to_owned()),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn table_insert_update_delete() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let database_schema = setup_test_database_schema(&database, TableSelection::UsersOnly).await;
    let lake = create_test_lake("table_insert_update_delete").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();

    let users_table_name =
        table_name_to_ducklake_table_name(&database_schema.users_schema().name).unwrap();

    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let users_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.users_schema().id).await;

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    pipeline.start().await.unwrap();
    users_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![EventCondition::TableCount(
            EventType::Insert,
            database_schema.users_schema().id,
            1,
        )])
        .await;

    database
        .insert_values(
            database_schema.users_schema().name.clone(),
            &["name", "age"],
            &[&"user_1", &1],
        )
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(query_user_rows(&conn, &users_table_name), vec![(1, "user_1".to_owned(), 1)]);
    drop(conn);

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    pipeline.start().await.unwrap();

    let events_notify = destination
        .wait_for_events(vec![EventCondition::TableCount(
            EventType::Update,
            database_schema.users_schema().id,
            1,
        )])
        .await;

    database
        .update_values(
            database_schema.users_schema().name.clone(),
            &["name", "age"],
            &[&"user_10", &10i32],
        )
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(query_user_rows(&conn, &users_table_name), vec![(1, "user_10".to_owned(), 10)]);
    drop(conn);

    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    pipeline.start().await.unwrap();

    let events_notify = destination
        .wait_for_events(vec![EventCondition::TableCount(
            EventType::Delete,
            database_schema.users_schema().id,
            1,
        )])
        .await;

    database
        .delete_values(database_schema.users_schema().name.clone(), &["id"], &["1"], "")
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert!(query_user_rows(&conn, &users_table_name).is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn cdc_streaming_with_truncate() {
    init_test_tracing();

    let mut database = spawn_source_database().await;
    let database_schema = setup_test_database_schema(&database, TableSelection::Both).await;
    let lake = create_test_lake("cdc_streaming_with_truncate").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();

    let users_table_name =
        table_name_to_ducklake_table_name(&database_schema.users_schema().name).unwrap();
    let orders_table_name =
        table_name_to_ducklake_table_name(&database_schema.orders_schema().name).unwrap();

    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;

    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        database_schema.publication_name(),
        store.clone(),
        destination.clone(),
    );

    let users_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.users_schema().id).await;
    let orders_sync_complete_notify =
        store.notify_on_table_sync_complete(database_schema.orders_schema().id).await;

    pipeline.start().await.unwrap();

    users_sync_complete_notify.notified().await;
    orders_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Insert, database_schema.users_schema().id, 2),
            EventCondition::TableCount(EventType::Insert, database_schema.orders_schema().id, 2),
        ])
        .await;

    insert_mock_data(
        &mut database,
        &database_schema.users_schema().name,
        &database_schema.orders_schema().name,
        1..=2,
        false,
    )
    .await;

    events_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Truncate, database_schema.users_schema().id, 1),
            EventCondition::TableCount(EventType::Truncate, database_schema.orders_schema().id, 1),
        ])
        .await;

    database.truncate_table(database_schema.users_schema().name.clone()).await.unwrap();
    database.truncate_table(database_schema.orders_schema().name.clone()).await.unwrap();

    events_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Insert, database_schema.users_schema().id, 4),
            EventCondition::TableCount(EventType::Insert, database_schema.orders_schema().id, 4),
        ])
        .await;

    insert_mock_data(
        &mut database,
        &database_schema.users_schema().name,
        &database_schema.orders_schema().name,
        3..=4,
        false,
    )
    .await;

    events_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_user_rows(&conn, &users_table_name),
        vec![(3, "user_3".to_owned(), 3), (4, "user_4".to_owned(), 4),]
    );
    assert_eq!(
        query_order_rows(&conn, &orders_table_name),
        vec![(3, "description_3".to_owned()), (4, "description_4".to_owned()),]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_change_add_column() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_schema_add_col");
    let table_id = database
        .create_table(
            table_name.clone(),
            true,
            &[("name", "text not null"), ("age", "integer not null")],
        )
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_schema_add";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, age) values ('Alice', 25)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    let lake = create_test_lake("schema_change_add_column").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let initial_metadata = store.get_destination_table_metadata(table_id).await.unwrap().unwrap();
    assert!(initial_metadata.is_applied());
    let initial_snapshot_id = initial_metadata.snapshot_id();

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;

    database
        .alter_table(
            table_name.clone(),
            &[
                TableModification::AddColumn { name: "email", data_type: "text" },
                TableModification::AddColumn {
                    name: "score",
                    data_type: "integer not null default 0",
                },
            ],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, age, email, score) values ('Bob', 30, 'bob@example.com', 7)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let final_metadata = store.get_destination_table_metadata(table_id).await.unwrap().unwrap();
    assert!(final_metadata.is_applied());
    assert!(final_metadata.snapshot_id() > initial_snapshot_id);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_schema_add_rows(&conn, &ducklake_table_name),
        vec![
            SchemaAddRow { id: 1, name: "Alice".to_owned(), age: 25, email: None, score: Some(0) },
            SchemaAddRow {
                id: 2,
                name: "Bob".to_owned(),
                age: 30,
                email: Some("bob@example.com".to_owned()),
                score: Some(7),
            },
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn case_only_rename_keeps_canonical_ducklake_column() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_case_only_rename");
    let table_id = database
        .create_table(table_name.clone(), true, &[("display_name", "text not null")])
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "alter table {} rename column {} to {}",
            table_name.as_quoted_identifier(),
            quote_identifier("display_name"),
            quote_identifier("DISPLAY_NAME")
        ))
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_case_only_rename";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();
    database
        .run_sql(&format!(
            "insert into {} ({}) values ('Alice')",
            table_name.as_quoted_identifier(),
            quote_identifier("DISPLAY_NAME")
        ))
        .await
        .unwrap();

    let lake = create_test_lake("case_only_rename_keeps_canonical_ducklake_column").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        random(),
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;
    database
        .run_sql(&format!(
            "alter table {} rename column {} to {}",
            table_name.as_quoted_identifier(),
            quote_identifier("DISPLAY_NAME"),
            quote_identifier("display_name")
        ))
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} ({}) values ('Bob')",
            table_name.as_quoted_identifier(),
            quote_identifier("display_name")
        ))
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(query_table_columns(&conn, &ducklake_table_name), ["id", "display_name"]);
    assert_eq!(
        query_case_mapped_rows(&conn, &ducklake_table_name),
        [(1, "Alice".to_owned()), (2, "Bob".to_owned())]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn publication_mask_adds_existing_defaulted_column_without_backfill() {
    init_test_tracing();

    let database = spawn_source_database().await;
    if below_version!(database.server_version(), POSTGRES_15) {
        eprintln!("Skipping test: PostgreSQL 15+ required for publication column lists");
        return;
    }

    let table_name = test_table_name("ducklake_publication_add_col");
    let table_id = database
        .create_table(
            table_name.clone(),
            true,
            &[("name", "text not null"), ("status", "text not null default 'pending'::text")],
        )
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_publication_add";
    database
        .run_sql(&format!(
            "create publication {} for table {} (id, name)",
            quote_identifier(publication_name),
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name) values ('Alice')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    let lake = create_test_lake("publication_mask_adds_existing_defaulted_column").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;

    database
        .run_sql(&format!(
            "alter publication {} set table {} (id, name, status)",
            quote_identifier(publication_name),
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, status) values ('Bob', 'replicated')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_publication_add_rows(&conn, &ducklake_table_name),
        vec![(1, "Alice".to_owned(), None), (2, "Bob".to_owned(), Some("replicated".to_owned())),]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_change_is_visible_to_already_open_connection() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_schema_open_conn");
    let table_id = database
        .create_table(
            table_name.clone(),
            true,
            &[("name", "text not null"), ("age", "integer not null")],
        )
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_schema_open_conn";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, age) values ('Alice', 25)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    let lake = create_test_lake("schema_change_is_visible_to_already_open_connection").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    checkpoint_lake(&catalog_url, &data_url);
    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(query_table_columns(&conn, &ducklake_table_name), vec!["id", "name", "age"]);

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;

    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "email", data_type: "text" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, age, email) values ('Bob', 30, 'bob@example.com')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    assert_eq!(
        query_table_columns(&conn, &ducklake_table_name),
        vec!["id", "name", "age", "email"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_change_ordered_name_reuse() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_schema_multi");
    let table_id = database
        .create_table(
            table_name.clone(),
            true,
            &[("name", "text not null"), ("age", "integer not null"), ("status", "text")],
        )
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_schema_multi";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, age, status) values ('Alice', 25, 'active')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    let lake = create_test_lake("schema_change_ordered_name_reuse").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;

    database
        .alter_table(
            table_name.clone(),
            &[TableModification::RenameColumn {
                old_name: "name",
                new_name: "supabase_etl_source_swap",
            }],
        )
        .await
        .unwrap();
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::RenameColumn { old_name: "status", new_name: "name" }],
        )
        .await
        .unwrap();
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::RenameColumn {
                old_name: "supabase_etl_source_swap",
                new_name: "status",
            }],
        )
        .await
        .unwrap();
    database
        .alter_table(
            table_name.clone(),
            &[
                TableModification::DropColumn { name: "age" },
                TableModification::AddColumn { name: "age", data_type: "text" },
            ],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (status, name, age) values ('Bob', 'pending', 'thirty')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_table_columns(&conn, &ducklake_table_name),
        vec!["id", "status", "name", "age"]
    );
    assert_eq!(
        query_schema_multi_rows(&conn, &ducklake_table_name),
        vec![
            SchemaMultiRow {
                id: 1,
                status: "Alice".to_owned(),
                name: Some("active".to_owned()),
                age: None,
            },
            SchemaMultiRow {
                id: 2,
                status: "Bob".to_owned(),
                name: Some("pending".to_owned()),
                age: Some("thirty".to_owned()),
            },
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_change_then_update_and_delete() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_schema_mutation");
    let table_id = database
        .create_table(
            table_name.clone(),
            true,
            &[("name", "text not null"), ("visits", "integer not null")],
        )
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_schema_mutation";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();
    database
        .run_sql(&format!(
            "insert into {} (name, visits) values ('Alice', 1)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    let lake = create_test_lake("schema_change_then_update_and_delete").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
            EventCondition::TableCount(EventType::Update, table_id, 1),
            EventCondition::TableCount(EventType::Delete, table_id, 1),
        ])
        .await;

    database
        .alter_table(
            table_name.clone(),
            &[TableModification::RenameColumn { old_name: "name", new_name: "full_name" }],
        )
        .await
        .unwrap();
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "email", data_type: "text" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (full_name, visits, email) values ('Bob', 1, 'bob@example.com')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "update {} set full_name = 'Alice Smith', visits = 2, email = 'alice@example.com' \
             where id = 1",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "delete from {} where full_name = 'Bob'",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();

    events_notify.notified().await;
    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_table_columns(&conn, &ducklake_table_name),
        vec!["id", "full_name", "visits", "email"]
    );
    assert_eq!(
        query_schema_mutation_rows(&conn, &ducklake_table_name),
        vec![SchemaMutationRow {
            id: 1,
            full_name: "Alice Smith".to_owned(),
            visits: 2,
            email: Some("alice@example.com".to_owned()),
        }]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_change_matches_simulator_generated_column_rotation() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_schema_simulator_rotation");
    let table_id = database
        .create_table(table_name.clone(), true, &[("text_col", "text not null")])
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_schema_simulator_rotation";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();

    let lake = create_test_lake("schema_change_matches_simulator_generated_column_rotation").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_0_0", data_type: "text" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0) values ('after_add', 'slot0_v0')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 2),
            EventCondition::TableCount(EventType::Insert, table_id, 2),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::RenameColumn { old_name: "ddl_col_0_0", new_name: "ddl_col_0_1" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_1) values ('after_rename', 'slot0_v1')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 3),
            EventCondition::TableCount(EventType::Insert, table_id, 3),
        ])
        .await;
    database
        .alter_table(table_name.clone(), &[TableModification::DropColumn { name: "ddl_col_0_1" }])
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col) values ('after_drop')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 4),
            EventCondition::TableCount(EventType::Insert, table_id, 4),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_0_0", data_type: "text" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0) values ('after_readd', 'slot0_readded')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_table_columns(&conn, &ducklake_table_name),
        vec!["id", "text_col", "ddl_col_0_0"]
    );
    assert_eq!(
        query_simulator_ddl_rotation_rows(&conn, &ducklake_table_name),
        vec![
            SimulatorDdlRotationRow { id: 1, text_col: "after_add".to_owned(), ddl_col_0_0: None },
            SimulatorDdlRotationRow {
                id: 2,
                text_col: "after_rename".to_owned(),
                ddl_col_0_0: None,
            },
            SimulatorDdlRotationRow { id: 3, text_col: "after_drop".to_owned(), ddl_col_0_0: None },
            SimulatorDdlRotationRow {
                id: 4,
                text_col: "after_readd".to_owned(),
                ddl_col_0_0: Some("slot0_readded".to_owned()),
            },
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_change_matches_simulator_generated_column_types() {
    init_test_tracing();

    let database = spawn_source_database().await;
    let table_name = test_table_name("ducklake_schema_simulator_types");
    let table_id = database
        .create_table(table_name.clone(), true, &[("text_col", "text not null")])
        .await
        .unwrap();
    let publication_name = "test_pub_ducklake_schema_simulator_types";
    database.create_publication(publication_name, std::slice::from_ref(&table_name)).await.unwrap();

    let lake = create_test_lake("schema_change_matches_simulator_generated_column_types").await;
    let catalog_url = lake.catalog_url.clone();
    let data_url = lake.data_url.clone();
    let ducklake_table_name = table_name_to_ducklake_table_name(&table_name).unwrap();
    let store = NotifyingStore::new();
    let pipeline_id: PipelineId = random();
    let destination = build_destination(&catalog_url, &data_url, store.clone()).await;
    let mut pipeline = create_pipeline(
        &database.config,
        pipeline_id,
        publication_name.to_owned(),
        store.clone(),
        destination.clone(),
    );
    let table_sync_complete_notify = store.notify_on_table_sync_complete(table_id).await;

    pipeline.start().await.unwrap();
    table_sync_complete_notify.notified().await;

    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 1),
            EventCondition::TableCount(EventType::Insert, table_id, 1),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_0_0", data_type: "text" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0) values ('after_text', 'text_0')",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 2),
            EventCondition::TableCount(EventType::Insert, table_id, 2),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_1_0", data_type: "integer" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0, ddl_col_1_0) values ('after_integer', \
             'text_1', 11)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 3),
            EventCondition::TableCount(EventType::Insert, table_id, 3),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_2_0", data_type: "boolean" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0, ddl_col_1_0, ddl_col_2_0) values \
             ('after_boolean', 'text_2', 22, true)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 4),
            EventCondition::TableCount(EventType::Insert, table_id, 4),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_3_0", data_type: "timestamptz" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0, ddl_col_1_0, ddl_col_2_0, ddl_col_3_0) values \
             ('after_timestamptz', 'text_3', 33, false, '2026-01-02 03:04:05+00'::timestamptz)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;
    let events_notify = destination
        .wait_for_events(vec![
            EventCondition::TableCount(EventType::Relation, table_id, 5),
            EventCondition::TableCount(EventType::Insert, table_id, 5),
        ])
        .await;
    database
        .alter_table(
            table_name.clone(),
            &[TableModification::AddColumn { name: "ddl_col_4_0", data_type: "numeric(10, 2)" }],
        )
        .await
        .unwrap();
    database
        .run_sql(&format!(
            "insert into {} (text_col, ddl_col_0_0, ddl_col_1_0, ddl_col_2_0, ddl_col_3_0, \
             ddl_col_4_0) values ('after_numeric', 'text_4', 44, true, '2026-01-02 \
             03:04:05+00'::timestamptz, 12345.67)",
            table_name.as_quoted_identifier()
        ))
        .await
        .unwrap();
    events_notify.notified().await;

    pipeline.shutdown_and_wait().await.unwrap();
    drop(destination);
    checkpoint_lake(&catalog_url, &data_url);

    let conn = open_lake_conn(&catalog_url, &data_url);
    assert_eq!(
        query_table_columns(&conn, &ducklake_table_name),
        vec![
            "id",
            "text_col",
            "ddl_col_0_0",
            "ddl_col_1_0",
            "ddl_col_2_0",
            "ddl_col_3_0",
            "ddl_col_4_0",
        ]
    );
    assert_eq!(
        query_simulator_ddl_type_row(&conn, &ducklake_table_name),
        SimulatorDdlTypeRow {
            id: 5,
            text_col: "after_numeric".to_owned(),
            ddl_col_0_0: Some("text_4".to_owned()),
            ddl_col_1_0: Some(44),
            ddl_col_2_0: Some(true),
            ddl_col_3_0_is_present: true,
            ddl_col_4_0: Some(Decimal::new(1234567, 2)),
        }
    );
}

/// Embeds the stock destination with host setup, routing and maintenance.
#[tokio::test(flavor = "multi_thread")]
async fn embedded_destination_keeps_copy_cdc_and_maintenance_on_one_instance() {
    use std::time::Duration;

    use etl::{error::ErrorKind, etl_error};

    init_test_tracing();
    let database = spawn_source_database().await;
    let schema = setup_test_database_schema(&database, TableSelection::UsersOnly).await;
    let users = schema.users_schema();
    database.insert_values(users.name.clone(), &["name", "age"], &[&"before", &1]).await.unwrap();
    let lake = create_test_lake("embedded_destination").await;
    let store = NotifyingStore::new();
    let setup = format!(
        "{} attach {} as lake (data_path {}, data_inlining_row_limit 0); create table host_marker \
         as select 42 as id;",
        ducklake_load_sql(),
        quote_literal(&format!("ducklake:{}", catalog_attach_target(&lake.catalog_url))),
        quote_literal(lake.data_url.as_str()),
    );
    let raw = DuckLakeDestination::builder(lake.catalog_url.clone(), lake.data_url.clone(), 1, store.clone())
        .connection_initializer(move || {
            let conn = Connection::open_in_memory().map_err(|source| {
                etl_error!(ErrorKind::DestinationConnectionFailed, "Host open failed", source: source)
            })?;
            conn.execute_batch(&setup).map_err(|source| {
                etl_error!(ErrorKind::DestinationQueryFailed, "Host setup failed", source: source)
            })?;
            Ok(conn)
        })
        .table_name_mapper(|name| Ok(DuckLakeTableName::new("replicated", format!("{}_{}", name.schema, name.name))))
        .build().await.unwrap();
    let wrapped = TestDestinationWrapper::wrap(raw.clone());
    let ready = store.notify_on_table_sync_complete(users.id).await;
    let mut pipeline = create_pipeline(
        &database.config,
        random(),
        schema.publication_name(),
        store.clone(),
        wrapped.clone(),
    );
    pipeline.start().await.unwrap();
    ready.notified().await;

    // Cancelling the caller must not release the pause around native work.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let maintenance_destination = raw.clone();
    let maintenance = tokio::spawn(async move {
        maintenance_destination.run_maintenance(Duration::from_secs(30), move |conn| {
            let marker: i64 = conn.query_row("select id from host_marker", [], |row| row.get(0)).map_err(|source| {
                etl_error!(ErrorKind::DestinationQueryFailed, "Marker query failed", source: source)
            })?;
            assert_eq!(marker, 42);
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            Ok(())
        }).await
    });
    started_rx.await.unwrap();
    maintenance.abort();
    assert!(maintenance.await.unwrap_err().is_cancelled());
    assert!(
        tokio::time::timeout(Duration::from_millis(50), raw.acquire_external_maintenance_pause())
            .await
            .is_err()
    );
    release_tx.send(()).unwrap();
    drop(
        tokio::time::timeout(Duration::from_secs(10), raw.acquire_external_maintenance_pause())
            .await
            .unwrap(),
    );

    let inserted = wrapped
        .wait_for_events(vec![EventCondition::TableCount(EventType::Insert, users.id, 1)])
        .await;
    database.insert_values(users.name.clone(), &["name", "age"], &[&"after", &2]).await.unwrap();
    inserted.notified().await;
    let mapped_for_inline_check =
        DuckLakeTableName::new("replicated", format!("{}_{}", users.name.schema, users.name.name));
    let flushed = raw.run_maintenance(Duration::from_secs(30), move |conn| {
        let sql = format!("select coalesce(sum(rows_flushed), 0) from ducklake_flush_inlined_data('lake', {}, schema => {})", quote_literal(mapped_for_inline_check.table()), quote_literal(mapped_for_inline_check.schema()));
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0)).map_err(|source| {
            etl_error!(ErrorKind::DestinationQueryFailed, "Inline verification failed", source: source)
        })
    }).await.unwrap();
    assert_eq!(flushed, 0);
    pipeline.shutdown_and_wait().await.unwrap();
    drop(wrapped);
    drop(raw);
    checkpoint_lake(&lake.catalog_url, &lake.data_url);
    let conn = open_lake_conn(&lake.catalog_url, &lake.data_url);
    let mapped =
        DuckLakeTableName::new("replicated", format!("{}_{}", users.name.schema, users.name.name));
    assert_eq!(
        query_user_rows(&conn, &mapped),
        vec![(1, "before".to_owned(), 1), (2, "after".to_owned(), 2)]
    );
}
