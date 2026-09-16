//! Table batches are the atomic per-table write units used by DuckLake writes.
//! Copy, mutation, and truncate inputs are normalized into prepared batches so
//! each attempt can replay the same SQL and replay bookkeeping. Copy batches
//! receive opaque upstream ids and retain them across destination-local
//! retries. Streaming mutation and truncate batches advance a per-table
//! progress watermark.
//! Bounded batch sizes preserve table-local ordering without letting one
//! transaction grow unbounded.

#[cfg(feature = "test-utils")]
use std::sync::LazyLock;
use std::{
    collections::HashMap,
    error, fmt,
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use etl::{
    data::{Cell, OldTableRow, PartialTableRow, SizeHint, TableRow, UpdatedTableRow},
    destination::TableCopyBatchId,
    error::{ErrorKind, EtlResult},
    etl_error,
    event::EventSequenceKey,
    schema::ReplicatedTableSchema,
};
use metrics::{counter, histogram};
#[cfg(feature = "test-utils")]
use parking_lot::Mutex;
use pg_escape::quote_literal;
use rand::Rng;
use tokio::{sync::Semaphore, time::Instant};
use tokio_postgres::types::PgLsn;
use tracing::{debug, trace, warn};

use crate::{
    ducklake::{
        DUCKLAKE_COLUMN_NAME_MAPPING, DuckLakeTableName, LAKE_CATALOG,
        client::{
            DuckLakeBlockingOperationContext, DuckLakeConnectionManager, format_query_error_detail,
            is_ducklake_shutdown_requested_error, run_duckdb_blocking,
            run_duckdb_blocking_with_context,
        },
        core::is_create_table_conflict,
        encoding::{
            PreparedRows, cell_to_sql_literal_ref, prepare_copy_rows, prepare_rows,
            table_row_to_sql_literal_ref,
        },
        metrics::{
            BATCH_KIND_LABEL, DELETE_ORIGIN_LABEL, ETL_DUCKLAKE_BATCH_COMMIT_DURATION_SECONDS,
            ETL_DUCKLAKE_BATCH_PREPARED_MUTATIONS, ETL_DUCKLAKE_DELETE_PREDICATES,
            ETL_DUCKLAKE_FAILED_BATCHES_TOTAL, ETL_DUCKLAKE_REPLAYED_BATCHES_TOTAL,
            ETL_DUCKLAKE_RETRIES_TOTAL, ETL_DUCKLAKE_UPSERT_ROWS, PREPARED_ROWS_KIND_LABEL,
            RETRY_SCOPE_LABEL, SUB_BATCH_KIND_LABEL,
        },
        replay_epoch::LEGACY_REPLAY_EPOCH,
        sql::{qualified_lake_table_name, quote_identifier},
    },
    retry::{RetryAttempt, RetryDecision, RetryPolicy, retry_with_backoff},
};

/// Maximum number of rows per SQL `INSERT ... VALUES` batch when nested values
/// force the staging path to bypass DuckDB's appender API.
const SQL_INSERT_BATCH_SIZE: usize = 128;
/// Maximum number of primary-key predicates per SQL `DELETE` batch.
///
/// Keep this small so each delete statement remains cheap while still avoiding
/// one round-trip per deleted row.
const SQL_DELETE_BATCH_SIZE: usize = 16;
/// Maximum number of ordered CDC mutations grouped into one atomic DuckLake
/// transaction.
///
/// Keeping mixed insert/delete/update streams in the same batch improves
/// insert throughput on interleaved workloads while still capping transaction
/// lifetime for DuckLake conflict handling.
const CDC_MUTATION_BATCH_SIZE: usize = 16;
/// ETL-managed marker table storing per-table applied copy batches.
const APPLIED_BATCHES_TABLE: &str = "__etl_applied_table_batches";
/// Data inlining limit for append-only DuckLake helper tables.
///
/// This per-table option intentionally overrides the COPY pool's attach-level
/// limit of zero. Helper markers and progress stay inline, while rows written
/// to the replicated table during COPY still become Parquet files. Maintenance
/// performs helper-table deletions while foreground mutations are paused.
const HELPER_TABLE_DATA_INLINING_ROW_LIMIT: usize = 256;
/// Replay epoch column shared by ETL helper tables.
const REPLAY_EPOCH_COLUMN: &str = "replay_epoch";

/// Formats an optional LSN without using debug output.
fn format_optional_lsn(lsn: Option<PgLsn>) -> String {
    lsn.map_or_else(|| "none".to_owned(), |lsn| lsn.to_string())
}

/// Formats a sequence key using DuckLake's existing fixed-width hexadecimal
/// representation.
fn format_sequence_key(sequence_key: EventSequenceKey) -> String {
    let commit_lsn = u64::from(sequence_key.commit_lsn);
    format!("{commit_lsn:016x}/{:016x}", sequence_key.tx_ordinal)
}

/// Formats an optional sequence key without using debug output.
fn format_optional_sequence_key(sequence_key: Option<EventSequenceKey>) -> String {
    sequence_key.map_or_else(|| "none".to_owned(), format_sequence_key)
}

/// Returns whether one DuckDB error is the standard interrupted query error.
fn is_duckdb_interrupt_error(error: &duckdb::Error) -> bool {
    error.to_string().contains("INTERRUPT Error: Interrupted")
}

/// Query failure whose value-bearing diagnostics require explicit opt-in.
struct DuckDbSensitiveQueryError {
    error: duckdb::Error,
    sql: String,
}
impl fmt::Display for DuckDbSensitiveQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if cfg!(feature = "ducklake-query-error-details") {
            write!(f, "{}; SQL: {}", self.error, self.sql)
        } else {
            write!(
                f,
                "DuckDB query failed; error message omitted because it may contain row values"
            )
        }
    }
}
impl fmt::Debug for DuckDbSensitiveQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl error::Error for DuckDbSensitiveQueryError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        if cfg!(feature = "ducklake-query-error-details") { Some(&self.error) } else { None }
    }
}

/// Formats query context for a delete mutation without row values.
fn format_delete_mutation_error_detail(
    target_table: &str,
    predicate_count: usize,
    chunk_index: usize,
    chunk_count: usize,
    chunk_predicate_count: usize,
) -> String {
    format!(
        "sql: DELETE FROM {target_table} WHERE [redacted predicates]; predicate_count: \
         {predicate_count}; chunk_index: {chunk_index}; chunk_count: {chunk_count}; \
         chunk_predicate_count: {chunk_predicate_count}"
    )
}

/// Formats query context for an update mutation without row values.
fn format_update_mutation_error_detail(
    target_table: &str,
    assignment_count: usize,
    has_predicate: bool,
) -> String {
    format!(
        "sql: UPDATE {target_table} SET [redacted assignments] WHERE [redacted predicate]; \
         assignment_count: {assignment_count}; has_predicate: {has_predicate}"
    )
}

/// ETL-managed per-table streaming replay progress for steady-state CDC
/// retries.
const STREAMING_PROGRESS_TABLE: &str = "__etl_streaming_progress";
/// Maximum number of times a failed write attempt is retried before giving up.
const MAX_COMMIT_RETRIES: u32 = 10;
/// Initial backoff duration before the first retry.
const INITIAL_RETRY_DELAY_MS: u64 = 50;
/// Upper bound on backoff duration.
const MAX_RETRY_DELAY_MS: u64 = 2_000;
/// Minimum retry delay for transient delete-file visibility failures.
const TRANSIENT_DELETE_FILE_RETRY_DELAY_MS: u64 = 5_000;

/// Decides whether DuckLake-owned retry loops should retry one failure.
fn ducklake_retry_decision(error: &etl::error::EtlError) -> RetryDecision {
    if is_ducklake_shutdown_requested_error(error) {
        RetryDecision::Stop
    } else {
        RetryDecision::Retry
    }
}

/// Event-level table mutations that must be applied in order.
pub(super) enum TableMutation {
    Insert(TableRow),
    Delete(OldTableRow),
    Update { delete_row: OldTableRow, new_row: UpdatedTableRow },
    Replace(TableRow),
}

/// Prepared table mutations ready for execution and retries.
enum PreparedTableMutation {
    Upsert(PreparedRows),
    Delete {
        // For WHERE clause predicates used in DELETE statements.
        predicates: Vec<String>,
        // To know if it's coming from an update or delete operation.
        origin: &'static str,
    },
    Update {
        // For SET clause assignments used in UPDATE statements. Example: "value=1, id=3"
        assignments: Vec<String>,
        // For the WHERE clause predicate used in the UPDATE statement. Example: "id = 4 AND
        // content = 'hello'"
        predicate: String,
    },
}

/// Borrowed row shape used to build delete predicates.
enum DeletePredicateRowRef<'a> {
    Full(&'a TableRow),
    Key(&'a TableRow),
}

impl<'a> From<&'a TableRow> for DeletePredicateRowRef<'a> {
    fn from(value: &'a TableRow) -> Self {
        Self::Full(value)
    }
}

impl<'a> From<&'a OldTableRow> for DeletePredicateRowRef<'a> {
    fn from(value: &'a OldTableRow) -> Self {
        match value {
            OldTableRow::Full(row) => Self::Full(row),
            OldTableRow::Key(row) => Self::Key(row),
        }
    }
}

/// Event-level table mutation annotated for idempotent replay.
pub(super) struct TrackedTableMutation {
    sequence_key: EventSequenceKey,
    mutation: TableMutation,
}

impl TrackedTableMutation {
    /// Creates one tracked mutation preserved for retry-safe replay.
    pub(super) fn new(sequence_key: EventSequenceKey, mutation: TableMutation) -> Self {
        Self { sequence_key, mutation }
    }

    /// Returns the stable event sequence key for this mutation.
    fn sequence_key(&self) -> EventSequenceKey {
        self.sequence_key
    }
}

/// Truncate event metadata preserved for idempotent replay.
#[derive(Clone, Copy)]
pub(super) struct TrackedTruncateEvent {
    sequence_key: EventSequenceKey,
    options: i8,
}

impl TrackedTruncateEvent {
    /// Creates one tracked truncate event preserved for retry-safe replay.
    pub(super) fn new(sequence_key: EventSequenceKey, options: i8) -> Self {
        Self { sequence_key, options }
    }

    /// Returns the stable event sequence key for this truncate.
    fn sequence_key(&self) -> EventSequenceKey {
        self.sequence_key
    }
}

/// Stable hash used to derive per-table batch identifiers.
struct BatchIdHasher(u64);

impl BatchIdHasher {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }
}

impl Default for BatchIdHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for BatchIdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
}

/// Atomic DuckLake batch kinds used by replay bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DuckLakeTableBatchKind {
    Copy,
    CopyComplete,
    Mutation,
    Truncate,
}

impl DuckLakeTableBatchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::CopyComplete => "copy_complete",
            Self::Mutation => "mutation",
            Self::Truncate => "truncate",
        }
    }
}

/// Deterministic identity for one table batch.
struct DuckLakeBatchIdentity {
    batch_id: String,
    first_start_lsn: Option<PgLsn>,
    last_commit_lsn: Option<PgLsn>,
}

/// Returns destination-visible column names in replicated write order.
fn replicated_column_names(replicated_table_schema: &ReplicatedTableSchema) -> Vec<String> {
    replicated_table_schema
        .destination_column_schemas(DUCKLAKE_COLUMN_NAME_MAPPING)
        .map(|column| column.name)
        .collect()
}

/// Prepared per-table work executed atomically in one DuckLake transaction.
enum PreparedDuckLakeTableBatchAction {
    Mutation(Vec<PreparedTableMutation>),
    Truncate,
}

/// Prepared atomic DuckLake table batch with replay metadata.
pub(super) struct PreparedDuckLakeTableBatch {
    table_name: DuckLakeTableName,
    replay_epoch: String,
    batch_id: String,
    batch_kind: DuckLakeTableBatchKind,
    first_start_lsn: Option<PgLsn>,
    last_commit_lsn: Option<PgLsn>,
    first_sequence_key: Option<EventSequenceKey>,
    last_sequence_key: Option<EventSequenceKey>,
    insert_column_names: Vec<String>,
    action: PreparedDuckLakeTableBatchAction,
}

/// Compact durable marker retained while COPY rows remain staged.
struct PreparedDuckLakeBatchMarker {
    batch_kind: DuckLakeTableBatchKind,
    first_start_lsn: Option<PgLsn>,
    last_commit_lsn: Option<PgLsn>,
}

/// Prepared initial-copy rows with a lightweight size estimate for buffering.
pub(super) struct PreparedDuckLakeCopyBatch {
    batch: PreparedDuckLakeTableBatch,
    estimated_bytes: u64,
}

impl PreparedDuckLakeCopyBatch {
    /// Returns the approximate decoded bytes owned by this row batch.
    pub(super) fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }

    /// Converts this prepared copy payload into the existing atomic-batch path.
    pub(super) fn into_atomic_batch(self) -> PreparedDuckLakeTableBatch {
        self.batch
    }

    /// Separates staged row data from its durable applied-batch marker.
    fn into_buffer_parts(
        mut self,
    ) -> EtlResult<(String, PreparedDuckLakeBatchMarker, PreparedRows)> {
        let PreparedDuckLakeTableBatchAction::Mutation(prepared_mutations) = &mut self.batch.action
        else {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake buffered copy batch is not a mutation"
            ));
        };
        if prepared_mutations.len() != 1 {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake buffered copy batch has an invalid mutation count",
                format!("Expected 1 mutation, got {}", prepared_mutations.len())
            ));
        }
        let prepared_mutation = prepared_mutations.pop().ok_or_else(|| {
            etl_error!(ErrorKind::InvalidState, "DuckLake buffered copy mutation is missing")
        })?;
        let PreparedTableMutation::Upsert(prepared_rows) = prepared_mutation else {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake buffered copy mutation is not an upsert"
            ));
        };

        let marker = PreparedDuckLakeBatchMarker {
            batch_kind: self.batch.batch_kind,
            first_start_lsn: self.batch.first_start_lsn,
            last_commit_lsn: self.batch.last_commit_lsn,
        };

        Ok((self.batch.batch_id, marker, prepared_rows))
    }
}

impl PreparedDuckLakeTableBatch {
    /// Returns the destination table this batch targets.
    pub(super) fn table_name(&self) -> &DuckLakeTableName {
        &self.table_name
    }

    /// Returns whether this batch uses the streaming progress replay path.
    fn uses_streaming_progress(&self) -> bool {
        matches!(
            self.batch_kind,
            DuckLakeTableBatchKind::Mutation | DuckLakeTableBatchKind::Truncate
        )
    }
}

/// One table-local streaming replay watermark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TableStreamingProgress {
    last_sequence_key: EventSequenceKey,
}

/// Ensures the ETL-managed replay marker table exists.
pub(super) async fn ensure_applied_batches_table_exists(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    table_creation_slots: Arc<Semaphore>,
    applied_batches_table_created: Arc<AtomicBool>,
) -> EtlResult<()> {
    if applied_batches_table_created.load(Ordering::Relaxed) {
        return Ok(());
    }

    let _table_creation_permit = table_creation_slots.acquire_owned().await.map_err(|_| {
        etl_error!(ErrorKind::InvalidState, "DuckLake table creation semaphore closed")
    })?;

    if applied_batches_table_created.load(Ordering::Relaxed) {
        return Ok(());
    }

    let ddl = format!(
        r#"CREATE TABLE IF NOT EXISTS {LAKE_CATALOG}."{APPLIED_BATCHES_TABLE}" (
             table_name VARCHAR NOT NULL,
             replay_epoch VARCHAR,
             batch_id VARCHAR NOT NULL,
             batch_kind VARCHAR NOT NULL,
             first_start_lsn UBIGINT,
             last_commit_lsn UBIGINT,
             applied_at TIMESTAMPTZ NOT NULL
             );"#
    );
    let created = Arc::clone(&applied_batches_table_created);
    let table_name = APPLIED_BATCHES_TABLE.to_owned();

    run_duckdb_blocking(pool, blocking_slots, move |conn| -> EtlResult<()> {
        match conn.execute_batch(&ddl) {
            Ok(()) => {}
            Err(error) if is_create_table_conflict(&error, &table_name) => {}
            Err(error) => {
                return Err(etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake CREATE TABLE failed",
                    format_query_error_detail(&ddl),
                    source: error
                ));
            }
        }
        ensure_helper_table_replay_epoch_column(conn, APPLIED_BATCHES_TABLE)?;

        let set_option_sql = format!(
            "CALL {LAKE_CATALOG}.set_option('data_inlining_row_limit', {}, table_name => {});",
            HELPER_TABLE_DATA_INLINING_ROW_LIMIT,
            quote_literal(APPLIED_BATCHES_TABLE),
        );
        conn.execute_batch(&set_option_sql).map_err(|err| {
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake set_option failed",
                format_query_error_detail(&set_option_sql),
                source: err
            )
        })?;

        created.store(true, Ordering::Relaxed);
        Ok(())
    })
    .await
}

/// Ensures the ETL-managed streaming progress table exists.
pub(super) async fn ensure_streaming_progress_table_exists(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    table_creation_slots: Arc<Semaphore>,
    streaming_progress_table_created: Arc<AtomicBool>,
) -> EtlResult<()> {
    if streaming_progress_table_created.load(Ordering::Relaxed) {
        return Ok(());
    }

    let _table_creation_permit = table_creation_slots.acquire_owned().await.map_err(|_| {
        etl_error!(ErrorKind::InvalidState, "DuckLake table creation semaphore closed")
    })?;

    if streaming_progress_table_created.load(Ordering::Relaxed) {
        return Ok(());
    }

    let ddl = format!(
        r#"CREATE TABLE IF NOT EXISTS {LAKE_CATALOG}."{STREAMING_PROGRESS_TABLE}" (
             table_name VARCHAR NOT NULL,
             replay_epoch VARCHAR,
             last_commit_lsn UBIGINT NOT NULL,
             last_tx_ordinal UBIGINT NOT NULL,
             updated_at TIMESTAMPTZ NOT NULL
             );"#
    );
    let created = Arc::clone(&streaming_progress_table_created);
    let table_name = STREAMING_PROGRESS_TABLE.to_owned();

    run_duckdb_blocking(pool, blocking_slots, move |conn| -> EtlResult<()> {
        match conn.execute_batch(&ddl) {
            Ok(()) => {}
            Err(err) if is_create_table_conflict(&err, &table_name) => {}
            Err(err) => {
                return Err(etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake CREATE TABLE failed",
                    format_query_error_detail(&ddl),
                    source: err
                ));
            }
        }
        ensure_helper_table_replay_epoch_column(conn, STREAMING_PROGRESS_TABLE)?;

        let set_option_sql = format!(
            "CALL {LAKE_CATALOG}.set_option('data_inlining_row_limit', {}, table_name => {});",
            HELPER_TABLE_DATA_INLINING_ROW_LIMIT,
            quote_literal(STREAMING_PROGRESS_TABLE),
        );
        conn.execute_batch(&set_option_sql).map_err(|error| {
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake set_option failed",
                format_query_error_detail(&set_option_sql),
                source: error
            )
        })?;

        created.store(true, Ordering::Relaxed);
        Ok(())
    })
    .await
}

/// Adds the replay epoch column to helper tables created by older versions.
fn ensure_helper_table_replay_epoch_column(
    conn: &duckdb::Connection,
    table_name: &str,
) -> EtlResult<()> {
    if helper_table_has_column(conn, table_name, REPLAY_EPOCH_COLUMN)? {
        return Ok(());
    }

    let table_name = format!(r#"{LAKE_CATALOG}."{table_name}""#);
    let column_name = quote_identifier(REPLAY_EPOCH_COLUMN);
    let sql = format!("ALTER TABLE {table_name} ADD COLUMN {column_name} VARCHAR;");
    conn.execute_batch(&sql).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake helper table migration failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;

    Ok(())
}

fn helper_table_has_column(
    conn: &duckdb::Connection,
    table_name: &str,
    column_name: &str,
) -> EtlResult<bool> {
    let sql = format!(
        "SELECT 1 FROM information_schema.columns WHERE table_catalog = {} AND table_name = {} \
         AND column_name = {} LIMIT 1;",
        quote_literal(LAKE_CATALOG),
        quote_literal(table_name),
        quote_literal(column_name)
    );
    let mut statement = conn.prepare(&sql).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake helper table schema lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;
    let mut rows = statement.query([]).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake helper table schema lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;

    rows.next().map(|row| row.is_some()).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake helper table schema row fetch failed",
            format_query_error_detail(&sql),
            source: source
        )
    })
}

/// Applies all prepared atomic batches for one table, reusing one DuckDB
/// connection per attempt and skipping already committed segments.
pub(super) async fn apply_table_batches_with_retry(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    batches: Vec<PreparedDuckLakeTableBatch>,
) -> EtlResult<()> {
    if batches.is_empty() {
        return Ok(());
    }

    let batch_count = batches.len();
    let batches = Arc::new(batches);
    let table_name = batches[0].table_name.clone();

    retry_with_backoff(
        RetryPolicy {
            max_retries: MAX_COMMIT_RETRIES,
            initial_delay: Duration::from_millis(INITIAL_RETRY_DELAY_MS),
            max_delay: Duration::from_millis(MAX_RETRY_DELAY_MS),
        },
        ducklake_retry_decision,
        jitter_ducklake_retry_delay,
        |attempt: RetryAttempt<'_, etl::error::EtlError>| {
            counter!(
                ETL_DUCKLAKE_RETRIES_TOTAL,
                BATCH_KIND_LABEL => DuckLakeTableBatchKind::Mutation.as_str(),
                RETRY_SCOPE_LABEL => "table_sequence",
            )
            .increment(1);
            warn!(
                attempt = attempt.retry_index,
                max = attempt.max_retries,
                table = %table_name,
                batch_count,
                error = %attempt.error,
                "ducklake table batch sequence failed, retrying"
            );
        },
        move || {
            let attempt_batches = Arc::clone(&batches);
            let pool = Arc::clone(&pool);
            let blocking_slots = Arc::clone(&blocking_slots);
            async move {
                run_duckdb_blocking_with_context(pool, blocking_slots, move |conn, context| {
                    apply_table_batches(conn, attempt_batches.as_ref(), context)?;
                    Ok(())
                })
                .await
            }
        },
    )
    .await
    .map_err(|failure| {
        if is_ducklake_shutdown_requested_error(&failure.last_error) {
            return failure.last_error;
        }

        counter!(
            ETL_DUCKLAKE_FAILED_BATCHES_TOTAL,
            BATCH_KIND_LABEL => DuckLakeTableBatchKind::Mutation.as_str(),
            RETRY_SCOPE_LABEL => "table_sequence",
        )
        .increment(1);
        etl_error!(
            ErrorKind::DestinationAtomicBatchRetryable,
            "DuckLake atomic table batch sequence failed after retries",
            format!("table={table_name}, batch_count={batch_count}"),
            source: failure.last_error
        )
    })
}

/// Applies one atomic per-table batch and retries on failure.
pub(super) async fn apply_table_batch_with_retry(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    batch: PreparedDuckLakeTableBatch,
) -> EtlResult<()> {
    let table_name = batch.table_name.clone();
    let batch_id = batch.batch_id.clone();
    let batch_kind = batch.batch_kind;
    let batch = Arc::new(batch);

    retry_with_backoff(
        RetryPolicy {
            max_retries: MAX_COMMIT_RETRIES,
            initial_delay: Duration::from_millis(INITIAL_RETRY_DELAY_MS),
            max_delay: Duration::from_millis(
                MAX_RETRY_DELAY_MS.max(TRANSIENT_DELETE_FILE_RETRY_DELAY_MS),
            ),
        },
        ducklake_retry_decision,
        jitter_ducklake_retry_delay,
        |attempt: RetryAttempt<'_, etl::error::EtlError>| {
            counter!(
                ETL_DUCKLAKE_RETRIES_TOTAL,
                BATCH_KIND_LABEL => batch_kind.as_str(),
                RETRY_SCOPE_LABEL => "single_batch",
            )
            .increment(1);
            warn!(
                attempt = attempt.retry_index,
                max = attempt.max_retries,
                table = %table_name,
                batch_id = %batch_id,
                error = %attempt.error,
                "ducklake table mutation attempt failed, retrying"
            );
        },
        move || {
            let attempt_batch = Arc::clone(&batch);
            let pool = Arc::clone(&pool);
            let blocking_slots = Arc::clone(&blocking_slots);
            async move {
                run_duckdb_blocking_with_context(pool, blocking_slots, move |conn, context| {
                    if batch_kind == DuckLakeTableBatchKind::Copy {
                        if applied_batch_marker_exists(conn, attempt_batch.as_ref())? {
                            record_replayed_batch_skip(attempt_batch.as_ref());
                            return Ok(());
                        }

                        apply_table_batch(conn, attempt_batch.as_ref(), context)?;
                        return Ok(());
                    }

                    apply_table_batches(
                        conn,
                        std::slice::from_ref(attempt_batch.as_ref()),
                        context,
                    )?;
                    Ok(())
                })
                .await
            }
        },
    )
    .await
    .map_err(|failure| {
        if is_ducklake_shutdown_requested_error(&failure.last_error) {
            return failure.last_error;
        }

        counter!(
            ETL_DUCKLAKE_FAILED_BATCHES_TOTAL,
            BATCH_KIND_LABEL => batch_kind.as_str(),
            RETRY_SCOPE_LABEL => "single_batch",
        )
        .increment(1);
        etl_error!(
            ErrorKind::DestinationAtomicBatchRetryable,
            "DuckLake atomic table batch failed after retries",
            format!(
                "table={table_name}, batch_id={batch_id}, batch_kind={}",
                batch_kind.as_str()
            ),
            source: failure.last_error
        )
    })
}

/// Prepares ordered atomic batches for one table's CDC mutations.
///
/// Mutations stay in source order and are split only at the batch-size cap so
/// mixed CDC streams can commit larger insert groups without breaking atomic
/// ordering.
pub(super) fn prepare_mutation_table_batches(
    batch_size: Option<std::num::NonZeroUsize>,
    replicated_table_schema: &ReplicatedTableSchema,
    table_name: DuckLakeTableName,
    replay_epoch: String,
    tracked_mutations: Vec<TrackedTableMutation>,
) -> EtlResult<Vec<PreparedDuckLakeTableBatch>> {
    let batch_size = batch_size.map_or(CDC_MUTATION_BATCH_SIZE, std::num::NonZeroUsize::get);
    let mut prepared_batches = Vec::new();
    let mut pending_mutations = Vec::new();

    for tracked_mutation in tracked_mutations {
        pending_mutations.push(tracked_mutation);
        if pending_mutations.len() >= batch_size {
            push_prepared_mutation_batch(
                &mut prepared_batches,
                replicated_table_schema,
                &table_name,
                &replay_epoch,
                std::mem::take(&mut pending_mutations),
            )?;
        }
    }

    push_prepared_mutation_batch(
        &mut prepared_batches,
        replicated_table_schema,
        &table_name,
        &replay_epoch,
        pending_mutations,
    )?;

    Ok(prepared_batches)
}

/// Prepares one retry-safe atomic batch for a table-copy row chunk.
pub(super) fn prepare_copy_table_batch(
    replicated_table_schema: &ReplicatedTableSchema,
    table_name: DuckLakeTableName,
    replay_epoch: String,
    batch_id: TableCopyBatchId,
    table_rows: Vec<TableRow>,
) -> EtlResult<PreparedDuckLakeCopyBatch> {
    let estimated_bytes =
        table_rows.iter().map(SizeHint::size_hint).fold(0usize, usize::saturating_add);
    Ok(PreparedDuckLakeCopyBatch {
        batch: PreparedDuckLakeTableBatch {
            table_name,
            replay_epoch,
            batch_id: batch_id.to_string(),
            batch_kind: DuckLakeTableBatchKind::Copy,
            first_start_lsn: None,
            last_commit_lsn: None,
            first_sequence_key: None,
            last_sequence_key: None,
            insert_column_names: replicated_column_names(replicated_table_schema),
            action: PreparedDuckLakeTableBatchAction::Mutation(vec![
                PreparedTableMutation::Upsert(prepare_copy_rows(
                    replicated_table_schema,
                    table_rows,
                )?),
            ]),
        },
        estimated_bytes: u64::try_from(estimated_bytes).unwrap_or(u64::MAX),
    })
}

/// Prepares the durable marker written after every copy worker finishes.
pub(super) fn prepare_copy_complete_table_batch(
    table_name: DuckLakeTableName,
    replay_epoch: String,
) -> PreparedDuckLakeTableBatch {
    let identity = build_copy_complete_batch_identity(&table_name);
    PreparedDuckLakeTableBatch {
        table_name,
        replay_epoch,
        batch_id: identity.batch_id,
        batch_kind: DuckLakeTableBatchKind::CopyComplete,
        first_start_lsn: None,
        last_commit_lsn: None,
        first_sequence_key: None,
        last_sequence_key: None,
        insert_column_names: Vec::new(),
        action: PreparedDuckLakeTableBatchAction::Mutation(Vec::new()),
    }
}

/// Prepares the ordered atomic batch for one table's truncate events.
pub(super) fn prepare_truncate_table_batch(
    table_name: DuckLakeTableName,
    replay_epoch: String,
    tracked_truncates: Vec<TrackedTruncateEvent>,
) -> PreparedDuckLakeTableBatch {
    let identity = build_truncate_batch_identity(&table_name, &tracked_truncates);
    PreparedDuckLakeTableBatch {
        table_name,
        replay_epoch,
        batch_id: identity.batch_id,
        batch_kind: DuckLakeTableBatchKind::Truncate,
        first_start_lsn: identity.first_start_lsn,
        last_commit_lsn: identity.last_commit_lsn,
        first_sequence_key: tracked_truncates.first().map(TrackedTruncateEvent::sequence_key),
        last_sequence_key: tracked_truncates.last().map(TrackedTruncateEvent::sequence_key),
        insert_column_names: Vec::new(),
        action: PreparedDuckLakeTableBatchAction::Truncate,
    }
}

/// Applies jitter to one DuckLake retry delay.
fn jitter_ducklake_retry_delay(base_delay: Duration) -> Duration {
    let jitter_ratio = rand::rng().random_range(0.5..=1.5_f64);
    base_delay.mul_f64(jitter_ratio)
}

/// Replay decision for one streaming batch after reading the table watermark.
enum StreamingReplayDecision {
    Skip,
    Apply,
}

/// Records that one replay-safe batch was skipped because it was already
/// committed.
fn record_replayed_batch_skip(batch: &PreparedDuckLakeTableBatch) {
    counter!(
        ETL_DUCKLAKE_REPLAYED_BATCHES_TOTAL,
        BATCH_KIND_LABEL => batch.batch_kind.as_str(),
    )
    .increment(1);
    debug!(
        table = %batch.table_name,
        batch_id = %batch.batch_id,
        batch_kind = batch.batch_kind.as_str(),
        "ducklake table batch already committed, skipping replay"
    );
}

/// Reads the steady-state streaming replay watermark for one table.
fn read_table_streaming_progress(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
    replay_epoch: &str,
) -> EtlResult<Option<TableStreamingProgress>> {
    let table_id = table_name.id();
    let sql = format!(
        r#"SELECT last_commit_lsn, last_tx_ordinal
         FROM {LAKE_CATALOG}."{STREAMING_PROGRESS_TABLE}"
         WHERE table_name = {} AND COALESCE({REPLAY_EPOCH_COLUMN}, {}) = {}
         ORDER BY last_commit_lsn DESC, last_tx_ordinal DESC
         LIMIT 1;"#,
        quote_literal(&table_id),
        quote_literal(LEGACY_REPLAY_EPOCH),
        quote_literal(replay_epoch),
    );
    let mut statement = conn.prepare(&sql).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake streaming progress query prepare failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;
    let mut rows = statement.query([]).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake streaming progress query failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;

    let Some(row) = rows.next().map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake streaming progress row fetch failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?
    else {
        return Ok(None);
    };

    let last_commit_lsn: u64 = row.get(0).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake streaming progress commit lsn read failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;
    let last_tx_ordinal: u64 = row.get(1).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake streaming progress tx ordinal read failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;

    Ok(Some(TableStreamingProgress {
        last_sequence_key: EventSequenceKey::new(PgLsn::from(last_commit_lsn), last_tx_ordinal),
    }))
}

/// Reads the last applied streaming sequence key for one table.
pub(super) fn read_table_streaming_progress_sequence_key(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
    replay_epoch: &str,
) -> EtlResult<Option<EventSequenceKey>> {
    Ok(read_table_streaming_progress(conn, table_name, replay_epoch)?
        .map(|progress| progress.last_sequence_key))
}

/// Drops already-applied tracked mutations using the persisted sequence key.
pub(super) fn retain_mutations_after_sequence_key(
    tracked_mutations: Vec<TrackedTableMutation>,
    last_sequence_key: Option<EventSequenceKey>,
) -> Vec<TrackedTableMutation> {
    match last_sequence_key {
        Some(last_sequence_key) => tracked_mutations
            .into_iter()
            .filter(|tracked_mutation| {
                compare_sequence_keys(tracked_mutation.sequence_key(), last_sequence_key)
                    == std::cmp::Ordering::Greater
            })
            .collect(),
        None => tracked_mutations,
    }
}

/// Drops already-applied tracked truncates using the persisted sequence key.
pub(super) fn retain_truncates_after_sequence_key(
    tracked_truncates: Vec<TrackedTruncateEvent>,
    last_sequence_key: Option<EventSequenceKey>,
) -> Vec<TrackedTruncateEvent> {
    match last_sequence_key {
        Some(last_sequence_key) => tracked_truncates
            .into_iter()
            .filter(|tracked_truncate| {
                compare_sequence_keys(tracked_truncate.sequence_key(), last_sequence_key)
                    == std::cmp::Ordering::Greater
            })
            .collect(),
        None => tracked_truncates,
    }
}

/// Decides whether a streaming batch must be replayed or skipped.
fn streaming_replay_decision(
    progress: TableStreamingProgress,
    batch: &PreparedDuckLakeTableBatch,
) -> EtlResult<StreamingReplayDecision> {
    let first_sequence_key = batch.first_sequence_key.ok_or_else(|| {
        etl_error!(
            ErrorKind::InvalidState,
            "DuckLake streaming batch is missing its first sequence key",
            format!("table={}, batch_kind={}", batch.table_name, batch.batch_kind.as_str())
        )
    })?;
    let last_sequence_key = batch.last_sequence_key.ok_or_else(|| {
        etl_error!(
            ErrorKind::InvalidState,
            "DuckLake streaming batch is missing its last sequence key",
            format!("table={}, batch_kind={}", batch.table_name, batch.batch_kind.as_str())
        )
    })?;

    if compare_sequence_keys(progress.last_sequence_key, first_sequence_key)
        != std::cmp::Ordering::Less
    {
        if compare_sequence_keys(progress.last_sequence_key, last_sequence_key)
            == std::cmp::Ordering::Less
        {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake streaming progress landed inside an atomic batch",
                format!(
                    "table={}, progress={}, first={}, last={}",
                    batch.table_name,
                    format_sequence_key(progress.last_sequence_key),
                    format_sequence_key(first_sequence_key),
                    format_sequence_key(last_sequence_key)
                )
            ));
        }

        return Ok(StreamingReplayDecision::Skip);
    }

    Ok(StreamingReplayDecision::Apply)
}

/// Compares two ETL event sequence keys using commit LSN then transaction
/// ordinal.
fn compare_sequence_keys(left: EventSequenceKey, right: EventSequenceKey) -> std::cmp::Ordering {
    (u64::from(left.commit_lsn), left.tx_ordinal)
        .cmp(&(u64::from(right.commit_lsn), right.tx_ordinal))
}

/// Applies all prepared atomic batches for one table on the same connection.
fn apply_table_batches(
    conn: &duckdb::Connection,
    batches: &[PreparedDuckLakeTableBatch],
    operation_context: &DuckLakeBlockingOperationContext,
) -> EtlResult<()> {
    if batches.is_empty() {
        return Ok(());
    }

    let mut streaming_progress = if batches[0].uses_streaming_progress() {
        read_table_streaming_progress(conn, batches[0].table_name(), &batches[0].replay_epoch)?
    } else {
        None
    };

    for batch in batches {
        if !batch.uses_streaming_progress() {
            // Copy batches keep the marker path because initial-copy retries
            // still depend on per-batch idempotency.
            if applied_batch_marker_exists(conn, batch)? {
                record_replayed_batch_skip(batch);
                continue;
            }

            apply_table_batch(conn, batch, operation_context).map_err(|error| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake atomic table batch failed",
                    format!(
                        "table={}, batch_id={}, batch_kind={}",
                        batch.table_name,
                        batch.batch_id,
                        batch.batch_kind.as_str()
                    ),
                    source: error
                )
            })?;
            continue;
        }

        if let Some(progress) = streaming_progress {
            match streaming_replay_decision(progress, batch)? {
                StreamingReplayDecision::Skip => {
                    record_replayed_batch_skip(batch);
                    continue;
                }
                StreamingReplayDecision::Apply => {}
            }
        }

        apply_table_batch(conn, batch, operation_context).map_err(|error| {
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake atomic table batch failed",
                format!(
                    "table={}, batch_id={}, batch_kind={}",
                    batch.table_name,
                    batch.batch_id,
                    batch.batch_kind.as_str()
                ),
                source: error
            )
        })?;

        streaming_progress = batch
            .last_sequence_key
            .map(|last_sequence_key| TableStreamingProgress { last_sequence_key });
    }

    Ok(())
}

/// Builds one prepared atomic batch from an ordered slice of tracked mutations.
fn push_prepared_mutation_batch(
    prepared_batches: &mut Vec<PreparedDuckLakeTableBatch>,
    replicated_table_schema: &ReplicatedTableSchema,
    table_name: &DuckLakeTableName,
    replay_epoch: &str,
    tracked_mutations: Vec<TrackedTableMutation>,
) -> EtlResult<()> {
    if tracked_mutations.is_empty() {
        return Ok(());
    }

    let identity =
        build_mutation_batch_identity(table_name, replicated_table_schema, &tracked_mutations)?;
    let first_sequence_key = tracked_mutations.first().map(TrackedTableMutation::sequence_key);
    let last_sequence_key = tracked_mutations.last().map(TrackedTableMutation::sequence_key);
    let mutations = tracked_mutations.into_iter().map(|tracked| tracked.mutation).collect();

    prepared_batches.push(PreparedDuckLakeTableBatch {
        table_name: table_name.clone(),
        replay_epoch: replay_epoch.to_owned(),
        batch_id: identity.batch_id,
        batch_kind: DuckLakeTableBatchKind::Mutation,
        first_start_lsn: identity.first_start_lsn,
        last_commit_lsn: identity.last_commit_lsn,
        first_sequence_key,
        last_sequence_key,
        insert_column_names: replicated_column_names(replicated_table_schema),
        action: PreparedDuckLakeTableBatchAction::Mutation(prepare_table_mutations(
            replicated_table_schema,
            mutations,
        )?),
    });

    Ok(())
}

/// Groups ordered row mutations into retryable DuckDB operations.
fn prepare_table_mutations(
    replicated_table_schema: &ReplicatedTableSchema,
    mutations: Vec<TableMutation>,
) -> EtlResult<Vec<PreparedTableMutation>> {
    let mut prepared_mutations = Vec::new();
    let mut upsert_rows = Vec::new();
    let mut delete_predicates = Vec::new();

    for mutation in mutations {
        match mutation {
            TableMutation::Insert(row) => {
                if !delete_predicates.is_empty() {
                    prepared_mutations.push(PreparedTableMutation::Delete {
                        predicates: std::mem::take(&mut delete_predicates),
                        origin: "delete",
                    });
                }
                upsert_rows.push(row);
            }
            TableMutation::Delete(row) => {
                if !upsert_rows.is_empty() {
                    prepared_mutations.push(PreparedTableMutation::Upsert(prepare_rows(
                        std::mem::take(&mut upsert_rows),
                    )));
                }
                delete_predicates.push(delete_predicate_from_row(replicated_table_schema, &row)?);
            }
            TableMutation::Update { delete_row, new_row } => {
                if !upsert_rows.is_empty() {
                    prepared_mutations.push(PreparedTableMutation::Upsert(prepare_rows(
                        std::mem::take(&mut upsert_rows),
                    )));
                }
                if !delete_predicates.is_empty() {
                    prepared_mutations.push(PreparedTableMutation::Delete {
                        predicates: std::mem::take(&mut delete_predicates),
                        origin: "delete",
                    });
                }
                match new_row {
                    UpdatedTableRow::Full(upsert_row) => {
                        prepared_mutations.push(PreparedTableMutation::Delete {
                            predicates: vec![delete_predicate_from_row(
                                replicated_table_schema,
                                &delete_row,
                            )?],
                            origin: "update",
                        });
                        prepared_mutations
                            .push(PreparedTableMutation::Upsert(prepare_rows(vec![upsert_row])));
                    }
                    UpdatedTableRow::Partial(partial_row) => {
                        prepared_mutations.push(PreparedTableMutation::Update {
                            assignments: update_assignments_from_partial_row(
                                replicated_table_schema,
                                &partial_row,
                            )?,
                            predicate: delete_predicate_from_row(
                                replicated_table_schema,
                                &delete_row,
                            )?,
                        });
                    }
                }
            }
            TableMutation::Replace(row) => {
                if !upsert_rows.is_empty() {
                    prepared_mutations.push(PreparedTableMutation::Upsert(prepare_rows(
                        std::mem::take(&mut upsert_rows),
                    )));
                }
                if !delete_predicates.is_empty() {
                    prepared_mutations.push(PreparedTableMutation::Delete {
                        predicates: std::mem::take(&mut delete_predicates),
                        origin: "delete",
                    });
                }

                prepared_mutations.push(PreparedTableMutation::Delete {
                    predicates: vec![delete_predicate_from_row(replicated_table_schema, &row)?],
                    origin: "replace",
                });
                prepared_mutations.push(PreparedTableMutation::Upsert(prepare_rows(vec![row])));
            }
        }
    }

    if !upsert_rows.is_empty() {
        prepared_mutations.push(PreparedTableMutation::Upsert(prepare_rows(upsert_rows)));
    }
    if !delete_predicates.is_empty() {
        prepared_mutations.push(PreparedTableMutation::Delete {
            predicates: delete_predicates,
            origin: "delete",
        });
    }

    Ok(prepared_mutations)
}

/// Builds a `WHERE` clause from the replica-identity values stored in `row`.
fn delete_predicate_from_row<'a>(
    replicated_table_schema: &ReplicatedTableSchema,
    row: impl Into<DeletePredicateRowRef<'a>>,
) -> EtlResult<String> {
    let row = row.into();
    let replicated_column_schemas: Vec<_> = replicated_table_schema.column_schemas().collect();
    let identity_column_schemas: Vec<_> =
        replicated_table_schema.identity_column_schemas().collect();
    if identity_column_schemas.is_empty() {
        return Err(etl_error!(
            ErrorKind::SourceReplicaIdentityError,
            "DuckLake delete requires a replica identity",
            format!(
                "Table '{}' has no replicated replica-identity columns",
                replicated_table_schema.name()
            )
        ));
    }

    let key_values: Vec<_> = match row {
        DeletePredicateRowRef::Full(row) => {
            if row.values().len() != replicated_column_schemas.len() {
                return Err(etl_error!(
                    ErrorKind::InvalidState,
                    "DuckLake row shape does not match schema",
                    format!(
                        "Expected {} values for table '{}', got {}",
                        replicated_column_schemas.len(),
                        replicated_table_schema.name(),
                        row.values().len()
                    )
                ));
            }

            let mut identity_columns = identity_column_schemas.iter().copied().peekable();
            let mut key_values = Vec::with_capacity(identity_column_schemas.len());

            for (column_schema, value) in replicated_column_schemas.iter().zip(row.values()) {
                if identity_columns.peek().is_some_and(|identity_column| {
                    identity_column.ordinal_position == column_schema.ordinal_position
                }) {
                    let Some(identity_column) = identity_columns.next() else {
                        return Err(etl_error!(
                            ErrorKind::InvalidState,
                            "DuckLake replica identity schema is inconsistent",
                            format!(
                                "Table '{}' identity columns ended unexpectedly",
                                replicated_table_schema.name()
                            )
                        ));
                    };

                    key_values.push((identity_column, value));
                }
            }

            key_values
        }
        DeletePredicateRowRef::Key(row) => {
            if row.values().len() != identity_column_schemas.len() {
                return Err(etl_error!(
                    ErrorKind::InvalidState,
                    "DuckLake key image does not match replica identity",
                    format!(
                        "Expected {} key values for table '{}', got {}",
                        identity_column_schemas.len(),
                        replicated_table_schema.name(),
                        row.values().len()
                    )
                ));
            }

            identity_column_schemas.iter().copied().zip(row.values()).collect()
        }
    };

    let mut predicates = Vec::new();
    for (column_schema, value) in key_values {
        let quoted_column =
            quote_identifier(&DUCKLAKE_COLUMN_NAME_MAPPING.map_name(&column_schema.name));
        let predicate = match value {
            Cell::Null => format!("{quoted_column} IS NULL"),
            _ => format!("{quoted_column} = {}", cell_to_sql_literal_ref(value)),
        };
        predicates.push(predicate);
    }

    Ok(predicates.join(" AND "))
}

/// Builds SQL `SET` assignments from a partial update row.
fn update_assignments_from_partial_row(
    replicated_table_schema: &ReplicatedTableSchema,
    partial_row: &PartialTableRow,
) -> EtlResult<Vec<String>> {
    let replicated_column_schemas: Vec<_> = replicated_table_schema.column_schemas().collect();
    if partial_row.total_columns() != replicated_column_schemas.len() {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake partial update row does not match schema",
            format!(
                "Expected {} replicated columns for table '{}', got {}",
                replicated_column_schemas.len(),
                replicated_table_schema.name(),
                partial_row.total_columns()
            )
        ));
    }

    if partial_row.values().is_empty() {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake partial update row has no assignments",
            format!(
                "Table '{}' emitted an empty partial update row",
                replicated_table_schema.name()
            )
        ));
    }

    if partial_row.values().len() + partial_row.missing_column_indexes().len()
        != partial_row.total_columns()
    {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake partial update row shape is inconsistent",
            format!(
                "Table '{}' partial row reports {} total columns but has {} present and {} missing",
                replicated_table_schema.name(),
                partial_row.total_columns(),
                partial_row.values().len(),
                partial_row.missing_column_indexes().len()
            )
        ));
    }

    let mut assignments = Vec::with_capacity(partial_row.values().len());
    let mut missing_indexes = partial_row.missing_column_indexes().iter().copied().peekable();
    let mut present_values = partial_row.values().iter();
    for (column_index, column_schema) in replicated_column_schemas.iter().enumerate() {
        if missing_indexes.peek().copied() == Some(column_index) {
            missing_indexes.next();
            continue;
        }

        let Some(value) = present_values.next() else {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake partial update row ended early",
                format!(
                    "Table '{}' did not provide enough values for its partial update row",
                    replicated_table_schema.name()
                )
            ));
        };
        let quoted_column =
            quote_identifier(&DUCKLAKE_COLUMN_NAME_MAPPING.map_name(&column_schema.name));
        assignments.push(format!("{quoted_column} = {}", cell_to_sql_literal_ref(value)));
    }

    if missing_indexes.next().is_some() || present_values.next().is_some() {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake partial update row shape is inconsistent",
            format!(
                "Table '{}' partial row has leftover values or missing indexes after decoding",
                replicated_table_schema.name()
            )
        ));
    }

    Ok(assignments)
}

/// Builds a deterministic identity for one ordered mutation batch.
fn build_mutation_batch_identity(
    table_name: &DuckLakeTableName,
    replicated_table_schema: &ReplicatedTableSchema,
    tracked_mutations: &[TrackedTableMutation],
) -> EtlResult<DuckLakeBatchIdentity> {
    let mut hasher = BatchIdHasher::new();
    "mutation".hash(&mut hasher);
    table_name.id().hash(&mut hasher);

    for tracked_mutation in tracked_mutations {
        u64::from(tracked_mutation.sequence_key.commit_lsn).hash(&mut hasher);
        tracked_mutation.sequence_key.tx_ordinal.hash(&mut hasher);

        match &tracked_mutation.mutation {
            TableMutation::Insert(row) => {
                "insert".hash(&mut hasher);
                hash_table_row_ref(&mut hasher, row);
            }
            TableMutation::Delete(row) => {
                "delete".hash(&mut hasher);
                delete_predicate_from_row(replicated_table_schema, row)?.hash(&mut hasher);
            }
            TableMutation::Update { delete_row, new_row } => {
                "update".hash(&mut hasher);
                delete_predicate_from_row(replicated_table_schema, delete_row)?.hash(&mut hasher);
                match new_row {
                    UpdatedTableRow::Full(row) => hash_table_row_ref(&mut hasher, row),
                    UpdatedTableRow::Partial(row) => hash_partial_table_row_ref(&mut hasher, row)?,
                }
            }
            TableMutation::Replace(row) => {
                "replace".hash(&mut hasher);
                delete_predicate_from_row(replicated_table_schema, row)?.hash(&mut hasher);
                hash_table_row_ref(&mut hasher, row);
            }
        }
    }

    Ok(build_batch_identity(
        DuckLakeTableBatchKind::Mutation,
        None,
        tracked_mutations.last().map(|tracked_mutation| tracked_mutation.sequence_key.commit_lsn),
        hasher.finish(),
    ))
}

/// Builds the deterministic identity shared by retries of a copy barrier.
fn build_copy_complete_batch_identity(table_name: &DuckLakeTableName) -> DuckLakeBatchIdentity {
    let mut hasher = BatchIdHasher::new();
    "copy_complete".hash(&mut hasher);
    table_name.id().hash(&mut hasher);

    build_batch_identity(DuckLakeTableBatchKind::CopyComplete, None, None, hasher.finish())
}

/// Builds a deterministic identity for one ordered truncate batch.
fn build_truncate_batch_identity(
    table_name: &DuckLakeTableName,
    tracked_truncates: &[TrackedTruncateEvent],
) -> DuckLakeBatchIdentity {
    let mut hasher = BatchIdHasher::new();
    "truncate".hash(&mut hasher);
    table_name.id().hash(&mut hasher);

    for tracked_truncate in tracked_truncates {
        u64::from(tracked_truncate.sequence_key.commit_lsn).hash(&mut hasher);
        tracked_truncate.sequence_key.tx_ordinal.hash(&mut hasher);
        tracked_truncate.options.hash(&mut hasher);
    }

    build_batch_identity(
        DuckLakeTableBatchKind::Truncate,
        None,
        tracked_truncates.last().map(|tracked_truncate| tracked_truncate.sequence_key.commit_lsn),
        hasher.finish(),
    )
}

/// Builds the final persisted batch identity string.
fn build_batch_identity(
    batch_kind: DuckLakeTableBatchKind,
    first_start_lsn: Option<PgLsn>,
    last_commit_lsn: Option<PgLsn>,
    fingerprint: u64,
) -> DuckLakeBatchIdentity {
    let first_start_lsn_u64 = first_start_lsn.map(u64::from).unwrap_or_default();
    let last_commit_lsn_u64 = last_commit_lsn.map(u64::from).unwrap_or_default();

    DuckLakeBatchIdentity {
        batch_id: format!(
            "{}:{first_start_lsn_u64:016x}:{last_commit_lsn_u64:016x}:{fingerprint:016x}",
            batch_kind.as_str()
        ),
        first_start_lsn,
        last_commit_lsn,
    }
}

/// Hashes a row using its SQL literal form so retries are independent of
/// appender encoding.
fn hash_table_row_ref(hasher: &mut BatchIdHasher, row: &TableRow) {
    table_row_to_sql_literal_ref(row).hash(hasher);
}

/// Hashes a partial row using column indexes and SQL literal forms.
fn hash_partial_table_row_ref(hasher: &mut BatchIdHasher, row: &PartialTableRow) -> EtlResult<()> {
    row.total_columns().hash(hasher);
    let mut missing_indexes = row.missing_column_indexes().iter().copied().peekable();
    let mut present_values = row.values().iter();

    for column_index in 0..row.total_columns() {
        if missing_indexes.peek().copied() == Some(column_index) {
            missing_indexes.next();
            continue;
        }

        let Some(value) = present_values.next() else {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake partial row shape is inconsistent",
                format!("Partial row ended before replicated column index {}", column_index)
            ));
        };

        column_index.hash(hasher);
        cell_to_sql_literal_ref(value).hash(hasher);
    }

    if present_values.next().is_some() {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake partial row shape is inconsistent",
            "Partial row contained more present values than its missing indexes allow"
        ));
    }

    Ok(())
}

/// Returns whether the atomic batch marker already exists.
fn applied_batch_marker_exists(
    conn: &duckdb::Connection,
    batch: &PreparedDuckLakeTableBatch,
) -> EtlResult<bool> {
    applied_batch_marker_id_exists(conn, &batch.table_name, &batch.replay_epoch, &batch.batch_id)
}

/// Returns whether one marker identity already exists.
fn applied_batch_marker_id_exists(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
    replay_epoch: &str,
    batch_id: &str,
) -> EtlResult<bool> {
    let table_id = table_name.id();
    let sql = format!(
        r#"SELECT 1 FROM {LAKE_CATALOG}."{APPLIED_BATCHES_TABLE}"
         WHERE table_name = {}
           AND COALESCE({REPLAY_EPOCH_COLUMN}, {}) = {}
           AND batch_id = {}
         LIMIT 1;"#,
        quote_literal(&table_id),
        quote_literal(LEGACY_REPLAY_EPOCH),
        quote_literal(replay_epoch),
        quote_literal(batch_id)
    );
    let mut statement = conn.prepare(&sql).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake marker query prepare failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;
    let mut rows = statement.query([]).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake marker query failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;

    rows.next().map(|row| row.is_some()).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake marker query row fetch failed",
            format_query_error_detail(&sql),
            source: err
        )
    })
}

/// Inserts the atomic batch marker inside the open DuckLake transaction.
fn insert_applied_batch_marker(
    conn: &duckdb::Connection,
    batch: &PreparedDuckLakeTableBatch,
) -> EtlResult<()> {
    let marker = PreparedDuckLakeBatchMarker {
        batch_kind: batch.batch_kind,
        first_start_lsn: batch.first_start_lsn,
        last_commit_lsn: batch.last_commit_lsn,
    };
    insert_applied_batch_marker_fields(
        conn,
        &batch.table_name,
        &batch.replay_epoch,
        &batch.batch_id,
        &marker,
    )
}

/// Inserts one compact marker inside the open DuckLake transaction.
fn insert_applied_batch_marker_fields(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
    replay_epoch: &str,
    batch_id: &str,
    marker: &PreparedDuckLakeBatchMarker,
) -> EtlResult<()> {
    let table_id = table_name.id();
    let sql = format!(
        r#"INSERT INTO {LAKE_CATALOG}."{APPLIED_BATCHES_TABLE}"
         (table_name, replay_epoch, batch_id, batch_kind, first_start_lsn, last_commit_lsn, applied_at)
         VALUES ({}, {}, {}, {}, {}, {}, current_timestamp);"#,
        quote_literal(&table_id),
        quote_literal(replay_epoch),
        quote_literal(batch_id),
        quote_literal(marker.batch_kind.as_str()),
        optional_lsn_to_sql_literal(marker.first_start_lsn),
        optional_lsn_to_sql_literal(marker.last_commit_lsn),
    );
    conn.execute_batch(&sql).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake batch marker insert failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;
    Ok(())
}

/// Appends the steady-state streaming replay watermark inside the open
/// transaction.
fn update_table_streaming_progress(
    conn: &duckdb::Connection,
    batch: &PreparedDuckLakeTableBatch,
) -> EtlResult<()> {
    let last_sequence_key = batch.last_sequence_key.ok_or_else(|| {
        etl_error!(
            ErrorKind::InvalidState,
            "DuckLake streaming batch is missing its last sequence key",
            format!("table={}, batch_kind={}", batch.table_name, batch.batch_kind.as_str())
        )
    })?;
    let table_id = batch.table_name.id();
    let sql = format!(
        r#"INSERT INTO {LAKE_CATALOG}."{STREAMING_PROGRESS_TABLE}"
         (table_name, replay_epoch, last_commit_lsn, last_tx_ordinal, updated_at)
         VALUES ({}, {}, {}, {}, current_timestamp);"#,
        quote_literal(&table_id),
        quote_literal(&batch.replay_epoch),
        u64::from(last_sequence_key.commit_lsn),
        last_sequence_key.tx_ordinal,
    );
    conn.execute_batch(&sql).map_err(|err| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake streaming progress update failed",
            format_query_error_detail(&sql),
            source: err
        )
    })?;
    Ok(())
}

/// Joins quoted column identifiers for insert/select lists.
fn quoted_column_list(column_names: &[String]) -> String {
    column_names
        .iter()
        .map(|column_name| quote_identifier(column_name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Reusable per-batch temp staging table for DuckLake upserts.
struct ReusableStagingTable {
    table_name: DuckLakeTableName,
    staging_name: String,
    created: bool,
    insert_column_names: Vec<String>,
}

impl ReusableStagingTable {
    /// Creates a fresh staging-table manager for one destination table.
    fn new(table_name: &DuckLakeTableName, insert_column_names: Vec<String>) -> Self {
        Self {
            table_name: table_name.clone(),
            staging_name: format!("__staging_{}", table_name.id()),
            created: false,
            insert_column_names,
        }
    }

    /// Loads one prepared row set into staging and applies it to the target
    /// table.
    fn stage_and_insert(
        &mut self,
        conn: &duckdb::Connection,
        prepared_rows: &PreparedRows,
    ) -> EtlResult<()> {
        self.prepare(conn)?;
        self.load_rows(conn, prepared_rows)?;
        self.insert_staged_rows(conn)
    }

    /// Drops the temp staging table after the batch finishes.
    fn cleanup(&self, conn: &duckdb::Connection) {
        if !self.created {
            return;
        }

        let staging_table = quote_identifier(&self.staging_name);
        if let Err(error) = conn.execute_batch(&format!("drop table if exists {staging_table}")) {
            tracing::error!(error = %error, "error drop table staging");
        }
    }

    /// Creates the temp table once, then clears it before each reuse.
    fn prepare(&mut self, conn: &duckdb::Connection) -> EtlResult<()> {
        if self.created {
            return self.clear(conn);
        }

        self.ensure_created(conn)
    }

    /// Creates the temporary staging table without clearing existing rows.
    fn ensure_created(&mut self, conn: &duckdb::Connection) -> EtlResult<()> {
        if self.created {
            return Ok(());
        }

        #[cfg(feature = "test-utils")]
        {
            let mut counts = STAGING_TABLE_CREATIONS_BY_TABLE.lock();
            *counts.entry(self.table_name.id()).or_default() += 1;
        }

        let staging_table = quote_identifier(&self.staging_name);
        let column_list = quoted_column_list(&self.insert_column_names);
        let target_table = qualified_lake_table_name(&self.table_name);
        conn.execute_batch(&format!(
            "create or replace temp table {staging_table} as
             select {column_list} from {target_table} limit 0;"
        ))
        .map_err(|error| {
            tracing::error!(error = %error, "error creating temporary table");

            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake staging table creation failed",
                source: error
            )
        })?;
        self.created = true;
        Ok(())
    }

    /// Inserts every currently staged row into the destination table.
    fn insert_staged_rows(&self, conn: &duckdb::Connection) -> EtlResult<()> {
        let column_list = quoted_column_list(&self.insert_column_names);
        let target_table = qualified_lake_table_name(&self.table_name);
        let staging_table = quote_identifier(&self.staging_name);
        let sql = format!(
            "insert into {target_table} ({column_list}) select {column_list} from {staging_table};"
        );
        conn.execute_batch(&sql).map_err(|error| {
            tracing::error!(error = %error, "error inserting rows");
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake INSERT SELECT failed",
                format_query_error_detail(&sql),
                source: error
            )
        })?;
        Ok(())
    }

    /// Clears all rows from an existing temporary staging table.
    fn clear(&self, conn: &duckdb::Connection) -> EtlResult<()> {
        if !self.created {
            return Ok(());
        }

        let staging_table = quote_identifier(&self.staging_name);
        let sql = format!("truncate table {staging_table};");
        conn.execute_batch(&sql).map_err(|error| {
            tracing::error!(error = %error, "error clear staging");
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake staging table clear failed",
                source: error
            )
        })?;
        Ok(())
    }

    /// Loads one prepared row payload into the temp staging table.
    fn load_rows(&self, conn: &duckdb::Connection, prepared_rows: &PreparedRows) -> EtlResult<()> {
        match prepared_rows {
            PreparedRows::Appender(all_values) => {
                let mut appender = conn.appender(&self.staging_name).map_err(|error| {
                    tracing::error!(error = %error, "error appender");
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake staging appender creation failed",
                        source: error
                    )
                })?;
                for values in all_values {
                    appender.append_row(duckdb::appender_params_from_iter(values)).map_err(
                        |err| {
                            tracing::error!(error = %err, "error append row");
                            etl_error!(
                                ErrorKind::DestinationQueryFailed,
                                "DuckLake staging append_row failed",
                                source: err
                            )
                        },
                    )?;
                }
                appender.flush().map_err(|err| {
                    tracing::error!(error = %err, "error flush");
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake staging appender flush failed",
                        source: err
                    )
                })?;
            }
            PreparedRows::ArrowRecordBatch(record_batch) => {
                let mut appender = conn.appender(&self.staging_name).map_err(|error| {
                    tracing::error!(error = %error, "error appender");
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake staging appender creation failed",
                        source: error
                    )
                })?;
                appender.append_record_batch(record_batch.clone()).map_err(|err| {
                    tracing::error!(error = %err, "error append record batch");
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake staging append_record_batch failed",
                        source: err
                    )
                })?;
                appender.flush().map_err(|err| {
                    tracing::error!(error = %err, "error flush");
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake staging appender flush failed",
                        source: err
                    )
                })?;
            }
            PreparedRows::SqlLiterals(row_literals) => {
                insert_rows_into_staging_with_sql(
                    conn,
                    &self.staging_name,
                    row_literals.as_slice(),
                )?;
            }
        }
        Ok(())
    }
}

/// Connection-local accumulator for one table-copy replay epoch.
pub(super) struct DuckLakeCopyAccumulator {
    table_name: DuckLakeTableName,
    replay_epoch: String,
    staging: ReusableStagingTable,
    pending_markers: HashMap<String, PreparedDuckLakeBatchMarker>,
    staged_bytes: u64,
}

impl DuckLakeCopyAccumulator {
    /// Creates an empty accumulator shaped from its first prepared batch.
    pub(super) fn new(batch: &PreparedDuckLakeCopyBatch) -> Self {
        Self {
            table_name: batch.batch.table_name.clone(),
            replay_epoch: batch.batch.replay_epoch.clone(),
            staging: ReusableStagingTable::new(
                &batch.batch.table_name,
                batch.batch.insert_column_names.clone(),
            ),
            pending_markers: HashMap::new(),
            staged_bytes: 0,
        }
    }

    /// Returns the approximate decoded bytes awaiting a durable flush.
    pub(super) fn staged_bytes(&self) -> u64 {
        self.staged_bytes
    }

    /// Appends one prepared copy batch to the connection-local staging table.
    ///
    /// Returns `false` when the deterministic batch marker proves that this
    /// batch was already committed during the same replay epoch.
    pub(super) fn append(
        &mut self,
        conn: &duckdb::Connection,
        batch: PreparedDuckLakeCopyBatch,
    ) -> EtlResult<bool> {
        if batch.batch.table_name != self.table_name
            || batch.batch.replay_epoch != self.replay_epoch
            || batch.batch.insert_column_names != self.staging.insert_column_names
        {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake buffered copy batch does not match its staging session",
                format!("table={}", self.table_name)
            ));
        }

        if self.pending_markers.contains_key(&batch.batch.batch_id)
            || applied_batch_marker_exists(conn, &batch.batch)?
        {
            record_replayed_batch_skip(&batch.batch);
            return Ok(false);
        }

        let estimated_bytes = batch.estimated_bytes;
        let (batch_id, marker, prepared_rows) = batch.into_buffer_parts()?;
        self.staging.ensure_created(conn)?;
        self.staging.load_rows(conn, &prepared_rows)?;
        self.pending_markers.insert(batch_id, marker);
        self.staged_bytes = self.staged_bytes.saturating_add(estimated_bytes);

        Ok(true)
    }

    /// Commits all staged rows and their markers as one DuckLake change.
    ///
    /// When supplied, `copy_complete` is committed in the same transaction as
    /// the final staged window. A failure before `COMMIT` is rolled back. A
    /// `COMMIT` failure has an ambiguous outcome, so the dedicated connection
    /// and complete table-copy attempt are invalidated instead of retried.
    /// Staging is cleared only after `COMMIT` succeeds.
    pub(super) fn flush(
        &mut self,
        conn: &duckdb::Connection,
        copy_complete: Option<PreparedDuckLakeTableBatch>,
    ) -> EtlResult<()> {
        if self.pending_markers.is_empty() && copy_complete.is_none() {
            return Ok(());
        }

        let started = Instant::now();
        conn.execute_batch("begin transaction").map_err(|error| {
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake buffered copy BEGIN TRANSACTION failed",
                source: error
            )
        })?;

        let result = (|| -> EtlResult<()> {
            if !self.pending_markers.is_empty() {
                self.staging.insert_staged_rows(conn)?;
                for (batch_id, marker) in &self.pending_markers {
                    insert_applied_batch_marker_fields(
                        conn,
                        &self.table_name,
                        &self.replay_epoch,
                        batch_id,
                        marker,
                    )?;
                }
            }

            if let Some(copy_complete) = &copy_complete {
                if copy_complete.table_name != self.table_name
                    || copy_complete.replay_epoch != self.replay_epoch
                {
                    return Err(etl_error!(
                        ErrorKind::InvalidState,
                        "DuckLake copy-complete marker does not match its buffered session",
                        format!("table={}", self.table_name)
                    ));
                }
                if !applied_batch_marker_exists(conn, copy_complete)? {
                    insert_applied_batch_marker(conn, copy_complete)?;
                }
            }

            Ok(())
        })();

        if let Err(error) = result {
            if let Err(rollback_error) = conn.execute_batch("rollback") {
                tracing::error!(error = %rollback_error, "error rollback buffered copy");
            }
            return Err(error);
        }

        // A COMMIT error does not prove whether DuckLake committed. Returning
        // the error makes the shared runner mark this connection broken; the
        // owner then fences the buffer until recovery drops and recopies the
        // complete table instead of retrying an ambiguous transaction.
        conn.execute_batch("commit").map_err(|error| {
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake buffered copy COMMIT failed",
                source: error
            )
        })?;

        #[cfg(feature = "test-utils")]
        maybe_fail_after_committed_batch_for_tests(DuckLakeTableBatchKind::Copy, &self.table_name)?;

        self.staging.clear(conn)?;
        histogram!(
            ETL_DUCKLAKE_BATCH_COMMIT_DURATION_SECONDS,
            BATCH_KIND_LABEL => DuckLakeTableBatchKind::Copy.as_str(),
            SUB_BATCH_KIND_LABEL => "buffered_insert",
        )
        .record(started.elapsed().as_secs_f64());
        histogram!(
            ETL_DUCKLAKE_BATCH_PREPARED_MUTATIONS,
            BATCH_KIND_LABEL => DuckLakeTableBatchKind::Copy.as_str(),
            SUB_BATCH_KIND_LABEL => "buffered_insert",
        )
        .record(self.pending_markers.len() as f64);
        trace!(
            table = %self.table_name,
            copy_batch_count = self.pending_markers.len(),
            staged_bytes = self.staged_bytes,
            copy_complete = copy_complete.is_some(),
            "ducklake buffered copy committed"
        );
        self.pending_markers.clear();
        self.staged_bytes = 0;

        Ok(())
    }
}

/// Applies one atomic per-table batch in a single DuckLake transaction.
fn apply_table_batch(
    conn: &duckdb::Connection,
    batch: &PreparedDuckLakeTableBatch,
    operation_context: &DuckLakeBlockingOperationContext,
) -> EtlResult<()> {
    let batch_started = Instant::now();

    conn.execute_batch("BEGIN TRANSACTION").map_err(|error| {
        tracing::error!(error = %error, "error transaction");
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake BEGIN TRANSACTION failed",
            source: error
        )
    })?;

    let mut reusable_staging_table =
        ReusableStagingTable::new(&batch.table_name, batch.insert_column_names.clone());
    let result = (|| -> EtlResult<()> {
        match &batch.action {
            PreparedDuckLakeTableBatchAction::Mutation(prepared_mutations) => {
                for prepared_mutation in prepared_mutations {
                    apply_table_mutation(
                        conn,
                        batch,
                        prepared_mutation,
                        &mut reusable_staging_table,
                        operation_context,
                    )?;
                }
            }
            PreparedDuckLakeTableBatchAction::Truncate => {
                apply_truncate_batch_action(conn, &batch.table_name)?;
            }
        }

        if batch.uses_streaming_progress() {
            update_table_streaming_progress(conn, batch)?;
        } else {
            insert_applied_batch_marker(conn, batch)?;
        }
        Ok(())
    })();

    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT").map_err(|error| {
                tracing::error!(error = %error, "error commit");
                reusable_staging_table.cleanup(conn);
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake COMMIT failed",
                    source: error
                )
            })?;
            reusable_staging_table.cleanup(conn);
            histogram!(
                ETL_DUCKLAKE_BATCH_COMMIT_DURATION_SECONDS,
                BATCH_KIND_LABEL => batch.batch_kind.as_str(),
                SUB_BATCH_KIND_LABEL => batch_log_kind(batch),
            )
            .record(batch_started.elapsed().as_secs_f64());
            histogram!(
                ETL_DUCKLAKE_BATCH_PREPARED_MUTATIONS,
                BATCH_KIND_LABEL => batch.batch_kind.as_str(),
                SUB_BATCH_KIND_LABEL => batch_log_kind(batch),
            )
            .record(prepared_mutation_count(batch) as f64);
            trace!(
                table = %batch.table_name,
                batch_id = %batch.batch_id,
                batch_kind = batch.batch_kind.as_str(),
                first_start_lsn = %format_optional_lsn(batch.first_start_lsn),
                last_commit_lsn = %format_optional_lsn(batch.last_commit_lsn),
                sub_batch_kind = batch_log_kind(batch),
                insert_sub_batch_rows = apply_sub_batch_rows(batch),
                "ducklake batch committed"
            );

            #[cfg(feature = "test-utils")]
            maybe_fail_after_committed_batch_for_tests(batch.batch_kind, &batch.table_name)?;

            Ok(())
        }
        Err(err) => {
            let rollback = conn.execute_batch("ROLLBACK");
            reusable_staging_table.cleanup(conn);
            if let Err(err) = rollback {
                tracing::error!(error = %err, "error rollback");
            }

            Err(err)
        }
    }
}

/// Applies the truncate action inside an open transaction.
fn apply_truncate_batch_action(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
) -> EtlResult<()> {
    let target_table = qualified_lake_table_name(table_name);
    let sql = format!("TRUNCATE TABLE {target_table};");
    conn.execute_batch(&sql).map_err(|error| {
        tracing::error!(error = %error, "error truncating table");
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake TRUNCATE TABLE failed",
            format_query_error_detail(&sql),
            source: error
        )
    })?;
    Ok(())
}

/// Formats an optional LSN for marker-table inserts.
fn optional_lsn_to_sql_literal(lsn: Option<PgLsn>) -> String {
    lsn.map_or_else(|| "NULL".to_owned(), |value| u64::from(value).to_string())
}

/// Applies one prepared table mutation inside an open transaction.
fn apply_table_mutation(
    conn: &duckdb::Connection,
    batch: &PreparedDuckLakeTableBatch,
    prepared_mutation: &PreparedTableMutation,
    reusable_staging_table: &mut ReusableStagingTable,
    operation_context: &DuckLakeBlockingOperationContext,
) -> EtlResult<()> {
    match prepared_mutation {
        PreparedTableMutation::Upsert(prepared_rows) => {
            histogram!(
                ETL_DUCKLAKE_UPSERT_ROWS,
                BATCH_KIND_LABEL => batch.batch_kind.as_str(),
                PREPARED_ROWS_KIND_LABEL => prepared_rows_kind(prepared_rows),
            )
            .record(prepared_rows_count(prepared_rows) as f64);
            apply_upsert_mutation(conn, prepared_rows, reusable_staging_table)
        }
        PreparedTableMutation::Delete { predicates, origin } => {
            histogram!(
                ETL_DUCKLAKE_DELETE_PREDICATES,
                BATCH_KIND_LABEL => batch.batch_kind.as_str(),
                DELETE_ORIGIN_LABEL => *origin,
            )
            .record(predicates.len() as f64);
            apply_delete_mutation(conn, batch, predicates.as_slice(), origin, operation_context)
        }
        PreparedTableMutation::Update { assignments, predicate } => apply_update_mutation(
            conn,
            &reusable_staging_table.table_name,
            assignments.as_slice(),
            predicate,
        ),
    }
}

/// Applies one upsert batch inside an open DuckLake transaction.
fn apply_upsert_mutation(
    conn: &duckdb::Connection,
    prepared_rows: &PreparedRows,
    reusable_staging_table: &mut ReusableStagingTable,
) -> EtlResult<()> {
    let row_count = prepared_rows_count(prepared_rows);

    if row_count == 0 {
        return Ok(());
    }

    reusable_staging_table.stage_and_insert(conn, prepared_rows)
}

/// Applies one delete batch inside an open DuckLake transaction.
fn apply_delete_mutation(
    conn: &duckdb::Connection,
    batch: &PreparedDuckLakeTableBatch,
    predicates: &[String],
    origin: &'static str,
    operation_context: &DuckLakeBlockingOperationContext,
) -> EtlResult<()> {
    if predicates.is_empty() {
        return Ok(());
    }

    let target_table = qualified_lake_table_name(&batch.table_name);
    let chunk_count = predicates.len().div_ceil(SQL_DELETE_BATCH_SIZE);
    for (chunk_index, chunk) in predicates.chunks(SQL_DELETE_BATCH_SIZE).enumerate() {
        let where_clause =
            chunk.iter().map(|predicate| format!("({predicate})")).collect::<Vec<_>>().join(" OR ");

        let sql_query = format!("DELETE FROM {target_table} WHERE {where_clause};");
        conn.execute_batch(&sql_query).map_err(|error| {
            let duckdb_interrupted = is_duckdb_interrupt_error(&error);
            let error = DuckDbSensitiveQueryError { error, sql: sql_query.clone() };
            tracing::error!(
                error = "DuckDB DELETE failed; row-bearing diagnostics omitted from logs",
                table = %batch.table_name,
                batch_id = %batch.batch_id,
                batch_kind = batch.batch_kind.as_str(),
                first_start_lsn = %format_optional_lsn(batch.first_start_lsn),
                last_commit_lsn = %format_optional_lsn(batch.last_commit_lsn),
                first_sequence_key = %format_optional_sequence_key(batch.first_sequence_key),
                last_sequence_key = %format_optional_sequence_key(batch.last_sequence_key),
                delete_origin = origin,
                delete_predicate_count = predicates.len(),
                delete_chunk_index = chunk_index,
                delete_chunk_count = chunk_count,
                delete_chunk_predicate_count = chunk.len(),
                duckdb_interrupted,
                ducklake_interrupt_reason = operation_context.interrupt_reason_label(),
                ducklake_operation_id = operation_context.operation_id(),
                ducklake_operation_kind = operation_context.operation_kind(),
                ducklake_operation_timeout_ms = operation_context.timeout_ms(),
                "error deleting rows"
            );
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake DELETE failed",
                format_delete_mutation_error_detail(
                    &target_table,
                    predicates.len(),
                    chunk_index,
                    chunk_count,
                    chunk.len(),
                ),
                source: error
            )
        })?;
    }

    Ok(())
}

/// Applies one update statement inside an open DuckLake transaction.
fn apply_update_mutation(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
    assignments: &[String],
    predicate: &str,
) -> EtlResult<()> {
    if assignments.is_empty() {
        return Ok(());
    }

    let set_clause = assignments.join(", ");
    let target_table = qualified_lake_table_name(table_name);
    let sql_query = format!("UPDATE {target_table} SET {set_clause} WHERE {predicate};");
    conn.execute_batch(&sql_query).map_err(|error| {
        let error = DuckDbSensitiveQueryError { error, sql: sql_query.clone() };
        tracing::error!(error = "DuckDB UPDATE failed; row-bearing diagnostics omitted from logs", table = %table_name, "error updating rows");
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake UPDATE failed",
            format_update_mutation_error_detail(&target_table, assignments.len(), !predicate.is_empty()),
            source: error
        )
    })?;

    Ok(())
}

/// Returns the number of values carried by a prepared row payload.
fn prepared_rows_count(prepared_rows: &PreparedRows) -> usize {
    match prepared_rows {
        PreparedRows::Appender(values) => values.len(),
        PreparedRows::ArrowRecordBatch(record_batch) => record_batch.num_rows(),
        PreparedRows::SqlLiterals(values) => values.len(),
    }
}

/// Returns the encoding strategy used by a prepared row payload.
fn prepared_rows_kind(prepared_rows: &PreparedRows) -> &'static str {
    match prepared_rows {
        PreparedRows::Appender(_) => "appender",
        PreparedRows::ArrowRecordBatch(_) => "arrow_record_batch",
        PreparedRows::SqlLiterals(_) => "sql_literals",
    }
}

/// Returns the number of prepared mutation statements in one atomic batch.
fn prepared_mutation_count(batch: &PreparedDuckLakeTableBatch) -> usize {
    match &batch.action {
        PreparedDuckLakeTableBatchAction::Truncate => 1,
        PreparedDuckLakeTableBatchAction::Mutation(prepared_mutations) => prepared_mutations.len(),
    }
}

/// Returns the insert row count when the batch is a pure insert sub-batch.
fn apply_sub_batch_rows(batch: &PreparedDuckLakeTableBatch) -> Option<usize> {
    let PreparedDuckLakeTableBatchAction::Mutation(prepared_mutations) = &batch.action else {
        return None;
    };

    if prepared_mutations.len() != 1 {
        return None;
    }

    match &prepared_mutations[0] {
        PreparedTableMutation::Upsert(prepared_rows) => Some(prepared_rows_count(prepared_rows)),
        PreparedTableMutation::Delete { .. } | PreparedTableMutation::Update { .. } => None,
    }
}

/// Classifies a prepared batch for concise INFO logging.
fn batch_log_kind(batch: &PreparedDuckLakeTableBatch) -> &'static str {
    match &batch.action {
        PreparedDuckLakeTableBatchAction::Truncate => "truncate",
        PreparedDuckLakeTableBatchAction::Mutation(prepared_mutations) => {
            match prepared_mutations.as_slice() {
                [PreparedTableMutation::Upsert(_)] => "insert",
                [PreparedTableMutation::Delete { origin, .. }]
                | [
                    PreparedTableMutation::Delete { origin, .. },
                    PreparedTableMutation::Upsert(_),
                ] => origin,
                [PreparedTableMutation::Update { .. }] => "update",
                _ => "mutation",
            }
        }
    }
}

/// Inserts rows into the local staging table using SQL literals.
fn insert_rows_into_staging_with_sql(
    conn: &duckdb::Connection,
    staging: &str,
    row_literals: &[String],
) -> EtlResult<()> {
    let staging_table = quote_identifier(staging);
    for chunk in row_literals.chunks(SQL_INSERT_BATCH_SIZE) {
        conn.execute_batch(&format!("INSERT INTO {staging_table} VALUES {};", chunk.join(", ")))
            .map_err(|err| {
                tracing::error!(error = %err, "error insert_rows_into_staging_with_sql");
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake staging row insert failed",
                    source: err
                )
            })?;
    }

    Ok(())
}

#[cfg(feature = "test-utils")]
static FAIL_AFTER_ATOMIC_BATCH_COMMIT_TABLE: LazyLock<Mutex<Option<String>>> =
    LazyLock::new(|| Mutex::new(None));
#[cfg(feature = "test-utils")]
static FAIL_AFTER_COPY_BATCH_COMMIT_TABLE: LazyLock<Mutex<Option<String>>> =
    LazyLock::new(|| Mutex::new(None));
#[cfg(feature = "test-utils")]
static STAGING_TABLE_CREATIONS_BY_TABLE: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Arms a test hook that injects one post-commit failure for the next atomic
/// batch.
#[cfg(feature = "test-utils")]
pub fn arm_fail_after_atomic_batch_commit_once_for_tests(table_name: &str) {
    *FAIL_AFTER_ATOMIC_BATCH_COMMIT_TABLE.lock() = Some(table_name.to_owned());
}

/// Arms a test hook that injects one post-commit failure for the next copy
/// batch.
#[cfg(feature = "test-utils")]
pub fn arm_fail_after_copy_batch_commit_once_for_tests(table_name: &str) {
    *FAIL_AFTER_COPY_BATCH_COMMIT_TABLE.lock() = Some(table_name.to_owned());
}

/// Clears DuckLake destination test hooks.
#[cfg(feature = "test-utils")]
pub fn reset_ducklake_test_hooks() {
    *FAIL_AFTER_ATOMIC_BATCH_COMMIT_TABLE.lock() = None;
    *FAIL_AFTER_COPY_BATCH_COMMIT_TABLE.lock() = None;
    STAGING_TABLE_CREATIONS_BY_TABLE.lock().clear();
}

/// Returns the number of staging-table creations performed for one table since
/// the last reset.
#[cfg(feature = "test-utils")]
pub fn ducklake_staging_table_creations_for_tests(table_name: &str) -> usize {
    STAGING_TABLE_CREATIONS_BY_TABLE.lock().get(table_name).copied().unwrap_or_default()
}

/// Injects a synthetic failure after commit so retries must rely on the correct
/// marker path.
#[cfg(feature = "test-utils")]
fn maybe_fail_after_committed_batch_for_tests(
    batch_kind: DuckLakeTableBatchKind,
    table_name: &DuckLakeTableName,
) -> EtlResult<()> {
    match batch_kind {
        DuckLakeTableBatchKind::Copy | DuckLakeTableBatchKind::CopyComplete => {
            maybe_fail_after_copy_batch_commit_for_tests(table_name)
        }
        DuckLakeTableBatchKind::Mutation | DuckLakeTableBatchKind::Truncate => {
            maybe_fail_after_atomic_batch_commit_for_tests(table_name)
        }
    }
}

/// Injects a synthetic failure after commit so retries must rely on the
/// progress row.
#[cfg(feature = "test-utils")]
fn maybe_fail_after_atomic_batch_commit_for_tests(table_name: &DuckLakeTableName) -> EtlResult<()> {
    let mut fail_table = FAIL_AFTER_ATOMIC_BATCH_COMMIT_TABLE.lock();
    if fail_table.as_deref() == Some(table_name.id().as_str()) {
        *fail_table = None;
        return Err(etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake test hook injected post-commit failure"
        ));
    }

    Ok(())
}

/// Injects a synthetic failure after commit so copy retries must rely on the
/// marker table.
#[cfg(feature = "test-utils")]
fn maybe_fail_after_copy_batch_commit_for_tests(table_name: &DuckLakeTableName) -> EtlResult<()> {
    let mut fail_table = FAIL_AFTER_COPY_BATCH_COMMIT_TABLE.lock();
    if fail_table.as_deref() == Some(table_name.id().as_str()) {
        *fail_table = None;
        return Err(etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake test hook injected copy post-commit failure"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use etl::{
        data::{OldTableRow, PartialTableRow, UpdatedTableRow},
        destination::TableCopyAttemptId,
        schema::{
            ColumnSchema, IdentityMask, ReplicatedTableSchema, ReplicationMask, TableId, TableName,
            TableSchema, Type as PgType,
        },
    };

    use super::*;

    #[test]
    fn sequence_key_format_preserves_fixed_width_hex_encoding() {
        let sequence_key = EventSequenceKey::new(PgLsn::from(1), 2);

        assert_eq!(format_sequence_key(sequence_key), "0000000000000001/0000000000000002");
    }

    fn make_schema() -> TableSchema {
        TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true),
            ],
        )
    }

    fn make_replicated_schema() -> ReplicatedTableSchema {
        ReplicatedTableSchema::all(Arc::new(make_schema()))
    }

    fn make_replicated_schema_with_columns(columns: Vec<ColumnSchema>) -> ReplicatedTableSchema {
        ReplicatedTableSchema::all(Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            columns,
        )))
    }

    #[test]
    fn replicated_column_names_use_ducklake_mapping() {
        let schema = make_replicated_schema_with_columns(vec![
            ColumnSchema::new("ID".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
            ColumnSchema::new("Display_Name".to_owned(), PgType::TEXT, -1, 2, true),
        ]);

        assert_eq!(replicated_column_names(&schema), ["id", "display_name"]);
    }

    fn ducklake_table_name() -> DuckLakeTableName {
        DuckLakeTableName::new("public", "users")
    }

    fn attach_lake_catalog(conn: &duckdb::Connection) {
        conn.execute_batch("attach ':memory:' as lake;").unwrap();
    }

    fn make_prepared_batch(table_name: DuckLakeTableName) -> PreparedDuckLakeTableBatch {
        PreparedDuckLakeTableBatch {
            table_name,
            replay_epoch: LEGACY_REPLAY_EPOCH.to_owned(),
            batch_id: "test-batch".to_owned(),
            batch_kind: DuckLakeTableBatchKind::Mutation,
            first_start_lsn: None,
            last_commit_lsn: None,
            first_sequence_key: None,
            last_sequence_key: None,
            insert_column_names: vec![],
            action: PreparedDuckLakeTableBatchAction::Mutation(vec![]),
        }
    }

    fn assert_query_failure_omits_sensitive_value(
        error: &etl::error::EtlError,
        description: &'static str,
        sensitive_value: &str,
    ) {
        assert_eq!(error.kind(), ErrorKind::DestinationQueryFailed);
        assert_eq!(error.description(), Some(description));
        if cfg!(feature = "ducklake-query-error-details") {
            assert!(error.to_string().contains(sensitive_value));
            assert!(error.to_string().contains("SQL:"));
            return;
        }
        assert!(!error.to_string().contains(sensitive_value));
        assert!(!error.detail().is_some_and(|detail| detail.contains(sensitive_value)));
        let source = error.source().expect("expected sanitized source");
        assert!(!source.to_string().contains(sensitive_value));
        assert!(source.to_string().contains("omitted because it may contain row values"));
    }

    #[test]
    fn applied_batch_marker_exists_filters_by_replay_epoch() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        attach_lake_catalog(&conn);
        conn.execute_batch(
            r#"create table lake."__etl_applied_table_batches" (
                 table_name varchar not null,
                 replay_epoch varchar,
                 batch_id varchar not null,
                 batch_kind varchar not null,
                 first_start_lsn ubigint,
                 last_commit_lsn ubigint,
                 applied_at timestamptz not null
               );"#,
        )
        .unwrap();
        let mut batch = make_prepared_batch(ducklake_table_name());
        batch.replay_epoch = "current".to_owned();
        let table_id = batch.table_name.id();

        conn.execute_batch(&format!(
            r#"insert into lake."__etl_applied_table_batches"
               (table_name, replay_epoch, batch_id, batch_kind, applied_at)
               values ({}, 'other', 'test-batch', 'mutation', current_timestamp);"#,
            quote_literal(&table_id)
        ))
        .unwrap();
        assert!(!applied_batch_marker_exists(&conn, &batch).unwrap());

        conn.execute_batch(&format!(
            r#"insert into lake."__etl_applied_table_batches"
               (table_name, replay_epoch, batch_id, batch_kind, applied_at)
               values ({}, 'current', 'test-batch', 'mutation', current_timestamp);"#,
            quote_literal(&table_id)
        ))
        .unwrap();
        assert!(applied_batch_marker_exists(&conn, &batch).unwrap());
    }

    #[test]
    fn applied_batch_marker_exists_treats_null_replay_epoch_as_legacy() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        attach_lake_catalog(&conn);
        conn.execute_batch(
            r#"create table lake."__etl_applied_table_batches" (
                 table_name varchar not null,
                 replay_epoch varchar,
                 batch_id varchar not null,
                 batch_kind varchar not null,
                 first_start_lsn ubigint,
                 last_commit_lsn ubigint,
                 applied_at timestamptz not null
               );"#,
        )
        .unwrap();
        let legacy_batch = make_prepared_batch(ducklake_table_name());
        let mut current_batch = make_prepared_batch(ducklake_table_name());
        current_batch.replay_epoch = "current".to_owned();
        let table_id = legacy_batch.table_name.id();

        conn.execute_batch(&format!(
            r#"insert into lake."__etl_applied_table_batches"
               (table_name, replay_epoch, batch_id, batch_kind, applied_at)
               values ({}, null, 'test-batch', 'mutation', current_timestamp);"#,
            quote_literal(&table_id)
        ))
        .unwrap();

        assert!(applied_batch_marker_exists(&conn, &legacy_batch).unwrap());
        assert!(!applied_batch_marker_exists(&conn, &current_batch).unwrap());
    }

    #[test]
    fn ensure_helper_table_replay_epoch_column_adds_missing_column() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        attach_lake_catalog(&conn);
        conn.execute_batch(
            r#"create table lake."__etl_applied_table_batches" (
                 table_name varchar not null,
                 batch_id varchar not null
               );"#,
        )
        .unwrap();

        assert!(
            !helper_table_has_column(&conn, APPLIED_BATCHES_TABLE, REPLAY_EPOCH_COLUMN).unwrap()
        );
        ensure_helper_table_replay_epoch_column(&conn, APPLIED_BATCHES_TABLE).unwrap();
        assert!(
            helper_table_has_column(&conn, APPLIED_BATCHES_TABLE, REPLAY_EPOCH_COLUMN).unwrap()
        );
        ensure_helper_table_replay_epoch_column(&conn, APPLIED_BATCHES_TABLE).unwrap();
    }

    #[test]
    fn read_table_streaming_progress_sequence_key_filters_by_replay_epoch() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        attach_lake_catalog(&conn);
        conn.execute_batch(
            r#"create table lake."__etl_streaming_progress" (
                 table_name varchar not null,
                 replay_epoch varchar,
                 last_commit_lsn ubigint not null,
                 last_tx_ordinal ubigint not null,
                 updated_at timestamptz not null
               );"#,
        )
        .unwrap();
        let table_name = ducklake_table_name();
        let table_id = table_name.id();

        conn.execute_batch(&format!(
            r#"insert into lake."__etl_streaming_progress"
               (table_name, replay_epoch, last_commit_lsn, last_tx_ordinal, updated_at)
               values
                 ({0}, 'current', 20, 1, current_timestamp),
                 ({0}, 'other', 999, 0, current_timestamp),
                 ({0}, 'current', 20, 2, current_timestamp),
                 ({0}, null, 30, 0, current_timestamp);"#,
            quote_literal(&table_id)
        ))
        .unwrap();

        let current_key =
            read_table_streaming_progress_sequence_key(&conn, &table_name, "current").unwrap();
        assert_eq!(current_key, Some(EventSequenceKey::new(PgLsn::from(20), 2)));

        let legacy_key =
            read_table_streaming_progress_sequence_key(&conn, &table_name, LEGACY_REPLAY_EPOCH)
                .unwrap();
        assert_eq!(legacy_key, Some(EventSequenceKey::new(PgLsn::from(30), 0)));
    }

    #[test]
    fn staging_load_rows_appends_arrow_record_batch() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "create table staging_arrow_copy (id integer, name varchar, created_at timestamp);",
        )
        .unwrap();
        let replicated_table_schema = make_replicated_schema_with_columns(vec![
            ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false),
            ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true),
            ColumnSchema::new("created_at".to_owned(), PgType::TIMESTAMP, -1, 3, true),
        ]);
        let prepared_rows = prepare_copy_rows(
            &replicated_table_schema,
            vec![
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
            ],
        )
        .unwrap();
        assert!(matches!(prepared_rows, PreparedRows::ArrowRecordBatch(_)));

        let staging_table = ReusableStagingTable {
            table_name: ducklake_table_name(),
            staging_name: "staging_arrow_copy".to_owned(),
            created: true,
            insert_column_names: vec!["id".to_owned(), "name".to_owned(), "created_at".to_owned()],
        };

        staging_table.load_rows(&conn, &prepared_rows).unwrap();

        let count: i64 = conn
            .query_row("select count(*) from staging_arrow_copy", [], |row| row.get(0))
            .unwrap();
        let id_sum: i64 =
            conn.query_row("select sum(id) from staging_arrow_copy", [], |row| row.get(0)).unwrap();
        assert_eq!(count, 2);
        assert_eq!(id_sum, 3);
    }

    #[test]
    fn apply_delete_mutation_failure_omits_row_values_from_detail() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        let batch = make_prepared_batch(ducklake_table_name());
        let operation_context = DuckLakeBlockingOperationContext::for_tests();
        let sensitive_value = "alice@example.com";
        let predicates = vec![format!("\"email\" = '{sensitive_value}'")];

        let error = apply_delete_mutation(
            &conn,
            &batch,
            predicates.as_slice(),
            "delete",
            &operation_context,
        )
        .unwrap_err();

        assert_query_failure_omits_sensitive_value(
            &error,
            "DuckLake DELETE failed",
            sensitive_value,
        );
    }

    #[test]
    fn apply_update_mutation_failure_omits_row_values_from_detail() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        let sensitive_value = "secret-token";
        let assignments = vec![format!("\"token\" = '{sensitive_value}'")];
        let predicate = format!("\"token\" = '{sensitive_value}'");

        let error = apply_update_mutation(
            &conn,
            &ducklake_table_name(),
            assignments.as_slice(),
            &predicate,
        )
        .unwrap_err();

        assert_query_failure_omits_sensitive_value(
            &error,
            "DuckLake UPDATE failed",
            sensitive_value,
        );
    }

    #[test]
    fn delete_predicate_from_row_uses_only_replica_identity_columns() {
        let replicated_table_schema = ReplicatedTableSchema::all(Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("tenant_id".to_owned(), PgType::INT4, -1, 1, false)
                    .with_primary_key(1),
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 2, false).with_primary_key(2),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true),
            ],
        )));
        let row =
            TableRow::new(vec![Cell::I32(7), Cell::I32(42), Cell::String("alice".to_owned())]);

        assert_eq!(
            delete_predicate_from_row(&replicated_table_schema, &row).unwrap(),
            "\"tenant_id\" = 7 AND \"id\" = 42"
        );
    }

    #[test]
    fn delete_predicate_from_row_supports_alternative_identity_without_primary_key() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false),
                ColumnSchema::new("email".to_owned(), PgType::TEXT, -1, 2, false),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true),
            ],
        ));
        let replicated_table_schema = ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![0, 1, 0]),
        );
        let row = TableRow::new(vec![
            Cell::I32(7),
            Cell::String("alice@example.com".to_owned()),
            Cell::String("alice".to_owned()),
        ]);

        assert_eq!(
            delete_predicate_from_row(&replicated_table_schema, &row).unwrap(),
            "\"email\" = 'alice@example.com'"
        );
    }

    #[test]
    fn delete_predicate_from_row_uses_full_replica_identity_columns() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("email".to_owned(), PgType::TEXT, -1, 2, false),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true),
            ],
        ));
        let replicated_table_schema = ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![1, 1, 1]),
        );
        let row = TableRow::new(vec![
            Cell::I32(7),
            Cell::String("alice@example.com".to_owned()),
            Cell::String("alice".to_owned()),
        ]);

        assert_eq!(
            delete_predicate_from_row(&replicated_table_schema, &row).unwrap(),
            "\"id\" = 7 AND \"email\" = 'alice@example.com' AND \"name\" = 'alice'"
        );
    }

    #[test]
    fn delete_predicate_from_row_rejects_missing_replica_identity() {
        let table_schema = Arc::new(make_schema());
        let replicated_table_schema = ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![0, 0]),
        );
        let row = TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())]);

        let error = delete_predicate_from_row(&replicated_table_schema, &row).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::SourceReplicaIdentityError);
        assert_eq!(error.description(), Some("DuckLake delete requires a replica identity"));
    }

    #[test]
    fn prepare_table_mutations_replace_emits_delete_then_upsert() {
        let replicated_table_schema = make_replicated_schema();
        let row = TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())]);

        let prepared =
            prepare_table_mutations(&replicated_table_schema, vec![TableMutation::Replace(row)])
                .unwrap();

        assert_eq!(prepared.len(), 2);
        match &prepared[0] {
            PreparedTableMutation::Delete { predicates, origin } => {
                assert_eq!(predicates, &vec!["\"id\" = 1".to_owned()]);
                assert_eq!(origin, &"replace");
            }
            PreparedTableMutation::Upsert(_) | PreparedTableMutation::Update { .. } => {
                panic!("expected delete first")
            }
        }
        match &prepared[1] {
            PreparedTableMutation::Upsert(PreparedRows::Appender(rows)) => {
                assert_eq!(rows.len(), 1);
            }
            PreparedTableMutation::Upsert(PreparedRows::SqlLiterals(_)) => {
                panic!("expected appender payload")
            }
            PreparedTableMutation::Upsert(PreparedRows::ArrowRecordBatch(_)) => {
                panic!("expected appender payload")
            }
            PreparedTableMutation::Delete { .. } | PreparedTableMutation::Update { .. } => {
                panic!("expected upsert second")
            }
        }
    }

    #[test]
    fn prepare_table_mutations_update_emits_update_statement() {
        let replicated_table_schema = make_replicated_schema();
        let prepared = prepare_table_mutations(
            &replicated_table_schema,
            vec![TableMutation::Update {
                delete_row: OldTableRow::Key(TableRow::new(vec![Cell::I32(1)])),
                new_row: UpdatedTableRow::Partial(PartialTableRow::new(
                    2,
                    TableRow::new(vec![Cell::I32(1), Cell::String("after".to_owned())]),
                    vec![],
                )),
            }],
        )
        .unwrap();

        assert_eq!(prepared.len(), 1);
        match &prepared[0] {
            PreparedTableMutation::Update { assignments, predicate } => {
                assert_eq!(
                    assignments,
                    &vec!["\"id\" = 1".to_owned(), "\"name\" = 'after'".to_owned()]
                );
                assert_eq!(predicate, "\"id\" = 1");
            }
            PreparedTableMutation::Upsert(_) | PreparedTableMutation::Delete { .. } => {
                panic!("expected update")
            }
        }
    }

    #[test]
    fn prepare_table_mutations_update_uses_alternative_identity_key_for_changed_key_update() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("email".to_owned(), PgType::TEXT, -1, 2, false),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true),
                ColumnSchema::new("payload".to_owned(), PgType::TEXT, -1, 4, true),
            ],
        ));
        let replicated_table_schema = ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![0, 1, 0, 0]),
        );

        let prepared = prepare_table_mutations(
            &replicated_table_schema,
            vec![TableMutation::Update {
                delete_row: OldTableRow::Key(TableRow::new(vec![Cell::String(
                    "alice@example.com".to_owned(),
                )])),
                new_row: UpdatedTableRow::Partial(PartialTableRow::new(
                    4,
                    TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("alice@new.example.com".to_owned()),
                        Cell::String("ripe".to_owned()),
                    ]),
                    vec![3],
                )),
            }],
        )
        .unwrap();

        assert_eq!(prepared.len(), 1);
        match &prepared[0] {
            PreparedTableMutation::Update { assignments, predicate } => {
                assert_eq!(
                    assignments,
                    &vec![
                        "\"id\" = 1".to_owned(),
                        "\"email\" = 'alice@new.example.com'".to_owned(),
                        "\"name\" = 'ripe'".to_owned(),
                    ]
                );
                assert_eq!(predicate, "\"email\" = 'alice@example.com'");
            }
            PreparedTableMutation::Upsert(_) | PreparedTableMutation::Delete { .. } => {
                panic!("expected update")
            }
        }
    }

    #[test]
    fn prepare_table_mutations_update_uses_full_replica_identity_predicate() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("email".to_owned(), PgType::TEXT, -1, 2, false),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true),
                ColumnSchema::new("payload".to_owned(), PgType::TEXT, -1, 4, true),
            ],
        ));
        let replicated_table_schema = ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![1, 1, 1, 1]),
        );

        let prepared = prepare_table_mutations(
            &replicated_table_schema,
            vec![TableMutation::Update {
                delete_row: OldTableRow::Full(TableRow::new(vec![
                    Cell::I32(1),
                    Cell::String("alice@example.com".to_owned()),
                    Cell::String("seed".to_owned()),
                    Cell::String("toast".to_owned()),
                ])),
                new_row: UpdatedTableRow::Partial(PartialTableRow::new(
                    4,
                    TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("alice@example.com".to_owned()),
                        Cell::String("grown".to_owned()),
                    ]),
                    vec![3],
                )),
            }],
        )
        .unwrap();

        assert_eq!(prepared.len(), 1);
        match &prepared[0] {
            PreparedTableMutation::Update { assignments, predicate } => {
                assert_eq!(
                    assignments,
                    &vec![
                        "\"id\" = 1".to_owned(),
                        "\"email\" = 'alice@example.com'".to_owned(),
                        "\"name\" = 'grown'".to_owned(),
                    ]
                );
                assert_eq!(
                    predicate,
                    "\"id\" = 1 AND \"email\" = 'alice@example.com' AND \"name\" = 'seed' AND \
                     \"payload\" = 'toast'"
                );
            }
            PreparedTableMutation::Upsert(_) | PreparedTableMutation::Delete { .. } => {
                panic!("expected update")
            }
        }
    }

    #[test]
    fn prepare_mutation_table_batches_insert_only_uses_single_upsert_operation() {
        let replicated_table_schema = make_replicated_schema();
        let batches = prepare_mutation_table_batches(
            None,
            &replicated_table_schema,
            ducklake_table_name(),
            LEGACY_REPLAY_EPOCH.to_owned(),
            vec![
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(20), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("alice".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(20), 1),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(2),
                        Cell::String("bob".to_owned()),
                    ])),
                ),
            ],
        )
        .unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].batch_kind, DuckLakeTableBatchKind::Mutation);
        match &batches[0].action {
            PreparedDuckLakeTableBatchAction::Mutation(prepared) => {
                assert_eq!(prepared.len(), 1);
                match &prepared[0] {
                    PreparedTableMutation::Upsert(PreparedRows::Appender(rows)) => {
                        assert_eq!(rows.len(), 2);
                    }
                    PreparedTableMutation::Upsert(PreparedRows::SqlLiterals(_)) => {
                        panic!("expected appender payload")
                    }
                    PreparedTableMutation::Upsert(PreparedRows::ArrowRecordBatch(_)) => {
                        panic!("expected appender payload")
                    }
                    PreparedTableMutation::Delete { .. } | PreparedTableMutation::Update { .. } => {
                        panic!("expected upsert")
                    }
                }
            }
            PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
        }
    }

    #[test]
    fn prepare_mutation_table_batches_split_mixed_cdc_at_delete_boundaries() {
        let replicated_table_schema = make_replicated_schema();
        let batches = prepare_mutation_table_batches(
            None,
            &replicated_table_schema,
            ducklake_table_name(),
            LEGACY_REPLAY_EPOCH.to_owned(),
            vec![
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(0),
                        Cell::String("seed".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 1),
                    TableMutation::Delete(OldTableRow::Full(TableRow::new(vec![
                        Cell::I32(0),
                        Cell::String("seed".to_owned()),
                    ]))),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 2),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(999),
                        Cell::String("tail".to_owned()),
                    ])),
                ),
            ],
        )
        .unwrap();

        assert_eq!(batches.len(), 1);

        match &batches[0].action {
            PreparedDuckLakeTableBatchAction::Mutation(prepared) => {
                assert_eq!(prepared.len(), 3);
                assert!(matches!(
                    prepared[0],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
                assert!(matches!(prepared[1], PreparedTableMutation::Delete { .. }));
                assert!(matches!(
                    prepared[2],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
            }
            PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
        }
    }

    #[test]
    fn prepare_mutation_table_batches_group_contiguous_deletes() {
        let replicated_table_schema = make_replicated_schema();
        let batches = prepare_mutation_table_batches(
            None,
            &replicated_table_schema,
            ducklake_table_name(),
            LEGACY_REPLAY_EPOCH.to_owned(),
            vec![
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 0),
                    TableMutation::Delete(OldTableRow::Full(TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("alice".to_owned()),
                    ]))),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(120), 0),
                    TableMutation::Delete(OldTableRow::Full(TableRow::new(vec![
                        Cell::I32(2),
                        Cell::String("bob".to_owned()),
                    ]))),
                ),
            ],
        )
        .unwrap();

        assert_eq!(batches.len(), 1);
        match &batches[0].action {
            PreparedDuckLakeTableBatchAction::Mutation(prepared) => {
                assert_eq!(prepared.len(), 1);
                match &prepared[0] {
                    PreparedTableMutation::Delete { predicates, origin } => {
                        assert_eq!(origin, &"delete");
                        assert_eq!(
                            predicates,
                            &vec!["\"id\" = 1".to_owned(), "\"id\" = 2".to_owned()]
                        );
                    }
                    PreparedTableMutation::Upsert(_) | PreparedTableMutation::Update { .. } => {
                        panic!("expected delete batch")
                    }
                }
            }
            PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
        }
    }

    #[test]
    fn prepare_mutation_table_batches_group_contiguous_updates() {
        let replicated_table_schema = make_replicated_schema();
        let batches = prepare_mutation_table_batches(
            None,
            &replicated_table_schema,
            ducklake_table_name(),
            LEGACY_REPLAY_EPOCH.to_owned(),
            vec![
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 0),
                    TableMutation::Update {
                        delete_row: OldTableRow::Full(TableRow::new(vec![
                            Cell::I32(1),
                            Cell::String("before-a".to_owned()),
                        ])),
                        new_row: UpdatedTableRow::Full(TableRow::new(vec![
                            Cell::I32(1),
                            Cell::String("after-a".to_owned()),
                        ])),
                    },
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(120), 0),
                    TableMutation::Update {
                        delete_row: OldTableRow::Full(TableRow::new(vec![
                            Cell::I32(2),
                            Cell::String("before-b".to_owned()),
                        ])),
                        new_row: UpdatedTableRow::Full(TableRow::new(vec![
                            Cell::I32(2),
                            Cell::String("after-b".to_owned()),
                        ])),
                    },
                ),
            ],
        )
        .unwrap();

        assert_eq!(batches.len(), 1);
        match &batches[0].action {
            PreparedDuckLakeTableBatchAction::Mutation(prepared) => {
                assert_eq!(prepared.len(), 4);
                assert!(matches!(prepared[0], PreparedTableMutation::Delete { .. }));
                assert!(matches!(
                    prepared[1],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
                assert!(matches!(prepared[2], PreparedTableMutation::Delete { .. }));
                assert!(matches!(
                    prepared[3],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
            }
            PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
        }
    }

    #[test]
    fn prepare_mutation_table_batches_split_non_inserts_at_cap() {
        for configured_size in [None, std::num::NonZeroUsize::new(CDC_MUTATION_BATCH_SIZE * 2)] {
            let replicated_table_schema = make_replicated_schema();
            let batch_size =
                configured_size.map_or(CDC_MUTATION_BATCH_SIZE, std::num::NonZeroUsize::get);
            let tracked = (0..=batch_size)
                .map(|idx| {
                    TrackedTableMutation::new(
                        EventSequenceKey::new(PgLsn::from(200 + idx as u64), 0),
                        TableMutation::Delete(OldTableRow::Full(TableRow::new(vec![
                            Cell::I32(idx as i32),
                            Cell::String(format!("name-{idx}")),
                        ]))),
                    )
                })
                .collect();
            let batches = prepare_mutation_table_batches(
                configured_size,
                &replicated_table_schema,
                ducklake_table_name(),
                LEGACY_REPLAY_EPOCH.to_owned(),
                tracked,
            )
            .unwrap();

            assert_eq!(batches.len(), 2);

            match &batches[0].action {
                PreparedDuckLakeTableBatchAction::Mutation(prepared) => match &prepared[0] {
                    PreparedTableMutation::Delete { predicates, .. } => {
                        assert_eq!(predicates.len(), batch_size);
                    }
                    PreparedTableMutation::Upsert(_) | PreparedTableMutation::Update { .. } => {
                        panic!("expected delete batch")
                    }
                },
                PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
            }

            match &batches[1].action {
                PreparedDuckLakeTableBatchAction::Mutation(prepared) => match &prepared[0] {
                    PreparedTableMutation::Delete { predicates, .. } => {
                        assert_eq!(predicates.len(), 1);
                    }
                    PreparedTableMutation::Upsert(_) | PreparedTableMutation::Update { .. } => {
                        panic!("expected delete batch")
                    }
                },
                PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
            }
        }
    }

    #[test]
    fn prepare_mutation_table_batches_isolate_update_in_its_own_atomic_batch() {
        let replicated_table_schema = make_replicated_schema();
        let batches = prepare_mutation_table_batches(
            None,
            &replicated_table_schema,
            ducklake_table_name(),
            LEGACY_REPLAY_EPOCH.to_owned(),
            vec![
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(0),
                        Cell::String("seed".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(120), 1),
                    TableMutation::Update {
                        delete_row: OldTableRow::Full(TableRow::new(vec![
                            Cell::I32(0),
                            Cell::String("seed".to_owned()),
                        ])),
                        new_row: UpdatedTableRow::Full(TableRow::new(vec![
                            Cell::I32(0),
                            Cell::String("grown".to_owned()),
                        ])),
                    },
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(130), 2),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(999),
                        Cell::String("tail".to_owned()),
                    ])),
                ),
            ],
        )
        .unwrap();

        assert_eq!(batches.len(), 1);

        match &batches[0].action {
            PreparedDuckLakeTableBatchAction::Mutation(prepared) => {
                assert_eq!(prepared.len(), 4);
                assert!(matches!(
                    prepared[0],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
                assert!(matches!(prepared[1], PreparedTableMutation::Delete { .. }));
                assert!(matches!(
                    prepared[2],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
                assert!(matches!(
                    prepared[3],
                    PreparedTableMutation::Upsert(PreparedRows::Appender(_))
                ));
            }
            PreparedDuckLakeTableBatchAction::Truncate => panic!("expected mutation batch"),
        }
    }

    #[test]
    fn retain_mutations_after_sequence_key_drops_applied_prefix() {
        let retained = retain_mutations_after_sequence_key(
            vec![
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(110), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("one".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(120), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(2),
                        Cell::String("two".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(130), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(3),
                        Cell::String("three".to_owned()),
                    ])),
                ),
            ],
            Some(EventSequenceKey::new(PgLsn::from(120), 0)),
        );

        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].sequence_key(), EventSequenceKey::new(PgLsn::from(130), 0));
    }

    #[test]
    fn retain_truncates_after_sequence_key_drops_applied_prefix() {
        let retained = retain_truncates_after_sequence_key(
            vec![
                TrackedTruncateEvent::new(EventSequenceKey::new(PgLsn::from(200), 0), 0),
                TrackedTruncateEvent::new(EventSequenceKey::new(PgLsn::from(200), 1), 0),
                TrackedTruncateEvent::new(EventSequenceKey::new(PgLsn::from(210), 0), 0),
            ],
            Some(EventSequenceKey::new(PgLsn::from(200), 0)),
        );

        assert_eq!(retained.len(), 2);
        assert_eq!(retained[0].sequence_key(), EventSequenceKey::new(PgLsn::from(200), 1));
        assert_eq!(retained[1].sequence_key(), EventSequenceKey::new(PgLsn::from(210), 0));
    }

    #[test]
    fn prepare_copy_table_batch_uses_propagated_id() {
        let replicated_table_schema = make_replicated_schema();
        let batch_id = TableCopyBatchId::new(TableCopyAttemptId::from_u128(1), 2);
        let expected_batch_id = batch_id.to_string();
        let prepared = prepare_copy_table_batch(
            &replicated_table_schema,
            ducklake_table_name(),
            LEGACY_REPLAY_EPOCH.to_owned(),
            batch_id,
            vec![TableRow::new(vec![Cell::I32(1), Cell::String("identical".to_owned())])],
        )
        .unwrap();

        assert_eq!(prepared.batch.batch_id, expected_batch_id);
    }

    #[test]
    fn build_mutation_batch_identity_is_deterministic() {
        let replicated_table_schema = make_replicated_schema();
        let tracked = vec![
            TrackedTableMutation::new(
                EventSequenceKey::new(PgLsn::from(200), 0),
                TableMutation::Insert(TableRow::new(vec![
                    Cell::I32(1),
                    Cell::String("alice".to_owned()),
                ])),
            ),
            TrackedTableMutation::new(
                EventSequenceKey::new(PgLsn::from(200), 1),
                TableMutation::Delete(OldTableRow::Full(TableRow::new(vec![
                    Cell::I32(1),
                    Cell::String("alice".to_owned()),
                ]))),
            ),
        ];

        let table_name = ducklake_table_name();
        let first =
            build_mutation_batch_identity(&table_name, &replicated_table_schema, &tracked).unwrap();
        let second =
            build_mutation_batch_identity(&table_name, &replicated_table_schema, &tracked).unwrap();

        assert_eq!(first.batch_id, second.batch_id);
        assert_eq!(first.first_start_lsn, None);
        assert_eq!(first.last_commit_lsn, Some(PgLsn::from(200)));
    }

    #[test]
    fn build_mutation_batch_identity_changes_with_order_and_sequence_key() {
        let replicated_table_schema = make_replicated_schema();
        let table_name = ducklake_table_name();
        let original = build_mutation_batch_identity(
            &table_name,
            &replicated_table_schema,
            &[
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(200), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("alice".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(200), 1),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(2),
                        Cell::String("bob".to_owned()),
                    ])),
                ),
            ],
        )
        .unwrap();
        let reordered = build_mutation_batch_identity(
            &table_name,
            &replicated_table_schema,
            &[
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(200), 0),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(2),
                        Cell::String("bob".to_owned()),
                    ])),
                ),
                TrackedTableMutation::new(
                    EventSequenceKey::new(PgLsn::from(200), 1),
                    TableMutation::Insert(TableRow::new(vec![
                        Cell::I32(1),
                        Cell::String("alice".to_owned()),
                    ])),
                ),
            ],
        )
        .unwrap();
        let changed_lsn = build_mutation_batch_identity(
            &table_name,
            &replicated_table_schema,
            &[TrackedTableMutation::new(
                EventSequenceKey::new(PgLsn::from(201), 0),
                TableMutation::Insert(TableRow::new(vec![
                    Cell::I32(1),
                    Cell::String("alice".to_owned()),
                ])),
            )],
        )
        .unwrap();

        assert_ne!(original.batch_id, reordered.batch_id);
        assert_ne!(original.batch_id, changed_lsn.batch_id);
    }

    #[test]
    fn build_truncate_batch_identity_changes_with_sequence_key() {
        let table_name = ducklake_table_name();
        let first = build_truncate_batch_identity(
            &table_name,
            &[TrackedTruncateEvent::new(EventSequenceKey::new(PgLsn::from(400), 0), 0)],
        );
        let second = build_truncate_batch_identity(
            &table_name,
            &[TrackedTruncateEvent::new(EventSequenceKey::new(PgLsn::from(401), 0), 0)],
        );

        assert_ne!(first.batch_id, second.batch_id);
    }
}
