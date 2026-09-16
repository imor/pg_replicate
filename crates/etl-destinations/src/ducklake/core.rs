use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use etl::{
    data::{OldTableRow, PartialTableRow, TableRow, UpdatedTableRow},
    destination::{
        Destination, DestinationTableMetadata, DestinationTableSchema, DestinationWriteStatus,
        DropTableForCopyResult, TableCopyBatchId, TaskSet, WriteEventsDurability,
        WriteEventsResult, WriteTableRowsResult,
    },
    error::{ErrorKind, EtlResult},
    etl_error,
    event::{Event, EventSequenceKey},
    schema::{
        ColumnAlterationKind, ColumnPresenceChangeReason, ColumnSchema, ReplicatedTableSchema,
        SchemaDiff, SchemaOperation, SchemaPlan, SnapshotId, TableId, TableName, TableSchema,
    },
    store::{DestinationStore, TableStateType},
};
use etl_config::{
    ducklake_catalog_metadata_connect_options,
    shared::{
        DuckLakeCopyBufferConfig, DuckLakeSortBy, DuckLakeSortColumn, DuckLakeSortDirection,
        DuckLakeSortNulls, DuckLakeTableSortingConfig, DuckLakeWriterConfig, Validate,
    },
};
use metrics::gauge;
use parking_lot::{Mutex, RwLock as ParkingLotRwLock};
use pg_escape::{quote_identifier as quote_postgres_identifier, quote_literal};
use sqlx::{AssertSqlSafe, PgPool, postgres::PgPoolOptions};
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};
#[cfg(feature = "test-utils")]
use tokio::sync::oneshot;
use tokio::{
    sync::{
        OwnedRwLockReadGuard, OwnedRwLockWriteGuard, OwnedSemaphorePermit, RwLock, Semaphore,
        TryAcquireError,
    },
    task::JoinSet,
};
use tracing::{debug, info, warn};
use url::Url;

use crate::{
    ducklake::{
        ATTACH_DATA_INLINING_ROW_LIMIT, COPY_DATA_INLINING_ROW_LIMIT, DUCKLAKE_COLUMN_NAME_MAPPING,
        DuckLakeTableName, LAKE_CATALOG, S3Config,
        batches::{
            DuckLakeCopyAccumulator, PreparedDuckLakeCopyBatch, TableMutation,
            TrackedTableMutation, TrackedTruncateEvent, apply_table_batch_with_retry,
            apply_table_batches_with_retry, ensure_applied_batches_table_exists,
            ensure_streaming_progress_table_exists, prepare_copy_complete_table_batch,
            prepare_copy_table_batch, prepare_mutation_table_batches, prepare_truncate_table_batch,
            read_table_streaming_progress_sequence_key, retain_mutations_after_sequence_key,
            retain_truncates_after_sequence_key,
        },
        client::{
            DuckLakeConnectionManager, DuckLakeDedicatedConnection, DuckLakeInterruptRegistry,
            build_warm_ducklake_pool, format_query_error_detail,
            is_ducklake_shutdown_requested_error, run_duckdb_blocking,
            run_duckdb_dedicated_blocking_with_context,
        },
        config::{
            DuckLakeSetupPlan, MIN_EXPIRE_SNAPSHOTS_OLDER_THAN, build_setup_plan,
            current_duckdb_extension_strategy, maintenance_target_file_size_sql,
            resolve_expire_snapshots_older_than, validate_expire_snapshots_older_than_sql,
        },
        embedding::EmbeddingOptions,
        external_maintenance::ExternalMaintenanceOperations,
        inline_size::DuckLakePendingInlineSizeSampler,
        metrics::{
            DuckLakeMetricsSampler, ETL_DUCKLAKE_POOL_SIZE, query_catalog_maintenance_metrics,
            query_table_storage_metrics, register_metrics,
            resolve_ducklake_metadata_schema_blocking, spawn_ducklake_metrics_sampler,
        },
        replay_epoch::{
            begin_table_replay_epoch_transition, complete_table_replay_epoch_transition,
            ensure_replay_epoch_table_exists, read_table_replay_epoch,
        },
        schema::{
            build_add_column_sql_ducklake, build_create_table_sql_ducklake,
            build_disable_sort_on_insert_sql_ducklake, build_drop_column_sql_ducklake,
            build_drop_default_sql_ducklake, build_drop_not_null_sql_ducklake,
            build_rename_column_sql_ducklake, build_reset_sorted_by_sql_ducklake,
            build_set_default_sql_ducklake, build_set_sorted_by_sql_ducklake,
        },
        sql::{qualified_lake_table_name, quote_identifier},
    },
    recovery::{
        ensure_destination_schema_matches_metadata, ensure_relation_schema_transition,
        warn_unsupported_column_type_change,
    },
};

/// Shared Postgres metadata pool size for DuckLake background samplers.
///
/// One connection is enough because inline-size sampling and metrics sampling
/// are both best-effort background reads and can safely serialize.
const DUCKLAKE_METADATA_PG_POOL_SIZE: u32 = 1;
/// Prefix for ETL-owned tombstone columns that keep same-name replacement DDL
/// replay-safe.
pub(super) const DUCKLAKE_DROPPED_COLUMN_PREFIX: &str = "supabase_etl_ducklake_dropped_";

/// Returns whether a name has an ETL-generated DuckLake tombstone shape.
fn is_ducklake_tombstone_column_name(column_name: &str) -> bool {
    column_name
        .strip_prefix(DUCKLAKE_DROPPED_COLUMN_PREFIX)
        .and_then(|rest| {
            let (ordinal, hash) = rest.split_once('_')?;
            let parsed_ordinal = ordinal.parse::<i32>().ok()?;
            Some((ordinal, parsed_ordinal, hash))
        })
        .is_some_and(|(ordinal, parsed_ordinal, hash)| {
            parsed_ordinal > 0
                && ordinal == parsed_ordinal.to_string()
                && hash.len() == 16
                && hash.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

/// Returns whether a source name occupies DuckLake's tombstone namespace.
///
/// DuckDB compares identifiers without ASCII case distinctions, so source
/// validation must reserve case-equivalent canonical names. Cleanup continues
/// to use [`is_ducklake_tombstone_column_name`] so ETL owns only exact names it
/// generated itself.
fn is_ducklake_tombstone_namespace_name(column_name: &str) -> bool {
    is_ducklake_tombstone_column_name(&DUCKLAKE_COLUMN_NAME_MAPPING.map_name(column_name))
}

/// Rejects source names that are indistinguishable from ETL-owned tombstones.
fn validate_ducklake_tombstone_namespace(schema: &ReplicatedTableSchema) -> EtlResult<()> {
    if let Some(column) =
        schema.column_schemas().find(|column| is_ducklake_tombstone_namespace_name(&column.name))
    {
        return Err(etl_error!(
            ErrorKind::SourceSchemaError,
            "DuckLake source column uses a reserved tombstone name",
            format!(
                "Table '{}' column '{}' matches an ETL-generated DuckLake tombstone name.",
                schema.name(),
                column.name
            )
        ));
    }

    Ok(())
}

/// Validates destination-specific DuckLake schema capabilities.
fn validate_ducklake_schema_capabilities(schema: &ReplicatedTableSchema) -> EtlResult<()> {
    validate_ducklake_tombstone_namespace(schema)
}

/// Validates a complete schema before creating or writing a DuckLake table.
fn validate_ducklake_table_shape(schema: &ReplicatedTableSchema) -> EtlResult<()> {
    schema.validate_destination_column_names(DUCKLAKE_COLUMN_NAME_MAPPING)?;
    validate_ducklake_schema_capabilities(schema)
}

/// Builds the shared Postgres metadata pool used by background samplers.
fn build_ducklake_metadata_pg_pool(catalog_url: &Url) -> EtlResult<PgPool> {
    let options = ducklake_catalog_metadata_connect_options(catalog_url).map_err(|source| {
        etl_error!(
            ErrorKind::ConfigError,
            "DuckLake metadata pool configuration failed",
            source: source
        )
    })?;

    Ok(PgPoolOptions::new()
        .max_connections(DUCKLAKE_METADATA_PG_POOL_SIZE)
        .min_connections(0)
        .acquire_timeout(std::time::Duration::from_secs(5))
        .idle_timeout(Some(std::time::Duration::from_secs(30)))
        .connect_lazy_with(options))
}

/// Drains accepted table work and reports every failure before returning one.
async fn finish_table_tasks(tasks: &mut tokio::task::JoinSet<EtlResult<()>>) -> EtlResult<()> {
    let mut first_error = None;
    while let Some(result) = tasks.join_next().await {
        let result = result.map_err(|source| etl_error!(ErrorKind::ApplyWorkerPanic, "DuckLake table task failed", source: source)).and_then(|result| result);
        if let Err(error) = result {
            // A returned error may carry opt-in row-bearing diagnostics.
            // Automatic logs keep only the owned description and error kind.
            tracing::error!(error = error.description().unwrap_or("DuckLake table task failed"), error_kind = ?error.kind(), "ducklake table task failed");
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Rejects destination names reserved for ETL replay bookkeeping.
fn validate_ducklake_table_name(table_name: &DuckLakeTableName) -> EtlResult<()> {
    if table_name.is_internal_helper() {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake destination table uses a reserved ETL helper name",
            format!("Table {table_name} cannot be used for replication")
        ));
    }
    Ok(())
}

/// Rejects another source's durable claim before creating or recovering a
/// table. Callers serialize initial creation with `table_creation_slots`;
/// pending metadata reserves a name even when physical DDL has not completed.
async fn ensure_unique_table_identity<S: DestinationStore>(
    store: &S,
    table_id: TableId,
    table_name: &DuckLakeTableName,
) -> EtlResult<()> {
    validate_ducklake_table_name(table_name)?;
    for schema in store.get_table_schemas().await? {
        if schema.id != table_id
            && let Some(metadata) = store.get_destination_table_metadata(schema.id).await?
            && let owner = DuckLakeTableName::from_metadata_id(metadata.table_id())?
            // DuckDB resolves even quoted identifiers using ASCII case folding.
            && owner.schema().eq_ignore_ascii_case(table_name.schema())
            && owner.table().eq_ignore_ascii_case(table_name.table())
        {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake destination table is already owned by another source",
                format!(
                    "Table {table_name} is owned by source {}; cannot assign it to {table_id}",
                    schema.id
                )
            ));
        }
    }
    Ok(())
}

/// Returns whether a DuckLake DDL error indicates another transaction already
/// created the requested table.
pub(super) fn is_create_table_conflict(error: &duckdb::Error, table_name: &str) -> bool {
    let message = error.to_string();
    message.contains("has been created by another transaction already")
        && message.contains(table_name)
}

/// Parses `expire_snapshots_older_than` into seconds for cheap metadata-only
/// trigger sampling.
fn expire_snapshots_retention_seconds(value: &str) -> Option<i64> {
    humantime::parse_duration(value)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

/// Returns whether file maintenance should run for a table.
///
/// Initial COPY deliberately creates Parquet files in each batch. Deferring
/// rewrites and compaction until every copy completes keeps maintenance from
/// competing with the writer and avoids repeatedly compacting a growing table.
fn should_request_file_maintenance(
    copy_phase_active: bool,
    active_data_files: i64,
    rewrite_data_files_min_active_data_files: i64,
) -> bool {
    !copy_phase_active && active_data_files > rewrite_data_files_min_active_data_files
}

/// One active sort key read from the DuckLake metadata catalog.
#[derive(Debug, PartialEq, Eq)]
struct ActiveDuckLakeSortColumn {
    expression: String,
    direction: String,
    null_order: String,
}

/// Validates and indexes configured DuckLake table sort orders.
fn index_table_sorting_config(
    config: DuckLakeTableSortingConfig,
) -> EtlResult<HashMap<DuckLakeTableName, DuckLakeSortBy>> {
    let mut table_sorting = HashMap::with_capacity(config.tables.len());

    for table in config.tables {
        if table.schema.is_empty() {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "DuckLake table sorting configuration is invalid",
                "A table sorting schema name must not be empty"
            ));
        }
        if table.table.is_empty() {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "DuckLake table sorting configuration is invalid",
                format!("Table name must not be empty for schema `{}`", table.schema)
            ));
        }

        if let DuckLakeSortBy::Columns { columns } = &table.sort_by {
            if columns.is_empty() {
                return Err(etl_error!(
                    ErrorKind::ConfigError,
                    "DuckLake table sorting configuration is invalid",
                    format!(
                        "Explicit sort columns must not be empty for `{}.{}`",
                        table.schema, table.table
                    )
                ));
            }

            let mut column_names = HashSet::with_capacity(columns.len());
            for column in columns {
                if column.name.is_empty() {
                    return Err(etl_error!(
                        ErrorKind::ConfigError,
                        "DuckLake table sorting configuration is invalid",
                        format!(
                            "Sort column names must not be empty for `{}.{}`",
                            table.schema, table.table
                        )
                    ));
                }
                if !column_names.insert(column.name.as_str()) {
                    return Err(etl_error!(
                        ErrorKind::ConfigError,
                        "DuckLake table sorting configuration is invalid",
                        format!(
                            "Sort column `{}` is configured more than once for `{}.{}`",
                            column.name, table.schema, table.table
                        )
                    ));
                }
            }
        }

        let table_name = DuckLakeTableName::new(table.schema, table.table);
        if table_sorting.insert(table_name.clone(), table.sort_by).is_some() {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "DuckLake table sorting configuration is invalid",
                format!("Table `{table_name}` is configured more than once")
            ));
        }
    }

    Ok(table_sorting)
}

/// Resolves a configured selector against the current replicated table schema.
fn resolve_table_sort_columns(
    table_sorting: &HashMap<DuckLakeTableName, DuckLakeSortBy>,
    table_name: &DuckLakeTableName,
    table_schema: &ReplicatedTableSchema,
) -> EtlResult<Option<Vec<DuckLakeSortColumn>>> {
    let Some(sort_by) = table_sorting.get(table_name) else {
        return Ok(None);
    };

    match sort_by {
        DuckLakeSortBy::Columns { columns } => {
            let replicated_columns: HashSet<_> =
                table_schema.column_schemas().map(|column| column.name.as_str()).collect();
            for column in columns {
                if !replicated_columns.contains(column.name.as_str()) {
                    return Err(etl_error!(
                        ErrorKind::ConfigError,
                        "DuckLake table sorting column is not replicated",
                        format!("Table `{table_name}` does not replicate column `{}`", column.name)
                    ));
                }
            }
            Ok(Some(
                columns
                    .iter()
                    .cloned()
                    .map(|mut column| {
                        column.name = DUCKLAKE_COLUMN_NAME_MAPPING.map_name(&column.name);
                        column
                    })
                    .collect(),
            ))
        }
        DuckLakeSortBy::PrimaryKey => {
            if !table_schema.all_primary_key_columns_replicated() {
                let columns = table_schema
                    .unreplicated_primary_key_column_schemas()
                    .map(|column| column.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(etl_error!(
                    ErrorKind::ConfigError,
                    "DuckLake primary-key sorting requires every primary-key column",
                    format!(
                        "Table `{table_name}` does not replicate primary-key columns: {columns}"
                    )
                ));
            }

            let mut columns = table_schema
                .primary_key_column_schemas()
                .filter_map(|column| {
                    column.primary_key_ordinal_position.map(|ordinal| {
                        (
                            ordinal,
                            DuckLakeSortColumn {
                                name: DUCKLAKE_COLUMN_NAME_MAPPING.map_name(&column.name),
                                direction: DuckLakeSortDirection::Asc,
                                nulls: None,
                            },
                        )
                    })
                })
                .collect::<Vec<_>>();
            if columns.is_empty() {
                return Err(etl_error!(
                    ErrorKind::ConfigError,
                    "DuckLake primary-key sorting requires a primary key",
                    format!("Table `{table_name}` has no primary key")
                ));
            }
            columns.sort_unstable_by_key(|(ordinal, _)| *ordinal);

            Ok(Some(columns.into_iter().map(|(_, column)| column).collect()))
        }
    }
}

/// Returns whether catalog metadata matches the desired column sort order.
fn active_sort_order_matches(
    active: &[ActiveDuckLakeSortColumn],
    desired: &[DuckLakeSortColumn],
) -> bool {
    active.len() == desired.len()
        && active.iter().zip(desired).all(|(active, desired)| {
            let expression_matches = active.expression == desired.name
                || active.expression == quote_identifier(&desired.name);
            let direction = match desired.direction {
                DuckLakeSortDirection::Asc => "ASC",
                DuckLakeSortDirection::Desc => "DESC",
            };
            let null_order = match desired.nulls.unwrap_or(DuckLakeSortNulls::Last) {
                DuckLakeSortNulls::First => "NULLS_FIRST",
                DuckLakeSortNulls::Last => "NULLS_LAST",
            };

            expression_matches
                && active.direction.eq_ignore_ascii_case(direction)
                && active.null_order.eq_ignore_ascii_case(null_order)
        })
}

// ── destination
// ───────────────────────────────────────────────────────────────

/// Shared handle to one initialized DuckDB connection pool.
type DuckLakePool = Arc<r2d2::Pool<DuckLakeConnectionManager>>;

/// Live connection-local buffer for one table-copy attempt.
struct DuckLakeCopyBufferHandle {
    connection: DuckLakeDedicatedConnection,
    accumulator: Arc<Mutex<DuckLakeCopyAccumulator>>,
    reservations: Mutex<Vec<OwnedSemaphorePermit>>,
    _session_permit: OwnedSemaphorePermit,
    _session_guard: OwnedRwLockReadGuard<()>,
}

impl DuckLakeCopyBufferHandle {
    /// Creates a new buffer pinned to one copy-pool connection.
    fn new(
        pool: DuckLakePool,
        batch: &PreparedDuckLakeCopyBatch,
        session_permit: OwnedSemaphorePermit,
        session_guard: OwnedRwLockReadGuard<()>,
    ) -> Self {
        Self {
            connection: DuckLakeDedicatedConnection::new(pool),
            accumulator: Arc::new(Mutex::new(DuckLakeCopyAccumulator::new(batch))),
            reservations: Mutex::new(Vec::new()),
            _session_permit: session_permit,
            _session_guard: session_guard,
        }
    }
}

/// Table-local replay state retained while deciding whether work is pending.
struct DuckLakeTableReplayCursor {
    table_name: DuckLakeTableName,
    replay_epoch: String,
    last_sequence_key: Option<EventSequenceKey>,
    table_write_permit: OwnedSemaphorePermit,
}

/// Streaming and initial-copy pools installed as one destination generation.
#[derive(Clone)]
struct DuckLakePools {
    /// Connections attached with streaming data inlining enabled.
    streaming: DuckLakePool,
    /// Connections attached with data inlining disabled for initial copy.
    copy: DuckLakePool,
}

impl DuckLakePools {
    /// Creates one fully initialized pair of DuckLake connection pools.
    fn new(streaming: DuckLakePool, copy: DuckLakePool) -> Self {
        Self { streaming, copy }
    }
}

/// Atomically replaceable DuckLake pool pair shared by destination clones.
struct DuckLakePoolHandle {
    /// Currently installed pools, or `None` after a failed maintenance refresh.
    current: ParkingLotRwLock<Option<DuckLakePools>>,
}

impl DuckLakePoolHandle {
    /// Creates a handle with an initialized pool pair.
    fn new(pools: DuckLakePools) -> Self {
        Self { current: ParkingLotRwLock::new(Some(pools)) }
    }

    /// Returns the currently installed streaming pool.
    fn streaming(&self) -> EtlResult<DuckLakePool> {
        self.current.read().as_ref().map(|pools| Arc::clone(&pools.streaming)).ok_or_else(|| {
            etl_error!(
                ErrorKind::DestinationConnectionFailed,
                "DuckLake connection pools are unavailable",
                "Pool refresh failed after external maintenance"
            )
        })
    }

    /// Returns the currently installed initial-copy pool.
    fn copy(&self) -> EtlResult<DuckLakePool> {
        self.current.read().as_ref().map(|pools| Arc::clone(&pools.copy)).ok_or_else(|| {
            etl_error!(
                ErrorKind::DestinationConnectionFailed,
                "DuckLake connection pools are unavailable",
                "Pool refresh failed after external maintenance"
            )
        })
    }

    /// Installs a fully initialized pair and returns the previous pools.
    fn replace(&self, pools: DuckLakePools) -> Option<DuckLakePools> {
        self.current.write().replace(pools)
    }

    /// Removes every installed pool after replacement initialization fails.
    fn invalidate(&self) -> Option<DuckLakePools> {
        self.current.write().take()
    }
}

/// A DuckLake destination that implements the ETL [`Destination`] trait.
///
/// Writes data to a DuckLake data lake. DuckDB connections are pre-initialized,
/// pooled, and bounded by a semaphore so operations can reuse attached lake
/// catalogs without oversubscribing Tokio's blocking threads. Data is persisted
/// as Parquet files at `data_path`; metadata is tracked in a PostgreSQL catalog
/// database.
///
/// All writes are wrapped in explicit transactions so that each batch of rows
/// is committed atomically in DuckLake while file materialization can be
/// deferred to coordinated maintenance.
#[derive(Clone)]
pub struct DuckLakeDestination<S> {
    /// Host policies retained across destination clones and pool replacement.
    embedding: EmbeddingOptions,
    manager: Arc<DuckLakeConnectionManager>,
    /// Connection manager for the pool dedicated to initial-copy writes.
    copy_manager: DuckLakeConnectionManager,
    /// Atomically replaceable streaming and initial-copy connection pools.
    pools: Arc<DuckLakePoolHandle>,
    /// Number of warm connections created in each DuckDB pool.
    pool_size: u32,
    blocking_slots: Arc<Semaphore>,
    /// Global gate that excludes external maintenance from foreground and
    /// table-scoped mutations after pinned copy sessions have drained.
    checkpoint_gate: Arc<RwLock<()>>,
    /// Gate held by connection-pinned copy sessions and acquired exclusively
    /// by maintenance before it queues on [`Self::checkpoint_gate`].
    copy_session_gate: Arc<RwLock<()>>,
    tasks: TaskSet,
    metrics_sampler: Arc<Option<DuckLakeMetricsSampler>>,
    metadata_schema: Arc<str>,
    expire_snapshots_older_than: Arc<str>,
    metadata_pg_pool: PgPool,
    /// Desired per-table sort orders indexed by source schema and table.
    table_sorting: Arc<HashMap<DuckLakeTableName, DuckLakeSortBy>>,
    table_creation_slots: Arc<Semaphore>,
    table_write_slots: Arc<Mutex<HashMap<DuckLakeTableName, Arc<Semaphore>>>>,
    /// Experimental initial-copy buffering policy.
    copy_buffer_config: DuckLakeCopyBufferConfig,
    /// Process-wide capacity for accepted but not durably committed copy rows.
    copy_buffer_capacity: Arc<Semaphore>,
    /// Maximum number of byte-denominated permits in
    /// [`Self::copy_buffer_capacity`].
    copy_buffer_max_permits: usize,
    /// Peak process-wide accepted-copy capacity observed by this destination.
    copy_buffer_peak_staged_bytes: Arc<AtomicU64>,
    /// Admission permits for connection-pinned initial-copy sessions.
    copy_session_slots: Arc<Semaphore>,
    /// Live connection-local buffers keyed by destination table.
    copy_buffers: Arc<Mutex<HashMap<DuckLakeTableName, Arc<DuckLakeCopyBufferHandle>>>>,
    /// Attempts invalidated by a staging or flush failure until table reset.
    failed_copy_buffers: Arc<Mutex<HashSet<DuckLakeTableName>>>,
    /// Shared-instance tables currently forced to write COPY rows to Parquet.
    copy_direct_to_parquet_tables: Arc<Mutex<HashSet<DuckLakeTableName>>>,
    store: S,
    /// Applied ETL-owned tables used by the background metrics sampler.
    ///
    /// This cache is rebuilt from durable destination metadata after restart.
    /// It never proves physical table existence and never drives table repair.
    applied_tables: Arc<Mutex<HashSet<DuckLakeTableName>>>,
    /// Cache tracking whether the ETL batch marker table already exists. If
    /// it's set then the table has already been created
    applied_batches_table_created: Arc<AtomicBool>,
    /// Cache tracking whether the ETL streaming progress table already exists.
    streaming_progress_table_created: Arc<AtomicBool>,
}

/// Builder for a [`DuckLakeDestination`].
///
/// The catalog URL, data path, pool size, and store are required. All runtime
/// policies start with their existing defaults and can be configured without
/// adding another constructor for each combination.
pub struct DuckLakeDestinationBuilder<S> {
    /// Optional host initialization and table mapping.
    embedding: EmbeddingOptions,
    /// DuckLake PostgreSQL catalog URL.
    catalog_url: Url,
    /// Parquet data path.
    data_path: Url,
    /// Number of warm DuckDB connections per pool.
    pool_size: u32,
    /// Optional S3 credentials and endpoint configuration.
    s3: Option<S3Config>,
    /// Optional schema that stores DuckLake metadata.
    metadata_schema: Option<String>,
    /// Optional target size used by DuckLake maintenance.
    maintenance_target_file_size: Option<String>,
    /// Optional Parquet row-group byte limit.
    parquet_row_group_size_bytes: Option<String>,
    /// Optional Parquet row-group row limit.
    parquet_row_group_size: Option<String>,
    /// Optional snapshot-retention interval.
    expire_snapshots_older_than: Option<String>,
    /// Initial-copy buffering policy.
    copy_buffer_config: DuckLakeCopyBufferConfig,
    /// Desired per-table sort orders.
    table_sorting: DuckLakeTableSortingConfig,
    /// External-maintenance coordination policy.
    external_maintenance: DuckLakeExternalMaintenanceConfig,
    /// Durable destination store.
    store: S,
}

impl<S> DuckLakeDestinationBuilder<S> {
    /// Creates a builder with every optional runtime policy disabled or
    /// defaulted.
    fn new(catalog_url: Url, data_path: Url, pool_size: u32, store: S) -> Self {
        Self {
            embedding: EmbeddingOptions::default(),
            catalog_url,
            data_path,
            pool_size,
            s3: None,
            metadata_schema: None,
            maintenance_target_file_size: None,
            parquet_row_group_size_bytes: None,
            parquet_row_group_size: None,
            expire_snapshots_older_than: None,
            copy_buffer_config: DuckLakeCopyBufferConfig::default(),
            table_sorting: DuckLakeTableSortingConfig::default(),
            external_maintenance: DuckLakeExternalMaintenanceConfig::default(),
            store,
        }
    }

    /// Uses a host-initialized DuckDB instance instead of the default setup.
    ///
    /// Called on a blocking thread at startup and on each pool replacement.
    /// Return a fresh instance with DuckLake attached as `lake`, using the same
    /// catalog, metadata schema and data path passed to this builder. The host
    /// owns extension loading, secrets, resource limits, attachment options and
    /// catalog writer options for both COPY and CDC. Standalone S3, Parquet
    /// setup and temporary COPY inlining overrides are bypassed.
    /// ETL still configures writer sessions and manages replay helper tables.
    /// Do not retain connections to retired instances in the callback.
    pub fn connection_initializer<F>(mut self, initialize: F) -> Self
    where
        F: Fn() -> EtlResult<duckdb::Connection> + Send + Sync + 'static,
    {
        self.embedding.connection_initializer = Some(Arc::new(initialize));
        self
    }

    /// Maps newly discovered tables to their durable destination names.
    ///
    /// The mapping must be deterministic, injective under ASCII
    /// case-insensitive identifier comparison, and avoid the reserved
    /// `__etl_` prefix in any case. Invalid names and identities already
    /// owned by another source are rejected. Existing tables retain their
    /// stored identity, including after a restart or a source rename;
    /// changing this callback does not move data. Table sorting policies
    /// refer to the resulting destination names.
    pub fn table_name_mapper<F>(mut self, map: F) -> Self
    where
        F: Fn(&TableName) -> EtlResult<DuckLakeTableName> + Send + Sync + 'static,
    {
        self.embedding.table_name_mapper = Some(Arc::new(map));
        self
    }

    /// Sets the maximum number of CDC mutations in one DuckLake transaction.
    ///
    /// Defaults to 16. Larger transactions can reduce file creation when data
    /// inlining is disabled, at the cost of longer transactions and retries.
    /// This only groups mutations already delivered by the pipeline; it does
    /// not wait for additional events or change the pipeline batch limits.
    pub fn cdc_batch_size(mut self, size: NonZeroUsize) -> Self {
        self.embedding.cdc_batch_size = Some(size);
        self
    }

    /// Sets optional S3 credentials and endpoint configuration.
    pub fn s3(mut self, s3: Option<S3Config>) -> Self {
        self.s3 = s3;
        self
    }

    /// Sets the optional PostgreSQL schema that stores DuckLake metadata.
    pub fn metadata_schema(mut self, metadata_schema: Option<String>) -> Self {
        self.metadata_schema = metadata_schema;
        self
    }

    /// Sets the optional DuckLake maintenance target file size.
    pub fn maintenance_target_file_size(
        mut self,
        maintenance_target_file_size: Option<String>,
    ) -> Self {
        self.maintenance_target_file_size = maintenance_target_file_size;
        self
    }

    /// Sets the optional Parquet row-group byte limit.
    pub fn parquet_row_group_size_bytes(
        mut self,
        parquet_row_group_size_bytes: Option<String>,
    ) -> Self {
        self.parquet_row_group_size_bytes = parquet_row_group_size_bytes;
        self
    }

    /// Sets the optional Parquet row-group row limit.
    pub fn parquet_row_group_size(mut self, parquet_row_group_size: Option<String>) -> Self {
        self.parquet_row_group_size = parquet_row_group_size;
        self
    }

    /// Sets the optional DuckLake snapshot-retention interval.
    pub fn expire_snapshots_older_than(
        mut self,
        expire_snapshots_older_than: Option<String>,
    ) -> Self {
        self.expire_snapshots_older_than = expire_snapshots_older_than;
        self
    }

    /// Sets the initial-copy buffering policy.
    pub fn copy_buffer(mut self, copy_buffer_config: DuckLakeCopyBufferConfig) -> Self {
        self.copy_buffer_config = copy_buffer_config;
        self
    }

    /// Sets the desired per-table sort orders.
    pub fn table_sorting(mut self, table_sorting: DuckLakeTableSortingConfig) -> Self {
        self.table_sorting = table_sorting;
        self
    }

    /// Sets the external-maintenance coordination policy.
    pub fn external_maintenance(
        mut self,
        external_maintenance: DuckLakeExternalMaintenanceConfig,
    ) -> Self {
        self.external_maintenance = external_maintenance;
        self
    }
}

impl<S> DuckLakeDestinationBuilder<S>
where
    S: DestinationStore,
{
    /// Validates the configuration and creates the destination.
    pub async fn build(self) -> EtlResult<DuckLakeDestination<S>> {
        let writer_config = DuckLakeWriterConfig::new(
            self.maintenance_target_file_size,
            self.parquet_row_group_size_bytes,
            self.parquet_row_group_size,
        );
        DuckLakeDestination::new_inner(
            self.catalog_url,
            self.data_path,
            self.pool_size,
            self.s3,
            self.metadata_schema,
            writer_config,
            self.expire_snapshots_older_than,
            self.copy_buffer_config,
            self.table_sorting,
            self.external_maintenance,
            self.embedding,
            self.store,
        )
        .await
    }
}

/// Held by an external DuckLake maintenance coordinator while foreground
/// mutations must be quiesced.
pub struct DuckLakeExternalMaintenancePause {
    _copy_session_guard: OwnedRwLockWriteGuard<()>,
    _checkpoint_guard: OwnedRwLockWriteGuard<()>,
}

/// Maintenance operations sampled from DuckLake catalog state.
pub(super) struct ExternalMaintenanceOperationSample {
    /// Operations the replicator should request.
    pub operations: ExternalMaintenanceOperations,
    /// Whether any table is currently in initial copy.
    pub copy_phase_active: bool,
}

/// Runtime backend used for DuckLake external maintenance coordination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DuckLakeMaintenanceMode {
    #[default]
    Disabled,
    Kubernetes,
    Postgres,
}

/// Runtime configuration for DuckLake external maintenance coordination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuckLakeExternalMaintenanceConfig {
    pub mode: DuckLakeMaintenanceMode,
    pub pipeline_id: u64,
}

impl DuckLakeExternalMaintenanceConfig {
    pub const fn disabled() -> Self {
        Self { mode: DuckLakeMaintenanceMode::Disabled, pipeline_id: 0 }
    }

    pub const fn kubernetes(pipeline_id: u64) -> Self {
        Self { mode: DuckLakeMaintenanceMode::Kubernetes, pipeline_id }
    }

    pub const fn postgres(pipeline_id: u64) -> Self {
        Self { mode: DuckLakeMaintenanceMode::Postgres, pipeline_id }
    }
}

impl Default for DuckLakeExternalMaintenanceConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Returns the table-local semaphore shared by concurrent foreground writes.
fn table_write_slot(
    table_write_slots: &Arc<Mutex<HashMap<DuckLakeTableName, Arc<Semaphore>>>>,
    table_name: &DuckLakeTableName,
) -> Arc<Semaphore> {
    let mut slots = table_write_slots.lock();
    let slot = slots.entry(table_name.clone()).or_insert_with(|| Arc::new(Semaphore::new(1)));
    Arc::clone(slot)
}

/// Waits for process shutdown signals and interrupts active DuckDB calls.
#[cfg(unix)]
async fn interrupt_duckdb_connections_on_process_shutdown(manager: Arc<DuckLakeConnectionManager>) {
    let Ok(mut sigterm) = signal(SignalKind::terminate()) else {
        warn!("ducklake failed to register sigterm interrupt handler");
        return;
    };
    let Ok(mut sigint) = signal(SignalKind::interrupt()) else {
        warn!("ducklake failed to register sigint interrupt handler");
        return;
    };

    let signal_name = tokio::select! {
        _ = sigterm.recv() => "sigterm",
        _ = sigint.recv() => "sigint",
    };

    let interrupted_connections = manager.interrupt_all_connections_for_process_shutdown();
    info!(
        interrupted_connections,
        signal = signal_name,
        "ducklake process shutdown signal received, interrupted active duckdb connections"
    );
}

/// Waits for process shutdown signals and interrupts active DuckDB calls.
#[cfg(not(unix))]
async fn interrupt_duckdb_connections_on_process_shutdown(manager: Arc<DuckLakeConnectionManager>) {
    if tokio::signal::ctrl_c().await.is_err() {
        warn!("ducklake failed to register ctrl-c interrupt handler");
        return;
    }

    let interrupted_connections = manager.interrupt_all_connections_for_process_shutdown();
    info!(
        interrupted_connections,
        signal = "ctrl_c",
        "ducklake process shutdown signal received, interrupted active duckdb connections"
    );
}

impl<S> Destination for DuckLakeDestination<S>
where
    S: DestinationStore,
{
    fn name() -> &'static str {
        etl_config::shared::DestinationKind::Ducklake.as_str()
    }

    async fn shutdown(&self) -> EtlResult<()> {
        let interrupted_connections = self.manager.interrupt_all_connections_for_shutdown();
        info!(
            interrupted_connections,
            "ducklake shutdown requested, interrupted active duckdb connections"
        );
        self.copy_buffers.lock().clear();
        self.failed_copy_buffers.lock().clear();
        self.copy_direct_to_parquet_tables.lock().clear();
        self.tasks.shutdown().await?;
        self.shutdown_metrics_sampler().await?;

        Ok(())
    }

    async fn startup(&self) -> EtlResult<()> {
        self.prepare_tables_after_restart().await
    }

    async fn drop_table_for_copy(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
        async_result: DropTableForCopyResult<()>,
    ) -> EtlResult<()> {
        let result = self.drop_table_for_copy_inner(replicated_table_schema).await;
        async_result.send(result);

        Ok(())
    }

    async fn write_table_rows(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
        batch_id: Option<TableCopyBatchId>,
        table_rows: Vec<TableRow>,
        async_result: WriteTableRowsResult,
    ) -> EtlResult<()> {
        let copy_complete = table_rows.is_empty();
        let result = DuckLakeDestination::write_table_rows(
            self,
            replicated_table_schema,
            batch_id,
            table_rows,
        )
        .await;
        async_result.send(result.map(|_| {
            if copy_complete {
                DestinationWriteStatus::Durable
            } else {
                DestinationWriteStatus::Accepted
            }
        }));

        Ok(())
    }

    async fn write_events(
        &self,
        events: Vec<Event>,
        _durability: WriteEventsDurability,
        async_result: WriteEventsResult,
    ) -> EtlResult<()> {
        self.tasks.try_reap().await?;

        let destination = self.clone();
        self.tasks
            .spawn_with(move || async move {
                let result = destination.write_events(events).await;
                async_result.send(result.map(|_| DestinationWriteStatus::Durable));
            })
            .await;

        Ok(())
    }
}

/// Validates that a replicated table schema can be applied to one DuckLake
/// row-matching mutation.
///
/// DuckLake can stream inserts without replica identity, but update and delete
/// paths need replicated replica-identity columns so the destination can match
/// existing rows safely.
fn validate_ducklake_replica_identity(
    replicated_table_schema: &ReplicatedTableSchema,
    operation: &'static str,
) -> EtlResult<()> {
    if replicated_table_schema.identity_column_schemas().len() == 0 {
        let description = match operation {
            "update" => "DuckLake update requires a replica identity",
            "delete" => "DuckLake delete requires a replica identity",
            _ => "DuckLake mutation requires a replica identity",
        };
        return Err(etl_error!(
            ErrorKind::SourceReplicaIdentityError,
            description,
            format!(
                "Table '{}' has no replicated replica-identity columns",
                replicated_table_schema.name()
            )
        ));
    }

    Ok(())
}

/// Builds the query used to inspect a DuckLake table shape.
fn ducklake_table_columns_sql(table_name: &DuckLakeTableName) -> String {
    format!(
        "select column_name from information_schema.columns where table_catalog = {} and \
         table_schema = {} and table_name = {} order by ordinal_position",
        quote_literal(LAKE_CATALOG),
        quote_literal(table_name.schema()),
        quote_literal(table_name.table())
    )
}

/// Reads DuckLake table column names using blocking DuckDB APIs.
///
/// Call only from a [`run_duckdb_blocking`] closure.
fn read_ducklake_table_column_names_blocking(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
) -> EtlResult<Vec<String>> {
    let sql = ducklake_table_columns_sql(table_name);
    let mut statement = conn.prepare(&sql).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake table schema lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;
    let mut rows = statement.query([]).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake table schema lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;
    let mut column_names = Vec::new();
    while let Some(row) = rows.next().map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake table schema lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })? {
        column_names.push(row.get(0).map_err(|source| {
            etl_error!(
                ErrorKind::DestinationQueryFailed,
                "DuckLake table schema lookup failed",
                format_query_error_detail(&sql),
                source: source
            )
        })?);
    }

    Ok(column_names)
}

/// Reads names of currently nullable DuckLake columns.
///
/// Call only from a [`run_duckdb_blocking`] closure.
fn read_ducklake_nullable_column_names_blocking(
    conn: &duckdb::Connection,
    table_name: &DuckLakeTableName,
) -> EtlResult<HashSet<String>> {
    let sql = format!(
        "select column_name from information_schema.columns where table_catalog = {} and \
         table_schema = {} and table_name = {} and is_nullable = 'YES'",
        quote_literal(LAKE_CATALOG),
        quote_literal(table_name.schema()),
        quote_literal(table_name.table())
    );
    let mut statement = conn.prepare(&sql).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake table nullability lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;
    let rows = statement.query_map([], |row| row.get(0)).map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake table nullability lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })?;

    rows.collect::<Result<_, _>>().map_err(|source| {
        etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake table nullability lookup failed",
            format_query_error_detail(&sql),
            source: source
        )
    })
}

/// Finds a destination column using DuckLake identifier semantics.
fn find_ducklake_column(column_names: &[String], column_name: &str) -> Option<usize> {
    column_names.iter().position(|name| DUCKLAKE_COLUMN_NAME_MAPPING.equivalent(name, column_name))
}

/// One planned DuckLake DDL statement.
#[derive(Debug, PartialEq, Eq)]
struct DuckLakeSchemaDdlStatement {
    /// SQL to execute.
    sql: String,
    /// Error description to attach if execution fails.
    error_description: &'static str,
}

/// Planned DuckLake schema DDL and the expected resulting column names.
#[derive(Debug, PartialEq, Eq)]
struct DuckLakeSchemaDdlPlan {
    /// Statements to execute in order.
    statements: Vec<DuckLakeSchemaDdlStatement>,
    /// Destination column names after applying the plan.
    column_names: Vec<String>,
}

/// Ordered CDC mutations that all use the same replicated table schema.
struct TableMutationSegment {
    /// Replicated schema used to encode every mutation in this segment.
    replicated_table_schema: ReplicatedTableSchema,
    /// Ordered mutations for the schema.
    mutations: Vec<TrackedTableMutation>,
}

/// Returns whether two replicated schemas have the same row shape and identity.
fn replicated_table_schemas_match(
    left: &ReplicatedTableSchema,
    right: &ReplicatedTableSchema,
) -> bool {
    left.id() == right.id()
        && left.inner().snapshot_id == right.inner().snapshot_id
        && left.replication_mask() == right.replication_mask()
        && left.identity_mask() == right.identity_mask()
}

/// Appends a mutation to the latest compatible schema segment.
fn push_table_mutation_segment(
    segments: &mut Vec<TableMutationSegment>,
    replicated_table_schema: ReplicatedTableSchema,
    mutation: TrackedTableMutation,
) {
    if let Some(segment) = segments.last_mut()
        && replicated_table_schemas_match(
            &segment.replicated_table_schema,
            &replicated_table_schema,
        )
    {
        segment.mutations.push(mutation);
        return;
    }

    segments.push(TableMutationSegment { replicated_table_schema, mutations: vec![mutation] });
}

/// Returns a deterministic hash for generated DuckLake identifiers.
fn stable_ducklake_identifier_hash(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Builds the tombstone name used when a dropped column name is reused.
fn dropped_column_tombstone_name_ducklake(column: &ColumnSchema) -> String {
    format!(
        "{DUCKLAKE_DROPPED_COLUMN_PREFIX}{}_{:016x}",
        column.ordinal_position,
        stable_ducklake_identifier_hash(&column.name)
    )
}

/// Returns ETL tombstone columns that are not active replicated columns.
fn tombstone_columns_to_cleanup_ducklake(
    column_names: &[String],
    target_schema: &ReplicatedTableSchema,
) -> Vec<String> {
    let active_column_names: HashSet<_> = target_schema
        .destination_column_schemas(DUCKLAKE_COLUMN_NAME_MAPPING)
        .map(|column| column.name)
        .collect();

    column_names
        .iter()
        .filter(|column_name| {
            is_ducklake_tombstone_column_name(column_name)
                && !active_column_names.contains(column_name.as_str())
        })
        .cloned()
        .collect()
}

/// Rejects recovery plans whose old and target schemas have the same cycle
/// name set and therefore cannot be distinguished without a durable marker.
fn ensure_ducklake_schema_plan_recoverable(
    table_name: &DuckLakeTableName,
    plan: &SchemaPlan,
) -> EtlResult<()> {
    if plan.has_rename_cycles() {
        return Err(etl_error!(
            ErrorKind::InvalidState,
            "DuckLake rename cycle recovery is ambiguous",
            format!(
                "Table '{table_name}' has Applying metadata for a rename cycle; the transaction \
                 may contain either the old or target schema, so manual recovery is required."
            )
        ));
    }

    Ok(())
}

/// Renders a shared plan using live DuckLake name and nullability state.
fn plan_ducklake_schema_ddl_with_nullability(
    table_name: &DuckLakeTableName,
    mut column_names: Vec<String>,
    mut nullable_column_names: HashSet<String>,
    plan: &SchemaPlan,
) -> EtlResult<DuckLakeSchemaDdlPlan> {
    let diff = plan.diff();
    let added_column_names: Vec<_> =
        diff.added_columns.iter().map(|change| change.after_column_schema.name.as_str()).collect();
    let rename_after_names: Vec<_> = diff
        .altered_columns
        .iter()
        .filter_map(|change| {
            change.name_changed().then_some(change.after_column_schema().name.as_str())
        })
        .collect();
    let reused_dropped_column_names: HashSet<_> = diff
        .dropped_columns
        .iter()
        .filter(|change| {
            added_column_names.iter().chain(&rename_after_names).any(|candidate| {
                DUCKLAKE_COLUMN_NAME_MAPPING
                    .equivalent(&change.before_column_schema.name, candidate)
            })
        })
        .map(|change| DUCKLAKE_COLUMN_NAME_MAPPING.map_name(&change.before_column_schema.name))
        .collect();
    let mut statements = Vec::new();

    // Preserve shared operation order while rendering replay-safe DuckLake DDL.
    for operation in plan.ordered_operations() {
        match operation {
            SchemaOperation::DropColumn { before_column_schema, reason: _ }
                if reused_dropped_column_names.contains(&before_column_schema.name) =>
            {
                // Keep a dropped column under a deterministic tombstone name
                // until the replacement has been applied and metadata is
                // durable. This makes same-name replacement replayable without
                // changing the shared operation order.
                let tombstone_name = dropped_column_tombstone_name_ducklake(before_column_schema);
                let before_index = find_ducklake_column(&column_names, &before_column_schema.name);
                let tombstone_index = find_ducklake_column(&column_names, &tombstone_name);

                match (before_index, tombstone_index) {
                    (Some(index), None) => {
                        statements.push(DuckLakeSchemaDdlStatement {
                            sql: build_rename_column_sql_ducklake(
                                table_name,
                                &before_column_schema.name,
                                &tombstone_name,
                            ),
                            error_description: "DuckLake alter table rename dropped column failed",
                        });
                        if nullable_column_names.remove(&before_column_schema.name) {
                            nullable_column_names.insert(tombstone_name.clone());
                        }
                        column_names[index] = tombstone_name;
                    }
                    (Some(_), Some(_)) => {
                        debug!(
                            table = %table_name,
                            column = %before_column_schema.name,
                            tombstone_column = %tombstone_name,
                            "ducklake drop column skipped because reused column name was already \
                             tombstoned"
                        );
                    }
                    (None, Some(_)) => {
                        debug!(
                            table = %table_name,
                            column = %before_column_schema.name,
                            tombstone_column = %tombstone_name,
                            "ducklake drop column skipped because destination column is already \
                             tombstoned"
                        );
                    }
                    (None, None) => {
                        debug!(
                            table = %table_name,
                            column = %before_column_schema.name,
                            "ducklake drop column skipped because destination column is already \
                             absent"
                        );
                    }
                }
            }
            SchemaOperation::DropColumn { before_column_schema, reason: _ } => {
                let Some(index) = find_ducklake_column(&column_names, &before_column_schema.name)
                else {
                    debug!(
                        table = %table_name,
                        column = %before_column_schema.name,
                        "ducklake drop column skipped because destination column is already absent"
                    );
                    continue;
                };

                statements.push(DuckLakeSchemaDdlStatement {
                    sql: build_drop_column_sql_ducklake(table_name, &before_column_schema.name),
                    error_description: "DuckLake alter table drop column failed",
                });
                nullable_column_names.remove(&before_column_schema.name);
                column_names.remove(index);
            }
            SchemaOperation::AddColumn { after_column_schema, reason } => {
                if !after_column_schema.nullable {
                    warn!(
                        table_name = %table_name,
                        column_name = %after_column_schema.name,
                        "adding a source not null column as nullable in ducklake; the destination \
                         schema will be more permissive"
                    );
                }

                if find_ducklake_column(&column_names, &after_column_schema.name).is_some() {
                    debug!(
                        table = %table_name,
                        column = %after_column_schema.name,
                        "ducklake add column skipped because destination column already exists"
                    );

                    if *reason == ColumnPresenceChangeReason::ReplicationMask {
                        if after_column_schema.default_expression.is_some() {
                            warn!(
                                table_name = %table_name,
                                column_name = %after_column_schema.name,
                                "not applying the source default to a publication-added ducklake \
                                 column; the destination schema will differ from the logical source \
                                 schema"
                            );
                        }
                    } else if let Some(default_expression) =
                        after_column_schema.default_expression.as_deref()
                    {
                        let Some(sql) = build_set_default_sql_ducklake(
                            table_name,
                            &after_column_schema.name,
                            &after_column_schema.typ,
                            default_expression,
                        ) else {
                            warn!(
                                table_name = %table_name,
                                column_name = %after_column_schema.name,
                                "skipping unsupported source column default for ducklake"
                            );
                            continue;
                        };
                        statements.push(DuckLakeSchemaDdlStatement {
                            sql,
                            error_description: "DuckLake alter table set default failed",
                        });
                    }

                    continue;
                }

                let mut destination_column_schema = after_column_schema.clone();
                if *reason == ColumnPresenceChangeReason::ReplicationMask {
                    destination_column_schema.default_expression = None;
                    if after_column_schema.default_expression.is_some() {
                        warn!(
                            table_name = %table_name,
                            column_name = %after_column_schema.name,
                            "not applying the source default to a publication-added ducklake \
                             column; the destination schema will differ from the logical source \
                             schema"
                        );
                    }
                }
                statements.push(DuckLakeSchemaDdlStatement {
                    sql: build_add_column_sql_ducklake(table_name, &destination_column_schema),
                    error_description: "DuckLake alter table add column failed",
                });
                column_names.push(after_column_schema.name.clone());
                nullable_column_names.insert(after_column_schema.name.clone());
            }
            SchemaOperation::AlterColumn { alteration }
                if alteration.kind() == ColumnAlterationKind::Rename =>
            {
                let before = alteration.before_column_schema();
                let after = alteration.after_column_schema();
                let before_index = find_ducklake_column(&column_names, &before.name);
                let after_index = find_ducklake_column(&column_names, &after.name);

                match (before_index, after_index) {
                    (Some(index), None) => {
                        statements.push(DuckLakeSchemaDdlStatement {
                            sql: build_rename_column_sql_ducklake(
                                table_name,
                                &before.name,
                                &after.name,
                            ),
                            error_description: "DuckLake alter table rename column failed",
                        });
                        if nullable_column_names.remove(&before.name) {
                            nullable_column_names.insert(after.name.clone());
                        }
                        column_names[index] = after.name.clone();
                    }
                    (None, Some(_)) => {
                        debug!(
                            table = %table_name,
                            before_column = %before.name,
                            after_column = %after.name,
                            "ducklake rename column skipped because destination column already \
                             has the after name"
                        );
                    }
                    (None, None) => {
                        return Err(etl_error!(
                            ErrorKind::CorruptedTableSchema,
                            "DuckLake destination column for rename is missing",
                            format!(
                                "Table '{table_name}' has neither before column '{}' nor after \
                                 column '{}'",
                                before.name, after.name
                            )
                        ));
                    }
                    (Some(_), Some(_))
                        if added_column_names.iter().chain(&rename_after_names).any(
                            |candidate| {
                                DUCKLAKE_COLUMN_NAME_MAPPING.equivalent(&before.name, candidate)
                            },
                        ) =>
                    {
                        debug!(
                            table = %table_name,
                            before_column = %before.name,
                            after_column = %after.name,
                            "ducklake rename column skipped because destination has both names \
                             after replay"
                        );
                    }
                    (Some(index), Some(_)) => {
                        debug!(
                            table = %table_name,
                            before_column = %before.name,
                            after_column = %after.name,
                            "ducklake dropping stale rename before column because destination \
                             already has the after name"
                        );
                        statements.push(DuckLakeSchemaDdlStatement {
                            sql: build_drop_column_sql_ducklake(table_name, &before.name),
                            error_description: "DuckLake alter table drop stale rename before \
                                                column failed",
                        });
                        nullable_column_names.remove(&before.name);
                        column_names.remove(index);
                    }
                }
            }
            SchemaOperation::AlterColumn { alteration }
                if alteration.kind() == ColumnAlterationKind::Type =>
            {
                warn_unsupported_column_type_change("ducklake", table_name, alteration);
            }
            SchemaOperation::AlterColumn { alteration }
                if alteration.kind() == ColumnAlterationKind::Nullability =>
            {
                let before = alteration.before_column_schema();
                let after = alteration.after_column_schema();
                if find_ducklake_column(&column_names, &before.name).is_none() {
                    debug!(
                        table = %table_name,
                        column = %before.name,
                        "ducklake column update skipped because destination column is absent"
                    );
                    continue;
                }

                if !before.nullable && after.nullable {
                    if nullable_column_names.contains(&before.name) {
                        debug!(
                            table = %table_name,
                            column = %before.name,
                            "ducklake column is already nullable"
                        );
                    } else {
                        statements.push(DuckLakeSchemaDdlStatement {
                            sql: build_drop_not_null_sql_ducklake(table_name, &before.name),
                            error_description: "DuckLake alter table drop not null failed",
                        });
                        nullable_column_names.insert(before.name.clone());
                    }
                } else {
                    warn!(
                        table_name = %table_name,
                        column_name = %before.name,
                        "ducklake does not tighten an existing nullable column to not null; \
                         keeping the destination column nullable"
                    );
                }
            }
            SchemaOperation::AlterColumn { alteration }
                if alteration.kind() == ColumnAlterationKind::Default =>
            {
                let before = alteration.before_column_schema();
                let after = alteration.after_column_schema();
                if find_ducklake_column(&column_names, &before.name).is_none() {
                    debug!(
                        table = %table_name,
                        column = %before.name,
                        "ducklake column update skipped because destination column is absent"
                    );
                    continue;
                }

                if before.default_expression.is_some() {
                    statements.push(DuckLakeSchemaDdlStatement {
                        sql: build_drop_default_sql_ducklake(table_name, &before.name),
                        error_description: "DuckLake alter table drop default failed",
                    });
                }

                if let Some(after_default_expression) = after.default_expression.as_deref() {
                    let Some(sql) = build_set_default_sql_ducklake(
                        table_name,
                        &before.name,
                        &after.typ,
                        after_default_expression,
                    ) else {
                        warn!(
                            table_name = %table_name,
                            column_name = %before.name,
                            "skipping unsupported source column default for ducklake"
                        );
                        continue;
                    };
                    statements.push(DuckLakeSchemaDdlStatement {
                        sql,
                        error_description: "DuckLake alter table set default failed",
                    });
                }
            }
            SchemaOperation::AlterColumn { .. } => unreachable!(
                "column alteration kind should match one of the supported planner kinds"
            ),
        }
    }

    Ok(DuckLakeSchemaDdlPlan { statements, column_names })
}

/// Returns target replicated columns that are missing from DuckLake.
fn missing_replicated_columns_ducklake(
    ducklake_columns: &[String],
    target_schema: &ReplicatedTableSchema,
) -> Vec<ColumnSchema> {
    target_schema
        .column_schemas()
        .filter(|column| find_ducklake_column(ducklake_columns, &column.name).is_none())
        .cloned()
        .collect()
}

/// Classifies a COPY inlining transition failure for the runtime retry policy.
fn classify_copy_data_inlining_error(
    table_name: &DuckLakeTableName,
    row_limit: u64,
    error: etl::error::EtlError,
) -> etl::error::EtlError {
    if is_ducklake_shutdown_requested_error(&error) {
        return error;
    }

    etl_error!(
        ErrorKind::DestinationAtomicBatchRetryable,
        "DuckLake COPY data inlining configuration failed",
        format!("table={table_name}, row_limit={row_limit}"),
        source: error
    )
}

impl<S> DuckLakeDestination<S>
where
    S: DestinationStore,
{
    /// Builds a key-only row from a partial update row when PostgreSQL omits
    /// the old key image because the replica identity did not change.
    fn key_row_from_updated_partial_row(
        replicated_table_schema: &ReplicatedTableSchema,
        partial_row: &PartialTableRow,
    ) -> EtlResult<TableRow> {
        let column_count = replicated_table_schema.column_schemas().len();
        if partial_row.total_columns() != column_count {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                "DuckLake partial update row does not match schema",
                format!(
                    "Expected {} replicated columns for table '{}', got {}",
                    column_count,
                    replicated_table_schema.name(),
                    partial_row.total_columns()
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
                    "Table '{}' partial row reports {} total columns but has {} present and {} \
                     missing",
                    replicated_table_schema.name(),
                    partial_row.total_columns(),
                    partial_row.values().len(),
                    partial_row.missing_column_indexes().len()
                )
            ));
        }

        validate_ducklake_replica_identity(replicated_table_schema, "update")?;

        let mut missing_indexes = partial_row.missing_column_indexes().iter().copied().peekable();
        let mut present_values = partial_row.values().iter();
        let identity_column_count = replicated_table_schema.identity_column_schemas().len();
        let mut identity_columns = replicated_table_schema.identity_column_schemas().peekable();
        let mut key_values = Vec::with_capacity(identity_column_count);

        for (column_index, column_schema) in replicated_table_schema.column_schemas().enumerate() {
            let is_identity = identity_columns.peek().is_some_and(|identity_column| {
                identity_column.ordinal_position == column_schema.ordinal_position
            });

            if missing_indexes.peek().copied() == Some(column_index) {
                missing_indexes.next();
                if is_identity {
                    return Err(etl_error!(
                        ErrorKind::SourceReplicaIdentityError,
                        "DuckLake partial update is missing replica-identity columns",
                        format!(
                            "Table '{}' emitted a partial update without key column '{}'",
                            replicated_table_schema.name(),
                            column_schema.name
                        )
                    ));
                }
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

            if is_identity {
                identity_columns.next();
                key_values.push(value.clone());
            }
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

        Ok(TableRow::new(key_values))
    }

    /// Deletes all rows from the destination table.
    ///
    /// This convenience wrapper preserves the pre-async-result direct-call API
    /// by awaiting the truncate work inline.
    pub async fn truncate_table(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        self.truncate_table_inner(replicated_table_schema).await
    }

    /// Writes an initial-copy batch to the destination table or copy buffer.
    ///
    /// With deferred copy buffering enabled, a nonempty call may return after
    /// staging rows on its dedicated DuckDB connection. An empty call is the
    /// terminal durability barrier and flushes all remaining rows.
    pub async fn write_table_rows(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
        batch_id: Option<TableCopyBatchId>,
        table_rows: Vec<TableRow>,
    ) -> EtlResult<()> {
        self.write_table_rows_inner(replicated_table_schema, batch_id, table_rows).await
    }

    /// Writes one streaming CDC batch directly to the destination.
    ///
    /// This convenience wrapper preserves the pre-async-result direct-call API
    /// by awaiting the batch inline.
    pub async fn write_events(&self, events: Vec<Event>) -> EtlResult<()> {
        #[cfg(feature = "test-utils")]
        wait_if_streaming_write_paused_for_tests().await;

        self.write_events_inner(events).await
    }

    /// Returns the peak process-wide copy bytes retained by deferred buffering.
    pub fn copy_buffer_peak_staged_bytes(&self) -> u64 {
        self.copy_buffer_peak_staged_bytes.load(Ordering::Relaxed)
    }

    /// Starts configuring a DuckLake destination.
    pub fn builder(
        catalog_url: Url,
        data_path: Url,
        pool_size: u32,
        store: S,
    ) -> DuckLakeDestinationBuilder<S> {
        DuckLakeDestinationBuilder::new(catalog_url, data_path, pool_size, store)
    }

    /// Creates a new DuckLake destination.
    ///
    /// - `catalog_url`: DuckLake catalog location. Use a PostgreSQL URL
    ///   (`postgres://user:pass@localhost:5432/mydb`).
    /// - `data_path`: Where Parquet files are stored. Use a local file URL (`file:///tmp/lake_data`)
    ///   or a cloud URL (`s3://bucket/prefix/`, `gs://bucket/prefix/`).
    /// - `pool_size`: Number of warm DuckDB connections maintained in the pool.
    ///   `4` is a reasonable default; higher values allow more tables to be
    ///   written in parallel.
    /// - `s3`: Optional S3 credentials. Required when `data_path` is an S3 URI
    ///   and the bucket is not publicly accessible.
    /// - `metadata_schema`: Optional Postgres schema for DuckLake metadata
    ///   tables (e.g. `"ducklake"`). Uses the catalog default schema when not
    ///   set.
    /// - `maintenance_target_file_size`: Optional DuckLake maintenance
    ///   `target_file_size` value (e.g. `"256MiB"`). Defaults to `"256MiB"`.
    /// - `expire_snapshots_older_than`: Optional DuckLake snapshot-retention
    ///   interval (e.g. `"7 days"`). Defaults to `7 days`.
    /// - `duckdb_log`: Optional DuckDB log storage and shutdown dump paths.
    /// - On Linux and macOS, DuckDB extensions are loaded from vendored local
    ///   files when a vendored directory is available. The root directory can
    ///   be forced with `ETL_DUCKDB_EXTENSION_ROOT`. Otherwise, DuckDB uses the
    ///   legacy online `INSTALL` flow. On Windows, DuckDB always uses the
    ///   legacy online `INSTALL` flow.
    ///
    /// Pool initialization is blocking because DuckDB extensions are loaded and
    /// the lake catalog is attached synchronously. This constructor offloads
    /// that warm-up work to Tokio's blocking pool.
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        catalog_url: Url,
        data_path: Url,
        pool_size: u32,
        s3: Option<S3Config>,
        metadata_schema: Option<String>,
        maintenance_target_file_size: Option<String>,
        expire_snapshots_older_than: Option<String>,
        store: S,
    ) -> EtlResult<Self> {
        Self::builder(catalog_url, data_path, pool_size, store)
            .s3(s3)
            .metadata_schema(metadata_schema)
            .maintenance_target_file_size(maintenance_target_file_size)
            .expire_snapshots_older_than(expire_snapshots_older_than)
            .build()
            .await
    }

    /// Creates a new DuckLake destination with explicit writer configuration.
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_writer_config(
        catalog_url: Url,
        data_path: Url,
        pool_size: u32,
        s3: Option<S3Config>,
        metadata_schema: Option<String>,
        writer_config: DuckLakeWriterConfig,
        expire_snapshots_older_than: Option<String>,
        store: S,
    ) -> EtlResult<Self> {
        Self::new_inner(
            catalog_url,
            data_path,
            pool_size,
            s3,
            metadata_schema,
            writer_config,
            expire_snapshots_older_than,
            DuckLakeCopyBufferConfig::default(),
            DuckLakeTableSortingConfig::default(),
            DuckLakeExternalMaintenanceConfig::default(),
            EmbeddingOptions::default(),
            store,
        )
        .await
    }

    /// Creates a new DuckLake destination with explicit external maintenance
    /// runtime configuration.
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_external_maintenance(
        catalog_url: Url,
        data_path: Url,
        pool_size: u32,
        s3: Option<S3Config>,
        metadata_schema: Option<String>,
        maintenance_target_file_size: Option<String>,
        expire_snapshots_older_than: Option<String>,
        external_maintenance: DuckLakeExternalMaintenanceConfig,
        store: S,
    ) -> EtlResult<Self> {
        Self::builder(catalog_url, data_path, pool_size, store)
            .s3(s3)
            .metadata_schema(metadata_schema)
            .maintenance_target_file_size(maintenance_target_file_size)
            .expire_snapshots_older_than(expire_snapshots_older_than)
            .external_maintenance(external_maintenance)
            .build()
            .await
    }

    /// Creates a new DuckLake destination with table sorting and external
    /// maintenance configuration.
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_table_sorting_and_external_maintenance(
        catalog_url: Url,
        data_path: Url,
        pool_size: u32,
        s3: Option<S3Config>,
        metadata_schema: Option<String>,
        maintenance_target_file_size: Option<String>,
        expire_snapshots_older_than: Option<String>,
        table_sorting: DuckLakeTableSortingConfig,
        external_maintenance: DuckLakeExternalMaintenanceConfig,
        store: S,
    ) -> EtlResult<Self> {
        Self::builder(catalog_url, data_path, pool_size, store)
            .s3(s3)
            .metadata_schema(metadata_schema)
            .maintenance_target_file_size(maintenance_target_file_size)
            .expire_snapshots_older_than(expire_snapshots_older_than)
            .table_sorting(table_sorting)
            .external_maintenance(external_maintenance)
            .build()
            .await
    }

    /// Creates a new DuckLake destination from fully resolved runtime policies.
    #[allow(clippy::too_many_arguments)]
    async fn new_inner(
        catalog_url: Url,
        data_path: Url,
        pool_size: u32,
        s3: Option<S3Config>,
        metadata_schema: Option<String>,
        writer_config: DuckLakeWriterConfig,
        expire_snapshots_older_than: Option<String>,
        copy_buffer_config: DuckLakeCopyBufferConfig,
        table_sorting: DuckLakeTableSortingConfig,
        external_maintenance: DuckLakeExternalMaintenanceConfig,
        embedding: EmbeddingOptions,
        store: S,
    ) -> EtlResult<Self> {
        register_metrics();

        if !matches!(catalog_url.scheme(), "postgres" | "postgresql") {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "DuckLake destination requires a PostgreSQL catalog URL",
                format!("unsupported catalog URL scheme `{}`", catalog_url.scheme())
            ));
        }

        if pool_size == 0 {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "DuckLake pool size must be greater than zero",
                "Pool size must be at least 1"
            ));
        }
        copy_buffer_config.validate().map_err(|error| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake copy buffer configuration is invalid",
                source: error
            )
        })?;
        if !table_sorting.is_empty()
            && external_maintenance.mode == DuckLakeMaintenanceMode::Disabled
        {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "DuckLake table sorting requires external maintenance",
                "Set DuckLake maintenance_mode to `kubernetes` or `postgres`"
            ));
        }
        let table_sorting = Arc::new(index_table_sorting_config(table_sorting)?);

        let target_file_size = Arc::<str>::from(writer_config.target_file_size());
        let expire_snapshots_older_than = Arc::<str>::from(
            resolve_expire_snapshots_older_than(expire_snapshots_older_than.as_deref()).to_owned(),
        );
        let custom_initialization = embedding.connection_initializer.is_some();
        let (setup_plan, disable_extension_autoload) = if custom_initialization {
            (Arc::new(DuckLakeSetupPlan::for_embedded_instance()), false)
        } else {
            let extension_strategy = current_duckdb_extension_strategy()?;
            let disable_extension_autoload = extension_strategy.disables_autoload();
            if let crate::ducklake::config::DuckDbExtensionStrategy::VendoredLocal {
                platform_dir,
            } = extension_strategy
            {
                info!(platform = platform_dir, "using vendored duckdb extensions");
            }
            let setup_plan = Arc::new(build_setup_plan(
                &catalog_url,
                &data_path,
                s3.as_ref(),
                metadata_schema.as_deref(),
                &writer_config,
                ATTACH_DATA_INLINING_ROW_LIMIT,
            )?);

            (setup_plan, disable_extension_autoload)
        };

        let interrupt_registry = Arc::new(DuckLakeInterruptRegistry::default());
        let shutdown_requested = Arc::new(AtomicBool::new(false));
        let manager = if embedding.connection_initializer.is_some() {
            DuckLakeConnectionManager::new_with_initializer(
                setup_plan,
                disable_extension_autoload,
                interrupt_registry,
                shutdown_requested,
                embedding.connection_initializer.clone(),
            )
            .await?
        } else {
            DuckLakeConnectionManager::new(
                setup_plan,
                disable_extension_autoload,
                interrupt_registry,
                shutdown_requested,
            )
            .await?
        };
        let copy_manager = manager.new_pool_manager();
        let manager = Arc::new(manager);
        let pool =
            Arc::new(build_warm_ducklake_pool(manager.as_ref().clone(), pool_size, "write").await?);
        let blocking_slots = Arc::new(Semaphore::new(pool_size as usize));

        // A host initializer owns catalog options, including schema-scoped
        // target sizes. Only standalone setup applies the catalog-wide default.
        if !custom_initialization {
            let target_file_size_sql = maintenance_target_file_size_sql(target_file_size.as_ref());
            run_duckdb_blocking(
                Arc::clone(&pool),
                Arc::clone(&blocking_slots),
                move |conn| -> EtlResult<()> {
                    conn.execute_batch(&target_file_size_sql).map_err(|error| {
                        etl_error!(
                            ErrorKind::DestinationQueryFailed,
                            "DuckLake target_file_size configuration failed",
                            source: error
                        )
                    })?;
                    Ok(())
                },
            )
            .await?;
        }
        let expire_snapshots_validation_sql =
            validate_expire_snapshots_older_than_sql(expire_snapshots_older_than.as_ref());
        let expire_snapshots_older_than_for_error = Arc::clone(&expire_snapshots_older_than);
        run_duckdb_blocking(
            Arc::clone(&pool),
            Arc::clone(&blocking_slots),
            move |conn| -> EtlResult<()> {
                let retention_is_safe: bool = conn
                    .query_row(&expire_snapshots_validation_sql, [], |row| row.get(0))
                    .map_err(|source| {
                        etl_error!(
                            ErrorKind::ConfigError,
                            "DuckLake expire_snapshots_older_than configuration failed",
                            format!(
                                "Invalid expire_snapshots_older_than value `{}`",
                                expire_snapshots_older_than_for_error
                            ),
                            source: source
                        )
                    })?;
                if !retention_is_safe {
                    return Err(etl_error!(
                        ErrorKind::ConfigError,
                        "DuckLake expire_snapshots_older_than configuration failed",
                        format!(
                            "Snapshot retention must be at least {}, got `{}`",
                            MIN_EXPIRE_SNAPSHOTS_OLDER_THAN, expire_snapshots_older_than_for_error
                        )
                    ));
                }
                Ok(())
            },
        )
        .await?;
        let metadata_schema = match metadata_schema {
            Some(metadata_schema) => metadata_schema,
            None => {
                run_duckdb_blocking(
                    Arc::clone(&pool),
                    Arc::clone(&blocking_slots),
                    resolve_ducklake_metadata_schema_blocking,
                )
                .await?
            }
        };
        let metadata_schema = Arc::<str>::from(metadata_schema);
        let metadata_pg_pool = build_ducklake_metadata_pg_pool(&catalog_url)?;
        ensure_replay_epoch_table_exists(&metadata_pg_pool, metadata_schema.as_ref()).await?;
        let table_creation_slots = Arc::new(Semaphore::new(1));
        let applied_batches_table_created = Arc::new(AtomicBool::new(false));
        let streaming_progress_table_created = Arc::new(AtomicBool::new(false));
        let copy_buffer_max_permits =
            u32::try_from(copy_buffer_config.max_total_bytes).map_err(|error| {
                etl_error!(
                    ErrorKind::ConfigError,
                    "DuckLake copy buffer maximum is too large",
                    source: error
                )
            })?;
        let copy_buffer_max_permits =
            usize::try_from(copy_buffer_max_permits).map_err(|error| {
                etl_error!(
                    ErrorKind::ConfigError,
                    "DuckLake copy buffer maximum does not fit this platform",
                    source: error
                )
            })?;
        let copy_session_permits = usize::try_from(pool_size).map_err(|error| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake pool size does not fit this platform",
                source: error
            )
        })?;

        // Persist helper-table inlining options before warming COPY
        // connections. Actively copied user tables temporarily override the
        // shared attachment's default, while these more specific options keep
        // ETL metadata rows inline.
        ensure_applied_batches_table_exists(
            Arc::clone(&pool),
            Arc::clone(&blocking_slots),
            Arc::clone(&table_creation_slots),
            Arc::clone(&applied_batches_table_created),
        )
        .await?;
        ensure_streaming_progress_table_exists(
            Arc::clone(&pool),
            Arc::clone(&blocking_slots),
            Arc::clone(&table_creation_slots),
            Arc::clone(&streaming_progress_table_created),
        )
        .await?;

        let copy_pool =
            Arc::new(build_warm_ducklake_pool(copy_manager.clone(), pool_size, "copy").await?);
        let pools =
            Arc::new(DuckLakePoolHandle::new(DuckLakePools::new(Arc::clone(&pool), copy_pool)));
        let applied_tables = Arc::default();
        let checkpoint_gate = Arc::new(RwLock::new(()));
        let copy_session_gate = Arc::new(RwLock::new(()));
        let mut destination = Self {
            embedding,
            manager: Arc::clone(&manager),
            copy_manager,
            pools,
            pool_size,
            blocking_slots: Arc::clone(&blocking_slots),
            checkpoint_gate: Arc::clone(&checkpoint_gate),
            copy_session_gate,
            tasks: TaskSet::new(),
            metrics_sampler: Arc::new(None),
            metadata_schema: Arc::clone(&metadata_schema),
            expire_snapshots_older_than: Arc::clone(&expire_snapshots_older_than),
            metadata_pg_pool: metadata_pg_pool.clone(),
            table_sorting,
            table_creation_slots,
            table_write_slots: Arc::default(),
            copy_buffer_config,
            copy_buffer_capacity: Arc::new(Semaphore::new(copy_buffer_max_permits)),
            copy_buffer_max_permits,
            copy_buffer_peak_staged_bytes: Arc::new(AtomicU64::new(0)),
            copy_session_slots: Arc::new(Semaphore::new(copy_session_permits)),
            copy_buffers: Arc::new(Mutex::new(HashMap::with_capacity(copy_session_permits))),
            failed_copy_buffers: Arc::default(),
            copy_direct_to_parquet_tables: Arc::default(),
            store,
            applied_tables: Arc::clone(&applied_tables),
            applied_batches_table_created,
            streaming_progress_table_created,
        };
        gauge!(ETL_DUCKLAKE_POOL_SIZE).set(pool_size as f64);
        let shutdown_signal_manager = Arc::clone(&manager);
        destination
            .tasks
            .spawn_with(move || async move {
                interrupt_duckdb_connections_on_process_shutdown(shutdown_signal_manager).await;
            })
            .await;
        destination.metrics_sampler = Arc::new(
            spawn_ducklake_metrics_sampler(
                metadata_schema.to_string(),
                metadata_pg_pool.clone(),
                Arc::clone(&applied_tables),
            )?
            .into(),
        );
        match external_maintenance.mode {
            DuckLakeMaintenanceMode::Disabled => {
                info!("ducklake external maintenance watcher disabled by configuration");
            }
            DuckLakeMaintenanceMode::Kubernetes => {
                use crate::ducklake::external_maintenance::run_kubernetes_external_maintenance_watcher;

                let watcher_destination = destination.clone();
                destination
                    .tasks
                    .spawn_with(move || async move {
                        if let Err(error) =
                            run_kubernetes_external_maintenance_watcher(watcher_destination).await
                        {
                            warn!(
                                error = %error,
                                "ducklake external maintenance watcher exited"
                            );
                        }
                    })
                    .await;
            }
            DuckLakeMaintenanceMode::Postgres => {
                use crate::ducklake::external_maintenance::run_postgres_external_maintenance_watcher;

                let watcher_destination = destination.clone();
                let maintenance_pool = metadata_pg_pool.clone();
                let pipeline_id = external_maintenance.pipeline_id as i64;
                destination
                    .tasks
                    .spawn_with(move || async move {
                        if let Err(error) = run_postgres_external_maintenance_watcher(
                            watcher_destination,
                            pipeline_id,
                            maintenance_pool,
                        )
                        .await
                        {
                            warn!(
                                error = %error,
                                "ducklake external maintenance watcher exited"
                            );
                        }
                    })
                    .await;
            }
        }

        Ok(destination)
    }

    /// Truncates the destination table while keeping its schema and name.
    async fn truncate_table_inner(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        let table_name =
            self.ensure_table_ready_for_streaming_schema(replicated_table_schema).await?;
        let _table_write_permit = self.acquire_table_write_slot(&table_name).await?;
        self.ensure_applied_batches_table_exists().await?;
        self.ensure_streaming_progress_table_exists().await?;
        let _checkpoint_guard = self.acquire_mutation_guard().await;
        let replay_epoch = self.begin_table_replay_epoch_transition(&table_name).await?;
        let table_name_for_truncate = table_name.clone();
        self.run_duckdb_blocking(move |conn| -> EtlResult<()> {
            conn.execute_batch("BEGIN TRANSACTION").map_err(|e| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake BEGIN TRANSACTION failed",
                    source: e
                )
            })?;

            let result = (|| -> EtlResult<()> {
                let target_table = qualified_lake_table_name(&table_name_for_truncate);
                let truncate_table_sql = format!("TRUNCATE TABLE {target_table};");
                conn.execute_batch(&truncate_table_sql).map_err(|e| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake TRUNCATE TABLE failed",
                        format_query_error_detail(&truncate_table_sql),
                        source: e
                    )
                })?;
                Ok(())
            })();

            match result {
                Ok(()) => conn.execute_batch("COMMIT").map_err(|e| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake COMMIT failed",
                        source: e
                    )
                }),
                Err(error) => {
                    let err = conn.execute_batch("ROLLBACK");
                    if let Err(err) = err {
                        tracing::error!(error = %err, "error rollback");
                    }
                    Err(error)
                }
            }
        })
        .await?;
        self.complete_table_replay_epoch_transition(&table_name, &replay_epoch).await?;
        debug!(
            table = %table_name,
            replay_epoch,
            "ducklake table replay epoch rotated after truncate"
        );

        Ok(())
    }

    /// Drops the destination table and rotates replay state before restarting a
    /// copy.
    async fn drop_table_for_copy_inner(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        let table_name = self.resolve_destination_table_name(replicated_table_schema).await?;
        let _table_write_permit = self.acquire_table_write_slot(&table_name).await?;
        self.copy_buffers.lock().remove(&table_name);
        #[cfg(feature = "test-utils")]
        maybe_fail_drop_table_for_copy_for_tests()?;
        self.ensure_applied_batches_table_exists().await?;
        self.ensure_streaming_progress_table_exists().await?;
        let _checkpoint_guard = self.acquire_mutation_guard().await;
        let replay_epoch = self.begin_table_replay_epoch_transition(&table_name).await?;
        let table_name_for_drop = table_name.clone();

        self.run_duckdb_blocking(move |conn| -> EtlResult<()> {
            conn.execute_batch("begin transaction").map_err(|e| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake BEGIN TRANSACTION failed",
                    source: e
                )
            })?;

            let result = (|| -> EtlResult<()> {
                let table_name = qualified_lake_table_name(&table_name_for_drop);
                let drop_table_sql = format!("drop table if exists {table_name};");
                conn.execute_batch(&drop_table_sql).map_err(|e| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake DROP TABLE failed",
                        format_query_error_detail(&drop_table_sql),
                        source: e
                    )
                })?;
                Ok(())
            })();

            match result {
                Ok(()) => conn.execute_batch("commit").map_err(|e| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake COMMIT failed",
                        source: e
                    )
                }),
                Err(error) => {
                    let err = conn.execute_batch("rollback");
                    if let Err(err) = err {
                        tracing::error!(error = %err, "error rollback");
                    }
                    Err(error)
                }
            }
        })
        .await?;
        self.complete_table_replay_epoch_transition(&table_name, &replay_epoch).await?;
        debug!(
            table = %table_name,
            replay_epoch,
            "ducklake table replay epoch rotated after drop-for-copy"
        );

        self.applied_tables.lock().remove(&table_name);
        self.failed_copy_buffers.lock().remove(&table_name);
        self.copy_direct_to_parquet_tables.lock().remove(&table_name);

        Ok(())
    }

    /// Forces one shared-instance table to materialize initial-copy rows as
    /// Parquet instead of inheriting the streaming attachment's inline limit.
    async fn ensure_copy_writes_direct_to_parquet(
        &self,
        table_name: &DuckLakeTableName,
    ) -> EtlResult<()> {
        // Host setup owns inlining for both COPY and CDC. Do not introduce a
        // table override that would mask its attachment or schema settings.
        if self.embedding.connection_initializer.is_some() {
            return Ok(());
        }

        if self.copy_direct_to_parquet_tables.lock().contains(table_name) {
            return Ok(());
        }

        self.set_copy_data_inlining_row_limit(table_name, COPY_DATA_INLINING_ROW_LIMIT).await?;
        self.copy_direct_to_parquet_tables.lock().insert(table_name.clone());
        Ok(())
    }

    /// Restores the streaming inline limit after a shared-instance copy reaches
    /// its terminal durability barrier.
    async fn restore_streaming_data_inlining(
        &self,
        table_name: &DuckLakeTableName,
    ) -> EtlResult<()> {
        // Host setup owns inlining for both COPY and CDC. Do not introduce a
        // table override that would mask its attachment or schema settings.
        if self.embedding.connection_initializer.is_some() {
            return Ok(());
        }

        self.set_copy_data_inlining_row_limit(table_name, ATTACH_DATA_INLINING_ROW_LIMIT).await?;
        self.copy_direct_to_parquet_tables.lock().remove(table_name);
        Ok(())
    }

    /// Sets one table's COPY-related inlining threshold with retryable error
    /// classification.
    async fn set_copy_data_inlining_row_limit(
        &self,
        table_name: &DuckLakeTableName,
        row_limit: u64,
    ) -> EtlResult<()> {
        self.set_table_data_inlining_row_limit(table_name, row_limit)
            .await
            .map_err(|error| classify_copy_data_inlining_error(table_name, row_limit, error))
    }

    /// Sets one DuckLake table's data-inlining threshold.
    async fn set_table_data_inlining_row_limit(
        &self,
        table_name: &DuckLakeTableName,
        row_limit: u64,
    ) -> EtlResult<()> {
        let sql = format!(
            "CALL {LAKE_CATALOG}.set_option('data_inlining_row_limit', {row_limit}, schema => {}, \
             table_name => {});",
            quote_literal(table_name.schema()),
            quote_literal(table_name.table()),
        );
        let table_name = table_name.clone();
        self.run_duckdb_blocking(move |conn| {
            debug!(table = %table_name, row_limit, "ducklake table data inlining configuration begin");
            conn.execute_batch(&sql).map_err(|error| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake table data inlining configuration failed",
                    format_query_error_detail(&sql),
                    source: error
                )
            })?;
            debug!(table = %table_name, row_limit, "ducklake table data inlining configuration finished");
            Ok(())
        })
        .await
    }

    /// Bulk-inserts rows into the destination table inside a single
    /// transaction.
    ///
    /// Wrapping all inserts in one `BEGIN` / `COMMIT` ensures they are written
    /// as one atomic DuckLake change rather than one file per row.
    ///
    /// Copy batches are recorded in the replay marker table so a retry after an
    /// ambiguous post-commit failure can detect already applied rows.
    ///
    /// Initial-copy rows are written directly to Parquet files. This avoids
    /// accumulating large snapshot loads in the catalog when source batches
    /// are smaller than the regular streaming inline threshold.
    async fn write_table_rows_inner(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
        batch_id: Option<TableCopyBatchId>,
        table_rows: Vec<TableRow>,
    ) -> EtlResult<()> {
        if self.copy_buffer_config.enabled {
            return self
                .write_table_rows_buffered_inner(replicated_table_schema, batch_id, table_rows)
                .await;
        }

        self.write_table_rows_immediate_inner(replicated_table_schema, batch_id, table_rows).await
    }

    /// Writes one initial-copy batch using the existing per-batch commit path.
    async fn write_table_rows_immediate_inner(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
        batch_id: Option<TableCopyBatchId>,
        table_rows: Vec<TableRow>,
    ) -> EtlResult<()> {
        let table_name = self.prepare_table_for_writes(replicated_table_schema).await?;
        let copy_complete = table_rows.is_empty();

        // Copy batches for the same table must still serialize so concurrent
        // callers do not race each other inside DuckDB.
        self.ensure_applied_batches_table_exists().await?;
        let _table_write_permit = self.acquire_table_write_slot(&table_name).await?;
        let _checkpoint_guard = self.acquire_mutation_guard().await;
        if !copy_complete {
            self.ensure_copy_writes_direct_to_parquet(&table_name).await?;
        }
        let replay_epoch = self.read_table_replay_epoch(&table_name).await?;
        let prepared_batch = match (batch_id, copy_complete) {
            (Some(batch_id), false) => prepare_copy_table_batch(
                replicated_table_schema,
                table_name.clone(),
                replay_epoch,
                batch_id,
                table_rows,
            )?
            .into_atomic_batch(),
            (None, true) => prepare_copy_complete_table_batch(table_name.clone(), replay_epoch),
            _ => {
                return Err(etl_error!(
                    ErrorKind::InvalidState,
                    "Table copy batch ID and rows are inconsistent"
                ));
            }
        };
        apply_table_batch_with_retry(
            self.copy_pool()?,
            Arc::clone(&self.blocking_slots),
            prepared_batch,
        )
        .await?;
        if copy_complete {
            self.restore_streaming_data_inlining(&table_name).await?;
        }

        Ok(())
    }

    /// Stages initial-copy rows and commits larger windows at the configured
    /// threshold or terminal durability barrier.
    async fn write_table_rows_buffered_inner(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
        batch_id: Option<TableCopyBatchId>,
        table_rows: Vec<TableRow>,
    ) -> EtlResult<()> {
        let table_name = self.prepare_table_for_writes(replicated_table_schema).await?;
        self.ensure_applied_batches_table_exists().await?;
        let _table_write_permit = self.acquire_table_write_slot(&table_name).await?;
        if !table_rows.is_empty() {
            let _checkpoint_guard = self.acquire_mutation_guard().await;
            self.ensure_copy_writes_direct_to_parquet(&table_name).await?;
        }
        let replay_epoch = self.read_table_replay_epoch(&table_name).await?;

        if self.failed_copy_buffers.lock().contains(&table_name) {
            return Err(Self::invalidated_copy_buffer_error(&table_name));
        }

        if table_rows.is_empty() {
            if batch_id.is_some() {
                return Err(etl_error!(
                    ErrorKind::InvalidState,
                    "Table copy batch ID and rows are inconsistent"
                ));
            }
            let copy_complete = prepare_copy_complete_table_batch(table_name.clone(), replay_epoch);
            let handle = self.copy_buffers.lock().get(&table_name).cloned();
            let Some(handle) = handle else {
                let _checkpoint_guard = self.acquire_mutation_guard().await;
                apply_table_batch_with_retry(
                    self.copy_pool()?,
                    Arc::clone(&self.blocking_slots),
                    copy_complete,
                )
                .await?;
                self.restore_streaming_data_inlining(&table_name).await?;
                return Ok(());
            };

            if let Err(error) = self.flush_copy_buffer(&handle, Some(copy_complete)).await {
                self.invalidate_copy_buffer(&table_name);
                return Err(error);
            }
            self.restore_streaming_data_inlining(&table_name).await?;
            handle.reservations.lock().clear();
            self.copy_buffers.lock().remove(&table_name);
            return Ok(());
        }

        let batch_id = batch_id.ok_or_else(|| {
            etl_error!(ErrorKind::InvalidState, "Table copy batch ID and rows are inconsistent")
        })?;
        let prepared_batch = prepare_copy_table_batch(
            replicated_table_schema,
            table_name.clone(),
            replay_epoch,
            batch_id,
            table_rows,
        )?;
        self.validate_copy_buffer_batch_size(prepared_batch.estimated_bytes())?;
        let handle = self.copy_buffer_handle(&table_name, &prepared_batch).await?;
        let reservation = self
            .reserve_copy_buffer_capacity(&table_name, &handle, prepared_batch.estimated_bytes())
            .await?;
        // Store the reservation before the blocking call starts. If the async
        // caller is cancelled, the detached blocking task and live session
        // retain both the capacity and copy-session fences until reset.
        handle.reservations.lock().push(reservation);
        let blocking_handle = Arc::clone(&handle);
        let connection = handle.connection.clone();
        let target_bytes = self.copy_buffer_config.target_bytes;
        let append_result = run_duckdb_dedicated_blocking_with_context(
            connection,
            Arc::clone(&self.blocking_slots),
            move |conn, _operation_context| {
                #[cfg(feature = "test-utils")]
                wait_if_copy_append_paused_for_tests();
                let mut accumulator = blocking_handle.accumulator.lock();
                let appended = accumulator.append(conn, prepared_batch)?;
                let flushed = appended && accumulator.staged_bytes() >= target_bytes;
                if flushed {
                    accumulator.flush(conn, None)?;
                }
                Ok((appended, flushed))
            },
        )
        .await;

        let (appended, flushed) = match append_result {
            Ok(result) => result,
            Err(error) => {
                self.invalidate_copy_buffer(&table_name);
                return Err(error);
            }
        };
        if flushed {
            handle.reservations.lock().clear();
        } else if !appended {
            handle.reservations.lock().pop();
        }

        Ok(())
    }

    /// Returns or creates the dedicated staging session for one copied table.
    async fn copy_buffer_handle(
        &self,
        table_name: &DuckLakeTableName,
        batch: &PreparedDuckLakeCopyBatch,
    ) -> EtlResult<Arc<DuckLakeCopyBufferHandle>> {
        if let Some(handle) = self.copy_buffers.lock().get(table_name).cloned() {
            return Ok(handle);
        }
        if self.failed_copy_buffers.lock().contains(table_name) {
            return Err(Self::invalidated_copy_buffer_error(table_name));
        }

        // Bound connection-pinned sessions independently from per-operation
        // blocking work. Waiting here must not consume a blocking permit that
        // an existing session needs in order to flush and release its pool
        // connection.
        let session_permit =
            Arc::clone(&self.copy_session_slots).acquire_owned().await.map_err(|_| {
                etl_error!(ErrorKind::InvalidState, "DuckLake copy session semaphore closed")
            })?;
        if let Some(handle) = self.copy_buffers.lock().get(table_name).cloned() {
            return Ok(handle);
        }
        if self.failed_copy_buffers.lock().contains(table_name) {
            return Err(Self::invalidated_copy_buffer_error(table_name));
        }

        // External maintenance waits on this session-specific gate before it
        // queues on the destination-wide checkpoint gate. Existing buffered
        // sessions can therefore drain without blocking unrelated mutations.
        let session_guard = Arc::clone(&self.copy_session_gate).read_owned().await;
        let handle = Arc::new(DuckLakeCopyBufferHandle::new(
            self.copy_pool()?,
            batch,
            session_permit,
            session_guard,
        ));
        self.copy_buffers.lock().insert(table_name.clone(), Arc::clone(&handle));
        Ok(handle)
    }

    /// Rejects one batch that cannot fit inside the configured global bound.
    fn validate_copy_buffer_batch_size(&self, estimated_bytes: u64) -> EtlResult<()> {
        if estimated_bytes > self.copy_buffer_config.max_total_bytes {
            return Err(etl_error!(
                ErrorKind::ValidationError,
                "DuckLake copy batch exceeds the configured buffer maximum",
                format!(
                    "estimated_bytes={estimated_bytes}, max_total_bytes={}",
                    self.copy_buffer_config.max_total_bytes
                )
            ));
        }

        Ok(())
    }

    /// Reserves process-wide accepted-copy capacity, flushing the current
    /// table first when its existing staged rows are preventing progress.
    async fn reserve_copy_buffer_capacity(
        &self,
        table_name: &DuckLakeTableName,
        handle: &Arc<DuckLakeCopyBufferHandle>,
        estimated_bytes: u64,
    ) -> EtlResult<OwnedSemaphorePermit> {
        let reserved_bytes = estimated_bytes.max(1);
        let permits = u32::try_from(reserved_bytes).map_err(|error| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake copy buffer reservation is too large",
                source: error
            )
        })?;

        match Arc::clone(&self.copy_buffer_capacity).try_acquire_many_owned(permits) {
            Ok(reservation) => {
                self.record_copy_buffer_reservation();
                return Ok(reservation);
            }
            Err(TryAcquireError::Closed) => {
                return Err(etl_error!(
                    ErrorKind::InvalidState,
                    "DuckLake copy buffer capacity semaphore closed"
                ));
            }
            Err(TryAcquireError::NoPermits) => {}
        }

        // This may wait for a cancelled caller's detached append. Both that
        // wait and the conditional DuckDB flush must stay on the blocking pool.
        if let Err(error) = self.flush_copy_buffer(handle, None).await {
            self.invalidate_copy_buffer(table_name);
            return Err(error);
        }
        handle.reservations.lock().clear();

        let reservation = Arc::clone(&self.copy_buffer_capacity)
            .acquire_many_owned(permits)
            .await
            .map_err(|_| {
                etl_error!(
                    ErrorKind::InvalidState,
                    "DuckLake copy buffer capacity semaphore closed"
                )
            })?;
        self.record_copy_buffer_reservation();

        Ok(reservation)
    }

    /// Records current deferred-copy bytes after one successful reservation.
    fn record_copy_buffer_reservation(&self) {
        let reserved_permits = self
            .copy_buffer_max_permits
            .saturating_sub(self.copy_buffer_capacity.available_permits());
        let reserved_bytes = u64::try_from(reserved_permits).unwrap_or(u64::MAX);
        self.copy_buffer_peak_staged_bytes.fetch_max(reserved_bytes, Ordering::Relaxed);
    }

    /// Flushes one dedicated copy session without retrying an ambiguous commit.
    async fn flush_copy_buffer(
        &self,
        handle: &Arc<DuckLakeCopyBufferHandle>,
        copy_complete: Option<crate::ducklake::batches::PreparedDuckLakeTableBatch>,
    ) -> EtlResult<()> {
        let blocking_handle = Arc::clone(handle);
        let connection = handle.connection.clone();
        run_duckdb_dedicated_blocking_with_context(
            connection,
            Arc::clone(&self.blocking_slots),
            move |conn, _operation_context| {
                blocking_handle.accumulator.lock().flush(conn, copy_complete)
            },
        )
        .await
    }

    /// Invalidates a failed table-copy session without allowing transparent
    /// connection replacement.
    fn invalidate_copy_buffer(&self, table_name: &DuckLakeTableName) {
        self.copy_buffers.lock().remove(table_name);
        self.failed_copy_buffers.lock().insert(table_name.clone());
    }

    /// Builds the sticky error returned after a buffered copy attempt fails.
    fn invalidated_copy_buffer_error(table_name: &DuckLakeTableName) -> etl::error::EtlError {
        etl_error!(
            ErrorKind::DestinationAtomicBatchRetryable,
            "DuckLake buffered table copy was invalidated",
            format!("table={table_name}; restart the table-copy attempt")
        )
    }

    /// Handles a schema-change relation event by applying the destination DDL
    /// diff and advancing destination table metadata.
    async fn handle_relation_event(
        &self,
        new_replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        validate_ducklake_schema_capabilities(new_replicated_table_schema)?;
        let table_id = new_replicated_table_schema.id();
        let new_snapshot_id = new_replicated_table_schema.inner().snapshot_id;
        let new_replication_mask = new_replicated_table_schema.replication_mask().clone();

        let metadata =
            self.store.get_destination_table_metadata(table_id).await?.ok_or_else(|| {
                etl_error!(
                    ErrorKind::CorruptedTableSchema,
                    "Destination metadata missing for DuckLake schema change",
                    format!(
                        "Table {} received schema snapshot {}, but destination metadata from \
                         initial synchronization was not found.",
                        table_id, new_snapshot_id
                    )
                )
            })?;
        let table_name = DuckLakeTableName::from_metadata_id(metadata.table_id())?;
        let metadata = if metadata.is_pending() {
            self.recover_pending_metadata(
                table_id,
                &table_name,
                metadata,
                Some(new_replicated_table_schema),
            )
            .await?
        } else {
            metadata
        };

        let current_snapshot_id = metadata.snapshot_id();
        let current_replication_mask = metadata.replication_mask().clone();

        if new_snapshot_id < current_snapshot_id {
            info!(
                table_id = %table_id,
                applied_snapshot_id = %current_snapshot_id,
                replayed_snapshot_id = %new_snapshot_id,
                "ducklake stale schema relation replay skipped"
            );
            return Ok(());
        }

        // A relation carries no durable DML sequence key. Reject both schema
        // advancement and an equal-snapshot mask conflict before either can
        // drive DuckLake DDL. Older relations are harmless replay markers;
        // later row and truncate handling validates retained events after the
        // durable streaming watermark has removed an already-applied prefix.
        ensure_relation_schema_transition(
            "DuckLake",
            table_id,
            current_snapshot_id,
            &current_replication_mask,
            new_snapshot_id,
            &new_replication_mask,
        )?;

        if current_snapshot_id == new_snapshot_id {
            info!(
                table_id = %table_id,
                snapshot_id = %new_snapshot_id,
                replication_mask = %new_replication_mask,
                "ducklake schema unchanged"
            );
            return Ok(());
        }

        info!(
            table_id = %table_id,
            current_snapshot_id = %current_snapshot_id,
            new_snapshot_id = %new_snapshot_id,
            current_replication_mask = %current_replication_mask,
            new_replication_mask = %new_replication_mask,
            "ducklake schema change detected"
        );

        let current_table_schema = self
            .load_exact_table_schema(
                table_id,
                current_snapshot_id,
                "Stored schema snapshot missing for DuckLake schema change",
            )
            .await?;
        let current_schema = ReplicatedTableSchema::from_mask(
            current_table_schema,
            current_replication_mask.clone(),
        );
        let plan = current_schema
            .plan_schema_change(new_replicated_table_schema, DUCKLAKE_COLUMN_NAME_MAPPING)?;
        self.cleanup_tombstone_columns_after_applied(&table_name, &current_schema).await;

        let updated_metadata = DestinationTableMetadata::new_applied(
            table_name.to_metadata_id()?,
            current_snapshot_id,
            current_replication_mask,
        )
        .with_schema_change(new_snapshot_id, new_replication_mask)?;
        self.applied_tables.lock().remove(&table_name);
        self.store.store_destination_table_metadata(table_id, updated_metadata.clone()).await?;

        if let Err(error) = self.apply_schema_plan(&table_name, &plan).await {
            warn!(
                error = %error,
                table_id = %table_id,
                table = %table_name,
                "ducklake schema change failed"
            );
            return Err(error);
        }
        self.reconcile_missing_replicated_columns(&table_name, new_replicated_table_schema).await?;
        self.reconcile_table_sorting(&table_name, new_replicated_table_schema).await?;

        let applied_metadata = updated_metadata.to_applied();
        self.store.store_destination_table_metadata(table_id, applied_metadata).await?;
        self.cleanup_tombstone_columns_after_applied(&table_name, new_replicated_table_schema)
            .await;
        self.applied_tables.lock().insert(table_name.clone());

        info!(
            table_id = %table_id,
            table = %table_name,
            snapshot_id = %new_snapshot_id,
            "ducklake schema change completed"
        );

        Ok(())
    }

    /// Applies a schema plan while serializing with table-local writes and
    /// external maintenance.
    ///
    /// A table that was copied with data inlining disabled must restore its
    /// streaming setting before DuckLake applies a later schema change. This
    /// lets DuckLake update its inlined-data representation with the DDL.
    async fn apply_schema_plan(
        &self,
        table_name: &DuckLakeTableName,
        plan: &SchemaPlan,
    ) -> EtlResult<()> {
        if plan.is_empty() {
            debug!(table = %table_name, "ducklake schema plan is empty");
            return Ok(());
        }

        info!(
            table = %table_name,
            additions = plan.diff().added_columns.len(),
            drops = plan.diff().dropped_columns.len(),
            alterations = plan.diff().altered_columns.len(),
            "ducklake applying schema plan"
        );

        let _table_write_permit = self.acquire_table_write_slot(table_name).await?;
        let _checkpoint_guard = self.acquire_mutation_guard().await;
        let table_name = table_name.clone();
        let plan = plan.clone();

        run_duckdb_blocking(self.streaming_pool()?, Arc::clone(&self.blocking_slots), move |conn| {
            let execute_ddl = |sql: &str, description: &'static str| -> EtlResult<()> {
                conn.execute_batch(sql).map_err(|source| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        description,
                        format_query_error_detail(sql),
                        source: source
                    )
                })
            };

            execute_ddl("begin transaction", "DuckLake DDL transaction failed")?;

            let apply_result = (|| -> EtlResult<()> {
                let column_names = read_ducklake_table_column_names_blocking(conn, &table_name)?;
                let nullable_column_names =
                    read_ducklake_nullable_column_names_blocking(conn, &table_name)?;
                let DuckLakeSchemaDdlPlan { statements, column_names: _column_names } =
                    plan_ducklake_schema_ddl_with_nullability(
                        &table_name,
                        column_names,
                        nullable_column_names,
                        &plan,
                    )?;

                for statement in statements {
                    execute_ddl(&statement.sql, statement.error_description)?;
                }

                Ok(())
            })();

            if let Err(error) = apply_result {
                if let Err(rollback_error) = conn.execute_batch("rollback") {
                    warn!(
                        error = %rollback_error,
                        table = %table_name,
                        "ducklake schema change rollback failed"
                    );
                }
                return Err(error);
            }

            execute_ddl("commit", "DuckLake DDL transaction commit failed")
        })
        .await
    }

    /// Reconciles the configured sort order and maintenance-only insert policy.
    async fn reconcile_table_sorting(
        &self,
        table_name: &DuckLakeTableName,
        table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        let desired =
            resolve_table_sort_columns(self.table_sorting.as_ref(), table_name, table_schema)?;
        let _table_write_permit = self.acquire_table_write_slot(table_name).await?;
        let _checkpoint_guard = self.acquire_mutation_guard().await;

        let sql = format!(
            "select e.expression, e.sort_direction, e.null_order from {}.{} as e join {}.{} as i \
             on i.sort_id = e.sort_id and i.table_id = e.table_id join {}.{} as t on t.table_id = \
             i.table_id join {}.{} as s on s.schema_id = t.schema_id where s.schema_name = $1 and \
             t.table_name = $2 and s.end_snapshot is null and t.end_snapshot is null and \
             i.end_snapshot is null order by e.sort_key_index",
            quote_postgres_identifier(self.metadata_schema.as_ref()),
            quote_postgres_identifier("ducklake_sort_expression"),
            quote_postgres_identifier(self.metadata_schema.as_ref()),
            quote_postgres_identifier("ducklake_sort_info"),
            quote_postgres_identifier(self.metadata_schema.as_ref()),
            quote_postgres_identifier("ducklake_table"),
            quote_postgres_identifier(self.metadata_schema.as_ref()),
            quote_postgres_identifier("ducklake_schema")
        );
        let active: Vec<(String, String, String)> = sqlx::query_as(AssertSqlSafe(sql))
            .bind(table_name.schema())
            .bind(table_name.table())
            .fetch_all(&self.metadata_pg_pool)
            .await
            .map_err(|source| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake table sort order query failed",
                    format!("table={table_name}"),
                    source: source
                )
            })?;
        let active = active
            .into_iter()
            .map(|(expression, direction, null_order)| ActiveDuckLakeSortColumn {
                expression,
                direction,
                null_order,
            })
            .collect::<Vec<_>>();

        let ddl = match desired {
            Some(columns) => {
                let mut statements = Vec::with_capacity(2);
                if !active_sort_order_matches(&active, &columns) {
                    statements.push(build_set_sorted_by_sql_ducklake(table_name, &columns));
                }
                // Keep foreground insert latency unchanged. Flush and compaction
                // still use the table's active sort order.
                statements.push(build_disable_sort_on_insert_sql_ducklake(table_name));
                statements.join(";\n")
            }
            None if !active.is_empty() => build_reset_sorted_by_sql_ducklake(table_name),
            None => return Ok(()),
        };
        let table_name = table_name.clone();

        run_duckdb_blocking(self.streaming_pool()?, Arc::clone(&self.blocking_slots), move |conn| {
            conn.execute_batch(&ddl).map_err(|source| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake table sorting reconciliation failed",
                    format_query_error_detail(&ddl),
                    source: source
                )
            })?;
            debug!(table = %table_name, "ducklake table sorting reconciled");
            Ok(())
        })
        .await
    }

    /// Adds target replicated columns missing from the physical DuckLake table.
    async fn reconcile_missing_replicated_columns(
        &self,
        table_name: &DuckLakeTableName,
        target_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        let table_name_for_read = table_name.clone();
        let ducklake_columns = {
            let _checkpoint_guard = self.acquire_mutation_guard().await;
            run_duckdb_blocking(
                self.streaming_pool()?,
                Arc::clone(&self.blocking_slots),
                move |conn| read_ducklake_table_column_names_blocking(conn, &table_name_for_read),
            )
            .await?
        };
        if ducklake_columns.is_empty() {
            return Err(etl_error!(
                ErrorKind::DestinationTableMissing,
                "DuckLake destination table is missing",
                format!(
                    "Pending destination metadata identifies table '{table_name}', but the \
                     physical table does not exist and recovery could not recreate it."
                )
            ));
        }
        let missing_columns = missing_replicated_columns_ducklake(&ducklake_columns, target_schema);

        if missing_columns.is_empty() {
            return Ok(());
        }

        warn!(
            table = %table_name,
            missing_column_count = missing_columns.len(),
            "ducklake destination table is missing replicated columns, reconciling"
        );

        // Physical recovery exposes column names but not a complete current
        // source schema, so this is intentionally a synthetic add-only plan.
        let mut target_column_names = ducklake_columns.clone();
        target_column_names.extend(missing_columns.iter().map(|column| column.name.clone()));
        let plan = SchemaDiff::new(missing_columns, Vec::new(), Vec::new()).plan_for_column_names(
            ducklake_columns,
            target_column_names,
            DUCKLAKE_COLUMN_NAME_MAPPING,
        )?;

        self.apply_schema_plan(table_name, &plan).await
    }

    /// Drops ETL tombstone columns after schema metadata is durably `Applied`.
    async fn cleanup_tombstone_columns(
        &self,
        table_name: &DuckLakeTableName,
        target_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        let _table_write_permit = self.acquire_table_write_slot(table_name).await?;
        let _checkpoint_guard = self.acquire_mutation_guard().await;
        let table_name = table_name.clone();
        let target_schema = target_schema.clone();

        run_duckdb_blocking(self.streaming_pool()?, Arc::clone(&self.blocking_slots), move |conn| {
            let execute_ddl = |sql: &str, description: &'static str| -> EtlResult<()> {
                conn.execute_batch(sql).map_err(|source| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        description,
                        format_query_error_detail(sql),
                        source: source
                    )
                })
            };

            let column_names = read_ducklake_table_column_names_blocking(conn, &table_name)?;
            let columns_to_drop =
                tombstone_columns_to_cleanup_ducklake(&column_names, &target_schema);
            if columns_to_drop.is_empty() {
                return Ok(());
            }

            info!(
                table = %table_name,
                tombstone_column_count = columns_to_drop.len(),
                "ducklake cleaning up tombstone columns"
            );

            execute_ddl("begin transaction", "DuckLake tombstone cleanup transaction failed")?;

            let cleanup_result = (|| -> EtlResult<()> {
                for column_name in columns_to_drop {
                    let sql = build_drop_column_sql_ducklake(&table_name, &column_name);
                    execute_ddl(&sql, "DuckLake tombstone column drop failed")?;
                }

                Ok(())
            })();

            if let Err(error) = cleanup_result {
                if let Err(rollback_error) = conn.execute_batch("rollback") {
                    warn!(
                        error = %rollback_error,
                        table = %table_name,
                        "ducklake tombstone cleanup rollback failed"
                    );
                }
                return Err(error);
            }

            execute_ddl("commit", "DuckLake tombstone cleanup transaction commit failed")
        })
        .await
    }

    /// Best-effort wrapper for post-`Applied` tombstone cleanup.
    async fn cleanup_tombstone_columns_after_applied(
        &self,
        table_name: &DuckLakeTableName,
        target_schema: &ReplicatedTableSchema,
    ) {
        if let Err(error) = self.cleanup_tombstone_columns(table_name, target_schema).await {
            warn!(
                error = %error,
                table = %table_name,
                "ducklake tombstone column cleanup failed"
            );
        }
    }

    /// Recovers pending operations and rebuilds applied-table process state.
    ///
    /// [`DestinationTableSchema::Applied`] is authoritative: startup trusts the
    /// recorded destination table and does not inspect or repair its physical
    /// schema. Missing or externally modified tables fail on ordinary use.
    async fn prepare_tables_after_restart(&self) -> EtlResult<()> {
        let table_schemas = self.store.get_table_schemas().await?;
        let table_ids: HashSet<_> = table_schemas.iter().map(|schema| schema.id).collect();

        if table_ids.is_empty() {
            return Ok(());
        }

        info!(
            table_count = table_ids.len(),
            "ducklake recovering pending tables and rebuilding applied table state"
        );

        for table_id in table_ids {
            let Some(metadata) = self.store.get_destination_table_metadata(table_id).await? else {
                continue;
            };

            let table_name = DuckLakeTableName::from_metadata_id(metadata.table_id())?;
            ensure_unique_table_identity(&self.store, table_id, &table_name).await?;
            if metadata.is_pending() {
                self.recover_pending_metadata(table_id, &table_name, metadata, None).await?;
                continue;
            }

            self.applied_tables.lock().insert(table_name);
        }

        Ok(())
    }

    /// Writes streaming CDC events to the destination.
    ///
    /// Insert, Update, and Delete events are grouped by table and written in
    /// parallel, each table in its own async task. Each DuckDB attempt acquires
    /// one blocking slot before entering `spawn_blocking`. Each table's ordered
    /// CDC stream is split into atomic sub-batches, applied on a reused DuckDB
    /// connection per retry attempt, and acknowledged through one per-table
    /// streaming replay watermark so retries can safely detect already
    /// committed work.
    async fn write_events_inner(&self, events: Vec<Event>) -> EtlResult<()> {
        let mut event_iter = events.into_iter().peekable();

        while event_iter.peek().is_some() {
            let mut table_id_to_mutations: HashMap<TableId, Vec<TableMutationSegment>> =
                HashMap::new();

            // Accumulate row events, stopping at the first DDL or truncate boundary.
            while let Some(event) = event_iter.peek() {
                if matches!(event, Event::Relation(_) | Event::Truncate(_)) {
                    break;
                }

                let Some(event) = event_iter.next() else {
                    break;
                };
                match event {
                    Event::Insert(insert) => {
                        let table_id = insert.replicated_table_schema.id();
                        let mutation = TrackedTableMutation::new(
                            insert.event_sequence_key(),
                            TableMutation::Insert(insert.table_row),
                        );
                        push_table_mutation_segment(
                            table_id_to_mutations.entry(table_id).or_default(),
                            insert.replicated_table_schema,
                            mutation,
                        );
                    }
                    Event::Update(update) => {
                        validate_ducklake_replica_identity(
                            &update.replicated_table_schema,
                            "update",
                        )?;
                        let sequence_key = update.event_sequence_key();
                        let table_id = update.replicated_table_schema.id();
                        let replicated_table_schema = update.replicated_table_schema;
                        let table_row = update.updated_table_row;
                        let old_table_row = update.old_table_row;
                        let segments = table_id_to_mutations.entry(table_id).or_default();
                        if let Some(old_row) = old_table_row {
                            let mutation = TrackedTableMutation::new(
                                sequence_key,
                                TableMutation::Update { delete_row: old_row, new_row: table_row },
                            );
                            push_table_mutation_segment(
                                segments,
                                replicated_table_schema,
                                mutation,
                            );
                        } else {
                            match table_row {
                                UpdatedTableRow::Full(table_row) => {
                                    debug!(
                                        "update event has no old row, deleting by replica \
                                         identity from new row"
                                    );
                                    let mutation = TrackedTableMutation::new(
                                        sequence_key,
                                        TableMutation::Replace(table_row),
                                    );
                                    push_table_mutation_segment(
                                        segments,
                                        replicated_table_schema,
                                        mutation,
                                    );
                                }
                                UpdatedTableRow::Partial(partial_row) => {
                                    let key_row = Self::key_row_from_updated_partial_row(
                                        &replicated_table_schema,
                                        &partial_row,
                                    )?;
                                    debug!(
                                        "update event has no old row, building key image from \
                                         partial new row"
                                    );
                                    let mutation = TrackedTableMutation::new(
                                        sequence_key,
                                        TableMutation::Update {
                                            delete_row: OldTableRow::Key(key_row),
                                            new_row: UpdatedTableRow::Partial(partial_row),
                                        },
                                    );
                                    push_table_mutation_segment(
                                        segments,
                                        replicated_table_schema,
                                        mutation,
                                    );
                                }
                            }
                        }
                    }
                    Event::Delete(delete) => {
                        validate_ducklake_replica_identity(
                            &delete.replicated_table_schema,
                            "delete",
                        )?;
                        let sequence_key = delete.event_sequence_key();
                        let Some(old_row) = delete.old_table_row else {
                            return Err(etl_error!(
                                ErrorKind::SourceReplicaIdentityError,
                                "DuckLake delete requires an old row image",
                                format!(
                                    "Table '{}' emitted a delete without an old row despite \
                                     exposing replica-identity columns",
                                    delete.replicated_table_schema.name()
                                )
                            ));
                        };
                        let table_id = delete.replicated_table_schema.id();
                        let mutation =
                            TrackedTableMutation::new(sequence_key, TableMutation::Delete(old_row));
                        push_table_mutation_segment(
                            table_id_to_mutations.entry(table_id).or_default(),
                            delete.replicated_table_schema,
                            mutation,
                        );
                    }
                    event => {
                        debug!(event_type = %event.event_type(), "skipping unsupported event type");
                    }
                }
            }

            if !table_id_to_mutations.is_empty() {
                self.ensure_applied_batches_table_exists().await?;
                self.ensure_streaming_progress_table_exists().await?;
                let mut join_set = JoinSet::new();

                for (_, mutation_segments) in table_id_to_mutations {
                    let destination = self.clone();

                    join_set.spawn(async move {
                        for segment in mutation_segments {
                            let DuckLakeTableReplayCursor {
                                table_name: destination_table_name,
                                replay_epoch,
                                last_sequence_key,
                                table_write_permit: replay_table_write_permit,
                            } = destination
                                .read_table_replay_cursor(&segment.replicated_table_schema)
                                .await?;
                            let pending_mutations = retain_mutations_after_sequence_key(
                                segment.mutations,
                                last_sequence_key,
                            );
                            if pending_mutations.is_empty() {
                                debug!(
                                    table = %destination_table_name,
                                    "ducklake streaming mutation replay skipped, no pending events"
                                );
                                continue;
                            }
                            // Schema reconciliation also acquires the table write slot.
                            drop(replay_table_write_permit);
                            let ready_table_name = destination
                                .ensure_table_ready_for_streaming_schema(
                                    &segment.replicated_table_schema,
                                )
                                .await?;
                            debug_assert_eq!(ready_table_name, destination_table_name);
                            let _table_write_permit = destination
                                .acquire_table_write_slot(&destination_table_name)
                                .await?;
                            let checkpoint_wait_started = tokio::time::Instant::now();
                            let _checkpoint_guard =
                                Arc::clone(&destination.checkpoint_gate).read_owned().await;
                            let checkpoint_wait = checkpoint_wait_started.elapsed();
                            if checkpoint_wait > Duration::from_secs(1) {
                                info!(
                                    table = %destination_table_name,
                                    checkpoint_wait_ms = checkpoint_wait.as_millis() as u64,
                                    "ducklake waited for checkpoint gate before streaming write"
                                );
                            }
                            let is_first_streaming_batch = last_sequence_key.is_none();
                            info!(
                                table = %destination_table_name,
                                pending_mutation_count = pending_mutations.len(),
                                is_first_streaming_batch,
                                "ducklake applying streaming mutations"
                            );

                            let prepared_batches = prepare_mutation_table_batches(
                                destination.embedding.cdc_batch_size,
                                &segment.replicated_table_schema,
                                destination_table_name.clone(),
                                replay_epoch,
                                pending_mutations,
                            )?;
                            apply_table_batches_with_retry(
                                destination.streaming_pool()?,
                                Arc::clone(&destination.blocking_slots),
                                prepared_batches,
                            )
                            .await?;
                            info!(
                                table = %destination_table_name,
                                is_first_streaming_batch,
                                "ducklake applied streaming mutations"
                            );
                        }

                        Ok::<(), etl::error::EtlError>(())
                    });
                }

                finish_table_tasks(&mut join_set).await?;
            }

            // Apply schema changes sequentially before any later row events
            // are encoded with the new replicated schema.
            while let Some(Event::Relation(_)) = event_iter.peek() {
                if let Some(Event::Relation(relation)) = event_iter.next() {
                    self.handle_relation_event(&relation.replicated_table_schema).await?;
                }
            }

            // Collect contiguous truncate events while preserving table-local order.
            let mut truncate_table_ids: HashMap<
                TableId,
                (ReplicatedTableSchema, Vec<TrackedTruncateEvent>),
            > = HashMap::new();
            while let Some(Event::Truncate(_)) = event_iter.peek() {
                if let Some(Event::Truncate(truncate)) = event_iter.next() {
                    let sequence_key = truncate.event_sequence_key();
                    for replicated_table_schema in truncate.truncated_tables {
                        let table_id = replicated_table_schema.id();
                        match truncate_table_ids.entry(table_id) {
                            std::collections::hash_map::Entry::Occupied(mut entry) => {
                                let (schema, truncates) = entry.get_mut();
                                *schema = replicated_table_schema;
                                truncates.push(TrackedTruncateEvent::new(
                                    sequence_key,
                                    truncate.options,
                                ));
                            }
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                entry.insert((
                                    replicated_table_schema,
                                    vec![TrackedTruncateEvent::new(sequence_key, truncate.options)],
                                ));
                            }
                        }
                    }
                }
            }

            if !truncate_table_ids.is_empty() {
                self.ensure_applied_batches_table_exists().await?;
                self.ensure_streaming_progress_table_exists().await?;
                let mut join_set = JoinSet::new();

                for (_, (replicated_table_schema, truncates)) in truncate_table_ids {
                    let destination = self.clone();
                    join_set.spawn(async move {
                        let DuckLakeTableReplayCursor {
                            table_name,
                            replay_epoch,
                            last_sequence_key,
                            table_write_permit: replay_table_write_permit,
                        } = destination.read_table_replay_cursor(&replicated_table_schema).await?;
                        let pending_truncates =
                            retain_truncates_after_sequence_key(truncates, last_sequence_key);
                        if pending_truncates.is_empty() {
                            debug!(
                                table = %table_name,
                                "ducklake streaming truncate replay skipped, no pending events"
                            );
                            return Ok(());
                        }
                        // Schema reconciliation also acquires the table write slot.
                        drop(replay_table_write_permit);
                        let ready_table_name = destination
                            .ensure_table_ready_for_streaming_schema(&replicated_table_schema)
                            .await?;
                        debug_assert_eq!(ready_table_name, table_name);
                        let _table_write_permit =
                            destination.acquire_table_write_slot(&table_name).await?;
                        let _checkpoint_guard =
                            Arc::clone(&destination.checkpoint_gate).read_owned().await;
                        let pool = destination.streaming_pool()?;

                        let prepared_batch = prepare_truncate_table_batch(
                            table_name,
                            replay_epoch,
                            pending_truncates,
                        );
                        apply_table_batch_with_retry(
                            pool,
                            Arc::clone(&destination.blocking_slots),
                            prepared_batch,
                        )
                        .await
                    });
                }

                finish_table_tasks(&mut join_set).await?;
            }
        }

        Ok(())
    }

    /// Creates a DuckLake table and brackets the DDL with destination metadata
    /// state transitions.
    async fn create_table_with_metadata(
        &self,
        table_id: TableId,
        table_name: &DuckLakeTableName,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        validate_ducklake_table_shape(replicated_table_schema)?;
        ensure_unique_table_identity(&self.store, table_id, table_name).await?;
        let metadata = DestinationTableMetadata::new_creating(
            table_name.to_metadata_id()?,
            replicated_table_schema.inner().snapshot_id,
            replicated_table_schema.replication_mask().clone(),
        );
        self.store.store_destination_table_metadata(table_id, metadata.clone()).await?;

        self.issue_create_table_stmt(table_name, replicated_table_schema).await?;
        self.reconcile_missing_replicated_columns(table_name, replicated_table_schema).await?;
        self.reconcile_table_sorting(table_name, replicated_table_schema).await?;

        self.store.store_destination_table_metadata(table_id, metadata.to_applied()).await?;
        self.applied_tables.lock().insert(table_name.clone());

        Ok(())
    }

    /// Issues DuckLake's idempotent `create table if not exists` statement.
    async fn issue_create_table_stmt(
        &self,
        table_name: &DuckLakeTableName,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<()> {
        let column_schemas: Vec<_> = replicated_table_schema
            .destination_column_schemas(DUCKLAKE_COLUMN_NAME_MAPPING)
            .collect();
        let ddl = build_create_table_sql_ducklake(table_name, &column_schemas);
        let table_name = table_name.clone();
        let _checkpoint_guard = self.acquire_mutation_guard().await;

        run_duckdb_blocking(
            self.streaming_pool()?,
            Arc::clone(&self.blocking_slots),
            move |conn| -> EtlResult<()> {
                debug!(table = %table_name, "ducklake create table begin");
                match conn.execute_batch(&ddl) {
                    Ok(()) => {}
                    Err(error) if is_create_table_conflict(&error, table_name.table()) => {}
                    Err(error) => {
                        return Err(etl_error!(
                            ErrorKind::DestinationQueryFailed,
                            "DuckLake create table failed",
                            format_query_error_detail(&ddl),
                            source: error
                        ));
                    }
                }
                debug!(table = %table_name, "ducklake create table finished");
                Ok(())
            },
        )
        .await
    }

    /// Loads an exact table schema snapshot from the store.
    async fn load_exact_table_schema(
        &self,
        table_id: TableId,
        snapshot_id: SnapshotId,
        missing_schema_description: &'static str,
    ) -> EtlResult<Arc<TableSchema>> {
        let table_schema =
            self.store.get_table_schema(&table_id, snapshot_id).await?.ok_or_else(|| {
                etl_error!(
                    ErrorKind::InvalidState,
                    missing_schema_description,
                    format!(
                        "Table {} needs stored schema snapshot {}, but it was not found.",
                        table_id, snapshot_id
                    )
                )
            })?;

        if table_schema.snapshot_id != snapshot_id {
            return Err(etl_error!(
                ErrorKind::InvalidState,
                missing_schema_description,
                format!(
                    "Table {} needs exact schema snapshot {}, but only snapshot {} was found.",
                    table_id, snapshot_id, table_schema.snapshot_id
                )
            ));
        }

        Ok(table_schema)
    }

    /// Loads the exact previous schema for interrupted DDL recovery.
    async fn load_previous_recovery_table_schema(
        &self,
        table_id: TableId,
        previous_snapshot_id: SnapshotId,
    ) -> EtlResult<Arc<TableSchema>> {
        self.load_exact_table_schema(
            table_id,
            previous_snapshot_id,
            "DuckLake schema recovery previous schema not found",
        )
        .await
    }

    /// Resolves the target replicated schema for interrupted DDL recovery.
    async fn target_schema_for_recovery(
        &self,
        table_id: TableId,
        metadata: &DestinationTableMetadata,
        provided_target_schema: Option<&ReplicatedTableSchema>,
    ) -> EtlResult<ReplicatedTableSchema> {
        if let Some(schema) = provided_target_schema {
            ensure_destination_schema_matches_metadata("DuckLake", table_id, metadata, schema)?;
            return Ok(schema.clone());
        }

        let target_table_schema = self
            .load_exact_table_schema(
                table_id,
                metadata.snapshot_id(),
                "DuckLake schema recovery target schema not found",
            )
            .await?;

        Ok(ReplicatedTableSchema::from_mask(
            target_table_schema,
            metadata.replication_mask().clone(),
        ))
    }

    /// Replays interrupted DuckLake DDL and transitions metadata back to
    /// `Applied`.
    async fn recover_pending_metadata(
        &self,
        table_id: TableId,
        table_name: &DuckLakeTableName,
        metadata: DestinationTableMetadata,
        target_schema: Option<&ReplicatedTableSchema>,
    ) -> EtlResult<DestinationTableMetadata> {
        warn!(
            table_id = %table_id,
            table = %table_name,
            "ducklake table has pending metadata, recovering interrupted operation"
        );

        let target_schema =
            self.target_schema_for_recovery(table_id, &metadata, target_schema).await?;
        validate_ducklake_table_shape(&target_schema)?;

        match metadata.table_schema().clone() {
            DestinationTableSchema::Applying {
                previous_snapshot_id,
                previous_replication_mask,
                ..
            } => {
                let previous_table_schema = self
                    .load_previous_recovery_table_schema(table_id, previous_snapshot_id)
                    .await?;
                let old_schema = ReplicatedTableSchema::from_mask(
                    previous_table_schema,
                    previous_replication_mask,
                );
                let plan =
                    old_schema.plan_schema_change(&target_schema, DUCKLAKE_COLUMN_NAME_MAPPING)?;
                ensure_ducklake_schema_plan_recoverable(table_name, &plan)?;
                self.apply_schema_plan(table_name, &plan).await?;
            }
            DestinationTableSchema::Creating { .. } => {
                self.issue_create_table_stmt(table_name, &target_schema).await?;
            }
            DestinationTableSchema::Applied { .. } => {
                return Err(etl_error!(
                    ErrorKind::InvalidState,
                    "DuckLake recovery received applied destination metadata",
                    format!("Table {table_id} does not have an interrupted destination operation")
                ));
            }
        }
        self.reconcile_missing_replicated_columns(table_name, &target_schema).await?;
        self.reconcile_table_sorting(table_name, &target_schema).await?;

        let metadata = metadata.to_applied();
        self.store.store_destination_table_metadata(table_id, metadata.clone()).await?;
        self.cleanup_tombstone_columns_after_applied(table_name, &target_schema).await;
        self.applied_tables.lock().insert(table_name.clone());

        Ok(metadata)
    }

    /// Resolves the ETL-owned destination table and completes pending setup.
    ///
    /// Applied metadata is authoritative and only repopulates process state;
    /// it never causes physical table inspection or repair.
    async fn prepare_table_for_writes(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<DuckLakeTableName> {
        let table_id = replicated_table_schema.id();
        let metadata = self.store.get_destination_table_metadata(table_id).await?;
        let table_name = metadata.as_ref().map_or_else(
            || self.map_new_table_name(replicated_table_schema.name()),
            |metadata| DuckLakeTableName::from_metadata_id(metadata.table_id()),
        )?;

        if let Some(metadata) = &metadata {
            ensure_destination_schema_matches_metadata(
                "DuckLake",
                table_id,
                metadata,
                replicated_table_schema,
            )?;
            if metadata.is_applied() {
                self.applied_tables.lock().insert(table_name.clone());
                return Ok(table_name);
            }
        }

        // Only initial setup and pending-operation recovery may validate or
        // mutate the physical destination schema.
        validate_ducklake_table_shape(replicated_table_schema)?;

        info!(
            table_id = %table_id,
            table = %table_name,
            "ducklake destination table requires setup or recovery"
        );

        let _table_creation_permit =
            Arc::clone(&self.table_creation_slots).acquire_owned().await.map_err(|_| {
                etl_error!(ErrorKind::InvalidState, "DuckLake table creation semaphore closed")
            })?;

        let metadata = self.store.get_destination_table_metadata(table_id).await?;
        let table_name = metadata.as_ref().map_or_else(
            || self.map_new_table_name(replicated_table_schema.name()),
            |metadata| DuckLakeTableName::from_metadata_id(metadata.table_id()),
        )?;

        if let Some(metadata) = &metadata {
            ensure_destination_schema_matches_metadata(
                "DuckLake",
                table_id,
                metadata,
                replicated_table_schema,
            )?;
            if metadata.is_applied() {
                self.applied_tables.lock().insert(table_name.clone());
                return Ok(table_name);
            }
        }

        match metadata {
            None => {
                self.create_table_with_metadata(table_id, &table_name, replicated_table_schema)
                    .await?;
            }
            Some(metadata) if metadata.is_pending() => {
                self.recover_pending_metadata(
                    table_id,
                    &table_name,
                    metadata,
                    Some(replicated_table_schema),
                )
                .await?;
            }
            Some(_) => unreachable!("applied DuckLake metadata should return before setup"),
        }

        Ok(table_name)
    }

    /// Resolves an applied table without validating a replayed event schema.
    ///
    /// This is only for reading the durable streaming watermark before schema
    /// validation. Callers must validate the schema before applying any event
    /// that remains after the already-applied replay prefix is removed.
    async fn applied_table_name_for_replay(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<Option<DuckLakeTableName>> {
        let Some(metadata) =
            self.store.get_destination_table_metadata(replicated_table_schema.id()).await?
        else {
            return Ok(None);
        };
        if !metadata.is_applied() {
            return Ok(None);
        }

        DuckLakeTableName::from_metadata_id(metadata.table_id()).map(Some)
    }

    /// Ensures destination metadata and physical columns can accept a row
    /// schema.
    async fn ensure_table_ready_for_streaming_schema(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<DuckLakeTableName> {
        let table_id = replicated_table_schema.id();
        if let Some(metadata) = self.store.get_destination_table_metadata(table_id).await?
            && metadata.is_applied()
            && (metadata.snapshot_id() != replicated_table_schema.inner().snapshot_id
                || metadata.replication_mask() != replicated_table_schema.replication_mask())
        {
            ensure_relation_schema_transition(
                "DuckLake",
                table_id,
                metadata.snapshot_id(),
                metadata.replication_mask(),
                replicated_table_schema.inner().snapshot_id,
                replicated_table_schema.replication_mask(),
            )?;
            self.handle_relation_event(replicated_table_schema).await?;
        }

        self.prepare_table_for_writes(replicated_table_schema).await
    }

    /// Ensures the ETL-managed replay marker table exists.
    async fn ensure_applied_batches_table_exists(&self) -> EtlResult<()> {
        // Avoid acquiring the global mutation gate or running DuckDB work when
        // initialization has already completed.
        if self.applied_batches_table_created.load(Ordering::Relaxed) {
            return Ok(());
        }

        let _checkpoint_guard = self.acquire_mutation_guard().await;
        ensure_applied_batches_table_exists(
            self.streaming_pool()?,
            Arc::clone(&self.blocking_slots),
            Arc::clone(&self.table_creation_slots),
            Arc::clone(&self.applied_batches_table_created),
        )
        .await
    }

    /// Ensures the ETL-managed streaming progress table exists.
    async fn ensure_streaming_progress_table_exists(&self) -> EtlResult<()> {
        if self.streaming_progress_table_created.load(Ordering::Relaxed) {
            return Ok(());
        }

        let _checkpoint_guard = self.acquire_mutation_guard().await;
        ensure_streaming_progress_table_exists(
            self.streaming_pool()?,
            Arc::clone(&self.blocking_slots),
            Arc::clone(&self.table_creation_slots),
            Arc::clone(&self.streaming_progress_table_created),
        )
        .await
    }

    /// Maps a table only before its durable destination identity exists.
    fn map_new_table_name(&self, source: &TableName) -> EtlResult<DuckLakeTableName> {
        let table_name = match &self.embedding.table_name_mapper {
            Some(map) => map(source),
            None => table_name_to_ducklake_table_name(source),
        }?;
        validate_ducklake_table_name(&table_name)?;
        Ok(table_name)
    }

    /// Returns the stored destination table name or the deterministic default.
    async fn resolve_destination_table_name(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<DuckLakeTableName> {
        let table_id = replicated_table_schema.id();

        if let Some(existing) = self.store.get_destination_table_metadata(table_id).await? {
            return DuckLakeTableName::from_metadata_id(existing.table_id());
        }

        self.map_new_table_name(replicated_table_schema.name())
    }

    /// Serializes table-local truncate and CDC mutation writes.
    async fn acquire_table_write_slot(
        &self,
        table_name: &DuckLakeTableName,
    ) -> EtlResult<OwnedSemaphorePermit> {
        let table_slot = table_write_slot(&self.table_write_slots, table_name);
        match Arc::clone(&table_slot).try_acquire_owned() {
            Ok(permit) => Ok(permit),
            Err(TryAcquireError::NoPermits) => {
                info!(
                    table = %table_name,
                    "ducklake waiting for table write slot"
                );
                let permit = table_slot.acquire_owned().await.map_err(|_| {
                    etl_error!(ErrorKind::InvalidState, "DuckLake table write semaphore closed")
                })?;
                info!(
                    table = %table_name,
                    "ducklake acquired table write slot after wait"
                );
                Ok(permit)
            }
            Err(TryAcquireError::Closed) => {
                Err(etl_error!(ErrorKind::InvalidState, "DuckLake table write semaphore closed"))
            }
        }
    }

    /// Returns the currently installed streaming connection pool.
    fn streaming_pool(&self) -> EtlResult<DuckLakePool> {
        self.pools.streaming()
    }

    /// Returns the currently installed initial-copy connection pool.
    fn copy_pool(&self) -> EtlResult<DuckLakePool> {
        self.pools.copy()
    }

    /// Recreates and atomically installs both connection pools after successful
    /// external maintenance.
    ///
    /// The caller must retain the exclusive external-maintenance pause until
    /// this method returns. If either replacement pool fails to initialize, the
    /// old pools are removed so later replication cannot reuse stale
    /// connections after the pause guard is released.
    pub(super) async fn recreate_pools_after_external_maintenance(&self) -> EtlResult<()> {
        #[cfg(feature = "test-utils")]
        if FAIL_POOL_REFRESH_ONCE.swap(false, std::sync::atomic::Ordering::Relaxed) {
            drop(self.pools.invalidate());
            return Err(etl_error!(
                ErrorKind::DestinationConnectionFailed,
                "Failed to recreate DuckLake connection pools",
                "Injected pool recreation failure"
            ));
        }

        if let Err(error) = self.manager.recreate_shared_instance().await {
            drop(self.pools.invalidate());
            return Err(error);
        }

        let streaming =
            match build_warm_ducklake_pool(self.manager.as_ref().clone(), self.pool_size, "write")
                .await
            {
                Ok(pool) => Arc::new(pool),
                Err(error) => {
                    drop(self.pools.invalidate());
                    return Err(error);
                }
            };
        let copy = match build_warm_ducklake_pool(self.copy_manager.clone(), self.pool_size, "copy")
            .await
        {
            Ok(pool) => Arc::new(pool),
            Err(error) => {
                drop(streaming);
                drop(self.pools.invalidate());
                return Err(error);
            }
        };

        let previous = self.pools.replace(DuckLakePools::new(streaming, copy));
        drop(previous);
        info!(
            pool_size = self.pool_size,
            "ducklake connection pools recreated after external maintenance"
        );

        Ok(())
    }

    /// Acquires shared mutation access so exclusive external maintenance cannot
    /// start in the middle of a foreground write sequence.
    async fn acquire_mutation_guard(&self) -> OwnedRwLockReadGuard<()> {
        Arc::clone(&self.checkpoint_gate).read_owned().await
    }

    /// Reads one table's durable replay cursor while retaining table-local
    /// ordering through the pending-work decision.
    async fn read_table_replay_cursor(
        &self,
        replicated_table_schema: &ReplicatedTableSchema,
    ) -> EtlResult<DuckLakeTableReplayCursor> {
        let table_name = match self.applied_table_name_for_replay(replicated_table_schema).await? {
            Some(table_name) => table_name,
            None => self.ensure_table_ready_for_streaming_schema(replicated_table_schema).await?,
        };
        let table_write_permit = self.acquire_table_write_slot(&table_name).await?;
        let replay_epoch = self.read_table_replay_epoch(&table_name).await?;
        let checkpoint_guard = self.acquire_mutation_guard().await;
        let last_sequence_key = read_table_streaming_progress_sequence_key_blocking(
            self.streaming_pool()?,
            Arc::clone(&self.blocking_slots),
            table_name.clone(),
            replay_epoch.clone(),
        )
        .await?;
        // Schema reconciliation acquires the checkpoint gate itself.
        drop(checkpoint_guard);

        Ok(DuckLakeTableReplayCursor {
            table_name,
            replay_epoch,
            last_sequence_key,
            table_write_permit,
        })
    }

    async fn read_table_replay_epoch(&self, table_name: &DuckLakeTableName) -> EtlResult<String> {
        read_table_replay_epoch(&self.metadata_pg_pool, self.metadata_schema.as_ref(), table_name)
            .await
    }

    /// Starts or resumes the replay epoch transition for a table reset.
    async fn begin_table_replay_epoch_transition(
        &self,
        table_name: &DuckLakeTableName,
    ) -> EtlResult<String> {
        begin_table_replay_epoch_transition(
            &self.metadata_pg_pool,
            self.metadata_schema.as_ref(),
            table_name,
        )
        .await
    }

    /// Promotes a pending replay epoch after its table reset commits.
    async fn complete_table_replay_epoch_transition(
        &self,
        table_name: &DuckLakeTableName,
        pending_replay_epoch: &str,
    ) -> EtlResult<()> {
        complete_table_replay_epoch_transition(
            &self.metadata_pg_pool,
            self.metadata_schema.as_ref(),
            table_name,
            pending_replay_epoch,
        )
        .await
    }

    /// Acquires exclusive DuckLake mutation access for an external maintenance
    /// run. While this guard is held, new foreground writes and in-process
    /// background maintenance operations wait before mutating the catalog.
    pub async fn acquire_external_maintenance_pause(&self) -> DuckLakeExternalMaintenancePause {
        // Drain pinned COPY connections before queueing the global writer. A
        // writer queued directly on `checkpoint_gate` would prevent unrelated
        // readers from entering while a long-lived COPY session drains.
        let copy_session_guard = Arc::clone(&self.copy_session_gate).write_owned().await;
        let checkpoint_guard = Arc::clone(&self.checkpoint_gate).write_owned().await;
        DuckLakeExternalMaintenancePause {
            _copy_session_guard: copy_session_guard,
            _checkpoint_guard: checkpoint_guard,
        }
    }

    /// Samples catalog state and returns which externally coordinated
    /// maintenance operations should be requested now.
    pub(super) async fn sample_external_maintenance_operations(
        &self,
        inline_flush_min_inlined_bytes: u64,
        rewrite_data_files_min_active_data_files: i64,
    ) -> EtlResult<ExternalMaintenanceOperationSample> {
        let table_names = self.list_active_ducklake_tables().await?;
        let inline_sampler = DuckLakePendingInlineSizeSampler::new(
            self.metadata_schema.to_string(),
            self.metadata_pg_pool.clone(),
        );
        let copy_phase_active = self.has_active_table_copy().await?;
        let mut operations = ExternalMaintenanceOperations::default();
        let catalog_metrics = query_catalog_maintenance_metrics(
            &self.metadata_pg_pool,
            self.metadata_schema.as_ref(),
        )
        .await?;

        match expire_snapshots_retention_seconds(self.expire_snapshots_older_than.as_ref()) {
            Some(retention_seconds) => {
                operations.expire_snapshots = catalog_metrics.snapshots_total > 1
                    && catalog_metrics.oldest_snapshot_age_seconds >= retention_seconds;
                debug!(
                    metadata_schema = %self.metadata_schema,
                    expire_snapshots_older_than = %self.expire_snapshots_older_than,
                    retention_seconds,
                    snapshots_total = catalog_metrics.snapshots_total,
                    oldest_snapshot_age_seconds = catalog_metrics.oldest_snapshot_age_seconds,
                    expire_snapshots = operations.expire_snapshots,
                    "ducklake sampled expire snapshots trigger: metadata_schema={}, \
                     expire_snapshots_older_than={}, retention_seconds={}, snapshots_total={}, \
                     oldest_snapshot_age_seconds={}, expire_snapshots={}",
                    self.metadata_schema,
                    self.expire_snapshots_older_than,
                    retention_seconds,
                    catalog_metrics.snapshots_total,
                    catalog_metrics.oldest_snapshot_age_seconds,
                    operations.expire_snapshots
                );
            }
            None => {
                warn!(
                    metadata_schema = %self.metadata_schema,
                    expire_snapshots_older_than = %self.expire_snapshots_older_than,
                    "ducklake could not parse expire_snapshots_older_than for external maintenance \
                     trigger sampling: metadata_schema={}, expire_snapshots_older_than={}",
                    self.metadata_schema,
                    self.expire_snapshots_older_than
                );
            }
        }

        for table_name in table_names {
            if table_name.is_internal_helper() {
                continue;
            }

            if !operations.inline_flush {
                let sizes = inline_sampler.sample_table(&table_name).await?;
                operations.inline_flush = sizes.inlined_bytes >= inline_flush_min_inlined_bytes;
            }

            if !operations.rewrite_data_files && !copy_phase_active {
                let metrics = query_table_storage_metrics(
                    &self.metadata_pg_pool,
                    self.metadata_schema.as_ref(),
                    &table_name,
                )
                .await?;
                operations.rewrite_data_files = should_request_file_maintenance(
                    copy_phase_active,
                    metrics.active_data_files,
                    rewrite_data_files_min_active_data_files,
                );
            }

            if operations.inline_flush && operations.rewrite_data_files {
                break;
            }
        }

        if !copy_phase_active && operations.rewrite_data_files {
            operations.merge_adjacent_files = true;
            operations.cleanup_old_files = true;
        }

        Ok(ExternalMaintenanceOperationSample { operations, copy_phase_active })
    }

    /// Returns whether any table is currently in initial copy.
    async fn has_active_table_copy(&self) -> EtlResult<bool> {
        let table_states = self.store.get_table_states().await?;
        Ok(table_states.values().any(|state| state.as_type() == TableStateType::DataSync))
    }

    /// Lists active DuckLake table names from the metadata catalog.
    async fn list_active_ducklake_tables(&self) -> EtlResult<Vec<DuckLakeTableName>> {
        let sql = format!(
            "SELECT s.schema_name, t.table_name FROM {}.{} AS t JOIN {}.{} AS s ON s.schema_id = \
             t.schema_id WHERE t.end_snapshot IS NULL AND s.end_snapshot IS NULL ORDER BY \
             s.schema_name, t.table_name",
            quote_postgres_identifier(self.metadata_schema.as_ref()),
            quote_postgres_identifier("ducklake_table"),
            quote_postgres_identifier(self.metadata_schema.as_ref()),
            quote_postgres_identifier("ducklake_schema")
        );
        let rows: Vec<(String, String)> = sqlx::query_as(AssertSqlSafe(sql))
            .fetch_all(&self.metadata_pg_pool)
            .await
            .map_err(|source| {
                etl_error!(
                    ErrorKind::DestinationQueryFailed,
                    "DuckLake table list query failed",
                    format!("metadata_schema={}", self.metadata_schema.as_ref()),
                    source: source
                )
            })?;
        Ok(rows
            .into_iter()
            .map(|(schema_name, table_name)| DuckLakeTableName::new(schema_name, table_name))
            .collect())
    }

    /// Runs one DuckDB operation on Tokio's blocking pool after acquiring a
    /// permit that matches the configured DuckDB concurrency limit.
    async fn run_duckdb_blocking<R, F>(&self, operation: F) -> EtlResult<R>
    where
        R: Send + 'static,
        F: FnOnce(&duckdb::Connection) -> EtlResult<R> + Send + 'static,
    {
        run_duckdb_blocking(self.streaming_pool()?, Arc::clone(&self.blocking_slots), operation)
            .await
    }

    /// Stops the background DuckLake metrics sampler.
    async fn shutdown_metrics_sampler(&self) -> EtlResult<()> {
        if let Some(metrics_sampler) = &*self.metrics_sampler {
            let _ = metrics_sampler.shutdown_tx.send(());
            let handle = metrics_sampler.handle.lock().take();
            if let Some(handle) = handle {
                handle.abort();
                if let Err(err) = handle.await
                    && !err.is_cancelled()
                {
                    return Err(etl_error!(
                        ErrorKind::ApplyWorkerPanic,
                        "DuckLake metrics sampler task panicked"
                    ));
                }
            }
        }

        Ok(())
    }
    /// Returns how many COPY-pool DuckDB connections have been initialized for
    /// tests.
    #[cfg(feature = "test-utils")]
    pub fn copy_connection_open_count_for_tests(&self) -> usize {
        self.copy_manager.open_count_for_tests()
    }

    /// Returns how many streaming-pool DuckDB connections have been initialized
    /// for tests.
    #[cfg(feature = "test-utils")]
    pub fn streaming_connection_open_count_for_tests(&self) -> usize {
        self.manager.open_count_for_tests()
    }
}

/// Reads the persisted streaming replay watermark for one table on DuckDB's
/// blocking pool.
async fn read_table_streaming_progress_sequence_key_blocking(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    table_name: DuckLakeTableName,
    replay_epoch: String,
) -> EtlResult<Option<EventSequenceKey>> {
    run_duckdb_blocking(pool, blocking_slots, move |conn| {
        read_table_streaming_progress_sequence_key(conn, &table_name, &replay_epoch)
    })
    .await
}

#[cfg(feature = "test-utils")]
struct PausedStreamingWriteHook {
    reached_tx: oneshot::Sender<()>,
    resume_rx: oneshot::Receiver<()>,
}

#[cfg(feature = "test-utils")]
type CopyAppendResume = Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

#[cfg(feature = "test-utils")]
struct PausedCopyAppendHook {
    reached_tx: oneshot::Sender<()>,
    resume: CopyAppendResume,
}

#[cfg(feature = "test-utils")]
static PAUSED_STREAMING_WRITE_HOOK: std::sync::LazyLock<Mutex<Option<PausedStreamingWriteHook>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));
#[cfg(feature = "test-utils")]
static PAUSED_STREAMING_WRITE_RESUME_TX: std::sync::LazyLock<Mutex<Option<oneshot::Sender<()>>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));
#[cfg(feature = "test-utils")]
static PAUSED_COPY_APPEND_HOOK: std::sync::LazyLock<Mutex<Option<PausedCopyAppendHook>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));
#[cfg(feature = "test-utils")]
static PAUSED_COPY_APPEND_RESUME: std::sync::LazyLock<Mutex<Option<CopyAppendResume>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));
#[cfg(feature = "test-utils")]
static FAIL_POOL_REFRESH_ONCE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "test-utils")]
static FAIL_DROP_TABLE_FOR_COPY_ONCE: AtomicBool = AtomicBool::new(false);

/// Injects one pool-refresh failure for tests.
#[cfg(feature = "test-utils")]
pub fn arm_fail_pool_refresh_once_for_tests() {
    FAIL_POOL_REFRESH_ONCE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Returns whether the one-shot pool-refresh failure remains armed for tests.
#[cfg(feature = "test-utils")]
pub fn pool_refresh_failure_armed_for_tests() -> bool {
    FAIL_POOL_REFRESH_ONCE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clears the one-shot pool-refresh failure for tests.
#[cfg(feature = "test-utils")]
pub fn reset_pool_refresh_failure_for_tests() {
    FAIL_POOL_REFRESH_ONCE.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Injects one drop-for-copy failure after the old session is removed.
#[cfg(feature = "test-utils")]
pub fn arm_fail_drop_table_for_copy_once_for_tests() {
    FAIL_DROP_TABLE_FOR_COPY_ONCE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Clears the one-shot drop-for-copy failure for tests.
#[cfg(feature = "test-utils")]
pub fn reset_drop_table_for_copy_failure_for_tests() {
    FAIL_DROP_TABLE_FOR_COPY_ONCE.store(false, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "test-utils")]
fn maybe_fail_drop_table_for_copy_for_tests() -> EtlResult<()> {
    if FAIL_DROP_TABLE_FOR_COPY_ONCE.swap(false, std::sync::atomic::Ordering::Relaxed) {
        return Err(etl_error!(
            ErrorKind::DestinationQueryFailed,
            "DuckLake test hook injected drop-for-copy failure"
        ));
    }

    Ok(())
}

/// Arms a one-shot hook that pauses the next streaming write before DuckLake
/// starts applying it.
#[cfg(feature = "test-utils")]
pub fn arm_pause_next_streaming_write_for_tests() -> oneshot::Receiver<()> {
    let (reached_tx, reached_rx) = oneshot::channel();
    let (resume_tx, resume_rx) = oneshot::channel();
    *PAUSED_STREAMING_WRITE_HOOK.lock() = Some(PausedStreamingWriteHook { reached_tx, resume_rx });
    *PAUSED_STREAMING_WRITE_RESUME_TX.lock() = Some(resume_tx);
    reached_rx
}

/// Releases the paused streaming-write test hook, if one is armed.
#[cfg(feature = "test-utils")]
pub fn release_paused_streaming_write_for_tests() {
    if let Some(resume_tx) = PAUSED_STREAMING_WRITE_RESUME_TX.lock().take() {
        let _ = resume_tx.send(());
    }
}

/// Clears the paused streaming-write hook without waiting for it to be
/// released.
#[cfg(feature = "test-utils")]
pub fn reset_paused_streaming_write_for_tests() {
    PAUSED_STREAMING_WRITE_HOOK.lock().take();
    PAUSED_STREAMING_WRITE_RESUME_TX.lock().take();
}

/// Arms a one-shot hook that pauses the next buffered COPY append inside its
/// blocking task.
#[cfg(feature = "test-utils")]
pub fn arm_pause_next_copy_append_for_tests() -> oneshot::Receiver<()> {
    let (reached_tx, reached_rx) = oneshot::channel();
    let resume = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    *PAUSED_COPY_APPEND_HOOK.lock() =
        Some(PausedCopyAppendHook { reached_tx, resume: Arc::clone(&resume) });
    *PAUSED_COPY_APPEND_RESUME.lock() = Some(resume);
    reached_rx
}

/// Releases the paused buffered COPY append, if one is armed.
#[cfg(feature = "test-utils")]
pub fn release_paused_copy_append_for_tests() {
    let Some(resume) = PAUSED_COPY_APPEND_RESUME.lock().take() else {
        return;
    };
    let (resumed, resume_ready) = &*resume;
    let mut resumed = resumed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    *resumed = true;
    resume_ready.notify_all();
}

/// Clears and releases the paused buffered COPY append hook.
#[cfg(feature = "test-utils")]
pub fn reset_paused_copy_append_for_tests() {
    PAUSED_COPY_APPEND_HOOK.lock().take();
    release_paused_copy_append_for_tests();
}

#[cfg(feature = "test-utils")]
async fn wait_if_streaming_write_paused_for_tests() {
    let Some(PausedStreamingWriteHook { reached_tx, resume_rx }) =
        PAUSED_STREAMING_WRITE_HOOK.lock().take()
    else {
        return;
    };

    let _ = reached_tx.send(());
    let _ = resume_rx.await;
}

#[cfg(feature = "test-utils")]
fn wait_if_copy_append_paused_for_tests() {
    let Some(PausedCopyAppendHook { reached_tx, resume }) = PAUSED_COPY_APPEND_HOOK.lock().take()
    else {
        return;
    };

    let _ = reached_tx.send(());
    let (resumed, resume_ready) = &*resume;
    let mut resumed = resumed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    while !*resumed {
        resumed = resume_ready.wait(resumed).unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

/// Converts a Postgres [`TableName`] to the matching DuckLake schema/table.
///
/// Source schemas are preserved in DuckLake. For example, `public.my_table`
/// becomes the DuckLake table `lake.public.my_table`.
pub fn table_name_to_ducklake_table_name(table_name: &TableName) -> EtlResult<DuckLakeTableName> {
    Ok(DuckLakeTableName::from_source(table_name))
}

impl<S: DestinationStore> DuckLakeDestination<S> {
    /// Runs host maintenance on the writer instance while mutations are paused.
    ///
    /// The operation runs on a blocking thread under the normal query watchdog.
    /// Its pause guard survives cancellation until the native operation exits.
    /// Finish all transactions before returning, and do not retain or clone the
    /// connection outside the operation. Scheduling and table selection belong
    /// to the caller; this method does not retry a partially completed
    /// operation.
    pub async fn run_maintenance<R, F>(&self, timeout: Duration, operation: F) -> EtlResult<R>
    where
        R: Send + 'static,
        F: FnOnce(&duckdb::Connection) -> EtlResult<R> + Send + 'static,
    {
        let pause = self.acquire_external_maintenance_pause().await;
        crate::ducklake::client::run_duckdb_blocking_with_timeout(
            self.streaming_pool()?,
            Arc::clone(&self.blocking_slots),
            timeout,
            move |connection| {
                let _pause = pause;
                operation(connection)
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
        time::Instant,
    };

    use duckdb::{Config, Connection};
    use etl::{
        config::{PgConnectionConfig, TcpKeepaliveConfig},
        data::{Cell, PartialTableRow, TableRow},
        destination::{TableCopyAttemptId, TableCopyBatchId},
        schema::{
            ColumnMetadataChange, ColumnSchema, IdentityMask, ReplicationMask, SchemaDiff,
            TableSchema, Type as PgType,
        },
        store::{MemoryStore, SchemaStore, StateStore},
    };
    use etl_maintenance::ducklake::flush_table_inlined_data;
    use etl_postgres::{test_utils::local_tls_config_from_env, tokio::test_utils::PgDatabase};
    use pg_escape::{quote_identifier, quote_literal};
    use tempfile::TempDir;
    use tokio_postgres::Client;
    use url::Url;
    use uuid::Uuid;

    use super::*;
    use crate::ducklake::{
        LAKE_CATALOG,
        config::catalog_conninfo_from_url,
        metrics::{query_catalog_maintenance_metrics, query_table_storage_metrics},
    };

    const POSTGRES_SCANNER_EXTENSION_FILE: &str = "postgres_scanner.duckdb_extension";

    /// Returns a unique table-copy batch ID for direct unit-test calls.
    fn test_table_copy_batch_id() -> TableCopyBatchId {
        static NEXT_BATCH_ID: AtomicU64 = AtomicU64::new(0);

        let batch_id = NEXT_BATCH_ID.fetch_add(1, Ordering::Relaxed);
        TableCopyBatchId::new(TableCopyAttemptId::from_u128(1), batch_id)
    }

    #[tokio::test]
    async fn failing_table_does_not_abort_other_accepted_work() {
        let completed = Arc::new(AtomicBool::new(false));
        let other_completed = Arc::clone(&completed);
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(async {
            Err(etl_error!(ErrorKind::DestinationQueryFailed, "First write failed"))
        });
        tasks.spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            other_completed.store(true, Ordering::SeqCst);
            Err(etl_error!(ErrorKind::DestinationQueryFailed, "Second write failed"))
        });
        assert!(finish_table_tasks(&mut tasks).await.is_err());
        assert!(completed.load(Ordering::SeqCst));
        assert!(tasks.is_empty());
    }

    #[test]
    fn copy_data_inlining_errors_are_retryable_except_during_shutdown() {
        let table_name = DuckLakeTableName::new("public", "users");
        let query_error =
            etl_error!(ErrorKind::DestinationQueryFailed, "Synthetic DuckLake inlining failure");

        let error = classify_copy_data_inlining_error(&table_name, 1_000_000, query_error);
        assert_eq!(error.kind(), ErrorKind::DestinationAtomicBatchRetryable);

        let shutdown_error = crate::ducklake::client::ducklake_shutdown_requested_error();
        let error = classify_copy_data_inlining_error(&table_name, 1_000_000, shutdown_error);
        assert!(is_ducklake_shutdown_requested_error(&error));
    }

    /// Keeps compaction from competing with batches that are creating Parquet
    /// files during an initial copy.
    #[test]
    fn file_maintenance_is_deferred_during_copy() {
        assert!(!should_request_file_maintenance(true, 41, 40));
        assert!(!should_request_file_maintenance(false, 40, 40));
        assert!(should_request_file_maintenance(false, 41, 40));
    }

    #[test]
    fn invalidated_connection_pools_return_destination_connection_error() {
        let pools = DuckLakePoolHandle { current: ParkingLotRwLock::new(None) };

        for result in [pools.streaming(), pools.copy()] {
            let Err(error) = result else {
                panic!("invalidated DuckLake pool handle should not return a pool");
            };
            assert_eq!(error.kind(), ErrorKind::DestinationConnectionFailed);
            assert_eq!(error.description(), Some("DuckLake connection pools are unavailable"));
        }
    }

    #[test]
    fn expire_snapshots_retention_seconds_uses_humantime_duration_syntax() {
        assert_eq!(expire_snapshots_retention_seconds("7 days"), Some(604_800));
        assert_eq!(expire_snapshots_retention_seconds("2h 30min"), Some(9_000));
        assert_eq!(expire_snapshots_retention_seconds("bad interval"), None);
    }

    fn make_schema(table_id: u32, schema: &str, table: &str) -> TableSchema {
        TableSchema::new(
            TableId::new(table_id),
            TableName::new(schema.to_owned(), table.to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true),
            ],
        )
    }

    fn ducklake_table_name() -> DuckLakeTableName {
        DuckLakeTableName::new("public", "users")
    }

    #[test]
    fn primary_key_sorting_uses_key_definition_order() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("tenant_id".to_owned(), PgType::INT4, -1, 1, false)
                    .with_primary_key(2),
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 2, false).with_primary_key(1),
            ],
        ));
        let replicated_table_schema = ReplicatedTableSchema::all(table_schema);
        let table_name = ducklake_table_name();
        let sorting = index_table_sorting_config(DuckLakeTableSortingConfig {
            tables: vec![etl_config::shared::DuckLakeTableSortConfig {
                schema: "public".to_owned(),
                table: "users".to_owned(),
                sort_by: DuckLakeSortBy::PrimaryKey,
            }],
        })
        .unwrap();

        let columns = resolve_table_sort_columns(&sorting, &table_name, &replicated_table_schema)
            .unwrap()
            .unwrap();

        assert_eq!(
            columns.iter().map(|column| column.name.as_str()).collect::<Vec<_>>(),
            vec!["id", "tenant_id"]
        );
        assert!(columns.iter().all(|column| {
            column.direction == DuckLakeSortDirection::Asc && column.nulls.is_none()
        }));
    }

    #[test]
    fn explicit_sorting_rejects_unreplicated_columns() {
        let table_schema = Arc::new(make_schema(1, "public", "users"));
        let replicated_table_schema =
            ReplicatedTableSchema::from_mask(table_schema, ReplicationMask::from_bytes(vec![1, 0]));
        let table_name = ducklake_table_name();
        let sorting = index_table_sorting_config(DuckLakeTableSortingConfig {
            tables: vec![etl_config::shared::DuckLakeTableSortConfig {
                schema: "public".to_owned(),
                table: "users".to_owned(),
                sort_by: DuckLakeSortBy::Columns {
                    columns: vec![DuckLakeSortColumn {
                        name: "name".to_owned(),
                        direction: DuckLakeSortDirection::Desc,
                        nulls: Some(DuckLakeSortNulls::First),
                    }],
                },
            }],
        })
        .unwrap();

        let error = resolve_table_sort_columns(&sorting, &table_name, &replicated_table_schema)
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ConfigError);
        assert!(error.to_string().contains("not replicated"));
    }

    #[test]
    fn active_sort_order_comparison_uses_ducklake_defaults() {
        let active = vec![ActiveDuckLakeSortColumn {
            expression: r#""id""#.to_owned(),
            direction: "ASC".to_owned(),
            null_order: "NULLS_LAST".to_owned(),
        }];
        let desired = vec![DuckLakeSortColumn {
            name: "id".to_owned(),
            direction: DuckLakeSortDirection::Asc,
            nulls: None,
        }];

        assert!(active_sort_order_matches(&active, &desired));
    }

    #[tokio::test]
    async fn table_sorting_requires_external_maintenance() {
        let sorting = DuckLakeTableSortingConfig {
            tables: vec![etl_config::shared::DuckLakeTableSortConfig {
                schema: "public".to_owned(),
                table: "users".to_owned(),
                sort_by: DuckLakeSortBy::PrimaryKey,
            }],
        };

        let error = DuckLakeDestination::new_with_table_sorting_and_external_maintenance(
            Url::parse("postgres://localhost/ducklake").unwrap(),
            Url::parse("file:///tmp/ducklake").unwrap(),
            1,
            None,
            None,
            None,
            None,
            sorting,
            DuckLakeExternalMaintenanceConfig::disabled(),
            MemoryStore::new(),
        )
        .await
        .err()
        .unwrap();

        assert_eq!(error.kind(), ErrorKind::ConfigError);
    }

    fn make_alternative_identity_schema() -> ReplicatedTableSchema {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(2),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("email".to_owned(), PgType::TEXT, -1, 2, false),
                ColumnSchema::new("payload".to_owned(), PgType::TEXT, -1, 3, true),
            ],
        ));

        ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![0, 1, 0]),
        )
    }

    fn make_missing_identity_schema() -> ReplicatedTableSchema {
        let table_schema = Arc::new(make_schema(3, "public", "users"));

        ReplicatedTableSchema::from_masks(
            Arc::clone(&table_schema),
            ReplicationMask::all(&table_schema),
            IdentityMask::from_bytes(vec![0, 0]),
        )
    }

    fn rename_change(
        old_name: &str,
        new_name: &str,
        ordinal_position: i32,
    ) -> ColumnMetadataChange {
        let before_column_schema =
            ColumnSchema::new(old_name.to_owned(), PgType::TEXT, -1, ordinal_position, true);
        let after_column_schema =
            ColumnSchema::new(new_name.to_owned(), PgType::TEXT, -1, ordinal_position, true);

        ColumnMetadataChange::between(&before_column_schema, &after_column_schema).unwrap()
    }

    fn default_change(
        name: &str,
        ordinal_position: i32,
        old_expression: Option<&str>,
        new_expression: Option<&str>,
    ) -> ColumnMetadataChange {
        let before_column_schema =
            ColumnSchema::new(name.to_owned(), PgType::TEXT, -1, ordinal_position, true)
                .with_default_expression_option(old_expression.map(ToOwned::to_owned));
        let after_column_schema =
            ColumnSchema::new(name.to_owned(), PgType::TEXT, -1, ordinal_position, true)
                .with_default_expression_option(new_expression.map(ToOwned::to_owned));

        ColumnMetadataChange::between(&before_column_schema, &after_column_schema).unwrap()
    }

    fn shared_schema_plan<I, S, J, T>(
        diff: SchemaDiff,
        current_column_names: I,
        target_column_names: J,
    ) -> SchemaPlan
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        J: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let current_column_names = current_column_names.into_iter().map(Into::into).collect();
        let target_column_names = target_column_names.into_iter().map(Into::into).collect();

        diff.plan_for_column_names(
            current_column_names,
            target_column_names,
            DUCKLAKE_COLUMN_NAME_MAPPING,
        )
        .unwrap()
    }

    fn plan_ducklake_schema_ddl(
        table_name: &DuckLakeTableName,
        column_names: Vec<String>,
        plan: &SchemaPlan,
    ) -> EtlResult<DuckLakeSchemaDdlPlan> {
        plan_ducklake_schema_ddl_with_nullability(table_name, column_names, HashSet::new(), plan)
    }

    #[test]
    fn ducklake_nullability_relaxation_is_idempotent_against_physical_state() {
        let before = ColumnSchema::new("value".to_owned(), PgType::TEXT, -1, 1, false);
        let after = ColumnSchema::new("value".to_owned(), PgType::TEXT, -1, 1, true);
        let change = ColumnMetadataChange::between(&before, &after).unwrap();
        let plan = shared_schema_plan(
            SchemaDiff::new(Vec::new(), Vec::new(), vec![change]),
            ["value"],
            ["value"],
        );

        let old_state = plan_ducklake_schema_ddl_with_nullability(
            &ducklake_table_name(),
            vec!["value".to_owned()],
            HashSet::new(),
            &plan,
        )
        .unwrap();
        let target_state = plan_ducklake_schema_ddl_with_nullability(
            &ducklake_table_name(),
            vec!["value".to_owned()],
            HashSet::from(["value".to_owned()]),
            &plan,
        )
        .unwrap();

        assert_eq!(
            old_state.statements,
            vec![DuckLakeSchemaDdlStatement {
                sql: r#"alter table "lake"."public"."users" alter column "value" drop not null"#
                    .to_owned(),
                error_description: "DuckLake alter table drop not null failed",
            }]
        );
        assert!(target_state.statements.is_empty());
    }

    #[test]
    fn key_row_from_updated_partial_row_uses_alternative_identity_columns() {
        let replicated_table_schema = make_alternative_identity_schema();
        let partial_row = PartialTableRow::new(
            3,
            TableRow::new(vec![Cell::I32(1), Cell::String("alice@example.com".to_owned())]),
            vec![2],
        );

        let key_row = DuckLakeDestination::<MemoryStore>::key_row_from_updated_partial_row(
            &replicated_table_schema,
            &partial_row,
        )
        .unwrap();

        assert_eq!(key_row, TableRow::new(vec![Cell::String("alice@example.com".to_owned())]));
    }

    #[test]
    fn key_row_from_updated_partial_row_rejects_missing_replica_identity() {
        let replicated_table_schema = make_missing_identity_schema();
        let partial_row = PartialTableRow::new(
            2,
            TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())]),
            vec![],
        );

        let error = DuckLakeDestination::<MemoryStore>::key_row_from_updated_partial_row(
            &replicated_table_schema,
            &partial_row,
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::SourceReplicaIdentityError);
        assert_eq!(error.description(), Some("DuckLake update requires a replica identity"));
    }

    #[test]
    fn plan_ducklake_schema_ddl_renames_before_adding_reused_source_name() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                vec![ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true)],
                Vec::new(),
                vec![rename_change("name", "full_name", 2)],
            ),
            ["id", "name"],
            ["id", "full_name", "name"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "name".to_owned()],
            &shared_plan,
        )
        .unwrap();
        let statement_sql: Vec<_> =
            plan.statements.iter().map(|statement| statement.sql.clone()).collect();

        assert_eq!(
            statement_sql,
            vec![
                r#"alter table "lake"."public"."users" rename column "name" to "full_name""#,
                r#"alter table "lake"."public"."users" add column "name" varchar"#,
            ]
        );
        assert_eq!(plan.column_names, vec!["id", "full_name", "name"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_applies_rename_cycle_for_each_physical_order() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![rename_change("a", "b", 1), rename_change("b", "a", 2)],
            ),
            ["a", "b"],
            ["b", "a"],
        );

        for (physical_column_names, expected_column_names) in [
            (vec!["a".to_owned(), "b".to_owned()], vec!["b", "a"]),
            (vec!["b".to_owned(), "a".to_owned()], vec!["a", "b"]),
        ] {
            let plan = plan_ducklake_schema_ddl(
                &ducklake_table_name(),
                physical_column_names,
                &shared_plan,
            )
            .unwrap();
            let statement_sql: Vec<_> =
                plan.statements.iter().map(|statement| statement.sql.clone()).collect();

            assert_eq!(
                statement_sql,
                vec![
                    r#"alter table "lake"."public"."users" rename column "a" to "supabase_etl_ddl_tmp_column_1_0""#,
                    r#"alter table "lake"."public"."users" rename column "b" to "a""#,
                    r#"alter table "lake"."public"."users" rename column "supabase_etl_ddl_tmp_column_1_0" to "b""#,
                ]
            );
            assert_eq!(plan.column_names, expected_column_names);
        }
    }

    #[test]
    fn plan_ducklake_schema_ddl_skips_planned_type_change() {
        let before_column_schema = ColumnSchema::new("value".to_owned(), PgType::INT4, -1, 1, true);
        let after_column_schema = ColumnSchema::new("value".to_owned(), PgType::INT8, -1, 1, true);
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![
                    ColumnMetadataChange::between(&before_column_schema, &after_column_schema)
                        .unwrap(),
                ],
            ),
            ["value"],
            ["value"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["value".to_owned()],
            &shared_plan,
        )
        .unwrap();

        assert!(plan.statements.is_empty());
        assert_eq!(plan.column_names, ["value"]);
    }

    #[test]
    fn ducklake_schema_plan_recovery_rejects_rename_cycle_without_durable_marker() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![rename_change("a", "b", 1), rename_change("b", "a", 2)],
            ),
            ["a", "b"],
            ["b", "a"],
        );

        let error = ensure_ducklake_schema_plan_recoverable(&ducklake_table_name(), &shared_plan)
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidState);
    }

    #[test]
    fn plan_ducklake_schema_ddl_adds_column_with_supported_default() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                vec![
                    ColumnSchema::new("status".to_owned(), PgType::TEXT, -1, 3, true)
                        .with_default_expression("'pending'::text".to_owned()),
                ],
                Vec::new(),
                Vec::new(),
            ),
            ["id", "name"],
            ["id", "name", "status"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "name".to_owned()],
            &shared_plan,
        )
        .unwrap();
        let statement_sql: Vec<_> =
            plan.statements.iter().map(|statement| statement.sql.clone()).collect();

        assert_eq!(
            statement_sql,
            vec![
                r#"alter table "lake"."public"."users" add column "status" varchar default 'pending'"#
            ]
        );
        assert_eq!(plan.column_names, vec!["id", "name", "status"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_adds_publication_column_nullable_without_default() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("status".to_owned(), PgType::TEXT, -1, 2, false)
                    .with_default_expression("'pending'::text".to_owned()),
            ],
        ));
        let before = ReplicatedTableSchema::from_mask(
            Arc::clone(&table_schema),
            ReplicationMask::from_bytes(vec![1, 0]),
        );
        let after =
            ReplicatedTableSchema::from_mask(table_schema, ReplicationMask::from_bytes(vec![1, 1]));
        let shared_plan = before.plan_schema_change(&after, DUCKLAKE_COLUMN_NAME_MAPPING).unwrap();

        let plan =
            plan_ducklake_schema_ddl(&ducklake_table_name(), vec!["id".to_owned()], &shared_plan)
                .unwrap();

        assert_eq!(
            plan.statements.iter().map(|statement| statement.sql.as_str()).collect::<Vec<_>>(),
            [r#"alter table "lake"."public"."users" add column "status" varchar"#]
        );
        assert_eq!(plan.column_names, vec!["id", "status"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_drops_previous_default_without_rechecking_support() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![default_change("status", 3, Some("array['unsupported']::text[]"), None)],
            ),
            ["id", "name", "status"],
            ["id", "name", "status"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "name".to_owned(), "status".to_owned()],
            &shared_plan,
        )
        .unwrap();

        assert_eq!(
            plan.statements.iter().map(|statement| statement.sql.as_str()).collect::<Vec<_>>(),
            [r#"alter table "lake"."public"."users" alter column "status" drop default"#]
        );
        assert_eq!(plan.column_names, vec!["id", "name", "status"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_drops_before_setting_replacement_default() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![default_change(
                    "status",
                    3,
                    Some("array['unsupported']::text[]"),
                    Some("'queued'::text"),
                )],
            ),
            ["id", "name", "status"],
            ["id", "name", "status"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "name".to_owned(), "status".to_owned()],
            &shared_plan,
        )
        .unwrap();

        assert_eq!(
            plan.statements.iter().map(|statement| statement.sql.as_str()).collect::<Vec<_>>(),
            [
                r#"alter table "lake"."public"."users" alter column "status" drop default"#,
                r#"alter table "lake"."public"."users" alter column "status" set default 'queued'"#,
            ]
        );
        assert_eq!(plan.column_names, vec!["id", "name", "status"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_skips_replayed_rename_with_reused_source_name() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                vec![ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true)],
                Vec::new(),
                vec![rename_change("name", "full_name", 2)],
            ),
            ["id", "name"],
            ["id", "full_name", "name"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "full_name".to_owned(), "name".to_owned()],
            &shared_plan,
        )
        .unwrap();

        assert!(plan.statements.is_empty());
        assert_eq!(plan.column_names, vec!["id", "full_name", "name"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_skips_replayed_rename_with_source_name_as_rename_target() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![rename_change("name", "full_name", 2), rename_change("email", "name", 3)],
            ),
            ["id", "name", "email"],
            ["id", "full_name", "name"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "full_name".to_owned(), "name".to_owned()],
            &shared_plan,
        )
        .unwrap();

        assert!(plan.statements.is_empty());
        assert_eq!(plan.column_names, vec!["id", "full_name", "name"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_drops_stale_rename_source_when_target_exists() {
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                Vec::new(),
                vec![rename_change("ddl_col_4_1", "ddl_col_4_0", 4)],
            ),
            ["id", "ddl_col_4_1"],
            ["id", "ddl_col_4_0"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "ddl_col_4_1".to_owned(), "ddl_col_4_0".to_owned()],
            &shared_plan,
        )
        .unwrap();
        let statement_sql: Vec<_> =
            plan.statements.iter().map(|statement| statement.sql.as_str()).collect();

        assert_eq!(
            statement_sql,
            vec![r#"alter table "lake"."public"."users" drop column "ddl_col_4_1""#]
        );
        assert_eq!(plan.column_names, vec!["id", "ddl_col_4_0"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_tombstones_dropped_column_when_name_is_reused_by_rename() {
        let dropped_column = ColumnSchema::new("status".to_owned(), PgType::TEXT, -1, 3, true);
        let tombstone_name = dropped_column_tombstone_name_ducklake(&dropped_column);
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                Vec::new(),
                vec![dropped_column],
                vec![rename_change("name", "status", 2)],
            ),
            ["id", "name", "status"],
            ["id", "status"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "name".to_owned(), "status".to_owned()],
            &shared_plan,
        )
        .unwrap();
        let statement_sql: Vec<_> =
            plan.statements.iter().map(|statement| statement.sql.clone()).collect();

        assert_eq!(
            statement_sql,
            vec![
                format!(
                    r#"alter table "lake"."public"."users" rename column "status" to "{tombstone_name}""#
                ),
                r#"alter table "lake"."public"."users" rename column "name" to "status""#
                    .to_owned(),
            ]
        );
        assert_eq!(plan.column_names, vec!["id", "status", tombstone_name.as_str()]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_tombstones_dropped_column_when_name_is_reused_by_add() {
        let dropped_column = ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true);
        let tombstone_name = dropped_column_tombstone_name_ducklake(&dropped_column);
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                vec![ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true)],
                vec![dropped_column],
                Vec::new(),
            ),
            ["id", "name"],
            ["id", "name"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), "name".to_owned()],
            &shared_plan,
        )
        .unwrap();
        let statement_sql: Vec<_> =
            plan.statements.iter().map(|statement| statement.sql.as_str()).collect();

        assert_eq!(
            statement_sql,
            vec![
                format!(
                    r#"alter table "lake"."public"."users" rename column "name" to "{tombstone_name}""#
                ),
                r#"alter table "lake"."public"."users" add column "name" varchar"#.to_owned(),
            ]
        );
        assert_eq!(plan.column_names, vec!["id", tombstone_name.as_str(), "name"]);
    }

    #[test]
    fn plan_ducklake_schema_ddl_skips_replayed_reused_dropped_column_name() {
        let dropped_column = ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true);
        let tombstone_name = dropped_column_tombstone_name_ducklake(&dropped_column);
        let shared_plan = shared_schema_plan(
            SchemaDiff::new(
                vec![ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 3, true)],
                vec![dropped_column],
                Vec::new(),
            ),
            ["id", "name"],
            ["id", "name"],
        );

        let plan = plan_ducklake_schema_ddl(
            &ducklake_table_name(),
            vec!["id".to_owned(), tombstone_name.clone(), "name".to_owned()],
            &shared_plan,
        )
        .unwrap();

        assert!(plan.statements.is_empty());
        assert_eq!(plan.column_names, vec!["id", tombstone_name.as_str(), "name"]);
    }

    #[test]
    fn tombstone_columns_to_cleanup_keeps_active_prefixed_columns() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new(
                    "supabase_etl_ducklake_dropped_business".to_owned(),
                    PgType::TEXT,
                    -1,
                    2,
                    true,
                ),
            ],
        ));
        let replicated_table_schema =
            ReplicatedTableSchema::from_mask(table_schema, ReplicationMask::from_bytes(vec![1, 1]));
        validate_ducklake_tombstone_namespace(&replicated_table_schema).unwrap();
        let column_names = vec![
            "id".to_owned(),
            "supabase_etl_ducklake_dropped_business".to_owned(),
            "supabase_etl_ducklake_dropped_3_0000000000abcdef".to_owned(),
        ];

        let columns_to_cleanup =
            tombstone_columns_to_cleanup_ducklake(&column_names, &replicated_table_schema);

        assert_eq!(columns_to_cleanup, vec!["supabase_etl_ducklake_dropped_3_0000000000abcdef"]);
    }

    #[test]
    fn tombstone_name_recognition_requires_the_canonical_generated_shape() {
        assert!(is_ducklake_tombstone_column_name(
            "supabase_etl_ducklake_dropped_3_0000000000abcdef"
        ));

        for column_name in [
            "supabase_etl_ducklake_dropped_business",
            "supabase_etl_ducklake_dropped_03_0000000000abcdef",
            "supabase_etl_ducklake_dropped_3_0000000000ABCDEF",
            "supabase_etl_ducklake_dropped_3_abcdef",
        ] {
            assert!(!is_ducklake_tombstone_column_name(column_name));
        }
    }

    #[test]
    fn tombstone_namespace_recognition_uses_duckdb_identifier_equivalence() {
        assert!(is_ducklake_tombstone_namespace_name(
            "SUPABASE_ETL_DUCKLAKE_DROPPED_3_0000000000ABCDEF"
        ));
        assert!(!is_ducklake_tombstone_column_name(
            "SUPABASE_ETL_DUCKLAKE_DROPPED_3_0000000000ABCDEF"
        ));

        for column_name in
            ["supabase_etl_ducklake_dropped_business", "SUPABASE_ETL_DUCKLAKE_DROPPED_BUSINESS"]
        {
            assert!(!is_ducklake_tombstone_namespace_name(column_name));
        }
    }

    #[test]
    fn ducklake_rejects_source_column_matching_generated_tombstone_name() {
        for column_name in [
            "supabase_etl_ducklake_dropped_2_0000000000abcdef",
            "SUPABASE_ETL_DUCKLAKE_DROPPED_2_0000000000ABCDEF",
        ] {
            let table_schema = Arc::new(TableSchema::new(
                TableId::new(1),
                TableName::new("public".to_owned(), "users".to_owned()),
                vec![ColumnSchema::new(column_name.to_owned(), PgType::TEXT, -1, 1, true)],
            ));
            let schema = ReplicatedTableSchema::all(table_schema);

            let error = validate_ducklake_tombstone_namespace(&schema).unwrap_err();

            assert_eq!(error.kind(), ErrorKind::SourceSchemaError);
        }
    }

    #[test]
    fn ducklake_rejects_case_colliding_source_columns() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(1),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("value".to_owned(), PgType::TEXT, -1, 1, true),
                ColumnSchema::new("VALUE".to_owned(), PgType::TEXT, -1, 2, true),
            ],
        ));
        let schema = ReplicatedTableSchema::all(table_schema);

        let error = validate_ducklake_table_shape(&schema).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::SourceSchemaError);
        assert_eq!(error.description(), Some("Source column names collide in the destination"));
        assert!(error.detail().is_some_and(|detail| detail.contains("'value' and 'VALUE'")));
    }

    #[test]
    fn missing_replicated_columns_ducklake_returns_only_active_target_columns() {
        let table_schema = Arc::new(TableSchema::new(
            TableId::new(4),
            TableName::new("public".to_owned(), "users".to_owned()),
            vec![
                ColumnSchema::new("id".to_owned(), PgType::INT4, -1, 1, false).with_primary_key(1),
                ColumnSchema::new("name".to_owned(), PgType::TEXT, -1, 2, true),
                ColumnSchema::new("hidden".to_owned(), PgType::TEXT, -1, 3, true),
                ColumnSchema::new("email".to_owned(), PgType::TEXT, -1, 4, true),
            ],
        ));
        let target_schema = ReplicatedTableSchema::from_mask(
            Arc::clone(&table_schema),
            ReplicationMask::from_bytes(vec![1, 1, 0, 1]),
        );

        let missing_columns = missing_replicated_columns_ducklake(
            &["id".to_owned(), "name".to_owned()],
            &target_schema,
        );
        let missing_column_names: Vec<_> =
            missing_columns.iter().map(|column| column.name.as_str()).collect();

        assert_eq!(missing_column_names, vec!["email"]);
    }

    fn path_to_file_url(path: &Path) -> Url {
        Url::from_file_path(path).expect("failed to convert path to file url")
    }

    fn local_pg_connection_config(database_name: String) -> PgConnectionConfig {
        PgConnectionConfig {
            host: env::var("TESTS_DATABASE_HOST").expect("TESTS_DATABASE_HOST must be set"),
            hostaddr: None,
            port: env::var("TESTS_DATABASE_PORT")
                .expect("TESTS_DATABASE_PORT must be set")
                .parse()
                .expect("TESTS_DATABASE_PORT must be a valid port number"),
            name: database_name,
            username: env::var("TESTS_DATABASE_USERNAME")
                .expect("TESTS_DATABASE_USERNAME must be set"),
            password: env::var("TESTS_DATABASE_PASSWORD").ok().map(Into::into),
            tls: local_tls_config_from_env(),
            keepalive: TcpKeepaliveConfig::default(),
        }
    }

    async fn create_catalog_database() -> (PgDatabase<Client>, Url) {
        let database_name = Uuid::new_v4().to_string();
        let host = env::var("TESTS_DATABASE_HOST").expect("TESTS_DATABASE_HOST must be set");
        let port = env::var("TESTS_DATABASE_PORT")
            .expect("TESTS_DATABASE_PORT must be set")
            .parse()
            .expect("TESTS_DATABASE_PORT must be a valid port number");
        let username =
            env::var("TESTS_DATABASE_USERNAME").expect("TESTS_DATABASE_USERNAME must be set");
        let password = env::var("TESTS_DATABASE_PASSWORD").ok();
        let database = PgDatabase::new(local_pg_connection_config(database_name.clone())).await;

        let mut catalog_url = Url::parse("postgres://localhost").expect("failed to parse base url");
        catalog_url.set_host(Some(&host)).expect("failed to set catalog host");
        catalog_url.set_port(Some(port)).expect("failed to set catalog port");
        catalog_url.set_username(&username).expect("failed to set catalog username");
        catalog_url.set_password(password.as_deref()).expect("failed to set catalog password");
        catalog_url.set_path(&database_name);

        (database, catalog_url)
    }

    fn current_vendored_extension_dir() -> Option<PathBuf> {
        let platform_dir = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("linux", "x86_64" | "amd64") => "linux_amd64",
            ("linux", "aarch64" | "arm64") => "linux_arm64",
            ("macos", "x86_64" | "amd64") => "osx_amd64",
            ("macos", "aarch64" | "arm64") => "osx_arm64",
            _ => return None,
        };
        let env_override = std::env::var_os("ETL_DUCKDB_EXTENSION_ROOT").map(PathBuf::from);
        let candidate_roots = env_override
            .into_iter()
            .chain([
                PathBuf::from("/app/duckdb_extensions"),
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../vendor/duckdb/extensions"),
            ])
            .collect::<Vec<_>>();

        for root in candidate_roots {
            let extension_dir = root.join("1.5.3").join(platform_dir);
            let ducklake_extension = extension_dir.join("ducklake.duckdb_extension");
            let postgres_scanner_extension = extension_dir.join(POSTGRES_SCANNER_EXTENSION_FILE);

            if ducklake_extension.is_file() && postgres_scanner_extension.is_file() {
                return Some(extension_dir);
            }
        }

        None
    }

    fn open_verification_connection() -> Connection {
        let duckdb_dir = tempfile::Builder::new()
            .prefix("etl_ducklake_verify_")
            .tempdir()
            .expect("failed to create verification duckdb dir")
            .keep();
        let duckdb_path = duckdb_dir.join("verify.duckdb");

        let conn = if current_vendored_extension_dir().is_some() {
            Connection::open_with_flags(
                &duckdb_path,
                Config::default()
                    .enable_autoload_extension(false)
                    .expect("failed to disable DuckDB extension autoload"),
            )
            .expect("failed to open verification DuckDB")
        } else {
            Connection::open(&duckdb_path).expect("failed to open verification DuckDB")
        };

        conn.execute_batch("SET preserve_insertion_order = false;")
            .expect("failed to configure verification DuckDB session");
        conn
    }

    fn ducklake_load_sql() -> String {
        if let Some(extension_dir) = current_vendored_extension_dir() {
            let ducklake_extension = extension_dir.join("ducklake.duckdb_extension");
            let postgres_scanner_extension = extension_dir.join(POSTGRES_SCANNER_EXTENSION_FILE);

            return format!(
                "LOAD {}; LOAD {};",
                quote_literal(&ducklake_extension.display().to_string()),
                quote_literal(&postgres_scanner_extension.display().to_string()),
            );
        }

        "INSTALL ducklake; LOAD ducklake; INSTALL postgres_scanner; LOAD postgres_scanner;"
            .to_owned()
    }

    fn open_lake_conn(catalog: &Url, data: &Url) -> Connection {
        let conn = open_verification_connection();
        let catalog_attach_target =
            catalog_conninfo_from_url(catalog).expect("invalid catalog url");
        conn.execute_batch(&format!(
            "{} ATTACH {} AS {} (DATA_PATH {});",
            ducklake_load_sql(),
            quote_literal(&format!("ducklake:{catalog_attach_target}")),
            quote_identifier(LAKE_CATALOG),
            quote_literal(data.as_str()),
        ))
        .expect("failed to attach DuckLake catalog");
        conn
    }

    fn lake_table_exists(conn: &Connection, table_name: &DuckLakeTableName) -> bool {
        conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_catalog = {} AND \
                 table_schema = {} AND table_name = {}",
                quote_literal(LAKE_CATALOG),
                quote_literal(table_name.schema()),
                quote_literal(table_name.table()),
            ),
            [],
            |row| row.get::<_, i64>(0),
        )
        .is_ok_and(|count| count > 0)
    }

    async fn open_lake_conn_when_table_visible(
        catalog: &Url,
        data: &Url,
        table_name: &DuckLakeTableName,
    ) -> Connection {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let conn = open_lake_conn(catalog, data);
            if lake_table_exists(&conn, table_name) {
                return conn;
            }

            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for DuckLake table `{table_name}` to become visible",
            );
            drop(conn);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    #[test]
    fn table_name_to_ducklake_table_name_preserves_schema() {
        assert_eq!(
            table_name_to_ducklake_table_name(&TableName {
                schema: "public".to_owned(),
                name: "orders".to_owned(),
            })
            .unwrap(),
            DuckLakeTableName::new("public", "orders")
        );
        assert_eq!(
            table_name_to_ducklake_table_name(&TableName {
                schema: "my_schema".to_owned(),
                name: "my_table".to_owned(),
            })
            .unwrap(),
            DuckLakeTableName::new("my_schema", "my_table")
        );
    }

    #[test]
    fn is_create_table_conflict_matches_ducklake_commit_conflict() {
        let error = duckdb::Error::DuckDBFailure(
            duckdb::ffi::Error::new(1),
            Some(
                "TransactionContext Error: Failed to commit: Failed to commit DuckLake \
                 transaction. Transaction conflict - attempting to create table \"public_users\" \
                 in schema \"main\" - but this table has been created by another transaction \
                 already"
                    .to_owned(),
            ),
        );

        assert!(is_create_table_conflict(&error, "public_users"));
        assert!(!is_create_table_conflict(&error, "public_orders"));
    }

    mod postgres_backed {
        use super::*;

        /// A pending replay epoch is stable until its table reset commits.
        #[tokio::test(flavor = "multi_thread")]
        async fn replay_epoch_transition_reuses_pending_epoch_until_reset_completes() {
            let dir = TempDir::new().expect("failed to create temp dir");
            let data = path_to_file_url(&dir.path().join("data"));
            let (_catalog_database, catalog) = create_catalog_database().await;
            let store = MemoryStore::new();
            let schema = make_schema(1, "public", "users");
            let replicated_table_schema = ReplicatedTableSchema::all(Arc::new(schema.clone()));
            let table_name = table_name_to_ducklake_table_name(&schema.name).unwrap();

            store.store_table_schema(schema).await.expect("failed to seed schema");
            let destination =
                DuckLakeDestination::new(catalog, data, 1, None, None, None, None, store)
                    .await
                    .expect("failed to create destination");
            destination
                .write_table_rows(
                    &replicated_table_schema,
                    Some(test_table_copy_batch_id()),
                    vec![TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())])],
                )
                .await
                .expect("failed to write row");
            destination
                .write_table_rows(&replicated_table_schema, None, Vec::new())
                .await
                .expect("failed to finish table copy");

            let pending_replay_epoch = destination
                .begin_table_replay_epoch_transition(&table_name)
                .await
                .expect("failed to begin replay epoch transition");
            let resumed_replay_epoch = destination
                .begin_table_replay_epoch_transition(&table_name)
                .await
                .expect("failed to resume replay epoch transition");
            assert_eq!(resumed_replay_epoch, pending_replay_epoch);
            assert_eq!(
                destination
                    .read_table_replay_epoch(&table_name)
                    .await
                    .expect("failed to read committed replay epoch"),
                crate::ducklake::replay_epoch::LEGACY_REPLAY_EPOCH
            );

            // A retry repeats the idempotent table reset with the persisted
            // pending epoch and promotes it only after DuckLake commits.
            destination
                .truncate_table(&replicated_table_schema)
                .await
                .expect("failed to retry table reset");
            assert_eq!(
                destination
                    .read_table_replay_epoch(&table_name)
                    .await
                    .expect("failed to read promoted replay epoch"),
                pending_replay_epoch
            );

            let epochs_table = format!(
                "{}.{}",
                quote_postgres_identifier(destination.metadata_schema.as_ref()),
                quote_postgres_identifier("__etl_replay_epochs")
            );
            let sql =
                format!("select pending_replay_epoch from {epochs_table} where table_name = $1;");
            let pending_replay_epoch = sqlx::query_scalar::<_, Option<String>>(AssertSqlSafe(sql))
                .bind(table_name.id())
                .fetch_one(&destination.metadata_pg_pool)
                .await
                .expect("failed to read pending replay epoch");
            assert_eq!(pending_replay_epoch, None);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn query_table_storage_metrics_reads_ducklake_metadata() {
            let dir = TempDir::new().expect("failed to create temp dir");
            let data = path_to_file_url(&dir.path().join("data"));
            let (_catalog_database, catalog) = create_catalog_database().await;
            let store = MemoryStore::new();
            let schema = make_schema(1, "public", "users");
            let replicated_table_schema = ReplicatedTableSchema::all(Arc::new(schema.clone()));
            let table_name = table_name_to_ducklake_table_name(&schema.name).unwrap();

            store.store_table_schema(schema.clone()).await.expect("failed to seed schema");

            let destination = DuckLakeDestination::new(
                catalog.clone(),
                data.clone(),
                1,
                None,
                None,
                None,
                None,
                store,
            )
            .await
            .expect("failed to create destination");

            destination
                .write_table_rows(
                    &replicated_table_schema,
                    Some(test_table_copy_batch_id()),
                    vec![
                        TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())]),
                        TableRow::new(vec![Cell::I32(2), Cell::String("bob".to_owned())]),
                    ],
                )
                .await
                .expect("failed to write rows");
            destination
                .write_table_rows(&replicated_table_schema, None, Vec::new())
                .await
                .expect("failed to finish table copy");

            let conn = open_lake_conn_when_table_visible(&catalog, &data, &table_name).await;
            let metadata_schema = resolve_ducklake_metadata_schema_blocking(&conn)
                .expect("failed to resolve metadata schema");
            let metadata_pg_pool =
                build_ducklake_metadata_pg_pool(&catalog).expect("failed to create metadata pool");
            let _rows_flushed =
                flush_table_inlined_data(&conn, table_name.schema(), table_name.table())
                    .expect("failed to materialize inlined rows for storage metrics test");
            let deadline = Instant::now() + Duration::from_secs(10);
            let metrics = loop {
                let metrics =
                    query_table_storage_metrics(&metadata_pg_pool, &metadata_schema, &table_name)
                        .await
                        .expect("failed to query storage metrics");
                if metrics.active_data_files >= 1 {
                    break metrics;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for storage metrics after materialization"
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            };

            assert!(metrics.active_data_files >= 1);
            assert!(metrics.active_data_bytes > 0);
            assert_eq!(metrics.active_delete_files, 0);
            assert_eq!(metrics.deleted_rows, 0);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn query_catalog_maintenance_metrics_reports_active_data_files_total() {
            let dir = TempDir::new().expect("failed to create temp dir");
            let data = path_to_file_url(&dir.path().join("data"));
            let (_catalog_database, catalog) = create_catalog_database().await;
            let store = MemoryStore::new();
            let schema = make_schema(1, "public", "users");
            let replicated_table_schema = ReplicatedTableSchema::all(Arc::new(schema.clone()));
            let table_name = table_name_to_ducklake_table_name(&schema.name).unwrap();

            store.store_table_schema(schema.clone()).await.expect("failed to seed schema");

            let destination = DuckLakeDestination::new(
                catalog.clone(),
                data.clone(),
                1,
                None,
                None,
                None,
                None,
                store,
            )
            .await
            .expect("failed to create destination");

            destination
                .write_table_rows(
                    &replicated_table_schema,
                    Some(test_table_copy_batch_id()),
                    vec![
                        TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())]),
                        TableRow::new(vec![Cell::I32(2), Cell::String("bob".to_owned())]),
                    ],
                )
                .await
                .expect("failed to write rows");
            destination
                .write_table_rows(&replicated_table_schema, None, Vec::new())
                .await
                .expect("failed to finish table copy");

            let conn = open_lake_conn_when_table_visible(&catalog, &data, &table_name).await;
            let metadata_schema = resolve_ducklake_metadata_schema_blocking(&conn)
                .expect("failed to resolve metadata schema");
            let metadata_pg_pool =
                build_ducklake_metadata_pg_pool(&catalog).expect("failed to create metadata pool");
            let _rows_flushed =
                flush_table_inlined_data(&conn, table_name.schema(), table_name.table())
                    .expect("failed to materialize inlined rows for catalog metrics test");
            let deadline = Instant::now() + Duration::from_secs(10);
            let metrics = loop {
                let metrics =
                    query_catalog_maintenance_metrics(&metadata_pg_pool, &metadata_schema)
                        .await
                        .expect("failed to query catalog maintenance metrics");
                if metrics.active_data_files_total >= 1 {
                    break metrics;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for active data files total after materialization"
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            };

            assert!(metrics.active_data_files_total >= 1);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn query_catalog_maintenance_metrics_reads_ducklake_metadata() {
            let dir = TempDir::new().expect("failed to create temp dir");
            let data = path_to_file_url(&dir.path().join("data"));
            let (_catalog_database, catalog) = create_catalog_database().await;
            let store = MemoryStore::new();
            let schema = make_schema(1, "public", "users");
            let replicated_table_schema = ReplicatedTableSchema::all(Arc::new(schema.clone()));
            let table_name = table_name_to_ducklake_table_name(&schema.name).unwrap();

            store.store_table_schema(schema.clone()).await.expect("failed to seed schema");

            let destination = DuckLakeDestination::new(
                catalog.clone(),
                data.clone(),
                1,
                None,
                None,
                None,
                None,
                store,
            )
            .await
            .expect("failed to create destination");

            destination
                .write_table_rows(
                    &replicated_table_schema,
                    Some(test_table_copy_batch_id()),
                    vec![TableRow::new(vec![Cell::I32(1), Cell::String("alice".to_owned())])],
                )
                .await
                .expect("failed to write rows");
            destination
                .write_table_rows(&replicated_table_schema, None, Vec::new())
                .await
                .expect("failed to finish table copy");
            destination
                .truncate_table(&replicated_table_schema)
                .await
                .expect("failed to truncate table");

            destination.shutdown().await.expect("failed to shutdown destination");
            drop(destination);

            let conn = open_lake_conn_when_table_visible(&catalog, &data, &table_name).await;
            let metadata_schema = resolve_ducklake_metadata_schema_blocking(&conn)
                .expect("failed to resolve metadata schema");
            let metadata_pg_pool =
                build_ducklake_metadata_pg_pool(&catalog).expect("failed to create metadata pool");
            let metrics = query_catalog_maintenance_metrics(&metadata_pg_pool, &metadata_schema)
                .await
                .expect("failed to query catalog maintenance metrics");

            assert!(metrics.active_data_files_total >= 0);
            assert!(metrics.snapshots_total >= 1);
            assert!(metrics.oldest_snapshot_age_seconds >= 0);
            assert!(metrics.files_scheduled_for_deletion_total >= 0);
            assert!(metrics.files_scheduled_for_deletion_bytes >= 0);
            assert!(metrics.oldest_scheduled_deletion_age_seconds >= 0);
        }
    }
    #[tokio::test]
    async fn mapped_table_identity_rejects_pending_and_applied_owners() {
        let store = MemoryStore::new();
        let first = make_schema(1, "public", "first");
        let second = make_schema(2, "public", "second");
        store.store_table_schema(first.clone()).await.unwrap();
        store.store_table_schema(second.clone()).await.unwrap();
        let name = ducklake_table_name();
        let schema = ReplicatedTableSchema::all(Arc::new(first.clone()));
        let pending = DestinationTableMetadata::new_creating(
            name.to_metadata_id().unwrap(),
            first.snapshot_id,
            schema.replication_mask().clone(),
        );
        for metadata in [pending.clone(), pending.to_applied()] {
            store.store_destination_table_metadata(first.id, metadata).await.unwrap();
            ensure_unique_table_identity(&store, first.id, &name).await.unwrap();
            for candidate in [
                name.clone(),
                DuckLakeTableName::new("PUBLIC", "users"),
                DuckLakeTableName::new("public", "Users"),
                DuckLakeTableName::new("PUBLIC", "USERS"),
            ] {
                ensure_unique_table_identity(&store, first.id, &candidate).await.unwrap();
                let error =
                    ensure_unique_table_identity(&store, second.id, &candidate).await.unwrap_err();
                assert_eq!(error.kind(), ErrorKind::InvalidState);
                assert!(store.get_destination_table_metadata(second.id).await.unwrap().is_none());
            }
            ensure_unique_table_identity(
                &store,
                second.id,
                &DuckLakeTableName::new("other", "users"),
            )
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn mapped_table_identity_rejects_reserved_helpers() {
        let store = MemoryStore::new();
        let schema = make_schema(1, "public", "users");
        store.store_table_schema(schema.clone()).await.unwrap();
        for table in [
            "__etl_applied_table_batches",
            "__etl_streaming_progress",
            "__ETL_APPLIED_TABLE_BATCHES",
            "__EtL_streaming_progress",
        ] {
            for namespace in ["main", "MAIN", "public"] {
                let name = DuckLakeTableName::new(namespace, table);
                let error =
                    ensure_unique_table_identity(&store, schema.id, &name).await.unwrap_err();
                assert_eq!(error.kind(), ErrorKind::InvalidState);
                assert!(store.get_destination_table_metadata(schema.id).await.unwrap().is_none());
            }
        }
        for table in ["__etl", "__et", "users", "éééusers", "__étl_users"] {
            ensure_unique_table_identity(&store, schema.id, &DuckLakeTableName::new("main", table))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn mapped_table_identity_preserves_non_ascii_distinctions() {
        let store = MemoryStore::new();
        let first = make_schema(1, "public", "first");
        let second = make_schema(2, "public", "second");
        store.store_table_schema(first.clone()).await.unwrap();
        store.store_table_schema(second.clone()).await.unwrap();
        let name = DuckLakeTableName::new("schéma", "Üsers");
        let schema = ReplicatedTableSchema::all(Arc::new(first.clone()));
        let metadata = DestinationTableMetadata::new_creating(
            name.to_metadata_id().unwrap(),
            first.snapshot_id,
            schema.replication_mask().clone(),
        );
        store.store_destination_table_metadata(first.id, metadata).await.unwrap();
        for candidate in
            [DuckLakeTableName::new("schéma", "üsers"), DuckLakeTableName::new("schÉma", "Üsers")]
        {
            ensure_unique_table_identity(&store, second.id, &candidate).await.unwrap();
        }
        let error = ensure_unique_table_identity(
            &store,
            second.id,
            &DuckLakeTableName::new("SCHéMA", "ÜSERS"),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidState);
    }
}
