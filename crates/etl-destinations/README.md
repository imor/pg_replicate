# `etl-destinations`

Built-in destination implementations for [Supabase ETL](https://supabase.github.io/etl/).
Enable only the destination feature you need when embedding the `etl` crate or
building `etl-replicator`.

| Feature | Destination | Status |
| --- | --- | --- |
| `clickhouse` | ClickHouse | In progress |
| `bigquery` | Google BigQuery | Stable |
| `ducklake` | DuckLake | In progress |
| `snowflake` | Snowflake | In progress |
| `iceberg` | Apache Iceberg | Deprecated |

ClickHouse is the easiest destination to run locally. BigQuery is the most
mature cloud destination. See the
[Destinations reference](https://supabase.github.io/etl/reference/destinations/)
for each implementation's maturity, requirements, and limitations.

DuckLake external maintenance is configured at runtime with
`maintenance_mode`: `disabled`, `kubernetes`, or `postgres`. The default is
`disabled`. Kubernetes coordination expects
`ETL_DUCKLAKE_MAINTENANCE_CR_NAME` and
`ETL_DUCKLAKE_MAINTENANCE_CR_NAMESPACE`. Postgres coordination uses the same
Postgres catalog connection as DuckLake and stores coordination state in the
`etl` schema.

When embedding DuckLake, the builder also accepts a `connection_initializer`
and a `table_name_mapper`. The initializer returns a fresh DuckDB instance with
`lake` already attached to the configured catalog and data path. It owns
extension loading, credentials, resource/spill limits and catalog options,
including schema-scoped options and the attachment's data inlining limit for both
COPY and CDC. Host initialization bypasses the standalone COPY inlining override.
ETL continues to own COPY, CDC, replay state, pooled connection clones and session
settings. The initializer runs again on instance replacement; it must not retain
connections to retired instances. Existing persisted table names take precedence
over a new mapping, and sorting configuration uses destination names.

`DuckLakeDestination::run_maintenance` runs caller-selected work on the writer's
instance under the existing mutation pause and query watchdog. The caller owns
the schedule and table selection. Cancellation does not release the pause until
the blocking operation exits. Callbacks must finish transactions and must not
retain cloned connections outside the operation.

The builder's `cdc_batch_size` controls the transaction cap for mutations already
received from the pipeline (default: 16). Increasing it can reduce file creation
with inlining disabled, with longer transactions and larger retry units. It does
not change how long the pipeline waits to fill an input batch.

The `bundled` feature is enabled by default and keeps the bundled DuckDB/JSON/
Parquet build. Hosts supplying a compatible native DuckDB library can disable
default features and select `ducklake` plus their TLS feature. Their initializer
must load extensions matching that library. Other dependencies enabling `bundled`
will still enable it through Cargo feature unification.

`ducklake-query-error-details` explicitly includes original DuckDB UPDATE/DELETE
errors, their source chains and SQL in returned errors. Automatic mutation/task
logs retain structural context without rendering those errors. Returned errors can contain row values;
leave the feature disabled to retain the default redaction. Concurrent table
failures are logged individually and accepted table tasks finish before the
first error is returned to the apply loop.
