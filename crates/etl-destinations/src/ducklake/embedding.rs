//! Host configuration for an embedded DuckLake destination.

use std::{num::NonZeroUsize, sync::Arc};

use etl::{error::EtlResult, schema::TableName};

use crate::ducklake::DuckLakeTableName;

/// Creates a fully initialized database for a new connection-pool generation.
pub(super) type ConnectionInitializer =
    Arc<dyn Fn() -> EtlResult<duckdb::Connection> + Send + Sync>;

/// Maps a source table before its destination identity is persisted.
pub(super) type TableNameMapper =
    Arc<dyn Fn(&TableName) -> EtlResult<DuckLakeTableName> + Send + Sync>;

/// Optional host policies; absent callbacks retain the standalone defaults.
#[derive(Clone, Default)]
pub(super) struct EmbeddingOptions {
    pub(super) connection_initializer: Option<ConnectionInitializer>,
    pub(super) table_name_mapper: Option<TableNameMapper>,
    pub(super) cdc_batch_size: Option<NonZeroUsize>,
}
