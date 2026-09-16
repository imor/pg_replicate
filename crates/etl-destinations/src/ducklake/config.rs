use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    path::{Path, PathBuf},
    str::FromStr,
};

use etl::{
    error::{ErrorKind, EtlResult},
    etl_error,
};
use etl_config::{libpq_tcp_host, shared::DuckLakeWriterConfig};
use pg_escape::{quote_identifier, quote_literal};
use tokio_postgres::{
    Config as PgConfig,
    config::{Host, SslMode},
};
use url::Url;

use crate::ducklake::LAKE_CATALOG;

const DUCKDB_EXTENSION_ROOT_ENV_VAR: &str = "ETL_DUCKDB_EXTENSION_ROOT";
const CONTAINER_DUCKDB_EXTENSION_ROOT: &str = "/app/duckdb_extensions";
const DUCKDB_EXTENSION_VERSION: &str = "1.5.3";
const DUCKLAKE_EXTENSION_FILE: &str = "ducklake.duckdb_extension";
const HTTPFS_EXTENSION_FILE: &str = "httpfs.duckdb_extension";
const POSTGRES_SCANNER_EXTENSION_FILE: &str = "postgres_scanner.duckdb_extension";
pub(super) const TARGET_FILE_SIZE_OPTION_NAME: &str = "target_file_size";
pub(super) const EXPIRE_SNAPSHOTS_OLDER_THAN: &str = "7 days";
/// Minimum snapshot-retention interval accepted for DuckLake maintenance.
pub(super) const MIN_EXPIRE_SNAPSHOTS_OLDER_THAN: &str = "1 day";
const PARQUET_COMPRESSION_OPTION_NAME: &str = "parquet_compression";
const PARQUET_COMPRESSION_OPTION_VALUE: &str = "zstd";
const PARQUET_ROW_GROUP_SIZE_BYTES_OPTION_NAME: &str = "parquet_row_group_size_bytes";
const PARQUET_ROW_GROUP_SIZE_OPTION_NAME: &str = "parquet_row_group_size";
const PARQUET_VERSION_OPTION_NAME: &str = "parquet_version";
const PARQUET_VERSION_OPTION_VALUE: u8 = 2;
const PRESERVE_INSERTION_ORDER_OPTION_NAME: &str = "preserve_insertion_order";

/// Builds the SQL that configures DuckDB's session-level write ordering.
fn configure_writer_session_sql() -> String {
    format!("SET {PRESERVE_INSERTION_ORDER_OPTION_NAME} = false;")
}

/// Builds the SQL that configures DuckLake's global Parquet writer settings.
fn configure_parquet_settings_sql(writer_config: &DuckLakeWriterConfig) -> String {
    format!(
        "CALL {LAKE_CATALOG}.set_option({}, {}); CALL {LAKE_CATALOG}.set_option({}, {}); CALL \
         {LAKE_CATALOG}.set_option({}, {}); CALL {LAKE_CATALOG}.set_option({}, {});",
        quote_literal(PARQUET_COMPRESSION_OPTION_NAME),
        quote_literal(PARQUET_COMPRESSION_OPTION_VALUE),
        quote_literal(PARQUET_ROW_GROUP_SIZE_BYTES_OPTION_NAME),
        quote_literal(writer_config.parquet_row_group_size_bytes()),
        quote_literal(PARQUET_ROW_GROUP_SIZE_OPTION_NAME),
        quote_literal(writer_config.parquet_row_group_size()),
        quote_literal(PARQUET_VERSION_OPTION_NAME),
        PARQUET_VERSION_OPTION_VALUE,
    )
}

/// Builds the SQL that sets DuckLake's maintenance target file size.
pub(super) fn maintenance_target_file_size_sql(target_file_size: &str) -> String {
    format!(
        "CALL ducklake_set_option({}, {}, {});",
        quote_literal(LAKE_CATALOG),
        quote_literal(TARGET_FILE_SIZE_OPTION_NAME),
        quote_literal(target_file_size),
    )
}

/// Resolves the configured snapshot-retention window or falls back to the
/// default.
pub(super) fn resolve_expire_snapshots_older_than(
    expire_snapshots_older_than: Option<&str>,
) -> &str {
    expire_snapshots_older_than.unwrap_or(EXPIRE_SNAPSHOTS_OLDER_THAN)
}

/// Builds the SQL used to validate the configured snapshot-retention window.
pub(super) fn validate_expire_snapshots_older_than_sql(
    expire_snapshots_older_than: &str,
) -> String {
    // Interval ordering treats months as 30 days, unlike timestamp arithmetic.
    format!(
        "select cast(now() as timestamp) - cast({} as interval) <= cast(now() as timestamp) - \
         cast({} as interval);",
        quote_literal(expire_snapshots_older_than),
        quote_literal(MIN_EXPIRE_SNAPSHOTS_OLDER_THAN),
    )
}

/// One named DuckDB setup phase for a DuckLake connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DuckLakeSetupStep {
    /// Log-friendly setup phase name.
    pub(super) label: &'static str,
    /// SQL batch executed for this phase.
    pub(super) sql: String,
}

/// Ordered DuckDB setup phases for one DuckLake connection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DuckLakeSetupPlan {
    steps: Vec<DuckLakeSetupStep>,
}

impl DuckLakeSetupPlan {
    /// Retains connection-local writer settings for a host-initialized
    /// database.
    pub(super) fn for_embedded_instance() -> Self {
        Self {
            steps: vec![DuckLakeSetupStep {
                label: "configure_writer_session",
                sql: configure_writer_session_sql(),
            }],
        }
    }

    /// Returns the ordered setup steps.
    pub(super) fn steps(&self) -> &[DuckLakeSetupStep] {
        &self.steps
    }

    /// Returns setup steps that must run for every connection cloned from one
    /// initialized DuckDB database.
    ///
    /// Extension loading, object-store secrets, DuckLake attachment, and
    /// catalog options belong to the shared database instance. Writer session
    /// settings remain connection-local.
    pub(super) fn connection_steps(&self) -> impl Iterator<Item = &DuckLakeSetupStep> {
        self.steps.iter().filter(|step| step.label == "configure_writer_session")
    }

    /// Builds a setup plan from explicit setup steps.
    #[cfg(test)]
    pub(super) fn from_steps(steps: Vec<DuckLakeSetupStep>) -> Self {
        Self { steps }
    }

    /// Returns all setup SQL concatenated into one batch.
    #[cfg(test)]
    pub(super) fn combined_sql(&self) -> String {
        self.steps.iter().map(|step| step.sql.as_str()).collect::<Vec<_>>().join(" ")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DuckDbExtensionStrategy {
    VendoredLocal { platform_dir: &'static str },
    InstallFromRepository,
}

/// Resolved extension-loading inputs for one setup plan.
#[derive(Debug, Clone, Copy)]
struct DuckDbExtensionSetup<'a> {
    /// Strategy used to load DuckDB extensions.
    strategy: DuckDbExtensionStrategy,
    /// Vendored extension root required by local-loading strategies.
    vendored_root: Option<&'a Path>,
}

impl DuckDbExtensionStrategy {
    pub(super) fn disables_autoload(self) -> bool {
        matches!(self, Self::VendoredLocal { .. })
    }
}

/// Returns the DuckDB extension strategy for the current platform.
pub(super) fn current_duckdb_extension_strategy() -> EtlResult<DuckDbExtensionStrategy> {
    duckdb_extension_strategy(
        env::consts::OS,
        env::consts::ARCH,
        env::var_os(DUCKDB_EXTENSION_ROOT_ENV_VAR).map(PathBuf::from),
        Path::new(CONTAINER_DUCKDB_EXTENSION_ROOT),
        &repo_vendored_extension_root(),
    )
}

fn duckdb_extension_strategy(
    os: &str,
    arch: &str,
    env_override: Option<PathBuf>,
    container_root: &Path,
    repo_root: &Path,
) -> EtlResult<DuckDbExtensionStrategy> {
    match os {
        "linux" => {
            let platform_dir = match arch {
                "x86_64" | "amd64" => "linux_amd64",
                "aarch64" | "arm64" => "linux_arm64",
                _ => {
                    return Err(etl_error!(
                        ErrorKind::ConfigError,
                        "Unsupported DuckDB extension platform",
                        format!(
                            "linux architecture `{arch}` is not supported for vendored DuckDB \
                             extensions"
                        )
                    ));
                }
            };

            if vendored_extension_dir(platform_dir, env_override, container_root, repo_root)?
                .is_some()
            {
                Ok(DuckDbExtensionStrategy::VendoredLocal { platform_dir })
            } else {
                Ok(DuckDbExtensionStrategy::InstallFromRepository)
            }
        }
        "macos" => {
            let platform_dir = match arch {
                "x86_64" | "amd64" => "osx_amd64",
                "aarch64" | "arm64" => "osx_arm64",
                _ => {
                    return Err(etl_error!(
                        ErrorKind::ConfigError,
                        "Unsupported DuckDB extension platform",
                        format!(
                            "macos architecture `{arch}` is not supported for vendored DuckDB \
                             extensions"
                        )
                    ));
                }
            };

            if vendored_extension_dir(platform_dir, env_override, container_root, repo_root)?
                .is_some()
            {
                Ok(DuckDbExtensionStrategy::VendoredLocal { platform_dir })
            } else {
                Ok(DuckDbExtensionStrategy::InstallFromRepository)
            }
        }
        "windows" => Ok(DuckDbExtensionStrategy::InstallFromRepository),
        _ => Err(etl_error!(
            ErrorKind::ConfigError,
            "Unsupported DuckDB extension platform",
            format!("operating system `{os}` is not supported for DuckDB extensions")
        )),
    }
}

fn repo_vendored_extension_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
        .join("vendor/duckdb/extensions")
}

fn vendored_extension_dir(
    platform_dir: &str,
    env_override: Option<PathBuf>,
    container_root: &Path,
    repo_root: &Path,
) -> EtlResult<Option<PathBuf>> {
    if let Some(root) = env_override {
        let directory = root.join(DUCKDB_EXTENSION_VERSION).join(platform_dir);
        ensure_vendored_extension_dir(&directory)?;
        return Ok(Some(directory));
    }

    for root in [container_root, repo_root] {
        let directory = root.join(DUCKDB_EXTENSION_VERSION).join(platform_dir);
        if !directory.exists() {
            continue;
        }
        ensure_vendored_extension_dir(&directory)?;
        return Ok(Some(directory));
    }

    Ok(None)
}

fn require_vendored_extension_dir(
    platform_dir: &str,
    env_override: Option<PathBuf>,
    container_root: &Path,
    repo_root: &Path,
) -> EtlResult<PathBuf> {
    vendored_extension_dir(platform_dir, env_override, container_root, repo_root)?.ok_or_else(
        || {
            etl_error!(
                ErrorKind::ConfigError,
                "Vendored DuckDB extensions not found",
                format!(
                    "expected vendored DuckDB extensions in one of: {}, {}",
                    container_root.join(DUCKDB_EXTENSION_VERSION).join(platform_dir).display(),
                    repo_root.join(DUCKDB_EXTENSION_VERSION).join(platform_dir).display()
                )
            )
        },
    )
}

fn ensure_vendored_extension_dir(directory: &Path) -> EtlResult<()> {
    let missing = [DUCKLAKE_EXTENSION_FILE]
        .into_iter()
        .filter(|filename| !directory.join(filename).is_file())
        .collect::<Vec<_>>();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(etl_error!(
            ErrorKind::ConfigError,
            "Vendored DuckDB extensions not found",
            format!("missing {} in `{}`", missing.join(", "), directory.display())
        ))
    }
}

fn vendored_extension_path(extension_dir: &Path, filename: &str) -> EtlResult<String> {
    extension_dir.join(filename).to_str().map(str::to_owned).ok_or_else(|| {
        etl_error!(
            ErrorKind::ConfigError,
            "Vendored DuckDB extension path contains non-utf8 characters",
            extension_dir.display().to_string()
        )
    })
}

/// S3-compatible storage credentials for DuckDB's httpfs extension.
///
/// Used when `data_path` points to an S3 or GCS URI. The `endpoint`
/// field supports an optional path prefix (e.g. `localhost:5000/s3` for
/// Supabase Storage or other S3-compatible services mounted at a sub-path).
#[derive(Debug, Clone)]
pub struct S3Config {
    pub access_key_id: String,
    pub secret_access_key: String,
    /// AWS region or equivalent (e.g. `"us-east-1"`).
    pub region: String,
    /// Host, host:port, or host:port/path of the S3-compatible endpoint.
    /// Defaults to AWS S3 if not set.
    pub endpoint: Option<String>,
    /// `"path"` (default for MinIO / Supabase Storage) or `"vhost"` (AWS S3
    /// default).
    pub url_style: String,
    /// Whether to use HTTPS. Set to `false` for local S3-compatible services.
    pub use_ssl: bool,
}

/// Serializes a PostgreSQL catalog [`Url`] to the libpq key=value format that
/// DuckLake expects for `postgres:` catalog attachments.
pub(super) fn catalog_conninfo_from_url(catalog_url: &Url) -> EtlResult<String> {
    match catalog_url.scheme() {
        "postgres" | "postgresql" => {}
        scheme => {
            return Err(etl_error!(
                ErrorKind::ConfigError,
                "Unsupported DuckLake catalog URL scheme",
                format!("catalog URL scheme `{scheme}` is not supported")
            ));
        }
    }

    let config = PgConfig::from_str(catalog_url.as_str()).map_err(|e| {
        etl_error!(
            ErrorKind::ConfigError,
            "Invalid DuckLake PostgreSQL catalog URL",
            source: e
        )
    })?;
    let explicit_query_options = explicit_query_options(catalog_url);
    reject_unsupported_query_options(&explicit_query_options)?;

    let mut parts = Vec::new();

    let hosts = serialize_hosts(config.get_hosts())?;
    if !hosts.is_empty() {
        push_conninfo_pair(&mut parts, "host", &hosts);
    }

    let hostaddrs = config
        .get_hostaddrs()
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if !hostaddrs.is_empty() {
        push_conninfo_pair(&mut parts, "hostaddr", &hostaddrs);
    }

    let ports = config
        .get_ports()
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if !ports.is_empty() {
        push_conninfo_pair(&mut parts, "port", &ports);
    }

    if let Some(dbname) = config.get_dbname() {
        push_conninfo_pair(&mut parts, "dbname", dbname);
    }
    if let Some(user) = config.get_user() {
        push_conninfo_pair(&mut parts, "user", user);
    }
    if let Some(password) = config.get_password() {
        let password = std::str::from_utf8(password).map_err(|e| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake PostgreSQL catalog URL contains non-utf8 password",
                source: e
            )
        })?;
        push_conninfo_pair(&mut parts, "password", password);
    }
    if let Some(options) = config.get_options() {
        push_conninfo_pair(&mut parts, "options", options);
    }
    if let Some(application_name) = config.get_application_name() {
        push_conninfo_pair(&mut parts, "application_name", application_name);
    }
    if let Some(ssl_cert) = config.get_ssl_cert() {
        let ssl_cert = std::str::from_utf8(ssl_cert).map_err(|e| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake PostgreSQL catalog URL contains non-utf8 sslcert",
                source: e
            )
        })?;
        push_conninfo_pair(&mut parts, "sslcert", ssl_cert);
    }
    if let Some(ssl_key) = config.get_ssl_key() {
        let ssl_key = std::str::from_utf8(ssl_key).map_err(|e| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake PostgreSQL catalog URL contains non-utf8 sslkey",
                source: e
            )
        })?;
        push_conninfo_pair(&mut parts, "sslkey", ssl_key);
    }
    if let Some(ssl_root_cert) = config.get_ssl_root_cert() {
        let ssl_root_cert = std::str::from_utf8(ssl_root_cert).map_err(|e| {
            etl_error!(
                ErrorKind::ConfigError,
                "DuckLake PostgreSQL catalog URL contains non-utf8 sslrootcert",
                source: e
            )
        })?;
        push_conninfo_pair(&mut parts, "sslrootcert", ssl_root_cert);
    }

    if explicit_query_options.contains("sslmode") {
        parts.push(format!("sslmode={}", ssl_mode_to_str(config.get_ssl_mode())?));
    }

    if explicit_query_options.contains("connect_timeout")
        && let Some(connect_timeout) = config.get_connect_timeout()
    {
        parts.push(format!("connect_timeout={}", connect_timeout.as_secs()));
    }
    if explicit_query_options.contains("tcp_user_timeout")
        && let Some(tcp_user_timeout) = config.get_tcp_user_timeout()
    {
        parts.push(format!("tcp_user_timeout={}", tcp_user_timeout.as_millis()));
    }

    if explicit_query_options.contains("keepalives") {
        parts.push(format!("keepalives={}", if config.get_keepalives() { "1" } else { "0" }));
    }
    if explicit_query_options.contains("keepalives_idle") {
        parts.push(format!("keepalives_idle={}", config.get_keepalives_idle().as_secs()));
    }
    if explicit_query_options.contains("keepalives_interval")
        && let Some(keepalives_interval) = config.get_keepalives_interval()
    {
        parts.push(format!("keepalives_interval={}", keepalives_interval.as_secs()));
    }
    if explicit_query_options.contains("keepalives_retries")
        && let Some(keepalives_retries) = config.get_keepalives_retries()
    {
        parts.push(format!("keepalives_retries={keepalives_retries}"));
    }

    Ok(format!("postgres:{}", parts.join(" ")))
}

fn explicit_query_options(catalog_url: &Url) -> BTreeSet<String> {
    catalog_url.query_pairs().map(|(key, _)| key.into_owned()).collect()
}

fn reject_unsupported_query_options(explicit_query_options: &BTreeSet<String>) -> EtlResult<()> {
    let unsupported =
        ["channel_binding", "load_balance_hosts", "replication", "target_session_attrs"]
            .into_iter()
            .filter(|key| explicit_query_options.contains(*key))
            .collect::<Vec<_>>();

    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(etl_error!(
            ErrorKind::ConfigError,
            "DuckLake PostgreSQL catalog URL uses unsupported query parameters",
            format!("unsupported parameters: {}", unsupported.join(", "))
        ))
    }
}

/// Converts a supported DuckLake catalog URL into the target string used by
/// DuckDB's `ATTACH 'ducklake:...'`.
pub(super) fn catalog_attach_target(catalog_url: &Url) -> EtlResult<String> {
    match catalog_url.scheme() {
        "file" => Ok(catalog_url.as_str().to_owned()),
        "postgres" | "postgresql" => catalog_conninfo_from_url(catalog_url),
        scheme => Err(etl_error!(
            ErrorKind::ConfigError,
            "Unsupported DuckLake catalog URL scheme",
            format!("catalog URL scheme `{scheme}` is not supported")
        )),
    }
}

/// Serializes hosts from [`PgConfig`] into a libpq-compatible host string.
pub(super) fn serialize_hosts(hosts: &[Host]) -> EtlResult<String> {
    let mut values = Vec::with_capacity(hosts.len());
    for host in hosts {
        match host {
            Host::Tcp(host) => values.push(libpq_tcp_host(host).to_owned()),
            #[cfg(unix)]
            Host::Unix(path) => {
                let path = path.to_str().ok_or_else(|| {
                    etl_error!(
                        ErrorKind::ConfigError,
                        "DuckLake PostgreSQL catalog URL contains non-utf8 unix socket path"
                    )
                })?;
                values.push(path.to_owned());
            }
        }
    }

    Ok(values.join(","))
}

/// Adds a `key='value'` pair to a libpq conninfo string.
pub(super) fn push_conninfo_pair(parts: &mut Vec<String>, key: &str, value: &str) {
    parts.push(format!("{key}={}", quote_libpq_conninfo_value(value)));
}

/// Quotes a libpq conninfo value.
///
/// This is not SQL quoting. DuckLake expects the PostgreSQL catalog connection
/// as libpq `key=value` conninfo.
pub(super) fn quote_libpq_conninfo_value(value: &str) -> String {
    let mut quoted = String::from("'");
    for ch in value.chars() {
        if matches!(ch, '\'' | '\\') {
            quoted.push('\\');
        }
        quoted.push(ch);
    }
    quoted.push('\'');
    quoted
}

pub(super) fn ssl_mode_to_str(ssl_mode: SslMode) -> EtlResult<&'static str> {
    match ssl_mode {
        SslMode::Disable => Ok("disable"),
        SslMode::Prefer => Ok("prefer"),
        SslMode::Require => Ok("require"),
        SslMode::VerifyCa => Ok("verify-ca"),
        SslMode::VerifyFull => Ok("verify-full"),
        _ => Err(etl_error!(
            ErrorKind::ConfigError,
            "DuckLake PostgreSQL catalog URL uses an unsupported sslmode"
        )),
    }
}

/// Returns a validated DuckLake data path string.
pub(super) fn validate_data_path(data_path: &Url) -> EtlResult<&str> {
    match data_path.scheme() {
        "file" | "s3" | "gs" => Ok(data_path.as_str()),
        scheme => Err(etl_error!(
            ErrorKind::ConfigError,
            "Unsupported DuckLake data URL scheme",
            format!("data URL scheme `{scheme}` is not supported")
        )),
    }
}

/// Builds the one-time setup SQL executed for each new pool connection.
///
/// On Linux and macOS, required extensions are loaded from vendored local
/// files when a vendored directory is available. Otherwise, DuckDB falls back
/// to the legacy `INSTALL` + `LOAD` flow. The vendored root can be forced with
/// `ETL_DUCKDB_EXTENSION_ROOT`. On Windows, the legacy `INSTALL` + `LOAD` flow
/// is always retained for local development.
#[cfg(test)]
pub(super) fn build_setup_sql(
    catalog_url: &Url,
    data_path: &Url,
    s3: Option<&S3Config>,
    metadata_schema: Option<&str>,
) -> EtlResult<String> {
    Ok(build_setup_plan(
        catalog_url,
        data_path,
        s3,
        metadata_schema,
        &DuckLakeWriterConfig::default(),
        super::ATTACH_DATA_INLINING_ROW_LIMIT,
    )?
    .combined_sql())
}

/// Builds the ordered setup phases executed for each new pool connection.
pub(super) fn build_setup_plan(
    catalog_url: &Url,
    data_path: &Url,
    s3: Option<&S3Config>,
    metadata_schema: Option<&str>,
    writer_config: &DuckLakeWriterConfig,
    data_inlining_row_limit: u64,
) -> EtlResult<DuckLakeSetupPlan> {
    let strategy = current_duckdb_extension_strategy()?;
    let vendored_root = match strategy {
        DuckDbExtensionStrategy::VendoredLocal { platform_dir } => {
            Some(require_vendored_extension_dir(
                platform_dir,
                env::var_os(DUCKDB_EXTENSION_ROOT_ENV_VAR).map(PathBuf::from),
                Path::new(CONTAINER_DUCKDB_EXTENSION_ROOT),
                &repo_vendored_extension_root(),
            )?)
        }
        DuckDbExtensionStrategy::InstallFromRepository => None,
    };
    build_setup_plan_with_strategy(
        catalog_url,
        data_path,
        s3,
        metadata_schema,
        writer_config,
        DuckDbExtensionSetup { strategy, vendored_root: vendored_root.as_deref() },
        data_inlining_row_limit,
    )
}

#[cfg(test)]
fn build_setup_sql_with_strategy(
    catalog_url: &Url,
    data_path: &Url,
    s3: Option<&S3Config>,
    metadata_schema: Option<&str>,
    strategy: DuckDbExtensionStrategy,
    vendored_root: Option<&Path>,
) -> EtlResult<String> {
    Ok(build_setup_plan_with_strategy(
        catalog_url,
        data_path,
        s3,
        metadata_schema,
        &DuckLakeWriterConfig::default(),
        DuckDbExtensionSetup { strategy, vendored_root },
        super::ATTACH_DATA_INLINING_ROW_LIMIT,
    )?
    .combined_sql())
}

fn build_setup_plan_with_strategy(
    catalog_url: &Url,
    data_path: &Url,
    s3: Option<&S3Config>,
    metadata_schema: Option<&str>,
    writer_config: &DuckLakeWriterConfig,
    extension_setup: DuckDbExtensionSetup<'_>,
    data_inlining_row_limit: u64,
) -> EtlResult<DuckLakeSetupPlan> {
    let catalog_target = catalog_attach_target(catalog_url)?;
    let data_path = validate_data_path(data_path)?;

    let needs_postgres = matches!(catalog_url.scheme(), "postgres" | "postgresql");
    let needs_httpfs = matches!(data_path.split(':').next(), Some("s3" | "gs"));
    let lake_catalog = quote_identifier(LAKE_CATALOG);
    let mut steps = vec![DuckLakeSetupStep {
        label: "configure_writer_session",
        sql: configure_writer_session_sql(),
    }];
    let mut secret_options = BTreeMap::from([
        ("KEY_ID", quote_literal(s3.map(|s| s.access_key_id.as_str()).unwrap_or_default())),
        ("REGION", quote_literal(s3.map(|s| s.region.as_str()).unwrap_or_default())),
        ("SECRET", quote_literal(s3.map(|s| s.secret_access_key.as_str()).unwrap_or_default())),
        ("SCOPE", quote_literal(data_path)),
        ("URL_COMPATIBILITY_MODE", "true".to_owned()),
        ("URL_STYLE", quote_literal(s3.map(|s| s.url_style.as_str()).unwrap_or_default())),
    ]);

    let extension_sql = match extension_setup.strategy {
        DuckDbExtensionStrategy::VendoredLocal { .. } => {
            let extension_root = extension_setup.vendored_root.ok_or_else(|| {
                etl_error!(
                    ErrorKind::ConfigError,
                    "Vendored DuckDB extensions not found",
                    "Missing vendored DuckDB extension root"
                )
            })?;
            let ducklake_extension =
                vendored_extension_path(extension_root, DUCKLAKE_EXTENSION_FILE)?;
            let mut sql = format!("LOAD {};", quote_literal(&ducklake_extension));
            if needs_postgres {
                let postgres_extension =
                    vendored_extension_path(extension_root, POSTGRES_SCANNER_EXTENSION_FILE)?;
                sql.push_str(&format!(" LOAD {};", quote_literal(&postgres_extension)));
            }
            if needs_httpfs {
                let httpfs_extension =
                    vendored_extension_path(extension_root, HTTPFS_EXTENSION_FILE)?;
                sql.push_str(&format!(" LOAD {};", quote_literal(&httpfs_extension)));
            }
            sql
        }
        DuckDbExtensionStrategy::InstallFromRepository => {
            let mut sql = String::from("INSTALL ducklake; LOAD ducklake;");
            if needs_postgres {
                sql.push_str(" INSTALL postgres; LOAD postgres;");
            }
            if needs_httpfs {
                sql.push_str(" INSTALL httpfs; LOAD httpfs;");
            }
            sql
        }
    };
    steps.push(DuckLakeSetupStep { label: "load_extensions", sql: extension_sql });

    if needs_httpfs && let Some(s3) = s3 {
        let secret_name = quote_identifier("ducklake_s3");
        if let Some(endpoint) = &s3.endpoint {
            secret_options.insert("ENDPOINT", quote_literal(endpoint));
        }

        let secret_body = secret_options
            .iter()
            .map(|(key, value)| format!("{key} {value}"))
            .collect::<Vec<_>>()
            .join(", ");

        steps.push(DuckLakeSetupStep {
            label: "configure_object_store",
            sql: format!(
                "SET enable_http_metadata_cache = true; SET parquet_metadata_cache = true; CREATE \
                 OR REPLACE SECRET {secret_name} (TYPE S3, PROVIDER config, {secret_body}, \
                 USE_SSL {});",
                if s3.use_ssl { "true" } else { "false" }
            ),
        });
    }
    let metadata_schema_clause = metadata_schema
        .map(|schema| format!(", METADATA_SCHEMA {}", quote_literal(schema)))
        .unwrap_or_default();

    steps.push(DuckLakeSetupStep {
        label: "attach_catalog",
        sql: format!(
            "ATTACH {} AS {lake_catalog} (DATA_PATH {}, DATA_INLINING_ROW_LIMIT {}, \
             AUTOMATIC_MIGRATION true{metadata_schema_clause});",
            quote_literal(&format!("ducklake:{catalog_target}")),
            quote_literal(data_path),
            data_inlining_row_limit
        ),
    });
    steps.push(DuckLakeSetupStep {
        label: "configure_parquet",
        sql: configure_parquet_settings_sql(writer_config),
    });

    Ok(DuckLakeSetupPlan { steps })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn vendored_root_with_files(platform_dir: &str) -> TempDir {
        let tempdir = TempDir::new().unwrap();
        let extension_dir = tempdir.path().join(DUCKDB_EXTENSION_VERSION).join(platform_dir);
        fs::create_dir_all(&extension_dir).unwrap();
        for filename in
            [DUCKLAKE_EXTENSION_FILE, HTTPFS_EXTENSION_FILE, POSTGRES_SCANNER_EXTENSION_FILE]
        {
            fs::write(extension_dir.join(filename), []).unwrap();
        }

        tempdir
    }

    #[test]
    fn catalog_conninfo_from_file_url() {
        let catalog_url = Url::from_file_path("/tmp/catalog.ducklake").unwrap();
        assert_eq!(catalog_attach_target(&catalog_url).unwrap(), catalog_url.as_str());
    }

    #[test]
    fn quote_libpq_conninfo_value_test() {
        assert_eq!(quote_libpq_conninfo_value("pa'ss\\word"), "'pa\\'ss\\\\word'");
        assert_eq!(quote_libpq_conninfo_value("value with spaces"), "'value with spaces'");
    }

    #[test]
    fn catalog_conninfo_from_postgres_url_round_trip() {
        let url = Url::parse("postgres://bnj@localhost:5432/ducklake_catalog").unwrap();
        let conninfo = catalog_conninfo_from_url(&url).unwrap();
        let parsed = PgConfig::from_str(conninfo.strip_prefix("postgres:").unwrap()).unwrap();

        assert!(!conninfo.contains("load_balance_hosts"));
        assert!(!conninfo.contains("channel_binding"));
        assert!(!conninfo.contains("target_session_attrs"));
        assert!(!conninfo.contains("sslmode="));
        assert!(!conninfo.contains("keepalives="));
        assert_eq!(parsed.get_user(), Some("bnj"));
        assert_eq!(parsed.get_dbname(), Some("ducklake_catalog"));
        assert_eq!(parsed.get_ports(), &[5432]);
        assert_eq!(serialize_hosts(parsed.get_hosts()).unwrap(), "localhost");
    }

    #[test]
    fn catalog_conninfo_from_postgres_ipv6_url_uses_libpq_host_syntax() {
        let url = Url::parse(
            "postgres://user@[2406:da18:1d63:9b01:df66:7be6:5158:2151]:5432/postgres?\
             sslmode=prefer",
        )
        .unwrap();
        let conninfo = catalog_conninfo_from_url(&url).unwrap();
        let parsed = PgConfig::from_str(conninfo.strip_prefix("postgres:").unwrap()).unwrap();

        assert!(!conninfo.contains("[2406:da18"));
        assert_eq!(
            serialize_hosts(parsed.get_hosts()).unwrap(),
            "2406:da18:1d63:9b01:df66:7be6:5158:2151"
        );
        assert_eq!(parsed.get_ports(), &[5432]);
        assert_eq!(parsed.get_ssl_mode(), SslMode::Prefer);
    }

    #[test]
    fn catalog_conninfo_from_postgres_ipv6_url_with_password_matches_expected_shape() {
        let url = Url::parse(
            "postgres://postgres:debug-password@[2406:da18:1d63:9b01:7f24:6b65:7aa:cac8]:5432/\
             postgres?sslmode=prefer",
        )
        .unwrap();
        let conninfo = catalog_conninfo_from_url(&url).unwrap();

        assert_eq!(
            conninfo,
            "postgres:host='2406:da18:1d63:9b01:7f24:6b65:7aa:cac8' port='5432' dbname='postgres' \
             user='postgres' password='debug-password' sslmode=prefer"
        );
    }

    #[test]
    fn sqlx_connect_options_from_postgres_ipv6_url_keeps_bracketed_host() {
        let options = sqlx::postgres::PgConnectOptions::from_str(
            "postgres://postgres:debug-password@[2406:da18:1d63:9b01:7f24:6b65:7aa:cac8]:5432/\
             postgres?sslmode=prefer",
        )
        .unwrap();

        assert_eq!(options.get_host(), "[2406:da18:1d63:9b01:7f24:6b65:7aa:cac8]");
        assert_eq!(options.get_port(), 5432);
        assert_eq!(options.get_database(), Some("postgres"));
        assert_eq!(options.get_username(), "postgres");
        assert!(matches!(options.get_ssl_mode(), sqlx::postgres::PgSslMode::Prefer));
    }

    #[test]
    fn catalog_conninfo_from_postgres_url_with_password_and_query_params_round_trip() {
        let url = Url::parse(
            "postgres://user:pa%27ss%5Cword@localhost:5433/mydb?sslmode=disable&\
             connect_timeout=10&tcp_user_timeout=1500&application_name=myapp",
        )
        .unwrap();
        let conninfo = catalog_conninfo_from_url(&url).unwrap();
        let parsed = PgConfig::from_str(conninfo.strip_prefix("postgres:").unwrap()).unwrap();

        assert_eq!(parsed.get_user(), Some("user"));
        assert_eq!(parsed.get_dbname(), Some("mydb"));
        assert_eq!(parsed.get_ports(), &[5433]);
        assert_eq!(std::str::from_utf8(parsed.get_password().unwrap()).unwrap(), "pa'ss\\word");
        assert_eq!(parsed.get_ssl_mode(), SslMode::Disable);
        assert_eq!(parsed.get_connect_timeout().unwrap().as_secs(), 10);
        assert!(conninfo.contains("tcp_user_timeout=1500"));
        assert_eq!(parsed.get_application_name(), Some("myapp"));
    }

    #[test]
    fn catalog_conninfo_rejects_explicit_unsupported_query_parameters() {
        for parameter in [
            "channel_binding=require",
            "load_balance_hosts=random",
            "replication=database",
            "target_session_attrs=read-write",
        ] {
            let url = Url::parse(&format!("postgres://user@localhost/mydb?{parameter}")).unwrap();
            let err = catalog_conninfo_from_url(&url).unwrap_err();

            assert_eq!(err.kind(), ErrorKind::ConfigError);
            assert!(err.to_string().contains(parameter.split('=').next().unwrap()));
        }
    }

    #[test]
    fn catalog_conninfo_rejects_unsupported_catalog_scheme() {
        let err =
            catalog_attach_target(&Url::parse("https://example.com/catalog").unwrap()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigError);
    }

    #[test]
    fn validate_data_path_rejects_unsupported_scheme() {
        let err = build_setup_sql(
            &Url::from_file_path("/tmp/catalog.ducklake").unwrap(),
            &Url::parse("https://example.com/lake").unwrap(),
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigError);

        let err = build_setup_sql(
            &Url::from_file_path("/tmp/catalog.ducklake").unwrap(),
            &Url::parse("az://container/lake").unwrap(),
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigError);
    }

    #[test]
    fn duckdb_extension_strategy_linux_amd64_uses_install_flow_without_vendored_extensions() {
        let tempdir = TempDir::new().unwrap();

        assert_eq!(
            duckdb_extension_strategy("linux", "x86_64", None, tempdir.path(), tempdir.path())
                .unwrap(),
            DuckDbExtensionStrategy::InstallFromRepository
        );
    }

    #[test]
    fn duckdb_extension_strategy_linux_arm64_uses_vendored_extensions_when_present() {
        let repo_root = vendored_root_with_files("linux_arm64");
        let container_root = TempDir::new().unwrap();

        assert_eq!(
            duckdb_extension_strategy(
                "linux",
                "aarch64",
                None,
                container_root.path(),
                repo_root.path(),
            )
            .unwrap(),
            DuckDbExtensionStrategy::VendoredLocal { platform_dir: "linux_arm64" }
        );
    }

    #[test]
    fn duckdb_extension_strategy_macos_uses_install_flow_without_vendored_extensions() {
        let tempdir = TempDir::new().unwrap();

        assert_eq!(
            duckdb_extension_strategy("macos", "aarch64", None, tempdir.path(), tempdir.path())
                .unwrap(),
            DuckDbExtensionStrategy::InstallFromRepository
        );
    }

    #[test]
    fn duckdb_extension_strategy_macos_arm64_uses_vendored_extensions_when_present() {
        let repo_root = vendored_root_with_files("osx_arm64");
        let container_root = TempDir::new().unwrap();

        assert_eq!(
            duckdb_extension_strategy(
                "macos",
                "aarch64",
                None,
                container_root.path(),
                repo_root.path(),
            )
            .unwrap(),
            DuckDbExtensionStrategy::VendoredLocal { platform_dir: "osx_arm64" }
        );
    }

    #[test]
    fn duckdb_extension_strategy_windows_uses_install_flow() {
        let tempdir = TempDir::new().unwrap();

        assert_eq!(
            duckdb_extension_strategy("windows", "x86_64", None, tempdir.path(), tempdir.path())
                .unwrap(),
            DuckDbExtensionStrategy::InstallFromRepository
        );
    }

    #[test]
    fn duckdb_extension_strategy_rejects_unsupported_linux_architecture() {
        let tempdir = TempDir::new().unwrap();
        let err =
            duckdb_extension_strategy("linux", "riscv64", None, tempdir.path(), tempdir.path())
                .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigError);
    }

    #[test]
    fn vendored_extension_dir_uses_env_override_first() {
        let env_root = vendored_root_with_files("linux_amd64");
        let fallback_root = TempDir::new().unwrap();

        let resolved = vendored_extension_dir(
            "linux_amd64",
            Some(env_root.path().to_path_buf()),
            fallback_root.path(),
            fallback_root.path(),
        )
        .unwrap();

        assert_eq!(
            resolved.unwrap(),
            env_root.path().join(DUCKDB_EXTENSION_VERSION).join("linux_amd64")
        );
    }

    #[test]
    fn vendored_extension_dir_falls_back_to_repo_root() {
        let repo_root = vendored_root_with_files("linux_arm64");
        let container_root = TempDir::new().unwrap();

        let resolved =
            vendored_extension_dir("linux_arm64", None, container_root.path(), repo_root.path())
                .unwrap();

        assert_eq!(
            resolved.unwrap(),
            repo_root.path().join(DUCKDB_EXTENSION_VERSION).join("linux_arm64")
        );
    }

    #[test]
    fn vendored_extension_dir_falls_back_to_container_root() {
        let container_root = vendored_root_with_files("linux_amd64");
        let repo_root = TempDir::new().unwrap();

        let resolved =
            vendored_extension_dir("linux_amd64", None, container_root.path(), repo_root.path())
                .unwrap();

        assert_eq!(
            resolved.unwrap(),
            container_root.path().join(DUCKDB_EXTENSION_VERSION).join("linux_amd64")
        );
    }

    #[test]
    fn vendored_extension_dir_rejects_missing_extensions() {
        let tempdir = TempDir::new().unwrap();
        let err = vendored_extension_dir(
            "linux_amd64",
            Some(tempdir.path().to_path_buf()),
            tempdir.path(),
            tempdir.path(),
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigError);
    }

    #[test]
    fn vendored_extension_dir_returns_none_when_not_present() {
        let tempdir = TempDir::new().unwrap();
        let resolved =
            vendored_extension_dir("linux_amd64", None, tempdir.path(), tempdir.path()).unwrap();

        assert!(resolved.is_none());
    }

    #[test]
    fn build_setup_sql_linux_vendored_local() {
        let vendored_root = vendored_root_with_files("linux_amd64");
        let catalog_url = Url::from_file_path("/tmp/catalog.ducklake").unwrap();
        let data_url = Url::from_file_path("/tmp/lake_data").unwrap();
        let sql = build_setup_sql_with_strategy(
            &catalog_url,
            &data_url,
            None,
            None,
            DuckDbExtensionStrategy::VendoredLocal { platform_dir: "linux_amd64" },
            Some(vendored_root.path().join(DUCKDB_EXTENSION_VERSION).join("linux_amd64").as_path()),
        )
        .unwrap();

        assert!(!sql.contains("INSTALL"));
        assert!(sql.contains("LOAD '/"));
        assert!(sql.contains(DUCKLAKE_EXTENSION_FILE));
        assert!(!sql.contains("postgres"));
        assert!(!sql.contains("httpfs"));
        assert!(!sql.contains("json"));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_COMPRESSION_OPTION_NAME),
            quote_literal(PARQUET_COMPRESSION_OPTION_VALUE),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_ROW_GROUP_SIZE_BYTES_OPTION_NAME),
            quote_literal(DuckLakeWriterConfig::default().parquet_row_group_size_bytes()),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_VERSION_OPTION_NAME),
            PARQUET_VERSION_OPTION_VALUE,
        )));
        assert!(sql.contains(&quote_literal(&format!("ducklake:{}", catalog_url.as_str()))));
        assert!(sql.contains(&format!("DATA_PATH {}", quote_literal(data_url.as_str()))));
        assert!(sql.contains(&format!(
            "DATA_INLINING_ROW_LIMIT {}",
            crate::ducklake::ATTACH_DATA_INLINING_ROW_LIMIT
        )));
        assert!(sql.contains("AUTOMATIC_MIGRATION true"));
    }

    #[test]
    fn build_setup_sql_macos_vendored_local() {
        let vendored_root = vendored_root_with_files("osx_arm64");
        let catalog_url = Url::from_file_path("/tmp/catalog.ducklake").unwrap();
        let data_url = Url::from_file_path("/tmp/lake_data").unwrap();
        let extension_dir = vendored_root.path().join(DUCKDB_EXTENSION_VERSION).join("osx_arm64");
        let sql = build_setup_sql_with_strategy(
            &catalog_url,
            &data_url,
            None,
            None,
            DuckDbExtensionStrategy::VendoredLocal { platform_dir: "osx_arm64" },
            Some(&extension_dir),
        )
        .unwrap();

        assert!(!sql.contains("INSTALL"));
        assert!(sql.contains("LOAD '/"));
        assert!(sql.contains(DUCKLAKE_EXTENSION_FILE));
        assert!(!sql.contains("postgres"));
        assert!(!sql.contains("httpfs"));
        assert!(!sql.contains("json"));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_COMPRESSION_OPTION_NAME),
            quote_literal(PARQUET_COMPRESSION_OPTION_VALUE),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_ROW_GROUP_SIZE_BYTES_OPTION_NAME),
            quote_literal(DuckLakeWriterConfig::default().parquet_row_group_size_bytes()),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_VERSION_OPTION_NAME),
            PARQUET_VERSION_OPTION_VALUE,
        )));
    }

    #[test]
    fn build_setup_sql_legacy_install_postgres_local_data() {
        let catalog_url = Url::parse("postgres://bnj@localhost:5432/ducklake_catalog").unwrap();
        let data_url = Url::from_file_path("/tmp/lake_data").unwrap();
        let sql = build_setup_sql_with_strategy(
            &catalog_url,
            &data_url,
            None,
            None,
            DuckDbExtensionStrategy::InstallFromRepository,
            None,
        )
        .unwrap();

        assert!(sql.contains("INSTALL ducklake"));
        assert!(sql.contains("LOAD ducklake"));
        assert!(sql.contains("INSTALL postgres"));
        assert!(sql.contains("LOAD postgres"));
        assert!(!sql.contains("httpfs"));
        assert!(!sql.contains("json"));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_COMPRESSION_OPTION_NAME),
            quote_literal(PARQUET_COMPRESSION_OPTION_VALUE),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_ROW_GROUP_SIZE_BYTES_OPTION_NAME),
            quote_literal(DuckLakeWriterConfig::default().parquet_row_group_size_bytes()),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_VERSION_OPTION_NAME),
            PARQUET_VERSION_OPTION_VALUE,
        )));
        assert!(!sql.contains("ducklake:postgres://"));
        assert!(sql.contains("ducklake:postgres:"));
        assert!(sql.contains(&format!("DATA_PATH {}", quote_literal(data_url.as_str()))));
        assert!(sql.contains(&format!(
            "DATA_INLINING_ROW_LIMIT {}",
            crate::ducklake::ATTACH_DATA_INLINING_ROW_LIMIT
        )));
        assert!(sql.contains("AUTOMATIC_MIGRATION true"));
    }

    #[test]
    fn build_setup_sql_linux_vendored_postgres_s3_data() {
        let vendored_root = vendored_root_with_files("linux_arm64");
        let catalog_url = Url::parse("postgres://user:pass@host/db").unwrap();
        let data_url = Url::parse("s3://my-bucket/lake/").unwrap();
        let extension_dir = vendored_root.path().join(DUCKDB_EXTENSION_VERSION).join("linux_arm64");
        let sql = build_setup_sql_with_strategy(
            &catalog_url,
            &data_url,
            None,
            None,
            DuckDbExtensionStrategy::VendoredLocal { platform_dir: "linux_arm64" },
            Some(&extension_dir),
        )
        .unwrap();

        assert!(!sql.contains("INSTALL"));
        assert!(sql.contains(DUCKLAKE_EXTENSION_FILE));
        assert!(sql.contains(POSTGRES_SCANNER_EXTENSION_FILE));
        assert!(sql.contains(HTTPFS_EXTENSION_FILE));
        assert!(!sql.contains("json"));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_COMPRESSION_OPTION_NAME),
            quote_literal(PARQUET_COMPRESSION_OPTION_VALUE),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_ROW_GROUP_SIZE_BYTES_OPTION_NAME),
            quote_literal(DuckLakeWriterConfig::default().parquet_row_group_size_bytes()),
        )));
        assert!(sql.contains(&format!(
            "CALL {LAKE_CATALOG}.set_option({}, {});",
            quote_literal(PARQUET_VERSION_OPTION_NAME),
            PARQUET_VERSION_OPTION_VALUE,
        )));
        assert!(!sql.contains("ducklake:postgres://"));
        assert!(sql.contains(&format!("DATA_PATH {}", quote_literal(data_url.as_str()))));
        assert!(sql.contains(&format!(
            "DATA_INLINING_ROW_LIMIT {}",
            crate::ducklake::ATTACH_DATA_INLINING_ROW_LIMIT
        )));
        assert!(sql.contains("AUTOMATIC_MIGRATION true"));
    }

    #[test]
    fn build_setup_plan_linux_vendored_postgres_s3_data_has_expected_phases() {
        let vendored_root = vendored_root_with_files("linux_arm64");
        let catalog_url = Url::parse("postgres://user:pass@host/db").unwrap();
        let data_url = Url::parse("s3://my-bucket/lake/").unwrap();
        let s3 = S3Config {
            access_key_id: "key".to_owned(),
            secret_access_key: "secret".to_owned(),
            region: "eu-west-3".to_owned(),
            endpoint: Some("localhost:9000".to_owned()),
            url_style: "path".to_owned(),
            use_ssl: false,
        };
        let extension_dir = vendored_root.path().join(DUCKDB_EXTENSION_VERSION).join("linux_arm64");
        let plan = build_setup_plan_with_strategy(
            &catalog_url,
            &data_url,
            Some(&s3),
            Some("ducklake"),
            &DuckLakeWriterConfig::default(),
            DuckDbExtensionSetup {
                strategy: DuckDbExtensionStrategy::VendoredLocal { platform_dir: "linux_arm64" },
                vendored_root: Some(&extension_dir),
            },
            crate::ducklake::COPY_DATA_INLINING_ROW_LIMIT,
        )
        .unwrap();

        assert_eq!(plan.steps().len(), 5);
        assert_eq!(plan.steps()[0].label, "configure_writer_session");
        assert_eq!(plan.steps()[1].label, "load_extensions");
        assert_eq!(plan.steps()[2].label, "configure_object_store");
        assert_eq!(plan.steps()[3].label, "attach_catalog");
        assert_eq!(plan.steps()[4].label, "configure_parquet");
        assert!(plan.steps()[3].sql.contains("DATA_INLINING_ROW_LIMIT 0"));
        assert_eq!(plan.steps()[0].sql, configure_writer_session_sql());
        assert!(plan.steps()[1].sql.contains(HTTPFS_EXTENSION_FILE));
        assert!(!plan.steps()[1].sql.contains("json"));
        assert!(plan.steps()[2].sql.contains("CREATE OR REPLACE SECRET"));
        assert!(plan.steps()[2].sql.contains("PROVIDER config"));
        assert!(
            plan.steps()[2].sql.contains(&format!("SCOPE {}", quote_literal(data_url.as_str())))
        );
        assert!(plan.steps()[2].sql.contains("URL_COMPATIBILITY_MODE true"));
        assert!(plan.steps()[3].sql.contains("ATTACH"));
        assert!(plan.steps()[3].sql.contains("METADATA_SCHEMA 'ducklake'"));
        assert_eq!(
            plan.steps()[4].sql,
            configure_parquet_settings_sql(&DuckLakeWriterConfig::default())
        );
    }

    #[test]
    fn ducklake_writer_settings_use_recommended_defaults() {
        let writer_config = DuckLakeWriterConfig::default();
        let parquet_sql = configure_parquet_settings_sql(&writer_config);

        assert!(parquet_sql.contains("'parquet_row_group_size_bytes', '128MiB'"));
        assert!(parquet_sql.contains("'parquet_row_group_size', '2500000'"));
        assert_eq!(
            maintenance_target_file_size_sql(writer_config.target_file_size()),
            "CALL ducklake_set_option('lake', 'target_file_size', '256MiB');"
        );
    }

    #[test]
    fn ducklake_writer_settings_use_configured_overrides() {
        let writer_config = DuckLakeWriterConfig::new(
            Some("64MB".to_owned()),
            Some("32MB".to_owned()),
            Some("500000".to_owned()),
        );
        let parquet_sql = configure_parquet_settings_sql(&writer_config);

        assert!(parquet_sql.contains("'parquet_row_group_size_bytes', '32MB'"));
        assert!(parquet_sql.contains("'parquet_row_group_size', '500000'"));
        assert_eq!(
            maintenance_target_file_size_sql(writer_config.target_file_size()),
            "CALL ducklake_set_option('lake', 'target_file_size', '64MB');"
        );
    }

    #[test]
    fn resolve_expire_snapshots_older_than_uses_default() {
        assert_eq!(resolve_expire_snapshots_older_than(None), EXPIRE_SNAPSHOTS_OLDER_THAN);
        assert_eq!(resolve_expire_snapshots_older_than(Some("2 days")), "2 days");
    }

    /// Checks actual elapsed retention across calendar and mixed-sign
    /// intervals.
    #[test]
    fn validate_expire_snapshots_older_than_sql_checks_minimum_retention() {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch("create macro now() as timestamp '2026-03-01 12:00:00';").unwrap();

        for (retention, expected) in [
            ("1 day", true),
            ("24 hours", true),
            ("7 days", true),
            ("1 month", true),
            ("1 year", true),
            ("2 days -12 hours", true),
            ("0 seconds", false),
            ("-1 day", false),
            ("23 hours", false),
            ("1 day -1 microsecond", false),
            ("-1 year 361 days", false),
            ("1 month -29 days", false),
        ] {
            let sql = validate_expire_snapshots_older_than_sql(retention);
            let actual: bool = conn.query_row(&sql, [], |row| row.get(0)).unwrap();
            assert_eq!(actual, expected, "retention: {retention}");
        }

        for retention in ["", "not an interval", "1 day'"] {
            let sql = validate_expire_snapshots_older_than_sql(retention);
            assert!(conn.query_row(&sql, [], |row| row.get::<_, bool>(0)).is_err());
        }
    }

    #[test]
    fn vendored_extension_strategy_disables_autoload() {
        assert!(
            DuckDbExtensionStrategy::VendoredLocal { platform_dir: "osx_arm64" }
                .disables_autoload()
        );
        assert!(!DuckDbExtensionStrategy::InstallFromRepository.disables_autoload());
    }

    #[test]
    fn build_setup_sql_uses_pg_escape_for_literals() {
        let catalog_url = Url::parse("postgres://user:pa%27ss%5Cword@localhost:5432/mydb").unwrap();
        let data_url = Url::parse("s3://bucket/path").unwrap();
        let s3 = S3Config {
            access_key_id: "key'with".to_owned(),
            secret_access_key: "sec\\ret".to_owned(),
            region: "eu-west-1".to_owned(),
            endpoint: Some("localhost:9000/o'hare".to_owned()),
            url_style: "path".to_owned(),
            use_ssl: false,
        };
        let metadata_schema = "duck'lake";
        let sql =
            build_setup_sql(&catalog_url, &data_url, Some(&s3), Some(metadata_schema)).unwrap();

        assert!(sql.contains(&format!("KEY_ID {}", quote_literal(&s3.access_key_id))));
        assert!(sql.contains(&format!("SECRET {}", quote_literal(&s3.secret_access_key))));
        assert!(sql.contains(&format!("SCOPE {}", quote_literal(data_url.as_str()))));
        assert!(
            sql.contains(&format!("ENDPOINT {}", quote_literal(s3.endpoint.as_deref().unwrap())))
        );
        assert!(sql.contains(&format!("METADATA_SCHEMA {}", quote_literal(metadata_schema))));
    }
}
