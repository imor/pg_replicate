#[cfg(feature = "test-utils")]
use std::sync::atomic::AtomicUsize;
use std::{
    borrow::Cow,
    error, fmt, process,
    sync::{
        Arc, LazyLock, Mutex, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Duration,
};

use duckdb::Config;
use etl::{
    error::{ErrorKind, EtlError, EtlResult},
    etl_error,
};
use metrics::histogram;
use regex::Regex;
use tokio::{
    sync::{Semaphore, oneshot},
    task::JoinHandle,
    time::Instant,
};
use tracing::{info, trace, warn};

use crate::ducklake::{
    config::{DuckLakeSetupPlan, DuckLakeSetupStep},
    embedding::ConnectionInitializer,
    metrics::{
        ETL_DUCKLAKE_BLOCKING_OPERATION_DURATION_SECONDS, ETL_DUCKLAKE_BLOCKING_SLOT_WAIT_SECONDS,
        ETL_DUCKLAKE_POOL_CHECKOUT_WAIT_SECONDS,
    },
};

/// Monotonic identifier assigned to each DuckLake connection setup attempt.
static NEXT_CONNECTION_INIT_ID: AtomicU64 = AtomicU64::new(1);
/// Monotonic identifier assigned to each timed DuckDB blocking operation.
static NEXT_DUCKDB_BLOCKING_OPERATION_ID: AtomicU64 = AtomicU64::new(1);
/// Matches one libpq password field inside a conninfo string.
static POSTGRES_PASSWORD_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"password=(?:'([^'\\]|\\.)*'|[^\s,);]+)")
        .expect("postgres password redaction regex should compile")
});

/// Timeout applied to each foreground DuckLake blocking operation.
pub(super) const FOREGROUND_QUERY_TIMEOUT: Duration = Duration::from_secs(3 * 60);
/// Stable log and error label for DuckDB blocking operations.
const DUCKDB_BLOCKING_OPERATION_KIND: &str = "foreground";
/// Extra time allowed for a timed-out DuckDB operation to return after
/// interrupt() has been called. If the operation is still stuck after this,
/// the process is no longer safe to keep running.
const BLOCKING_ABORT_GRACE: Duration = Duration::from_secs(30);
/// Description used when DuckLake rejects new blocking work during shutdown.
const DUCKLAKE_SHUTDOWN_REQUESTED: &str = "DuckLake shutdown requested";

/// Reason recorded for the first interrupt sent to a DuckLake connection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DuckLakeInterruptReason {
    None = 0,
    QueryTimeout = 1,
    DestinationShutdown = 2,
    ProcessShutdown = 3,
    ExternalInterrupt = 4,
}

impl DuckLakeInterruptReason {
    /// Returns the stable label used in logs and error details.
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::QueryTimeout => "query_timeout",
            Self::DestinationShutdown => "destination_shutdown",
            Self::ProcessShutdown => "process_shutdown",
            Self::ExternalInterrupt => "external_interrupt",
        }
    }

    /// Converts one stored atomic value back into an interrupt reason.
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::QueryTimeout,
            2 => Self::DestinationShutdown,
            3 => Self::ProcessShutdown,
            4 => Self::ExternalInterrupt,
            _ => Self::None,
        }
    }
}

/// Shared interrupt state for one managed DuckDB connection.
#[derive(Default)]
struct DuckLakeInterruptState {
    reason: AtomicU8,
}

impl DuckLakeInterruptState {
    /// Clears the previously recorded interrupt reason before a new query.
    fn clear(&self) {
        self.reason.store(DuckLakeInterruptReason::None as u8, Ordering::Relaxed);
    }

    /// Records the first reason that interrupted the current query.
    fn record(&self, reason: DuckLakeInterruptReason) {
        let _ = self.reason.compare_exchange(
            DuckLakeInterruptReason::None as u8,
            reason as u8,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Returns the currently recorded interrupt reason.
    fn reason(&self) -> DuckLakeInterruptReason {
        DuckLakeInterruptReason::from_u8(self.reason.load(Ordering::Relaxed))
    }
}

/// Context for one DuckDB blocking operation.
pub(super) struct DuckLakeBlockingOperationContext {
    operation_id: u64,
    operation_kind: &'static str,
    timeout: Duration,
    interrupt_state: Arc<DuckLakeInterruptState>,
}

impl DuckLakeBlockingOperationContext {
    /// Creates context for one blocking DuckDB operation.
    fn new(
        operation_id: u64,
        operation_kind: &'static str,
        timeout: Duration,
        interrupt_state: Arc<DuckLakeInterruptState>,
    ) -> Self {
        Self { operation_id, operation_kind, timeout, interrupt_state }
    }

    /// Returns the operation id assigned to the blocking DuckDB operation.
    pub(super) fn operation_id(&self) -> u64 {
        self.operation_id
    }

    /// Returns the stable operation-kind label.
    pub(super) fn operation_kind(&self) -> &'static str {
        self.operation_kind
    }

    /// Returns the operation timeout in milliseconds.
    pub(super) fn timeout_ms(&self) -> u64 {
        self.timeout.as_millis() as u64
    }

    /// Returns the currently recorded interrupt reason label.
    pub(super) fn interrupt_reason_label(&self) -> &'static str {
        self.interrupt_state.reason().as_str()
    }

    /// Creates a minimal context for unit tests that call batch helpers
    /// directly.
    #[cfg(test)]
    pub(super) fn for_tests() -> Self {
        Self::new(
            0,
            DUCKDB_BLOCKING_OPERATION_KIND,
            Duration::ZERO,
            Arc::new(DuckLakeInterruptState::default()),
        )
    }
}

trait DuckDbQueryInterrupt: Send + Sync + 'static {
    fn interrupt(&self);
}

impl DuckDbQueryInterrupt for duckdb::InterruptHandle {
    fn interrupt(&self) {
        duckdb::InterruptHandle::interrupt(self);
    }
}

type DuckDbQueryInterruptHandle = Arc<dyn DuckDbQueryInterrupt>;

/// Interrupt handle registered for one managed DuckLake connection.
struct RegisteredDuckLakeInterrupt {
    handle: Arc<duckdb::InterruptHandle>,
    state: Arc<DuckLakeInterruptState>,
}

impl RegisteredDuckLakeInterrupt {
    /// Creates a registered interrupt handle for one DuckDB connection.
    fn new(handle: Arc<duckdb::InterruptHandle>) -> Self {
        Self { handle, state: Arc::new(DuckLakeInterruptState::default()) }
    }

    /// Clears the previously recorded interrupt reason before a new query.
    fn clear_reason(&self) {
        self.state.clear();
    }

    /// Returns the shared interrupt state for context passed to callers.
    fn interrupt_state(&self) -> Arc<DuckLakeInterruptState> {
        Arc::clone(&self.state)
    }

    /// Interrupts the connection and records why the interrupt was sent.
    fn interrupt_with_reason(&self, reason: DuckLakeInterruptReason) {
        self.state.record(reason);
        self.handle.interrupt();
    }
}

impl DuckDbQueryInterrupt for RegisteredDuckLakeInterrupt {
    fn interrupt(&self) {
        self.interrupt_with_reason(DuckLakeInterruptReason::QueryTimeout);
    }
}

/// Async watchdog that interrupts one timed DuckDB query when its deadline
/// expires.
pub(super) struct DuckDbQueryWatchdog {
    timed_out: Arc<AtomicBool>,
    interrupt_tx: Option<oneshot::Sender<DuckDbQueryInterruptHandle>>,
    done_tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl DuckDbQueryWatchdog {
    fn spawn(deadline: Instant) -> Self {
        let timed_out = Arc::new(AtomicBool::new(false));
        let timeout_flag = Arc::clone(&timed_out);
        let (interrupt_tx, interrupt_rx) = oneshot::channel::<DuckDbQueryInterruptHandle>();
        let (done_tx, done_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut interrupt_rx = Box::pin(interrupt_rx);
            let mut done_rx = Box::pin(done_rx);
            let interrupt_handle = tokio::select! {
                biased;
                _ = &mut done_rx => return,
                result = &mut interrupt_rx => match result {
                    Ok(handle) => handle,
                    Err(_) => return,
                },
                _ = tokio::time::sleep_until(deadline) => {
                    timeout_flag.store(true, Ordering::Relaxed);
                    // If we didn't receive the interrupt_rx yet, make sure to get it to call interrupt() later
                    tokio::select! {
                        biased;
                        _ = &mut done_rx => return,
                        result = &mut interrupt_rx => match result {
                            Ok(handle) => handle,
                            Err(_) => return,
                        },
                    }
                },
            };

            if timeout_flag.load(Ordering::Relaxed) {
                interrupt_handle.interrupt();
                return;
            }

            tokio::select! {
                biased;
                _ = &mut done_rx => {}
                _ = tokio::time::sleep_until(deadline) => {
                    timeout_flag.store(true, Ordering::Relaxed);
                    interrupt_handle.interrupt();
                }
            }
        });

        Self {
            timed_out,
            interrupt_tx: Some(interrupt_tx),
            done_tx: Some(done_tx),
            task: Some(task),
        }
    }

    fn publish_interrupt_handle(&mut self, handle: DuckDbQueryInterruptHandle) {
        self.publish_query_interrupt_handle(handle);
    }

    fn publish_query_interrupt_handle(&mut self, handle: DuckDbQueryInterruptHandle) {
        if let Some(interrupt_tx) = self.interrupt_tx.take() {
            let _ = interrupt_tx.send(handle);
        }
    }

    fn finish(&mut self) {
        if let Some(done_tx) = self.done_tx.take() {
            let _ = done_tx.send(());
        }
    }

    fn timed_out(&self) -> bool {
        self.timed_out.load(Ordering::Relaxed)
    }

    fn async_task_handle(&mut self) -> EtlResult<JoinHandle<()>> {
        self.task.take().ok_or_else(|| {
            etl_error!(
                ErrorKind::DestinationError,
                "Cannot get async task handle from watchdog: task is None"
            )
        })
    }
}

/// Shared registry of interrupt handles for live DuckLake connections.
#[derive(Default)]
pub(super) struct DuckLakeInterruptRegistry {
    handles: Mutex<Vec<Weak<RegisteredDuckLakeInterrupt>>>,
}

impl DuckLakeInterruptRegistry {
    /// Registers one live DuckLake connection interrupt handle.
    fn register(&self, handle: &Arc<RegisteredDuckLakeInterrupt>) {
        let mut handles =
            self.handles.lock().expect("ducklake interrupt registry mutex should not be poisoned");
        handles.retain(|registered| registered.strong_count() > 0);
        handles.push(Arc::downgrade(handle));
    }

    /// Interrupts all currently live registered DuckLake connections.
    #[cfg(test)]
    pub(super) fn interrupt_all(&self) -> usize {
        self.interrupt_all_with_reason(DuckLakeInterruptReason::ExternalInterrupt)
    }

    /// Interrupts all live connections and records one shared reason.
    fn interrupt_all_with_reason(&self, reason: DuckLakeInterruptReason) -> usize {
        let mut interrupted = 0;
        let mut handles =
            self.handles.lock().expect("ducklake interrupt registry mutex should not be poisoned");
        handles.retain(|handle| {
            let Some(handle) = handle.upgrade() else {
                return false;
            };
            handle.interrupt_with_reason(reason);
            interrupted += 1;
            true
        });
        interrupted
    }
}

/// Custom r2d2 connection manager for one DuckLake pool.
///
/// Every managed connection is cloned from one initialized database instance
/// shared by both destination pools. The anchor mutex only serializes
/// connection cloning and database-generation replacement; pooled connections
/// remain distinct and execute independently.
#[derive(Clone)]
pub(super) struct DuckLakeConnectionManager {
    /// Ordered setup phases executed immediately after a new connection opens.
    pub(super) setup_plan: Arc<DuckLakeSetupPlan>,
    /// Disables DuckDB extension autoload/autoinstall when vendored local
    /// extensions are required.
    pub(super) disable_extension_autoload: bool,
    /// Shared registry of interrupt handles for live connections.
    pub(super) interrupt_registry: Arc<DuckLakeInterruptRegistry>,
    /// Set once process or destination shutdown should stop new DuckDB work.
    pub(super) shutdown_requested: Arc<AtomicBool>,
    /// Anchor connection used to clone pooled connections from one database.
    shared_instance: Arc<Mutex<duckdb::Connection>>,
    /// Recreates host-specific configuration and credentials on every
    /// generation.
    connection_initializer: Option<ConnectionInitializer>,
    /// Counts successfully initialized DuckDB connections for tests.
    #[cfg(feature = "test-utils")]
    pub(super) open_count: Arc<AtomicUsize>,
}

/// DuckDB connection state tracked while a pooled connection is checked out.
pub(super) struct ManagedDuckLakeConnection {
    conn: duckdb::Connection,
    interrupt_handle: Arc<RegisteredDuckLakeInterrupt>,
    shutdown_requested: Arc<AtomicBool>,
    broken: bool,
}

/// One pooled DuckLake connection pinned across a multi-call copy session.
///
/// Ordinary operations check a connection out for one call and may use a
/// different pooled connection on the next call. Buffered copies instead store
/// rows in a DuckDB temporary table, which exists only on the connection that
/// created it. This handle lazily checks out one connection, shares that same
/// connection across append and flush calls, and never replaces it after an
/// operation fails. The owner must then restart the complete table-copy
/// attempt.
#[derive(Clone)]
pub(super) struct DuckLakeDedicatedConnection {
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    /// Exclusive access is intentional because DuckDB connections,
    /// transactions, and connection-local temporary tables cannot be used by
    /// concurrent operations in the same copy session.
    connection: Arc<Mutex<Option<r2d2::PooledConnection<DuckLakeConnectionManager>>>>,
}

impl DuckLakeDedicatedConnection {
    /// Creates a lazy dedicated session backed by one DuckLake pool.
    pub(super) fn new(pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>) -> Self {
        Self { pool, connection: Arc::new(Mutex::new(None)) }
    }
}

/// Supplies one managed connection to the shared blocking-operation runner.
///
/// Implementations differ only in connection lifetime: ordinary operations
/// check out a connection per call, while buffered-copy operations retain one
/// connection across calls.
trait DuckLakeConnectionProvider: Send + 'static {
    /// Runs `operation` with one managed connection before `deadline`.
    fn with_connection<R, F>(
        self,
        deadline: Instant,
        timeout: Duration,
        operation: F,
    ) -> EtlResult<R>
    where
        F: FnOnce(&mut ManagedDuckLakeConnection) -> EtlResult<R>;
}

impl DuckLakeConnectionProvider for Arc<r2d2::Pool<DuckLakeConnectionManager>> {
    fn with_connection<R, F>(
        self,
        deadline: Instant,
        timeout: Duration,
        operation: F,
    ) -> EtlResult<R>
    where
        F: FnOnce(&mut ManagedDuckLakeConnection) -> EtlResult<R>,
    {
        let checkout_timeout =
            deadline.checked_duration_since(Instant::now()).unwrap_or(Duration::ZERO);
        if checkout_timeout.is_zero() {
            return Err(duckdb_blocking_timeout_error(timeout, "pool_checkout"));
        }
        let checkout_started = Instant::now();
        let mut connection = self.get_timeout(checkout_timeout).map_err(|error| {
            if Instant::now() >= deadline {
                duckdb_blocking_timeout_error(timeout, "pool_checkout")
            } else {
                etl_error!(
                    ErrorKind::DestinationConnectionFailed,
                    "Failed to check out DuckLake connection",
                    source: error
                )
            }
        })?;
        histogram!(ETL_DUCKLAKE_POOL_CHECKOUT_WAIT_SECONDS)
            .record(checkout_started.elapsed().as_secs_f64());
        trace!(
            wait_ms = checkout_started.elapsed().as_millis() as u64,
            "wait for ducklake pool checkout"
        );

        operation(&mut connection)
    }
}

impl DuckLakeConnectionProvider for DuckLakeDedicatedConnection {
    fn with_connection<R, F>(
        self,
        deadline: Instant,
        timeout: Duration,
        operation: F,
    ) -> EtlResult<R>
    where
        F: FnOnce(&mut ManagedDuckLakeConnection) -> EtlResult<R>,
    {
        let mut connection = self.connection.lock().map_err(|_| {
            etl_error!(ErrorKind::InvalidState, "DuckLake dedicated connection mutex was poisoned")
        })?;

        if connection.is_none() {
            let checkout_timeout =
                deadline.checked_duration_since(Instant::now()).unwrap_or(Duration::ZERO);
            if checkout_timeout.is_zero() {
                return Err(duckdb_blocking_timeout_error(timeout, "pool_checkout"));
            }
            let checkout_started = Instant::now();
            let pooled_connection = self.pool.get_timeout(checkout_timeout).map_err(|error| {
                if Instant::now() >= deadline {
                    duckdb_blocking_timeout_error(timeout, "pool_checkout")
                } else {
                    etl_error!(
                        ErrorKind::DestinationConnectionFailed,
                        "Failed to check out dedicated DuckLake connection",
                        source: error
                    )
                }
            })?;
            histogram!(ETL_DUCKLAKE_POOL_CHECKOUT_WAIT_SECONDS)
                .record(checkout_started.elapsed().as_secs_f64());
            trace!(
                wait_ms = checkout_started.elapsed().as_millis() as u64,
                "wait for dedicated ducklake pool checkout"
            );
            *connection = Some(pooled_connection);
        }

        let connection = connection.as_mut().ok_or_else(|| {
            etl_error!(ErrorKind::InvalidState, "DuckLake dedicated connection is unavailable")
        })?;
        operation(connection)
    }
}

/// DuckDB connection and registered interrupt handle opened by the manager.
struct OpenDuckLakeConnection {
    conn: duckdb::Connection,
    interrupt_handle: Arc<RegisteredDuckLakeInterrupt>,
}

/// Error returned while opening or validating one DuckLake pooled connection.
#[derive(Debug)]
pub(super) struct DuckLakeConnectionError {
    message: String,
}

impl DuckLakeConnectionError {
    /// Creates one setup-phase error and redacts a PostgreSQL catalog password
    /// when present.
    fn setup_phase(step: &DuckLakeSetupStep, error: duckdb::Error) -> Self {
        let error_message = error.to_string();
        let error_message = if attach_step_uses_postgres_catalog(step) {
            redact_ducklake_connection_error_message(&error_message)
        } else {
            Cow::Owned(error_message)
        };

        Self {
            message: format!(
                "ducklake duckdb connection setup phase `{}` failed: {error_message}",
                step.label
            ),
        }
    }

    /// Creates one connection validation error.
    fn validation(error: duckdb::Error) -> Self {
        Self { message: format!("DuckLake DuckDB connection validation failed: {error}") }
    }

    /// Creates one shared-instance state error.
    fn shared_instance(message: &'static str) -> Self {
        Self { message: message.to_owned() }
    }
}

impl fmt::Display for DuckLakeConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.write_str(&self.message)
    }
}

impl error::Error for DuckLakeConnectionError {}

impl DuckLakeConnectionManager {
    /// Creates a manager anchored to one initialized in-memory DuckDB database.
    pub(super) async fn new(
        setup_plan: Arc<DuckLakeSetupPlan>,
        disable_extension_autoload: bool,
        interrupt_registry: Arc<DuckLakeInterruptRegistry>,
        shutdown_requested: Arc<AtomicBool>,
    ) -> EtlResult<Self> {
        Self::new_with_initializer(
            setup_plan,
            disable_extension_autoload,
            interrupt_registry,
            shutdown_requested,
            None,
        )
        .await
    }

    /// Creates a manager with an optional host-owned instance initializer.
    pub(super) async fn new_with_initializer(
        setup_plan: Arc<DuckLakeSetupPlan>,
        disable_extension_autoload: bool,
        interrupt_registry: Arc<DuckLakeInterruptRegistry>,
        shutdown_requested: Arc<AtomicBool>,
        connection_initializer: Option<ConnectionInitializer>,
    ) -> EtlResult<Self> {
        let initialize = connection_initializer.clone();
        let setup_plan_for_initialization = Arc::clone(&setup_plan);
        let instance = tokio::task::spawn_blocking(move || {
            if let Some(initialize) = initialize {
                return initialize();
            }
            Self::open_initialized_duckdb_instance(
                setup_plan_for_initialization.as_ref(),
                disable_extension_autoload,
            )
            .map_err(|error| {
                etl_error!(
                    ErrorKind::DestinationConnectionFailed,
                    "Failed to initialize shared DuckLake DuckDB instance",
                    source: error
                )
            })
        })
        .await
        .map_err(|_| {
            etl_error!(
                ErrorKind::ApplyWorkerPanic,
                "DuckLake shared instance initialization task panicked"
            )
        })??;

        Ok(Self {
            setup_plan,
            disable_extension_autoload,
            interrupt_registry,
            shutdown_requested,
            shared_instance: Arc::new(Mutex::new(instance)),
            connection_initializer,
            #[cfg(feature = "test-utils")]
            open_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Creates another pool manager backed by the same database instance.
    pub(super) fn new_pool_manager(&self) -> Self {
        Self {
            setup_plan: Arc::clone(&self.setup_plan),
            disable_extension_autoload: self.disable_extension_autoload,
            interrupt_registry: Arc::clone(&self.interrupt_registry),
            shutdown_requested: Arc::clone(&self.shutdown_requested),
            shared_instance: Arc::clone(&self.shared_instance),
            connection_initializer: self.connection_initializer.clone(),
            #[cfg(feature = "test-utils")]
            open_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Records DuckLake shutdown and interrupts all live managed connections.
    pub(super) fn interrupt_all_connections_for_shutdown(&self) -> usize {
        self.shutdown_requested.store(true, Ordering::Relaxed);
        self.interrupt_registry
            .interrupt_all_with_reason(DuckLakeInterruptReason::DestinationShutdown)
    }

    /// Records process shutdown and interrupts all live managed connections.
    pub(super) fn interrupt_all_connections_for_process_shutdown(&self) -> usize {
        self.shutdown_requested.store(true, Ordering::Relaxed);
        self.interrupt_registry.interrupt_all_with_reason(DuckLakeInterruptReason::ProcessShutdown)
    }

    /// Interrupts all currently live managed DuckLake connections.
    #[cfg(test)]
    pub(super) fn interrupt_all_connections(&self) -> usize {
        self.interrupt_registry.interrupt_all()
    }

    /// Opens and configures one new in-memory DuckDB database.
    fn open_initialized_duckdb_instance(
        setup_plan: &DuckLakeSetupPlan,
        disable_extension_autoload: bool,
    ) -> Result<duckdb::Connection, DuckLakeConnectionError> {
        let connection_init_id = NEXT_CONNECTION_INIT_ID.fetch_add(1, Ordering::Relaxed);
        let conn = if disable_extension_autoload {
            duckdb::Connection::open_in_memory_with_flags(
                Config::default()
                    .enable_autoload_extension(false)
                    .map_err(DuckLakeConnectionError::validation)?,
            )
            .map_err(DuckLakeConnectionError::validation)?
        } else {
            duckdb::Connection::open_in_memory().map_err(DuckLakeConnectionError::validation)?
        };
        Self::run_setup_steps(connection_init_id, &conn, setup_plan.steps())?;
        Ok(conn)
    }

    /// Runs named setup phases on one DuckDB connection.
    fn run_setup_steps<'a>(
        connection_init_id: u64,
        conn: &duckdb::Connection,
        steps: impl IntoIterator<Item = &'a DuckLakeSetupStep>,
    ) -> Result<(), DuckLakeConnectionError> {
        for step in steps {
            let phase_started = Instant::now();
            info!(
                connection_init_id,
                phase = step.label,
                "starting ducklake duckdb connection setup phase"
            );
            if let Err(error) = conn.execute_batch(&step.sql) {
                return Err(DuckLakeConnectionError::setup_phase(step, error));
            }
            info!(
                connection_init_id,
                phase = step.label,
                elapsed_ms = phase_started.elapsed().as_millis() as u64,
                "ducklake duckdb connection setup phase finished"
            );
        }
        Ok(())
    }

    /// Opens one managed connection cloned from the shared database instance.
    fn open_duckdb_connection(&self) -> Result<OpenDuckLakeConnection, DuckLakeConnectionError> {
        let connection_init_id = NEXT_CONNECTION_INIT_ID.fetch_add(1, Ordering::Relaxed);
        let conn = self
            .shared_instance
            .lock()
            .map_err(|_| {
                DuckLakeConnectionError::shared_instance(
                    "DuckLake shared DuckDB instance mutex was poisoned",
                )
            })?
            .try_clone()
            .map_err(DuckLakeConnectionError::validation)?;
        let interrupt_handle = Arc::new(RegisteredDuckLakeInterrupt::new(conn.interrupt_handle()));
        self.interrupt_registry.register(&interrupt_handle);
        Self::run_setup_steps(connection_init_id, &conn, self.setup_plan.connection_steps())?;

        #[cfg(feature = "test-utils")]
        self.open_count.fetch_add(1, Ordering::Relaxed);
        Ok(OpenDuckLakeConnection { conn, interrupt_handle })
    }

    /// Replaces the shared database used by future pooled connections.
    pub(super) async fn recreate_shared_instance(&self) -> EtlResult<()> {
        let shared_instance = Arc::clone(&self.shared_instance);
        let setup_plan = Arc::clone(&self.setup_plan);
        let disable_extension_autoload = self.disable_extension_autoload;
        let initialize = self.connection_initializer.clone();
        tokio::task::spawn_blocking(move || -> EtlResult<()> {
            let instance = if let Some(initialize) = initialize {
                initialize()?
            } else {
                Self::open_initialized_duckdb_instance(
                    setup_plan.as_ref(),
                    disable_extension_autoload,
                )
                .map_err(|error| {
                    etl_error!(
                        ErrorKind::DestinationConnectionFailed,
                        "Failed to recreate shared DuckLake DuckDB instance",
                        source: error
                    )
                })?
            };
            let mut current = shared_instance.lock().map_err(|_| {
                etl_error!(
                    ErrorKind::InvalidState,
                    "DuckLake shared DuckDB instance mutex was poisoned"
                )
            })?;
            *current = instance;
            Ok(())
        })
        .await
        .map_err(|_| {
            etl_error!(
                ErrorKind::ApplyWorkerPanic,
                "DuckLake shared instance recreation task panicked"
            )
        })?
    }

    /// Returns the number of successfully initialized DuckDB connections.
    #[cfg(feature = "test-utils")]
    pub(super) fn open_count_for_tests(&self) -> usize {
        self.open_count.load(Ordering::Relaxed)
    }
}

impl r2d2::ManageConnection for DuckLakeConnectionManager {
    type Connection = ManagedDuckLakeConnection;
    type Error = DuckLakeConnectionError;

    fn connect(&self) -> Result<ManagedDuckLakeConnection, DuckLakeConnectionError> {
        let OpenDuckLakeConnection { conn, interrupt_handle } = self.open_duckdb_connection()?;

        Ok(ManagedDuckLakeConnection {
            conn,
            interrupt_handle,
            shutdown_requested: Arc::clone(&self.shutdown_requested),
            broken: false,
        })
    }

    fn is_valid(
        &self,
        conn: &mut ManagedDuckLakeConnection,
    ) -> Result<(), DuckLakeConnectionError> {
        if conn.shutdown_requested.load(Ordering::Relaxed) {
            return Ok(());
        }

        conn.conn.execute_batch("SELECT 1").map_err(DuckLakeConnectionError::validation)?;
        Ok(())
    }

    fn has_broken(&self, conn: &mut ManagedDuckLakeConnection) -> bool {
        conn.broken
    }
}

/// Formats owned query context for a DuckDB query failure.
pub(super) fn format_query_error_detail(sql: &str) -> String {
    let compact_sql = sql.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("sql: {compact_sql}")
}

/// Returns whether one setup step attaches a PostgreSQL-backed catalog.
fn attach_step_uses_postgres_catalog(step: &DuckLakeSetupStep) -> bool {
    step.label == "attach_catalog" && step.sql.contains("ducklake:postgres:")
}

/// Redacts PostgreSQL passwords from one DuckDB or r2d2 error message.
fn redact_ducklake_connection_error_message(message: &str) -> Cow<'_, str> {
    POSTGRES_PASSWORD_REGEX.replace_all(message, "password='[redacted]'")
}

/// Builds and warms an r2d2 pool of initialized DuckDB connections.
pub(super) async fn build_warm_ducklake_pool(
    manager: DuckLakeConnectionManager,
    pool_size: u32,
    purpose: &'static str,
) -> EtlResult<r2d2::Pool<DuckLakeConnectionManager>> {
    tokio::task::spawn_blocking(move || -> EtlResult<_> {
        let started = Instant::now();
        let pool = r2d2::Pool::builder()
            .max_size(pool_size)
            .min_idle(Some(0))
            .connection_timeout(Duration::from_mins(4))
            .test_on_check_out(true)
            // Callers log the returned pool initialization failure once, so
            // suppress r2d2's internal per-attempt logging here.
            .error_handler(Box::new(r2d2::NopErrorHandler))
            .build(manager)
            .map_err(|e| {
                etl_error!(
                    ErrorKind::DestinationConnectionFailed,
                    "Failed to build DuckLake connection pool",
                    source: e
                )
            })?;

        let mut warmed_connections = Vec::with_capacity(pool_size as usize);
        for _ in 0..pool_size {
            let conn = pool.get().map_err(|e| {
                etl_error!(
                    ErrorKind::DestinationConnectionFailed,
                    "Failed to warm DuckLake connection pool",
                    source: e
                )
            })?;
            warmed_connections.push(conn);
        }
        drop(warmed_connections);

        trace!(
            purpose,
            pool_size,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "ducklake connection pool warmed"
        );

        Ok(pool)
    })
    .await
    .map_err(|_| {
        etl_error!(
            ErrorKind::ApplyWorkerPanic,
            "DuckLake connection pool initialization task panicked"
        )
    })?
}

/// Builds a consistent timeout error for one blocking DuckDB stage.
#[inline]
pub(super) fn duckdb_blocking_timeout_error(timeout: Duration, stage: &'static str) -> EtlError {
    etl_error!(
        ErrorKind::DestinationQueryFailed,
        "DuckLake blocking operation timed out",
        format!(
            "Operation kind={}, stage={stage}, timeout_ms={}",
            DUCKDB_BLOCKING_OPERATION_KIND,
            timeout.as_millis()
        )
    )
}

/// Builds the error returned when shutdown has stopped new DuckDB work.
pub(super) fn ducklake_shutdown_requested_error() -> EtlError {
    etl_error!(ErrorKind::DestinationConnectionFailed, DUCKLAKE_SHUTDOWN_REQUESTED)
}

/// Returns whether an error was produced by DuckLake shutdown admission checks.
pub(super) fn is_ducklake_shutdown_requested_error(error: &EtlError) -> bool {
    error.kind() == ErrorKind::DestinationConnectionFailed
        && error.description() == Some(DUCKLAKE_SHUTDOWN_REQUESTED)
}

fn abort_stuck_duckdb_blocking_operation(timeout: Duration, abort_grace: Duration) -> ! {
    tracing::error!(
        operation_kind = DUCKDB_BLOCKING_OPERATION_KIND,
        timeout_ms = timeout.as_millis() as u64,
        abort_grace_ms = abort_grace.as_millis() as u64,
        "ducklake blocking operation did not return after timeout interrupt; aborting process"
    );
    process::abort();
}

/// Runs one DuckDB operation on Tokio's blocking pool after acquiring a permit
/// that matches the configured DuckDB concurrency limit and then checking out a
/// warm pooled DuckDB connection.
pub(super) async fn run_duckdb_blocking<R, F>(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    operation: F,
) -> EtlResult<R>
where
    R: Send + 'static,
    F: FnOnce(&duckdb::Connection) -> EtlResult<R> + Send + 'static,
{
    run_duckdb_blocking_with_timeout(pool, blocking_slots, FOREGROUND_QUERY_TIMEOUT, operation)
        .await
}

/// Runs one DuckDB operation with context for operation-local diagnostics.
pub(super) async fn run_duckdb_blocking_with_context<R, F>(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    operation: F,
) -> EtlResult<R>
where
    R: Send + 'static,
    F: FnOnce(&duckdb::Connection, &DuckLakeBlockingOperationContext) -> EtlResult<R>
        + Send
        + 'static,
{
    run_duckdb_blocking_with_provider_and_context(
        pool,
        blocking_slots,
        FOREGROUND_QUERY_TIMEOUT,
        operation,
    )
    .await
}

/// Runs one operation on a connection pinned to a multi-call DuckLake session.
///
/// This retains the same timeout, interrupt, and blocking-slot behavior as
/// [`run_duckdb_blocking_with_context`], but keeps the checked-out connection
/// until every clone of `session` is dropped. A failed operation marks the
/// connection broken and all later calls fail without checking out a
/// replacement connection.
pub(super) async fn run_duckdb_dedicated_blocking_with_context<R, F>(
    session: DuckLakeDedicatedConnection,
    blocking_slots: Arc<Semaphore>,
    operation: F,
) -> EtlResult<R>
where
    R: Send + 'static,
    F: FnOnce(&duckdb::Connection, &DuckLakeBlockingOperationContext) -> EtlResult<R>
        + Send
        + 'static,
{
    run_duckdb_blocking_with_provider_and_context(
        session,
        blocking_slots,
        FOREGROUND_QUERY_TIMEOUT,
        operation,
    )
    .await
}

/// Runs one DuckDB operation with an explicit timeout budget.
pub(super) async fn run_duckdb_blocking_with_timeout<R, F>(
    pool: Arc<r2d2::Pool<DuckLakeConnectionManager>>,
    blocking_slots: Arc<Semaphore>,
    timeout: Duration,
    operation: F,
) -> EtlResult<R>
where
    R: Send + 'static,
    F: FnOnce(&duckdb::Connection) -> EtlResult<R> + Send + 'static,
{
    run_duckdb_blocking_with_provider_and_context(pool, blocking_slots, timeout, move |conn, _| {
        operation(conn)
    })
    .await
}

/// Runs one DuckDB operation with an explicit timeout and diagnostic context.
async fn run_duckdb_blocking_with_provider_and_context<P, R, F>(
    provider: P,
    blocking_slots: Arc<Semaphore>,
    timeout: Duration,
    operation: F,
) -> EtlResult<R>
where
    P: DuckLakeConnectionProvider,
    R: Send + 'static,
    F: FnOnce(&duckdb::Connection, &DuckLakeBlockingOperationContext) -> EtlResult<R>
        + Send
        + 'static,
{
    let operation_kind = DUCKDB_BLOCKING_OPERATION_KIND;
    let operation_id = NEXT_DUCKDB_BLOCKING_OPERATION_ID.fetch_add(1, Ordering::Relaxed);
    let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
        etl_error!(ErrorKind::ConfigError, "DuckLake operation timeout is too large")
    })?;
    let abort_deadline = deadline.checked_add(BLOCKING_ABORT_GRACE).ok_or_else(|| {
        etl_error!(ErrorKind::ConfigError, "DuckLake operation timeout leaves no abort grace")
    })?;
    let slot_wait_started = Instant::now();
    let permit = tokio::time::timeout_at(deadline, Arc::clone(&blocking_slots).acquire_owned())
        .await
        .map_err(|_| duckdb_blocking_timeout_error(timeout, "slot_wait"))?
        .map_err(|_| {
            etl_error!(ErrorKind::ApplyWorkerPanic, "DuckLake blocking slot acquisition failed")
        })?;
    histogram!(ETL_DUCKLAKE_BLOCKING_SLOT_WAIT_SECONDS)
        .record(slot_wait_started.elapsed().as_secs_f64());
    trace!(
        wait_ms = slot_wait_started.elapsed().as_millis() as u64,
        "wait for ducklake blocking slot"
    );

    // This is needed to make sure we properly interrupt the blocking operation if
    // it exceeds the timeout, we don't just cancel the task and leave the
    // connection active.
    let mut watchdog = DuckDbQueryWatchdog::spawn(deadline);
    let watchdog_task = watchdog.async_task_handle()?;

    let blocking_task = tokio::task::spawn_blocking(move || -> EtlResult<R> {
        // Please if you modify the code inside this blocking task do not add any
        // blocking operations that could delay other tasks waiting on this slot.
        let _permit = permit;
        provider.with_connection(deadline, timeout, move |pooled_conn| {
            if pooled_conn.broken {
                return Err(etl_error!(
                    ErrorKind::DestinationConnectionFailed,
                    "DuckLake connection was invalidated",
                    "Restart the operation that owned this connection"
                ));
            }
            if pooled_conn.shutdown_requested.load(Ordering::Relaxed) {
                warn!(
                    operation_id,
                    operation_kind = operation_kind,
                    "ducklake blocking operation skipped because shutdown was requested"
                );
                return Err(ducklake_shutdown_requested_error());
            }
            let operation_timeout =
                deadline.checked_duration_since(Instant::now()).unwrap_or(Duration::ZERO);
            if operation_timeout.is_zero() {
                return Err(duckdb_blocking_timeout_error(timeout, "query_execution"));
            }
            pooled_conn.interrupt_handle.clear_reason();
            let operation_context = DuckLakeBlockingOperationContext::new(
                operation_id,
                operation_kind,
                timeout,
                pooled_conn.interrupt_handle.interrupt_state(),
            );
            let interrupt_handle: Arc<RegisteredDuckLakeInterrupt> =
                Arc::clone(&pooled_conn.interrupt_handle);
            let interrupt_handle: DuckDbQueryInterruptHandle = interrupt_handle;
            watchdog.publish_interrupt_handle(interrupt_handle);
            if watchdog.timed_out() {
                pooled_conn.broken = true;
                return Err(duckdb_blocking_timeout_error(timeout, "query_execution"));
            }
            let operation_started = Instant::now();
            let result = operation(&pooled_conn.conn, &operation_context);
            watchdog.finish();
            histogram!(ETL_DUCKLAKE_BLOCKING_OPERATION_DURATION_SECONDS)
                .record(operation_started.elapsed().as_secs_f64());
            trace!(
                duration_ms = operation_started.elapsed().as_millis() as u64,
                "ducklake blocking operation finished"
            );
            if watchdog.timed_out() {
                pooled_conn.broken = true;
                return Err(duckdb_blocking_timeout_error(timeout, "query_execution"));
            }
            if result.is_err() {
                pooled_conn.broken = true;
            }

            result
        })
    });

    let blocking_result = tokio::select! {
        biased;
        result = blocking_task => result,
        _ = tokio::time::sleep_until(abort_deadline) => {
            // The blocking task still owns the pooled connection here, so the
            // async side cannot mark it broken and return it to r2d2 for
            // eviction. If DuckDB does not return after interrupt plus grace,
            // the stuck native call also keeps holding its semaphore permit and
            // blocking thread, so restarting the process is the recoverable
            // boundary.
            abort_stuck_duckdb_blocking_operation(timeout, BLOCKING_ABORT_GRACE);
        }
    };

    // Await the watchdog so it cannot outlive the finished blocking task and
    // accidentally interrupt a later operation that reuses the connection.
    watchdog_task.await.map_err(|_| {
        etl_error!(ErrorKind::ApplyWorkerPanic, "DuckLake query watchdog task panicked")
    })?;

    blocking_result.map_err(|_| {
        etl_error!(ErrorKind::ApplyWorkerPanic, "DuckLake blocking operation task panicked")
    })?
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{Barrier, Semaphore, oneshot};

    use super::*;

    const WATCHDOG_DEADLINE: Duration = Duration::from_millis(20);
    const WATCHDOG_JOIN_TIMEOUT: Duration = Duration::from_secs(1);

    fn make_blocking_test_manager() -> DuckLakeConnectionManager {
        DuckLakeConnectionManager {
            setup_plan: Arc::new(DuckLakeSetupPlan::default()),
            disable_extension_autoload: cfg!(target_os = "linux"),
            interrupt_registry: Arc::new(DuckLakeInterruptRegistry::default()),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            shared_instance: Arc::new(Mutex::new(duckdb::Connection::open_in_memory().unwrap())),
            connection_initializer: None,
            #[cfg(feature = "test-utils")]
            open_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[tokio::test]
    async fn shared_connections_see_database_and_isolate_temporary_tables() {
        let manager = make_blocking_test_manager();
        let copy_manager = manager.new_pool_manager();

        let first = r2d2::ManageConnection::connect(&manager).unwrap();
        let second = r2d2::ManageConnection::connect(&copy_manager).unwrap();
        first
            .conn
            .execute_batch(
                "create table shared_table (id integer); insert into shared_table values (1); \
                 create temporary table local_table (id integer); insert into local_table values \
                 (2);",
            )
            .unwrap();

        let shared_value: i32 =
            second.conn.query_row("select id from shared_table", [], |row| row.get(0)).unwrap();
        assert_eq!(shared_value, 1);
        assert!(second.conn.execute_batch("select * from local_table").is_err());
        let local_value: i32 =
            first.conn.query_row("select id from local_table", [], |row| row.get(0)).unwrap();
        assert_eq!(local_value, 2);
    }

    #[tokio::test]
    async fn shared_connections_support_concurrent_appends() {
        let manager = make_blocking_test_manager();
        let copy_manager = manager.new_pool_manager();
        let setup = r2d2::ManageConnection::connect(&manager).unwrap();
        setup.conn.execute_batch("create table concurrent_rows (id integer)").unwrap();
        drop(setup);

        let connections =
            (0..4).map(|_| r2d2::ManageConnection::connect(&manager).unwrap()).collect::<Vec<_>>();
        let barrier = Arc::new(std::sync::Barrier::new(connections.len()));
        std::thread::scope(|scope| {
            for (worker, connection) in connections.into_iter().enumerate() {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    connection
                        .conn
                        .execute(
                            "insert into concurrent_rows select ? * 1000 + range from range(1000)",
                            [i32::try_from(worker).unwrap()],
                        )
                        .unwrap();
                });
            }
        });

        let reader = r2d2::ManageConnection::connect(&copy_manager).unwrap();
        let count: i64 = reader
            .conn
            .query_row("select count(*) from concurrent_rows", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 4_000);
    }

    #[tokio::test]
    async fn recreated_shared_instance_is_a_new_database_generation() {
        let manager = make_blocking_test_manager();
        let copy_manager = manager.new_pool_manager();
        let old_connection = r2d2::ManageConnection::connect(&manager).unwrap();
        old_connection.conn.execute_batch("create table old_generation (id integer)").unwrap();

        manager.recreate_shared_instance().await.unwrap();
        let new_connection = r2d2::ManageConnection::connect(&copy_manager).unwrap();

        assert!(new_connection.conn.execute_batch("select * from old_generation").is_err());
        old_connection.conn.execute_batch("select * from old_generation").unwrap();
    }

    fn watchdog_interrupt_handle() -> (duckdb::Connection, DuckDbQueryInterruptHandle) {
        let conn =
            duckdb::Connection::open_in_memory().expect("failed to open watchdog test connection");
        let handle = conn.interrupt_handle();
        (conn, handle)
    }

    struct DelayedInterruptHandle {
        delay: Duration,
        calls: std::sync::atomic::AtomicUsize,
        started: AtomicBool,
        completed: AtomicBool,
    }

    impl DelayedInterruptHandle {
        fn new(delay: Duration) -> Self {
            Self {
                delay,
                calls: std::sync::atomic::AtomicUsize::new(0),
                started: AtomicBool::new(false),
                completed: AtomicBool::new(false),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }

        fn started(&self) -> bool {
            self.started.load(Ordering::Relaxed)
        }

        fn completed(&self) -> bool {
            self.completed.load(Ordering::Relaxed)
        }
    }

    impl DuckDbQueryInterrupt for DelayedInterruptHandle {
        fn interrupt(&self) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.started.store(true, Ordering::Relaxed);
            std::thread::sleep(self.delay);
            self.completed.store(true, Ordering::Relaxed);
        }
    }

    fn delayed_interrupt_handle(
        delay: Duration,
    ) -> (Arc<DelayedInterruptHandle>, DuckDbQueryInterruptHandle) {
        let handle = Arc::new(DelayedInterruptHandle::new(delay));
        (Arc::clone(&handle), handle)
    }

    async fn wait_for_delayed_interrupt_to_start(handle: &DelayedInterruptHandle) {
        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, async {
            while !handle.started() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("delayed interrupt should start");
    }

    #[derive(Clone, Copy, Debug)]
    enum WatchdogSignal {
        PublishInterrupt,
        Finish,
        DropInterruptSender,
        DropDoneSender,
    }

    type WatchdogSignalCase =
        (&'static str, &'static [WatchdogSignal], bool, &'static [WatchdogSignal], bool);

    fn apply_watchdog_signal(
        watchdog: &mut DuckDbQueryWatchdog,
        handle: &DuckDbQueryInterruptHandle,
        signal: WatchdogSignal,
    ) {
        match signal {
            WatchdogSignal::PublishInterrupt => {
                watchdog.publish_query_interrupt_handle(Arc::clone(handle));
            }
            WatchdogSignal::Finish => {
                watchdog.finish();
            }
            WatchdogSignal::DropInterruptSender => {
                drop(watchdog.interrupt_tx.take());
            }
            WatchdogSignal::DropDoneSender => {
                drop(watchdog.done_tx.take());
            }
        }
    }

    async fn assert_watchdog_sequence_completes(
        name: &str,
        before_deadline: &[WatchdogSignal],
        wait_past_deadline: bool,
        after_deadline: &[WatchdogSignal],
        expected_timed_out: bool,
    ) {
        let (_conn, interrupt_handle) = watchdog_interrupt_handle();
        let mut watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let watchdog_task = watchdog.async_task_handle().expect("failed to extract watchdog task");

        for signal in before_deadline {
            apply_watchdog_signal(&mut watchdog, &interrupt_handle, *signal);
            tokio::task::yield_now().await;
        }

        if wait_past_deadline {
            tokio::time::sleep(WATCHDOG_DEADLINE * 2).await;
        }

        for signal in after_deadline {
            apply_watchdog_signal(&mut watchdog, &interrupt_handle, *signal);
            tokio::task::yield_now().await;
        }

        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, watchdog_task)
            .await
            .unwrap_or_else(|_| panic!("watchdog sequence `{name}` deadlocked"))
            .unwrap_or_else(|error| panic!("watchdog sequence `{name}` panicked: {error}"));

        assert_eq!(
            watchdog.timed_out(),
            expected_timed_out,
            "unexpected timeout state for watchdog sequence `{name}`"
        );
    }

    #[tokio::test]
    async fn overflowing_timeout_returns_config_error_before_running_operation() {
        let pool = Arc::new(
            build_warm_ducklake_pool(make_blocking_test_manager(), 1, "test").await.unwrap(),
        );
        let slots = Arc::new(Semaphore::new(1));
        let error = run_duckdb_blocking_with_timeout(
            pool,
            Arc::clone(&slots),
            Duration::MAX,
            |_| -> EtlResult<()> {
                panic!("an invalid timeout must not start native work");
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ConfigError);
        assert_eq!(slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn run_duckdb_blocking_timeout_releases_resources_for_follow_up_queries() {
        let pool = Arc::new(
            build_warm_ducklake_pool(make_blocking_test_manager(), 1, "test")
                .await
                .expect("failed to build blocking test pool"),
        );
        let blocking_slots = Arc::new(Semaphore::new(1));

        let error = run_duckdb_blocking_with_timeout(
            Arc::clone(&pool),
            Arc::clone(&blocking_slots),
            Duration::from_millis(50),
            |_conn| -> EtlResult<()> {
                std::thread::sleep(Duration::from_millis(100));
                Ok(())
            },
        )
        .await
        .expect_err("expected blocking timeout");

        assert_eq!(error.kind(), ErrorKind::DestinationQueryFailed);
        assert_eq!(error.description(), Some("DuckLake blocking operation timed out"));
        // This timeout budget covers pool checkout and query execution. Under
        // full-suite load the deadline can expire before the blocking operation
        // starts, so either stage is acceptable as long as the pool remains
        // usable for the follow-up query below.
        assert!(
            error.detail().is_some_and(|detail| {
                detail.contains("stage=pool_checkout") || detail.contains("stage=query_execution")
            }),
            "unexpected error detail: {error:?}"
        );

        let value = run_duckdb_blocking_with_timeout(
            Arc::clone(&pool),
            Arc::clone(&blocking_slots),
            Duration::from_secs(1),
            |conn| -> EtlResult<i64> {
                conn.query_row("SELECT 1;", [], |row| row.get::<_, i64>(0)).map_err(|source| {
                    etl_error!(
                        ErrorKind::DestinationQueryFailed,
                        "DuckLake timeout verification query failed",
                        source: source
                    )
                })
            },
        )
        .await
        .expect("expected follow-up query to succeed");

        assert_eq!(value, 1);
    }

    #[test]
    fn format_query_error_detail_compacts_sql() {
        let sql = r#"CREATE TABLE lake."orders" ("id" INTEGER NOT NULL)"#;

        assert_eq!(
            format_query_error_detail(sql),
            "sql: CREATE TABLE lake.\"orders\" (\"id\" INTEGER NOT NULL)"
        );
    }

    #[test]
    fn ducklake_interrupt_state_records_first_reason() {
        let state = DuckLakeInterruptState::default();

        assert_eq!(state.reason(), DuckLakeInterruptReason::None);

        state.record(DuckLakeInterruptReason::DestinationShutdown);
        state.record(DuckLakeInterruptReason::QueryTimeout);

        assert_eq!(state.reason(), DuckLakeInterruptReason::DestinationShutdown);

        state.clear();

        assert_eq!(state.reason(), DuckLakeInterruptReason::None);
    }

    #[test]
    fn interrupt_registry_prunes_dropped_handles_during_registration() {
        let registry = DuckLakeInterruptRegistry::default();
        let first_conn =
            duckdb::Connection::open_in_memory().expect("failed to open first test connection");
        let first = Arc::new(RegisteredDuckLakeInterrupt::new(first_conn.interrupt_handle()));
        registry.register(&first);
        assert_eq!(registry.handles.lock().unwrap().len(), 1);

        drop(first);

        let second_conn =
            duckdb::Connection::open_in_memory().expect("failed to open second test connection");
        let second = Arc::new(RegisteredDuckLakeInterrupt::new(second_conn.interrupt_handle()));
        registry.register(&second);

        assert_eq!(registry.handles.lock().unwrap().len(), 1);
        assert_eq!(registry.interrupt_all(), 1);
    }

    #[test]
    fn ducklake_blocking_operation_context_reports_interrupt_reason() {
        let state = Arc::new(DuckLakeInterruptState::default());
        let context = DuckLakeBlockingOperationContext::new(
            42,
            DUCKDB_BLOCKING_OPERATION_KIND,
            Duration::from_millis(250),
            Arc::clone(&state),
        );

        assert_eq!(context.operation_id(), 42);
        assert_eq!(context.operation_kind(), DUCKDB_BLOCKING_OPERATION_KIND);
        assert_eq!(context.timeout_ms(), 250);
        assert_eq!(context.interrupt_reason_label(), "none");

        state.record(DuckLakeInterruptReason::ProcessShutdown);

        assert_eq!(context.interrupt_reason_label(), "process_shutdown");
    }

    #[test]
    fn redact_ducklake_connection_error_message_masks_quoted_password() {
        let message = "timed out waiting for connection: attach failed: postgres:host='localhost' \
                       user='ducklake' password='pa\\'ss\\\\word' dbname='catalog'";

        assert_eq!(
            redact_ducklake_connection_error_message(message),
            "timed out waiting for connection: attach failed: postgres:host='localhost' \
             user='ducklake' password='[redacted]' dbname='catalog'"
        );
    }

    #[test]
    fn redact_ducklake_connection_error_message_keeps_non_sensitive_messages() {
        let message = "timed out waiting for connection: parser error";

        assert_eq!(redact_ducklake_connection_error_message(message), message);
    }

    #[test]
    fn ducklake_connection_error_redacts_postgres_catalog_password() {
        let step = DuckLakeSetupStep {
            label: "attach_catalog",
            sql: "ATTACH 'ducklake:postgres:host=''localhost'' user=''ducklake'' \
                  password=''secret''' AS lake;"
                .to_owned(),
        };
        let error = duckdb::Error::DuckDBFailure(
            duckdb::ffi::Error::new(1),
            Some(
                "attach failed: postgres:host='localhost' user='ducklake' password='secret'"
                    .to_owned(),
            ),
        );

        assert_eq!(
            DuckLakeConnectionError::setup_phase(&step, error).to_string(),
            "ducklake duckdb connection setup phase `attach_catalog` failed: attach failed: \
             postgres:host='localhost' user='ducklake' password='[redacted]'"
        );
    }

    #[test]
    fn open_duckdb_connection_registers_interrupt_before_setup_sql() {
        let manager = DuckLakeConnectionManager {
            setup_plan: Arc::new(DuckLakeSetupPlan::from_steps(vec![DuckLakeSetupStep {
                label: "configure_writer_session",
                sql: "SELECT * FROM missing_setup_table;".to_owned(),
            }])),
            ..make_blocking_test_manager()
        };

        let Err(error) = manager.open_duckdb_connection() else {
            panic!("setup should fail");
        };

        assert!(
            error.to_string().contains("configure_writer_session"),
            "unexpected setup error: {error}"
        );
        assert_eq!(
            manager
                .interrupt_registry
                .handles
                .lock()
                .expect("ducklake interrupt registry mutex should not be poisoned")
                .len(),
            1,
            "interrupt handle should be registered before setup SQL runs"
        );
    }

    #[tokio::test]
    async fn query_watchdog_signal_order_matrix_does_not_deadlock() {
        use WatchdogSignal::{DropDoneSender, DropInterruptSender, Finish, PublishInterrupt};

        let cases: &[WatchdogSignalCase] = &[
            ("finish_before_interrupt_handle", &[Finish], false, &[], false),
            ("done_sender_dropped_before_interrupt_handle", &[DropDoneSender], false, &[], false),
            ("interrupt_sender_dropped_before_deadline", &[DropInterruptSender], false, &[], false),
            (
                "interrupt_handle_then_finish_before_deadline",
                &[PublishInterrupt, Finish],
                false,
                &[],
                false,
            ),
            (
                "interrupt_handle_then_done_sender_drop_before_deadline",
                &[PublishInterrupt, DropDoneSender],
                false,
                &[],
                false,
            ),
            (
                "finish_then_interrupt_handle_before_deadline",
                &[Finish, PublishInterrupt],
                false,
                &[],
                false,
            ),
            (
                "interrupt_handle_before_deadline_then_deadline",
                &[PublishInterrupt],
                true,
                &[],
                true,
            ),
            ("deadline_then_interrupt_handle", &[], true, &[PublishInterrupt], true),
            ("deadline_then_finish", &[], true, &[Finish], true),
            ("deadline_then_interrupt_sender_drop", &[], true, &[DropInterruptSender], true),
            ("deadline_then_done_sender_drop", &[], true, &[DropDoneSender], true),
            (
                "deadline_then_both_senders_drop",
                &[],
                true,
                &[DropDoneSender, DropInterruptSender],
                true,
            ),
            (
                "interrupt_handle_before_deadline_then_finish_after_deadline",
                &[PublishInterrupt],
                true,
                &[Finish],
                true,
            ),
        ];

        for (name, before_deadline, wait_past_deadline, after_deadline, expected_timed_out) in cases
        {
            assert_watchdog_sequence_completes(
                name,
                before_deadline,
                *wait_past_deadline,
                after_deadline,
                *expected_timed_out,
            )
            .await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_watchdog_concurrent_signal_races_do_not_deadlock() {
        for signal_after_deadline in [false, true] {
            for iteration in 0..50 {
                let (_conn, interrupt_handle) = watchdog_interrupt_handle();
                let deadline = if signal_after_deadline {
                    Instant::now() + WATCHDOG_DEADLINE
                } else {
                    Instant::now() + Duration::from_secs(1)
                };
                let mut watchdog = DuckDbQueryWatchdog::spawn(deadline);
                let interrupt_tx =
                    watchdog.interrupt_tx.take().expect("watchdog interrupt sender should exist");
                let done_tx = watchdog.done_tx.take().expect("watchdog done sender should exist");
                let watchdog_task =
                    watchdog.async_task_handle().expect("failed to extract watchdog task");
                let barrier = Arc::new(Barrier::new(3));

                let interrupt_sender = tokio::spawn({
                    let barrier = Arc::clone(&barrier);
                    let interrupt_handle = Arc::clone(&interrupt_handle);
                    async move {
                        barrier.wait().await;
                        let _ = interrupt_tx.send(interrupt_handle);
                    }
                });
                let done_sender = tokio::spawn({
                    let barrier = Arc::clone(&barrier);
                    async move {
                        barrier.wait().await;
                        let _ = done_tx.send(());
                    }
                });

                if signal_after_deadline {
                    tokio::time::sleep(WATCHDOG_DEADLINE * 2).await;
                }

                barrier.wait().await;

                tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, watchdog_task)
                    .await
                    .unwrap_or_else(|_| {
                        panic!(
                            "watchdog concurrent signal race deadlocked: \
                             signal_after_deadline={signal_after_deadline}, iteration={iteration}"
                        )
                    })
                    .unwrap_or_else(|error| {
                        panic!(
                            "watchdog concurrent signal race panicked: \
                             signal_after_deadline={signal_after_deadline}, \
                             iteration={iteration}, error={error}"
                        )
                    });
                interrupt_sender.await.expect("interrupt sender task should not panic");
                done_sender.await.expect("done sender task should not panic");

                assert_eq!(
                    watchdog.timed_out(),
                    signal_after_deadline,
                    "unexpected timeout state for concurrent signal race: \
                     signal_after_deadline={signal_after_deadline}, iteration={iteration}"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_watchdog_slow_interrupt_before_deadline_does_not_deadlock() {
        let (delayed_handle, interrupt_handle) =
            delayed_interrupt_handle(Duration::from_millis(100));
        let mut watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let watchdog_task = watchdog.async_task_handle().expect("failed to extract watchdog task");

        watchdog.publish_query_interrupt_handle(interrupt_handle);

        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, watchdog_task)
            .await
            .expect("watchdog with slow interrupt should not deadlock")
            .expect("watchdog task should not panic");

        assert!(watchdog.timed_out());
        assert_eq!(delayed_handle.calls(), 1);
        assert!(delayed_handle.completed());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_watchdog_slow_interrupt_after_deadline_does_not_deadlock() {
        let (delayed_handle, interrupt_handle) =
            delayed_interrupt_handle(Duration::from_millis(100));
        let mut watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let watchdog_task = watchdog.async_task_handle().expect("failed to extract watchdog task");

        tokio::time::sleep(WATCHDOG_DEADLINE * 2).await;
        watchdog.publish_query_interrupt_handle(interrupt_handle);

        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, watchdog_task)
            .await
            .expect("watchdog with slow late interrupt should not deadlock")
            .expect("watchdog task should not panic");

        assert!(watchdog.timed_out());
        assert_eq!(delayed_handle.calls(), 1);
        assert!(delayed_handle.completed());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_watchdog_finish_while_slow_interrupt_is_running_does_not_deadlock() {
        let (delayed_handle, interrupt_handle) =
            delayed_interrupt_handle(Duration::from_millis(100));
        let mut watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let watchdog_task = watchdog.async_task_handle().expect("failed to extract watchdog task");

        watchdog.publish_query_interrupt_handle(interrupt_handle);
        wait_for_delayed_interrupt_to_start(&delayed_handle).await;
        watchdog.finish();

        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, watchdog_task)
            .await
            .expect("watchdog should finish after slow interrupt returns")
            .expect("watchdog task should not panic");

        assert!(watchdog.timed_out());
        assert_eq!(delayed_handle.calls(), 1);
        assert!(delayed_handle.completed());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn query_watchdog_slow_interrupt_starves_single_worker_runtime_until_it_returns() {
        let interrupt_delay = Duration::from_millis(100);
        let (delayed_handle, interrupt_handle) = delayed_interrupt_handle(interrupt_delay);
        let mut watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let watchdog_task = watchdog.async_task_handle().expect("failed to extract watchdog task");
        let (observer_started_tx, observer_started_rx) = oneshot::channel();
        let started_at = std::time::Instant::now();
        let observer_task = tokio::spawn(async move {
            observer_started_tx.send(()).expect("observer start receiver should be open");
            tokio::time::sleep(WATCHDOG_DEADLINE * 2).await;
            std::time::Instant::now()
        });

        observer_started_rx.await.expect("observer should start");
        watchdog.publish_query_interrupt_handle(interrupt_handle);

        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, watchdog_task)
            .await
            .expect("watchdog should not deadlock on a single worker runtime")
            .expect("watchdog task should not panic");
        let observer_finished_at = observer_task.await.expect("observer task should not panic");

        assert!(watchdog.timed_out());
        assert_eq!(delayed_handle.calls(), 1);
        assert!(delayed_handle.completed());
        assert!(
            observer_finished_at.duration_since(started_at) >= interrupt_delay,
            "single worker runtime should not poll other async tasks while interrupt() blocks"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn query_watchdog_slow_interrupts_serialize_on_single_worker_runtime() {
        let interrupt_delay = Duration::from_millis(100);
        let (first_delayed_handle, first_interrupt_handle) =
            delayed_interrupt_handle(interrupt_delay);
        let (second_delayed_handle, second_interrupt_handle) =
            delayed_interrupt_handle(interrupt_delay);
        let mut first_watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let mut second_watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + WATCHDOG_DEADLINE);
        let first_watchdog_task =
            first_watchdog.async_task_handle().expect("failed to extract first watchdog task");
        let second_watchdog_task =
            second_watchdog.async_task_handle().expect("failed to extract second watchdog task");
        let started_at = std::time::Instant::now();

        first_watchdog.publish_query_interrupt_handle(first_interrupt_handle);
        second_watchdog.publish_query_interrupt_handle(second_interrupt_handle);

        tokio::time::timeout(WATCHDOG_JOIN_TIMEOUT, async {
            let (first_result, second_result) =
                tokio::join!(first_watchdog_task, second_watchdog_task);
            first_result.expect("first watchdog task should not panic");
            second_result.expect("second watchdog task should not panic");
        })
        .await
        .expect("serialized slow interrupts should not deadlock on a single worker runtime");

        assert!(first_watchdog.timed_out());
        assert!(second_watchdog.timed_out());
        assert_eq!(first_delayed_handle.calls(), 1);
        assert_eq!(second_delayed_handle.calls(), 1);
        assert!(first_delayed_handle.completed());
        assert!(second_delayed_handle.completed());
        assert!(
            started_at.elapsed() >= interrupt_delay * 2,
            "single worker runtime should serialize blocking interrupt() calls"
        );
    }

    #[tokio::test]
    async fn query_watchdog_marks_timeout_when_handle_arrives_after_deadline() {
        let opened = make_blocking_test_manager()
            .open_duckdb_connection()
            .expect("failed to open blocking test connection");
        let mut watchdog = DuckDbQueryWatchdog::spawn(Instant::now() + Duration::from_millis(10));
        let watchdog_task = watchdog.async_task_handle().expect("failed to extract watchdog task");

        tokio::time::sleep(Duration::from_millis(20)).await;
        watchdog.publish_interrupt_handle(opened.conn.interrupt_handle());

        watchdog_task.await.expect("watchdog task should not panic");

        assert!(
            watchdog.timed_out(),
            "watchdog should remain timed out when the interrupt handle is published after the \
             deadline"
        );
    }

    #[tokio::test]
    async fn interrupt_all_connections_cancels_running_query() {
        let manager = make_blocking_test_manager();
        let pool = Arc::new(
            build_warm_ducklake_pool(manager.clone(), 1, "test")
                .await
                .expect("failed to build blocking test pool"),
        );
        let blocking_slots = Arc::new(Semaphore::new(1));
        let (query_started_tx, query_started_rx) = oneshot::channel();

        let query_task = tokio::spawn({
            let pool = Arc::clone(&pool);
            let blocking_slots = Arc::clone(&blocking_slots);
            async move {
                run_duckdb_blocking_with_timeout(
                    pool,
                    blocking_slots,
                    Duration::from_secs(30),
                    move |conn| -> EtlResult<()> {
                        let _ = query_started_tx.send(());
                        conn.query_row(
                            "SELECT COUNT(*) FROM range(10000000) t1, range(1000000) t2;",
                            [],
                            |row| row.get::<_, i64>(0),
                        )
                        .map_err(|source| {
                            etl_error!(
                                ErrorKind::DestinationQueryFailed,
                                "DuckLake interrupt test query failed",
                                source: source
                            )
                        })?;
                        Ok(())
                    },
                )
                .await
            }
        });

        query_started_rx.await.expect("query should start before interrupt");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            manager.interrupt_all_connections() >= 1,
            "expected at least one registered connection to be interrupted"
        );

        let error = tokio::time::timeout(Duration::from_secs(5), query_task)
            .await
            .expect("interrupted query should finish promptly")
            .expect("interrupt test task should not panic")
            .expect_err("interrupted query should fail");

        assert_eq!(error.kind(), ErrorKind::DestinationQueryFailed);
        assert_eq!(error.description(), Some("DuckLake interrupt test query failed"));
    }

    #[tokio::test]
    async fn shutdown_requested_skips_new_blocking_operations() {
        let manager = make_blocking_test_manager();
        let pool = Arc::new(
            build_warm_ducklake_pool(manager.clone(), 1, "shutdown-skip-test")
                .await
                .expect("failed to build blocking test pool"),
        );
        let blocking_slots = Arc::new(Semaphore::new(1));

        manager.interrupt_all_connections_for_shutdown();

        let error = run_duckdb_blocking(pool, blocking_slots, |_| Ok(())).await.unwrap_err();
        assert!(is_ducklake_shutdown_requested_error(&error));
    }

    #[tokio::test]
    async fn embedded_initializer_is_shared_and_repeated_on_recreation() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let initialize_calls = Arc::clone(&calls);
        let initialize: ConnectionInitializer = Arc::new(move || {
            let generation = initialize_calls.fetch_add(1, Ordering::SeqCst);
            let connection = duckdb::Connection::open_in_memory().map_err(|source| {
                etl_error!(ErrorKind::DestinationConnectionFailed, "Open failed", source: source)
            })?;
            connection.execute_batch(&format!(
                "create table generation as select {generation} as id;"
            )).map_err(|source| {
                etl_error!(ErrorKind::DestinationQueryFailed, "Setup failed", source: source)
            })?;
            Ok(connection)
        });
        let manager = DuckLakeConnectionManager::new_with_initializer(
            Arc::new(DuckLakeSetupPlan::for_embedded_instance()),
            false,
            Arc::new(DuckLakeInterruptRegistry::default()),
            Arc::new(AtomicBool::new(false)),
            Some(initialize),
        )
        .await
        .unwrap();
        let copy_manager = manager.new_pool_manager();
        let first = manager.open_duckdb_connection().unwrap();
        first.conn.execute_batch("insert into generation values (42)").unwrap();
        let second = copy_manager.open_duckdb_connection().unwrap();
        assert_eq!(
            second
                .conn
                .query_row("select count(*) from generation", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(first);
        drop(second);
        manager.recreate_shared_instance().await.unwrap();
        let recreated = copy_manager.open_duckdb_connection().unwrap();
        assert_eq!(
            recreated
                .conn
                .query_row("select id from generation", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn embedded_initializer_propagates_failures() {
        let result = DuckLakeConnectionManager::new_with_initializer(
            Arc::new(DuckLakeSetupPlan::for_embedded_instance()),
            false,
            Arc::new(DuckLakeInterruptRegistry::default()),
            Arc::new(AtomicBool::new(false)),
            Some(Arc::new(|| {
                Err(etl_error!(ErrorKind::ConfigError, "Host initialization failed"))
            })),
        )
        .await;
        let error = result.err().unwrap();
        assert_eq!(error.kind(), ErrorKind::ConfigError);
        assert!(error.to_string().contains("Host initialization failed"));
    }
}
