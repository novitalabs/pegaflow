mod check_cuda_version;
pub mod fd_channel;
pub mod http_server;
pub mod metric;
pub mod proto;
pub mod registry;
pub mod service;
pub mod session;
#[cfg(feature = "tracing")]
mod trace;
#[cfg(not(feature = "tracing"))]
mod trace {
    pub(crate) fn init() {}
    pub(crate) fn flush() {}
}
mod utils;

pub use registry::{CudaTensorRegistry, RegistryHandle};
pub use service::GrpcEngineService;

use clap::Parser;
use cudarc::driver::result as cuda_driver;
use log::{error, info, warn};
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use pegaflow_common::grpc::{
    GRPC_SERVER_HTTP2_KEEPALIVE_INTERVAL, GRPC_SERVER_HTTP2_KEEPALIVE_TIMEOUT,
};
use pegaflow_core::PegaEngine;
use prometheus::Registry;
use proto::engine::engine_server::EngineServer;
use pyo3::{PyErr, Python, types::PyAnyMethods};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tonic::transport::Server;
use utils::parse_memory_size;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(
    name = "pega-engine-server",
    version,
    about = "PegaEngine gRPC server with CUDA IPC registry"
)]
pub struct Cli {
    /// Address to bind, e.g. 0.0.0.0:50055
    #[arg(long, default_value = "127.0.0.1:50055")]
    pub addr: SocketAddr,

    /// CUDA devices to initialize (comma-separated, e.g., "0,1,2,3").
    /// If not specified, auto-detects and initializes all available GPUs.
    #[arg(long, value_delimiter = ',')]
    pub devices: Vec<i32>,

    /// Pinned memory pool size (supports units: kb, mb, gb, tb)
    /// Examples: "10gb", "500mb", "1tb"
    #[arg(long, default_value = "30gb", value_parser = parse_memory_size)]
    pub pool_size: usize,

    /// Hint for typical value size (supports units: kb, mb, gb, tb); tunes cache + allocator
    #[arg(long, value_parser = parse_memory_size)]
    pub hint_value_size: Option<usize>,

    /// Use huge pages for pinned memory pool (faster allocation).
    /// Requires pre-configured huge pages via /proc/sys/vm/nr_hugepages
    #[arg(long, default_value_t = false)]
    pub use_hugepages: bool,

    /// Enable TinyLFU admission policy for cache (default: plain LRU)
    #[arg(long, default_value_t = false)]
    pub enable_lfu_admission: bool,

    /// Disable NUMA-aware memory allocation (use single pool instead of per-node pools)
    #[arg(long, default_value_t = false)]
    pub disable_numa_affinity: bool,

    /// HTTP server address for health check and Prometheus metrics.
    /// Always enabled for health check endpoint.
    #[arg(long, default_value = "0.0.0.0:9091")]
    pub http_addr: SocketAddr,

    /// Enable Prometheus /metrics endpoint on the HTTP server.
    #[arg(long, default_value_t = true)]
    pub enable_prometheus: bool,

    /// UDS path for native clients to send VMM allocation fds (SCM_RIGHTS). Empty disables.
    #[arg(long, default_value = "/tmp/pegaflow-fd.sock")]
    pub fd_socket_path: String,

    /// Init torch CUDA registry (vLLM path). False = torch-free, native VMM only.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub python_registry: bool,

    /// Enable OTLP metrics export over gRPC (e.g. http://127.0.0.1:4317).
    #[arg(long)]
    pub metrics_otel_endpoint: Option<String>,

    /// Period (seconds) for exporting OTLP metrics (only used when endpoint is set).
    #[arg(long, default_value_t = 10)]
    pub metrics_period_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Enable SSD cache for sealed blocks. Provide one or more cache directories.
    /// Repeat the flag to use multiple paths (e.g. one per SSD device).
    #[arg(long, num_args = 1..)]
    pub ssd_cache_path: Vec<String>,

    /// SSD cache capacity (supports units: kb, mb, gb, tb). Default: 512gb
    #[arg(long, default_value = "512gb", value_parser = parse_memory_size)]
    pub ssd_cache_capacity: usize,

    /// SSD cache file shards per path. Use >1 for parallel filesystems such as GPFS.
    /// When multiple --ssd-cache-path values are given each path receives this many
    /// shards so that every device is utilised. Default: 1
    #[arg(long, default_value = "1")]
    pub ssd_cache_shards: std::num::NonZeroUsize,

    /// SSD write queue depth (max pending write batches). Default: 8
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_WRITE_QUEUE_DEPTH)]
    pub ssd_write_queue_depth: usize,

    /// SSD prefetch queue depth (max pending prefetch batches). Default: 2
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_PREFETCH_QUEUE_DEPTH)]
    pub ssd_prefetch_queue_depth: usize,

    /// SSD write inflight (max concurrent block writes). Default: 2
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_WRITE_INFLIGHT)]
    pub ssd_write_inflight: usize,

    /// SSD prefetch inflight (max concurrent block reads). Default: 16
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_PREFETCH_INFLIGHT)]
    pub ssd_prefetch_inflight: usize,

    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch). Default: 1500
    #[arg(long, default_value_t = 800)]
    pub max_prefetch_blocks: usize,

    /// Trace sampling rate (0.0–1.0). E.g. 0.01 = 1%. Default: 1.0 (100%)
    #[arg(long, default_value_t = 1.0, value_parser = parse_sample_rate)]
    pub trace_sample_rate: f64,

    /// Number of shards for the pinned memory pool (reduces allocator lock contention).
    /// The pool is split into this many independent sub-pools with round-robin allocation.
    #[arg(long, default_value_t = 1)]
    pub pool_shards: usize,

    /// Allocate each block separately instead of contiguous batch allocation.
    /// Reduces memory fragmentation when blocks are freed in different order.
    #[arg(long, default_value_t = false)]
    pub blockwise_alloc: bool,

    /// RDMA NIC names for inter-node transfer (e.g. --nics mlx5_0,mlx5_1 or --nics mlx5_0 mlx5_1).
    /// When set, pinned memory is registered for RDMA access on these NICs.
    #[arg(long, value_delimiter = ',', value_parser = parse_nic_name, num_args = 1..)]
    pub nics: Option<Vec<String>>,

    /// Number of RC QPs per (local NIC, remote NIC) pair.
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_RDMA_QPS_PER_PEER)]
    pub qps_per_peer: usize,

    /// MetaServer address for cross-node block hash registration (e.g. http://127.0.0.1:50056).
    /// When set, sealed block hashes are automatically registered with the MetaServer.
    #[arg(long)]
    pub metaserver_addr: Option<String>,

    /// MetaServer command queue depth (pending insert/remove commands, shared channel).
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_METASERVER_QUEUE_DEPTH)]
    pub metaserver_queue_depth: usize,

    /// HLL sliding-window list for hit-rate estimation. Comma-separated humantime
    /// durations; each becomes a canonical `window` label in metrics (e.g. `15m,1h,1d`).
    /// Slot duration is derived as `clamp(window/24, 1min, 1h)`.
    #[arg(long, default_value = "15m,1h,24h", value_parser = parse_hll_windows_arg)]
    pub metric_hll_windows: String,

    /// HLL bucket index bits 4–18 (default: 14 → 16384 buckets, ~0.8% error)
    #[arg(long, default_value_t = 14, value_parser = parse_hll_bucket_bits)]
    pub metric_hll_bucket_bits: u8,

    /// Transfer lock timeout in seconds. Blocks held for cross-node RDMA transfer are
    /// locked for at most this duration before being force-released (crash recovery).
    #[arg(long, default_value_t = 120)]
    pub transfer_lock_timeout_secs: u64,
}

fn parse_hll_bucket_bits(s: &str) -> Result<u8, String> {
    use pegaflow_common::hll::{MAX_BUCKET_BITS, MIN_BUCKET_BITS};
    let v: u8 = s.parse().map_err(|e| format!("{e}"))?;
    if !(MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&v) {
        return Err(format!(
            "HLL bucket_bits must be in {MIN_BUCKET_BITS}..={MAX_BUCKET_BITS}, got {v}"
        ));
    }
    Ok(v)
}

fn parse_nic_name(s: &str) -> Result<String, String> {
    let name = s.trim();
    if name.is_empty() {
        return Err("--nics contains an empty NIC name".into());
    }
    Ok(name.to_string())
}

fn parse_hll_windows_arg(s: &str) -> Result<String, String> {
    parse_hll_windows(s)?;
    Ok(s.to_string())
}

/// Parse a comma-separated list of humantime windows (e.g. `15m,1h,24h`).
/// Each entry becomes `(label, duration)` where label is canonicalized from
/// the parsed duration.
fn parse_hll_windows(s: &str) -> Result<Vec<(String, Duration)>, String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (idx, token) in s.split(',').map(str::trim).enumerate() {
        if token.is_empty() {
            return Err(format!(
                "--metric-hll-windows contains an empty window at position {}",
                idx + 1
            ));
        }
        let dur = parse_humantime(token)?;
        if dur < Duration::from_secs(60) {
            return Err(format!("HLL window {token} must be at least 1 minute"));
        }
        if !seen.insert(dur) {
            return Err(format!(
                "duplicate HLL window duration: {token} ({})",
                format_hll_window_label(dur)
            ));
        }
        out.push((format_hll_window_label(dur), dur));
    }
    if out.is_empty() {
        return Err("--metric-hll-windows must list at least one window".into());
    }
    Ok(out)
}

/// Minimal humantime parser: `<number><unit>` where unit ∈ {s, m, h, d}.
fn parse_humantime(s: &str) -> Result<Duration, String> {
    let (num_str, unit) = s.split_at(
        s.find(|c: char| !c.is_ascii_digit())
            .ok_or_else(|| format!("missing time unit in {s}"))?,
    );
    let n: u64 = num_str
        .parse()
        .map_err(|e| format!("invalid number in {s}: {e}"))?;
    let unit_secs = match unit {
        "s" => n,
        "m" => n
            .checked_mul(60)
            .ok_or_else(|| format!("duration overflows seconds in {s}"))?,
        "h" => n
            .checked_mul(3600)
            .ok_or_else(|| format!("duration overflows seconds in {s}"))?,
        "d" => n
            .checked_mul(86400)
            .ok_or_else(|| format!("duration overflows seconds in {s}"))?,
        _ => return Err(format!("unknown time unit {unit:?} in {s}; use s/m/h/d")),
    };
    Ok(Duration::from_secs(unit_secs))
}

fn format_hll_window_label(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs.is_multiple_of(86400) {
        format!("{}d", secs / 86400)
    } else if secs.is_multiple_of(3600) {
        format!("{}h", secs / 3600)
    } else if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

fn parse_sample_rate(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("sample rate must be between 0.0 and 1.0, got {v}"));
    }
    Ok(v)
}

fn format_py_err(err: PyErr) -> String {
    Python::attach(|py| err.value(py).to_string())
}

fn init_cuda_driver() -> Result<(), std::io::Error> {
    cuda_driver::init()
        .map_err(|err| std::io::Error::other(format!("failed to initialize CUDA driver: {err}")))
}

fn detect_cuda_devices() -> Result<Vec<i32>, std::io::Error> {
    Python::attach(|py| -> pyo3::PyResult<Vec<i32>> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        let device_count: i32 = cuda.call_method0("device_count")?.extract()?;

        // Probe each device ID from 0 to device_count-1 to see if it's available
        let mut available_devices = Vec::new();
        for device_id in 0..device_count {
            // Try to get device properties to verify it's accessible
            match cuda.call_method1("get_device_properties", (device_id,)) {
                Ok(_) => available_devices.push(device_id),
                Err(_) => continue, // Skip unavailable devices
            }
        }
        Ok(available_devices)
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to detect CUDA devices: {}",
            format_py_err(err)
        ))
    })
}

fn init_python_cuda(device_ids: &[i32]) -> Result<(), std::io::Error> {
    if device_ids.is_empty() {
        return Err(std::io::Error::other("no CUDA devices to initialize"));
    }

    Python::attach(|py| -> pyo3::PyResult<()> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        cuda.call_method0("init")?;

        // Initialize CUDA context for each device by performing a real CUDA operation
        // PyTorch uses lazy initialization, so we need to actually allocate something
        // to force context creation on each device
        for &device_id in device_ids {
            let start = std::time::Instant::now();
            cuda.call_method1("set_device", (device_id,))?;

            // Allocate a small tensor to force CUDA context creation on this device
            // This ensures the CUDA driver creates a context for the device
            let device_str = format!("cuda:{}", device_id);
            let empty_args = (vec![1i64],);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("device", device_str)?;
            let _ = torch.call_method("empty", empty_args, Some(&kwargs))?;

            // Synchronize to ensure context is fully initialized
            cuda.call_method0("synchronize")?;

            let elapsed = start.elapsed();
            log::info!(
                "Initialized CUDA context for device {} in {:.2}s",
                device_id,
                elapsed.as_secs_f64()
            );
        }

        // Set the first device as the default
        cuda.call_method1("set_device", (device_ids[0],))?;
        Ok(())
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to initialize python/tensor CUDA runtime: {}",
            format_py_err(err)
        ))
    })
}

/// Primary CUDA contexts via cudarc when `--python-registry false`.
fn init_cudarc_cuda(device_ids: &[i32]) -> Result<(), std::io::Error> {
    if device_ids.is_empty() {
        return Err(std::io::Error::other("no CUDA devices to initialize"));
    }
    for &device_id in device_ids {
        let ordinal = usize::try_from(device_id)
            .map_err(|_| std::io::Error::other(format!("device_id {device_id} must be >= 0")))?;
        let ctx = cudarc::driver::CudaContext::new(ordinal)
            .map_err(|e| std::io::Error::other(format!("cudarc init device {device_id}: {e}")))?;
        ctx.bind_to_thread()
            .map_err(|e| std::io::Error::other(format!("cudarc bind device {device_id}: {e}")))?;
        info!("Initialized CUDA context for device {device_id} (cudarc, torch-free)");
    }
    Ok(())
}

struct MetricsState {
    meter_provider: Option<SdkMeterProvider>,
    prometheus_registry: Option<Registry>,
}

fn init_metrics(
    prometheus_enabled: bool,
    otlp_endpoint: Option<String>,
    otlp_period_secs: u64,
) -> Result<MetricsState, Box<dyn Error>> {
    let otlp_endpoint = otlp_endpoint.filter(|s| !s.is_empty());

    // If neither Prometheus nor OTLP is enabled, return empty state
    if !prometheus_enabled && otlp_endpoint.is_none() {
        info!("Metrics disabled (no Prometheus addr or OTLP endpoint configured)");
        return Ok(MetricsState {
            meter_provider: None,
            prometheus_registry: None,
        });
    }

    let mut builder = SdkMeterProvider::builder();
    let mut prometheus_registry = None;

    // Add Prometheus exporter if enabled
    if prometheus_enabled {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()?;
        builder = builder.with_reader(exporter);
        prometheus_registry = Some(registry);
        info!("Prometheus metrics exporter enabled");
    }

    // Add OTLP exporter if endpoint is configured
    if let Some(endpoint) = otlp_endpoint {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()?;

        let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
            .with_interval(Duration::from_secs(otlp_period_secs))
            .build();

        builder = builder.with_reader(reader);
        info!(
            "OTLP metrics exporter enabled (period={}s)",
            otlp_period_secs
        );
    }

    let meter_provider = builder.build();
    global::set_meter_provider(meter_provider.clone());

    Ok(MetricsState {
        meter_provider: Some(meter_provider),
        prometheus_registry,
    })
}

/// Main entry point for pegaflow-server
pub fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    pegaflow_common::logging::init_stdout_colored(&cli.log_level);
    info!("Starting pega-engine-server v{}", env!("CARGO_PKG_VERSION"));
    trace::init();
    pegaflow_core::set_trace_sample_rate(cli.trace_sample_rate);

    // Initialize CUDA in the main thread before starting Tokio runtime
    init_cuda_driver()?;
    check_cuda_version::preflight()?;

    // Determine which devices to initialize
    let devices = if cli.devices.is_empty() {
        // Auto-detect all available devices
        let detected = detect_cuda_devices()?;
        info!(
            "Auto-detected {} CUDA device(s): {:?}",
            detected.len(),
            detected
        );
        detected
    } else {
        info!("Using specified CUDA device(s): {:?}", cli.devices);
        cli.devices.clone()
    };

    if devices.is_empty() {
        return Err("No CUDA devices available".into());
    }

    if cli.python_registry {
        init_python_cuda(&devices)?;
    } else {
        init_cudarc_cuda(&devices)?;
    }
    info!(
        "CUDA runtime initialized for {} device(s): {:?}",
        devices.len(),
        devices
    );

    let registry = if cli.python_registry {
        CudaTensorRegistry::new().map_err(|err| {
            let msg = format_py_err(err);
            std::io::Error::other(format!("failed to initialize torch CUDA context: {msg}"))
        })?
    } else {
        info!("Python/torch registry disabled; serving native VMM clients only");
        CudaTensorRegistry::empty()
    };
    // Confine the registry to its own thread: GIL + CUDA work now happens off
    // the async runtime, so a wedged `empty_cache` can't starve tokio workers.
    let registry = RegistryHandle::spawn(registry);

    if let Some(hint_value_size) = cli.hint_value_size {
        if hint_value_size == 0 {
            return Err("--hint-value-size must be greater than zero when set".into());
        }
        info!("Value size hint set to {} bytes", hint_value_size);
    }

    info!(
        "Creating PegaEngine with pinned memory pool: {:.2} GiB ({} bytes), hugepages={}",
        cli.pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        cli.pool_size,
        cli.use_hugepages
    );

    let ssd_cache_config = if cli.ssd_cache_path.is_empty() {
        None
    } else {
        info!(
            "SSD cache enabled: paths={}, capacity={:.2} GiB, shards={}, write_queue={}, prefetch_queue={}, write_inflight={}, prefetch_inflight={}",
            cli.ssd_cache_path.join(", "),
            cli.ssd_cache_capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.ssd_cache_shards,
            cli.ssd_write_queue_depth,
            cli.ssd_prefetch_queue_depth,
            cli.ssd_write_inflight,
            cli.ssd_prefetch_inflight,
        );
        Some(pegaflow_core::SsdCacheConfig {
            cache_paths: cli.ssd_cache_path.iter().map(|p| p.into()).collect(),
            capacity_bytes: cli.ssd_cache_capacity as u64,
            shards: cli.ssd_cache_shards,
            write_queue_depth: cli.ssd_write_queue_depth,
            prefetch_queue_depth: cli.ssd_prefetch_queue_depth,
            write_inflight: cli.ssd_write_inflight,
            prefetch_inflight: cli.ssd_prefetch_inflight,
        })
    };

    let has_metaserver = cli.metaserver_addr.is_some();
    let has_nics = cli.nics.as_ref().is_some_and(|n| !n.is_empty());

    if has_metaserver != has_nics {
        log::warn!(
            "--metaserver-addr and --nics should be set together (got metaserver={}, nics={})",
            has_metaserver,
            has_nics,
        );
    }

    let advertise_addr = if has_metaserver {
        if cli.addr.ip().is_unspecified() || cli.addr.ip().is_loopback() {
            log::warn!(
                "P2P: --addr is {}, other nodes may not be able to reach this server",
                cli.addr.ip()
            );
        }
        Some(cli.addr.to_string())
    } else {
        None
    };

    let storage_config = pegaflow_core::StorageConfig {
        enable_lfu_admission: cli.enable_lfu_admission,
        hint_value_size_bytes: cli.hint_value_size,
        max_prefetch_blocks: cli.max_prefetch_blocks,
        ssd_cache_config,
        rdma_nic_names: cli.nics.clone(),
        rdma_qps_per_peer: cli.qps_per_peer,
        enable_numa_affinity: !cli.disable_numa_affinity,
        blockwise_alloc: cli.blockwise_alloc,
        transfer_lock_timeout: Duration::from_secs(cli.transfer_lock_timeout_secs),
        metaserver_addr: cli.metaserver_addr.clone(),
        advertise_addr,
        metaserver_queue_depth: cli.metaserver_queue_depth,
        pool_shards: cli.pool_shards,
    };

    if cli.pool_shards > 1 {
        info!(
            "Pinned memory pool sharding enabled: {} shards",
            cli.pool_shards
        );
    }
    if cli.enable_lfu_admission {
        info!("TinyLFU cache admission enabled");
    }
    if cli.disable_numa_affinity {
        info!("NUMA-aware memory allocation disabled");
    }

    // Create Tokio runtime early - needed for OTLP metrics gRPC exporter
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Initialize OTEL metrics BEFORE creating PegaEngine, so that core metrics
    // (pool, cache, save/load) use the real meter provider instead of noop.
    let metrics_state = runtime.block_on(async {
        init_metrics(
            cli.enable_prometheus,
            cli.metrics_otel_endpoint.clone(),
            cli.metrics_period_secs,
        )
    })?;

    let hll_tracker = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::MultiWindowHllTracker::new(
            parse_hll_windows(&cli.metric_hll_windows)
                .map_err(|err| format!("invalid --metric-hll-windows: {err}"))?,
            cli.metric_hll_bucket_bits,
        ),
    ));
    crate::metric::register_hll_gauges(&hll_tracker);

    let shutdown = Arc::new(Notify::new());

    runtime.block_on(async move {
        // Create PegaEngine inside tokio runtime context (needed for SSD cache tokio::spawn)
        let engine = Arc::new(PegaEngine::new_with_config(
            cli.pool_size,
            cli.use_hugepages,
            storage_config,
        )?);

        let fd_channel = if cli.fd_socket_path.is_empty() {
            None
        } else {
            Some(
                crate::fd_channel::FdChannel::bind(cli.fd_socket_path.clone())
                    .map_err(|e| format!("bind fd side-channel {}: {e}", cli.fd_socket_path))?,
            )
        };

        let service = GrpcEngineService::new(
            Arc::clone(&engine),
            registry.clone(),
            Arc::clone(&shutdown),
            Arc::clone(&hll_tracker),
            fd_channel,
        );

        // Spawn background GC task for stale inflight blocks and expired transfer locks
        {
            let engine = Arc::clone(&engine);
            let shutdown = Arc::clone(&shutdown);
            tokio::spawn(async move {
                const GC_INTERVAL: Duration = Duration::from_secs(60);
                const INFLIGHT_MAX_AGE: Duration = Duration::from_secs(300); // 5 min
                const FAILED_REMOTE_MAX_AGE: Duration = Duration::from_secs(300); // 5 min
                let mut interval = tokio::time::interval(GC_INTERVAL);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            let (cleaned, failed) = engine
                                .gc_stale_inflight(INFLIGHT_MAX_AGE, FAILED_REMOTE_MAX_AGE)
                                .await;
                            if cleaned > 0 {
                                info!("Inflight GC: cleaned {} stale blocks", cleaned);
                            }
                            if failed > 0 {
                                info!("Inflight GC: cleared {} stale failed_remote entries", failed);
                            }

                            let expired_locks = engine.gc_expired_transfer_locks();
                            if expired_locks > 0 {
                                warn!("Transfer lock GC: expired {} stale sessions", expired_locks);
                            }
                        }
                        _ = shutdown.notified() => {
                            info!("Background GC task shutting down");
                            break;
                        }
                    }
                }
            });
            info!("Background GC task started (interval=60s, inflight_max_age=5m, failed_remote_max_age=5m)");
        }

        // Start HTTP server for health check (always enabled)
        let http_server_handle = http_server::start_http_server(
            cli.http_addr,
            Arc::clone(&engine),
            registry,
            cli.enable_prometheus,
            metrics_state.prometheus_registry.clone(),
            Arc::clone(&shutdown),
        )
        .await?;

        let shutdown_signal = {
            let notify = Arc::clone(&shutdown);
            async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("Ctrl+C received, shutting down");
                    }
                    _ = notify.notified() => {
                        info!("Shutdown requested via RPC");
                    }
                }
            }
        };

        info!("PegaEngine gRPC server listening on {}", cli.addr);

        const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024; // 64 MiB

        let grpc_service = EngineServer::new(service)
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);

        if let Err(err) = Server::builder()
            .http2_keepalive_interval(Some(GRPC_SERVER_HTTP2_KEEPALIVE_INTERVAL))
            .http2_keepalive_timeout(Some(GRPC_SERVER_HTTP2_KEEPALIVE_TIMEOUT))
            .add_service(grpc_service)
            .serve_with_shutdown(cli.addr, shutdown_signal)
            .await
        {
            error!("Server error: {err}");
            return Err(err.into());
        }

        info!("Server stopped");

        // Stop HTTP server
        shutdown.notify_waiters();
        let _ = http_server_handle.await;

        engine.shutdown_metaserver_client().await;

        // Flush metrics before exit
        if let Some(provider) = metrics_state.meter_provider
            && let Err(err) = provider.shutdown()
        {
            error!("Failed to shutdown metrics provider: {err}");
        }

        trace::flush();

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hll_windows_canonicalizes_labels() {
        let windows = parse_hll_windows("15m,60m,24h").unwrap();

        assert_eq!(windows, expected_hll_windows());
    }

    #[test]
    fn cli_default_metric_hll_windows_parses_without_panic() {
        let cli = Cli::try_parse_from(["pegaflow-server"]).unwrap();

        assert_eq!(
            parse_hll_windows(&cli.metric_hll_windows).unwrap(),
            expected_hll_windows()
        );
    }

    #[test]
    fn cli_nics_accepts_comma_separated_values() {
        let cli =
            Cli::try_parse_from(["pegaflow-server", "--nics", "mlx5_0,mlx5_1,mlx5_2"]).unwrap();

        assert_eq!(
            cli.nics.unwrap(),
            [
                "mlx5_0".to_string(),
                "mlx5_1".to_string(),
                "mlx5_2".to_string()
            ]
        );
    }

    #[test]
    fn cli_nics_trims_comma_separated_values() {
        let cli =
            Cli::try_parse_from(["pegaflow-server", "--nics", "mlx5_0, mlx5_1, mlx5_2"]).unwrap();

        assert_eq!(
            cli.nics.unwrap(),
            [
                "mlx5_0".to_string(),
                "mlx5_1".to_string(),
                "mlx5_2".to_string()
            ]
        );
    }

    #[test]
    fn cli_nics_rejects_empty_comma_separated_values() {
        let err = Cli::try_parse_from(["pegaflow-server", "--nics", "mlx5_0,,mlx5_1"]).unwrap_err();

        assert!(err.to_string().contains("empty NIC name"), "{err}");
    }

    #[test]
    fn cli_nics_accepts_repeated_values_after_flag() {
        let cli = Cli::try_parse_from(["pegaflow-server", "--nics", "mlx5_0", "mlx5_1", "mlx5_2"])
            .unwrap();

        assert_eq!(
            cli.nics.unwrap(),
            [
                "mlx5_0".to_string(),
                "mlx5_1".to_string(),
                "mlx5_2".to_string()
            ]
        );
    }

    #[test]
    fn cli_explicit_metric_hll_windows_parses_without_panic() {
        let cli =
            Cli::try_parse_from(["pegaflow-server", "--metric-hll-windows", "15m,1h,24h"]).unwrap();

        assert_eq!(
            parse_hll_windows(&cli.metric_hll_windows).unwrap(),
            expected_hll_windows()
        );
    }

    #[test]
    fn cli_rejects_invalid_metric_hll_windows() {
        let err = Cli::try_parse_from(["pegaflow-server", "--metric-hll-windows", "15m,,1h"])
            .unwrap_err();

        assert!(err.to_string().contains("empty window"), "{err}");
    }

    fn expected_hll_windows() -> Vec<(String, Duration)> {
        vec![
            ("15m".to_string(), Duration::from_secs(15 * 60)),
            ("1h".to_string(), Duration::from_secs(60 * 60)),
            ("1d".to_string(), Duration::from_secs(24 * 60 * 60)),
        ]
    }

    #[test]
    fn parse_hll_windows_rejects_empty_tokens() {
        let err = parse_hll_windows("15m,,1h").unwrap_err();

        assert!(err.contains("empty window"), "{err}");
    }

    #[test]
    fn parse_hll_windows_rejects_duplicate_durations() {
        let err = parse_hll_windows("1h,60m").unwrap_err();

        assert!(err.contains("duplicate HLL window duration"), "{err}");
    }
}
