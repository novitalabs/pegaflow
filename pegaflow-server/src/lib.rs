pub mod http_server;
pub mod metric;
pub mod proto;
pub mod registry;
pub mod service;
#[cfg(feature = "tracing")]
mod trace;
#[cfg(not(feature = "tracing"))]
mod trace {
    pub fn init() {}
    pub fn flush() {}
}
mod utils;

pub use registry::CudaTensorRegistry;
pub use service::GrpcEngineService;

use clap::Parser;
use cudarc::driver::result as cuda_driver;
use log::{error, info};
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use parking_lot::Mutex;
use pegaflow_core::internode::PegaflowClientPool;
use pegaflow_core::internode::metaserver_query::MetaServerQueryClient;
use pegaflow_core::internode::remote_fetch_worker::{
    RdmaBatchReadFn, RdmaRegisterMemoryFn, RemoteFetchWorkerConfig, execute_remote_fetch,
};
use pegaflow_core::{PegaEngine, RemoteFetchFn};
use pegaflow_transfer::{
    HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
use prometheus::Registry;
use proto::engine::engine_server::EngineServer;
use pyo3::{PyErr, Python, types::PyAnyMethods};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
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

    /// Enable OTLP metrics export over gRPC (e.g. http://127.0.0.1:4317).
    #[arg(long)]
    pub metrics_otel_endpoint: Option<String>,

    /// Period (seconds) for exporting OTLP metrics (only used when endpoint is set).
    #[arg(long, default_value_t = 10)]
    pub metrics_period_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Enable SSD cache for sealed blocks. Provide the cache file path to enable.
    #[arg(long)]
    pub ssd_cache_path: Option<String>,

    /// SSD cache capacity (supports units: kb, mb, gb, tb). Default: 512gb
    #[arg(long, default_value = "512gb", value_parser = parse_memory_size)]
    pub ssd_cache_capacity: usize,

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

    /// Allocate each block separately instead of contiguous batch allocation.
    /// Reduces memory fragmentation when blocks are freed in different order.
    #[arg(long, default_value_t = false)]
    pub blockwise_alloc: bool,

    /// RDMA NIC names for inter-node transfer (e.g. --nics mlx5_0 mlx5_1).
    /// When set, pinned memory is registered for RDMA access on these NICs.
    #[arg(long, num_args = 1..)]
    pub nics: Option<Vec<String>>,

    /// MetaServer address for cross-node block hash registration (e.g. http://127.0.0.1:50056).
    /// When set, enables block registration, cross-node remote fetch, and RDMA transfer.
    /// Requires --advertise-addr and --rdma-nic to also be set.
    #[arg(long, requires_all = ["advertise_addr", "rdma_nic"])]
    pub metaserver_addr: Option<String>,

    /// Advertise address for this node in MetaServer registrations (e.g. 10.0.0.1:50055).
    /// Required when --metaserver-addr is set.
    #[arg(long)]
    pub advertise_addr: Option<String>,

    /// MetaServer registration queue depth (max pending registration batches).
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_METASERVER_QUEUE_DEPTH)]
    pub metaserver_queue_depth: usize,

    /// RDMA NIC name for the transfer engine (e.g. "mlx5_0").
    /// Required when --metaserver-addr is set.
    #[arg(long)]
    pub rdma_nic: Option<String>,

    /// RDMA transfer engine RPC port for session establishment.
    #[arg(long, default_value_t = 50057)]
    pub rdma_port: u16,

    /// Transfer lock timeout in seconds. Blocks locked for RDMA transfer
    /// are auto-released after this duration.
    #[arg(long, default_value_t = 30)]
    pub transfer_lock_timeout_secs: u64,

    /// Max blocks in remote fetch in-flight (backpressure limit).
    #[arg(long, default_value_t = 200)]
    pub max_remote_fetch_blocks: usize,
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

struct RemoteFetchRuntime {
    remote_fetch_fn: RemoteFetchFn,
    engine_cell: Arc<OnceLock<Arc<PegaEngine>>>,
    rdma_engine: Arc<TransferEngine>,
    rdma_session_id: Vec<u8>,
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

    init_python_cuda(&devices)?;
    info!(
        "CUDA runtime initialized for {} device(s): {:?}",
        devices.len(),
        devices
    );

    let registry = CudaTensorRegistry::new().map_err(|err| {
        let msg = format_py_err(err);
        std::io::Error::other(format!("failed to initialize torch CUDA context: {msg}"))
    })?;
    let registry = Arc::new(Mutex::new(registry));

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

    let ssd_cache_config = cli.ssd_cache_path.as_ref().map(|path| {
        info!(
            "SSD cache enabled: path={}, capacity={:.2} GiB, write_queue={}, prefetch_queue={}, write_inflight={}, prefetch_inflight={}",
            path,
            cli.ssd_cache_capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.ssd_write_queue_depth,
            cli.ssd_prefetch_queue_depth,
            cli.ssd_write_inflight,
            cli.ssd_prefetch_inflight,
        );
        pegaflow_core::SsdCacheConfig {
            cache_path: path.into(),
            capacity_bytes: cli.ssd_cache_capacity as u64,
            write_queue_depth: cli.ssd_write_queue_depth,
            prefetch_queue_depth: cli.ssd_prefetch_queue_depth,
            write_inflight: cli.ssd_write_inflight,
            prefetch_inflight: cli.ssd_prefetch_inflight,
        }
    });

    let remote_fetch_enabled = cli.metaserver_addr.is_some();

    let storage_config = pegaflow_core::StorageConfig {
        enable_lfu_admission: cli.enable_lfu_admission,
        hint_value_size_bytes: cli.hint_value_size,
        max_prefetch_blocks: cli.max_prefetch_blocks,
        ssd_cache_config,
        rdma_nic_names: cli.nics.clone(),
        enable_numa_affinity: !cli.disable_numa_affinity,
        blockwise_alloc: cli.blockwise_alloc,
        transfer_lock_timeout: if remote_fetch_enabled {
            Some(Duration::from_secs(cli.transfer_lock_timeout_secs))
        } else {
            None
        },
        max_remote_fetch_blocks: if remote_fetch_enabled {
            Some(cli.max_remote_fetch_blocks)
        } else {
            None
        },
    };

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

    let shutdown = Arc::new(Notify::new());

    runtime.block_on(async move {
        // Create MetaServer registrar if --metaserver-addr is set
        let metaserver_registrar = cli.metaserver_addr.as_ref().map(|addr| {
            // Safe to unwrap: clap `requires_all` guarantees advertise_addr is set
            let advertise = cli.advertise_addr.clone().unwrap();
            info!(
                "MetaServer registration enabled: metaserver={}, advertise={}, queue_depth={}",
                addr, advertise, cli.metaserver_queue_depth
            );
            let config = pegaflow_core::MetaServerRegistrarConfig::new(addr.clone(), advertise)
                .with_queue_depth(cli.metaserver_queue_depth);
            Arc::new(pegaflow_core::MetaServerRegistrar::new(config))
        });

        let remote_fetch_runtime = if let Some(addr) = cli.metaserver_addr.as_ref() {
            let rdma_nic = cli
                .rdma_nic
                .clone()
                .expect("clap requires rdma_nic when metaserver_addr is set");
            info!(
                "Remote RDMA fetch enabled: metaserver={} rdma_nic={} rdma_port={}",
                addr, rdma_nic, cli.rdma_port
            );

            let metaserver_client = Arc::new(MetaServerQueryClient::connect(addr).await?);
            let client_pool = Arc::new(PegaflowClientPool::without_registry());

            let transfer_engine = TransferEngine::new(&[rdma_nic]).map_err(|err| {
                std::io::Error::other(format!("failed to initialize RDMA transfer engine: {err}"))
            })?;

            let local_handshake = transfer_engine.prepare_handshake().map_err(|err| {
                std::io::Error::other(format!("failed to prepare RDMA handshake: {err}"))
            })?;
            let rdma_session_id = local_handshake.to_bytes();
            let rdma_engine = Arc::new(transfer_engine);
            let engine_cell: Arc<OnceLock<Arc<PegaEngine>>> = Arc::new(OnceLock::new());

            let rdma_read_fn: RdmaBatchReadFn = {
                let rdma_engine = Arc::clone(&rdma_engine);
                Arc::new(move |session_id, local_ptrs, remote_ptrs, sizes| {
                    let peer = HandshakeMetadata::from_bytes(session_id)
                        .map_err(|e| format!("invalid RDMA handshake metadata: {e}"))?;
                    let descs: Vec<TransferDesc> = local_ptrs
                        .iter()
                        .zip(remote_ptrs.iter())
                        .zip(sizes.iter())
                        .map(|((&local, &remote), &len)| TransferDesc {
                            local_ptr: std::ptr::NonNull::new(local as *mut u8).unwrap(),
                            remote_ptr: std::ptr::NonNull::new(remote as *mut u8).unwrap(),
                            len,
                        })
                        .collect();
                    let rx = rdma_engine
                        .batch_transfer_async(TransferOp::Read, &peer, &descs)
                        .map_err(|e| e.to_string())?;
                    rx.recv()
                        .map_err(|e| format!("RDMA transfer channel error: {e}"))?
                        .map_err(|e| e.to_string())
                })
            };
            let rdma_register_fn: RdmaRegisterMemoryFn = {
                let rdma_engine = Arc::clone(&rdma_engine);
                Arc::new(move |ptr, size| {
                    let region = MemoryRegion {
                        ptr: std::ptr::NonNull::new(ptr as *mut u8).unwrap(),
                        len: size,
                    };
                    rdma_engine
                        .register_memory(&[region])
                        .map_err(|e| e.to_string())
                })
            };
            let remote_fetch_fn: RemoteFetchFn = {
                let metaserver_client = Arc::clone(&metaserver_client);
                let client_pool = Arc::clone(&client_pool);
                let engine_cell = Arc::clone(&engine_cell);
                let rdma_read_fn = Arc::clone(&rdma_read_fn);
                let rdma_register_fn = Arc::clone(&rdma_register_fn);
                let cached_config: Arc<OnceLock<Arc<RemoteFetchWorkerConfig>>> =
                    Arc::new(OnceLock::new());
                Arc::new(move |keys, done_tx| {
                    let Some(engine) = engine_cell.get().cloned() else {
                        let _ = done_tx.send(Vec::new());
                        return;
                    };
                    let config = Arc::clone(cached_config.get_or_init(|| {
                        Arc::new(RemoteFetchWorkerConfig::new_from_engine(
                            Arc::clone(&metaserver_client),
                            Arc::clone(&client_pool),
                            engine,
                            Arc::clone(&rdma_read_fn),
                            Arc::clone(&rdma_register_fn),
                        ))
                    }));
                    tokio::spawn(execute_remote_fetch(config, keys, done_tx));
                })
            };

            Some(RemoteFetchRuntime {
                remote_fetch_fn,
                engine_cell,
                rdma_engine,
                rdma_session_id,
            })
        } else {
            None
        };

        // Create PegaEngine inside tokio runtime context (needed for SSD cache tokio::spawn)
        let engine = Arc::new(PegaEngine::new_with_remote_fetch(
            cli.pool_size,
            cli.use_hugepages,
            storage_config,
            metaserver_registrar,
            remote_fetch_runtime
                .as_ref()
                .map(|runtime| Arc::clone(&runtime.remote_fetch_fn)),
        ));

        if let Some(runtime) = &remote_fetch_runtime {
            runtime
                .engine_cell
                .set(Arc::clone(&engine))
                .map_err(|_| std::io::Error::other("remote fetch engine already initialized"))?;

            let pinned_regions = engine.pinned_memory_regions();
            let memory_regions: Vec<MemoryRegion> = pinned_regions
                .into_iter()
                .map(|(ptr, len)| MemoryRegion {
                    ptr: std::ptr::NonNull::new(ptr as *mut u8).unwrap(),
                    len,
                })
                .collect();
            runtime
                .rdma_engine
                .register_memory(&memory_regions)
                .map_err(|err| {
                    std::io::Error::other(format!(
                        "failed to register pinned memory with RDMA engine: {err}"
                    ))
                })?;
        }

        let service = GrpcEngineService::new(
            Arc::clone(&engine),
            Arc::clone(&registry),
            Arc::clone(&shutdown),
            remote_fetch_runtime
                .as_ref()
                .map(|runtime| runtime.rdma_session_id.clone()),
        );

        // Spawn background GC task for stale inflight blocks
        {
            let engine = Arc::clone(&engine);
            let shutdown = Arc::clone(&shutdown);
            tokio::spawn(async move {
                const GC_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes
                const GC_MAX_AGE: Duration = Duration::from_secs(3600); // 60 minutes
                let mut interval = tokio::time::interval(GC_INTERVAL);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            let cleaned = engine.gc_stale_inflight(GC_MAX_AGE).await;
                            if cleaned > 0 {
                                info!("Inflight GC: cleaned {} stale blocks", cleaned);
                            }
                        }
                        _ = shutdown.notified() => {
                            info!("Inflight GC task shutting down");
                            break;
                        }
                    }
                }
            });
            info!("Inflight GC task started (interval=5m, max_age=60m)");
        }

        // Spawn transfer lock GC task if remote fetch is enabled
        if remote_fetch_enabled {
            let engine = Arc::clone(&engine);
            let shutdown = Arc::clone(&shutdown);
            let lock_timeout_secs = cli.transfer_lock_timeout_secs;
            tokio::spawn(async move {
                // GC interval is half the lock timeout (check frequently enough to expire promptly)
                let gc_interval = Duration::from_secs((lock_timeout_secs / 2).max(5));
                let mut interval = tokio::time::interval(gc_interval);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            let expired = engine.gc_expired_transfer_locks();
                            if expired > 0 {
                                info!("Transfer lock GC: expired {} sessions", expired);
                            }
                        }
                        _ = shutdown.notified() => {
                            info!("Transfer lock GC task shutting down");
                            break;
                        }
                    }
                }
            });
            info!(
                "Transfer lock GC task started (interval={}s, timeout={}s)",
                (lock_timeout_secs / 2).max(5),
                lock_timeout_secs
            );
        }

        // Start HTTP server for health check (always enabled)
        let http_server_handle = http_server::start_http_server(
            cli.http_addr,
            Arc::clone(&engine),
            Arc::clone(&registry),
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
