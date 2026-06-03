//! SSD multi-path throughput benchmark.
//!
//! Measures how write and read throughput scale when cache files are spread
//! across multiple physical SSD paths versus a single path.  Each path receives
//! the configured number of shards so that every device is utilised.

use std::ffi::c_void;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudarc::driver::{CudaContext, sys};
use pegaflow_core::{
    LayerSave, LoadState, PegaEngine, PrefetchStatus, SsdCacheConfig, StorageConfig,
};
use tokio::runtime::Runtime;

const INSTANCE_ID: &str = "ssd-mp-bench";
const NAMESPACE: &str = "ssd-mp";
const LAYER_NAME: &str = "layer_0";
const DEVICE_ID: i32 = 0;

const NUM_BLOCKS: usize = 512;
const BYTES_PER_BLOCK: usize = 4096;
const SSD_CAPACITY: u64 = 256 * 1024 * 1024; // 256 MiB

// ---------------------------------------------------------------------------
// CUDA helpers (mirrored from cpu_path.rs)
// ---------------------------------------------------------------------------

struct GpuBuffer {
    ctx: Arc<CudaContext>,
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn new(ctx: Arc<CudaContext>, len: usize) -> Self {
        assert!(len > 0);
        ctx.bind_to_thread().expect("bind CUDA context");
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ctx, ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        self.ctx.bind_to_thread().expect("bind CUDA context");
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.ctx.bind_to_thread();
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    assert_eq!(host_data.len() % block_size, 0);
    for (idx, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        block.fill(((idx % 251) + 1) as u8);
    }
}

fn make_block_hashes(num_blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

fn block_ids(num_blocks: usize) -> Vec<i32> {
    (0..num_blocks).map(|i| i as i32).collect()
}

// ---------------------------------------------------------------------------
// Benchmark fixture
// ---------------------------------------------------------------------------

struct BenchFixture {
    engine: PegaEngine,
    _ctx: Arc<CudaContext>,
    _gpu: GpuBuffer,
    num_blocks: usize,
}

impl BenchFixture {
    fn new(num_blocks: usize, bytes_per_block: usize, ssd_config: Option<SsdCacheConfig>) -> Self {
        let ctx = CudaContext::new(DEVICE_ID as usize).expect("CUDA context");
        ctx.bind_to_thread().expect("bind CUDA context");

        let total_bytes = num_blocks
            .checked_mul(bytes_per_block)
            .expect("registered GPU size overflow");
        let gpu = GpuBuffer::new(Arc::clone(&ctx), total_bytes);
        let mut host = vec![0u8; total_bytes];
        fill_test_pattern(&mut host, bytes_per_block);
        gpu.copy_from_host(&host);

        let pool_size = total_bytes
            .checked_mul(4)
            .and_then(|size| size.checked_add(64 << 20))
            .expect("pool size overflow");

        let engine = PegaEngine::new_with_config(
            pool_size,
            false,
            StorageConfig {
                ssd_cache_config: ssd_config,
                ..StorageConfig::default()
            },
        )
        .expect("engine");

        engine
            .register_context_layer_batch(
                INSTANCE_ID,
                NAMESPACE,
                DEVICE_ID,
                0,
                0,
                1,
                1,
                1,
                &[LAYER_NAME.to_string()],
                &[gpu.as_u64()],
                &[total_bytes],
                &[num_blocks],
                &[bytes_per_block],
                &[0],
                &[1],
            )
            .expect("register layer");

        Self {
            engine,
            _ctx: ctx,
            _gpu: gpu,
            num_blocks,
        }
    }

    async fn save_and_flush(&self, hashes: Vec<Vec<u8>>) {
        self.engine
            .batch_save_kv_blocks_from_ipc(
                INSTANCE_ID,
                0,
                0,
                DEVICE_ID,
                vec![LayerSave {
                    layer_name: LAYER_NAME.to_string(),
                    block_ids: block_ids(self.num_blocks),
                    block_hashes: hashes,
                }],
            )
            .await
            .expect("save");
        self.engine.flush_saves().await;
    }

    async fn query_and_load(&self, hashes: &[Vec<u8>]) {
        let lease = match self
            .engine
            .count_prefix_hit_blocks_with_prefetch(INSTANCE_ID, "req-1", hashes)
            .await
            .expect("query")
        {
            PrefetchStatus::Ready { blocks, missing } => {
                assert_eq!(blocks.len(), hashes.len());
                assert_eq!(missing, 0);
                self.engine
                    .create_query_lease(INSTANCE_ID, blocks)
                    .expect("create query lease")
            }
            PrefetchStatus::Loading => panic!("bench should not return Loading"),
        };

        let load_state = LoadState::new().expect("create LoadState");
        self.engine
            .batch_load_kv_blocks_multi_layer(
                INSTANCE_ID,
                0,
                DEVICE_ID,
                load_state.shm_name(),
                &[LAYER_NAME],
                &[(lease, block_ids(self.num_blocks))],
            )
            .expect("submit load");

        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let state = load_state.get();
            if state == pegaflow_core::sync_state::LOAD_STATE_SUCCESS {
                break;
            }
            assert!(state != pegaflow_core::sync_state::LOAD_STATE_ERROR, "load error");
            assert!(Instant::now() < deadline, "load timeout");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Config {
    name: &'static str,
    paths: &'static [&'static str],
    shards: usize,
}

const CONFIGS: &[Config] = &[
    Config {
        name: "1path_1shard",
        paths: &["/tmp/pegaflow-ssd-bench-0"],
        shards: 1,
    },
    Config {
        name: "1path_2shard",
        paths: &["/tmp/pegaflow-ssd-bench-0"],
        shards: 2,
    },
    Config {
        name: "2path_1shard",
        paths: &["/tmp/pegaflow-ssd-bench-0", "/tmp/pegaflow-ssd-bench-1"],
        shards: 1,
    },
    Config {
        name: "2path_2shard",
        paths: &["/tmp/pegaflow-ssd-bench-0", "/tmp/pegaflow-ssd-bench-1"],
        shards: 2,
    },
];

fn make_ssd_config(cfg: &Config) -> SsdCacheConfig {
    SsdCacheConfig {
        cache_paths: cfg.paths.iter().map(|p| (*p).into()).collect(),
        capacity_bytes: SSD_CAPACITY,
        shards: NonZeroUsize::new(cfg.shards).unwrap(),
        write_queue_depth: 8,
        prefetch_queue_depth: 2,
        write_inflight: 2,
        prefetch_inflight: 16,
    }
}

fn io_uring_available() -> bool {
    unsafe {
        let mut params = std::mem::MaybeUninit::<[u8; 128]>::zeroed();
        let fd = libc::syscall(
            libc::SYS_io_uring_setup,
            1i32,
            params.as_mut_ptr() as *mut libc::c_void,
        );
        if fd >= 0 {
            libc::close(fd as i32);
            true
        } else {
            false
        }
    }
}

fn write_benchmarks(c: &mut Criterion) {
    if !io_uring_available() {
        eprintln!("Skipping SSD multi-path benchmarks: io_uring is not available in this environment");
        return;
    }
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("ssd_multi_path/write");
    let total_bytes = (NUM_BLOCKS * BYTES_PER_BLOCK) as u64;
    group.throughput(Throughput::Bytes(total_bytes));
    group.sample_size(10);

    for cfg in CONFIGS {
        group.bench_function(BenchmarkId::new("save_flush", cfg.name), |b| {
            // Clean up previous bench files so each iteration starts fresh.
            for path in cfg.paths {
                let _ = std::fs::remove_dir_all(path);
            }

            let fixture = BenchFixture::new(
                NUM_BLOCKS,
                BYTES_PER_BLOCK,
                Some(make_ssd_config(cfg)),
            );

            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for iter in 0..iters {
                        let hashes = make_block_hashes(NUM_BLOCKS, (iter as u8).wrapping_add(1));
                        let start = Instant::now();
                        fixture.save_and_flush(hashes).await;
                        measured += start.elapsed();
                        // Evict from memory so the next iteration forces SSD writes.
                        fixture.engine.cleanup_memory_cache();
                    }
                    measured
                })
            });

            for path in cfg.paths {
                let _ = std::fs::remove_dir_all(path);
            }
        });
    }
    group.finish();
}

fn read_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("ssd_multi_path/read");
    let total_bytes = (NUM_BLOCKS * BYTES_PER_BLOCK) as u64;
    group.throughput(Throughput::Bytes(total_bytes));
    group.sample_size(10);

    for cfg in CONFIGS {
        group.bench_function(BenchmarkId::new("prefetch_load", cfg.name), |b| {
            for path in cfg.paths {
                let _ = std::fs::remove_dir_all(path);
            }

            let fixture = BenchFixture::new(
                NUM_BLOCKS,
                BYTES_PER_BLOCK,
                Some(make_ssd_config(cfg)),
            );

            // Prime SSD cache.
            let prime_hashes = make_block_hashes(NUM_BLOCKS, 0);
            rt.block_on(fixture.save_and_flush(prime_hashes.clone()));
            // Evict from RAM so reads come from SSD.
            fixture.engine.cleanup_memory_cache();

            b.iter_custom(|iters| {
                rt.block_on(async {
                    let mut measured = Duration::ZERO;
                    for _ in 0..iters {
                        // Re-query the same hashes — they are already on SSD.
                        let start = Instant::now();
                        fixture.query_and_load(&prime_hashes).await;
                        measured += start.elapsed();
                        // Zero GPU memory so the next load is not a no-op.
                        fixture
                            .engine
                            .batch_save_kv_blocks_from_ipc(
                                INSTANCE_ID,
                                0,
                                0,
                                DEVICE_ID,
                                vec![LayerSave {
                                    layer_name: LAYER_NAME.to_string(),
                                    block_ids: block_ids(fixture.num_blocks),
                                    block_hashes: make_block_hashes(fixture.num_blocks, 255),
                                }],
                            )
                            .await
                            .expect("zero-save");
                        fixture.engine.flush_saves().await;
                        fixture.engine.cleanup_memory_cache();
                    }
                    measured
                })
            });

            for path in cfg.paths {
                let _ = std::fs::remove_dir_all(path);
            }
        });
    }
    group.finish();
}

criterion_group!(benches, write_benchmarks, read_benchmarks);
criterion_main!(benches);
