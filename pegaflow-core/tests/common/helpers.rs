use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use pegaflow_core::sync_state::{LOAD_STATE_ERROR, LOAD_STATE_SUCCESS};
use pegaflow_core::*;

pub const CACHE_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
pub const LOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
pub const DEFAULT_LAYER: &str = "layer_0";

pub fn test_engine() -> PegaEngine {
    test_engine_with_storage_config(StorageConfig {
        enable_lfu_admission: false,
        hint_value_size_bytes: None,
        max_prefetch_blocks: 100,
        ssd_cache_config: None,
        rdma_nic_names: None,
        enable_numa_affinity: false,
        blockwise_alloc: false,
        transfer_lock_timeout: Duration::from_secs(120),
        metaserver_addr: None,
        advertise_addr: None,
        metaserver_queue_depth: 256,
        pool_shards: 1,
    })
}

pub fn test_engine_with_storage_config(config: StorageConfig) -> PegaEngine {
    // 16 MB pool — enough for test blocks, small enough to be fast.
    PegaEngine::new_with_config(16 << 20, false, config)
}

/// Layer registration info for batch registration.
pub struct LayerInfo {
    pub name: String,
    pub gpu_ptr: u64,
    pub total_size: usize,
    pub num_blocks: usize,
    pub block_size: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn register_layers(
    engine: &PegaEngine,
    instance_id: &str,
    namespace: &str,
    layers: &[LayerInfo],
    device_id: i32,
    tp_rank: usize,
    tp_size: usize,
    world_size: usize,
    num_layers: usize,
) {
    let layer_names: Vec<String> = layers.iter().map(|l| l.name.clone()).collect();
    let gpu_ptrs: Vec<u64> = layers.iter().map(|l| l.gpu_ptr).collect();
    let total_sizes: Vec<usize> = layers.iter().map(|l| l.total_size).collect();
    let num_blocks_list: Vec<usize> = layers.iter().map(|l| l.num_blocks).collect();
    let block_sizes: Vec<usize> = layers.iter().map(|l| l.block_size).collect();
    let kv_strides: Vec<usize> = vec![0; layers.len()];
    let segments: Vec<usize> = vec![1; layers.len()];

    engine
        .register_context_layer_batch(
            instance_id,
            namespace,
            device_id,
            tp_rank,
            tp_size,
            world_size,
            num_layers,
            &layer_names,
            &gpu_ptrs,
            &total_sizes,
            &num_blocks_list,
            &block_sizes,
            &kv_strides,
            &segments,
        )
        .expect("register_context_layer_batch");
}

#[allow(clippy::too_many_arguments)]
pub fn register_single_layer(
    engine: &PegaEngine,
    instance_id: &str,
    namespace: &str,
    layer_name: &str,
    gpu_ptr: u64,
    total_size: usize,
    num_blocks: usize,
    block_size: usize,
    device_id: i32,
    tp_rank: usize,
    tp_size: usize,
    world_size: usize,
    num_layers: usize,
) {
    register_layers(
        engine,
        instance_id,
        namespace,
        &[LayerInfo {
            name: layer_name.to_string(),
            gpu_ptr,
            total_size,
            num_blocks,
            block_size,
        }],
        device_id,
        tp_rank,
        tp_size,
        world_size,
        num_layers,
    );
}

pub async fn save_single_layer(
    engine: &PegaEngine,
    instance_id: &str,
    tp_rank: usize,
    device_id: i32,
    layer_name: &str,
    block_ids: Vec<i32>,
    block_hashes: Vec<Vec<u8>>,
) -> Result<(), EngineError> {
    engine
        .batch_save_kv_blocks_from_ipc(
            instance_id,
            tp_rank,
            device_id,
            vec![LayerSave {
                layer_name: layer_name.to_string(),
                block_ids,
                block_hashes,
            }],
        )
        .await
}

/// Fill `host_data` with a deterministic pattern: block i = byte (i+1) repeated.
pub fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    assert_eq!(
        host_data.len() % block_size,
        0,
        "host_data must contain full blocks"
    );
    for (i, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        let fill = ((i % 251) + 1) as u8;
        block.fill(fill);
    }
}

pub fn make_block_ids(num_blocks: usize) -> Vec<i32> {
    (0..num_blocks)
        .map(|idx| i32::try_from(idx).expect("num_blocks exceeds i32 range"))
        .collect()
}

pub fn make_block_hashes(num_blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

/// Poll `count_prefix_hit_blocks_with_prefetch` until `expected_hit` blocks are cached, or timeout.
///
/// This helper is intentionally state-neutral for the caller: every probe query
/// drains the pin reservations it just created so tests can wait for cache
/// visibility without also performing the scheduler pin step.
pub async fn wait_for_cache(
    engine: &PegaEngine,
    instance_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit: usize,
    world_size: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let status = engine
            .count_prefix_hit_blocks_with_prefetch(instance_id, "wait-for-cache", block_hashes)
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        let hit = match status {
            PrefetchStatus::Done { hit, .. } => hit,
            PrefetchStatus::Loading { hit, .. } => hit,
        };
        if hit > 0 {
            let hit_hashes = &block_hashes[..hit];
            for _ in 0..world_size.max(1) {
                engine
                    .unpin_blocks(instance_id, hit_hashes)
                    .expect("unpin_blocks after cache probe");
            }
        }
        if hit >= expected_hit {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected_hit} cached blocks (got {hit})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Poll `LoadState::get()` until success or timeout.
pub async fn wait_for_load(load_state: &LoadState, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        let state = load_state.get();
        if state == LOAD_STATE_SUCCESS {
            return;
        }
        assert!(state != LOAD_STATE_ERROR, "load reported ERROR");
        assert!(Instant::now() < deadline, "timed out waiting for load");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

pub async fn wait_for_ssd_nonzero(cache_path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(mut f) = File::open(cache_path) {
            let mut buf = vec![0u8; 4096];
            if let Ok(n) = f.read(&mut buf)
                && n > 0
                && buf[..n].iter().any(|&b| b != 0)
            {
                return;
            }
        }

        assert!(
            Instant::now() < deadline,
            "timed out waiting for non-zero SSD data at {}",
            cache_path.display()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
