use std::sync::Arc;

use cudarc::driver::CudaContext;
use pegaflow_core::*;

use super::gpu_buffer::GpuBuffer;
use super::helpers::*;

// ---------------------------------------------------------------------------
// TestGpuData: GPU buffer + expected content
// ---------------------------------------------------------------------------

pub struct TestGpuData {
    gpu: GpuBuffer,
    expected: Vec<u8>,
    pub block_size: usize,
}

impl TestGpuData {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let total = num_blocks * block_size;
        let gpu = GpuBuffer::alloc(total);
        let mut expected = vec![0u8; total];
        fill_test_pattern(&mut expected, block_size);
        gpu.copy_from_host(&expected);
        Self {
            gpu,
            expected,
            block_size,
        }
    }

    pub fn overwrite(&mut self, modifier: u8) {
        for b in &mut self.expected {
            *b = b.wrapping_add(modifier);
        }
        self.gpu.copy_from_host(&self.expected);
    }

    pub fn zero_gpu(&self) {
        self.gpu.zero();
    }

    pub fn assert_gpu_matches_expected(&self) {
        assert_eq!(self.gpu.copy_to_host(), self.expected, "GPU data mismatch");
    }

    pub fn ptr(&self) -> u64 {
        self.gpu.as_u64()
    }

    pub fn total_size(&self) -> usize {
        self.expected.len()
    }
}

// ---------------------------------------------------------------------------
// TestEnv: engine + registered layers
// ---------------------------------------------------------------------------

pub struct RegisteredLayer {
    pub name: String,
    pub data: TestGpuData,
    pub num_blocks: usize,
}

pub struct TestEnv {
    pub _ctx: Arc<CudaContext>,
    pub engine: PegaEngine,
    pub instance_id: String,
    pub layers: Vec<RegisteredLayer>,
    pub world_size: usize,
}

pub struct TestEnvBuilder {
    instance_id: &'static str,
    namespace: &'static str,
    pool_size: usize,
    world_size: usize,
    storage_config: Option<StorageConfig>,
    layers: Vec<(&'static str, usize, usize)>,
}

impl TestEnvBuilder {
    pub fn new(instance_id: &'static str, namespace: &'static str) -> Self {
        Self {
            instance_id,
            namespace,
            pool_size: 16 << 20,
            world_size: 1,
            storage_config: None,
            layers: vec![],
        }
    }

    pub fn layer(mut self, name: &'static str, num_blocks: usize, block_size: usize) -> Self {
        self.layers.push((name, num_blocks, block_size));
        self
    }

    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    pub fn world_size(mut self, ws: usize) -> Self {
        self.world_size = ws;
        self
    }

    pub fn storage(mut self, config: StorageConfig) -> Self {
        self.storage_config = Some(config);
        self
    }

    pub fn build(self) -> TestEnv {
        let ctx = CudaContext::new(0).expect("CUDA init");
        let sc = self.storage_config.unwrap_or(StorageConfig {
            enable_lfu_admission: false,
            ..StorageConfig::default()
        });
        let engine = test_engine_with_pool(self.pool_size, sc);

        let layers: Vec<RegisteredLayer> = self
            .layers
            .iter()
            .map(|&(name, nb, bs)| RegisteredLayer {
                name: name.to_string(),
                data: TestGpuData::new(nb, bs),
                num_blocks: nb,
            })
            .collect();

        let layer_infos: Vec<LayerInfo> = layers
            .iter()
            .map(|l| LayerInfo {
                name: l.name.clone(),
                gpu_ptr: l.data.ptr(),
                total_size: l.data.total_size(),
                num_blocks: l.num_blocks,
                block_size: l.data.block_size,
            })
            .collect();

        register_layers(
            &engine,
            self.instance_id,
            self.namespace,
            &layer_infos,
            0,
            0,
            1,
            self.world_size,
            layers.len(),
        );

        TestEnv {
            _ctx: ctx,
            engine,
            instance_id: self.instance_id.to_string(),
            layers,
            world_size: self.world_size,
        }
    }
}

// ---------------------------------------------------------------------------
// TestEnv operations — mirror pub API, names spell out side effects
// ---------------------------------------------------------------------------

impl TestEnv {
    /// Shortcut for single-layer envs.
    pub fn data(&self) -> &TestGpuData {
        assert_eq!(self.layers.len(), 1, "data() requires exactly one layer");
        &self.layers[0].data
    }

    pub fn data_mut(&mut self) -> &mut TestGpuData {
        assert_eq!(
            self.layers.len(),
            1,
            "data_mut() requires exactly one layer"
        );
        &mut self.layers[0].data
    }

    pub fn num_blocks(&self) -> usize {
        assert_eq!(
            self.layers.len(),
            1,
            "num_blocks() requires exactly one layer"
        );
        self.layers[0].num_blocks
    }

    /// Generate hashes using this env's block count + a salt.
    pub fn hashes(&self, salt: u8) -> Vec<Vec<u8>> {
        make_block_hashes(self.num_blocks(), salt)
    }

    // -- Core API wrappers --

    /// Save one layer's blocks to cache.
    pub async fn save_layer(&self, layer_index: usize, hashes: &[Vec<u8>]) {
        let layer = &self.layers[layer_index];
        let block_ids: Vec<i32> = (0..hashes.len() as i32).collect();
        self.engine
            .batch_save_kv_blocks_from_ipc(
                &self.instance_id,
                0,
                0,
                vec![LayerSave {
                    layer_name: layer.name.clone(),
                    block_ids,
                    block_hashes: hashes.to_vec(),
                }],
            )
            .await
            .expect("save");
    }

    /// Save all layers, wait until all blocks appear in cache.
    pub async fn save_and_wait(&self, hashes: &[Vec<u8>]) {
        for i in 0..self.layers.len() {
            self.save_layer(i, hashes).await;
        }
        wait_for_cache(
            &self.engine,
            &self.instance_id,
            hashes,
            hashes.len(),
            self.world_size,
            CACHE_WAIT_TIMEOUT,
        )
        .await;
    }

    /// Query prefix hits. Returns raw PrefetchStatus. Leaves blocks pinned on hit.
    pub async fn query(&self, hashes: &[Vec<u8>]) -> PrefetchStatus {
        self.engine
            .count_prefix_hit_blocks_with_prefetch(&self.instance_id, "test", hashes)
            .await
            .expect("query")
    }

    /// Query, assert all hit, leave pinned (scheduler step before load).
    pub async fn assert_all_hit_and_pin(&self, hashes: &[Vec<u8>]) {
        match self.query(hashes).await {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, hashes.len(), "expected all blocks hit");
                assert_eq!(missing, 0);
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    pub fn unpin(&self, hashes: &[Vec<u8>]) {
        for _ in 0..self.world_size.max(1) {
            self.engine
                .unpin_blocks(&self.instance_id, hashes)
                .expect("unpin");
        }
    }

    /// Count cache hits, then unpin (for probing without consuming).
    pub async fn count_hits_then_unpin(&self, hashes: &[Vec<u8>]) -> usize {
        let hit = match self.query(hashes).await {
            PrefetchStatus::Done { hit, .. } => hit,
            PrefetchStatus::Loading { hit, .. } => hit,
        };
        if hit > 0 {
            self.unpin(&hashes[..hit]);
        }
        hit
    }

    /// Load blocks from cache to GPU (pin must already be held).
    pub async fn load_to_gpu(&self, hashes: &[Vec<u8>]) {
        let block_ids: Vec<i32> = (0..hashes.len() as i32).collect();
        let layer_names: Vec<&str> = self.layers.iter().map(|l| l.name.as_str()).collect();
        let load_state = LoadState::new().expect("create LoadState");
        let shm_name = load_state.shm_name().to_string();
        self.engine
            .batch_load_kv_blocks_multi_layer(
                &self.instance_id,
                0,
                0,
                &shm_name,
                &layer_names,
                &block_ids,
                hashes,
            )
            .expect("submit load");
        wait_for_load(&load_state, LOAD_WAIT_TIMEOUT).await;
    }

    /// Submit load and assert it fails synchronously with `expected_msg`.
    pub fn expect_load_error(&self, hashes: &[Vec<u8>], expected_msg: &str) {
        let block_ids: Vec<i32> = (0..hashes.len() as i32).collect();
        let layer_names: Vec<&str> = self.layers.iter().map(|l| l.name.as_str()).collect();
        let load_state = LoadState::new().expect("create LoadState");
        let shm_name = load_state.shm_name().to_string();
        let err = self
            .engine
            .batch_load_kv_blocks_multi_layer(
                &self.instance_id,
                0,
                0,
                &shm_name,
                &layer_names,
                &block_ids,
                hashes,
            )
            .expect_err("load should fail");
        assert!(
            err.to_string().contains(expected_msg),
            "unexpected error: {err}"
        );
        assert!(
            load_state.get() < 0,
            "LoadState should be ERROR after pre-submit failure"
        );
    }

    /// Wait until hashes appear in cache (without saving).
    pub async fn wait_cached(&self, hashes: &[Vec<u8>]) {
        wait_for_cache(
            &self.engine,
            &self.instance_id,
            hashes,
            hashes.len(),
            self.world_size,
            CACHE_WAIT_TIMEOUT,
        )
        .await;
    }
}
