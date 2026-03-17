use std::sync::Arc;

use cudarc::driver::CudaContext;
use pegaflow_core::*;

use super::gpu_buffer::GpuBuffer;
use super::helpers::*;

#[derive(Clone)]
pub struct HarnessConfig<'a> {
    pub instance_id: &'a str,
    pub namespace: &'a str,
    pub device_id: i32,
    pub tp_rank: usize,
    pub tp_size: usize,
    pub world_size: usize,
    pub num_layers: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub hash_salt: u8,
    pub storage_config: Option<StorageConfig>,
}

impl<'a> HarnessConfig<'a> {
    pub fn new(
        instance_id: &'a str,
        namespace: &'a str,
        num_blocks: usize,
        block_size: usize,
    ) -> Self {
        Self {
            instance_id,
            namespace,
            device_id: 0,
            tp_rank: 0,
            tp_size: 1,
            world_size: 1,
            num_layers: 1,
            num_blocks,
            block_size,
            hash_salt: 0,
            storage_config: None,
        }
    }

    pub fn with_world_size(mut self, world_size: usize) -> Self {
        self.world_size = world_size;
        self
    }

    pub fn with_hash_salt(mut self, hash_salt: u8) -> Self {
        self.hash_salt = hash_salt;
        self
    }

    pub fn with_storage_config(mut self, config: StorageConfig) -> Self {
        self.storage_config = Some(config);
        self
    }
}

pub struct RoundtripHarness {
    pub _cuda_ctx: Arc<CudaContext>,
    pub engine: PegaEngine,
    pub gpu_buf: GpuBuffer,
    pub host_data: Vec<u8>,
    pub block_ids: Vec<i32>,
    pub block_hashes: Vec<Vec<u8>>,
    pub instance_id: String,
    pub layer_name: String,
    pub device_id: i32,
    pub tp_rank: usize,
}

impl RoundtripHarness {
    pub fn new(config: HarnessConfig<'_>) -> Self {
        let cuda_device =
            usize::try_from(config.device_id).expect("device_id must be non-negative");
        let cuda_ctx =
            CudaContext::new(cuda_device).expect("CUDA init failed — is a GPU available?");

        let total_size = config.num_blocks * config.block_size;
        let gpu_buf = GpuBuffer::alloc(total_size);

        let mut host_data = vec![0u8; total_size];
        fill_test_pattern(&mut host_data, config.block_size);
        gpu_buf.copy_from_host(&host_data);

        let engine = match config.storage_config {
            Some(sc) => test_engine_with_storage_config(sc),
            None => test_engine(),
        };
        register_single_layer(
            &engine,
            config.instance_id,
            config.namespace,
            DEFAULT_LAYER,
            gpu_buf.as_u64(),
            total_size,
            config.num_blocks,
            config.block_size,
            config.device_id,
            config.tp_rank,
            config.tp_size,
            config.world_size,
            config.num_layers,
        );

        Self {
            _cuda_ctx: cuda_ctx,
            engine,
            gpu_buf,
            host_data,
            block_ids: make_block_ids(config.num_blocks),
            block_hashes: make_block_hashes(config.num_blocks, config.hash_salt),
            instance_id: config.instance_id.to_string(),
            layer_name: DEFAULT_LAYER.to_string(),
            device_id: config.device_id,
            tp_rank: config.tp_rank,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.block_hashes.len()
    }

    pub fn block_ids(&self) -> &[i32] {
        &self.block_ids
    }

    pub fn block_hashes(&self) -> &[Vec<u8>] {
        &self.block_hashes
    }

    pub async fn save_all(&self) {
        self.save_layer(self.block_ids.clone(), self.block_hashes.clone())
            .await
            .expect("save all blocks");
    }

    pub async fn save_layer(
        &self,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), EngineError> {
        save_single_layer(
            &self.engine,
            &self.instance_id,
            self.tp_rank,
            self.device_id,
            &self.layer_name,
            block_ids,
            block_hashes,
        )
        .await
    }

    pub async fn assert_cache_eventually_all(&self) {
        wait_for_cache(
            &self.engine,
            &self.instance_id,
            self.block_hashes(),
            self.num_blocks(),
            CACHE_WAIT_TIMEOUT,
        )
        .await;
    }

    pub fn query_hits(&self, block_hashes: &[Vec<u8>]) -> (usize, usize) {
        self.engine
            .count_prefix_hit_blocks(&self.instance_id, block_hashes)
            .expect("count_prefix_hit_blocks")
    }

    pub async fn expect_query_prefetch_done_all(&self) {
        match self
            .engine
            .count_prefix_hit_blocks_with_prefetch(
                &self.instance_id,
                "harness-req",
                self.block_hashes(),
            )
            .await
            .expect("query_prefetch")
        {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, self.num_blocks());
                assert_eq!(missing, 0);
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    pub async fn load_all_and_wait(&self) -> Result<(), EngineError> {
        self.load_and_wait(self.block_ids(), self.block_hashes())
            .await
    }

    pub async fn load_and_wait(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::new().expect("create LoadState");
        let shm_name = load_state.shm_name().to_string();
        let layers = [self.layer_name.as_str()];

        self.engine.batch_load_kv_blocks_multi_layer(
            &self.instance_id,
            self.tp_rank,
            self.device_id,
            &shm_name,
            &layers,
            block_ids,
            block_hashes,
        )?;

        wait_for_load(&load_state, LOAD_WAIT_TIMEOUT).await;
        Ok(())
    }

    pub fn expect_load_submit_error(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
        expected_msg: &str,
    ) {
        let load_state = LoadState::new().expect("create LoadState");
        let shm_name = load_state.shm_name().to_string();
        let layers = [self.layer_name.as_str()];

        let err = self
            .engine
            .batch_load_kv_blocks_multi_layer(
                &self.instance_id,
                self.tp_rank,
                self.device_id,
                &shm_name,
                &layers,
                block_ids,
                block_hashes,
            )
            .expect_err("load submission should fail");

        assert!(
            err.to_string().contains(expected_msg),
            "unexpected error: {}",
            err
        );
        assert!(
            load_state.get() < 0,
            "LoadState should be ERROR after pre-submit failure"
        );
    }

    pub fn unpin(&self, block_hashes: &[Vec<u8>]) -> usize {
        self.engine
            .unpin_blocks(&self.instance_id, block_hashes)
            .expect("unpin_blocks")
    }

    pub fn zero_gpu_and_assert(&self) {
        self.gpu_buf.zero();
        assert!(
            self.gpu_buf.copy_to_host().iter().all(|&b| b == 0),
            "GPU memory should be zeroed"
        );
    }

    pub fn assert_gpu_matches_host(&self) {
        assert_eq!(
            self.gpu_buf.copy_to_host(),
            self.host_data,
            "GPU round-trip data mismatch"
        );
    }
}
