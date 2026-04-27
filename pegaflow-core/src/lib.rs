//! PegaFlow Core Engine
//!
//! A GPU-aware KV cache offloading engine with support for:
//! - Multi-tenant instance isolation
//! - Tensor parallelism (TP) across multiple GPUs
//! - Split-storage layout for efficient K/V batch transfers
//! - SSD caching tier
//! - Kubernetes service discovery for inter-node communication

#[macro_use]
mod trace;

mod allocator;
mod backing;
mod block;
mod cache;
mod gpu_worker;
mod instance;
mod internode;
pub use pegaflow_common::logging;
mod load_plan;
mod metrics;
mod offload;
mod pd_receive;
mod pinned_mem;
mod pinned_pool;
mod seal_offload;
mod storage;
pub mod sync_state;
pub mod transfer;

pub use backing::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdCacheConfig,
};
pub use block::{
    BlockHash, BlockKey, BlockStatus, LayerBlock, LayerSave, PrefetchStatus, RawBlock, SealedBlock,
};
pub use instance::{GpuContext, InstanceContext, KVCacheRegistration};
pub use internode::{DEFAULT_METASERVER_QUEUE_DEPTH, MetaServerClient, MetaServerClientConfig};
pub use load_plan::{PrepareLoadOutcome, PrepareLoadRequest, PreparedLoadItem};
pub use pd_receive::{
    PdReceiveDescriptor, PdReceiveDescriptorLookup, PdReceiveLayerLayout, PdReceivePrepareRequest,
    PdReceivePrepareResponse, PdReceiveRankDesc, PdReceiveSlabDesc,
};
pub use pegaflow_common::NumaNode;
use pegaflow_common::NumaTopology;
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use storage::StorageConfig;
pub use sync_state::{LoadState, LoadStateError};
pub use sync_state::{
    PREPARE_LOAD_STATE_NO_PLAN, PREPARE_LOAD_STATE_PENDING, PREPARE_LOAD_STATE_PLAN,
    PrepareLoadState, PrepareLoadStateError,
};
pub use trace::{RequestTrace, set_trace_sample_rate, should_sample};

use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use log::{debug, info, warn};
use parking_lot::Mutex;

use crate::backing::SSD_ALIGNMENT;
use crate::gpu_worker::{LayerLoadData, LoadBlock, LoadTask};
use crate::load_plan::PreparedLoadPlan;
use crate::metrics::core_metrics;
use crate::pd_receive::PdReceiveCheckoutStatus;
use crate::storage::StorageEngine;

/// Errors that can occur during engine operations.
#[derive(Debug)]
pub enum EngineError {
    /// Instance not found in the registry.
    InstanceMissing(String),
    /// GPU worker not found for the specified device.
    WorkerMissing(String, i32),
    /// Invalid argument provided.
    InvalidArgument(String),
    /// CUDA initialization or runtime error.
    CudaInit(String),
    /// Storage engine error.
    Storage(String),
    /// Internal lock poisoned.
    Poisoned(&'static str),
    /// Topology mismatch between registration and existing instance.
    TopologyMismatch(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InstanceMissing(ctx) => write!(f, "instance {ctx} not found"),
            EngineError::WorkerMissing(ctx, device) => {
                write!(f, "device {device} not found in instance {ctx}")
            }
            EngineError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            EngineError::CudaInit(msg) => write!(f, "failed to initialize CUDA: {msg}"),
            EngineError::Storage(msg) => write!(f, "storage error: {msg}"),
            EngineError::Poisoned(what) => write!(f, "internal lock poisoned: {what}"),
            EngineError::TopologyMismatch(msg) => write!(f, "topology mismatch: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<LoadStateError> for EngineError {
    fn from(err: LoadStateError) -> Self {
        EngineError::Storage(format!("LoadState: {err}"))
    }
}

impl From<PrepareLoadStateError> for EngineError {
    fn from(err: PrepareLoadStateError) -> Self {
        EngineError::Storage(format!("PrepareLoadState: {err}"))
    }
}

struct PreparedLoadEntry {
    instance_id: String,
    request_id: String,
    plan: PreparedLoadPlan,
    remaining_loads: usize,
    trace: Option<RequestTrace>,
}

/// Main engine for managing KV cache offloading.
///
/// `PegaEngine` is the top-level orchestrator that:
/// - Manages multiple inference instances
/// - Coordinates GPU worker pools for async transfers
/// - Interfaces with the storage engine for block caching
/// - Tracks GPU-NUMA topology for optimal memory locality
///
/// The engine is thread-safe and can be shared across async tasks.
#[derive(Clone)]
pub struct PegaEngine {
    /// Active inference instances indexed by instance ID.
    instances: Arc<RwLock<HashMap<String, Arc<InstanceContext>>>>,
    /// Storage engine for pinned memory, block cache, and SSD tier.
    storage: Arc<StorageEngine>,
    /// GPU-NUMA topology for memory allocation decisions.
    topology: Arc<NumaTopology>,
    /// D-side P/D CPU-staging receive leases.
    pd_receive: Arc<pd_receive::PdReceiveManager>,
    /// Prepared load plans keyed by opaque plan id.
    prepared_loads: Arc<Mutex<HashMap<u64, PreparedLoadEntry>>>,
    /// Monotonic source for prepared-load plan handles. Zero is reserved for
    /// "no plan" in shared memory.
    next_prepared_load_id: Arc<AtomicU64>,
}

impl PegaEngine {
    /// Create an engine with full custom configuration.
    ///
    /// If `storage_config.enable_numa_affinity` is true and the system has multiple
    /// NUMA nodes, per-node pinned memory pools are created for optimal bandwidth.
    pub fn new_with_config(
        pool_size: usize,
        use_hugepages: bool,
        storage_config: storage::StorageConfig,
    ) -> Self {
        let topology = Arc::new(NumaTopology::detect());
        topology.log_summary();

        let config = storage_config;
        let numa_nodes: Vec<NumaNode> = if config.enable_numa_affinity && topology.is_multi_numa() {
            info!(
                "Auto-enabling NUMA-aware memory allocation for {} nodes",
                topology.num_nodes()
            );
            topology.numa_nodes().to_vec()
        } else {
            vec![]
        };

        let storage = StorageEngine::new_with_config(pool_size, use_hugepages, config, &numa_nodes);

        PegaEngine {
            instances: Arc::new(RwLock::new(HashMap::new())),
            storage,
            topology,
            pd_receive: Arc::new(pd_receive::PdReceiveManager::new()),
            prepared_loads: Arc::new(Mutex::new(HashMap::new())),
            next_prepared_load_id: Arc::new(AtomicU64::new(1)),
        }
    }

    fn new_prepared_load_id(&self) -> u64 {
        loop {
            let plan_id = self.next_prepared_load_id.fetch_add(1, Ordering::Relaxed);
            if plan_id != 0 {
                return plan_id;
            }
        }
    }

    /// Get or create an instance with the specified topology.
    ///
    /// If an instance with the same ID exists but different topology,
    /// returns a `TopologyMismatch` error.
    fn get_or_create_instance(
        &self,
        instance_id: &str,
        namespace: &str,
        num_layers: usize,
        tp_size: usize,
        world_size: usize,
    ) -> Result<Arc<InstanceContext>, EngineError> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");

        if let Some(instance) = instances.get(instance_id) {
            // Already exists, verify topology
            instance
                .verify_topology(num_layers, tp_size, world_size)
                .map_err(|e| {
                    EngineError::TopologyMismatch(format!("instance {instance_id} {e}"))
                })?;
            return Ok(Arc::clone(instance));
        }

        // Create new instance
        let instance = InstanceContext::new(
            instance_id.to_string(),
            namespace.to_string(),
            num_layers,
            tp_size,
            world_size,
        )
        .map_err(EngineError::InvalidArgument)?;

        let instance = Arc::new(instance);
        instances.insert(instance_id.to_string(), Arc::clone(&instance));
        Ok(instance)
    }

    /// Look up an instance by ID.
    fn get_instance(&self, instance_id: &str) -> Result<Arc<InstanceContext>, EngineError> {
        let instances = self.instances.read().expect("instances read lock poisoned");
        instances
            .get(instance_id)
            .cloned()
            .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))
    }

    /// Batch register multiple KV cache layers for a single GPU.
    ///
    /// This reduces gRPC round-trips from N to 1 and allows the engine to
    /// construct the GPU context with all layer registrations at once.
    #[allow(clippy::too_many_arguments)]
    pub fn register_context_layer_batch(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        tp_rank: usize,
        tp_size: usize,
        world_size: usize,
        num_layers: usize,
        layer_names: &[String],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
    ) -> Result<(), EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(
                "device_id must be >= 0".to_string(),
            ));
        }

        // Build all registrations
        let ssd_enabled = self.storage.is_ssd_enabled();
        let batch_size = layer_names.len();
        let mut kv_caches = HashMap::with_capacity(batch_size);

        for i in 0..batch_size {
            let layer_name = &layer_names[i];
            let mut registration = KVCacheRegistration::new(
                data_ptrs[i],
                size_bytes_list[i],
                num_blocks_list[i],
                bytes_per_block_list[i],
                kv_stride_bytes_list[i],
                segments_list[i],
            )
            .map_err(|e| EngineError::InvalidArgument(format!("layer {layer_name}: {e}")))?;

            if ssd_enabled {
                registration = registration.with_ssd_padding(SSD_ALIGNMENT);
                if registration.padded_bytes_per_block != registration.bytes_per_block {
                    info!(
                        "SSD alignment padding: layer={layer_name}, bytes_per_block={} -> padded={}",
                        registration.bytes_per_block, registration.padded_bytes_per_block
                    );
                }
            }

            kv_caches.insert(layer_name.clone(), registration);
        }

        // Get or create instance
        let instance =
            self.get_or_create_instance(instance_id, namespace, num_layers, tp_size, world_size)?;

        // Get NUMA affinity for this GPU
        let numa_node = self.topology.numa_for_gpu(device_id);

        // Validate NUMA topology if NUMA-aware allocation is enabled
        if self.storage.is_numa_enabled() && numa_node.is_unknown() {
            return Err(EngineError::InvalidArgument(format!(
                "NUMA-aware allocation is enabled, but GPU {} NUMA affinity is unknown. \
                 Please ensure nvidia-smi is available and GPU NUMA topology is detectable, \
                 or disable NUMA-aware allocation.",
                device_id
            )));
        }

        // Register GPU with all layers
        instance.register_new_gpu(device_id, tp_rank, numa_node, kv_caches)?;

        info!(
            "Registered context batch: instance={instance_id}, namespace={namespace}, \
             device={device_id}, num_layers={num_layers}, tp_rank={tp_rank}/{tp_size}"
        );
        Ok(())
    }

    /// Unregister an instance and release all associated resources.
    pub fn unregister_instance(&self, instance_id: &str) -> Result<(), EngineError> {
        let removed = self
            .instances
            .write()
            .expect("instances write lock poisoned")
            .remove(instance_id);

        if removed.is_none() {
            return Err(EngineError::InstanceMissing(instance_id.to_string()));
        }
        info!("Unregistered instance: {}", instance_id);
        Ok(())
    }

    /// Unregister all instances, returning the IDs that were removed.
    pub fn unregister_all_instances(&self) -> Vec<String> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");
        let ids: Vec<String> = instances.keys().cloned().collect();
        instances.clear();
        if !ids.is_empty() {
            info!("Unregistered all instances: {:?}", ids);
        }
        ids
    }

    /// List all registered instance IDs.
    pub fn list_instance_ids(&self) -> Vec<String> {
        self.instances
            .read()
            .expect("instances read lock poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Returns:
    /// - `Done { hit, missing: 0 }`: all blocks in memory cache
    /// - `Loading { hit, loading }`: some blocks being fetched from SSD
    /// - `Done { hit, missing }`: some blocks don't exist
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "prepare_load.count_prefix_hit")
    )]
    pub async fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<PrefetchStatus, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let num_workers = instance.world_size();
        self.count_prefix_hit_blocks_with_prefetch_for_workers(
            instance_id,
            req_id,
            block_hashes,
            num_workers,
        )
        .await
    }

    async fn count_prefix_hit_blocks_with_prefetch_for_workers(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> Result<PrefetchStatus, EngineError> {
        if req_id.is_empty() {
            warn!("count_prefix_hit_blocks_with_prefetch: empty req_id, returning 0 hits");
            return Ok(PrefetchStatus::Done {
                hit: 0,
                missing: block_hashes.len(),
            });
        }

        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let metrics = core_metrics();

        let status = self
            .storage
            .check_prefix_and_prefetch(instance_id, req_id, namespace, block_hashes, num_workers)
            .await;

        match &status {
            PrefetchStatus::Done { hit, missing } => {
                metrics.cache_block_hits.add(*hit as u64, &[]);
                if *missing > 0 {
                    metrics.cache_block_misses.add(*missing as u64, &[]);
                }
            }
            PrefetchStatus::Loading { hit, loading: _ } => {
                metrics.cache_block_hits.add(*hit as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Advance one server-owned prepare-load state-machine step.
    ///
    /// The caller may invoke this repeatedly from an async task until the
    /// outcome is terminal. Terminal plan outcomes are registered under an
    /// opaque plan id that the later load call consumes.
    pub async fn prepare_load_step(
        &self,
        request: &PrepareLoadRequest,
    ) -> Result<PrepareLoadOutcome, EngineError> {
        self.prepare_load_step_with_trace(request, None).await
    }

    /// Same prepare-load state-machine step, optionally attached to a
    /// request-lifecycle trace that will be carried into the later load.
    pub async fn prepare_load_step_with_trace(
        &self,
        request: &PrepareLoadRequest,
        trace: Option<RequestTrace>,
    ) -> Result<PrepareLoadOutcome, EngineError> {
        assert!(
            !self.has_prepared_load_for_request(&request.instance_id, &request.request_id),
            "prepared load plan already exists before prepare_load_step: instance_id={} request_id={}",
            request.instance_id,
            request.request_id
        );

        let computed_blocks = usize::try_from(
            request.num_computed_tokens / request.virtual_block_size,
        )
        .map_err(|_| {
            EngineError::InvalidArgument(format!(
                "num_computed_tokens={} / virtual_block_size={} does not fit usize",
                request.num_computed_tokens, request.virtual_block_size
            ))
        })?;

        if let Some(decode_req_id) = request
            .decode_request_id
            .as_ref()
            .filter(|request_id| !request_id.is_empty())
        {
            return self
                .prepare_staged_load_step(request, decode_req_id, computed_blocks, trace)
                .await;
        }

        self.prepare_cache_load_step(request, computed_blocks, trace)
            .await
    }

    async fn prepare_cache_load_step(
        &self,
        request: &PrepareLoadRequest,
        computed_blocks: usize,
        trace: Option<RequestTrace>,
    ) -> Result<PrepareLoadOutcome, EngineError> {
        let remaining_hashes = if computed_blocks >= request.block_hashes.len() {
            Vec::new()
        } else {
            request.block_hashes[computed_blocks..].to_vec()
        };
        if remaining_hashes.is_empty() {
            return Ok(PrepareLoadOutcome::NoPlan);
        }

        let status = self
            .count_prefix_hit_blocks_with_prefetch_for_workers(
                &request.instance_id,
                &request.request_id,
                &remaining_hashes,
                1,
            )
            .await?;

        let hit_blocks = match status {
            PrefetchStatus::Loading { .. } => return Ok(PrepareLoadOutcome::Pending),
            PrefetchStatus::Done { hit, .. } => hit.min(remaining_hashes.len()),
        };
        if hit_blocks == 0 {
            return Ok(PrepareLoadOutcome::NoPlan);
        }

        let num_tokens = (hit_blocks as u64)
            .checked_mul(request.virtual_block_size)
            .ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "prepared load token count overflows: blocks={} virtual_block_size={}",
                    hit_blocks, request.virtual_block_size
                ))
            })?;
        let block_hashes: Vec<Vec<u8>> = remaining_hashes.into_iter().take(hit_blocks).collect();
        let instance = self.get_instance(&request.instance_id)?;
        let block_entries = self
            .storage
            .consume_pinned_blocks(&request.instance_id, instance.namespace(), &block_hashes)
            .map_err(EngineError::Storage)?;
        let plan = PreparedLoadPlan::from_cache_blocks(&instance, &block_entries)?;
        let remaining_loads = instance.registered_shards().len().max(1);
        Ok(self.insert_prepared_load(
            self.new_prepared_load_id(),
            num_tokens,
            PreparedLoadEntry {
                instance_id: request.instance_id.clone(),
                request_id: request.request_id.clone(),
                plan,
                remaining_loads,
                trace,
            },
        ))
    }

    async fn prepare_staged_load_step(
        &self,
        request: &PrepareLoadRequest,
        decode_req_id: &str,
        computed_blocks: usize,
        trace: Option<RequestTrace>,
    ) -> Result<PrepareLoadOutcome, EngineError> {
        let num_external_tokens = request
            .num_prompt_tokens
            .saturating_sub(request.num_computed_tokens.saturating_add(1));
        let num_blocks_u64 = num_external_tokens.div_ceil(request.virtual_block_size);
        if num_external_tokens == 0 || num_blocks_u64 == 0 {
            return Ok(PrepareLoadOutcome::NoPlan);
        }
        let num_blocks = usize::try_from(num_blocks_u64).map_err(|_| {
            EngineError::InvalidArgument(format!(
                "prepared staged block count does not fit usize: {num_blocks_u64}"
            ))
        })?;

        let hash_start = computed_blocks.min(request.block_hashes.len());
        let hash_end = hash_start
            .saturating_add(num_blocks)
            .min(request.block_hashes.len());
        let staged_hashes = request.block_hashes[hash_start..hash_end].to_vec();

        let prepare = self.prepare_staging_receive(PdReceivePrepareRequest {
            instance_id: request.instance_id.clone(),
            request_id: decode_req_id.to_string(),
            block_hashes: staged_hashes,
            num_blocks,
            expected_imm_count: request.decode_expected_writes,
            expire_after: None,
        })?;

        let load_plans = match self.pd_receive.try_checkout_ready(
            &request.instance_id,
            decode_req_id,
            prepare.handle.as_str(),
        )? {
            PdReceiveCheckoutStatus::Pending => return Ok(PrepareLoadOutcome::Pending),
            PdReceiveCheckoutStatus::Expired => return Ok(PrepareLoadOutcome::NoPlan),
            PdReceiveCheckoutStatus::Ready(load_plans) => load_plans,
        };

        let plan = PreparedLoadPlan::from_staged_load_plans(&load_plans, num_blocks)?;
        let instance = self.get_instance(&request.instance_id)?;
        let remaining_loads = instance.registered_shards().len().max(1);
        Ok(self.insert_prepared_load(
            self.new_prepared_load_id(),
            num_external_tokens,
            PreparedLoadEntry {
                instance_id: request.instance_id.clone(),
                request_id: request.request_id.clone(),
                plan,
                remaining_loads,
                trace,
            },
        ))
    }

    fn insert_prepared_load(
        &self,
        plan_id: u64,
        num_tokens: u64,
        entry: PreparedLoadEntry,
    ) -> PrepareLoadOutcome {
        let outcome = PrepareLoadOutcome::Plan {
            plan_id,
            num_tokens,
        };
        RequestTrace::add_optional_property(&entry.trace, "result", "plan");
        RequestTrace::add_optional_property(&entry.trace, "plan_id", plan_id.to_string());
        RequestTrace::add_optional_property(
            &entry.trace,
            "external_tokens",
            num_tokens.to_string(),
        );
        let mut prepared = self.prepared_loads.lock();
        prepared.insert(plan_id, entry);
        outcome
    }

    fn get_prepared_load_entry(
        &self,
        instance_id: &str,
        plan_id: u64,
    ) -> Result<(PreparedLoadPlan, Option<RequestTrace>), EngineError> {
        let prepared = self.prepared_loads.lock();
        let entry = prepared.get(&plan_id).ok_or_else(|| {
            EngineError::InvalidArgument(format!("prepared load plan not found: {plan_id}"))
        })?;
        if entry.instance_id != instance_id {
            return Err(EngineError::InvalidArgument(format!(
                "prepared load plan {plan_id} belongs to instance {}, not {instance_id}",
                entry.instance_id
            )));
        }
        Ok((entry.plan.clone(), entry.trace.clone()))
    }

    fn has_prepared_load_for_request(&self, instance_id: &str, request_id: &str) -> bool {
        let prepared = self.prepared_loads.lock();
        prepared
            .values()
            .any(|entry| entry.instance_id == instance_id && entry.request_id == request_id)
    }

    fn mark_prepared_load_consumed(&self, plan_id: u64) {
        let mut prepared = self.prepared_loads.lock();
        let Some(entry) = prepared.get_mut(&plan_id) else {
            return;
        };
        entry.remaining_loads = entry.remaining_loads.saturating_sub(1);
        if entry.remaining_loads == 0 {
            prepared.remove(&plan_id);
        }
    }

    /// Unpin blocks that were pinned during query.
    ///
    /// Used when load is cancelled or preempted before consumption.
    pub fn unpin_blocks(
        &self,
        instance_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<usize, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let unpinned = self
            .storage
            .unpin_blocks(instance_id, namespace, block_hashes);
        debug!(
            "unpin_blocks: instance_id={instance_id} blocks={} unpinned={unpinned}",
            block_hashes.len()
        );
        Ok(unpinned)
    }

    /// Batch load KV blocks for multiple layers asynchronously.
    ///
    /// Returns immediately after submitting the task to the GPU worker pool.
    /// The connector spin-waits on the `LoadState` until completion.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::attach(load_state_shm)?;

        let result = self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            load_state_shm,
            layer_names,
            block_ids,
            block_hashes,
        );

        if let Err(ref e) = result {
            log::error!("batch_load_kv_blocks_multi_layer pre-submit error: {e:?}");
            load_state.set_error();
        }

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_load_kv_blocks_multi_layer_inner(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();

        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        // Consume all pinned blocks reserved for this load.
        trace_scope!("load.cache_lookup", _s);
        let block_cache = self
            .storage
            .consume_pinned_blocks(instance_id, namespace, block_hashes)
            .map_err(EngineError::Storage)?;
        trace_drop!(_s);

        // Build load tasks for each layer
        trace_scope!("load.build_tasks");
        let mut layers = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_id = instance.get_layer_id(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} unknown for instance {instance_id}"
                ))
            })?;

            let registration = gpu.get_registration(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} not registered on device {device_id}"
                ))
            })?;

            let slot_id = instance.get_slot_index(layer_id, tp_rank)?;

            let blocks: Vec<LoadBlock> = block_ids
                .iter()
                .zip(block_cache.iter())
                .filter_map(|(block_id, block_entry)| {
                    let block_idx = usize::try_from(*block_id).ok()?;
                    let block = block_entry.get_slot(slot_id)?.clone();
                    Some(LoadBlock {
                        block_idx,
                        source: block,
                    })
                })
                .collect();

            if !blocks.is_empty() {
                layers.push(LayerLoadData {
                    layer_name: (*layer_name).to_string(),
                    registration,
                    blocks,
                });
            }
        }

        // Complete immediately if no blocks to load
        if layers.is_empty() {
            debug!("No blocks to load, completing immediately");
            LoadState::attach(load_state_shm)?.set_completed();
            return Ok(());
        }

        // Submit to worker pool (fire and forget)
        gpu.worker_pool().submit_load(LoadTask {
            layers,
            load_state_shm: load_state_shm.to_string(),
            traces: Vec::new(),
        })
    }

    /// Load KV blocks from prepared load plans.
    ///
    /// The scheduler-facing prepare path returns opaque plan ids. Worker-side
    /// load supplies those ids plus destination block ids after vLLM allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn load_prepared_kv_blocks_multi_layer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        items: &[PreparedLoadItem],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::attach(load_state_shm)?;

        let result = self.load_prepared_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            load_state_shm,
            layer_names,
            items,
        );

        if let Err(ref e) = result {
            log::error!("load_prepared_kv_blocks_multi_layer pre-submit error: {e:?}");
            load_state.set_error();
        }

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn load_prepared_kv_blocks_multi_layer_inner(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        items: &[PreparedLoadItem],
    ) -> Result<(), EngineError> {
        if items.is_empty() {
            LoadState::attach(load_state_shm)?.set_completed();
            return Ok(());
        }

        let instance = self.get_instance(instance_id)?;
        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        let mut layers = Vec::with_capacity(layer_names.len());
        let mut layer_indices: HashMap<&str, usize> = HashMap::with_capacity(layer_names.len());
        for layer_name in layer_names {
            let layer_id = instance.get_layer_id(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} unknown for instance {instance_id}"
                ))
            })?;
            let registration = gpu.get_registration(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} not registered on device {device_id}"
                ))
            })?;
            let index = layers.len();
            layers.push(LayerLoadData {
                layer_name: (*layer_name).to_string(),
                registration,
                blocks: Vec::new(),
            });
            layer_indices.insert(*layer_name, index);
            instance.get_slot_index(layer_id, tp_rank)?;
        }

        let mut consumed_plan_ids: Vec<u64> = Vec::with_capacity(items.len());
        let mut traces: Vec<RequestTrace> = Vec::with_capacity(items.len());

        for item in items {
            if item.plan_id == 0 {
                return Err(EngineError::InvalidArgument(
                    "prepared load item plan_id must be non-zero".to_string(),
                ));
            }
            if item.block_ids.is_empty() {
                return Err(EngineError::InvalidArgument(
                    "prepared load item block_ids must not be empty".to_string(),
                ));
            }
            let (plan, trace) = self.get_prepared_load_entry(instance_id, item.plan_id)?;
            RequestTrace::push_load_plan(&mut traces, trace, item.plan_id, item.block_ids.len());
            let plan_blocks = plan.block_count();
            if item.block_ids.len() != plan_blocks {
                return Err(EngineError::InvalidArgument(format!(
                    "prepared load plan {} has {} blocks, load requested {}",
                    item.plan_id,
                    plan_blocks,
                    item.block_ids.len()
                )));
            }

            for layer_name in layer_names {
                let layer_id = instance.get_layer_id(layer_name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "layer {layer_name} unknown for instance {instance_id}"
                    ))
                })?;
                let slot_id = instance.get_slot_index(layer_id, tp_rank)?;
                let source_blocks = plan.blocks_for_slot(slot_id).ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "prepared load plan {} missing slot {} for layer {} tp_rank {}",
                        item.plan_id, slot_id, layer_name, tp_rank
                    ))
                })?;
                if source_blocks.len() != item.block_ids.len() {
                    return Err(EngineError::InvalidArgument(format!(
                        "prepared load plan {} slot {} has {} blocks, load requested {}",
                        item.plan_id,
                        slot_id,
                        source_blocks.len(),
                        item.block_ids.len()
                    )));
                }
                let layer_index = *layer_indices.get(*layer_name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "layer {layer_name} missing from prepared load batch builder"
                    ))
                })?;
                let layer_data = &mut layers[layer_index];
                for (block_id, source) in item.block_ids.iter().zip(source_blocks.iter()) {
                    let block_idx = usize::try_from(*block_id).map_err(|_| {
                        EngineError::InvalidArgument(format!(
                            "block_id {block_id} must be non-negative for prepared load"
                        ))
                    })?;
                    layer_data.blocks.push(LoadBlock {
                        block_idx,
                        source: Arc::clone(source),
                    });
                }
            }

            consumed_plan_ids.push(item.plan_id);
        }

        layers.retain(|layer| !layer.blocks.is_empty());
        if layers.is_empty() {
            debug!("No prepared blocks to load, completing immediately");
            LoadState::attach(load_state_shm)?.set_completed();
            return Ok(());
        }

        let batch_blocks = items.iter().map(|item| item.block_ids.len()).sum();
        let layer_count = layers.len();
        let item_count = items.len();
        let traces = RequestTrace::begin_load_wait_done_batch(
            traces,
            batch_blocks,
            layer_count,
            item_count,
            std::time::Instant::now(),
        );

        gpu.worker_pool().submit_load(LoadTask {
            layers,
            load_state_shm: load_state_shm.to_string(),
            traces,
        })?;

        for plan_id in consumed_plan_ids {
            self.mark_prepared_load_consumed(plan_id);
        }
        Ok(())
    }

    /// Wait until all previously submitted save batches have been processed
    /// by the insert worker.
    ///
    /// This is a flush barrier: it guarantees that every `batch_save_kv_blocks_from_ipc`
    /// call that returned before this call will have its blocks inserted into the
    /// read cache (or inflight map) by the time this future resolves.
    pub async fn flush_saves(&self) {
        self.storage.flush_write_pipeline().await;
    }

    /// Flush write pipeline and SSD writer.
    ///
    /// Guarantees that all saves submitted before this call are both
    /// cache-visible and persisted to SSD (if SSD is enabled).
    pub async fn flush_all(&self) {
        self.storage.flush_write_pipeline().await;
        self.storage.flush_ssd().await;
    }

    /// Remove stale inflight blocks and failed_remote entries (background GC).
    ///
    /// Should be called periodically (e.g., every 30 seconds).
    pub async fn gc_stale_inflight(
        &self,
        inflight_max_age: std::time::Duration,
        failed_remote_max_age: std::time::Duration,
    ) -> (usize, usize) {
        self.storage
            .gc_stale_inflight(inflight_max_age, failed_remote_max_age)
            .await
    }

    // =========================================================================
    // Cross-node transfer: serving side
    // =========================================================================

    /// Look up blocks and lock them for RDMA transfer. Returns metadata
    /// for each found block plus a session ID for later unlock.
    pub fn query_blocks_for_transfer(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        requester_id: &str,
    ) -> (String, Vec<(BlockKey, Arc<SealedBlock>)>) {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|h| BlockKey::new(namespace.to_string(), h.clone()))
            .collect();

        let found = self.storage.get_blocks_for_transfer(&keys);
        let session_id = self.storage.lock_blocks_for_transfer(requester_id, &found);

        debug!(
            "query_blocks_for_transfer: namespace={namespace} requested={} found={} session={session_id}",
            block_hashes.len(),
            found.len(),
        );

        (session_id, found)
    }

    pub fn transfer_lock_timeout(&self) -> std::time::Duration {
        self.storage.transfer_lock_timeout()
    }

    /// Release a transfer lock session. Returns the number of blocks released.
    pub fn release_transfer_lock(&self, session_id: &str) -> usize {
        self.storage.release_transfer_lock(session_id)
    }

    /// GC expired transfer lock sessions.
    pub fn gc_expired_transfer_locks(&self) -> usize {
        self.storage.gc_expired_transfer_locks()
    }

    /// Return `(base_ptr, size)` for each contiguous pinned memory region.
    /// Used for RDMA memory registration.
    pub fn pinned_memory_regions(&self) -> Vec<(u64, usize)> {
        self.storage.pinned_memory_regions()
    }

    /// Returns true if RDMA transport is available.
    pub fn has_rdma_transport(&self) -> bool {
        self.storage.rdma_transport().is_some()
    }

    /// Take the receiver-side WRITE_WITH_IMM completion stream for P/D receive.
    ///
    /// The transfer layer exposes a single consumer. The server owns that
    /// consumer and forwards opaque immediate values to the P/D receive manager.
    pub fn take_pd_receive_imm_receiver(&self) -> Option<pegaflow_transfer::ImmCompletionReceiver> {
        self.storage
            .rdma_transport()
            .and_then(|rdma| rdma.engine().take_imm_receiver())
    }

    /// Observe one opaque P/D WRITE_WITH_IMM value.
    ///
    /// Returns true when the immediate value matched a live receive lease.
    pub fn observe_pd_receive_imm(&self, imm_data: u32) -> bool {
        self.pd_receive.observe_imm(imm_data)
    }

    /// Perform server-side RDMA handshake with connection reuse.
    ///
    /// If `client_handshake_bytes` is empty, the client believes it is already
    /// connected -- return our cached local metadata (or empty if not found).
    /// Otherwise, establish (or re-establish) a connection to the client.
    ///
    /// Returns `Err` if the handshake fails (bad client metadata, QP creation, etc.).
    pub fn rdma_accept_handshake(
        &self,
        client_addr: &str,
        client_handshake_bytes: &[u8],
    ) -> Result<Vec<u8>, String> {
        let rdma = self
            .storage
            .rdma_transport()
            .ok_or_else(|| "RDMA transport not configured".to_string())?;

        if client_handshake_bytes.is_empty() {
            // Client thinks it's already connected -- return our cached meta if we have it
            return Ok(rdma
                .engine()
                .local_meta_for(client_addr)
                .map(|m| m.to_bytes())
                .unwrap_or_default());
        }

        let client_meta = pegaflow_transfer::HandshakeMetadata::from_bytes(client_handshake_bytes)
            .map_err(|e| format!("invalid client handshake metadata: {e}"))?;

        // Client sent handshake bytes → it has no connection. If we have a stale
        // one (e.g. client restarted), tear it down so get_or_prepare creates fresh QPs.
        rdma.engine().invalidate_connection(client_addr);

        let server_meta = match rdma
            .engine()
            .get_or_prepare(client_addr)
            .map_err(|e| format!("get_or_prepare failed: {e}"))?
        {
            pegaflow_transfer::ConnectionStatus::Prepared(m) => m,
            pegaflow_transfer::ConnectionStatus::Existing => {
                unreachable!("just invalidated connection for {client_addr}")
            }
            pegaflow_transfer::ConnectionStatus::Connecting => {
                return Err(format!("handshake to {client_addr} already in progress"));
            }
        };
        rdma.engine()
            .complete_handshake(client_addr, &server_meta, &client_meta)
            .map_err(|e| format!("complete_handshake failed: {e}"))?;
        info!("RDMA handshake accepted: client={client_addr}");
        Ok(server_meta.to_bytes())
    }
}
