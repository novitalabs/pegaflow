//! Instance and GPU context management for PegaFlow.
//!
//! This module provides the hierarchical context structure for managing
//! multi-tenant inference instances and their associated GPU resources.
//!
//! An instance goes through two phases:
//!
//! 1. **Registering** — workers attach their GPUs one by one, each declaring
//!    the KV cache layers that actually exist on that device. The engine makes
//!    no assumption about the model's layer count.
//! 2. **Sealed** — when the last of `world_size` workers has registered, the
//!    union of all declared layer names is sorted into a dense layer-id space
//!    and the slot topology is validated. Save/load/query are only possible on
//!    a sealed instance.
//!
//! Deriving the id space from what workers actually register (instead of a
//! connector-side guess from model config) is what makes optional speculative
//! MTP layers, external drafters, and hybrid attention layouts work without
//! special cases: the layer set *is* the topology.
//!
//! One engine instance serves the workers of one node: all `world_size`
//! workers must register with the same server. This is not new to sealing —
//! CUDA IPC registration is same-host only, and the session watcher and
//! query-lease consumption already count to `world_size` on a single server.

use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::CudaContext;
use log::info;

use crate::layout::KVCacheLayout;
use crate::{EngineError, TransferMode, gpu_worker::GpuWorkerPool};
use pegaflow_common::NumaNode;

/// Registration state protected by a single mutex.
struct RegistrationState {
    /// GPU contexts indexed by CUDA device ID.
    gpu_contexts: HashMap<i32, Arc<GpuContext>>,

    /// Sealed layer-id space. `None` while workers are still registering.
    topology: Option<Arc<LayerTopology>>,
}

/// Dense layer-id space sealed from the union of registered layers.
///
/// Layer ids are the rank of the layer name in sorted order, so every
/// instance that registers the same layer set derives the same ids — the
/// property block slot layout depends on.
#[derive(Debug)]
pub(crate) struct LayerTopology {
    name_to_id: HashMap<String, usize>,
    tp_size: usize,
}

impl LayerTopology {
    /// Look up the numeric ID for a layer name.
    pub(crate) fn layer_id(&self, layer_name: &str) -> Result<usize, EngineError> {
        self.name_to_id.get(layer_name).copied().ok_or_else(|| {
            EngineError::InvalidArgument(format!("layer {layer_name} is not registered"))
        })
    }

    /// Number of layers in the sealed id space.
    pub(crate) fn num_layers(&self) -> usize {
        self.name_to_id.len()
    }

    /// Total number of storage slots, organized as `[layer][tp_rank]`.
    pub(crate) fn total_slots(&self) -> usize {
        self.num_layers() * self.tp_size
    }

    /// Compute the slot index for a specific layer and TP rank.
    pub(crate) fn slot_index(&self, layer_id: usize, tp_rank: usize) -> Result<usize, EngineError> {
        if layer_id >= self.num_layers() {
            return Err(EngineError::InvalidArgument(format!(
                "layer_id {} out of range ({} layers)",
                layer_id,
                self.num_layers()
            )));
        }
        if tp_rank >= self.tp_size {
            return Err(EngineError::InvalidArgument(format!(
                "tp_rank {} out of range (tp_size {})",
                tp_rank, self.tp_size
            )));
        }
        Ok(layer_id * self.tp_size + tp_rank)
    }
}

/// Per-GPU execution context.
///
/// Each `GpuContext` manages:
/// - CUDA context lifetime for a specific device
/// - KV cache registrations for all layers on this GPU
/// - Asynchronous worker pool for load/save operations
/// - NUMA affinity for memory allocation optimization
pub struct GpuContext {
    /// CUDA device ID for diagnostics and duplicate registration checks.
    device_id: i32,

    /// Effective TP rank represented by this GPU context.
    tp_rank: usize,

    /// Pipeline-parallel rank represented by this GPU context.
    pp_rank: usize,

    /// Preferred NUMA node for this GPU (for memory allocation).
    preferred_numa: NumaNode,

    /// KV cache layouts by layer name.
    kv_caches: HashMap<String, KVCacheLayout>,

    /// CUDA context handle (kept alive for the lifetime of this context).
    _cuda_ctx: Arc<CudaContext>,

    /// Worker thread pool for asynchronous GPU operations.
    worker_pool: GpuWorkerPool,
}

impl GpuContext {
    /// Create a new GPU context for the specified device.
    ///
    /// The `numa_node` should be obtained from `NumaTopology::numa_for_gpu()`.
    /// Worker threads will be pinned to the specified NUMA node for optimal
    /// memory locality during transfers.
    ///
    /// # Errors
    /// Returns `EngineError::CudaInit` if CUDA context creation or worker
    /// pool initialization fails.
    fn new(
        cuda_ctx: Arc<CudaContext>,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
        kv_caches: HashMap<String, KVCacheLayout>,
    ) -> Result<Self, EngineError> {
        let worker_pool = GpuWorkerPool::spawn(device_id, numa_node, transfer_mode)?;

        Ok(Self {
            device_id,
            tp_rank,
            pp_rank,
            preferred_numa: numa_node,
            kv_caches,
            _cuda_ctx: cuda_ctx,
            worker_pool,
        })
    }

    /// Get the preferred NUMA node for this GPU.
    pub(crate) fn preferred_numa(&self) -> NumaNode {
        self.preferred_numa
    }

    /// CUDA device ID represented by this shard.
    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Effective TP rank represented by this shard.
    pub(crate) fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    /// Pipeline-parallel rank represented by this shard.
    pub(crate) fn pp_rank(&self) -> usize {
        self.pp_rank
    }

    /// Retrieve a layer's KV cache layout.
    pub(crate) fn get_layout(&self, layer_name: &str) -> Option<KVCacheLayout> {
        self.kv_caches.get(layer_name).cloned()
    }

    /// Access the worker pool for submitting GPU operations.
    pub(crate) fn worker_pool(&self) -> &GpuWorkerPool {
        &self.worker_pool
    }
}

pub(crate) struct GpuRegistration {
    pub(crate) device_id: i32,
    pub(crate) tp_rank: usize,
    pub(crate) pp_rank: usize,
    pub(crate) numa_node: NumaNode,
    pub(crate) transfer_mode: TransferMode,
    pub(crate) kv_caches: HashMap<String, KVCacheLayout>,
}

/// Instance context for a model inference process.
///
/// An `InstanceContext` represents a single inference instance (e.g., one
/// `vllm serve` process) and manages all its GPU contexts and layer metadata.
/// It supports tensor parallelism via the `tp_size` parameter.
pub struct InstanceContext {
    /// Unique instance identifier.
    id: String,

    /// Namespace for model isolation (e.g., model name or tenant ID).
    namespace: String,

    /// Tensor parallelism degree (number of GPUs per instance).
    tp_size: usize,

    /// Total worker count for this instance. Sealing happens when this many
    /// devices have registered.
    world_size: usize,

    /// Registration state and GPU contexts protected by a single mutex.
    state: Mutex<RegistrationState>,
}

impl InstanceContext {
    /// Create a new instance context.
    ///
    /// # Errors
    /// Returns an error string if topology parameters are invalid.
    pub(crate) fn new(
        id: String,
        namespace: String,
        tp_size: usize,
        world_size: usize,
    ) -> Result<Self, String> {
        if tp_size == 0 || world_size == 0 {
            return Err("tp_size and world_size must be > 0".into());
        }

        Ok(Self {
            id,
            namespace,
            tp_size,
            world_size,
            state: Mutex::new(RegistrationState {
                gpu_contexts: HashMap::new(),
                topology: None,
            }),
        })
    }

    /// Access the sealed layer topology.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` while workers are still
    /// registering — save/load must not run against a partial topology.
    pub(crate) fn sealed_topology(&self) -> Result<Arc<LayerTopology>, EngineError> {
        let state = self.state.lock();
        state.topology.clone().ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "instance {} registration incomplete: {}/{} workers registered",
                self.id,
                state.gpu_contexts.len(),
                self.world_size
            ))
        })
    }

    fn ensure_accepting_registrations(
        &self,
        state: &RegistrationState,
        device_id: i32,
    ) -> Result<(), EngineError> {
        // Check the duplicate device first: a restarted worker re-registering
        // against a stale sealed instance should hear "device already exists"
        // (the instance must be unregistered first), not a generic "fully
        // registered" that hides the actual cause.
        if state.gpu_contexts.contains_key(&device_id) {
            return Err(EngineError::InvalidArgument(format!(
                "GPU context for device {device_id} already exists"
            )));
        }
        if state.topology.is_some() {
            return Err(EngineError::InvalidArgument(format!(
                "instance {} is already fully registered ({} workers); \
                 device {device_id} cannot join",
                self.id, self.world_size
            )));
        }
        Ok(())
    }

    /// Seal the layer-id space from every registered GPU plus the pending one.
    ///
    /// Validates, in order:
    /// 1. Layers sharing a name declare the same block geometry on every
    ///    device (same name ⇒ same stored bytes).
    /// 2. Every `(layer, tp_rank)` slot has at least one owner. A slot may
    ///    have several owners: MLA models replicate KV across TP ranks, so the
    ///    connector collapses them to one effective tp_rank and every worker
    ///    registers the same slot range from its own device (rank 0 saves,
    ///    each device loads into its own copy). Replica owners can only
    ///    disagree on pp_rank, which would mean two pipeline stages claimed
    ///    the same layer and is rejected.
    fn seal_topology(
        &self,
        state: &RegistrationState,
        pending: &GpuContext,
    ) -> Result<LayerTopology, EngineError> {
        let gpus = || {
            state
                .gpu_contexts
                .values()
                .map(Arc::as_ref)
                .chain(std::iter::once(pending))
        };

        // Union of layer names with geometry consistency across devices.
        let mut geometry_by_name: HashMap<&str, (usize, bool)> = HashMap::new();
        for gpu in gpus() {
            for (name, layout) in &gpu.kv_caches {
                let geometry = (layout.segment_bytes(), layout.is_split());
                match geometry_by_name.insert(name, geometry) {
                    None => {}
                    Some(existing) if existing == geometry => {}
                    Some((existing_bytes, existing_split)) => {
                        return Err(EngineError::InvalidArgument(format!(
                            "layer {name} registered with inconsistent geometry: \
                             segment_bytes={existing_bytes} split={existing_split} vs \
                             segment_bytes={} split={} on device {}",
                            layout.segment_bytes(),
                            layout.is_split(),
                            gpu.device_id(),
                        )));
                    }
                }
            }
        }

        let mut names: Vec<String> = geometry_by_name.keys().map(|s| s.to_string()).collect();
        names.sort_unstable();
        let name_to_id: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(id, name)| (name.clone(), id))
            .collect();
        let topology = LayerTopology {
            name_to_id,
            tp_size: self.tp_size,
        };

        // Per slot: (device_id, pp_rank) of the first registered owner.
        let mut owners: Vec<Option<(i32, usize)>> = vec![None; topology.total_slots()];
        for gpu in gpus() {
            for layer_name in gpu.kv_caches.keys() {
                let layer_id = topology.layer_id(layer_name)?;
                let slot_id = layer_id * self.tp_size + gpu.tp_rank();
                match owners[slot_id] {
                    None => owners[slot_id] = Some((gpu.device_id(), gpu.pp_rank())),
                    Some((existing_device, existing_pp_rank)) => {
                        if existing_pp_rank != gpu.pp_rank() {
                            return Err(EngineError::InvalidArgument(format!(
                                "layer {layer_name} claimed by different pipeline stages: \
                                 device={existing_device} pp_rank={existing_pp_rank}; \
                                 device={} pp_rank={} tp_rank={}",
                                gpu.device_id(),
                                gpu.pp_rank(),
                                gpu.tp_rank(),
                            )));
                        }
                    }
                }
            }
        }

        if let Some(missing_slot) = owners.iter().position(Option::is_none) {
            let layer_id = missing_slot / self.tp_size;
            let tp_rank = missing_slot % self.tp_size;
            return Err(EngineError::InvalidArgument(format!(
                "instance {} has incomplete KV registration: layer {} has no owner \
                 for tp_rank {tp_rank} after all {} workers registered",
                self.id, names[layer_id], self.world_size
            )));
        }

        Ok(topology)
    }

    /// Build a GPU context for the specified device.
    ///
    /// This method lazily initializes CUDA contexts as devices are first accessed.
    /// The `numa_node` should be obtained from `NumaTopology::numa_for_gpu()`.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` for negative device IDs,
    /// or `EngineError::CudaInit` if CUDA context creation fails.
    fn build_gpu_context(
        &self,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
        kv_caches: HashMap<String, KVCacheLayout>,
    ) -> Result<Arc<GpuContext>, EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {} must be >= 0",
                device_id
            )));
        }

        let cuda_ctx = CudaContext::new(device_id as usize)
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;

        Ok(Arc::new(GpuContext::new(
            cuda_ctx,
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
        )?))
    }

    /// Get an existing GPU context without creating one.
    pub(crate) fn get_gpu(&self, device_id: i32) -> Option<Arc<GpuContext>> {
        let state = self.state.lock();
        state.gpu_contexts.get(&device_id).cloned()
    }

    /// Get a GPU context and verify it belongs to the requested save group.
    pub(crate) fn get_gpu_for_save_group(
        &self,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
    ) -> Result<Arc<GpuContext>, EngineError> {
        let gpu = self
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(self.id.clone(), device_id))?;

        if gpu.tp_rank() != tp_rank || gpu.pp_rank() != pp_rank {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {device_id} is registered for tp_rank {}, pp_rank {}, but save requested tp_rank {tp_rank}, pp_rank {pp_rank}",
                gpu.tp_rank(),
                gpu.pp_rank()
            )));
        }

        Ok(gpu)
    }

    /// Register a new GPU with all its KV cache layers.
    ///
    /// The registration that completes the worker set (`world_size` devices)
    /// also seals the instance topology; if sealing fails the registration is
    /// rejected as a whole and the instance keeps waiting for a valid worker
    /// set.
    ///
    /// # Errors
    /// - `EngineError::InvalidArgument` if the GPU is already registered, the
    ///   instance is already sealed, or sealing detects an invalid topology
    /// - `EngineError::CudaInit` if GPU context creation fails
    pub(crate) fn register_new_gpu(
        &self,
        registration: GpuRegistration,
    ) -> Result<(), EngineError> {
        let GpuRegistration {
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
        } = registration;

        if tp_rank >= self.tp_size {
            return Err(EngineError::InvalidArgument(format!(
                "tp_rank {} out of range (tp_size {})",
                tp_rank, self.tp_size
            )));
        }
        if kv_caches.is_empty() {
            return Err(EngineError::InvalidArgument(format!(
                "device {device_id} registered no KV cache layers"
            )));
        }

        {
            let state = self.state.lock();
            self.ensure_accepting_registrations(&state, device_id)?;
        }

        let ctx = self.build_gpu_context(
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
        )?;

        {
            let mut state = self.state.lock();
            self.ensure_accepting_registrations(&state, device_id)?;

            let topology = if state.gpu_contexts.len() + 1 == self.world_size {
                Some(Arc::new(self.seal_topology(&state, &ctx)?))
            } else {
                None
            };

            state.gpu_contexts.insert(device_id, ctx);
            if let Some(topology) = topology {
                info!(
                    "Sealed instance topology: instance={}, num_layers={}, tp_size={}, \
                     world_size={}, total_slots={}",
                    self.id,
                    topology.num_layers(),
                    self.tp_size,
                    self.world_size,
                    topology.total_slots()
                );
                state.topology = Some(topology);
            }
        }

        info!("Initialized GPU context: device_id={device_id}, numa_node={numa_node}");
        Ok(())
    }

    /// Access the instance namespace.
    pub(crate) fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Total worker count registered for this instance.
    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }

    /// Verify that the topology matches expected values.
    ///
    /// Returns `Ok(())` if matches, or an error message describing the mismatch.
    pub(crate) fn verify_topology(&self, tp_size: usize, world_size: usize) -> Result<(), String> {
        if self.tp_size != tp_size || self.world_size != world_size {
            return Err(format!(
                "exists with tp={}, world={}; requested tp={}, world={}",
                self.tp_size, self.world_size, tp_size, world_size
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
