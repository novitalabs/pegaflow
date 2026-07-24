use super::GrpcEngineService;
use crate::proto::engine::{
    QueryRequest, RegisterContextResponse, ResponseStatus, SaveLayer, TransferSlotInfo,
};
use crate::registry::RegistryHandle;
use log::{debug, info};
use pegaflow_core::{EngineError, PegaEngine};
use tonic::Status;

impl GrpcEngineService {
    /// Drop CUDA IPC tensors and engine-side instance state for `instance_id`.
    /// Idempotent — safe to call after the instance is already gone.
    pub(super) async fn cleanup_instance(
        engine: &PegaEngine,
        registry: &RegistryHandle,
        instance_id: &str,
        reason: &'static str,
    ) {
        let cleanup = registry.drop_instance(instance_id.to_string()).await;
        if cleanup.tensor_count() > 0 {
            info!(
                "Session cleanup ({}): dropped {} CUDA tensors for instance {}",
                reason,
                cleanup.tensor_count(),
                instance_id
            );
        }
        let unregister = engine.unregister_instance(instance_id);
        registry.finish_cleanup(cleanup).await;
        if let Err(err) = unregister {
            // `InstanceMissing` is normal if the instance was never registered
            // (vllm died before any register_context_batch). Log at debug.
            debug!(
                "Session cleanup ({}): engine.unregister_instance({}) returned {}",
                reason, instance_id, err
            );
        }
    }

    pub(super) fn context_key(
        instance_id: &str,
        tp_rank: u32,
        pp_rank: u32,
        device_id: i32,
    ) -> String {
        format!("{instance_id}:tp{tp_rank}:pp{pp_rank}:dev{device_id}")
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    pub(super) fn map_engine_error(err: EngineError) -> Status {
        match err {
            EngineError::InvalidArgument(_) => Status::invalid_argument(err.to_string()),
            EngineError::InstanceMissing(_) | EngineError::WorkerMissing(_, _) => {
                Status::failed_precondition(err.to_string())
            }
            EngineError::TopologyMismatch(_) => Status::failed_precondition(err.to_string()),
            EngineError::CudaInit(_) | EngineError::Storage(_) | EngineError::Poisoned(_) => {
                Status::internal(err.to_string())
            }
        }
    }

    pub(super) fn usize_from_u64(value: u64, field: &str) -> Result<usize, Status> {
        usize::try_from(value).map_err(|_| {
            Status::invalid_argument(format!("{field}={value} does not fit into usize"))
        })
    }

    pub(super) fn usize_from_u32(value: u32, field: &str) -> Result<usize, Status> {
        usize::try_from(value).map_err(|_| {
            Status::invalid_argument(format!("{field}={value} does not fit into usize"))
        })
    }

    pub(super) fn validate_device_id(device_id: i32) -> Result<(), Status> {
        if device_id < 0 {
            return Err(Status::invalid_argument(format!(
                "device_id {device_id} must be >= 0"
            )));
        }
        Ok(())
    }

    pub(super) fn validate_save_layers(saves: &[SaveLayer]) -> Result<(), Status> {
        for layer in saves {
            if layer.block_ids.len() != layer.block_hashes.len() {
                return Err(Status::invalid_argument(format!(
                    "block_ids length {} does not match block_hashes {} for layer {}",
                    layer.block_ids.len(),
                    layer.block_hashes.len(),
                    layer.layer_name
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_query_prefetch_request(req: &QueryRequest) -> Result<(), Status> {
        if req.req_id.is_empty() {
            return Err(Status::invalid_argument("req_id must not be empty"));
        }
        Ok(())
    }

    pub(super) fn build_register_context_response() -> RegisterContextResponse {
        RegisterContextResponse {
            status: Some(Self::ok_status()),
        }
    }

    pub(super) fn build_simple_response() -> ResponseStatus {
        Self::ok_status()
    }

    pub(super) fn build_transfer_slot_info(
        raw_block: &pegaflow_core::RawBlock,
        numa_node: pegaflow_common::NumaNode,
    ) -> TransferSlotInfo {
        let layer_block = pegaflow_core::LayerBlock::new(raw_block);
        if let Some(v_ptr) = layer_block.v_ptr() {
            TransferSlotInfo {
                k_ptr: layer_block.k_ptr() as u64,
                k_size: layer_block.k_size() as u64,
                v_ptr: v_ptr as u64,
                v_size: layer_block.v_size().unwrap_or(0) as u64,
                numa_node: numa_node.0,
            }
        } else {
            TransferSlotInfo {
                k_ptr: layer_block.k_ptr() as u64,
                k_size: layer_block.k_size() as u64,
                v_ptr: 0,
                v_size: 0,
                numa_node: numa_node.0,
            }
        }
    }
}
