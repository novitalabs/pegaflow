use tonic::Status;

use crate::proto::engine::RegisterContextRequest;

pub(super) fn register_context(req: &RegisterContextRequest) -> Result<(), Status> {
    let server_version = pegaflow_proto::VERSION;
    if req.client_version != server_version {
        return Err(Status::failed_precondition(format!(
            "PegaFlow version mismatch: client={} server={server_version}",
            if req.client_version.is_empty() {
                "<missing>"
            } else {
                &req.client_version
            }
        )));
    }
    if req.device_id < 0 {
        return Err(Status::invalid_argument("device_id must be non-negative"));
    }
    if req.tp_size == 0 {
        return Err(Status::invalid_argument("tp_size must be > 0"));
    }
    if req.world_size == 0 {
        return Err(Status::invalid_argument("world_size must be > 0"));
    }
    if req.tp_rank >= req.tp_size {
        return Err(Status::invalid_argument(format!(
            "tp_rank {} out of range (tp_size {})",
            req.tp_rank, req.tp_size
        )));
    }

    let layers = req.layer_names.len();
    let python = req.wrapper_bytes.len() == layers && req.native_kv_tensors.is_empty();
    let native = req.native_kv_tensors.len() == layers && req.wrapper_bytes.is_empty();
    if layers == 0 || (!python && !native) {
        return Err(Status::invalid_argument(
            "exactly one tensor payload must match layer_names",
        ));
    }
    if native && !crate::fd_channel::valid_instance_id(&req.instance_id) {
        return Err(Status::invalid_argument(
            "native registration requires a side-channel-safe instance_id",
        ));
    }
    if native && (req.tp_size != 1 || req.world_size != 1) {
        return Err(Status::invalid_argument(
            "native VMM registration currently requires tp_size=1 and world_size=1",
        ));
    }
    for tensor in &req.native_kv_tensors {
        if tensor.size_bytes == 0
            || tensor.block_stride_bytes == 0
            || tensor.offset_bytes.checked_add(tensor.size_bytes).is_none()
        {
            return Err(Status::invalid_argument(
                "invalid native KV tensor metadata",
            ));
        }
    }
    Ok(())
}
