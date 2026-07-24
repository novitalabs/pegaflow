use super::*;
use crate::proto::engine::{NativeKvTensor, SaveLayer};
use tonic::Code;

#[test]
fn validate_query_prefetch_rejects_empty_req_id() {
    let err = GrpcEngineService::validate_query_prefetch_request(&QueryRequest {
        instance_id: "instance".to_string(),
        block_hashes: Vec::new(),
        req_id: String::new(),
        wait_for_full_prefix: false,
    })
    .expect_err("empty req_id must be rejected before engine lookup");

    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("req_id"));
}

#[test]
fn validate_query_prefetch_allows_empty_hashes() {
    GrpcEngineService::validate_query_prefetch_request(&QueryRequest {
        instance_id: "instance".to_string(),
        block_hashes: Vec::new(),
        req_id: "request".to_string(),
        wait_for_full_prefix: false,
    })
    .expect("empty block_hashes are a valid zero-hit query");
}

#[test]
fn validate_register_context_rejects_pure_argument_errors() {
    let err = validation::register_context(&RegisterContextRequest {
        instance_id: "instance".to_string(),
        namespace: "namespace".to_string(),
        client_version: pegaflow_proto::VERSION.to_string(),
        tp_rank: 1,
        tp_size: 1,
        world_size: 1,
        device_id: 0,
        layer_names: Vec::new(),
        wrapper_bytes: Vec::new(),
        num_blocks: Vec::new(),
        bytes_per_block: Vec::new(),
        kv_stride_bytes: Vec::new(),
        segments: Vec::new(),
        pp_rank: 0,
        transfer_mode: ProtoTransferMode::Direct as i32,
        page_first: false,
        native_kv_tensors: Vec::new(),
        native_alloc_size: 0,
    })
    .expect_err("tp_rank outside tp_size must be rejected at RPC boundary");

    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("tp_rank"));
}

#[test]
fn validate_register_context_rejects_client_version_mismatch() {
    let err = validation::register_context(&RegisterContextRequest {
        instance_id: "instance".to_string(),
        namespace: "namespace".to_string(),
        client_version: "0.0.0-test-mismatch".to_string(),
        tp_rank: 0,
        tp_size: 1,
        world_size: 1,
        device_id: 0,
        layer_names: Vec::new(),
        wrapper_bytes: Vec::new(),
        num_blocks: Vec::new(),
        bytes_per_block: Vec::new(),
        kv_stride_bytes: Vec::new(),
        segments: Vec::new(),
        pp_rank: 0,
        transfer_mode: ProtoTransferMode::Direct as i32,
        page_first: false,
        native_kv_tensors: Vec::new(),
        native_alloc_size: 0,
    })
    .expect_err("client/server version mismatch must be rejected before registration");

    assert_eq!(err.code(), Code::FailedPrecondition);
    assert!(err.message().contains("PegaFlow version mismatch"));
    assert!(err.message().contains("client=0.0.0-test-mismatch"));
}

#[test]
fn native_registration_requires_side_channel_safe_instance_id() {
    let err = validation::register_context(&native_register_request("instance/with/slashes"))
        .expect_err("native fd correlation must use a side-channel-safe instance id");

    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("instance_id"));
}

#[test]
fn native_registration_rejects_multi_gpu_topology() {
    let mut request = native_register_request("qwen3-dev0-018f8f75-b82e-7c10-a7d4-01abc2345678");
    request.world_size = 2;
    let err = validation::register_context(&request)
        .expect_err("v1 native ownership supports one fused allocation on one GPU");

    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("world_size=1"));
}

fn native_register_request(instance_id: &str) -> RegisterContextRequest {
    RegisterContextRequest {
        instance_id: instance_id.to_string(),
        namespace: "namespace".to_string(),
        client_version: pegaflow_proto::VERSION.to_string(),
        tp_rank: 0,
        tp_size: 1,
        world_size: 1,
        device_id: 0,
        layer_names: vec!["layer".to_string()],
        wrapper_bytes: Vec::new(),
        num_blocks: vec![1],
        bytes_per_block: vec![16],
        kv_stride_bytes: vec![0],
        segments: vec![1],
        pp_rank: 0,
        transfer_mode: ProtoTransferMode::Direct as i32,
        page_first: false,
        native_kv_tensors: vec![NativeKvTensor {
            offset_bytes: 0,
            size_bytes: 16,
            block_stride_bytes: 16,
        }],
        native_alloc_size: 16,
    }
}

#[test]
fn validate_save_layers_rejects_mismatched_block_shapes() {
    let err = GrpcEngineService::validate_save_layers(&[SaveLayer {
        layer_name: "layer_0".to_string(),
        block_ids: vec![0, 1],
        block_hashes: vec![vec![1]],
    }])
    .expect_err("service must reject malformed save shape");

    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("does not match"));
}

#[test]
fn validate_device_id_rejects_negative_values() {
    let err = GrpcEngineService::validate_device_id(-1)
        .expect_err("negative device_id must be rejected at RPC boundary");

    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("device_id"));
}
