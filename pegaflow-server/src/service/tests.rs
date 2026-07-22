use super::*;
use crate::proto::engine::SaveLayer;
use tonic::Code;

#[test]
fn validate_query_prefetch_rejects_empty_req_id() {
    let err = GrpcEngineService::validate_query_prefetch_request(&QueryRequest {
        instance_id: "instance".to_string(),
        block_hashes: Vec::new(),
        req_id: String::new(),
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
