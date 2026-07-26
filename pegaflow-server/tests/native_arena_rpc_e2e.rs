//! Native arena registration + save/load over gRPC.
//!
//! Covers the torch-free path end to end:
//! `RegisterContextBatch(native_*)` → server allocates the arena and returns a
//! CUDA IPC handle → a child process imports it (CUDA forbids importing an IPC
//! handle in the exporting process) and writes a pattern → D2H save → child
//! wipes the arena → H2D load with `wait_for_completion` → child verifies the
//! restored bytes.
//!
//! Run: `cargo test -p pegaflow-server --test native_arena_rpc_e2e --features cuda-13,rdma`

use std::ffi::c_void;
use std::net::{SocketAddr, TcpListener};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::sys;
use pegaflow_core::{PegaEngine, StorageConfig};
use pegaflow_server::proto::engine::engine_client::EngineClient;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::proto::engine::{
    LeaseLoad, LoadRequest, NativeKvTensor, QueryRequest, RegisterContextRequest, SaveLayer,
    SaveRequest, TransferMode, UnregisterRequest, query_response,
};
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService, RegistryHandle};
use tokio::sync::Notify;
use tonic::transport::Server;

const INSTANCE_ID: &str = "native-arena-rpc-e2e";
const NAMESPACE: &str = "native-arena";
const LAYER_NAME: &str = "layer_0";
const BLOCK_COUNT: usize = 4;
const BYTES_PER_BLOCK: usize = 1024;
const TOTAL_BYTES: usize = BLOCK_COUNT * BYTES_PER_BLOCK;

const MODE_ENV: &str = "PEGAFLOW_ARENA_E2E_MODE";
const HANDLE_ENV: &str = "PEGAFLOW_ARENA_E2E_HANDLE_HEX";
const OUT_ENV: &str = "PEGAFLOW_ARENA_E2E_OUT";

#[tokio::test]
async fn native_arena_register_save_load_roundtrip() {
    let engine = Arc::new(
        PegaEngine::new_with_config(
            16 << 20,
            false,
            StorageConfig {
                enable_lfu_admission: false,
                ..StorageConfig::default()
            },
        )
        .expect("engine"),
    );
    // Torch-free registry: native registration only.
    let registry = RegistryHandle::spawn(CudaTensorRegistry::empty());
    let port = unused_port();
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let shutdown = Arc::new(Notify::new());
    let hll = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::MultiWindowHllTracker::new(
            vec![("24h".into(), Duration::from_secs(86400))],
            14,
        ),
    ));
    let service = GrpcEngineService::new(Arc::clone(&engine), registry, Arc::clone(&shutdown), hll);
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(addr)
            .await
            .expect("serve");
    });

    let mut client = connect(&format!("http://127.0.0.1:{port}")).await;

    // 1) Native RegisterContextBatch: the server allocates the arena and
    //    returns its CUDA IPC handle.
    let reg = client
        .register_context_batch(RegisterContextRequest {
            instance_id: INSTANCE_ID.to_string(),
            namespace: NAMESPACE.to_string(),
            client_version: pegaflow_proto::VERSION.to_string(),
            tp_rank: 0,
            tp_size: 1,
            world_size: 1,
            device_id: 0,
            layer_names: vec![LAYER_NAME.to_string()],
            wrapper_bytes: vec![],
            num_blocks: vec![BLOCK_COUNT as u64],
            bytes_per_block: vec![BYTES_PER_BLOCK as u64],
            kv_stride_bytes: vec![0],
            segments: vec![1],
            pp_rank: 0,
            transfer_mode: TransferMode::Direct as i32,
            page_first: false,
            native_kv_tensors: vec![NativeKvTensor {
                offset_bytes: 0,
                size_bytes: TOTAL_BYTES as u64,
                block_stride_bytes: BYTES_PER_BLOCK as u64,
            }],
            native_alloc_size: TOTAL_BYTES as u64,
        })
        .await
        .expect("register_context_batch")
        .into_inner();
    assert!(
        reg.status.as_ref().is_some_and(|s| s.ok),
        "register failed: {:?}",
        reg.status
    );
    assert_eq!(
        reg.arena_ipc_handle.len(),
        64,
        "expected a CUipcMemHandle in the response"
    );
    let handle_hex = hex_encode(&reg.arena_ipc_handle);

    // 2) The "client" (a separate process, as CUDA IPC requires) writes the
    //    pattern into its imported mapping.
    run_child("write", &handle_hex, None);

    // 3) D2H save reads through the server's own arena pointers.
    let hashes: Vec<Vec<u8>> = (0..BLOCK_COUNT)
        .map(|i| {
            let mut h = vec![7u8];
            h.extend_from_slice(&(i as u32).to_le_bytes());
            h
        })
        .collect();
    let save = client
        .save(SaveRequest {
            instance_id: INSTANCE_ID.to_string(),
            tp_rank: 0,
            device_id: 0,
            pp_rank: 0,
            saves: vec![SaveLayer {
                layer_name: LAYER_NAME.to_string(),
                block_ids: (0..BLOCK_COUNT as u32).collect(),
                block_hashes: hashes.clone(),
            }],
        })
        .await
        .expect("save")
        .into_inner();
    assert!(
        save.status.as_ref().is_some_and(|s| s.ok),
        "{:?}",
        save.status
    );
    engine.flush_saves().await;

    // 4) Query hits.
    let query = client
        .query_prefetch(QueryRequest {
            instance_id: INSTANCE_ID.to_string(),
            block_hashes: hashes.clone(),
            req_id: "native-arena-hit".into(),
            wait_for_full_prefix: false,
        })
        .await
        .expect("query")
        .into_inner();
    let ready = match query.outcome {
        Some(query_response::Outcome::Ready(r)) => r,
        other => panic!("expected Ready, got {other:?}"),
    };
    assert_eq!(ready.num_hit_blocks as usize, BLOCK_COUNT);
    assert!(!ready.lease.is_empty());

    // 5) Client wipes the arena, then a synchronous load restores it.
    run_child("wipe", &handle_hex, None);
    let load = client
        .load(LoadRequest {
            instance_id: INSTANCE_ID.to_string(),
            tp_rank: 0,
            device_id: 0,
            load_state_shm: String::new(),
            layer_names: vec![LAYER_NAME.to_string()],
            loads: vec![LeaseLoad {
                lease: ready.lease,
                block_ids: (0..BLOCK_COUNT as u32).collect(),
            }],
            wait_for_completion: true,
        })
        .await
        .expect("load")
        .into_inner();
    assert!(
        load.status.as_ref().is_some_and(|s| s.ok),
        "{:?}",
        load.status
    );

    // 6) Client reads its mapping back; must be bit-identical to the pattern.
    let out = std::env::temp_dir().join(format!("pegaflow-arena-e2e-{}", std::process::id()));
    run_child("dump", &handle_hex, Some(&out));
    let restored = std::fs::read(&out).expect("read child dump");
    let _ = std::fs::remove_file(&out);
    let mut expected = vec![0u8; TOTAL_BYTES];
    fill_pattern(&mut expected);
    assert_eq!(
        restored, expected,
        "restored arena must match saved pattern"
    );

    // 7) Unregister frees the arena.
    let unreg = client
        .unregister_context(UnregisterRequest {
            instance_id: INSTANCE_ID.to_string(),
        })
        .await
        .expect("unregister")
        .into_inner();
    assert!(unreg.status.as_ref().is_some_and(|s| s.ok));

    server.abort();
}

/// Child-process entry point, driven by `run_child`. Ignored so plain
/// `cargo test` never runs it directly.
#[test]
#[ignore = "spawned as a helper process by native_arena_register_save_load_roundtrip"]
fn arena_ipc_child_helper() {
    let mode = std::env::var(MODE_ENV).expect("child mode");
    let handle = hex_decode(&std::env::var(HANDLE_ENV).expect("child handle"));
    assert_eq!(handle.len(), 64);

    let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
    ctx.bind_to_thread().expect("bind CUDA context");

    let mut ipc_handle = sys::CUipcMemHandle { reserved: [0; 64] };
    for (dst, src) in ipc_handle.reserved.iter_mut().zip(&handle) {
        *dst = *src as i8;
    }
    let mut base_ptr: sys::CUdeviceptr = 0;
    // SAFETY: the handle references a live allocation owned by the parent
    // process (the server keeps it registered while children run).
    check_cuda(
        unsafe {
            sys::cuIpcOpenMemHandle_v2(
                &mut base_ptr,
                ipc_handle,
                sys::CUipcMem_flags_enum::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
            )
        },
        "cuIpcOpenMemHandle",
    );

    match mode.as_str() {
        "write" => {
            let mut pattern = vec![0u8; TOTAL_BYTES];
            fill_pattern(&mut pattern);
            // SAFETY: the imported mapping spans TOTAL_BYTES.
            check_cuda(
                unsafe {
                    sys::cuMemcpyHtoD_v2(base_ptr, pattern.as_ptr() as *const c_void, TOTAL_BYTES)
                },
                "cuMemcpyHtoD",
            );
        }
        "wipe" => {
            // SAFETY: same mapping bounds as above.
            check_cuda(
                unsafe { sys::cuMemsetD8_v2(base_ptr, 0, TOTAL_BYTES) },
                "cuMemsetD8",
            );
        }
        "dump" => {
            let out = std::env::var(OUT_ENV).expect("child out path");
            let mut bytes = vec![0u8; TOTAL_BYTES];
            // SAFETY: same mapping bounds as above.
            check_cuda(
                unsafe {
                    sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr() as *mut c_void, base_ptr, TOTAL_BYTES)
                },
                "cuMemcpyDtoH",
            );
            std::fs::write(out, bytes).expect("write dump");
        }
        other => panic!("unknown child mode {other}"),
    }

    // SAFETY: base_ptr came from cuIpcOpenMemHandle above.
    check_cuda(
        unsafe { sys::cuIpcCloseMemHandle(base_ptr) },
        "cuIpcCloseMemHandle",
    );
}

fn run_child(mode: &str, handle_hex: &str, out: Option<&std::path::Path>) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["arena_ipc_child_helper", "--exact", "--include-ignored"])
        .env(MODE_ENV, mode)
        .env(HANDLE_ENV, handle_hex);
    if let Some(out) = out {
        cmd.env(OUT_ENV, out);
    }
    let output = cmd.output().expect("spawn child helper");
    assert!(
        output.status.success(),
        "child '{mode}' failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn fill_pattern(buf: &mut [u8]) {
    for (i, byte) in buf.iter_mut().enumerate() {
        *byte = ((i * 31 + 7) % 251) as u8;
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn unused_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

async fn connect(endpoint: &str) -> EngineClient<tonic::transport::Channel> {
    for _ in 0..50 {
        if let Ok(client) = EngineClient::connect(endpoint.to_string()).await {
            return client;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("gRPC server did not come up at {endpoint}");
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "{op} failed");
}
