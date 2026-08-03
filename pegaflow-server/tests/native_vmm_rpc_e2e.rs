//! Native VMM registration + save/load over gRPC.
//!
//! Covers the torch-free path:
//! VMM alloc → SCM_RIGHTS fd side-channel → `RegisterContextBatch(native_*)`
//! → D2H save → H2D load → GPU pattern check.
//!
//! Run: `cargo test -p pegaflow-server --test native_vmm_rpc_e2e --features cuda-13,rdma`

use std::ffi::c_void;
use std::net::{SocketAddr, TcpListener};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use pegaflow_core::sync_state::{LOAD_STATE_ERROR, LOAD_STATE_SUCCESS};
use pegaflow_core::{LoadState, PegaEngine, StorageConfig};
use pegaflow_server::fd_channel::FdChannel;
use pegaflow_server::proto::engine::engine_client::EngineClient;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::proto::engine::{
    LeaseLoad, LoadRequest, NativeKvTensor, QueryRequest, RegisterContextRequest, SaveLayer,
    SaveRequest, TransferMode, query_response,
};
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService, RegistryHandle};
use tokio::sync::Notify;
use tonic::transport::Server;

const INSTANCE_ID: &str = "native-vmm-rpc-e2e";
const NAMESPACE: &str = "native-vmm";
const LAYER_NAME: &str = "layer_0";
const BLOCK_COUNT: usize = 4;
const BYTES_PER_BLOCK: usize = 1024;
const TOTAL_BYTES: usize = BLOCK_COUNT * BYTES_PER_BLOCK;

#[tokio::test]
async fn native_vmm_register_save_load_roundtrip() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    ctx.bind_to_thread().expect("bind CUDA context");

    // Client-side fused VMM allocation (same process; side-channel still exercises SCM_RIGHTS).
    let client_alloc = ClientVmmAlloc::create(0, TOTAL_BYTES).expect("create VMM alloc");
    let mut expected = vec![0u8; TOTAL_BYTES];
    fill_pattern(&mut expected);
    client_alloc.copy_from_host(&expected);

    let sock_path = temp_sock_path();
    let fd_channel = FdChannel::bind(sock_path.clone()).expect("bind fd channel");

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
    let service = GrpcEngineService::new(
        Arc::clone(&engine),
        registry,
        Arc::clone(&shutdown),
        hll,
        Some(fd_channel),
    );
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(addr)
            .await
            .expect("serve");
    });

    let mut client = connect(&format!("http://127.0.0.1:{port}")).await;

    // 1) fd before register
    send_fd(
        &sock_path,
        INSTANCE_ID,
        0,
        client_alloc.export_fd().as_raw_fd(),
    );

    // 2) native RegisterContextBatch
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
            native_alloc_size: client_alloc.alloc_size() as u64,
        })
        .await
        .expect("register_context_batch")
        .into_inner();
    assert!(
        reg.status.as_ref().is_some_and(|s| s.ok),
        "register failed: {:?}",
        reg.status
    );

    // 3) D2H save
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

    // 4) query hits
    let query = client
        .query_prefetch(QueryRequest {
            instance_id: INSTANCE_ID.to_string(),
            block_hashes: hashes.clone(),
            req_id: "native-vmm-hit".into(),
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

    // 5) wipe GPU, H2D load, verify
    client_alloc.zero();
    assert!(client_alloc.copy_to_host().iter().all(|&b| b == 0));

    let load_state = LoadState::new().expect("LoadState");
    let load = client
        .load(LoadRequest {
            instance_id: INSTANCE_ID.to_string(),
            tp_rank: 0,
            device_id: 0,
            load_state_shm: load_state.shm_name().to_string(),
            layer_names: vec![LAYER_NAME.to_string()],
            loads: vec![LeaseLoad {
                lease: ready.lease,
                block_ids: (0..BLOCK_COUNT as u32).collect(),
            }],
            wait_for_completion: false,
        })
        .await
        .expect("load")
        .into_inner();
    assert!(
        load.status.as_ref().is_some_and(|s| s.ok),
        "{:?}",
        load.status
    );
    wait_load(&load_state).await;

    assert_eq!(
        client_alloc.copy_to_host(),
        expected,
        "H2D restore must match pre-save GPU pattern"
    );

    server.abort();
}

// ── client VMM allocation (exportable POSIX fd) ─────────────────────────────

struct ClientVmmAlloc {
    _ctx: Arc<CudaContext>,
    handle: sys::CUmemGenericAllocationHandle,
    base_ptr: sys::CUdeviceptr,
    alloc_size: usize,
    export_fd: OwnedFd,
}

impl ClientVmmAlloc {
    fn create(device_id: i32, size: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device_id as usize).map_err(|e| e.to_string())?;
        ctx.bind_to_thread().map_err(|e| e.to_string())?;

        let mut props: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        props.type_ = sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = device_id;
        props.requestedHandleTypes =
            sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        // Consumer GPUs reject gpuDirectRDMACapable; IT only needs POSIX export.
        props.allocFlags.gpuDirectRDMACapable = 0;

        let mut granularity = 0usize;
        check_cuda(
            unsafe {
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &props,
                    sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        );
        let alloc_size = size.div_ceil(granularity) * granularity;

        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        check_cuda(
            unsafe { sys::cuMemCreate(&mut handle, alloc_size, &props, 0) },
            "cuMemCreate",
        );

        let mut base_ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAddressReserve(&mut base_ptr, alloc_size, 0, 0, 0) },
            "cuMemAddressReserve",
        );
        if let Err(e) = check_cuda_result(
            unsafe { sys::cuMemMap(base_ptr, alloc_size, 0, handle, 0) },
            "cuMemMap",
        ) {
            unsafe {
                sys::cuMemAddressFree(base_ptr, alloc_size);
                sys::cuMemRelease(handle);
            }
            return Err(e);
        }

        let access = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        if let Err(e) = check_cuda_result(
            unsafe { sys::cuMemSetAccess(base_ptr, alloc_size, &access, 1) },
            "cuMemSetAccess",
        ) {
            unsafe {
                sys::cuMemUnmap(base_ptr, alloc_size);
                sys::cuMemAddressFree(base_ptr, alloc_size);
                sys::cuMemRelease(handle);
            }
            return Err(e);
        }

        let mut raw_fd: i32 = -1;
        check_cuda(
            unsafe {
                sys::cuMemExportToShareableHandle(
                    &mut raw_fd as *mut i32 as *mut c_void,
                    handle,
                    sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                )
            },
            "cuMemExportToShareableHandle",
        );
        // SAFETY: export transferred a new fd to us.
        let export_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        Ok(Self {
            _ctx: ctx,
            handle,
            base_ptr,
            alloc_size,
            export_fd,
        })
    }

    fn alloc_size(&self) -> usize {
        self.alloc_size
    }

    fn export_fd(&self) -> &OwnedFd {
        &self.export_fd
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert!(data.len() <= self.alloc_size);
        check_cuda(
            unsafe {
                sys::cuMemcpyHtoD_v2(self.base_ptr, data.as_ptr() as *const c_void, data.len())
            },
            "cuMemcpyHtoD_v2",
        );
    }

    fn copy_to_host(&self) -> Vec<u8> {
        let mut out = vec![0u8; TOTAL_BYTES];
        check_cuda(
            unsafe {
                sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.base_ptr, out.len())
            },
            "cuMemcpyDtoH_v2",
        );
        out
    }

    fn zero(&self) {
        check_cuda(
            unsafe { sys::cuMemsetD8_v2(self.base_ptr, 0, TOTAL_BYTES) },
            "cuMemsetD8_v2",
        );
    }
}

impl Drop for ClientVmmAlloc {
    fn drop(&mut self) {
        let _ = self._ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuMemUnmap(self.base_ptr, self.alloc_size);
            let _ = sys::cuMemAddressFree(self.base_ptr, self.alloc_size);
            let _ = sys::cuMemRelease(self.handle);
        }
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn send_fd(sock_path: &str, instance_id: &str, device_id: i32, pass: RawFd) {
    let stream = StdUnixStream::connect(sock_path).expect("connect fd socket");
    let mut payload = format!("{instance_id}\0{device_id}").into_bytes();
    let mut iov = libc::iovec {
        iov_base: payload.as_mut_ptr().cast(),
        iov_len: payload.len(),
    };
    let cmsg_space =
        unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as libc::c_uint) as usize };
    let mut cmsg_buf = vec![0u8; cmsg_space];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_buf.len();
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        assert!(!cmsg.is_null());
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as libc::c_uint) as _;
        std::ptr::write_unaligned(libc::CMSG_DATA(cmsg).cast::<RawFd>(), pass);
        msg.msg_controllen = (*cmsg).cmsg_len;
        let n = libc::sendmsg(stream.as_raw_fd(), &msg, 0);
        assert!(n >= 0, "sendmsg: {}", std::io::Error::last_os_error());
    }
}

fn fill_pattern(buf: &mut [u8]) {
    for (i, block) in buf.chunks_exact_mut(BYTES_PER_BLOCK).enumerate() {
        block.fill(((i % 251) + 1) as u8);
    }
}

fn temp_sock_path() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir()
        .join(format!("pegaflow-native-vmm-e2e-{nanos}.sock"))
        .to_string_lossy()
        .into_owned()
}

fn unused_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn connect(endpoint: &str) -> EngineClient<tonic::transport::Channel> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match EngineClient::connect(endpoint.to_string()).await {
            Ok(c) => return c,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(e) => panic!("connect: {e}"),
        }
    }
}

async fn wait_load(state: &LoadState) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let s = state.get();
        if s == LOAD_STATE_SUCCESS {
            return;
        }
        assert_ne!(s, LOAD_STATE_ERROR, "load ERROR");
        assert!(Instant::now() < deadline, "load timeout");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    check_cuda_result(result, op).unwrap_or_else(|e| panic!("{e}"));
}

fn check_cuda_result(result: sys::CUresult, op: &str) -> Result<(), String> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{op} failed: {result:?}"))
    }
}
