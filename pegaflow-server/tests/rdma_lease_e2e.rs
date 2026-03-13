//! End-to-end RDMA lease test.
//!
//! Exercises the full flow:
//!   1. Save KV blocks via PegaEngine (GPU → pinned pool)
//!   2. Register pinned pool with owner's MooncakeTransferEngine
//!   3. Start gRPC server with RdmaTransfer service
//!   4. Client: AcquireLease → RDMA read each slot → verify data → ReleaseLease
//!
//! Requires:
//!   - CUDA GPU (device 0)
//!   - RDMA NIC (mlx5_0)
//!
//! Run with:
//!   cargo test -p pegaflow-server --test rdma_lease_e2e -- --ignored --nocapture

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, sys};
use pegaflow_core::{LayerSave, PegaEngine, StorageConfig};
use pegaflow_server::GrpcRdmaTransferService;
use pegaflow_server::proto::engine::rdma_transfer_server::RdmaTransferServer;
use pegaflow_server::proto::engine::{
    AcquireLeaseRequest, ReleaseLeaseRequest, rdma_transfer_client::RdmaTransferClient,
};
use pegaflow_transfer::MooncakeTransferEngine;
use tonic::transport::Server;

const NIC: &str = "mlx5_0";
const OWNER_RDMA_PORT: u16 = 56070;
const REQUESTER_RDMA_PORT: u16 = 56080;
const GRPC_PORT: u16 = 50199;

const NUM_BLOCKS: usize = 4;
const BLOCK_SIZE: usize = 4096;
const TOTAL_GPU_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE;
const NAMESPACE: &str = "test-ns";
const INSTANCE_ID: &str = "rdma-e2e-inst";
const LAYER_NAME: &str = "layer_0";

// ---------------------------------------------------------------------------
// GPU buffer helper (same pattern as pegaflow-core tests)
// ---------------------------------------------------------------------------

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        check_cuda(
            unsafe {
                sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const std::ffi::c_void, self.len)
            },
            "cuMemcpyHtoD_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_block_ids(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

fn make_block_hashes(n: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut h = vec![salt];
            h.extend_from_slice(&(i as u32).to_le_bytes());
            h
        })
        .collect()
}

fn fill_test_pattern(buf: &mut [u8], block_size: usize) {
    for (i, block) in buf.chunks_exact_mut(block_size).enumerate() {
        block.fill(((i % 251) + 1) as u8);
    }
}

fn test_engine() -> PegaEngine {
    PegaEngine::new_with_config(
        16 << 20, // 16 MB
        false,
        StorageConfig {
            enable_lfu_admission: false,
            hint_value_size_bytes: None,
            max_prefetch_blocks: 100,
            baking_store_config: None,
            ssd_cache_config: None,
            enable_numa_affinity: false,
            transfer_engine: None,
        },
    )
}

async fn wait_for_cache(
    engine: &PegaEngine,
    instance_id: &str,
    hashes: &[Vec<u8>],
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let (hit, _) = engine
            .count_prefix_hit_blocks(instance_id, hashes)
            .expect("count_prefix_hit_blocks");
        if hit >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} cached blocks (got {hit})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// ---------------------------------------------------------------------------
// E2E test
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires RDMA NIC (mlx5_0) + CUDA GPU"]
async fn rdma_lease_acquire_read_release() {
    let _ctx = CudaContext::new(0).expect("CUDA device 0");

    // 1. Create PegaEngine, save blocks via GPU → pinned pool
    let engine = Arc::new(test_engine());

    let gpu = GpuBuffer::alloc(TOTAL_GPU_SIZE);
    let mut host_data = vec![0u8; TOTAL_GPU_SIZE];
    fill_test_pattern(&mut host_data, BLOCK_SIZE);
    gpu.copy_from_host(&host_data);

    let block_hashes = make_block_hashes(NUM_BLOCKS, 0xAB);

    engine
        .register_context_layer(
            INSTANCE_ID,
            NAMESPACE,
            0, // device_id
            LAYER_NAME.to_string(),
            gpu.as_u64(),
            TOTAL_GPU_SIZE,
            NUM_BLOCKS,
            BLOCK_SIZE,
            0, // kv_stride_bytes (contiguous)
            1, // segments
            0, // tp_rank
            1, // tp_size
            1, // world_size
            1, // num_layers
        )
        .expect("register_context_layer");

    engine
        .batch_save_kv_blocks_from_ipc(
            INSTANCE_ID,
            0, // tp_rank
            0, // device_id
            vec![LayerSave {
                layer_name: LAYER_NAME.to_string(),
                block_ids: make_block_ids(NUM_BLOCKS),
                block_hashes: block_hashes.clone(),
            }],
        )
        .await
        .expect("batch_save");

    wait_for_cache(
        &engine,
        INSTANCE_ID,
        &block_hashes,
        NUM_BLOCKS,
        Duration::from_secs(5),
    )
    .await;
    eprintln!("[e2e] blocks cached: {NUM_BLOCKS}");

    // 2. Register pinned pool with owner RDMA engine
    let mut owner_rdma = MooncakeTransferEngine::new();
    owner_rdma
        .initialize(NIC, OWNER_RDMA_PORT)
        .expect("owner RDMA init");

    for (ptr, size) in engine.pinned_pool_regions() {
        owner_rdma
            .register_memory(ptr, size)
            .expect("register pinned pool region");
        eprintln!("[e2e] registered pool region: ptr={ptr:#x} size={size}");
    }

    let owner_session = owner_rdma.get_session_id();
    eprintln!("[e2e] owner session: {owner_session}");

    // 3. Start gRPC server with RdmaTransfer service
    let grpc_addr: SocketAddr = format!("127.0.0.1:{GRPC_PORT}").parse().unwrap();
    let rdma_service = GrpcRdmaTransferService::new(Arc::clone(&engine));

    let server_handle = tokio::spawn({
        async move {
            Server::builder()
                .add_service(RdmaTransferServer::new(rdma_service))
                .serve(grpc_addr)
                .await
                .expect("gRPC server");
        }
    });

    // Give the server a moment to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 4. Create requester RDMA engine + receive buffer
    let mut requester_rdma = MooncakeTransferEngine::new();
    requester_rdma
        .initialize(NIC, REQUESTER_RDMA_PORT)
        .expect("requester RDMA init");

    // Allocate page-aligned receive buffer via mmap
    let recv_buf_size = TOTAL_GPU_SIZE;
    let recv_buf = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            recv_buf_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
            -1,
            0,
        )
    };
    assert_ne!(recv_buf, libc::MAP_FAILED, "mmap failed");
    // Lock pages for RDMA
    unsafe {
        libc::mlock(recv_buf, recv_buf_size);
    }

    requester_rdma
        .register_memory(recv_buf as u64, recv_buf_size)
        .expect("register recv buffer");

    // 5. gRPC: AcquireLease
    let mut client = RdmaTransferClient::connect(format!("http://127.0.0.1:{GRPC_PORT}"))
        .await
        .expect("connect to gRPC server");

    let acquire_resp = client
        .acquire_lease(AcquireLeaseRequest {
            requester_node_id: "requester-node-1".to_string(),
            namespace: NAMESPACE.to_string(),
            block_hashes: block_hashes.clone(),
            lease_duration_secs: 60,
        })
        .await
        .expect("acquire_lease RPC")
        .into_inner();

    let lease_id = acquire_resp.lease_id.clone();
    eprintln!(
        "[e2e] lease acquired: id={} blocks={} missing={}",
        lease_id,
        acquire_resp.blocks.len(),
        acquire_resp.missing_hashes.len(),
    );

    assert_eq!(acquire_resp.blocks.len(), NUM_BLOCKS);
    assert!(acquire_resp.missing_hashes.is_empty());

    // 6. RDMA read each block's K/V segments
    let mut recv_offset = 0usize;
    for (block_idx, descriptor) in acquire_resp.blocks.iter().enumerate() {
        assert_eq!(descriptor.slots.len(), 1, "single TP rank → 1 slot");
        let slot = &descriptor.slots[0];

        // Read K segment
        let k_size = slot.k_size as usize;
        let local_k_ptr = (recv_buf as u64) + recv_offset as u64;
        let read_bytes = requester_rdma
            .transfer_sync_read(&owner_session, local_k_ptr, slot.k_addr, k_size)
            .unwrap_or_else(|e| panic!("RDMA read K of block {block_idx}: {e}"));
        assert_eq!(read_bytes, k_size);
        recv_offset += k_size;

        // Read V segment
        let v_size = slot.v_size as usize;
        let local_v_ptr = (recv_buf as u64) + recv_offset as u64;
        let read_bytes = requester_rdma
            .transfer_sync_read(&owner_session, local_v_ptr, slot.v_addr, v_size)
            .unwrap_or_else(|e| panic!("RDMA read V of block {block_idx}: {e}"));
        assert_eq!(read_bytes, v_size);
        recv_offset += v_size;
    }

    eprintln!("[e2e] RDMA read complete: {recv_offset} bytes");

    // 7. Verify data integrity
    //    The contiguous block layout stores K+V packed, so reading K (first half)
    //    and V (second half) back-to-back should reconstruct the original block.
    let received = unsafe { std::slice::from_raw_parts(recv_buf as *const u8, recv_offset) };
    assert_eq!(
        received,
        &host_data[..recv_offset],
        "RDMA-transferred data does not match original"
    );
    eprintln!("[e2e] data integrity verified");

    // 8. gRPC: ReleaseLease
    client
        .release_lease(ReleaseLeaseRequest {
            lease_id: lease_id.clone(),
            requester_node_id: "requester-node-1".to_string(),
        })
        .await
        .expect("release_lease RPC");

    eprintln!("[e2e] lease released: {lease_id}");

    // Cleanup
    requester_rdma
        .unregister_memory(recv_buf as u64)
        .expect("unregister recv buffer");
    unsafe {
        libc::munmap(recv_buf, recv_buf_size);
    }
    for (ptr, size) in engine.pinned_pool_regions() {
        owner_rdma.unregister_memory(ptr).unwrap_or_else(|e| {
            eprintln!("warn: unregister pool region {ptr:#x} ({size}): {e}");
        });
    }
    server_handle.abort();

    eprintln!("[e2e] PASSED");
}
