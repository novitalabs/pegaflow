//! Async worker for cross-node remote block fetching.
//!
//! Orchestrates: MetaServer query → remote gRPC QueryBlocksForTransfer → RDMA READ
//! → build SealedBlock → send result via oneshot channel.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Instant;

use log::{debug, info, warn};
use tokio::sync::oneshot;

use crate::block::{BlockKey, LayerBlock, SealedBlock};
use crate::internode::client::{PegaflowClient, PegaflowClientPool};
use crate::internode::metaserver_query::MetaServerQueryClient;
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocator;
use crate::storage::remote_fetch::RemoteFetchResult;
use pegaflow_common::NumaNode;

/// Abstraction for RDMA batch read operations. Injected by the server layer
/// to decouple pegaflow-core from pegaflow-transfer.
///
/// Arguments: (remote_session_id_bytes, local_ptrs, remote_ptrs, sizes)
/// Returns: total bytes transferred, or error message.
pub(crate) type RdmaBatchReadFn = Arc<
    dyn Fn(
            &[u8],    // remote_session_id (26-byte DomainAddress)
            &[u64],   // local_ptrs
            &[u64],   // remote_ptrs
            &[usize], // sizes
        ) -> Result<usize, String>
        + Send
        + Sync,
>;

/// Abstraction for registering local memory with RDMA.
pub(crate) type RdmaRegisterMemoryFn = Arc<dyn Fn(u64, usize) -> Result<(), String> + Send + Sync>;

/// Configuration for the remote fetch worker.
pub(crate) struct RemoteFetchWorkerConfig {
    pub metaserver_client: Arc<MetaServerQueryClient>,
    pub client_pool: Arc<PegaflowClientPool>,
    pub allocator: Arc<PinnedAllocator>,
    pub rdma_read_fn: RdmaBatchReadFn,
    pub rdma_register_fn: RdmaRegisterMemoryFn,
}

/// Execute a remote fetch for the given missing blocks.
///
/// This is spawned as a tokio task by the `RemoteFetchFn` closure.
pub(crate) async fn execute_remote_fetch(
    config: Arc<RemoteFetchWorkerConfig>,
    missing_keys: Vec<BlockKey>,
    done_tx: oneshot::Sender<RemoteFetchResult>,
) {
    let started_at = Instant::now();
    let result = execute_remote_fetch_inner(&config, &missing_keys).await;

    match &result {
        Ok(blocks) => {
            let elapsed = started_at.elapsed();
            info!(
                "Remote fetch completed: requested={} fetched={} elapsed={:?}",
                missing_keys.len(),
                blocks.len(),
                elapsed
            );
            core_metrics()
                .remote_fetch_latency_seconds
                .record(elapsed.as_secs_f64(), &[]);
        }
        Err(e) => {
            warn!(
                "Remote fetch failed: requested={} error={}",
                missing_keys.len(),
                e
            );
        }
    }

    let _ = done_tx.send(result.unwrap_or_default());
}

async fn execute_remote_fetch_inner(
    config: &RemoteFetchWorkerConfig,
    missing_keys: &[BlockKey],
) -> Result<RemoteFetchResult, String> {
    if missing_keys.is_empty() {
        return Ok(Vec::new());
    }

    let namespace = &missing_keys[0].namespace;
    let hashes: Vec<Vec<u8>> = missing_keys.iter().map(|k| k.hash.clone()).collect();

    // Step 1: Query MetaServer for block locations
    let meta_start = Instant::now();
    let meta_resp = config
        .metaserver_client
        .query_block_hashes(namespace, &hashes)
        .await
        .map_err(|e| {
            core_metrics().metaserver_query_failures.add(1, &[]);
            format!("MetaServer query failed: {e}")
        })?;

    let meta_elapsed = meta_start.elapsed();
    core_metrics()
        .metaserver_query_latency_seconds
        .record(meta_elapsed.as_secs_f64(), &[]);

    debug!(
        "MetaServer query: queried={} found={} nodes={} elapsed={:?}",
        hashes.len(),
        meta_resp.found_count,
        meta_resp.node_blocks.len(),
        meta_elapsed
    );

    if meta_resp.node_blocks.is_empty() {
        let missed = hashes.len();
        core_metrics()
            .remote_fetch_blocks_missed
            .add(missed as u64, &[]);
        return Ok(Vec::new());
    }

    let mut fetched_blocks: Vec<(BlockKey, Arc<SealedBlock>)> = Vec::new();

    // Step 2: For each remote node with matching blocks
    for node_blocks in &meta_resp.node_blocks {
        let endpoint = format!("http://{}", node_blocks.node);

        let client: PegaflowClient = match config.client_pool.get_or_connect(&endpoint).await {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to connect to remote node {endpoint}: {e}");
                core_metrics()
                    .remote_fetch_blocks_missed
                    .add(node_blocks.block_hashes.len() as u64, &[]);
                continue;
            }
        };

        // Step 3: Query remote node for RDMA metadata + lock blocks
        let transfer_resp = match client
            .query_blocks_for_transfer(namespace, &node_blocks.block_hashes, "remote-fetch")
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("QueryBlocksForTransfer failed for {endpoint}: {e}");
                core_metrics()
                    .remote_fetch_blocks_missed
                    .add(node_blocks.block_hashes.len() as u64, &[]);
                continue;
            }
        };

        if transfer_resp.rdma_session_id.is_empty() {
            warn!("Remote node {endpoint} returned empty RDMA session ID");
            core_metrics()
                .remote_fetch_blocks_missed
                .add(node_blocks.block_hashes.len() as u64, &[]);
            // Release lock since we can't do RDMA
            let _ = client
                .release_transfer_lock(&transfer_resp.transfer_session_id)
                .await;
            continue;
        }

        // Step 4: RDMA fetch for each block
        for block_info in &transfer_resp.blocks {
            match fetch_single_block(
                config,
                block_info,
                &transfer_resp.rdma_session_id,
                namespace,
            ) {
                Ok(result) => fetched_blocks.push(result),
                Err(e) => {
                    warn!("Failed to fetch block via RDMA: {e}");
                    core_metrics().remote_fetch_blocks_missed.add(1, &[]);
                }
            }
        }

        // Step 5: Release remote locks
        if let Err(e) = client
            .release_transfer_lock(&transfer_resp.transfer_session_id)
            .await
        {
            warn!("Failed to release transfer lock on {endpoint}: {e} (will auto-expire)");
        }
    }

    Ok(fetched_blocks)
}

fn fetch_single_block(
    config: &RemoteFetchWorkerConfig,
    block_info: &pegaflow_proto::proto::engine::TransferBlockInfo,
    rdma_session_id: &[u8],
    namespace: &str,
) -> Result<(BlockKey, Arc<SealedBlock>), String> {
    let mut local_ptrs = Vec::new();
    let mut remote_ptrs = Vec::new();
    let mut sizes = Vec::new();
    let mut allocations: Vec<(
        Arc<crate::pinned_pool::PinnedAllocation>,
        Option<Arc<crate::pinned_pool::PinnedAllocation>>,
    )> = Vec::new();

    // Allocate local pinned memory and prepare RDMA scatter list
    for slot_info in &block_info.slots {
        // Allocate for K segment
        let k_size = NonZeroU64::new(slot_info.k_size).ok_or("zero k_size")?;
        let k_alloc = config
            .allocator
            .allocate(k_size, NumaNode::UNKNOWN)
            .ok_or("pinned memory exhausted for K segment")?;
        let k_local_ptr = k_alloc.as_ptr() as u64;

        // Register local memory with RDMA engine
        (config.rdma_register_fn)(k_local_ptr, slot_info.k_size as usize)
            .map_err(|e| format!("RDMA register K failed: {e}"))?;

        local_ptrs.push(k_local_ptr);
        remote_ptrs.push(slot_info.k_ptr);
        sizes.push(slot_info.k_size as usize);

        if slot_info.v_ptr != 0 {
            // Split storage: allocate for V segment
            let v_size = NonZeroU64::new(slot_info.v_size).ok_or("zero v_size")?;
            let v_alloc = config
                .allocator
                .allocate(v_size, NumaNode::UNKNOWN)
                .ok_or("pinned memory exhausted for V segment")?;
            let v_local_ptr = v_alloc.as_ptr() as u64;

            (config.rdma_register_fn)(v_local_ptr, slot_info.v_size as usize)
                .map_err(|e| format!("RDMA register V failed: {e}"))?;

            local_ptrs.push(v_local_ptr);
            remote_ptrs.push(slot_info.v_ptr);
            sizes.push(slot_info.v_size as usize);

            allocations.push((k_alloc, Some(v_alloc)));
        } else {
            allocations.push((k_alloc, None));
        }
    }

    // RDMA READ
    let bytes_transferred =
        (config.rdma_read_fn)(rdma_session_id, &local_ptrs, &remote_ptrs, &sizes)
            .map_err(|e| format!("RDMA READ failed: {e}"))?;

    core_metrics()
        .remote_fetch_rdma_bytes
        .add(bytes_transferred as u64, &[]);

    // Build SealedBlock from received data
    let mut layer_blocks = Vec::with_capacity(block_info.slots.len());
    for (i, slot_info) in block_info.slots.iter().enumerate() {
        let (ref k_alloc, ref v_alloc) = allocations[i];
        let layer_block = if let Some(v_alloc) = v_alloc {
            LayerBlock::new_split(
                k_alloc.as_ptr().cast_mut(),
                v_alloc.as_ptr().cast_mut(),
                slot_info.k_size as usize,
                Arc::clone(k_alloc),
                Arc::clone(v_alloc),
            )
        } else {
            LayerBlock::new_contiguous(
                k_alloc.as_ptr().cast_mut(),
                slot_info.k_size as usize,
                Arc::clone(k_alloc),
            )
        };
        layer_blocks.push(Arc::new(layer_block));
    }

    let sealed = SealedBlock::from_slots(layer_blocks);
    let key = BlockKey::new(namespace.to_string(), block_info.block_hash.clone());

    Ok((key, Arc::new(sealed)))
}

#[cfg(test)]
impl RemoteFetchWorkerConfig {
    fn new_for_test(
        allocator: Arc<PinnedAllocator>,
        rdma_read_fn: RdmaBatchReadFn,
        rdma_register_fn: RdmaRegisterMemoryFn,
    ) -> Arc<Self> {
        use tonic::transport::Endpoint;
        let channel = Endpoint::from_static("http://[::]:1").connect_lazy();
        Arc::new(Self {
            metaserver_client: Arc::new(MetaServerQueryClient::from_channel(channel.clone())),
            client_pool: Arc::new(PegaflowClientPool::new_for_test()),
            allocator,
            rdma_read_fn,
            rdma_register_fn,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use pegaflow_proto::proto::engine::{TransferBlockInfo, TransferSlotInfo};

    use super::*;

    // ---- Mock helpers ----

    type RdmaCallLog = Arc<Mutex<Vec<(Vec<u64>, Vec<u64>, Vec<usize>)>>>;

    fn noop_rdma_register_fn() -> RdmaRegisterMemoryFn {
        Arc::new(|_ptr, _size| Ok(()))
    }

    fn recording_rdma_read_fn() -> (RdmaBatchReadFn, RdmaCallLog) {
        let calls: RdmaCallLog = Arc::new(Mutex::new(Vec::new()));
        let calls_clone = calls.clone();
        let f: RdmaBatchReadFn = Arc::new(move |_session_id, local_ptrs, remote_ptrs, sizes| {
            calls_clone.lock().unwrap().push((
                local_ptrs.to_vec(),
                remote_ptrs.to_vec(),
                sizes.to_vec(),
            ));
            Ok(sizes.iter().sum())
        });
        (f, calls)
    }

    fn failing_rdma_read_fn(msg: &str) -> RdmaBatchReadFn {
        let msg = msg.to_string();
        Arc::new(move |_, _, _, _| Err(msg.clone()))
    }

    fn failing_rdma_register_fn(msg: &str) -> RdmaRegisterMemoryFn {
        let msg = msg.to_string();
        Arc::new(move |_, _| Err(msg.clone()))
    }

    fn has_cuda() -> bool {
        cudarc::driver::CudaContext::new(0).is_ok()
    }

    fn make_allocator(size: usize) -> Arc<PinnedAllocator> {
        Arc::new(PinnedAllocator::new_global(size, false, false, None))
    }

    // ---- fetch_single_block tests ----

    #[tokio::test]
    async fn fetch_single_block_contiguous() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, calls) = recording_rdma_read_fn();
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn, noop_rdma_register_fn());

        let block_info = TransferBlockInfo {
            block_hash: vec![1, 2, 3],
            slots: vec![TransferSlotInfo {
                k_ptr: 0x1000,
                k_size: 512,
                v_ptr: 0,
                v_size: 0,
            }],
            rkey: 0,
        };

        let (key, sealed) = fetch_single_block(&config, &block_info, &[0u8; 26], "ns").unwrap();

        assert_eq!(key.namespace, "ns");
        assert_eq!(key.hash, vec![1, 2, 3]);
        assert_eq!(sealed.slots().len(), 1);
        assert!(sealed.slots()[0].v_ptr().is_none());

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2.len(), 1); // 1 size entry (K only)
        assert_eq!(calls[0].2[0], 512);
    }

    #[tokio::test]
    async fn fetch_single_block_split_storage() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, calls) = recording_rdma_read_fn();
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn, noop_rdma_register_fn());

        let block_info = TransferBlockInfo {
            block_hash: vec![4, 5, 6],
            slots: vec![TransferSlotInfo {
                k_ptr: 0x1000,
                k_size: 512,
                v_ptr: 0x2000,
                v_size: 512,
            }],
            rkey: 0,
        };

        let (key, sealed) = fetch_single_block(&config, &block_info, &[0u8; 26], "ns").unwrap();

        assert_eq!(key.hash, vec![4, 5, 6]);
        assert_eq!(sealed.slots().len(), 1);
        assert!(sealed.slots()[0].v_ptr().is_some());

        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2.len(), 2); // K + V
        assert_eq!(calls[0].2[0], 512);
        assert_eq!(calls[0].2[1], 512);
    }

    #[tokio::test]
    async fn fetch_single_block_multi_slot() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, _) = recording_rdma_read_fn();
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn, noop_rdma_register_fn());

        let block_info = TransferBlockInfo {
            block_hash: vec![7],
            slots: vec![
                TransferSlotInfo {
                    k_ptr: 0x1000,
                    k_size: 256,
                    v_ptr: 0,
                    v_size: 0,
                },
                TransferSlotInfo {
                    k_ptr: 0x2000,
                    k_size: 256,
                    v_ptr: 0,
                    v_size: 0,
                },
                TransferSlotInfo {
                    k_ptr: 0x3000,
                    k_size: 256,
                    v_ptr: 0,
                    v_size: 0,
                },
            ],
            rkey: 0,
        };

        let (_key, sealed) = fetch_single_block(&config, &block_info, &[0u8; 26], "ns").unwrap();

        assert_eq!(sealed.slots().len(), 3);
    }

    #[tokio::test]
    async fn fetch_single_block_zero_k_size() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, _) = recording_rdma_read_fn();
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn, noop_rdma_register_fn());

        let block_info = TransferBlockInfo {
            block_hash: vec![8],
            slots: vec![TransferSlotInfo {
                k_ptr: 0x1000,
                k_size: 0,
                v_ptr: 0,
                v_size: 0,
            }],
            rkey: 0,
        };

        let err = fetch_single_block(&config, &block_info, &[0u8; 26], "ns")
            .err()
            .expect("expected error");
        assert!(err.contains("zero k_size"));
    }

    #[tokio::test]
    async fn fetch_single_block_rdma_register_failure() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, _) = recording_rdma_read_fn();
        let config = RemoteFetchWorkerConfig::new_for_test(
            allocator,
            rdma_read_fn,
            failing_rdma_register_fn("register boom"),
        );

        let block_info = TransferBlockInfo {
            block_hash: vec![9],
            slots: vec![TransferSlotInfo {
                k_ptr: 0x1000,
                k_size: 512,
                v_ptr: 0,
                v_size: 0,
            }],
            rkey: 0,
        };

        let err = fetch_single_block(&config, &block_info, &[0u8; 26], "ns")
            .err()
            .expect("expected error");
        assert!(err.contains("register boom"));
    }

    #[tokio::test]
    async fn fetch_single_block_rdma_read_failure() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let config = RemoteFetchWorkerConfig::new_for_test(
            allocator,
            failing_rdma_read_fn("network error"),
            noop_rdma_register_fn(),
        );

        let block_info = TransferBlockInfo {
            block_hash: vec![10],
            slots: vec![TransferSlotInfo {
                k_ptr: 0x1000,
                k_size: 512,
                v_ptr: 0,
                v_size: 0,
            }],
            rkey: 0,
        };

        let err = fetch_single_block(&config, &block_info, &[0u8; 26], "ns")
            .err()
            .expect("expected error");
        assert!(err.contains("RDMA READ failed"));
    }

    #[tokio::test]
    async fn fetch_single_block_allocator_exhausted() {
        if !has_cuda() {
            return;
        }
        // 512-byte allocator can't fit a 1024-byte K segment
        let allocator = make_allocator(512);
        let (rdma_read_fn, _) = recording_rdma_read_fn();
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn, noop_rdma_register_fn());

        let block_info = TransferBlockInfo {
            block_hash: vec![11],
            slots: vec![TransferSlotInfo {
                k_ptr: 0x1000,
                k_size: 1024,
                v_ptr: 0,
                v_size: 0,
            }],
            rkey: 0,
        };

        let err = fetch_single_block(&config, &block_info, &[0u8; 26], "ns")
            .err()
            .expect("expected error");
        assert!(err.contains("pinned memory exhausted"));
    }

    // ---- Integration tests ----

    #[tokio::test]
    async fn full_pipeline_metaserver_to_rdma() {
        if !has_cuda() {
            return;
        }

        use crate::internode::client::test_utils::{MockEngine, start_mock_engine};
        use crate::internode::metaserver_query::test_utils::{
            MockMetaServer, start_mock_metaserver,
        };
        use pegaflow_proto::proto::engine::{QueryBlocksForTransferResponse, ResponseStatus};

        let allocator = make_allocator(1 << 20);

        // Start mock Engine with transfer data
        let engine_mock = MockEngine {
            transfer_response: QueryBlocksForTransferResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                blocks: vec![TransferBlockInfo {
                    block_hash: vec![1, 2, 3],
                    slots: vec![TransferSlotInfo {
                        k_ptr: 0x1000,
                        k_size: 512,
                        v_ptr: 0,
                        v_size: 0,
                    }],
                    rkey: 0,
                }],
                transfer_session_id: "test-session".to_string(),
                rdma_session_id: vec![0u8; 26],
            },
        };
        let (engine_addr, _) = start_mock_engine(engine_mock).await;
        let engine_endpoint = format!("127.0.0.1:{}", engine_addr.port());

        // Start mock MetaServer with hash mapped to engine node
        let meta_mock = MockMetaServer::new();
        meta_mock.insert("ns", vec![1, 2, 3], &engine_endpoint);
        let meta_addr = start_mock_metaserver(meta_mock).await;

        // Build config with real clients
        let metaserver_client = Arc::new(
            MetaServerQueryClient::connect(&format!("127.0.0.1:{}", meta_addr.port()))
                .await
                .unwrap(),
        );
        let (rdma_read_fn, rdma_calls) = recording_rdma_read_fn();
        let config = Arc::new(RemoteFetchWorkerConfig {
            metaserver_client,
            client_pool: Arc::new(PegaflowClientPool::new_for_test()),
            allocator,
            rdma_read_fn,
            rdma_register_fn: noop_rdma_register_fn(),
        });

        let missing_keys = vec![BlockKey::new("ns".to_string(), vec![1, 2, 3])];
        let result = execute_remote_fetch_inner(&config, &missing_keys).await;

        let blocks = result.unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0.hash, vec![1, 2, 3]);
        assert_eq!(blocks[0].1.slots().len(), 1);

        // Verify RDMA was called
        let calls = rdma_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2[0], 512);
    }

    #[tokio::test]
    async fn empty_keys_returns_empty() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, _) = recording_rdma_read_fn();
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn, noop_rdma_register_fn());

        let result = execute_remote_fetch_inner(&config, &[]).await;
        assert!(result.unwrap().is_empty());
    }
}
