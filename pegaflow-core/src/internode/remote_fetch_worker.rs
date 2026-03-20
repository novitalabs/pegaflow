//! Async worker for cross-node remote block fetching.
//!
//! Orchestrates: MetaServer query → remote gRPC QueryBlocksForTransfer → RDMA READ
//! → build SealedBlock → send result via oneshot channel.

use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use log::{debug, info, warn};
use tokio::sync::oneshot;

use crate::PegaEngine;
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::internode::client::PegaflowClientPool;
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
pub type RdmaBatchReadFn = Arc<
    dyn Fn(
            &[u8],    // remote_session_id (26-byte DomainAddress)
            &[u64],   // local_ptrs
            &[u64],   // remote_ptrs
            &[usize], // sizes
        ) -> Result<usize, String>
        + Send
        + Sync,
>;

/// Configuration for the remote fetch worker.
pub struct RemoteFetchWorkerConfig {
    metaserver_client: Arc<MetaServerQueryClient>,
    client_pool: Arc<PegaflowClientPool>,
    allocator: Arc<PinnedAllocator>,
    rdma_read_fn: RdmaBatchReadFn,
}

impl RemoteFetchWorkerConfig {
    pub fn new_from_engine(
        metaserver_client: Arc<MetaServerQueryClient>,
        client_pool: Arc<PegaflowClientPool>,
        engine: Arc<PegaEngine>,
        rdma_read_fn: RdmaBatchReadFn,
    ) -> Self {
        Self {
            metaserver_client,
            client_pool,
            allocator: engine.pinned_allocator(),
            rdma_read_fn,
        }
    }
}

/// Execute a remote fetch for the given missing blocks.
///
/// This is spawned as a tokio task by the `RemoteFetchFn` closure.
pub async fn execute_remote_fetch(
    config: Arc<RemoteFetchWorkerConfig>,
    missing_keys: Vec<BlockKey>,
    done_tx: oneshot::Sender<RemoteFetchResult>,
) {
    let started_at = Instant::now();
    let result = execute_remote_fetch_inner(Arc::clone(&config), &missing_keys).await;

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
    config: Arc<RemoteFetchWorkerConfig>,
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

    for node_blocks in &meta_resp.node_blocks {
        let endpoint = format!("http://{}", node_blocks.node);

        let client = match config.client_pool.get_or_connect(&endpoint).await {
            Ok(c) => c,
            Err(e) => {
                warn!("Skipping unreachable node {endpoint}: {e}");
                core_metrics()
                    .remote_fetch_blocks_missed
                    .add(node_blocks.block_hashes.len() as u64, &[]);
                continue;
            }
        };

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
            let _ = client
                .release_transfer_lock(&transfer_resp.transfer_session_id)
                .await;
            continue;
        }

        // Batch all blocks from this node into a single spawn_blocking call
        // to avoid per-block thread-pool round-trips and redundant clones.
        let blocks_for_node = transfer_resp.blocks.clone();
        let rdma_session_id = transfer_resp.rdma_session_id.clone();
        let ns = namespace.to_string();
        let config_clone = Arc::clone(&config);
        match tokio::task::spawn_blocking(move || {
            blocks_for_node
                .iter()
                .filter_map(|block_info| {
                    match fetch_single_block(&config_clone, block_info, &rdma_session_id, &ns) {
                        Ok(result) => Some(result),
                        Err(e) => {
                            warn!("Failed to fetch block via RDMA: {e}");
                            core_metrics().remote_fetch_blocks_missed.add(1, &[]);
                            None
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .await
        {
            Ok(results) => fetched_blocks.extend(results),
            Err(e) => {
                warn!("RDMA fetch task panicked: {e}");
                core_metrics()
                    .remote_fetch_blocks_missed
                    .add(transfer_resp.blocks.len() as u64, &[]);
            }
        }

        if let Err(e) = client
            .release_transfer_lock(&transfer_resp.transfer_session_id)
            .await
        {
            warn!("Failed to release transfer lock on {endpoint}: {e} (will auto-expire)");
        }
    }

    Ok(fetched_blocks)
}

/// Max size for a single K or V segment received from a remote node (256 MiB).
/// Rejects absurdly large values that could exhaust the pinned memory pool.
const MAX_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

fn validate_slot_info(
    slot_info: &pegaflow_proto::proto::engine::TransferSlotInfo,
) -> Result<(), String> {
    if slot_info.k_ptr == 0 {
        return Err("remote k_ptr is null".into());
    }
    if slot_info.k_size == 0 {
        return Err("zero k_size".into());
    }
    if slot_info.k_size > MAX_SEGMENT_SIZE {
        return Err(format!(
            "k_size {} exceeds max {}",
            slot_info.k_size, MAX_SEGMENT_SIZE
        ));
    }
    if slot_info.v_ptr != 0 {
        if slot_info.v_size == 0 {
            return Err("zero v_size with non-zero v_ptr".into());
        }
        if slot_info.v_size > MAX_SEGMENT_SIZE {
            return Err(format!(
                "v_size {} exceeds max {}",
                slot_info.v_size, MAX_SEGMENT_SIZE
            ));
        }
    }
    Ok(())
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
        // Validate remote-supplied slot metadata
        validate_slot_info(slot_info)?;

        // Allocate for K segment
        let k_size = NonZeroU64::new(slot_info.k_size).ok_or("zero k_size")?;
        let k_alloc = config
            .allocator
            .allocate(k_size, NumaNode::UNKNOWN)
            .ok_or("pinned memory exhausted for K segment")?;
        let k_local_ptr = k_alloc.as_ptr() as u64;

        // Note: per-block RDMA registration is skipped because the entire pinned
        // memory pool is registered at startup (see pegaflow-server/src/lib.rs).

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
    let mut raw_blocks = Vec::with_capacity(block_info.slots.len());
    for (i, slot_info) in block_info.slots.iter().enumerate() {
        let (ref k_alloc, ref v_alloc) = allocations[i];
        let mut segments = vec![Segment::new(
            NonNull::new(k_alloc.as_ptr().cast_mut()).unwrap(),
            slot_info.k_size as usize,
            Arc::clone(k_alloc),
        )];
        if let Some(v_alloc) = v_alloc {
            segments.push(Segment::new(
                NonNull::new(v_alloc.as_ptr().cast_mut()).unwrap(),
                slot_info.v_size as usize,
                Arc::clone(v_alloc),
            ));
        }
        raw_blocks.push(Arc::new(RawBlock::new(segments)));
    }

    let sealed = SealedBlock::from_slots(raw_blocks);
    let key = BlockKey::new(namespace.to_string(), block_info.block_hash.clone());

    Ok((key, Arc::new(sealed)))
}

#[cfg(test)]
impl RemoteFetchWorkerConfig {
    fn new_for_test(allocator: Arc<PinnedAllocator>, rdma_read_fn: RdmaBatchReadFn) -> Arc<Self> {
        use tonic::transport::Endpoint;
        let channel = Endpoint::from_static("http://[::]:1").connect_lazy();
        Arc::new(Self {
            metaserver_client: Arc::new(MetaServerQueryClient::from_channel(channel.clone())),
            client_pool: Arc::new(PegaflowClientPool::new_for_test()),
            allocator,
            rdma_read_fn,
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
        let config = RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn);

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
        assert_eq!(sealed.slots()[0].num_segments(), 1);

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
        let config = RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn);

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
        assert!(sealed.slots()[0].num_segments() >= 2);
        assert_eq!(sealed.slots()[0].memory_footprint(), 1024);

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
        let config = RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn);

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
        let config = RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn);

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

    #[test]
    fn validate_slot_info_rejects_null_k_ptr() {
        let slot = TransferSlotInfo {
            k_ptr: 0,
            k_size: 512,
            v_ptr: 0,
            v_size: 0,
        };
        assert!(
            validate_slot_info(&slot)
                .unwrap_err()
                .contains("k_ptr is null")
        );
    }

    #[test]
    fn validate_slot_info_rejects_oversized_k() {
        let slot = TransferSlotInfo {
            k_ptr: 0x1000,
            k_size: MAX_SEGMENT_SIZE + 1,
            v_ptr: 0,
            v_size: 0,
        };
        assert!(
            validate_slot_info(&slot)
                .unwrap_err()
                .contains("exceeds max")
        );
    }

    #[test]
    fn validate_slot_info_rejects_oversized_v() {
        let slot = TransferSlotInfo {
            k_ptr: 0x1000,
            k_size: 512,
            v_ptr: 0x2000,
            v_size: MAX_SEGMENT_SIZE + 1,
        };
        assert!(
            validate_slot_info(&slot)
                .unwrap_err()
                .contains("exceeds max")
        );
    }

    #[test]
    fn validate_slot_info_accepts_valid_slot() {
        let slot = TransferSlotInfo {
            k_ptr: 0x1000,
            k_size: 512,
            v_ptr: 0x2000,
            v_size: 512,
        };
        assert!(validate_slot_info(&slot).is_ok());
    }

    #[tokio::test]
    async fn fetch_single_block_rdma_read_failure() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let config =
            RemoteFetchWorkerConfig::new_for_test(allocator, failing_rdma_read_fn("network error"));

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
        let config = RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn);

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
        });

        let missing_keys = vec![BlockKey::new("ns".to_string(), vec![1, 2, 3])];
        let result = execute_remote_fetch_inner(config, &missing_keys).await;

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
    async fn full_pipeline_fetches_blocks_from_multiple_nodes() {
        if !has_cuda() {
            return;
        }

        use crate::internode::client::test_utils::{MockEngine, start_mock_engine};
        use crate::internode::metaserver_query::test_utils::{
            MockMetaServer, start_mock_metaserver,
        };
        use pegaflow_proto::proto::engine::{QueryBlocksForTransferResponse, ResponseStatus};

        let allocator = make_allocator(1 << 20);

        let engine_a = MockEngine {
            transfer_response: QueryBlocksForTransferResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                blocks: vec![TransferBlockInfo {
                    block_hash: vec![1],
                    slots: vec![TransferSlotInfo {
                        k_ptr: 0x1000,
                        k_size: 512,
                        v_ptr: 0,
                        v_size: 0,
                    }],
                    rkey: 0,
                }],
                transfer_session_id: "session-a".to_string(),
                rdma_session_id: vec![1u8; 26],
            },
        };
        let (engine_a_addr, _) = start_mock_engine(engine_a).await;

        let engine_b = MockEngine {
            transfer_response: QueryBlocksForTransferResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                blocks: vec![TransferBlockInfo {
                    block_hash: vec![2],
                    slots: vec![TransferSlotInfo {
                        k_ptr: 0x2000,
                        k_size: 512,
                        v_ptr: 0,
                        v_size: 0,
                    }],
                    rkey: 0,
                }],
                transfer_session_id: "session-b".to_string(),
                rdma_session_id: vec![2u8; 26],
            },
        };
        let (engine_b_addr, _) = start_mock_engine(engine_b).await;

        let meta_mock = MockMetaServer::new();
        meta_mock.insert(
            "ns",
            vec![1],
            &format!("127.0.0.1:{}", engine_a_addr.port()),
        );
        meta_mock.insert(
            "ns",
            vec![2],
            &format!("127.0.0.1:{}", engine_b_addr.port()),
        );
        let meta_addr = start_mock_metaserver(meta_mock).await;

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
        });

        let missing_keys = vec![
            BlockKey::new("ns".to_string(), vec![1]),
            BlockKey::new("ns".to_string(), vec![2]),
        ];
        let result = execute_remote_fetch_inner(config, &missing_keys)
            .await
            .unwrap();

        assert_eq!(result.len(), 2);

        let calls = rdma_calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[tokio::test]
    async fn empty_keys_returns_empty() {
        if !has_cuda() {
            return;
        }
        let allocator = make_allocator(1 << 20);
        let (rdma_read_fn, _) = recording_rdma_read_fn();
        let config = RemoteFetchWorkerConfig::new_for_test(allocator, rdma_read_fn);

        let result = execute_remote_fetch_inner(config, &[]).await;
        assert!(result.unwrap().is_empty());
    }
}
