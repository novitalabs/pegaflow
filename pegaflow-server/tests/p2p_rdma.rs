//! P2P RDMA integration tests: production cross-node fetch + transfer-plan NUMA.
//!
//! ```bash
//! cargo test -p pegaflow-server --test p2p_rdma -- --ignored --nocapture
//! ```
//!
//! The cases here enumerate how a stored block's slots map onto NUMA nodes,
//! because that mapping is what the transfer plan must reproduce and the
//! requester must rebuild:
//!
//! - **A (single writer)** — `SingleLayer` / `MultiLayerMultiBlock`: one worker,
//!   every slot on one NUMA. Baseline encode/fetch/load.
//! - **B (cross-slot NUMA)** — normal MHA TP>1: each rank owns a distinct slot
//!   column and saves with its own GPU's NUMA, so a *single block* spans NUMA
//!   when ranks land on different sockets. No server NUMA hint (eff tp_size>1).
//! - **C (cross-block NUMA)** — MLA replica without DCP: only effective rank 0
//!   saves, but the server round-robins each save RPC's NUMA across the replica
//!   candidates, so each block is single-NUMA while the *batch* spans NUMA.
//!
//! Multi-NUMA assertions self-adapt: on a single-NUMA box the distinct-NUMA set
//! collapses to one and the cross-NUMA shape is logged as not exercised, but the
//! multi-worker topology and the data roundtrip are still verified.

// The p2p harness is standalone: include it directly so this binary does not
// drag in the unrelated mock-vLLM helpers under `common/` (and vice versa).
#[path = "common/p2p_harness.rs"]
mod p2p_harness;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::CudaContext;
use pegaflow_common::NumaNode;
use pegaflow_core::{PegaEngine, StorageConfig};
use pegaflow_server::session::SessionRegistry;

use p2p_harness::{
    DeviceGpu, GpuBuffer, P2P_NAMESPACE, P2pCase, P2pShape, any_block_spans_multiple_numa,
    assert_transfer_plan_numa, cuda_device_count, distinct_slot_numas, fill_test_pattern_salted,
    get_free_port, holder_save_blocks, holder_slot_numas_per_block, make_block_hashes,
    query_holder_transfer_plan, register_worker, requester_load_and_verify,
    requester_load_worker_and_verify, save_worker_blocks, spawn_engine_server, spawn_metaserver,
    unique_chunk_numa_nodes, wait_for_cache, wait_for_metaserver_registration,
    wait_for_prefetch_done,
};

/// Cap multi-GPU fan-out: enough to span both sockets on a 2-NUMA box without
/// registering an absurd number of CUDA contexts.
const MAX_TEST_GPUS: usize = 8;

fn case_salt(case: P2pCase) -> u8 {
    match case {
        P2pCase::SingleLayer => 42,
        P2pCase::MultiLayerMultiBlock => 43,
    }
}

struct P2pCluster {
    meta_port: u16,
    port_holder: u16,
    port_requester: u16,
}

impl P2pCluster {
    fn new() -> Self {
        Self {
            meta_port: get_free_port(),
            port_holder: get_free_port(),
            port_requester: get_free_port(),
        }
    }

    fn holder_config(&self) -> StorageConfig {
        StorageConfig {
            metaserver_addr: Some(format!("http://127.0.0.1:{}", self.meta_port)),
            advertise_addr: Some(format!("127.0.0.1:{}", self.port_holder)),
            rdma_nic_names: Some(p2p_harness::rdma_nic_names()),
            ..StorageConfig::default()
        }
    }

    fn requester_config(&self) -> StorageConfig {
        StorageConfig {
            metaserver_addr: Some(format!("http://127.0.0.1:{}", self.meta_port)),
            advertise_addr: Some(format!("127.0.0.1:{}", self.port_requester)),
            rdma_nic_names: Some(p2p_harness::rdma_nic_names()),
            ..StorageConfig::default()
        }
    }
}

// =============================================================================
// Class A: single writer, single NUMA
// =============================================================================

async fn run_roundtrip(case: P2pCase) {
    pegaflow_common::logging::init_stdout_colored("debug");
    let shape = case.default_shape();
    let _ = CudaContext::new(0).expect("CUDA init");

    let cluster = P2pCluster::new();
    let meta_store = spawn_metaserver(cluster.meta_port).await;

    let engine_a = Arc::new(
        PegaEngine::new_with_config(256 << 20, false, cluster.holder_config())
            .expect("holder engine"),
    );
    spawn_engine_server(Arc::clone(&engine_a), cluster.port_holder).await;

    let block_hashes = make_block_hashes(shape.blocks, case_salt(case));
    let gpu_holder = GpuBuffer::alloc(shape.rank_bytes());
    let holder_host = holder_save_blocks(
        &engine_a,
        "inst-a",
        &shape,
        0,
        0,
        &block_hashes,
        &gpu_holder,
    )
    .await;

    wait_for_cache(
        &engine_a,
        "inst-a",
        &block_hashes,
        shape.blocks,
        Duration::from_secs(60),
    )
    .await;
    wait_for_metaserver_registration(
        &meta_store,
        P2P_NAMESPACE,
        &block_hashes,
        shape.blocks,
        Duration::from_secs(30),
    )
    .await;

    let plan = query_holder_transfer_plan(&engine_a, &block_hashes, "p2p-it-requester").await;
    assert_transfer_plan_numa(&plan);
    let (_, found) =
        engine_a.query_blocks_for_transfer(P2P_NAMESPACE, &block_hashes, "p2p-it-check");
    let holder_numas = holder_slot_numas_per_block(&found);
    assert_eq!(holder_numas.len(), shape.blocks);
    for slot_numas in &holder_numas {
        assert!(!slot_numas.is_empty());
        assert!(slot_numas.iter().all(|n| n.is_valid()));
    }
    let distinct = unique_chunk_numa_nodes(&plan);
    eprintln!(
        "{case:?}: plan blocks={} chunks={} distinct_chunk_numa={distinct:?}",
        plan.block_hashes.len(),
        plan.remote_chunks.len()
    );

    let engine_b = Arc::new(
        PegaEngine::new_with_config(256 << 20, false, cluster.requester_config())
            .expect("requester engine"),
    );

    let gpu_req = GpuBuffer::alloc(shape.rank_bytes());
    gpu_req.zero();
    register_worker(
        &engine_b,
        "inst-b",
        &shape,
        0,
        0,
        0,
        shape.tp_size,
        shape.world_size,
        gpu_req.as_u64(),
    );

    let lease = wait_for_prefetch_done(
        &engine_b,
        "inst-b",
        &format!("p2p-{case:?}"),
        &block_hashes,
        shape.blocks,
        Duration::from_secs(120),
    )
    .await;

    let block_ids: Vec<usize> = (0..shape.blocks).collect();
    requester_load_and_verify(
        &engine_b,
        "inst-b",
        &shape,
        0,
        0,
        lease,
        &block_ids,
        &holder_host,
        &gpu_req,
    )
    .await;
}

#[tokio::test]
#[ignore = "requires RDMA NIC + CUDA GPU"]
async fn p2p_rdma_single_layer_roundtrip() {
    run_roundtrip(P2pCase::SingleLayer).await;
}

#[tokio::test]
#[ignore = "requires RDMA NIC + CUDA GPU"]
async fn p2p_rdma_multi_layer_multi_block() {
    run_roundtrip(P2pCase::MultiLayerMultiBlock).await;
}

// =============================================================================
// Class B: normal MHA TP>1 — cross-slot NUMA (each rank owns a slot column)
// =============================================================================

#[tokio::test]
#[ignore = "requires RDMA NIC + multiple CUDA GPUs spanning NUMA"]
async fn p2p_rdma_normal_tp_cross_numa() {
    pegaflow_common::logging::init_stdout_colored("debug");

    let tp_size = cuda_device_count().min(MAX_TEST_GPUS);
    if tp_size < 2 {
        eprintln!("skip: need >= 2 CUDA devices for normal-TP cross-NUMA case");
        return;
    }

    let shape = P2pShape {
        layers: 4,
        blocks: 16,
        block_size: 512,
        tp_size,
        world_size: tp_size,
    };
    let rank_bytes = shape.rank_bytes();
    let block_ids: Vec<usize> = (0..shape.blocks).collect();
    let block_hashes = make_block_hashes(shape.blocks, 81);

    let cluster = P2pCluster::new();
    let meta_store = spawn_metaserver(cluster.meta_port).await;
    let engine_a = Arc::new(
        PegaEngine::new_with_config(512 << 20, false, cluster.holder_config())
            .expect("holder engine"),
    );
    spawn_engine_server(Arc::clone(&engine_a), cluster.port_holder).await;

    // Each TP rank gets its own device + a distinct payload under the same
    // hashes (different KV heads = different bytes). Register *all* ranks before
    // saving — sealing requires the full worker set.
    let mut holder_gpus: Vec<DeviceGpu> = Vec::with_capacity(tp_size);
    let mut holder_host: Vec<Vec<u8>> = Vec::with_capacity(tp_size);
    for rank in 0..tp_size {
        let gpu = DeviceGpu::alloc(rank as i32, rank_bytes);
        let mut host = vec![0u8; rank_bytes];
        fill_test_pattern_salted(&mut host, shape.block_size, rank);
        gpu.copy_from_host(&host);
        register_worker(
            &engine_a,
            "inst-a",
            &shape,
            rank as i32,
            rank,
            0,
            tp_size,
            tp_size,
            gpu.as_u64(),
        );
        holder_gpus.push(gpu);
        holder_host.push(host);
    }
    for rank in 0..tp_size {
        save_worker_blocks(
            &engine_a,
            "inst-a",
            &shape,
            rank as i32,
            rank,
            0,
            &block_ids,
            &block_hashes,
            None,
        )
        .await;
    }

    wait_for_cache(
        &engine_a,
        "inst-a",
        &block_hashes,
        shape.blocks,
        Duration::from_secs(60),
    )
    .await;
    wait_for_metaserver_registration(
        &meta_store,
        P2P_NAMESPACE,
        &block_hashes,
        shape.blocks,
        Duration::from_secs(30),
    )
    .await;

    let (_, found) =
        engine_a.query_blocks_for_transfer(P2P_NAMESPACE, &block_hashes, "p2p-it-check");
    let holder_numas = holder_slot_numas_per_block(&found);
    assert_eq!(holder_numas.len(), shape.blocks);
    for slot_numas in &holder_numas {
        assert_eq!(
            slot_numas.len(),
            shape.layers * tp_size,
            "each block has one slot per (layer, tp_rank)"
        );
        assert!(slot_numas.iter().all(|n| n.is_valid()));
    }

    let plan = query_holder_transfer_plan(&engine_a, &block_hashes, "p2p-it-requester").await;
    assert_transfer_plan_numa(&plan);
    let slot_numa = distinct_slot_numas(&found);
    assert_eq!(
        unique_chunk_numa_nodes(&plan),
        slot_numa,
        "plan remote-chunk NUMA set must equal the holder's slot NUMA set"
    );
    if slot_numa.len() > 1 {
        assert!(
            any_block_spans_multiple_numa(&found),
            "TP across sockets must produce blocks whose slots span >1 NUMA"
        );
        eprintln!("normal-TP cross-slot multi-NUMA exercised: distinct={slot_numa:?}");
    } else {
        eprintln!("single-NUMA box: cross-slot multi-NUMA not exercised (distinct={slot_numa:?})");
    }

    // Requester mirrors the TP topology; each rank loads its own column.
    let engine_b = Arc::new(
        PegaEngine::new_with_config(512 << 20, false, cluster.requester_config())
            .expect("requester engine"),
    );
    let mut req_gpus: Vec<DeviceGpu> = Vec::with_capacity(tp_size);
    for rank in 0..tp_size {
        let gpu = DeviceGpu::alloc(rank as i32, rank_bytes);
        gpu.zero();
        register_worker(
            &engine_b,
            "inst-b",
            &shape,
            rank as i32,
            rank,
            0,
            tp_size,
            tp_size,
            gpu.as_u64(),
        );
        req_gpus.push(gpu);
    }

    let lease = wait_for_prefetch_done(
        &engine_b,
        "inst-b",
        "p2p-normal-tp",
        &block_hashes,
        shape.blocks,
        Duration::from_secs(120),
    )
    .await;

    for rank in 0..tp_size {
        requester_load_worker_and_verify(
            &engine_b,
            "inst-b",
            &shape,
            rank as i32,
            rank,
            lease,
            &block_ids,
            &holder_host[rank],
            &req_gpus[rank],
        )
        .await;
    }
}

// =============================================================================
// Class C: MLA replica (no DCP) — cross-block NUMA via server round-robin
// =============================================================================

#[tokio::test]
#[ignore = "requires RDMA NIC + multiple CUDA GPUs spanning NUMA"]
async fn p2p_rdma_mla_replica_round_robin_numa() {
    pegaflow_common::logging::init_stdout_colored("debug");

    let replicas = cuda_device_count().min(MAX_TEST_GPUS);
    if replicas < 2 {
        eprintln!("skip: need >= 2 CUDA devices for MLA-replica round-robin case");
        return;
    }

    // MLA without DCP: effective tp_size == 1 (one slot column), world_size ==
    // replica count. KV is identical across replicas, so the latent layout is a
    // single contiguous segment per slot.
    let shape = P2pShape {
        layers: 4,
        blocks: 16,
        block_size: 512,
        tp_size: 1,
        world_size: replicas,
    };
    let rank_bytes = shape.rank_bytes();
    let block_ids: Vec<usize> = (0..shape.blocks).collect();
    let block_hashes = make_block_hashes(shape.blocks, 82);

    let cluster = P2pCluster::new();
    let meta_store = spawn_metaserver(cluster.meta_port).await;
    let engine_a = Arc::new(
        PegaEngine::new_with_config(512 << 20, false, cluster.holder_config())
            .expect("holder engine"),
    );
    spawn_engine_server(Arc::clone(&engine_a), cluster.port_holder).await;

    // Register every replica (so the save group's NUMA candidates span both
    // sockets), but only effective rank 0 (device 0) carries and writes data.
    let mut host = vec![0u8; rank_bytes];
    fill_test_pattern_salted(&mut host, shape.block_size, 0);
    let mut holder_gpus: Vec<DeviceGpu> = Vec::with_capacity(replicas);
    for device in 0..replicas {
        let gpu = DeviceGpu::alloc(device as i32, rank_bytes);
        if device == 0 {
            gpu.copy_from_host(&host);
        }
        register_worker(
            &engine_a,
            "inst-a",
            &shape,
            device as i32,
            0,
            0,
            shape.tp_size,
            shape.world_size,
            gpu.as_u64(),
        );
        holder_gpus.push(gpu);
    }

    // Drive the real server-side round-robin: one save RPC per block, each with
    // the next candidate NUMA. On a single-NUMA box candidates.len() < 2 makes
    // this fall back to the device's NUMA (no rotation).
    let candidates = engine_a
        .registered_numa_nodes_for_save_group("inst-a", 0, 0)
        .expect("save-group NUMA candidates");
    let session = SessionRegistry::new();
    session.install(
        "inst-a".to_string(),
        P2P_NAMESPACE.to_string(),
        replicas as u32,
        replicas as u32,
    );
    for &block_id in &block_ids {
        let hint: Option<NumaNode> = session
            .next_save_numa_hint("inst-a", 0, 0, &candidates)
            .map(|h| h.numa_node);
        save_worker_blocks(
            &engine_a,
            "inst-a",
            &shape,
            0,
            0,
            0,
            std::slice::from_ref(&block_id),
            std::slice::from_ref(&block_hashes[block_id]),
            hint,
        )
        .await;
    }

    wait_for_cache(
        &engine_a,
        "inst-a",
        &block_hashes,
        shape.blocks,
        Duration::from_secs(60),
    )
    .await;
    wait_for_metaserver_registration(
        &meta_store,
        P2P_NAMESPACE,
        &block_hashes,
        shape.blocks,
        Duration::from_secs(30),
    )
    .await;

    let (_, found) =
        engine_a.query_blocks_for_transfer(P2P_NAMESPACE, &block_hashes, "p2p-it-check");
    let holder_numas = holder_slot_numas_per_block(&found);
    assert_eq!(holder_numas.len(), shape.blocks);
    for slot_numas in &holder_numas {
        assert_eq!(
            slot_numas.len(),
            shape.layers,
            "MLA replica has one slot per layer (effective tp_size == 1)"
        );
        assert!(slot_numas.iter().all(|n| n.is_valid()));
        let block_set: BTreeSet<u32> = slot_numas.iter().map(|n| n.0).collect();
        assert_eq!(
            block_set.len(),
            1,
            "each replica block is written by one save RPC, so all its slots share one NUMA"
        );
    }

    let plan = query_holder_transfer_plan(&engine_a, &block_hashes, "p2p-it-requester").await;
    assert_transfer_plan_numa(&plan);
    let slot_numa = distinct_slot_numas(&found);
    assert_eq!(
        unique_chunk_numa_nodes(&plan),
        slot_numa,
        "plan remote-chunk NUMA set must equal the holder's slot NUMA set"
    );
    assert!(
        !any_block_spans_multiple_numa(&found),
        "MLA replica blocks must each be single-NUMA (cross-block, not cross-slot)"
    );
    if slot_numa.len() > 1 {
        eprintln!(
            "MLA-replica round-robin cross-block multi-NUMA exercised: distinct={slot_numa:?}"
        );
    } else {
        eprintln!("single-NUMA box: round-robin not exercised (distinct={slot_numa:?})");
    }

    // Requester mirrors the replica topology; every replica loads the full block.
    let engine_b = Arc::new(
        PegaEngine::new_with_config(512 << 20, false, cluster.requester_config())
            .expect("requester engine"),
    );
    let mut req_gpus: Vec<DeviceGpu> = Vec::with_capacity(replicas);
    for device in 0..replicas {
        let gpu = DeviceGpu::alloc(device as i32, rank_bytes);
        gpu.zero();
        register_worker(
            &engine_b,
            "inst-b",
            &shape,
            device as i32,
            0,
            0,
            shape.tp_size,
            shape.world_size,
            gpu.as_u64(),
        );
        req_gpus.push(gpu);
    }

    let lease = wait_for_prefetch_done(
        &engine_b,
        "inst-b",
        "p2p-mla-replica",
        &block_hashes,
        shape.blocks,
        Duration::from_secs(120),
    )
    .await;

    for (device, gpu) in req_gpus.iter().enumerate() {
        requester_load_worker_and_verify(
            &engine_b,
            "inst-b",
            &shape,
            device as i32,
            0,
            lease,
            &block_ids,
            &host,
            gpu,
        )
        .await;
    }
}
