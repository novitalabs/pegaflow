use super::*;
use crate::layout::BlockCopies;

// Pure layout validation tests live in `crate::layout::tests`.

// --- Real registration path ---
//
// Every assertion below exercises `register_new_gpu`, the exact path the gRPC
// service calls. There is no test-only shim that re-implements sealing:
// single-device flows commit device 0; multi-device flows are gated on the
// host exposing enough CUDA devices and skip otherwise.

/// Build a `GpuRegistration` for the given layer names. Data pointers are
/// assigned deterministically from the slice order.
fn gpu_registration(device_id: i32, tp_rank: usize, layers: &[&str]) -> GpuRegistration {
    gpu_registration_with_segment_bytes(device_id, tp_rank, layers, 1024)
}

fn gpu_registration_with_segment_bytes(
    device_id: i32,
    tp_rank: usize,
    layers: &[&str],
    segment_bytes: usize,
) -> GpuRegistration {
    let mut kv_caches = HashMap::new();
    for (index, name) in layers.iter().enumerate() {
        let layout = KVCacheLayout::new(
            0x1000 + index as u64 * 0x10000,
            1024 * 1024,
            100,
            segment_bytes,
            0,
            1,
        )
        .unwrap();
        kv_caches.insert((*name).to_string(), layout);
    }
    GpuRegistration {
        device_id,
        tp_rank,
        pp_rank: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        kv_caches,
    }
}

/// Single-worker registration seals immediately: ids are the sorted-name
/// ranks, slot math follows, and the layout stays retrievable. Commits
/// device 0.
#[test]
fn single_worker_registration_seals_topology() {
    let instance =
        InstanceContext::new("test-instance-1".into(), "model-ns".into(), 1, 1, false).unwrap();

    // Names intentionally out of registration order: ids come from sorted
    // names, not from declaration order.
    instance
        .register_new_gpu(gpu_registration(
            0,
            0,
            &["layer_b", "layer_a", "layer_d", "layer_c"],
        ))
        .expect("register gpu with layers");

    let topology = instance
        .sealed_topology()
        .expect("sealed after last worker");
    assert_eq!(topology.num_layers(), 4);
    assert_eq!(topology.layer_id("layer_a").unwrap(), 0);
    assert_eq!(topology.layer_id("layer_d").unwrap(), 3);
    assert!(topology.layer_id("layer_x").is_err());
    assert_eq!(topology.slot_index(2, 0).unwrap(), 2);
    assert_eq!(topology.total_slots(), 4);

    // Topology parameters are pinned at creation.
    assert!(instance.verify_topology(1, 1, false).is_ok());
    assert!(instance.verify_topology(2, 1, false).is_err());
    assert!(instance.verify_topology(1, 2, false).is_err());

    // A sealed instance accepts no further devices.
    let err = instance
        .register_new_gpu(gpu_registration(1, 0, &["layer_a"]))
        .expect_err("sealed instance must reject new devices");
    assert!(err.to_string().contains("already fully registered"));

    // The registered layer is retrievable with its original layout.
    let gpu = instance.get_gpu(0).expect("get gpu context");
    let layout = gpu.get_layout("layer_b").expect("get layout");
    assert_eq!(layout.num_blocks(), 100);
    match layout.block_copies(0).expect("block 0 in range") {
        BlockCopies::Contiguous(c) => assert_eq!(c.addr, 0x1000),
        BlockCopies::Split { .. } => panic!("dense test layout must be contiguous"),
    }
}

/// Page-first folds the layer dimension into one page slot per tp_rank and
/// lays layers out contiguously in sorted-name (layer-id) order. Commits
/// device 0.
#[test]
fn page_first_collapses_slots_and_lays_out_page() {
    let instance = InstanceContext::new("page-first".into(), "page-ns".into(), 1, 1, true).unwrap();

    // segment_bytes=1024, single segment, no SSD padding on this path, so each
    // layer's padded_block_bytes == 1024.
    instance
        .register_new_gpu(gpu_registration_with_segment_bytes(
            0,
            0,
            &["layer_b", "layer_a", "layer_c"],
            1024,
        ))
        .expect("register page-first gpu");

    let topology = instance.sealed_topology().expect("sealed");
    assert!(topology.is_page_first());
    assert_eq!(topology.num_layers(), 3);
    // Layer dimension folds away: one slot per tp_rank (here tp_size == 1).
    assert_eq!(topology.total_slots(), 1);
    for layer_id in 0..3 {
        assert_eq!(topology.slot_index(layer_id, 0).unwrap(), 0);
    }
    // Page = all layers concatenated in sorted-name order (a<b<c).
    assert_eq!(topology.page_size(), Some(3 * 1024));
    assert_eq!(topology.page_placement(0), Some((0, 1024)));
    assert_eq!(topology.page_placement(1), Some((1024, 1024)));
    assert_eq!(topology.page_placement(2), Some((2048, 1024)));

    // page_first is part of the topology contract: a mismatched re-registration
    // intent is rejected.
    assert!(instance.verify_topology(1, 1, true).is_ok());
    assert!(instance.verify_topology(1, 1, false).is_err());
}

/// Until every worker has registered there is no topology: save/load must be
/// rejected with a registration-progress error. Commits device 0.
#[test]
fn unsealed_instance_rejects_topology_access() {
    let instance =
        InstanceContext::new("partial".into(), "partial-ns".into(), 2, 2, false).unwrap();
    instance
        .register_new_gpu(gpu_registration(0, 0, &["layer_0"]))
        .expect("first of two workers");

    let err = instance
        .sealed_topology()
        .expect_err("topology must not exist before the last worker registers");
    assert!(err.to_string().contains("registration incomplete: 1/2"));

    // Duplicate device registration is still rejected while filling.
    let err = instance
        .register_new_gpu(gpu_registration(0, 0, &["layer_0"]))
        .expect_err("duplicate GPU registration should fail");
    assert!(err.to_string().contains("already exists"));
}

/// An empty layer set carries no information and would silently widen the
/// sealed topology contract; reject it outright.
#[test]
fn registration_without_layers_is_rejected() {
    let instance = InstanceContext::new("empty".into(), "empty-ns".into(), 1, 1, false).unwrap();
    let err = instance
        .register_new_gpu(gpu_registration(0, 0, &[]))
        .expect_err("empty registration must fail");
    assert!(err.to_string().contains("no KV cache layers"));
}

/// True when the host exposes at least `n` CUDA devices. Multi-worker sealing
/// contract tests need one real CUDA context per device; on smaller boxes
/// they skip instead of failing.
fn has_cuda_devices(n: usize) -> bool {
    (0..n).all(|device_id| CudaContext::new(device_id).is_ok())
}

/// The layer-id space is the union of what workers actually register — no
/// pre-declared layer count. A PP stage owning an extra trailing layer (the
/// optional speculative MTP predictor) widens the space; nothing has to be
/// guessed up front, and ids stay the sorted-name ranks.
#[test]
fn seal_derives_layer_space_from_union_of_workers() {
    if !has_cuda_devices(2) {
        eprintln!(
            "skipping seal_derives_layer_space_from_union_of_workers: needs >= 2 CUDA devices"
        );
        return;
    }

    let instance = InstanceContext::new("pp-mtp".into(), "pp-mtp-ns".into(), 1, 2, false).unwrap();
    instance
        .register_new_gpu(gpu_registration(0, 0, &["model.layers.0.self_attn.attn"]))
        .expect("stage 0 registers the main layer");
    instance
        .register_new_gpu(GpuRegistration {
            pp_rank: 1,
            ..gpu_registration(
                1,
                0,
                &[
                    "model.layers.1.self_attn.attn",
                    "model.layers.2.self_attn.attn", // speculative MTP layer
                ],
            )
        })
        .expect("stage 1 registers main + MTP layers");

    let topology = instance.sealed_topology().expect("sealed");
    assert_eq!(topology.num_layers(), 3);
    assert_eq!(
        topology.layer_id("model.layers.2.self_attn.attn").unwrap(),
        2
    );
}

/// GLM-5.1-style MLA + DSA: two caches per transformer layer (main attention
/// + sparse indexer) sharing an in-model index.
const MLA_DSA_LAYERS: &[&str] = &[
    "model.layers.0.self_attn.attn",
    "model.layers.0.self_attn.indexer.k_cache",
    "model.layers.1.self_attn.attn",
    "model.layers.1.self_attn.indexer.k_cache",
];

/// Registration contract for MLA replicas: KV is identical across TP ranks, so
/// the connector collapses them to effective tp_rank 0 and every worker
/// registers the full slot range from its own device (rank 0 saves, each
/// device loads into its own copy). Multiple owners per slot are valid and the
/// instance seals once all devices are in.
#[test]
fn mla_replica_registration_seals() {
    if !has_cuda_devices(2) {
        eprintln!("skipping mla_replica_registration_seals: needs >= 2 CUDA devices");
        return;
    }

    let instance =
        InstanceContext::new("mla-replica".into(), "mla-ns".into(), 1, 2, false).unwrap();
    instance
        .register_new_gpu(gpu_registration(0, 0, MLA_DSA_LAYERS))
        .expect("register replica on device 0");
    instance
        .register_new_gpu(gpu_registration(1, 0, MLA_DSA_LAYERS))
        .expect("register replica on device 1, sealing the instance");

    let topology = instance.sealed_topology().expect("sealed");
    assert_eq!(topology.num_layers(), 4);
    assert_eq!(topology.total_slots(), 4);
}

/// Replicas may only share a slot within one pipeline stage: the same layer
/// claimed from two pp_ranks is a topology error, not replication. The
/// conflict is detected when the last worker tries to seal, and that
/// registration does not commit.
#[test]
fn seal_rejects_replicas_across_pipeline_stages() {
    if !has_cuda_devices(2) {
        eprintln!("skipping seal_rejects_replicas_across_pipeline_stages: needs >= 2 CUDA devices");
        return;
    }

    let instance = InstanceContext::new("pp-conflict".into(), "pp-ns".into(), 1, 2, false).unwrap();
    let layers = &["model.layers.0.self_attn.attn"];
    instance
        .register_new_gpu(gpu_registration(0, 0, layers))
        .expect("register pp_rank 0 owner");

    let err = instance
        .register_new_gpu(GpuRegistration {
            pp_rank: 1,
            ..gpu_registration(1, 0, layers)
        })
        .expect_err("one layer on two pipeline stages must be rejected at seal");
    assert!(err.to_string().contains("different pipeline stages"));

    // The failed registration did not commit: the instance is still waiting
    // for a valid second worker.
    let err = instance.sealed_topology().expect_err("still unsealed");
    assert!(err.to_string().contains("registration incomplete: 1/2"));
}

/// A worker set that leaves a (layer, tp_rank) slot unowned is rejected when
/// the last worker registers: with per-rank KV (non-MLA), every rank must
/// register every layer.
#[test]
fn seal_rejects_missing_slot_owner() {
    if !has_cuda_devices(2) {
        eprintln!("skipping seal_rejects_missing_slot_owner: needs >= 2 CUDA devices");
        return;
    }

    let instance =
        InstanceContext::new("missing-slot".into(), "missing-ns".into(), 2, 2, false).unwrap();
    instance
        .register_new_gpu(gpu_registration(0, 0, &["layer_0", "layer_1"]))
        .expect("rank 0 registers both layers");

    let err = instance
        .register_new_gpu(gpu_registration(1, 1, &["layer_0"]))
        .expect_err("rank 1 missing layer_1 must fail the seal");
    assert!(err.to_string().contains("incomplete KV registration"));
    assert!(err.to_string().contains("layer_1"));

    // Re-registering rank 1 with the full layer set seals the instance.
    instance
        .register_new_gpu(gpu_registration(1, 1, &["layer_0", "layer_1"]))
        .expect("complete worker set seals");
    assert!(instance.sealed_topology().is_ok());
}

/// The same layer name must describe the same stored bytes on every device;
/// otherwise replicas would seal a slot whose data differs by device.
#[test]
fn seal_rejects_inconsistent_layer_geometry() {
    if !has_cuda_devices(2) {
        eprintln!("skipping seal_rejects_inconsistent_layer_geometry: needs >= 2 CUDA devices");
        return;
    }

    let instance = InstanceContext::new("geom".into(), "geom-ns".into(), 1, 2, false).unwrap();
    instance
        .register_new_gpu(gpu_registration_with_segment_bytes(
            0,
            0,
            &["layer_0"],
            1024,
        ))
        .expect("register first replica");

    let err = instance
        .register_new_gpu(gpu_registration_with_segment_bytes(
            1,
            0,
            &["layer_0"],
            2048,
        ))
        .expect_err("same name with different geometry must fail the seal");
    assert!(err.to_string().contains("inconsistent geometry"));
}
