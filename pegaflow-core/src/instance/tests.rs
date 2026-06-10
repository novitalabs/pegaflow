use super::*;
use crate::layout::BlockCopies;

// Pure layout validation tests live in `crate::layout::tests`.

// --- Real registration path ---
//
// Every layer-id assertion below exercises `register_new_gpu`, the exact path
// the gRPC service calls. There is no test-only shim that re-implements the
// validation: pre-CUDA rejections (out-of-range / in-batch / cross-device
// conflicts) never touch a device, while a committed GPU uses device 0.

/// Build a `GpuRegistration` whose layers carry explicit connector-declared ids.
/// Data pointers are assigned deterministically from the slice order so callers
/// can assert on `get_registration`.
fn gpu_registration(device_id: i32, tp_rank: usize, layers: &[(&str, usize)]) -> GpuRegistration {
    let mut kv_caches = HashMap::new();
    let mut layer_ids_by_name = HashMap::new();
    for (index, (name, id)) in layers.iter().enumerate() {
        let layout = KVCacheLayout::new(
            0x1000 + index as u64 * 0x10000,
            1024 * 1024,
            100,
            1024,
            0,
            1,
        )
        .unwrap();
        kv_caches.insert((*name).to_string(), layout);
        layer_ids_by_name.insert((*name).to_string(), *id);
    }
    GpuRegistration {
        device_id,
        tp_rank,
        pp_rank: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        kv_caches,
        layer_ids_by_name,
    }
}

/// Inference-side registration: instance creation -> GPU context creation ->
/// batch layer registration, then topology and lookup checks. Commits device 0.
#[test]
fn inference_registration_flow() {
    let instance =
        InstanceContext::new("test-instance-1".into(), "model-ns".into(), 64, 8, 8).unwrap();

    instance
        .register_new_gpu(gpu_registration(
            0,
            0,
            &[
                ("layer_0", 0),
                ("layer_1", 1),
                ("layer_2", 2),
                ("layer_3", 3),
            ],
        ))
        .expect("register gpu with layers");

    // num_layers is fixed by the connector-declared topology.
    assert!(instance.verify_topology(64, 8, 8).is_ok());
    assert!(instance.verify_topology(32, 8, 8).is_err());
    // tp_size / world_size mismatch is still rejected.
    assert!(instance.verify_topology(64, 4, 8).is_err());
    assert!(instance.verify_topology(64, 8, 4).is_err());

    // Explicit ids are stored and drive slot placement (tp_size = 8).
    assert_eq!(instance.get_layer_id("layer_2"), Some(2));
    assert_eq!(instance.get_slot_index(2, 0).unwrap(), 2 * 8);

    // Duplicate GPU registration fails.
    let err = instance
        .register_new_gpu(gpu_registration(0, 0, &[("layer_0", 0)]))
        .expect_err("duplicate GPU registration should fail");
    assert!(err.to_string().contains("already exists"));

    // The registered layer is retrievable with its original layout.
    let gpu = instance.get_gpu(0).expect("get gpu context");
    let layout = gpu.get_layout("layer_0").expect("get layout");
    assert_eq!(layout.num_blocks(), 100);
    match layout.block_copies(0).expect("block 0 in range") {
        BlockCopies::Contiguous(c) => assert_eq!(c.addr, 0x1000),
        BlockCopies::Split { .. } => panic!("dense test layout must be contiguous"),
    }
}

/// An id outside `[0, num_layers)` is rejected while building assignments,
/// before any device work, and commits nothing.
#[test]
fn register_rejects_out_of_range_layer_id() {
    let instance =
        InstanceContext::new("range-instance".into(), "range-ns".into(), 2, 1, 1).unwrap();
    let err = instance
        .register_new_gpu(gpu_registration(0, 0, &[("layer_2", 2)]))
        .expect_err("id must stay inside fixed num_layers");
    assert!(err.to_string().contains("out of range"));
    assert_eq!(instance.get_layer_id("layer_2"), None);
}

/// Two layers in one batch claiming the same id are rejected before commit,
/// and the rejection leaves no poison: a valid batch then commits cleanly.
#[test]
fn register_rejects_in_batch_duplicate_layer_id() {
    let instance =
        InstanceContext::new("batch-dup-instance".into(), "batch-dup-ns".into(), 3, 1, 1).unwrap();

    let err = instance
        .register_new_gpu(gpu_registration(0, 0, &[("layer_0", 0), ("layer_1", 0)]))
        .expect_err("duplicate ids in one batch must fail");
    assert!(err.to_string().contains("appears more than once"));
    assert_eq!(instance.get_layer_id("layer_0"), None);
    assert_eq!(instance.get_layer_id("layer_1"), None);

    instance
        .register_new_gpu(gpu_registration(0, 0, &[("layer_0", 0), ("layer_1", 1)]))
        .expect("valid batch commits after a rejected one");
    assert_eq!(instance.get_layer_id("layer_0"), Some(0));
    assert_eq!(instance.get_layer_id("layer_1"), Some(1));
}

/// A batch that conflicts with already-committed ids is rejected atomically:
/// neither the conflicting entry nor a valid prefix in the same batch commits,
/// and the prior state is preserved. Commits device 0; conflicting batches on
/// device 1 fail validation before any CUDA context is created.
#[test]
fn register_rejects_conflicting_id_and_preserves_committed_state() {
    let instance =
        InstanceContext::new("conflict-instance".into(), "conflict-ns".into(), 3, 1, 1).unwrap();
    instance
        .register_new_gpu(gpu_registration(0, 0, &[("layer_0", 0)]))
        .expect("commit first device");

    // Same name, different id.
    let err = instance
        .register_new_gpu(gpu_registration(1, 0, &[("layer_0", 1)]))
        .expect_err("same name cannot change id");
    assert!(err.to_string().contains("already registered with id 0"));

    // Same id, different name.
    let err = instance
        .register_new_gpu(gpu_registration(1, 0, &[("other_layer", 0)]))
        .expect_err("same id cannot point to another name");
    assert!(err.to_string().contains("already registered for layer_0"));

    // Valid prefix (layer_1 -> 1) + conflicting entry (other_layer_0 -> 0):
    // the whole batch is rejected, so the valid prefix must NOT leak in.
    let err = instance
        .register_new_gpu(gpu_registration(
            1,
            0,
            &[("layer_1", 1), ("other_layer_0", 0)],
        ))
        .expect_err("conflicting id must fail the whole batch");
    assert!(err.to_string().contains("already registered for layer_0"));

    assert_eq!(instance.get_layer_id("layer_0"), Some(0));
    assert_eq!(instance.get_layer_id("layer_1"), None);
    assert_eq!(instance.get_layer_id("other_layer"), None);
    assert_eq!(instance.get_layer_id("other_layer_0"), None);
}

/// Completeness check: a fully covered topology passes; a topology with an
/// unregistered slot is rejected. Both commit device 0.
#[test]
fn ensure_all_slots_registered_detects_missing_slot() {
    let complete =
        InstanceContext::new("complete-instance".into(), "complete-ns".into(), 1, 1, 1).unwrap();
    complete
        .register_new_gpu(gpu_registration(0, 0, &[("layer_0", 0)]))
        .expect("complete single-slot registration");
    complete
        .ensure_all_slots_registered()
        .expect("all slots are registered");

    let sparse =
        InstanceContext::new("sparse-instance".into(), "sparse-ns".into(), 2, 1, 1).unwrap();
    sparse
        .register_new_gpu(gpu_registration(0, 0, &[("layer_0", 0)]))
        .expect("register one of two slots");
    let err = sparse
        .ensure_all_slots_registered()
        .expect_err("missing slots must be rejected");
    assert!(err.to_string().contains("incomplete KV registration"));
}

/// True when the host exposes at least `n` CUDA devices. Replica-registration
/// contract tests need one real CUDA context per device; on smaller boxes they
/// skip instead of failing.
fn has_cuda_devices(n: usize) -> bool {
    (0..n).all(|device_id| CudaContext::new(device_id).is_ok())
}

/// GLM-5.1-FP8 scaled to 2 model layers x 2 caches (main MLA attention + DSA
/// sparse indexer); the layer at the last in-model index stands in for the MTP
/// next-n predictor, whose ids sit at the top of the space.
const MLA_DSA_LAYERS: &[(&str, usize)] = &[
    ("model.layers.0.self_attn.attn", 0),
    ("model.layers.0.self_attn.indexer.k_cache", 1),
    ("model.layers.1.self_attn.attn", 2),
    ("model.layers.1.self_attn.indexer.k_cache", 3),
];

/// Registration contract for MLA replicas: KV is identical across TP ranks, so
/// the connector collapses them to effective tp_rank 0 and every worker
/// registers the full slot range from its own device (rank 0 saves, each
/// device loads into its own copy). Multiple owners per slot are valid.
#[test]
fn mla_replica_registration_is_accepted() {
    if !has_cuda_devices(2) {
        eprintln!("skipping mla_replica_registration_is_accepted: needs >= 2 CUDA devices");
        return;
    }

    let instance = InstanceContext::new("mla-replica".into(), "mla-ns".into(), 4, 1, 2).unwrap();
    instance
        .register_new_gpu(gpu_registration(0, 0, MLA_DSA_LAYERS))
        .expect("register replica on device 0");
    instance
        .register_new_gpu(gpu_registration(1, 0, MLA_DSA_LAYERS))
        .expect("register replica on device 1");

    instance
        .ensure_all_slots_registered()
        .expect("replica owners on the same slots are a valid MLA topology");
}

/// Replicas may only share a slot within one pipeline stage: the same layer
/// claimed from two pp_ranks is a topology error, not replication.
#[test]
fn rejects_replicas_across_pipeline_stages() {
    if !has_cuda_devices(2) {
        eprintln!("skipping rejects_replicas_across_pipeline_stages: needs >= 2 CUDA devices");
        return;
    }

    let instance = InstanceContext::new("pp-conflict".into(), "pp-ns".into(), 1, 1, 2).unwrap();
    let layers = &[("model.layers.0.self_attn.attn", 0)];
    instance
        .register_new_gpu(gpu_registration(0, 0, layers))
        .expect("register pp_rank 0 owner");
    instance
        .register_new_gpu(GpuRegistration {
            pp_rank: 1,
            ..gpu_registration(1, 0, layers)
        })
        .expect("name/id mapping is consistent, registration itself succeeds");

    let err = instance
        .ensure_all_slots_registered()
        .expect_err("one layer on two pipeline stages must be rejected");
    assert!(err.to_string().contains("different pipeline stages"));
}

/// PP x MLA: pipeline stages partition the id space and each stage holds its
/// own replica group. Completeness requires every stage registered.
#[test]
fn pp_stages_partition_slots_with_replicas_per_stage() {
    if !has_cuda_devices(4) {
        eprintln!(
            "skipping pp_stages_partition_slots_with_replicas_per_stage: needs >= 4 CUDA devices"
        );
        return;
    }

    let stage0 = &MLA_DSA_LAYERS[..2];
    let stage1 = &MLA_DSA_LAYERS[2..];
    let instance = InstanceContext::new("pp-mla".into(), "pp-mla-ns".into(), 4, 1, 4).unwrap();

    instance
        .register_new_gpu(gpu_registration(0, 0, stage0))
        .expect("stage 0 replica on device 0");
    instance
        .register_new_gpu(gpu_registration(1, 0, stage0))
        .expect("stage 0 replica on device 1");

    let err = instance
        .ensure_all_slots_registered()
        .expect_err("stage 1 slots are still unregistered");
    assert!(err.to_string().contains("incomplete KV registration"));

    instance
        .register_new_gpu(GpuRegistration {
            pp_rank: 1,
            ..gpu_registration(2, 0, stage1)
        })
        .expect("stage 1 replica on device 2");
    instance
        .register_new_gpu(GpuRegistration {
            pp_rank: 1,
            ..gpu_registration(3, 0, stage1)
        })
        .expect("stage 1 replica on device 3");

    instance
        .ensure_all_slots_registered()
        .expect("both stages covered, replicas within each stage");
}

#[test]
fn save_numa_hint_validation_rejects_unknown_or_unregistered_nodes() {
    let instance =
        InstanceContext::new("hint-instance".to_string(), "hint-ns".to_string(), 1, 1, 1)
            .expect("create instance");

    let err = instance
        .validate_save_numa_hint(0, 0, NumaNode::UNKNOWN)
        .expect_err("unknown NUMA hint should fail");
    assert!(err.to_string().contains("valid NUMA node"));

    let err = instance
        .validate_save_numa_hint(0, 0, NumaNode(0))
        .expect_err("unregistered NUMA hint should fail");
    assert!(err.to_string().contains("not registered"));
}
