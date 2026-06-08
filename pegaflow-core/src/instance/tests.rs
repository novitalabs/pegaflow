use super::*;

#[test]
fn registration_valid() {
    let reg = KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
    assert_eq!(reg.block_size_bytes, 1024);
}

#[test]
fn registration_null_pointer_rejected() {
    assert!(KVCacheRegistration::new(0, 1024, 10, 64, 0, 1).is_err());
}

#[test]
fn registration_memory_too_small() {
    let err = KVCacheRegistration::new(0x1000, 5120, 10, 1024, 0, 1).unwrap_err();
    assert!(err.contains("too small"));
}

#[test]
fn padded_block_size() {
    // Unaligned: 8848 % 512 = 144, padded to 9216
    let reg = KVCacheRegistration::new(0x1000, 10_000_000, 100, 8848, 0, 1)
        .unwrap()
        .with_ssd_padding(512);
    assert_eq!(reg.padded_bytes_per_block, 9216);
    assert_eq!(reg.padded_block_size_bytes, 9216);

    // Already aligned: no change
    let reg = KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1)
        .unwrap()
        .with_ssd_padding(512);
    assert_eq!(reg.padded_bytes_per_block, 1024);
    assert_eq!(reg.padded_block_size_bytes, 1024);

    // Split layout: padded per segment, total = padded * segments
    let reg = KVCacheRegistration::new(0x1000, 10_000_000, 100, 8848, 900_000, 2)
        .unwrap()
        .with_ssd_padding(512);
    assert_eq!(reg.padded_bytes_per_block, 9216);
    assert_eq!(reg.padded_block_size_bytes, 9216 * 2);
}

/// Simulates the inference-side registration flow (single GPU).
/// Covers: instance creation -> GPU context creation -> batch layer registration.
#[test]
fn inference_registration_flow() {
    // 1. Create instance (like get_or_create_instance)
    let instance = InstanceContext::new(
        "test-instance-1".to_string(),
        "model-ns".to_string(),
        64, // num_layers
        8,  // tp_size
        8,  // world_size
    )
    .expect("create instance");

    // Use UNKNOWN NUMA node for tests (no actual GPU/NUMA in CI)
    let numa = NumaNode::UNKNOWN;

    // 2. Build all layer registrations for device 0
    let mut kv_caches = HashMap::new();
    let mut layer_ids = HashMap::new();
    for layer_id in 0..4 {
        let layer_name = format!("layer_{}", layer_id);
        let reg = KVCacheRegistration::new(
            0x1000 + layer_id as u64 * 0x10000,
            1024 * 1024,
            100,
            1024,
            0,
            1,
        )
        .unwrap();
        layer_ids.insert(layer_name.clone(), layer_id);
        kv_caches.insert(layer_name, reg);
    }

    // 3. Register all layers at once
    instance
        .register_new_gpu(GpuRegistration {
            device_id: 0,
            tp_rank: 0,
            pp_rank: 0,
            numa_node: numa,
            transfer_mode: TransferMode::Direct,
            kv_caches,
            layer_ids_by_name: layer_ids,
        })
        .expect("register gpu with layers");

    // 4. Verify topology checking
    assert!(instance.verify_topology(64, 8, 8).is_ok());
    // num_layers is fixed by the connector-declared topology.
    assert!(instance.verify_topology(32, 8, 8).is_err());
    // tp_size / world_size mismatch is still rejected
    assert!(instance.verify_topology(64, 4, 8).is_err());
    assert!(instance.verify_topology(64, 8, 4).is_err());

    // 5. Verify duplicate GPU registration fails
    let dup_caches = HashMap::from([(
        "layer_0".to_string(),
        KVCacheRegistration::new(0x2000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
    )]);
    let dup_ids = HashMap::from([("layer_0".to_string(), 0)]);
    let err = instance
        .register_new_gpu(GpuRegistration {
            device_id: 0,
            tp_rank: 0,
            pp_rank: 0,
            numa_node: numa,
            transfer_mode: TransferMode::Direct,
            kv_caches: dup_caches,
            layer_ids_by_name: dup_ids,
        })
        .expect_err("duplicate GPU registration should fail");
    assert!(err.to_string().contains("already exists"));

    // 6. Verify we can get the registered layer back
    let gpu = instance.get_gpu(0).expect("get gpu context");
    let reg = gpu.get_registration("layer_0").expect("get registration");
    assert_eq!(reg.data_ptr, 0x1000);
    assert_eq!(reg.num_blocks, 100);
}

/// Tests MLA-style registration where the same layer name is used
/// by multiple TP ranks. Verifies explicit layer IDs are idempotent.
#[test]
fn explicit_layer_id_registration_is_idempotent() {
    let instance = InstanceContext::new(
        "mla-instance".to_string(),
        "mla-ns".to_string(),
        10, // num_layers
        1,  // tp_size (MLA uses tp_size=1)
        8,  // world_size (8 TP ranks, but treated as 1 for storage)
    )
    .expect("create instance");

    for _ in 0..8 {
        instance
            .register_layer_id("layer_0", 0)
            .expect("same layer id is idempotent");
    }

    assert_eq!(instance.get_layer_id("layer_0"), Some(0));

    instance
        .register_layer_id("layer_1", 1)
        .expect("register another explicit layer id");

    assert_eq!(instance.get_layer_id("layer_0"), Some(0));
    assert_eq!(instance.get_layer_id("layer_1"), Some(1));
}

#[test]
fn explicit_layer_id_rejects_out_of_range_and_conflicts() {
    let instance = InstanceContext::new("id-instance".to_string(), "id-ns".to_string(), 2, 1, 1)
        .expect("create instance");

    instance
        .register_layer_id("layer_0", 0)
        .expect("register first layer");

    let err = instance
        .register_layer_id("layer_0", 1)
        .expect_err("same name cannot change id");
    assert!(err.to_string().contains("already registered with id 0"));

    let err = instance
        .register_layer_id("other_layer", 0)
        .expect_err("same id cannot point to another name");
    assert!(err.to_string().contains("already registered for layer_0"));

    let err = instance
        .register_layer_id("layer_2", 2)
        .expect_err("id must stay inside fixed num_layers");
    assert!(err.to_string().contains("out of range"));
}

#[test]
fn failed_batch_registration_does_not_commit_layer_ids() {
    let instance = InstanceContext::new(
        "batch-conflict-instance".to_string(),
        "batch-conflict-ns".to_string(),
        3,
        1,
        1,
    )
    .expect("create instance");

    let kv_caches = HashMap::from([
        (
            "layer_0".to_string(),
            KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
        ),
        (
            "layer_1".to_string(),
            KVCacheRegistration::new(0x2000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
        ),
    ]);
    let duplicate_ids = HashMap::from([("layer_0".to_string(), 0), ("layer_1".to_string(), 0)]);

    let err = instance
        .register_new_gpu(GpuRegistration {
            device_id: 0,
            tp_rank: 0,
            pp_rank: 0,
            numa_node: NumaNode::UNKNOWN,
            transfer_mode: TransferMode::Direct,
            kv_caches,
            layer_ids_by_name: duplicate_ids,
        })
        .expect_err("duplicate ids in one batch must fail");
    assert!(err.to_string().contains("appears more than once"));
    assert_eq!(instance.get_layer_id("layer_0"), None);
    assert_eq!(instance.get_layer_id("layer_1"), None);

    instance
        .register_layer_id("layer_0", 0)
        .expect("failed batch did not poison layer_0");
    instance
        .register_layer_id("layer_1", 1)
        .expect("failed batch did not poison layer_1");
}

#[test]
fn failed_batch_conflict_does_not_commit_valid_prefix() {
    let instance = InstanceContext::new(
        "batch-prefix-instance".to_string(),
        "batch-prefix-ns".to_string(),
        3,
        1,
        1,
    )
    .expect("create instance");
    instance
        .register_layer_id("layer_0", 0)
        .expect("register existing layer");

    let kv_caches = HashMap::from([
        (
            "layer_1".to_string(),
            KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
        ),
        (
            "other_layer_0".to_string(),
            KVCacheRegistration::new(0x2000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
        ),
    ]);
    let layer_ids = HashMap::from([("layer_1".to_string(), 1), ("other_layer_0".to_string(), 0)]);

    let err = instance
        .register_new_gpu(GpuRegistration {
            device_id: 0,
            tp_rank: 0,
            pp_rank: 0,
            numa_node: NumaNode::UNKNOWN,
            transfer_mode: TransferMode::Direct,
            kv_caches,
            layer_ids_by_name: layer_ids,
        })
        .expect_err("conflicting id must fail the whole batch");
    assert!(err.to_string().contains("already registered for layer_0"));
    assert_eq!(instance.get_layer_id("layer_0"), Some(0));
    assert_eq!(instance.get_layer_id("layer_1"), None);
    assert_eq!(instance.get_layer_id("other_layer_0"), None);
}

#[test]
fn cross_layer_pp_slots_are_order_independent() {
    let producer = InstanceContext::new(
        "producer-instance".to_string(),
        "pp-ns".to_string(),
        4,
        1,
        4,
    )
    .expect("create producer");
    let consumer = InstanceContext::new(
        "consumer-instance".to_string(),
        "pp-ns".to_string(),
        4,
        1,
        4,
    )
    .expect("create consumer");

    for pp_rank in [2, 3, 1, 0] {
        producer
            .register_layer_id(&format!("ALL_LAYERS_pp{pp_rank}"), pp_rank)
            .expect("producer registers explicit pp id");
    }
    for pp_rank in [3, 1, 2, 0] {
        consumer
            .register_layer_id(&format!("ALL_LAYERS_pp{pp_rank}"), pp_rank)
            .expect("consumer registers explicit pp id");
    }

    for pp_rank in 0..4 {
        let layer_name = format!("ALL_LAYERS_pp{pp_rank}");
        assert_eq!(producer.get_layer_id(&layer_name), Some(pp_rank));
        assert_eq!(consumer.get_layer_id(&layer_name), Some(pp_rank));
        assert_eq!(producer.get_slot_index(pp_rank, 0).unwrap(), pp_rank);
        assert_eq!(consumer.get_slot_index(pp_rank, 0).unwrap(), pp_rank);
    }
}

#[test]
fn completeness_check_rejects_missing_slots() {
    let instance = InstanceContext::new(
        "complete-instance".to_string(),
        "complete-ns".to_string(),
        1,
        1,
        1,
    )
    .expect("create instance");
    let kv_caches = HashMap::from([(
        "layer_0".to_string(),
        KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
    )]);
    let layer_ids = HashMap::from([("layer_0".to_string(), 0)]);
    instance
        .register_new_gpu(GpuRegistration {
            device_id: 0,
            tp_rank: 0,
            pp_rank: 0,
            numa_node: NumaNode::UNKNOWN,
            transfer_mode: TransferMode::Direct,
            kv_caches,
            layer_ids_by_name: layer_ids,
        })
        .expect("complete single-slot registration");
    instance
        .ensure_all_slots_registered()
        .expect("all slots are registered");

    let sparse = InstanceContext::new(
        "sparse-instance".to_string(),
        "sparse-ns".to_string(),
        2,
        1,
        1,
    )
    .expect("create sparse instance");
    sparse
        .register_layer_id("layer_0", 0)
        .expect("register one layer id");
    let err = sparse
        .ensure_all_slots_registered()
        .expect_err("missing slots must be rejected");
    assert!(err.to_string().contains("incomplete KV registration"));
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
