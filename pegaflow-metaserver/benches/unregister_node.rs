use criterion::{Criterion, criterion_group, criterion_main};
use pegaflow_metaserver::store::{BlockHashStore, StoreConfig};
use std::time::Duration;
use uuid::Uuid;

const TOTAL_KEYS: usize = 1_000_000;
const TARGET_OWNED_KEYS: usize = 10_000;

fn populate_store() -> (BlockHashStore, String, Uuid) {
    let store = BlockHashStore::with_config(StoreConfig {
        node_stale_after: Duration::from_secs(30),
        ttl: Duration::from_secs(7_200),
    });
    let target_node = "target-node:50055".to_string();
    let other_node = "other-node:50055".to_string();
    let target_id = Uuid::new_v4();
    let other_id = Uuid::new_v4();
    store.heartbeat_node(&target_node, target_id).unwrap();
    store.heartbeat_node(&other_node, other_id).unwrap();

    for chunk_start in (0..TOTAL_KEYS).step_by(1_000) {
        let chunk_end = (chunk_start + 1_000).min(TOTAL_KEYS);
        let hashes: Vec<Vec<u8>> = (chunk_start..chunk_end)
            .map(|i| (i as u64).to_le_bytes().to_vec())
            .collect();
        let (node, node_id) = if chunk_start < TARGET_OWNED_KEYS {
            (target_node.as_str(), target_id)
        } else {
            (other_node.as_str(), other_id)
        };
        store
            .insert_hashes("bench", &hashes, node, node_id)
            .unwrap();
    }

    (store, target_node, target_id)
}

fn bench_unregister_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("unregister_node");
    group.sample_size(10);
    group.bench_function("1m_keys_10k_owned", |b| {
        b.iter_batched(
            populate_store,
            |(store, target_node, target_id)| {
                let removed = store.unregister_node(&target_node, target_id).unwrap();
                assert_eq!(removed, TARGET_OWNED_KEYS);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_unregister_node);
criterion_main!(benches);
