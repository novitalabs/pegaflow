//! Microbench for `unregister_node` synchronous owner cleanup.
//!
//! Reference numbers on h20:
//!   cargo bench -p pegaflow-metaserver --bench unregister_node -- --sample-size 10
//!   1m_keys_10k_owned: 315.95 / 320.52 / 326.10 ms
//!
//! `remove_node_owners` does a full `blocks.retain`, so cost is O(total_blocks),
//! not O(owned). At roughly 3ms per 10k total blocks on h20, 10M total blocks
//! would sit close to the 3s client-side unregister timeout. If that threshold
//! becomes realistic, unregister should return after dropping the node record and
//! leave owner cleanup to the lifecycle sweep.

use criterion::{Criterion, criterion_group, criterion_main};
use pegaflow_common::hll::HllWindowSnapshot;
use pegaflow_metaserver::hll::{HllNodeReport, HllSchema};
use pegaflow_metaserver::store::{BlockHashStore, StoreConfig};
use std::time::Duration;
use uuid::Uuid;

const TOTAL_KEYS: usize = 1_000_000;
const TARGET_OWNED_KEYS: usize = 10_000;

fn empty_hll_report() -> HllNodeReport {
    HllNodeReport {
        windows: vec![HllWindowSnapshot {
            window: "15m".into(),
            window_secs: 900,
            bucket_bits: 4,
            registers: vec![0; 16],
            total_requests: 0,
        }],
    }
}

fn populate_store() -> (BlockHashStore, String, Uuid) {
    let schema = HllSchema::new(vec![("15m".into(), Duration::from_secs(900))], 4).unwrap();
    let store = BlockHashStore::with_config_and_hll_schema(
        StoreConfig {
            node_stale_after: Duration::from_secs(30),
            ttl: Duration::from_secs(7_200),
        },
        schema,
    );
    let target_node = "target-node:50055".to_string();
    let other_node = "other-node:50055".to_string();
    let target_id = Uuid::new_v4();
    let other_id = Uuid::new_v4();
    store
        .heartbeat_node(&target_node, target_id, empty_hll_report())
        .unwrap();
    store
        .heartbeat_node(&other_node, other_id, empty_hll_report())
        .unwrap();

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
