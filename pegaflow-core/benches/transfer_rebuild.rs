//! Reproduces the cross-node `rebuild_sealed_blocks` cost observed on the wire.
//!
//! The p2p stage breakdown showed rebuild dominating the small-block regime
//! (~359 ms for 4500 blocks, vs ~73 ms RDMA wait), scaling with cell count
//! (`blocks * slots * 2`), not bytes. This bench rebuilds vLLM-shaped plans
//! (`slots = layers*tp = 288`, split K/V) so the cost can be profiled locally
//! with no GPU or RDMA.
//!
//! Run:
//!   cargo bench -p pegaflow-core --no-default-features \
//!       --features bench,cuda-13,rdma --bench transfer_rebuild
//! Profile (flamegraph / perf) the same target binary under `--profile-time`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pegaflow_core::bench_support::RebuildFixture;

fn bench_rebuild(c: &mut Criterion) {
    // vLLM-shaped: slots = layers(36) * tp(8) = 288, split K/V (2 segments).
    // Cost is count-driven, so tiny segment bytes keep the slab small.
    const SLOTS: usize = 288;
    const SEG_BYTES: usize = 16;

    let mut group = c.benchmark_group("rebuild_sealed_blocks");
    // 70 = 256 KiB load, 281 = 64 KiB load, 4500 = 4 KiB load (per the p2p A/B).
    for &blocks in &[70usize, 281, 4500] {
        let fixture = RebuildFixture::new(blocks, SLOTS, SEG_BYTES);
        let (placements, chunks, cells) = fixture.stats();
        eprintln!(
            "shape blocks={blocks} slots={SLOTS} segs=2 cells={cells} \
             placements={placements} chunks={chunks}"
        );
        group.throughput(Throughput::Elements(cells as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(blocks),
            &fixture,
            |b, fixture| b.iter(|| black_box(fixture.rebuild())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_rebuild);
criterion_main!(benches);
