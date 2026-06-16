//! Isolated profiling driver for `rebuild_sealed_blocks`.
//!
//! Builds one vLLM-shaped fixture (slots = layers*tp = 288, split K/V) once,
//! then rebuilds it in a tight loop so a sampling profiler (samply / perf) sees
//! essentially only the rebuild hot path -- no criterion, no encode, no
//! fixture setup in the steady state.
//!
//!   cargo build --example rebuild_profile --release --no-default-features \
//!       --features bench,cuda-13,rdma
//!   samply record -- target/release/examples/rebuild_profile

use std::hint::black_box;
use std::time::Instant;

use pegaflow_core::bench_support::RebuildFixture;

const BLOCKS: usize = 4500; // 4 KiB load (the regime where rebuild dominates)
const SLOTS: usize = 288; // layers(36) * tp(8)
const SEG_BYTES: usize = 16; // cost is count-driven, not byte-driven
const ITERS: usize = 40;

fn main() {
    let fixture = RebuildFixture::new(BLOCKS, SLOTS, SEG_BYTES);
    let (placements, chunks, cells) = fixture.stats();
    eprintln!(
        "fixture blocks={BLOCKS} slots={SLOTS} cells={cells} placements={placements} chunks={chunks}; looping {ITERS}x"
    );
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..ITERS {
        acc += black_box(fixture.rebuild());
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    eprintln!("mean rebuild = {ms:.2} ms/iter (acc={acc})");
}
