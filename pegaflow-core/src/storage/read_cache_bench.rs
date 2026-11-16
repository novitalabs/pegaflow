//! Opt-in CPU metadata benchmark. Run with `cargo test --release -p
//! pegaflow-core --no-default-features --features cuda-13 --lib
//! storage::read_cache_clock::bench::cache_policy_benchmark -- --ignored --nocapture`.
//! Uses real ReadCache operations with empty SealedBlocks: no CUDA allocation,
//! KV payload copies, SSD/RDMA traffic, or model execution is measured.

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use super::ReadCache;
use crate::block::{BlockKey, SealedBlock};

const RESIDENTS: usize = 32_768;
const SAMPLES: usize = 16;
const LOOKUPS: usize = 4096;

fn key(id: usize) -> BlockKey {
    let mut hash = vec![0; 32];
    hash[..8].copy_from_slice(&(id as u64).to_le_bytes());
    BlockKey::new("bench".into(), hash)
}

fn block() -> Arc<SealedBlock> {
    Arc::new(SealedBlock::from_slots(Vec::new()))
}

fn populated(keys: &[BlockKey]) -> ReadCache {
    let cache = ReadCache::new(keys.len(), false, Some(1));
    cache.batch_insert(keys.iter().map(|key| (key.clone(), block())).collect());
    cache
}

fn evict(cache: &ReadCache, count: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    cache.remove_clock_batch(count)
}

fn next_random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as usize
}

fn report(name: &str, units: usize, samples: &[Duration], hits: usize, requested: usize) {
    let mut ns: Vec<_> = samples.iter().map(Duration::as_nanos).collect();
    ns.sort_unstable();
    let percentile = |p: usize| ns[(ns.len() - 1) * p / 100] / units as u128;
    println!(
        "CACHE_BENCH,{name},{},{},{},{},{},{hits},{requested}",
        units * samples.len(),
        ns.iter().sum::<u128>(),
        percentile(50),
        percentile(95),
        percentile(99),
    );
}

fn hits(keys: &[BlockKey], width: usize) {
    let cache = populated(keys);
    let mut random = 42;
    let offsets: Vec<_> = (0..LOOKUPS)
        .map(|_| next_random(&mut random) % (keys.len() / width) * width)
        .collect();
    for &start in &offsets {
        drop(cache.get_prefix_blocks(&keys[start..start + width]));
    }
    let mut samples = Vec::new();
    for _ in 0..SAMPLES {
        let started = Instant::now();
        for &start in &offsets {
            black_box(cache.get_prefix_blocks(&keys[start..start + width]));
        }
        samples.push(started.elapsed());
    }
    report(&format!("hit_prefix_{width}"), LOOKUPS, &samples, 0, 0);
}

fn contended_hits(keys: &[BlockKey]) {
    const THREADS: usize = 4;
    const PER_THREAD: usize = 65_536;
    let cache = populated(keys);
    let ready = Barrier::new(THREADS + 1);
    let start = Barrier::new(THREADS + 1);
    let done = Barrier::new(THREADS + 1);
    let elapsed = std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let cache = &cache;
            let ready = &ready;
            let start = &start;
            let done = &done;
            scope.spawn(move || {
                let mut random = 42 + thread as u64;
                let indices: Vec<_> = (0..PER_THREAD)
                    .map(|_| next_random(&mut random) % keys.len())
                    .collect();
                ready.wait();
                start.wait();
                for index in indices {
                    black_box(cache.get_prefix_blocks(&keys[index..index + 1]));
                }
                done.wait();
            });
        }
        ready.wait();
        let started = Instant::now();
        start.wait();
        done.wait();
        started.elapsed()
    });
    report("hit_contended_4", THREADS * PER_THREAD, &[elapsed], 0, 0);
}

fn parallel_workload(keys: &[BlockKey], threads: usize, hot: bool, mixed: bool) {
    const PER_THREAD: usize = 32_768;
    let cache = populated(keys);
    let ready = Barrier::new(threads + 1);
    let start = Barrier::new(threads + 1);
    let done = Barrier::new(threads + 1);
    let elapsed = std::thread::scope(|scope| {
        for worker in 0..threads {
            let cache = &cache;
            let ready = &ready;
            let start = &start;
            let done = &done;
            scope.spawn(move || {
                let mut random = 42 + worker as u64;
                let offsets: Vec<_> = (0..PER_THREAD)
                    .map(|_| next_random(&mut random) % if hot { 4 } else { keys.len() })
                    .collect();
                ready.wait();
                start.wait();
                for (i, index) in offsets.into_iter().enumerate() {
                    if mixed && i % 32 == 0 {
                        let new_key = key(keys.len() + worker * PER_THREAD + i);
                        cache.batch_insert(vec![(new_key, block())]);
                        black_box(evict(cache, 1));
                    } else {
                        black_box(cache.get_prefix_blocks(&keys[index..index + 1]));
                    }
                }
                done.wait();
            });
        }
        ready.wait();
        let started = Instant::now();
        start.wait();
        done.wait();
        started.elapsed()
    });
    report(
        &format!("parallel_n{}_t{threads}_hot{hot}_mixed{mixed}", keys.len()),
        threads * PER_THREAD,
        &[elapsed],
        0,
        0,
    );
}

fn inserts(size: usize, duplicate: bool) {
    let cache = ReadCache::new(RESIDENTS, false, Some(1));
    let entries: Vec<_> = (0..size).map(|id| (key(id), block())).collect();
    cache.batch_insert_refs(&entries);
    if !duplicate {
        drop(cache.remove_all());
    }
    let mut samples = Vec::new();
    for _ in 0..SAMPLES {
        let started = Instant::now();
        black_box(cache.batch_insert_refs(&entries));
        samples.push(started.elapsed());
        if !duplicate {
            drop(cache.remove_all());
        }
    }
    let mode = if duplicate { "duplicate" } else { "new_pinned" };
    report(&format!("insert_{mode}_{size}"), size, &samples, 0, 0);
}

fn pressure(keys: &[BlockKey], hot: bool, pinned: bool) {
    const BATCH: usize = 256;
    let mut samples = Vec::new();
    for _ in 0..SAMPLES {
        let cache = populated(keys);
        if hot {
            drop(cache.get_prefix_blocks(keys));
        }
        let held = if pinned {
            cache.peek_blocks(&keys[..keys.len() / 2])
        } else {
            Vec::new()
        };
        let started = Instant::now();
        let removed = evict(&cache, BATCH);
        samples.push(started.elapsed());
        assert_eq!(removed.len(), BATCH);
        drop(held);
    }
    report(&format!("evict_hot{hot}_pinned{pinned}"), 1, &samples, 0, 0);
}

fn document(id: usize, blocks: usize, shared: usize) -> Vec<BlockKey> {
    (0..blocks)
        .map(|index| {
            if index < shared {
                key(index)
            } else {
                key(1024 + id * 256 + index)
            }
        })
        .collect()
}

#[test]
#[ignore = "CPU maintenance comparison; run explicitly in release mode on an idle Linux host"]
fn cache_maintenance_benchmark() {
    let keys: Vec<_> = (0..RESIDENTS).map(key).collect();
    for hot in [false, true] {
        let mut samples = Vec::new();
        for _ in 0..SAMPLES {
            let cache = populated(&keys);
            if hot {
                drop(cache.get_prefix_blocks(&keys));
            }
            let started = Instant::now();
            black_box(cache.backup_candidates(u64::MAX, RESIDENTS));
            samples.push(started.elapsed());
        }
        report(&format!("backup_full_hot{hot}"), 1, &samples, 0, 0);
    }
    for mode in ["demote", "cleanup"] {
        let mut samples = Vec::new();
        for _ in 0..SAMPLES {
            let cache = populated(&keys);
            let started = Instant::now();
            if mode == "demote" {
                black_box(
                    cache.mark_reclaimable_keys(&keys, super::DemotionReason::TransferRelease),
                );
            } else {
                assert_eq!(cache.remove_all().len(), RESIDENTS);
            }
            samples.push(started.elapsed());
        }
        report(mode, RESIDENTS, &samples, 0, 0);
    }
    for threads in [1, 2, 4, 8] {
        let cache = populated(&keys);
        drop(cache.get_prefix_blocks(&keys));
        let start = Barrier::new(threads);
        let started = Instant::now();
        let removed: usize = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        let mut count = 0;
                        while count < RESIDENTS / threads {
                            count += evict(&cache, (RESIDENTS / threads - count).min(256)).len();
                        }
                        count
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .sum()
        });
        assert_eq!(removed, RESIDENTS);
        report(
            &format!("evict_parallel_t{threads}"),
            RESIDENTS,
            &[started.elapsed()],
            0,
            0,
        );
    }
}

fn trace(name: &str, capacity: usize, requests: &[Vec<BlockKey>]) {
    const WARMUP: usize = 512;
    let cache = ReadCache::new(capacity, false, Some(1));
    let mut residents = 0;
    let mut hits = 0;
    let mut requested = 0;
    let mut samples = Vec::new();
    for (index, keys) in requests.iter().enumerate() {
        let started = Instant::now();
        let (hit, held) = cache.get_prefix_blocks(keys);
        let suffix = &keys[hit..];
        let new_blocks = cache
            .contains_keys(suffix)
            .iter()
            .filter(|&&hit| !hit)
            .count();
        let entries: Vec<_> = suffix.iter().map(|key| (key.clone(), block())).collect();
        cache.batch_insert_refs(&entries);
        residents += new_blocks;
        drop(entries);
        drop(held);
        // Synthetic replay enforces a block-count capacity at request end.
        // It does not model byte allocation, fragmentation, or in-flight loads.
        if residents > capacity {
            let removed = evict(&cache, residents - capacity);
            assert_eq!(removed.len(), residents - capacity);
            residents -= removed.len();
        }
        if index >= WARMUP {
            samples.push(started.elapsed());
            hits += hit;
            requested += keys.len();
        }
    }
    report(name, 1, &samples, hits, requested);
}

#[test]
#[ignore = "CPU policy comparison; run explicitly in release mode on an idle Linux host"]
#[expect(
    clippy::assertions_on_constants,
    reason = "Reject debug benchmark runs without blocking normal debug test compilation."
)]
fn cache_policy_benchmark() {
    assert!(!cfg!(debug_assertions), "benchmark requires --release");
    println!(
        "CACHE_BENCH_HEADER,scenario,units,total_ns,p50_ns_per_unit,p95_ns_per_unit,p99_ns_per_unit,prefix_hit_blocks,requested_blocks"
    );
    let keys: Vec<_> = (0..RESIDENTS).map(key).collect();
    hits(&keys, 1);
    hits(&keys, 64);
    contended_hits(&keys);
    for threads in [1, 2, 4, 8] {
        parallel_workload(&keys, threads, false, false);
        parallel_workload(&keys, threads, true, false);
        parallel_workload(&keys, threads, false, true);
    }
    let large_keys: Vec<_> = (0..262_144).map(key).collect();
    for threads in [1, 2, 4, 8] {
        parallel_workload(&large_keys, threads, false, false);
    }
    for size in [128, 1024, 8192] {
        inserts(size, false);
        inserts(size, true);
    }
    for (hot, pinned) in [(false, false), (true, false), (false, true)] {
        pressure(&keys, hot, pinned);
    }

    let mut random = 42;
    let shared: Vec<_> = (0..4608)
        .map(|_| {
            let draw = next_random(&mut random);
            let id = if draw % 10 < 8 { draw % 8 } else { draw % 64 };
            document(id, 64, 16)
        })
        .collect();
    trace("trace_shared_prefix", 1024, &shared);

    let multi_turn: Vec<_> = (0..4608)
        .map(|i| document(i % 32 + i / 256 * 32, 8 + (i / 32 % 8 + 1) * 16, 8))
        .collect();
    trace("trace_multi_turn", 2048, &multi_turn);

    let pollution: Vec<_> = (0..4608)
        .map(|i| {
            let id = if i % 5 == 0 {
                64 + i
            } else {
                next_random(&mut random) % 8
            };
            document(id, 64, 8)
        })
        .collect();
    trace("trace_scan_pollution", 1024, &pollution);

    let over_capacity: Vec<_> = (0..4608)
        .map(|_| document(next_random(&mut random) % 64, 64, 8))
        .collect();
    trace("trace_over_capacity", 1024, &over_capacity);
}
