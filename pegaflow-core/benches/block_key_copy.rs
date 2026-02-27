use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pegaflow_core::block::BlockKey;

const HASH_BYTES: usize = 64; // SHA256
const PAGE_SIZES: [usize; 4] = [32, 64, 128, 256];
const TOKEN_COUNTS: [usize; 4] = [4_096, 65_536, 131_072, 262_144];

fn make_hashes(num_blocks: usize) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|i| {
            let mut h = vec![0u8; HASH_BYTES];
            let bytes = i.to_le_bytes();
            h[..bytes.len()].copy_from_slice(&bytes);
            h
        })
        .collect()
}

fn block_key_copy_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_key_copy");

    for &page_size in &PAGE_SIZES {
        for &num_tokens in &TOKEN_COUNTS {
            let num_blocks = num_tokens / page_size;
            let hashes = make_hashes(num_blocks);
            let namespace = "deepseek-ai/DeepSeek-V3";

            group.bench_function(
                BenchmarkId::new(
                    "clone_collect",
                    format!("{num_tokens}tok_page{page_size}_{num_blocks}blk"),
                ),
                |b| {
                    b.iter(|| {
                        let keys: Vec<BlockKey> = hashes
                            .iter()
                            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
                            .collect();
                        std::hint::black_box(&keys);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, block_key_copy_benchmarks);
criterion_main!(benches);
