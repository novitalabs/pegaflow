//! Common test utilities for pegaflow-core tests.

pub mod config;

use std::sync::Arc;

/// Skip test if CUDA is not available.
///
/// Usage:
/// ```ignore
/// #[test]
/// fn test_something() {
///     skip_without_cuda!();
///     // ... rest of test
/// }
/// ```
#[macro_export]
macro_rules! skip_without_cuda {
    () => {
        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }
    };
}

/// Skip test if not running on Linux (required for io_uring).
#[macro_export]
macro_rules! skip_without_linux {
    () => {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!("Skipping test: Linux required for io_uring");
            return;
        }
    };
}

/// Generate a random block hash for testing.
pub fn random_block_hash() -> Vec<u8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut hash = vec![0u8; 32];
    rng.fill(&mut hash[..]);
    hash
}

/// Generate multiple random block hashes.
pub fn random_block_hashes(count: usize) -> Vec<Vec<u8>> {
    (0..count).map(|_| random_block_hash()).collect()
}

/// Generate a deterministic block hash from an index (for reproducible tests).
pub fn block_hash_from_index(index: usize) -> Vec<u8> {
    let mut hash = vec![0u8; 32];
    let bytes = index.to_le_bytes();
    hash[..bytes.len()].copy_from_slice(&bytes);
    hash
}

/// Test namespace for isolation.
pub const TEST_NAMESPACE: &str = "test_instance";

/// Barrier helper for coordinating multiple threads.
pub struct TestBarrier {
    barrier: std::sync::Barrier,
}

impl TestBarrier {
    pub fn new(n: usize) -> Self {
        Self {
            barrier: std::sync::Barrier::new(n),
        }
    }

    pub fn wait(&self) {
        self.barrier.wait();
    }
}

/// Arc wrapper for sharing TestBarrier across threads.
pub type SharedBarrier = Arc<TestBarrier>;

/// Create a shared barrier for N threads.
pub fn shared_barrier(n: usize) -> SharedBarrier {
    Arc::new(TestBarrier::new(n))
}

/// Helper to run concurrent operations and collect results.
pub fn run_concurrent<T, F>(num_threads: usize, f: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(usize) -> T + Send + Sync + 'static,
{
    let f = Arc::new(f);
    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let f = Arc::clone(&f);
            std::thread::spawn(move || f(i))
        })
        .collect();

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

/// Helper to run concurrent operations with a barrier for synchronization.
pub fn run_concurrent_with_barrier<T, F>(num_threads: usize, f: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(usize, &TestBarrier) -> T + Send + Sync + 'static,
{
    let f = Arc::new(f);
    let barrier = shared_barrier(num_threads);

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let f = Arc::clone(&f);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || f(i, &barrier))
        })
        .collect();

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_block_hash() {
        let hash1 = random_block_hash();
        let hash2 = random_block_hash();
        assert_eq!(hash1.len(), 32);
        assert_eq!(hash2.len(), 32);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_block_hash_from_index() {
        let hash0 = block_hash_from_index(0);
        let hash1 = block_hash_from_index(1);
        let hash0_again = block_hash_from_index(0);

        assert_eq!(hash0.len(), 32);
        assert_ne!(hash0, hash1);
        assert_eq!(hash0, hash0_again);
    }

    #[test]
    fn test_run_concurrent() {
        let results = run_concurrent(4, |i| i * 2);
        assert_eq!(results.len(), 4);
        // Results may be in any order
        let mut sorted = results.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 2, 4, 6]);
    }
}
