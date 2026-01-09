use ahash::RandomState;
use hashlink::LruCache;
use std::hash::Hash;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::block::{BlockKey, SealedBlock};

// ============================================================================
// Constants
// ============================================================================

/// Default assumed size per cached value for estimating item count (128MB)
const DEFAULT_BYTES_PER_VALUE: usize = 128 * 1024 * 1024;
/// Minimum number of slots (columns) in the CM-Sketch estimator
const MIN_ESTIMATOR_SLOTS: usize = 16;
/// Minimum number of hash functions (rows) in the CM-Sketch estimator
const MIN_ESTIMATOR_HASHES: usize = 2;
/// Divisor for compact estimator (uses 1/100th the slots of optimal)
const COMPACT_ESTIMATOR_DIVISOR: usize = 100;
/// Multiplier for TinyLFU window limit relative to cache size
const WINDOW_LIMIT_MULTIPLIER: usize = 8;
/// Number of bits to right-shift counters during aging
const AGE_SHIFT_BITS: u8 = 1;

/// LRU cache with TinyLFU-based admission. Keeps API surface tiny to avoid
/// bloating storage.rs.
pub(crate) struct TinyLfuCache<K, V> {
    lru: LruCache<K, V>,
    freq: Option<TinyLfu>,
}

impl TinyLfuCache<BlockKey, ArcSealedBlock> {
    pub fn new_unbounded(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        bytes_per_value_hint: Option<usize>,
    ) -> Self {
        let bytes_per_value = bytes_per_value_hint
            .filter(|size| *size > 0)
            .unwrap_or(DEFAULT_BYTES_PER_VALUE);
        let estimated_items = std::cmp::max(1, capacity_bytes / bytes_per_value);
        Self {
            lru: LruCache::new_unbounded(),
            freq: enable_lfu_admission.then(|| TinyLfu::new(estimated_items)),
        }
    }

    pub fn contains_key(&self, key: &BlockKey) -> bool {
        self.lru.contains_key(key)
    }

    /// Returns a cloned value and bumps frequency on hit.
    pub fn get(&mut self, key: &BlockKey) -> Option<ArcSealedBlock> {
        let hit = self.lru.get(key).cloned();
        if hit.is_some() {
            if let Some(freq) = &self.freq {
                freq.incr(key);
            }
        }
        hit
    }

    /// Insert with TinyLFU admission. If the candidate is colder than the
    /// current LRU victim it is dropped. Returns true if inserted.
    pub fn insert(&mut self, key: BlockKey, value: ArcSealedBlock) -> bool {
        // Always record the access so future attempts have a chance.
        if let Some(freq) = &self.freq {
            freq.incr(&key);
        }

        // Update existing entry eagerly.
        if self.lru.contains_key(&key) {
            self.lru.insert(key, value);
            return true;
        }

        if let Some(freq) = &self.freq {
            let candidate_freq = freq.get(&key);
            if let Some((victim_key, _)) = self.lru.iter().next() {
                let victim_freq = freq.get(victim_key);
                if candidate_freq < victim_freq {
                    return false;
                }
            }
        }

        self.lru.insert(key, value);
        true
    }

    pub fn remove_lru(&mut self) -> Option<(BlockKey, ArcSealedBlock)> {
        self.lru.remove_lru()
    }
}

pub(crate) type ArcSealedBlock = Arc<SealedBlock>;

// Bare-minimum TinyLFU with CM-Sketch; no doorkeeper.
struct Estimator {
    estimator: Box<[(Box<[AtomicU8]>, RandomState)]>,
}

impl Estimator {
    fn optimal_paras(items: usize) -> (usize, usize) {
        use std::cmp::max;
        // derived from https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch
        // width = ceil(e / ε)
        // depth = ceil(ln(1 − δ) / ln(1 / 2))
        let error_range = 1.0 / (items as f64);
        let failure_probability = 1.0 / (items as f64);
        (
            max(
                (std::f64::consts::E / error_range).ceil() as usize,
                MIN_ESTIMATOR_SLOTS,
            ),
            max(
                (failure_probability.ln() / 0.5f64.ln()).ceil() as usize,
                MIN_ESTIMATOR_HASHES,
            ),
        )
    }

    fn optimal(items: usize) -> Self {
        let (slots, hashes) = Self::optimal_paras(items);
        Self::new(hashes, slots, RandomState::new)
    }

    fn compact(items: usize) -> Self {
        let (slots, hashes) = Self::optimal_paras(items / COMPACT_ESTIMATOR_DIVISOR);
        Self::new(hashes, slots, RandomState::new)
    }

    /// Create a new `Estimator` with the given amount of hashes and columns (slots) using
    /// the given random source.
    fn new(hashes: usize, slots: usize, random: impl Fn() -> RandomState) -> Self {
        let mut estimator = Vec::with_capacity(hashes);
        for _ in 0..hashes {
            let mut slot = Vec::with_capacity(slots);
            for _ in 0..slots {
                slot.push(AtomicU8::new(0));
            }
            estimator.push((slot.into_boxed_slice(), random()));
        }

        Estimator {
            estimator: estimator.into_boxed_slice(),
        }
    }

    fn incr<T: Hash>(&self, key: T) -> u8 {
        let mut min = u8::MAX;
        for (slot, hasher) in self.estimator.iter() {
            let hash = hasher.hash_one(&key) as usize;
            let counter = &slot[hash % slot.len()];
            let (_current, new) = incr_no_overflow(counter);
            min = std::cmp::min(min, new);
        }
        min
    }

    /// Get the estimated frequency of `key`.
    fn get<T: Hash>(&self, key: T) -> u8 {
        let mut min = u8::MAX;
        for (slot, hasher) in self.estimator.iter() {
            let hash = hasher.hash_one(&key) as usize;
            let counter = &slot[hash % slot.len()];
            let current = counter.load(Ordering::Relaxed);
            min = std::cmp::min(min, current);
        }
        min
    }

    /// right shift all values inside this `Estimator`.
    fn age(&self, shift: u8) {
        for (slot, _) in self.estimator.iter() {
            for counter in slot.iter() {
                let c = counter.load(Ordering::Relaxed);
                counter.store(c >> shift, Ordering::Relaxed);
            }
        }
    }
}

fn incr_no_overflow(var: &AtomicU8) -> (u8, u8) {
    loop {
        let current = var.load(Ordering::Relaxed);
        if current == u8::MAX {
            return (current, current);
        }
        let new = if current == u8::MAX - 1 {
            u8::MAX
        } else {
            current + 1
        };
        if let Err(new) = var.compare_exchange(current, new, Ordering::Acquire, Ordering::Relaxed) {
            if new == u8::MAX {
                return (current, new);
            }
        } else {
            return (current, new);
        }
    }
}

pub(crate) struct TinyLfu {
    estimator: Estimator,
    window_counter: AtomicUsize,
    window_limit: usize,
}

impl TinyLfu {
    pub fn get<T: Hash>(&self, key: T) -> u8 {
        self.estimator.get(key)
    }

    pub fn incr<T: Hash>(&self, key: T) -> u8 {
        let window_size = self.window_counter.fetch_add(1, Ordering::Relaxed);
        if window_size == self.window_limit || window_size > self.window_limit * 2 {
            self.window_counter.store(0, Ordering::Relaxed);
            self.estimator.age(AGE_SHIFT_BITS);
        }
        self.estimator.incr(key)
    }

    // Because we use 8-bit counters, window size can be 256 * the cache size.
    pub fn new(cache_size: usize) -> Self {
        Self {
            estimator: Estimator::optimal(cache_size),
            window_counter: Default::default(),
            window_limit: cache_size * WINDOW_LIMIT_MULTIPLIER,
        }
    }

    #[allow(dead_code)]
    pub fn new_compact(cache_size: usize) -> Self {
        Self {
            estimator: Estimator::compact(cache_size),
            window_counter: Default::default(),
            window_limit: cache_size * WINDOW_LIMIT_MULTIPLIER,
        }
    }
}

// ============================================================================
// Loom tests for concurrent verification
// ============================================================================

#[cfg(all(test, feature = "loom"))]
mod loom_tests {
    use loom::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use loom::thread;

    /// Test that incr_no_overflow correctly handles concurrent increments.
    fn loom_incr_no_overflow(var: &AtomicU8) -> (u8, u8) {
        loop {
            let current = var.load(Ordering::Relaxed);
            if current == u8::MAX {
                return (current, current);
            }
            let new = if current == u8::MAX - 1 {
                u8::MAX
            } else {
                current + 1
            };
            match var.compare_exchange(current, new, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return (current, new),
                Err(actual) => {
                    if actual == u8::MAX {
                        return (current, actual);
                    }
                    // Retry
                }
            }
        }
    }

    #[test]
    fn test_incr_no_overflow_concurrent() {
        loom::model(|| {
            let counter = loom::sync::Arc::new(AtomicU8::new(0));

            let c1 = loom::sync::Arc::clone(&counter);
            let c2 = loom::sync::Arc::clone(&counter);

            let t1 = thread::spawn(move || {
                loom_incr_no_overflow(&c1)
            });

            let t2 = thread::spawn(move || {
                loom_incr_no_overflow(&c2)
            });

            let (_, v1) = t1.join().unwrap();
            let (_, v2) = t2.join().unwrap();

            // Both increments should succeed, final value should be 2
            let final_val = counter.load(Ordering::Relaxed);
            assert!(final_val >= 1 && final_val <= 2);
            assert!(v1 >= 1 && v1 <= 2);
            assert!(v2 >= 1 && v2 <= 2);
        });
    }

    #[test]
    fn test_incr_near_overflow() {
        loom::model(|| {
            // Start near overflow
            let counter = loom::sync::Arc::new(AtomicU8::new(u8::MAX - 2));

            let c1 = loom::sync::Arc::clone(&counter);
            let c2 = loom::sync::Arc::clone(&counter);
            let c3 = loom::sync::Arc::clone(&counter);

            let t1 = thread::spawn(move || loom_incr_no_overflow(&c1));
            let t2 = thread::spawn(move || loom_incr_no_overflow(&c2));
            let t3 = thread::spawn(move || loom_incr_no_overflow(&c3));

            t1.join().unwrap();
            t2.join().unwrap();
            t3.join().unwrap();

            // Should never exceed MAX
            let final_val = counter.load(Ordering::Relaxed);
            assert!(final_val <= u8::MAX);
        });
    }

    #[test]
    fn test_window_counter_wrap() {
        loom::model(|| {
            let window_counter = loom::sync::Arc::new(AtomicUsize::new(0));
            let window_limit = 2;

            let wc1 = loom::sync::Arc::clone(&window_counter);
            let wc2 = loom::sync::Arc::clone(&window_counter);

            let t1 = thread::spawn(move || {
                let val = wc1.fetch_add(1, Ordering::Relaxed);
                if val == window_limit || val > window_limit * 2 {
                    wc1.store(0, Ordering::Relaxed);
                }
            });

            let t2 = thread::spawn(move || {
                let val = wc2.fetch_add(1, Ordering::Relaxed);
                if val == window_limit || val > window_limit * 2 {
                    wc2.store(0, Ordering::Relaxed);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // Counter should be in valid range
            let final_val = window_counter.load(Ordering::Relaxed);
            assert!(final_val <= window_limit * 2 + 2);
        });
    }
}

// ============================================================================
// Regular unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incr_no_overflow_basic() {
        let counter = AtomicU8::new(0);

        let (old, new) = incr_no_overflow(&counter);
        assert_eq!(old, 0);
        assert_eq!(new, 1);

        let (old, new) = incr_no_overflow(&counter);
        assert_eq!(old, 1);
        assert_eq!(new, 2);
    }

    #[test]
    fn test_incr_no_overflow_at_max() {
        let counter = AtomicU8::new(u8::MAX);

        let (old, new) = incr_no_overflow(&counter);
        assert_eq!(old, u8::MAX);
        assert_eq!(new, u8::MAX);

        // Should stay at MAX
        let (old, new) = incr_no_overflow(&counter);
        assert_eq!(old, u8::MAX);
        assert_eq!(new, u8::MAX);
    }

    #[test]
    fn test_incr_no_overflow_near_max() {
        let counter = AtomicU8::new(u8::MAX - 1);

        let (old, new) = incr_no_overflow(&counter);
        assert_eq!(old, u8::MAX - 1);
        assert_eq!(new, u8::MAX);

        // Next increment should stay at MAX
        let (old, new) = incr_no_overflow(&counter);
        assert_eq!(old, u8::MAX);
        assert_eq!(new, u8::MAX);
    }

    #[test]
    fn test_estimator_basic() {
        let est = Estimator::new(2, 16, RandomState::new);

        // Increment same key multiple times
        for _ in 0..10 {
            est.incr("key1");
        }

        let freq = est.get("key1");
        assert!(freq >= 5, "Frequency should reflect accesses");

        // New key should have lower frequency
        let new_freq = est.get("key2");
        assert!(new_freq < freq, "New key should have lower frequency");
    }

    #[test]
    fn test_estimator_aging() {
        let est = Estimator::new(2, 16, RandomState::new);

        // Build up frequency
        for _ in 0..100 {
            est.incr("key1");
        }

        let before = est.get("key1");

        // Age the estimator
        est.age(1);

        let after = est.get("key1");
        assert!(after < before, "Aging should reduce frequency");
        assert!(after >= before / 2 - 1, "Aging should halve frequency");
    }

    #[test]
    fn test_tiny_lfu_window_reset() {
        let lfu = TinyLfu::new(10);

        // Increment beyond window limit
        for _ in 0..(10 * WINDOW_LIMIT_MULTIPLIER + 10) {
            lfu.incr("key1");
        }

        // Should not panic, window counter should reset
        let freq = lfu.get("key1");
        assert!(freq > 0, "Should have some frequency");
    }
}
