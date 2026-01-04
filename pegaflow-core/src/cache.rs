use ahash::RandomState;
use hashlink::LruCache;
use std::hash::Hash;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::storage::{BlockKey, SealedBlock};

const DEFAULT_BYTES_PER_VALUE: usize = 1024 * 1024; // heuristic: 1MB per block

/// LRU cache with TinyLFU-based admission. Keeps API surface tiny to avoid
/// bloating storage.rs.
pub(crate) struct TinyLfuCache<K, V> {
    lru: LruCache<K, V>,
    freq: Option<TinyLfu>,
}

impl TinyLfuCache<BlockKey, ArcSealedBlock> {
    pub fn new_unbounded(capacity_bytes: usize, enable_lfu_admission: bool) -> Self {
        let estimated_items = std::cmp::max(1, capacity_bytes / DEFAULT_BYTES_PER_VALUE);
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
            max((std::f64::consts::E / error_range).ceil() as usize, 16),
            max((failure_probability.ln() / 0.5f64.ln()).ceil() as usize, 2),
        )
    }

    fn optimal(items: usize) -> Self {
        let (slots, hashes) = Self::optimal_paras(items);
        Self::new(hashes, slots, RandomState::new)
    }

    fn compact(items: usize) -> Self {
        let (slots, hashes) = Self::optimal_paras(items / 100);
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
            self.estimator.age(1); // right shift 1 bit
        }
        self.estimator.incr(key)
    }

    // Because we use 8-bit counters, window size can be 256 * the cache size.
    pub fn new(cache_size: usize) -> Self {
        Self {
            estimator: Estimator::optimal(cache_size),
            window_counter: Default::default(),
            window_limit: cache_size * 8,
        }
    }

    #[allow(dead_code)]
    pub fn new_compact(cache_size: usize) -> Self {
        Self {
            estimator: Estimator::compact(cache_size),
            window_counter: Default::default(),
            window_limit: cache_size * 8,
        }
    }
}
