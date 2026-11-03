use std::{collections::HashMap, fmt, str::FromStr, sync::Arc};

use crate::block::{BlockKey, SealedBlock};
use crate::metrics::{DEMOTION_BACKUP_HINT, DEMOTION_RECLAIM_HINT, DEMOTION_TRANSFER};

use super::read_cache_clock;
use super::read_cache_lru;

/// Why resident blocks moved from retained to reclaimable.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum DemotionReason {
    /// A peer finished copying the blocks from this node.
    TransferRelease,
    /// The MetaServer reported enough other owners at registration.
    ReclaimHint,
    /// The MetaServer reported a live decode backup for reported candidates.
    BackupHint,
}

impl DemotionReason {
    pub(crate) fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::TransferRelease => &*DEMOTION_TRANSFER,
            Self::ReclaimHint => &*DEMOTION_RECLAIM_HINT,
            Self::BackupHint => &*DEMOTION_BACKUP_HINT,
        }
    }
}

/// Default reclaimable share of capacity kept ready by decode backups.
pub(crate) const DEFAULT_BACKUP_RECLAIMABLE_PERCENT: u64 = 10;

/// Host-memory resident cache eviction policy, selected when the engine starts.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Keep the upstream least-recently-used replacement policy.
    #[default]
    Lru,
    /// Use FIFO queues with a CLOCK second chance for recently accessed blocks.
    ClockFifo,
}

impl FromStr for EvictionPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "lru" => Ok(Self::Lru),
            "clock-fifo" => Ok(Self::ClockFifo),
            other => Err(format!(
                "unknown eviction policy {other:?}; expected lru or clock-fifo"
            )),
        }
    }
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lru => formatter.write_str("lru"),
            Self::ClockFifo => formatter.write_str("clock-fifo"),
        }
    }
}

/// Static dispatch over the two resident-cache implementations. The default is
/// the upstream LRU backend; CLOCK-FIFO is only used when explicitly selected.
pub(crate) enum ReadCache {
    Lru(read_cache_lru::ReadCache),
    ClockFifo(read_cache_clock::ReadCache),
}

impl ReadCache {
    #[cfg(test)]
    pub(crate) fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
    ) -> Self {
        Self::with_policy(
            capacity_bytes,
            enable_lfu_admission,
            value_size_hint,
            EvictionPolicy::Lru,
        )
    }

    pub(crate) fn with_policy(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
        policy: EvictionPolicy,
    ) -> Self {
        match policy {
            EvictionPolicy::Lru => Self::Lru(read_cache_lru::ReadCache::new(
                capacity_bytes,
                enable_lfu_admission,
                value_size_hint,
            )),
            EvictionPolicy::ClockFifo => Self::ClockFifo(read_cache_clock::ReadCache::new(
                capacity_bytes,
                enable_lfu_admission,
                value_size_hint,
            )),
        }
    }

    pub(crate) fn policy(&self) -> EvictionPolicy {
        match self {
            Self::Lru(_) => EvictionPolicy::Lru,
            Self::ClockFifo(_) => EvictionPolicy::ClockFifo,
        }
    }

    pub(crate) fn with_backup_reclaimable_percent(self, percent: u64) -> Self {
        match self {
            Self::Lru(cache) => Self::Lru(cache.with_backup_reclaimable_percent(percent)),
            Self::ClockFifo(cache) => {
                Self::ClockFifo(cache.with_backup_reclaimable_percent(percent))
            }
        }
    }

    pub(super) fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        match self {
            Self::Lru(cache) => cache.contains_keys(keys),
            Self::ClockFifo(cache) => cache.contains_keys(keys),
        }
    }

    pub(super) fn peek_blocks(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        match self {
            Self::Lru(cache) => cache.peek_blocks(keys),
            Self::ClockFifo(cache) => cache.peek_blocks(keys),
        }
    }

    pub(super) fn get_prefix_blocks(&self, keys: &[BlockKey]) -> (usize, Vec<Arc<SealedBlock>>) {
        match self {
            Self::Lru(cache) => cache.get_prefix_blocks(keys),
            Self::ClockFifo(cache) => cache.get_prefix_blocks(keys),
        }
    }

    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        match self {
            Self::Lru(cache) => cache.get_blocks(keys),
            Self::ClockFifo(cache) => cache.get_blocks(keys),
        }
    }

    pub(super) fn get_blocks_aligned(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        match self {
            Self::Lru(cache) => cache.get_blocks_aligned(keys),
            Self::ClockFifo(cache) => cache.get_blocks_aligned(keys),
        }
    }

    pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
        match self {
            Self::Lru(cache) => cache.batch_insert(blocks),
            Self::ClockFifo(cache) => cache.batch_insert(blocks),
        }
    }

    pub(super) fn batch_insert_refs(
        &self,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) -> Vec<BlockKey> {
        match self {
            Self::Lru(cache) => cache.batch_insert_refs(blocks),
            Self::ClockFifo(cache) => cache.batch_insert_refs(blocks),
        }
    }

    pub(super) fn resident_generations(&self, keys: &[BlockKey]) -> Vec<(BlockKey, u64)> {
        match self {
            Self::Lru(cache) => cache.resident_generations(keys),
            Self::ClockFifo(cache) => cache.resident_generations(keys),
        }
    }

    /// Evict up to `batch_size` residents using the selected backend's policy.
    pub(super) fn evict_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        match self {
            Self::Lru(cache) => cache.remove_lru_batch(batch_size),
            Self::ClockFifo(cache) => cache.remove_clock_batch(batch_size),
        }
    }

    pub(super) fn remove_all(&self) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        match self {
            Self::Lru(cache) => cache.remove_all(),
            Self::ClockFifo(cache) => cache.remove_all(),
        }
    }

    pub(crate) fn mark_reclaimable_keys(&self, keys: &[BlockKey], reason: DemotionReason) -> usize {
        match self {
            Self::Lru(cache) => cache.mark_reclaimable_keys(keys, reason),
            Self::ClockFifo(cache) => cache.mark_reclaimable_keys(keys, reason),
        }
    }

    pub(crate) fn mark_reclaimable_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        reason: DemotionReason,
    ) -> usize {
        match self {
            Self::Lru(cache) => cache.mark_reclaimable_hashes(namespace, hashes, reason),
            Self::ClockFifo(cache) => cache.mark_reclaimable_hashes(namespace, hashes, reason),
        }
    }

    pub(crate) fn mark_reclaimable_hashes_if_generation(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        generations: &HashMap<Vec<u8>, u64>,
    ) {
        match self {
            Self::Lru(cache) => {
                cache.mark_reclaimable_hashes_if_generation(namespace, hashes, generations)
            }
            Self::ClockFifo(cache) => {
                cache.mark_reclaimable_hashes_if_generation(namespace, hashes, generations)
            }
        }
    }

    pub(crate) fn backup_report_budget(&self) -> u64 {
        match self {
            Self::Lru(cache) => cache.backup_report_budget(),
            Self::ClockFifo(cache) => cache.backup_report_budget(),
        }
    }

    pub(crate) fn backup_candidates(
        &self,
        max_bytes: u64,
        max_blocks: usize,
    ) -> Vec<(BlockKey, u64)> {
        match self {
            Self::Lru(cache) => cache.backup_candidates(max_bytes, max_blocks),
            Self::ClockFifo(cache) => cache.backup_candidates(max_bytes, max_blocks),
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_retained_for_test(&self, key: BlockKey, block: Arc<SealedBlock>) {
        match self {
            Self::Lru(cache) => cache.insert_retained_for_test(key, block),
            Self::ClockFifo(cache) => cache.insert_retained_for_test(key, block),
        }
    }

    #[cfg(test)]
    pub(crate) fn resident_generation_for_test(&self, key: &BlockKey) -> u64 {
        match self {
            Self::Lru(cache) => cache.resident_generation_for_test(key),
            Self::ClockFifo(cache) => cache.resident_generation_for_test(key),
        }
    }

    #[cfg(test)]
    pub(crate) fn remove_lru_batch_for_test(&self, batch_size: usize) {
        match self {
            Self::Lru(cache) => cache.remove_lru_batch_for_test(batch_size),
            Self::ClockFifo(cache) => cache.evict_batch_for_test(batch_size),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_reclaimable_for_test(&self, key: &BlockKey) -> bool {
        match self {
            Self::Lru(cache) => cache.is_reclaimable_for_test(key),
            Self::ClockFifo(cache) => cache.is_reclaimable_for_test(key),
        }
    }
}
