mod result;
mod scheduler;
mod task;

use std::sync::Arc;
use std::time::Duration;

use crate::block::{BlockKey, PrefetchStatus, SealedBlock};
use crate::storage::backing_tier::{BackingTier, FakeTier};
use crate::storage::read_cache::ReadCache;

use super::{PrefixScan, Scheduler};

const READY_TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_BYTES: usize = 64 * 1024 * 1024;

fn key(n: u8) -> BlockKey {
    BlockKey::new("ns".to_string(), vec![n])
}

fn hashes(n: u8) -> Vec<Vec<u8>> {
    (0..n).map(|i| vec![i]).collect()
}

fn block() -> Arc<SealedBlock> {
    Arc::new(SealedBlock::from_slots(Vec::new()))
}

fn keyed_blocks(n: u8) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    (0..n).map(|i| (key(i), block())).collect()
}

fn untracked(fake: FakeTier) -> BackingTier {
    BackingTier::Fake(Arc::new(fake))
}

fn tracked(fake: FakeTier) -> (Arc<FakeTier>, BackingTier) {
    let fake = Arc::new(fake);
    let tier = BackingTier::Fake(Arc::clone(&fake));
    (fake, tier)
}

struct Harness {
    scheduler: Scheduler,
    cache: ReadCache,
}

impl Harness {
    fn new(tiers: Vec<BackingTier>, max_prefetch_blocks: usize) -> Self {
        Self {
            scheduler: Scheduler::with_tiers(tiers, max_prefetch_blocks),
            cache: ReadCache::new(CACHE_BYTES, false, None),
        }
    }

    async fn await_ready(&self, req_id: &str, hashes: &[Vec<u8>]) -> PrefetchStatus {
        let poll_loop = async {
            loop {
                let status = self
                    .scheduler
                    .query(&self.cache, PrefixScan::new(req_id, "ns", hashes))
                    .await;
                if matches!(status, PrefetchStatus::Ready { .. }) {
                    return status;
                }
                tokio::task::yield_now().await;
            }
        };
        match tokio::time::timeout(READY_TIMEOUT, poll_loop).await {
            Ok(status) => status,
            Err(_) => panic!(
                "await_ready timed out after {:?}: req_id={} active_len={}",
                READY_TIMEOUT,
                req_id,
                self.scheduler.state.lock().active_len()
            ),
        }
    }
}
