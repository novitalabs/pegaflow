// Transfer lock manager: prevents LRU eviction of blocks during cross-node
// RDMA transfer by holding Arc<SealedBlock> references. When the TinyLFU cache
// evicts a key, the pinned memory stays allocated as long as this lock holds an Arc.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, info, warn};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::block::{BlockKey, SealedBlock};
use crate::metrics::core_metrics;

struct TransferSession {
    blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    created_at: Instant,
    requester_id: String,
}

pub(crate) struct TransferLockManager {
    inner: Mutex<HashMap<String, TransferSession>>,
    lock_timeout: Duration,
}

impl TransferLockManager {
    pub(crate) fn new(lock_timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            lock_timeout,
        }
    }

    /// Lock blocks for a transfer session. Returns the session ID.
    ///
    /// The caller must later call `release()` to free the locks. If the caller
    /// crashes, `gc_expired()` will auto-release after `lock_timeout`.
    pub(crate) fn lock_blocks(
        &self,
        requester_id: &str,
        blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> String {
        let session_id = Uuid::new_v4().to_string();
        let block_count = blocks.len();

        let mut inner = self.inner.lock();
        inner.insert(
            session_id.clone(),
            TransferSession {
                blocks,
                created_at: Instant::now(),
                requester_id: requester_id.to_string(),
            },
        );

        core_metrics()
            .transfer_lock_active
            .add(block_count as i64, &[]);
        debug!(
            "Transfer lock acquired: session={} requester={} blocks={}",
            session_id, requester_id, block_count
        );

        session_id
    }

    /// Take a session's blocks out of the manager for a push transfer.
    /// The caller's `Arc`s keep the blocks alive while the RDMA WRITE runs;
    /// the lock is gone once the returned Vec is dropped.
    ///
    /// # Security model
    ///
    /// Any caller with the session ID can take the session. This relies on:
    /// 1. Session IDs are UUIDv4 (cryptographically random, unguessable)
    /// 2. The gRPC port is network-isolated (internal cluster only)
    #[cfg(any(feature = "rdma", test))]
    pub(crate) fn take(&self, session_id: &str) -> Option<Vec<(BlockKey, Arc<SealedBlock>)>> {
        let session = self.inner.lock().remove(session_id)?;
        core_metrics()
            .transfer_lock_active
            .add(-(session.blocks.len() as i64), &[]);
        debug!(
            "Transfer lock taken for push: session={} requester={} blocks={}",
            session_id,
            session.requester_id,
            session.blocks.len()
        );
        Some(session.blocks)
    }

    /// Garbage-collect expired sessions. Returns the number of sessions removed.
    pub(crate) fn gc_expired(&self) -> usize {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let timeout = self.lock_timeout;

        let expired: Vec<String> = inner
            .iter()
            .filter(|(_, session)| now.duration_since(session.created_at) > timeout)
            .map(|(id, _)| id.clone())
            .collect();

        let mut expired_count = 0;
        let mut expired_blocks = 0usize;
        for id in &expired {
            if let Some(session) = inner.remove(id) {
                expired_blocks += session.blocks.len();
                expired_count += 1;
                warn!(
                    "Transfer lock expired: session={} requester={} blocks={} age={:?}",
                    id,
                    session.requester_id,
                    session.blocks.len(),
                    now.duration_since(session.created_at),
                );
            }
        }

        if expired_count > 0 {
            core_metrics()
                .transfer_lock_active
                .add(-(expired_blocks as i64), &[]);
            core_metrics()
                .transfer_lock_timeouts_total
                .add(expired_count as u64, &[]);
            info!(
                "Transfer lock GC: expired {} sessions ({} blocks)",
                expired_count, expired_blocks
            );
        }

        expired_count
    }

    /// Number of active transfer sessions.
    #[cfg(test)]
    fn active_session_count(&self) -> usize {
        self.inner.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_block() -> (BlockKey, Arc<SealedBlock>) {
        let key = BlockKey::new("ns".into(), vec![1, 2, 3]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        (key, block)
    }

    #[test]
    fn lock_then_take_consumes_session() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));
        let (key, block) = make_test_block();

        let session_id = mgr.lock_blocks("node-a", vec![(key, block.clone())]);
        assert_eq!(mgr.active_session_count(), 1);
        // The lock holds an Arc clone in addition to our local one.
        assert_eq!(Arc::strong_count(&block), 2);

        let taken = mgr.take(&session_id).expect("session should exist");
        assert_eq!(taken.len(), 1);
        assert_eq!(mgr.active_session_count(), 0);

        // Take is consuming: second take and unknown ids return None.
        assert!(mgr.take(&session_id).is_none());
        assert!(mgr.take("nonexistent").is_none());

        // Dropping the taken blocks drops the last lock reference.
        drop(taken);
        assert_eq!(Arc::strong_count(&block), 1);
    }

    #[test]
    fn gc_only_removes_expired_sessions() {
        let mgr = TransferLockManager::new(Duration::from_millis(10));
        let (key1, block1) = make_test_block();

        // Session 1: will expire
        let _s1 = mgr.lock_blocks("node-a", vec![(key1, block1)]);
        assert_eq!(mgr.gc_expired(), 0);

        std::thread::sleep(Duration::from_millis(20));

        // Session 2: fresh, should NOT expire
        let key2 = BlockKey::new("ns".into(), vec![7, 8, 9]);
        let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));
        let s2 = mgr.lock_blocks("node-b", vec![(key2, block2)]);

        assert_eq!(mgr.gc_expired(), 1);
        assert_eq!(mgr.active_session_count(), 1);
        assert!(mgr.take(&s2).is_some());
    }
}
