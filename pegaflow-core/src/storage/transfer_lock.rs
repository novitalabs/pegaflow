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

    pub(crate) fn lock_timeout(&self) -> Duration {
        self.lock_timeout
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

    /// Release a transfer session's locks. Returns the number of blocks released.
    ///
    /// # Security model
    ///
    /// Any caller with the session ID can release the lock. This relies on:
    /// 1. Session IDs are UUIDv4 (cryptographically random, unguessable)
    /// 2. The gRPC port is network-isolated (internal cluster only)
    pub(crate) fn release(&self, session_id: &str) -> usize {
        let mut inner = self.inner.lock();
        if let Some(session) = inner.remove(session_id) {
            let count = session.blocks.len();
            core_metrics()
                .transfer_lock_active
                .add(-(count as i64), &[]);
            debug!(
                "Transfer lock released: session={} requester={} blocks={}",
                session_id, session.requester_id, count
            );
            count
        } else {
            warn!("Transfer lock release: session not found: {}", session_id);
            0
        }
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

    /// Total locked blocks across all sessions.
    #[cfg(test)]
    fn total_locked_blocks(&self) -> usize {
        self.inner.lock().values().map(|s| s.blocks.len()).sum()
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
    fn lock_and_release() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));
        let (key, block) = make_test_block();

        let session_id = mgr.lock_blocks("node-a", vec![(key.clone(), block.clone())]);
        assert_eq!(mgr.active_session_count(), 1);
        assert_eq!(mgr.total_locked_blocks(), 1);

        let released = mgr.release(&session_id);
        assert_eq!(released, 1);
        assert_eq!(mgr.active_session_count(), 0);
        assert_eq!(mgr.total_locked_blocks(), 0);
    }

    #[test]
    fn release_unknown_session_returns_zero() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));
        assert_eq!(mgr.release("nonexistent"), 0);
    }

    #[test]
    fn gc_expired_sessions() {
        let mgr = TransferLockManager::new(Duration::from_millis(10));
        let (key, block) = make_test_block();
        let _session_id = mgr.lock_blocks("node-a", vec![(key, block)]);

        // Not expired yet
        assert_eq!(mgr.gc_expired(), 0);
        assert_eq!(mgr.active_session_count(), 1);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(mgr.gc_expired(), 1);
        assert_eq!(mgr.active_session_count(), 0);
    }

    #[test]
    fn multiple_concurrent_sessions() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));
        let (key1, block1) = make_test_block();
        let key2 = BlockKey::new("ns".into(), vec![4, 5, 6]);
        let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));

        let s1 = mgr.lock_blocks("node-a", vec![(key1.clone(), block1.clone())]);
        let s2 = mgr.lock_blocks("node-b", vec![(key1, block1), (key2, block2)]);

        assert_eq!(mgr.active_session_count(), 2);
        assert_eq!(mgr.total_locked_blocks(), 3);

        mgr.release(&s1);
        assert_eq!(mgr.active_session_count(), 1);
        assert_eq!(mgr.total_locked_blocks(), 2);

        mgr.release(&s2);
        assert_eq!(mgr.active_session_count(), 0);
    }

    #[test]
    fn arc_keeps_memory_alive() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        // Lock holds an Arc clone
        let session_id = mgr.lock_blocks("node-a", vec![(key, block.clone())]);

        // Block has 2 strong refs: our local `block` + the lock's copy
        assert_eq!(Arc::strong_count(&block), 2);

        mgr.release(&session_id);
        // After release, only our local ref remains
        assert_eq!(Arc::strong_count(&block), 1);
    }

    #[test]
    fn lock_empty_blocks_list() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));

        let session_id = mgr.lock_blocks("node-a", vec![]);
        assert_eq!(mgr.active_session_count(), 1);
        assert_eq!(mgr.total_locked_blocks(), 0);

        let released = mgr.release(&session_id);
        assert_eq!(released, 0);
        assert_eq!(mgr.active_session_count(), 0);
    }

    #[test]
    fn lock_with_zero_timeout_expires_immediately() {
        let mgr = TransferLockManager::new(Duration::from_secs(0));
        let (key, block) = make_test_block();

        let _session_id = mgr.lock_blocks("node-a", vec![(key, block)]);
        assert_eq!(mgr.active_session_count(), 1);

        // With zero timeout, any elapsed time > 0 means expired
        std::thread::sleep(Duration::from_millis(1));
        assert_eq!(mgr.gc_expired(), 1);
        assert_eq!(mgr.active_session_count(), 0);
        assert_eq!(mgr.total_locked_blocks(), 0);
    }

    #[test]
    fn gc_when_no_sessions_exist() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));

        // GC on empty manager returns 0 and does not panic
        assert_eq!(mgr.gc_expired(), 0);
        assert_eq!(mgr.active_session_count(), 0);
    }

    #[test]
    fn large_number_of_concurrent_sessions() {
        let mgr = TransferLockManager::new(Duration::from_secs(300));
        let mut session_ids = Vec::new();

        let blocks_per_session = 5;
        let num_sessions = 100;

        for i in 0..num_sessions {
            let blocks: Vec<(BlockKey, Arc<SealedBlock>)> = (0..blocks_per_session)
                .map(|j| {
                    let key = BlockKey::new("ns".into(), vec![i as u8, j as u8]);
                    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
                    (key, block)
                })
                .collect();
            let requester = format!("node-{}", i);
            let sid = mgr.lock_blocks(&requester, blocks);
            session_ids.push(sid);
        }

        assert_eq!(mgr.active_session_count(), num_sessions);
        assert_eq!(mgr.total_locked_blocks(), num_sessions * blocks_per_session);

        // Release half
        for sid in &session_ids[..num_sessions / 2] {
            mgr.release(sid);
        }
        assert_eq!(mgr.active_session_count(), num_sessions / 2);
        assert_eq!(
            mgr.total_locked_blocks(),
            (num_sessions / 2) * blocks_per_session
        );

        // Release the rest
        for sid in &session_ids[num_sessions / 2..] {
            mgr.release(sid);
        }
        assert_eq!(mgr.active_session_count(), 0);
        assert_eq!(mgr.total_locked_blocks(), 0);
    }

    #[test]
    fn release_same_session_twice_returns_zero_second_time() {
        let mgr = TransferLockManager::new(Duration::from_secs(30));
        let (key, block) = make_test_block();

        let session_id = mgr.lock_blocks("node-a", vec![(key, block)]);
        assert_eq!(mgr.release(&session_id), 1);
        // Second release is a no-op
        assert_eq!(mgr.release(&session_id), 0);
        assert_eq!(mgr.active_session_count(), 0);
    }

    #[test]
    fn gc_only_removes_expired_sessions() {
        let mgr = TransferLockManager::new(Duration::from_millis(10));
        let (key1, block1) = make_test_block();

        // Session 1: will expire
        let _s1 = mgr.lock_blocks("node-a", vec![(key1, block1)]);

        std::thread::sleep(Duration::from_millis(20));

        // Session 2: fresh, should NOT expire
        let key2 = BlockKey::new("ns".into(), vec![7, 8, 9]);
        let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));
        let s2 = mgr.lock_blocks("node-b", vec![(key2, block2)]);

        assert_eq!(mgr.active_session_count(), 2);
        assert_eq!(mgr.gc_expired(), 1);
        assert_eq!(mgr.active_session_count(), 1);

        // The surviving session is s2
        assert_eq!(mgr.release(&s2), 1);
        assert_eq!(mgr.active_session_count(), 0);
    }
}
