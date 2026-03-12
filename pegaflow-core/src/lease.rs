//! Lease manager for RDMA P2P block transfer.
//!
//! A lease is a time-bounded guarantee that specific sealed blocks will not be
//! evicted from pinned memory. The lease holds `Arc<SealedBlock>` references,
//! which prevents eviction even under memory pressure.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use log::{debug, info};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::block::{BlockKey, SealedBlock};

/// Unique lease identifier.
pub type LeaseId = Uuid;

// ============================================================================
// LeaseEntry
// ============================================================================

struct LeaseEntry {
    requester: String,
    /// Holding these Arcs prevents cache eviction.
    _pinned_blocks: Vec<Arc<SealedBlock>>,
    total_bytes: u64,
    block_count: usize,
    expires_at: Instant,
}

impl LeaseEntry {
    fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }
}

// ============================================================================
// Public types
// ============================================================================

/// Configuration for the lease manager.
#[derive(Debug, Clone)]
pub struct LeaseConfig {
    /// Max total bytes that can be leased across all active leases.
    pub max_leased_bytes: u64,
    /// Default lease duration when client doesn't specify.
    pub default_duration: Duration,
    /// Maximum allowed lease duration (server caps client requests).
    pub max_duration: Duration,
}

impl Default for LeaseConfig {
    fn default() -> Self {
        Self {
            max_leased_bytes: 8 * 1024 * 1024 * 1024,   // 8 GiB
            default_duration: Duration::from_secs(600), // 10 min
            max_duration: Duration::from_secs(1800),    // 30 min
        }
    }
}

/// Result of a successful lease acquisition.
pub struct LeaseGrant {
    pub lease_id: LeaseId,
    /// Blocks that were found and leased, with their sealed data.
    pub blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    /// Block hashes that were not found in cache.
    pub missing_hashes: Vec<Vec<u8>>,
    /// Lease expiry as unix timestamp (ms).
    pub expires_at_unix_ms: u64,
}

impl std::fmt::Debug for LeaseGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeaseGrant")
            .field("lease_id", &self.lease_id)
            .field("blocks_count", &self.blocks.len())
            .field("missing_count", &self.missing_hashes.len())
            .field("expires_at_unix_ms", &self.expires_at_unix_ms)
            .finish()
    }
}

/// Error from lease operations.
#[derive(Debug)]
pub enum LeaseError {
    /// Lease budget exceeded.
    ResourceExhausted { current: u64, max: u64 },
    /// Lease not found (already expired or released).
    NotFound(LeaseId),
    /// Requester mismatch.
    PermissionDenied { lease_id: LeaseId, expected: String },
}

impl std::fmt::Display for LeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeaseError::ResourceExhausted { current, max } => {
                write!(f, "lease budget exhausted: {current}/{max} bytes")
            }
            LeaseError::NotFound(id) => {
                write!(f, "lease {id} not found or already expired")
            }
            LeaseError::PermissionDenied { lease_id, expected } => {
                write!(
                    f,
                    "lease {lease_id}: requester mismatch, expected {expected}"
                )
            }
        }
    }
}

// ============================================================================
// LeaseManager
// ============================================================================

/// Manages time-bounded leases on sealed blocks for RDMA transfer.
pub struct LeaseManager {
    config: LeaseConfig,
    /// All state under a single lock to avoid TOCTOU on budget checks.
    state: Mutex<LeaseState>,
}

struct LeaseState {
    leases: HashMap<LeaseId, LeaseEntry>,
    current_leased_bytes: u64,
}

impl LeaseManager {
    pub fn new(config: LeaseConfig) -> Self {
        Self {
            config,
            state: Mutex::new(LeaseState {
                leases: HashMap::new(),
                current_leased_bytes: 0,
            }),
        }
    }

    /// Acquire a lease on the given blocks.
    ///
    /// `found_blocks` are sealed blocks looked up from the cache by the caller.
    /// `missing_hashes` are hashes not found in cache.
    /// The lease holds Arc references to prevent eviction.
    pub fn acquire(
        &self,
        requester_node_id: &str,
        _namespace: &str,
        found_blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
        missing_hashes: Vec<Vec<u8>>,
        requested_duration_secs: u32,
    ) -> Result<LeaseGrant, LeaseError> {
        if found_blocks.is_empty() {
            return Ok(LeaseGrant {
                lease_id: Uuid::new_v4(),
                blocks: vec![],
                missing_hashes,
                expires_at_unix_ms: system_time_to_unix_ms(SystemTime::now()),
            });
        }

        let total_bytes: u64 = found_blocks.iter().map(|(_, b)| b.memory_footprint()).sum();

        let duration = self.clamp_duration(requested_duration_secs);
        let lease_id = Uuid::new_v4();
        let block_count = found_blocks.len();

        // Budget check + insert atomically under the lock.
        {
            let mut state = self.state.lock();

            if state.current_leased_bytes + total_bytes > self.config.max_leased_bytes {
                return Err(LeaseError::ResourceExhausted {
                    current: state.current_leased_bytes,
                    max: self.config.max_leased_bytes,
                });
            }

            // Only clone the Arcs for the entry (no BlockKey clones needed).
            let pinned_arcs: Vec<Arc<SealedBlock>> =
                found_blocks.iter().map(|(_, sb)| Arc::clone(sb)).collect();

            let entry = LeaseEntry {
                requester: requester_node_id.to_string(),
                _pinned_blocks: pinned_arcs,
                total_bytes,
                block_count,
                expires_at: Instant::now() + duration,
            };

            state.leases.insert(lease_id, entry);
            state.current_leased_bytes += total_bytes;
        }

        debug!(
            "lease acquired: id={lease_id} requester={requester_node_id} \
             blocks={block_count} bytes={total_bytes} duration={duration:?}",
        );

        Ok(LeaseGrant {
            lease_id,
            blocks: found_blocks,
            missing_hashes,
            expires_at_unix_ms: system_time_to_unix_ms(SystemTime::now() + duration),
        })
    }

    /// Extend an existing lease.
    pub fn renew(
        &self,
        lease_id: LeaseId,
        requester_node_id: &str,
        extend_duration_secs: u32,
    ) -> Result<u64, LeaseError> {
        let mut state = self.state.lock();
        let entry = state
            .leases
            .get_mut(&lease_id)
            .ok_or(LeaseError::NotFound(lease_id))?;

        if entry.requester != requester_node_id {
            return Err(LeaseError::PermissionDenied {
                lease_id,
                expected: entry.requester.clone(),
            });
        }

        let extension = self.clamp_duration(extend_duration_secs);
        entry.expires_at = Instant::now() + extension;
        let new_unix_ms = system_time_to_unix_ms(SystemTime::now() + extension);

        debug!("lease renewed: id={lease_id} new_duration={extension:?}");
        Ok(new_unix_ms)
    }

    /// Release a lease, dropping Arc references so blocks become evictable.
    pub fn release(&self, lease_id: LeaseId, requester_node_id: &str) -> Result<(), LeaseError> {
        let mut state = self.state.lock();

        match state.leases.entry(lease_id) {
            Entry::Occupied(o) => {
                if o.get().requester != requester_node_id {
                    return Err(LeaseError::PermissionDenied {
                        lease_id,
                        expected: o.get().requester.clone(),
                    });
                }
                let entry = o.remove();
                state.current_leased_bytes -= entry.total_bytes;

                debug!(
                    "lease released: id={lease_id} blocks={} bytes={}",
                    entry.block_count, entry.total_bytes
                );
                Ok(())
            }
            Entry::Vacant(_) => Err(LeaseError::NotFound(lease_id)),
        }
    }

    /// Sweep expired leases. Call periodically from a background task.
    pub fn sweep_expired(&self) -> usize {
        // Collect expired entries under lock, log after releasing.
        let expired: Vec<(LeaseId, String, usize, u64)>;
        {
            let mut state = self.state.lock();
            let expired_ids: Vec<LeaseId> = state
                .leases
                .iter()
                .filter(|(_, e)| e.is_expired())
                .map(|(id, _)| *id)
                .collect();

            let mut freed_bytes = 0u64;
            expired = expired_ids
                .into_iter()
                .filter_map(|id| {
                    let entry = state.leases.remove(&id)?;
                    freed_bytes += entry.total_bytes;
                    Some((id, entry.requester, entry.block_count, entry.total_bytes))
                })
                .collect();

            state.current_leased_bytes -= freed_bytes;
        }

        for (id, requester, blocks, bytes) in &expired {
            info!("lease expired: id={id} requester={requester} blocks={blocks} bytes={bytes}");
        }
        expired.len()
    }

    /// Current total leased bytes.
    pub fn current_leased_bytes(&self) -> u64 {
        self.state.lock().current_leased_bytes
    }

    /// Number of active leases.
    pub fn active_lease_count(&self) -> usize {
        self.state.lock().leases.len()
    }

    fn clamp_duration(&self, requested_secs: u32) -> Duration {
        if requested_secs == 0 {
            return self.config.default_duration;
        }
        let requested = Duration::from_secs(requested_secs as u64);
        requested.min(self.config.max_duration)
    }
}

fn system_time_to_unix_ms(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LeaseConfig {
        LeaseConfig {
            max_leased_bytes: 1024,
            default_duration: Duration::from_secs(60),
            max_duration: Duration::from_secs(300),
        }
    }

    fn block(footprint: u64) -> (BlockKey, Arc<SealedBlock>) {
        let key = BlockKey::new("ns".into(), Uuid::new_v4().as_bytes().to_vec());
        let sealed = Arc::new(SealedBlock::with_footprint(footprint));
        (key, sealed)
    }

    #[test]
    fn acquire_and_release() {
        let mgr = LeaseManager::new(test_config());
        let b = block(128);

        let grant = mgr.acquire("node-a", "ns", vec![b], vec![], 60).unwrap();
        assert_eq!(grant.blocks.len(), 1);
        assert!(grant.missing_hashes.is_empty());
        assert_eq!(mgr.active_lease_count(), 1);
        assert_eq!(mgr.current_leased_bytes(), 128);

        mgr.release(grant.lease_id, "node-a").unwrap();
        assert_eq!(mgr.active_lease_count(), 0);
        assert_eq!(mgr.current_leased_bytes(), 0);
    }

    #[test]
    fn acquire_empty_blocks_no_lease_created() {
        let mgr = LeaseManager::new(test_config());
        let grant = mgr
            .acquire("node-a", "ns", vec![], vec![vec![9]], 60)
            .unwrap();
        assert!(grant.blocks.is_empty());
        assert_eq!(grant.missing_hashes, vec![vec![9]]);
        assert_eq!(mgr.active_lease_count(), 0);
    }

    #[test]
    fn resource_exhaustion() {
        let mgr = LeaseManager::new(test_config()); // max 1024

        let grant = mgr
            .acquire("node-a", "ns", vec![block(800)], vec![], 60)
            .unwrap();
        assert_eq!(mgr.current_leased_bytes(), 800);

        // 800 + 800 > 1024
        let err = mgr
            .acquire("node-b", "ns", vec![block(800)], vec![], 60)
            .unwrap_err();
        assert!(matches!(err, LeaseError::ResourceExhausted { .. }));

        mgr.release(grant.lease_id, "node-a").unwrap();

        // Now it fits
        mgr.acquire("node-b", "ns", vec![block(800)], vec![], 60)
            .unwrap();
    }

    #[test]
    fn renew_extends_lease() {
        let mgr = LeaseManager::new(test_config());

        let grant = mgr
            .acquire("node-a", "ns", vec![block(64)], vec![], 10)
            .unwrap();

        let new_ms = mgr.renew(grant.lease_id, "node-a", 120).unwrap();
        assert!(new_ms > grant.expires_at_unix_ms);
    }

    #[test]
    fn wrong_requester_denied() {
        let mgr = LeaseManager::new(test_config());

        let grant = mgr
            .acquire("node-a", "ns", vec![block(64)], vec![], 60)
            .unwrap();

        assert!(matches!(
            mgr.release(grant.lease_id, "node-b").unwrap_err(),
            LeaseError::PermissionDenied { .. }
        ));

        assert!(matches!(
            mgr.renew(grant.lease_id, "node-b", 60).unwrap_err(),
            LeaseError::PermissionDenied { .. }
        ));
    }

    #[test]
    fn sweep_expired() {
        let config = LeaseConfig {
            max_leased_bytes: 4096,
            default_duration: Duration::from_millis(1),
            max_duration: Duration::from_millis(1),
        };
        let mgr = LeaseManager::new(config);

        mgr.acquire("node-a", "ns", vec![block(64)], vec![], 0)
            .unwrap();

        std::thread::sleep(Duration::from_millis(5));

        assert_eq!(mgr.sweep_expired(), 1);
        assert_eq!(mgr.active_lease_count(), 0);
        assert_eq!(mgr.current_leased_bytes(), 0);
    }

    #[test]
    fn duration_clamped_to_max() {
        let mgr = LeaseManager::new(test_config()); // max 300s

        let grant = mgr
            .acquire("node-a", "ns", vec![block(64)], vec![], 9999)
            .unwrap();

        let now_ms = system_time_to_unix_ms(SystemTime::now());
        let delta_ms = grant.expires_at_unix_ms.saturating_sub(now_ms);
        assert!(delta_ms <= 301_000, "expected <= 301s, got {delta_ms}ms");
    }

    #[test]
    fn not_found_after_release() {
        let mgr = LeaseManager::new(test_config());

        let grant = mgr
            .acquire("node-a", "ns", vec![block(64)], vec![], 60)
            .unwrap();
        mgr.release(grant.lease_id, "node-a").unwrap();

        assert!(matches!(
            mgr.release(grant.lease_id, "node-a").unwrap_err(),
            LeaseError::NotFound(_)
        ));
    }
}
