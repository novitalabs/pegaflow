use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;
use uuid::Uuid;

#[cfg(feature = "rdma")]
use crate::backing::DirectQueryPlan;
use crate::block::SealedBlock;

const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(600);
const DEFAULT_LEASE_SWEEP_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryLeaseId([u8; 16]);

impl QueryLeaseId {
    pub fn fresh() -> Self {
        Self(*Uuid::new_v4().as_bytes())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("query lease id must be non-empty".to_string());
        }
        let token: [u8; 16] = bytes
            .try_into()
            .map_err(|_| format!("query lease id must be 16 bytes, got {}", bytes.len()))?;
        Ok(Self(token))
    }

    pub fn to_bytes(&self) -> [u8; 16] {
        self.0
    }
}

impl fmt::Debug for QueryLeaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryLeaseId({})", Uuid::from_bytes(self.0))
    }
}

struct QueryLease {
    instance_id: String,
    payload: QueryLeasePayload,
    remaining_consumers: usize,
    expires_at: Instant,
}

pub(crate) enum QueryLeasePayload {
    Cached(Vec<Arc<SealedBlock>>),
    #[cfg(feature = "rdma")]
    Direct(DirectQueryPlan),
}

pub(crate) struct QueryLeaseManager {
    inner: Arc<QueryLeaseInner>,
    sweeper: Option<JoinHandle<()>>,
}

struct QueryLeaseInner {
    leases: Mutex<HashMap<QueryLeaseId, QueryLease>>,
}

impl Default for QueryLeaseManager {
    fn default() -> Self {
        Self::new(DEFAULT_LEASE_SWEEP_INTERVAL)
    }
}

impl QueryLeaseManager {
    fn new(sweep_interval: Duration) -> Self {
        let inner = Arc::new(QueryLeaseInner {
            leases: Mutex::new(HashMap::new()),
        });
        let sweeper = tokio::runtime::Handle::try_current().ok().map(|handle| {
            let inner = Arc::clone(&inner);
            handle.spawn(async move {
                let mut interval = tokio::time::interval(sweep_interval);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    inner.sweep_expired();
                }
            })
        });

        Self { inner, sweeper }
    }

    pub(crate) fn create(
        &self,
        instance_id: &str,
        blocks: Vec<Arc<SealedBlock>>,
        consumers: usize,
    ) -> QueryLeaseId {
        self.sweep_expired();
        debug_assert!(!blocks.is_empty(), "query leases require ready blocks");

        let token = QueryLeaseId::fresh();
        let lease = QueryLease {
            instance_id: instance_id.to_string(),
            payload: QueryLeasePayload::Cached(blocks),
            remaining_consumers: consumers.max(1),
            expires_at: Instant::now() + DEFAULT_LEASE_TTL,
        };
        self.inner.insert(token, lease);
        token
    }

    #[cfg(feature = "rdma")]
    pub(crate) fn create_direct(
        &self,
        instance_id: &str,
        plan: DirectQueryPlan,
        consumers: usize,
    ) -> QueryLeaseId {
        self.sweep_expired();
        let token = QueryLeaseId::fresh();
        let lease = QueryLease {
            instance_id: instance_id.to_string(),
            payload: QueryLeasePayload::Direct(plan),
            remaining_consumers: consumers.max(1),
            expires_at: Instant::now() + DEFAULT_LEASE_TTL,
        };
        self.inner.insert(token, lease);
        token
    }

    pub(crate) fn consume(
        &self,
        instance_id: &str,
        token: &QueryLeaseId,
    ) -> Result<QueryLeasePayload, String> {
        self.sweep_expired();
        let mut leases = self
            .inner
            .leases
            .lock()
            .expect("query leases lock poisoned");
        let lease = leases
            .get_mut(token)
            .ok_or_else(|| "query lease is unknown or expired".to_string())?;
        if lease.instance_id != instance_id {
            return Err(format!(
                "query lease belongs to instance {}, got {}",
                lease.instance_id, instance_id
            ));
        }
        if lease.remaining_consumers > 1 {
            lease.remaining_consumers -= 1;
            return Ok(match &lease.payload {
                QueryLeasePayload::Cached(blocks) => QueryLeasePayload::Cached(blocks.clone()),
                #[cfg(feature = "rdma")]
                QueryLeasePayload::Direct(plan) => QueryLeasePayload::Direct(plan.clone()),
            });
        }

        Ok(leases
            .remove(token)
            .expect("query lease disappeared during consume")
            .payload)
    }

    pub(crate) fn release(&self, token: &QueryLeaseId) -> bool {
        self.sweep_expired();
        self.inner.remove(token)
    }

    pub(crate) fn release_instance(&self, instance_id: &str) {
        let mut leases = self
            .inner
            .leases
            .lock()
            .expect("query leases lock poisoned");
        leases.retain(|_, lease| lease.instance_id != instance_id);
    }

    pub(crate) fn sweep_expired(&self) {
        self.inner.sweep_expired();
    }
}

impl Drop for QueryLeaseManager {
    fn drop(&mut self) {
        if let Some(sweeper) = self.sweeper.take() {
            sweeper.abort();
        }
    }
}

impl QueryLeaseInner {
    fn insert(&self, token: QueryLeaseId, lease: QueryLease) {
        self.leases
            .lock()
            .expect("query leases lock poisoned")
            .insert(token, lease);
    }

    fn remove(&self, token: &QueryLeaseId) -> bool {
        self.leases
            .lock()
            .expect("query leases lock poisoned")
            .remove(token)
            .is_some()
    }

    fn sweep_expired(&self) {
        let now = Instant::now();
        self.leases
            .lock()
            .expect("query leases lock poisoned")
            .retain(|_, lease| lease.expires_at > now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consume_rejects_wrong_instance_without_removing_lease() {
        let manager = QueryLeaseManager::default();
        let lease_id = QueryLeaseId::fresh();
        manager
            .inner
            .leases
            .lock()
            .expect("query leases lock poisoned")
            .insert(
                lease_id,
                QueryLease {
                    instance_id: "inst-a".to_string(),
                    payload: QueryLeasePayload::Cached(Vec::new()),
                    remaining_consumers: 1,
                    expires_at: Instant::now() + DEFAULT_LEASE_TTL,
                },
            );

        let err = match manager.consume("inst-b", &lease_id) {
            Ok(_) => panic!("wrong instance consumed lease"),
            Err(err) => err,
        };
        assert!(err.contains("belongs to instance inst-a"));

        manager
            .consume("inst-a", &lease_id)
            .expect("original instance can still consume lease");
    }

    #[test]
    fn consume_allows_configured_number_of_consumers() {
        let manager = QueryLeaseManager::default();
        let blocks = vec![Arc::new(SealedBlock::from_slots(Vec::new()))];
        let lease_id = manager.create("inst-a", blocks, 2);

        assert!(
            matches!(manager.consume("inst-a", &lease_id).unwrap(), QueryLeasePayload::Cached(blocks) if blocks.len() == 1)
        );
        assert!(
            matches!(manager.consume("inst-a", &lease_id).unwrap(), QueryLeasePayload::Cached(blocks) if blocks.len() == 1)
        );

        let err = manager
            .consume("inst-a", &lease_id)
            .err()
            .expect("lease should be exhausted");
        assert!(err.contains("query lease is unknown or expired"));
    }

    #[cfg(feature = "rdma")]
    #[test]
    fn direct_lease_round_trips_plan_for_each_consumer() {
        for (local_count, remote_count) in [(1, 0), (0, 2), (1, 2)] {
            let manager = QueryLeaseManager::default();
            let block = Arc::new(SealedBlock::from_slots(Vec::new()));
            let remote = (remote_count > 0).then(|| crate::backing::DirectFetchPlan {
                namespace: "ns".into(),
                hashes: vec![vec![1], vec![2]],
                fetch_plan: crate::backing::rdma_fetch::FetchPlan {
                    segments: vec![
                        crate::backing::rdma_fetch::FetchPlanSegment {
                            node: "node-a".into(),
                            start: 0,
                            end: 1,
                        },
                        crate::backing::rdma_fetch::FetchPlanSegment {
                            node: "node-b".into(),
                            start: 1,
                            end: 2,
                        },
                    ],
                    block_count: 2,
                },
            });
            let plan = DirectQueryPlan {
                local_blocks: vec![Arc::clone(&block); local_count],
                remote: remote.clone(),
            };
            let lease_id = manager.create_direct("inst-a", plan.clone(), 2);
            drop(plan);
            assert!(manager.consume("inst-b", &lease_id).is_err());
            for _ in 0..2 {
                let QueryLeasePayload::Direct(received) =
                    manager.consume("inst-a", &lease_id).unwrap()
                else {
                    panic!("expected direct query plan");
                };
                assert_eq!(received.block_count(), local_count + remote_count);
                assert_eq!(received.remote, remote);
                assert_eq!(received.local_blocks.len(), local_count);
                for local in &received.local_blocks {
                    assert!(Arc::ptr_eq(local, &block));
                }
            }
            assert!(manager.consume("inst-a", &lease_id).is_err());
            assert_eq!(Arc::strong_count(&block), 1);
        }
    }

    #[cfg(feature = "rdma")]
    #[test]
    fn direct_lease_release_and_expiry_drop_local_pins() {
        let manager = QueryLeaseManager::default();
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        for expire in [false, true] {
            let lease = manager.create_direct(
                "inst-a",
                DirectQueryPlan {
                    local_blocks: vec![Arc::clone(&block)],
                    remote: None,
                },
                2,
            );
            assert_eq!(Arc::strong_count(&block), 2);
            if expire {
                manager
                    .inner
                    .leases
                    .lock()
                    .unwrap()
                    .get_mut(&lease)
                    .unwrap()
                    .expires_at = Instant::now();
                manager.sweep_expired();
            } else {
                assert!(manager.release(&lease));
            }
            assert_eq!(Arc::strong_count(&block), 1);
            assert!(manager.consume("inst-a", &lease).is_err());
        }
    }
}
