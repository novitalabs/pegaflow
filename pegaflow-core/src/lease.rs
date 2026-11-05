use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::block::{BlockKey, SealedBlock};

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
    blocks: Vec<Arc<SealedBlock>>,
    /// Object leases hold their hits by key; their loads consume them.
    objects: HashMap<BlockKey, Arc<SealedBlock>>,
    /// Consumers left; an object lease's first load sets it from `consumers`.
    remaining_consumers: usize,
    /// Loads an object lease takes in all, once its first load named it.
    consumers: Option<usize>,
    expires_at: Instant,
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
            blocks,
            objects: HashMap::new(),
            remaining_consumers: consumers.max(1),
            consumers: None,
            expires_at: Instant::now() + DEFAULT_LEASE_TTL,
        };
        self.inner.insert(token, lease);
        token
    }

    /// A lease over object hits; its loads consume it, or Release or expiry
    /// drops it.
    pub(crate) fn create_objects(
        &self,
        instance_id: &str,
        objects: HashMap<BlockKey, Arc<SealedBlock>>,
    ) -> QueryLeaseId {
        self.sweep_expired();
        let token = QueryLeaseId::fresh();
        let lease = QueryLease {
            instance_id: instance_id.to_string(),
            blocks: Vec::new(),
            objects,
            remaining_consumers: 0,
            consumers: None,
            expires_at: Instant::now() + DEFAULT_LEASE_TTL,
        };
        self.inner.insert(token, lease);
        token
    }

    /// The objects an object lease of `instance_id` holds for `keys`, `None`
    /// where it holds none, for a load of `consumers` loads in all: the
    /// lease's first load fixes that count, and a load naming another fails.
    pub(crate) fn leased_objects(
        &self,
        instance_id: &str,
        token: &QueryLeaseId,
        keys: &[BlockKey],
        consumers: usize,
    ) -> Result<Vec<Option<Arc<SealedBlock>>>, String> {
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
        if consumers == 0 {
            return Err("a load names 0 consumers".to_string());
        }
        match lease.consumers {
            Some(n) if n != consumers => {
                return Err(format!(
                    "query lease has {n} consumers, this load names {consumers}"
                ));
            }
            Some(_) => {}
            None => {
                lease.consumers = Some(consumers);
                lease.remaining_consumers = consumers;
            }
        }
        Ok(keys.iter().map(|k| lease.objects.get(k).cloned()).collect())
    }

    /// Count one load of an object lease; the last releases it. A lease
    /// released or expired meanwhile is already gone.
    pub(crate) fn consume_objects(&self, token: &QueryLeaseId) {
        let mut leases = self
            .inner
            .leases
            .lock()
            .expect("query leases lock poisoned");
        if let Some(lease) = leases.get_mut(token) {
            lease.remaining_consumers = lease.remaining_consumers.saturating_sub(1);
            if lease.remaining_consumers == 0 {
                leases.remove(token);
            }
        }
    }

    /// Drop lease `token` of `instance_id`, as a failed object load does:
    /// the restore it served cannot complete, so no later load consumes it.
    /// A lease of another instance stays.
    pub(crate) fn release_objects(&self, instance_id: &str, token: &QueryLeaseId) {
        let mut leases = self
            .inner
            .leases
            .lock()
            .expect("query leases lock poisoned");
        if leases
            .get(token)
            .is_some_and(|l| l.instance_id == instance_id)
        {
            leases.remove(token);
        }
    }

    pub(crate) fn consume(
        &self,
        instance_id: &str,
        token: &QueryLeaseId,
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
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
            return Ok(lease.blocks.clone());
        }

        Ok(leases
            .remove(token)
            .expect("query lease disappeared during consume")
            .blocks)
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
                    blocks: Vec::new(),
                    objects: HashMap::new(),
                    remaining_consumers: 1,
                    consumers: None,
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
        let blocks = vec![Arc::new(SealedBlock::from_slots(Vec::new(), 0))];
        let lease_id = manager.create("inst-a", blocks, 2);

        assert_eq!(manager.consume("inst-a", &lease_id).unwrap().len(), 1);
        assert_eq!(manager.consume("inst-a", &lease_id).unwrap().len(), 1);

        let err = manager
            .consume("inst-a", &lease_id)
            .err()
            .expect("lease should be exhausted");
        assert!(err.contains("query lease is unknown or expired"));
    }

    #[test]
    fn object_lease_is_consumed_by_its_named_loads() {
        let manager = QueryLeaseManager::new(Duration::from_secs(3600));
        let key = BlockKey::new("ns".into(), vec![1]);
        let objects = HashMap::from([(
            key.clone(),
            Arc::new(SealedBlock::from_slots(Vec::new(), 0)),
        )]);
        let lease = manager.create_objects("inst-a", objects);
        let read = |consumers| {
            manager.leased_objects("inst-a", &lease, std::slice::from_ref(&key), consumers)
        };

        assert!(read(0).is_err(), "0 consumers");
        assert!(read(2).unwrap()[0].is_some(), "first load fixes 2");
        assert!(
            read(3).err().expect("rejected").contains("2 consumers"),
            "another count"
        );
        manager.consume_objects(&lease);
        assert!(read(2).is_ok(), "one consumer left");
        manager.consume_objects(&lease);
        assert!(
            read(2)
                .err()
                .expect("rejected")
                .contains("unknown or expired"),
            "released"
        );

        // A failed load drops the lease, but only for its own instance.
        let lease = manager.create_objects(
            "inst-a",
            HashMap::from([(
                key.clone(),
                Arc::new(SealedBlock::from_slots(Vec::new(), 0)),
            )]),
        );
        let read = |consumers| {
            manager.leased_objects("inst-a", &lease, std::slice::from_ref(&key), consumers)
        };
        manager.release_objects("inst-b", &lease);
        assert!(read(2).is_ok(), "another instance cannot drop it");
        manager.release_objects("inst-a", &lease);
        assert!(read(2).is_err(), "dropped by its failed load");
    }
}
