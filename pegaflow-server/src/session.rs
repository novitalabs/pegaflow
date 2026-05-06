//! Liveness session registry.
//!
//! Tracks an active `Session` RPC per instance_id. Each install produces a
//! monotonically increasing token; a new session for the same instance_id
//! supersedes the previous token, so a stale session's cleanup hook becomes
//! a no-op.

use dashmap::DashMap;
use parking_lot::Mutex;
use pegaflow_common::NumaNode;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionTopology {
    pub namespace: String,
    pub tp_size: u32,
    pub world_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveNumaHint {
    pub session_tp_size: u32,
    pub numa_node: NumaNode,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SaveNumaGroup {
    tp_rank: usize,
    pp_rank: usize,
}

struct SessionEntry {
    token: u64,
    topology: SessionTopology,
    save_numa_cursors: Mutex<HashMap<SaveNumaGroup, usize>>,
}

#[derive(Default)]
pub struct SessionRegistry {
    sessions: DashMap<String, SessionEntry>,
    next_token: AtomicU64,
}

impl SessionRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Install a new session for `instance_id`, returning the token that
    /// identifies it. Overwrites any existing token (the old session's
    /// cleanup will observe `is_current == false` and skip).
    pub fn install(
        &self,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> u64 {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed) + 1;
        self.sessions.insert(
            instance_id,
            SessionEntry {
                token,
                topology: SessionTopology {
                    namespace,
                    tp_size,
                    world_size,
                },
                save_numa_cursors: Mutex::new(HashMap::new()),
            },
        );
        token
    }

    pub fn topology(&self, instance_id: &str) -> Option<SessionTopology> {
        self.sessions
            .get(instance_id)
            .map(|entry| entry.topology.clone())
    }

    pub fn next_save_numa_hint(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
        candidates: &[NumaNode],
    ) -> Option<SaveNumaHint> {
        if tp_rank != 0 || candidates.len() < 2 {
            return None;
        }
        let entry = self.sessions.get(instance_id)?;
        if entry.topology.tp_size <= 1 {
            return None;
        }
        let group = SaveNumaGroup { tp_rank, pp_rank };
        let index = {
            let mut cursors = entry.save_numa_cursors.lock();
            let cursor = cursors.entry(group).or_insert(0);
            let index = *cursor % candidates.len();
            *cursor = cursor.wrapping_add(1);
            index
        };
        Some(SaveNumaHint {
            session_tp_size: entry.topology.tp_size,
            numa_node: candidates[index],
        })
    }

    /// CAS-remove: only removes if `token` is still the current one.
    /// Returns true if this caller owns the cleanup.
    pub fn take(&self, instance_id: &str, token: u64) -> bool {
        let mut owned = false;
        self.sessions.remove_if(instance_id, |_, entry| {
            if entry.token == token {
                owned = true;
                true
            } else {
                false
            }
        });
        owned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_replaces_topology_and_token() {
        let registry = SessionRegistry::default();
        let first = registry.install("inst".to_string(), "ns1".to_string(), 8, 8);
        let second = registry.install("inst".to_string(), "ns2".to_string(), 4, 4);

        assert_ne!(first, second);
        assert_eq!(
            registry.topology("inst"),
            Some(SessionTopology {
                namespace: "ns2".to_string(),
                tp_size: 4,
                world_size: 4,
            })
        );
        assert!(!registry.take("inst", first));
        assert!(registry.take("inst", second));
        assert_eq!(registry.topology("inst"), None);
    }

    #[test]
    fn save_numa_hint_round_robins_candidates() {
        let registry = SessionRegistry::default();
        registry.install("inst".to_string(), "ns".to_string(), 8, 8);
        let candidates = [NumaNode(0), NumaNode(1), NumaNode(2)];

        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(0),
            })
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(1),
            })
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(2),
            })
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(0),
            })
        );
    }

    #[test]
    fn save_numa_hint_checks_eligibility_without_advancing_cursor() {
        let registry = SessionRegistry::default();
        let candidates = [NumaNode(0), NumaNode(1)];

        registry.install("inst".to_string(), "ns".to_string(), 1, 1);
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            None
        );

        registry.install("inst".to_string(), "ns".to_string(), 8, 8);
        assert_eq!(
            registry.next_save_numa_hint("inst", 1, 0, &candidates),
            None
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(0),
            })
        );
    }

    #[test]
    fn save_numa_hint_round_robins_pp_groups_independently() {
        let registry = SessionRegistry::default();
        registry.install("inst".to_string(), "ns".to_string(), 8, 16);
        let candidates = [NumaNode(0), NumaNode(1)];

        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(0),
            })
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 1, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(0),
            })
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 0, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(1),
            })
        );
        assert_eq!(
            registry.next_save_numa_hint("inst", 0, 1, &candidates),
            Some(SaveNumaHint {
                session_tp_size: 8,
                numa_node: NumaNode(1),
            })
        );
    }
}
