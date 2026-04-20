//! Liveness session registry.
//!
//! Tracks an active `Session` RPC per instance_id. Each install produces a
//! monotonically increasing token; a new session for the same instance_id
//! supersedes the previous token, so a stale session's cleanup hook becomes
//! a no-op.

use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct SessionRegistry {
    sessions: DashMap<String, u64>,
    next_token: AtomicU64,
}

impl SessionRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Install a new session for `instance_id`, returning the token that
    /// identifies it. Overwrites any existing token (the old session's
    /// cleanup will observe `is_current == false` and skip).
    pub fn install(&self, instance_id: String) -> u64 {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed) + 1;
        self.sessions.insert(instance_id, token);
        token
    }

    /// CAS-remove: only removes if `token` is still the current one.
    /// Returns true if this caller owns the cleanup.
    pub fn take(&self, instance_id: &str, token: u64) -> bool {
        let mut owned = false;
        self.sessions.remove_if(instance_id, |_, current| {
            if *current == token {
                owned = true;
                true
            } else {
                false
            }
        });
        owned
    }
}
