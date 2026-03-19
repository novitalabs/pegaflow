use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use sideway::ibverbs::memory_region::MemoryRegion;

use super::session::RcSession;
use crate::engine::RegisteredMemoryRegion;
use crate::error::{Result, TransferError};

pub(super) struct RegisteredMemoryEntry {
    pub(super) base_ptr: u64,
    pub(super) len: usize,
    pub(super) mr: Arc<MemoryRegion>,
}

#[derive(Clone, Copy)]
struct RemoteMemoryEntry {
    base_ptr: u64,
    end_ptr: u64,
    rkey: u32,
}

#[derive(Default)]
pub(super) struct RcBackendState {
    pub(super) registered: HashMap<u64, RegisteredMemoryEntry>,
    /// Pre-connect sessions in FIFO order (first prepared, first connected).
    pub(super) pending: VecDeque<Arc<RcSession>>,
    /// Connected sessions keyed by remote QP number.
    pub(super) sessions: HashMap<u32, Arc<RcSession>>,
    /// Remote memory cache keyed by remote QP number.
    remote_memory: HashMap<u32, Vec<RemoteMemoryEntry>>,
}

impl RcBackendState {
    /// Find a registered local MR that fully covers `[ptr, ptr+len)`.
    pub(super) fn find_local_mr(&self, ptr: u64, len: usize) -> Option<Arc<MemoryRegion>> {
        let end = ptr.checked_add(len as u64)?;
        self.registered.values().find_map(|entry| {
            let entry_end = entry.base_ptr.checked_add(entry.len as u64)?;
            if ptr >= entry.base_ptr && end <= entry_end {
                Some(Arc::clone(&entry.mr))
            } else {
                None
            }
        })
    }

    /// Look up the remote rkey for `[remote_ptr, remote_ptr+len)` from the
    /// handshake snapshot cached for `remote_qpn`.
    pub(super) fn find_remote_rkey(
        &self,
        remote_qpn: u32,
        remote_ptr: u64,
        len: usize,
    ) -> Option<u32> {
        let end = remote_ptr.checked_add(len as u64)?;
        let entries = self.remote_memory.get(&remote_qpn)?;
        let index = match entries.binary_search_by_key(&remote_ptr, |e| e.base_ptr) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        let entry = &entries[index];
        if remote_ptr >= entry.base_ptr && end <= entry.end_ptr {
            Some(entry.rkey)
        } else {
            None
        }
    }

    /// Validate and cache the remote memory regions received during handshake.
    pub(super) fn cache_remote_memory(
        &mut self,
        remote_qpn: u32,
        remote_memory_regions: &[RegisteredMemoryRegion],
    ) -> Result<()> {
        let mut cached = Vec::with_capacity(remote_memory_regions.len());
        for entry in remote_memory_regions.iter().copied() {
            if entry.len == 0 {
                return Err(TransferError::Backend(
                    "handshake response contains zero-length memory region".to_string(),
                ));
            }
            let Some(end_ptr) = entry.base_ptr.checked_add(entry.len) else {
                return Err(TransferError::Backend(
                    "handshake response contains memory region overflow".to_string(),
                ));
            };
            cached.push(RemoteMemoryEntry {
                base_ptr: entry.base_ptr,
                end_ptr,
                rkey: entry.rkey,
            });
        }
        cached.sort_unstable_by_key(|e| e.base_ptr);
        for pair in cached.windows(2) {
            if pair[1].base_ptr < pair[0].end_ptr {
                return Err(TransferError::Backend(
                    "handshake response contains overlapping memory regions".to_string(),
                ));
            }
        }
        self.remote_memory.insert(remote_qpn, cached);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_remote_memory_rejects_overlapping_regions() {
        let mut state = RcBackendState::default();
        let regions = vec![
            RegisteredMemoryRegion {
                base_ptr: 0x1000,
                len: 0x200,
                rkey: 1,
            },
            RegisteredMemoryRegion {
                base_ptr: 0x1100,
                len: 0x100,
                rkey: 2,
            },
        ];

        let error = state
            .cache_remote_memory(1, &regions)
            .expect_err("overlap should fail");
        assert_eq!(
            error,
            TransferError::Backend(
                "handshake response contains overlapping memory regions".to_string()
            )
        );
    }

    #[test]
    fn find_remote_rkey_uses_sorted_snapshot() {
        let mut state = RcBackendState::default();
        let remote_qpn = 42;
        let regions = vec![
            RegisteredMemoryRegion {
                base_ptr: 0x3000,
                len: 0x100,
                rkey: 3,
            },
            RegisteredMemoryRegion {
                base_ptr: 0x1000,
                len: 0x100,
                rkey: 1,
            },
            RegisteredMemoryRegion {
                base_ptr: 0x2000,
                len: 0x100,
                rkey: 2,
            },
        ];
        state
            .cache_remote_memory(remote_qpn, &regions)
            .expect("snapshot cache");

        let hit = state.find_remote_rkey(remote_qpn, 0x2080, 0x10);
        assert_eq!(hit, Some(2));

        let miss = state.find_remote_rkey(remote_qpn, 0x2500, 0x10);
        assert!(miss.is_none());
    }
}
