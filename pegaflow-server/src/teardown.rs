//! Per-instance teardown: fence → drain GPU workers → unmap CUDA IPC.
//!
//! Queued save/load tasks capture raw device pointers into a client's CUDA
//! IPC mapping and dereference them only when the per-GPU worker thread
//! executes them. Unmapping while such a task is still queued either poisons
//! the CUDA context (dead VA) or — if a new registration reuses the VA —
//! silently reads/writes another client's live KV cache. Every teardown
//! entrance (graceful Unregister RPC, session-watcher disconnect cleanup,
//! admin HTTP endpoint) therefore goes through [`InstanceTeardown`], which
//! enforces the one ordering that is safe:
//!
//! 1. fence: remove the instance from the engine and release its leases,
//! 2. drain: close the worker channels and wait for the threads to exit,
//! 3. unmap: only then drop the CUDA IPC tensors.
//!
//! Teardowns of different instances are independent (separate worker pools,
//! separate tensors); only same-instance callers need coordination. The
//! `draining` set arbitrates that: exactly one caller claims the instance
//! (the atomic engine-map removal) and owns drain + unmap; concurrent
//! callers for the same id observe the claim and back off instead of
//! unmapping memory the owner's drain still protects.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use log::{error, info};
use parking_lot::Mutex;
use pegaflow_core::{InstanceContext, PegaEngine};

use crate::registry::RegistryHandle;

/// How long to wait for a dead instance's GPU worker queues to drain. A deep
/// save backlog drains at PCIe D2H bandwidth (seconds for tens of GiB); only
/// a wedged GPU exceeds this.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) struct CleanupOutcome {
    /// Whether this caller found and tore down the instance.
    pub instance_found: bool,
    /// False when the GPU workers failed to drain within [`DRAIN_TIMEOUT`].
    /// The CUDA IPC mappings are deliberately leaked in that case — unmapping
    /// memory a stale task may still touch trades a bounded leak for memory
    /// corruption.
    pub drained: bool,
    pub dropped_tensors: usize,
}

pub(crate) struct CleanupAllOutcome {
    pub removed_instances: Vec<String>,
    pub dropped_tensors: usize,
    /// Instances whose drain timed out; their CUDA IPC mappings are leaked.
    pub leaked_instances: Vec<String>,
}

enum Claim {
    /// This caller removed the instance and owns drain + unmap.
    Owner(Arc<InstanceContext>),
    /// Another teardown owns this instance's drain (or leaked it after a
    /// drain timeout); do not touch its tensors.
    Draining,
    /// No instance and no drain in flight — only a tensor sweep applies.
    Missing,
}

pub struct InstanceTeardown {
    engine: Arc<PegaEngine>,
    registry: RegistryHandle,
    /// Instance ids whose drain is in flight. An id stays here forever if its
    /// drain timed out, so later sweeps can never unmap the leaked mappings.
    ///
    /// Known hole: registration does not consult this set, so a new
    /// registration reusing a leaked instance id overwrites the leaked
    /// tensors in the registry (unmapping them under the wedged tasks).
    /// Closing it needs incarnation-scoped registrations — deferred.
    draining: Mutex<HashSet<String>>,
}

impl InstanceTeardown {
    pub(crate) fn new(engine: Arc<PegaEngine>, registry: RegistryHandle) -> Arc<Self> {
        Arc::new(Self {
            engine,
            registry,
            draining: Mutex::new(HashSet::new()),
        })
    }

    /// Atomically decide this caller's role for `instance_id`. The lock makes
    /// "remove from engine map" and "appear in the draining set" one step, so
    /// a concurrent caller can never see neither.
    fn claim(&self, instance_id: &str) -> Claim {
        let mut draining = self.draining.lock();
        match self.engine.unregister_instance(instance_id) {
            Ok(instance) => {
                draining.insert(instance_id.to_string());
                Claim::Owner(instance)
            }
            Err(_) if draining.contains(instance_id) => Claim::Draining,
            Err(_) => Claim::Missing,
        }
    }

    fn release_claim(&self, instance_id: &str) {
        self.draining.lock().remove(instance_id);
    }

    /// Tear down one instance. Idempotent: late callers observe `Missing` and
    /// only sweep leftover tensors (e.g. a partial registration where the
    /// tensors were materialized but engine registration failed).
    pub(crate) async fn cleanup(&self, instance_id: &str, reason: &str) -> CleanupOutcome {
        match self.claim(instance_id) {
            Claim::Owner(instance) => {
                if !drain_instance(&instance, reason).await {
                    // Keep the id claimed forever: it must block any later
                    // tensor sweep for this id from unmapping the leaked
                    // mappings while the wedged tasks may still hold them.
                    return CleanupOutcome {
                        instance_found: true,
                        drained: false,
                        dropped_tensors: 0,
                    };
                }
                // Safe to unmap: fenced, and the worker threads have exited.
                let dropped_tensors = self.registry.drop_instance(instance_id.to_string()).await;
                self.release_claim(instance_id);
                info!(
                    "Teardown ({reason}): instance {instance_id} removed, \
                     {dropped_tensors} CUDA IPC tensor(s) dropped"
                );
                CleanupOutcome {
                    instance_found: true,
                    drained: true,
                    dropped_tensors,
                }
            }
            Claim::Draining => CleanupOutcome {
                instance_found: false,
                drained: true,
                dropped_tensors: 0,
            },
            Claim::Missing => {
                let dropped_tensors = self.registry.drop_instance(instance_id.to_string()).await;
                if dropped_tensors > 0 {
                    info!(
                        "Teardown ({reason}): swept {dropped_tensors} orphaned CUDA IPC \
                         tensor(s) for unregistered instance {instance_id}"
                    );
                }
                CleanupOutcome {
                    instance_found: false,
                    drained: true,
                    dropped_tensors,
                }
            }
        }
    }

    /// Tear down every instance (admin escape hatch). Orphaned tensor
    /// contexts are swept only when every instance drained cleanly, since the
    /// sweep cannot distinguish them from a leaked instance's tensors.
    pub(crate) async fn cleanup_all(&self, reason: &str) -> CleanupAllOutcome {
        // Claim every instance atomically, same contract as claim().
        let instances = {
            let mut draining = self.draining.lock();
            let instances = self.engine.unregister_all_instances();
            for instance in &instances {
                draining.insert(instance.id().to_string());
            }
            instances
        };
        let removed_instances: Vec<String> = instances.iter().map(|i| i.id().to_string()).collect();

        // Drain concurrently; each instance gets its own timeout.
        let drains: Vec<_> = instances
            .iter()
            .map(|instance| {
                let instance = Arc::clone(instance);
                let reason = reason.to_string();
                tokio::spawn(async move { drain_instance(&instance, &reason).await })
            })
            .collect();

        let mut leaked_instances = Vec::new();
        let mut dropped_tensors = 0;
        for (instance, drain) in instances.iter().zip(drains) {
            if drain.await.unwrap_or(false) {
                dropped_tensors += self.registry.drop_instance(instance.id().to_string()).await;
                self.release_claim(instance.id());
            } else {
                leaked_instances.push(instance.id().to_string());
            }
        }

        if leaked_instances.is_empty() {
            dropped_tensors += self.registry.clear().await;
        } else {
            error!(
                "Teardown ({reason}): {} instance(s) failed to drain, their CUDA IPC \
                 mappings stay leaked: {leaked_instances:?}",
                leaked_instances.len()
            );
        }

        CleanupAllOutcome {
            removed_instances,
            dropped_tensors,
            leaked_instances,
        }
    }
}

/// Returns true when the instance's GPU workers drained and exited in time.
async fn drain_instance(instance: &InstanceContext, reason: &str) -> bool {
    match tokio::time::timeout(DRAIN_TIMEOUT, instance.shutdown()).await {
        Ok(()) => true,
        Err(_) => {
            error!(
                "Teardown ({reason}): GPU workers for instance {} did not drain within \
                 {DRAIN_TIMEOUT:?}; leaking its CUDA IPC mappings instead of unmapping \
                 memory a stale task may still touch",
                instance.id()
            );
            false
        }
    }
}
