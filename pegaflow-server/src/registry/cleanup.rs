use super::VmmRegistration;
use std::sync::Arc;
use tokio::sync::OwnedRwLockWriteGuard;

pub(super) struct RemovedContexts {
    pub(super) tensor_count: usize,
    pub(super) registrations: Vec<Arc<VmmRegistration>>,
}

impl RemovedContexts {
    pub(super) fn new(tensor_count: usize, registrations: Vec<Arc<VmmRegistration>>) -> Self {
        Self {
            tensor_count,
            registrations,
        }
    }
}

/// Owns native mappings after their tensor views leave the registry.
///
/// The service removes the corresponding engine addresses before returning
/// this value to the registry actor for final CUDA teardown.
#[must_use = "finish through RegistryHandle after engine registration is removed"]
pub(crate) struct RegistryCleanup {
    pub(super) tensor_count: usize,
    pub(super) _drained: Vec<OwnedRwLockWriteGuard<()>>,
    pub(super) _registrations: Vec<Arc<VmmRegistration>>,
    pub(super) _registration_barrier: Option<OwnedRwLockWriteGuard<()>>,
}

impl RegistryCleanup {
    pub(crate) fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}
