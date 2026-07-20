use std::sync::Arc;

use pegaflow_core::{EngineError, PegaEngine};
use tokio::sync::{RwLock, RwLockReadGuard};

use crate::registry::RegistryHandle;
use crate::session::SessionRegistry;

#[derive(Debug)]
pub(crate) struct CleanupResult {
    pub(crate) engine_found: bool,
    pub(crate) removed_tensors: usize,
}

/// Keeps registration and session creation out of cleanup transactions.
#[derive(Clone, Default)]
pub struct InstanceCoordinator {
    gate: Arc<RwLock<()>>,
    sessions: Arc<SessionRegistry>,
}

impl InstanceCoordinator {
    pub(crate) async fn registration(&self) -> RwLockReadGuard<'_, ()> {
        self.gate.read().await
    }

    pub(crate) async fn open_session(
        &self,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> u64 {
        let _registration = self.registration().await;
        self.sessions
            .install(instance_id, namespace, tp_size, world_size)
    }

    pub(crate) async fn cleanup_instance(
        &self,
        engine: Arc<PegaEngine>,
        registry: RegistryHandle,
        instance_id: String,
    ) -> Result<CleanupResult, EngineError> {
        let coordinator = self.clone();
        tokio::spawn(async move {
            let _cleanup = coordinator.gate.write().await;
            let engine_found = match engine.unregister_instance_drained(&instance_id).await {
                Ok(()) => true,
                Err(EngineError::InstanceMissing(_)) => false,
                Err(err) => return Err(err),
            };
            let removed_tensors = registry.drop_instance(instance_id.clone()).await;
            coordinator.sessions.remove(&instance_id);
            Ok(CleanupResult {
                engine_found,
                removed_tensors,
            })
        })
        .await
        .expect("instance cleanup task panicked")
    }

    pub(crate) async fn cleanup_all(
        &self,
        engine: Arc<PegaEngine>,
        registry: RegistryHandle,
    ) -> (Vec<String>, usize) {
        let coordinator = self.clone();
        tokio::spawn(async move {
            let _cleanup = coordinator.gate.write().await;
            let removed_instances = engine.unregister_all_instances_drained().await;
            let removed_tensors = registry.clear().await;
            coordinator.sessions.clear();
            (removed_instances, removed_tensors)
        })
        .await
        .expect("all-instance cleanup task panicked")
    }

    pub(crate) async fn cleanup_session(
        &self,
        engine: Arc<PegaEngine>,
        registry: RegistryHandle,
        instance_id: String,
        token: u64,
    ) -> Result<Option<usize>, EngineError> {
        let coordinator = self.clone();
        tokio::spawn(async move {
            let _cleanup = coordinator.gate.write().await;
            if !coordinator.sessions.take(&instance_id, token) {
                return Ok(None);
            }
            match engine.unregister_instance_drained(&instance_id).await {
                Ok(()) | Err(EngineError::InstanceMissing(_)) => {}
                Err(err) => return Err(err),
            }
            Ok(Some(registry.drop_instance(instance_id).await))
        })
        .await
        .expect("session cleanup task panicked")
    }
}
