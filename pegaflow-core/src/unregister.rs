use std::collections::HashMap;
use std::sync::Arc;

use log::info;

use crate::instance::InstanceContext;
use crate::lease::QueryLeaseManager;
use crate::{EngineError, PegaEngine};

pub(super) fn remove_generation(
    instances: &mut HashMap<String, Arc<InstanceContext>>,
    query_leases: &QueryLeaseManager,
    instance_id: &str,
    expected: &Arc<InstanceContext>,
) -> bool {
    if !instances
        .get(instance_id)
        .is_some_and(|current| Arc::ptr_eq(current, expected))
    {
        return false;
    }

    query_leases.release_instance(instance_id);
    instances.remove(instance_id);
    true
}

impl PegaEngine {
    /// Unregister an instance immediately without draining accepted transfers.
    /// Allocation owners must use [`Self::unregister_instance_drained`] before
    /// releasing memory that GPU workers may still reference.
    pub fn unregister_instance(&self, instance_id: &str) -> Result<(), EngineError> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");
        let instance = instances
            .get(instance_id)
            .cloned()
            .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))?;
        instance.ensure_open()?;
        drop(
            instance
                .begin_close()
                .expect("open instance must elect its close"),
        );
        assert!(remove_generation(
            &mut instances,
            &self.query_leases,
            instance_id,
            &instance,
        ));
        info!("Unregistered instance: {}", instance_id);
        Ok(())
    }

    /// Drain accepted GPU transfers before unregistering an instance.
    ///
    /// Once this future starts, cleanup continues in an owned task even if the
    /// caller is canceled.
    pub async fn unregister_instance_drained(&self, instance_id: &str) -> Result<(), EngineError> {
        let (instance, drains) = {
            let instances = self.instances.read().expect("instances read lock poisoned");
            let instance = instances
                .get(instance_id)
                .cloned()
                .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))?;
            let drains = instance.begin_close();
            (instance, drains)
        };
        let instances = Arc::clone(&self.instances);
        let query_leases = Arc::clone(&self.query_leases);
        let instance_id = instance_id.to_string();
        tokio::spawn(async move {
            instance.finish_close(drains).await;
            let mut instances = instances.write().expect("instances write lock poisoned");
            if remove_generation(&mut instances, &query_leases, &instance_id, &instance) {
                info!("Unregistered drained instance: {}", instance_id);
            }
        })
        .await
        .expect("instance cleanup task panicked");
        Ok(())
    }

    /// Unregister all instances immediately without draining accepted transfers.
    /// Allocation owners must use [`Self::unregister_all_instances_drained`]
    /// before releasing memory that GPU workers may still reference.
    pub fn unregister_all_instances(&self) -> Vec<String> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");
        let targets = instances
            .iter()
            .filter(|(_, instance)| instance.ensure_open().is_ok())
            .map(|(id, instance)| (id.clone(), Arc::clone(instance)))
            .collect::<Vec<_>>();
        let mut ids = Vec::with_capacity(targets.len());
        for (id, instance) in targets {
            drop(
                instance
                    .begin_close()
                    .expect("open instance must elect its close"),
            );
            if remove_generation(&mut instances, &self.query_leases, &id, &instance) {
                ids.push(id);
            }
        }
        if !ids.is_empty() {
            info!("Unregistered all instances: {:?}", ids);
        }
        ids
    }

    /// Drain accepted GPU transfers before unregistering every instance.
    ///
    /// Once this future starts, cleanup continues in an owned task even if the
    /// caller is canceled.
    pub async fn unregister_all_instances_drained(&self) -> Vec<String> {
        let instances = {
            let instances = self
                .instances
                .read()
                .expect("instances write lock poisoned");
            instances
                .iter()
                .map(|(id, instance)| (id.clone(), Arc::clone(instance), instance.begin_close()))
                .collect::<Vec<_>>()
        };
        let current = Arc::clone(&self.instances);
        let query_leases = Arc::clone(&self.query_leases);
        tokio::spawn(async move {
            let mut instances = instances;
            futures::future::join_all(
                instances
                    .iter_mut()
                    .map(|(_, instance, drains)| instance.finish_close(drains.take())),
            )
            .await;
            let mut current = current.write().expect("instances write lock poisoned");
            let mut ids = Vec::with_capacity(instances.len());
            for (id, instance, _) in instances {
                if remove_generation(&mut current, &query_leases, &id, &instance) {
                    ids.push(id);
                }
            }
            if !ids.is_empty() {
                info!("Unregistered all drained instances: {:?}", ids);
            }
            ids
        })
        .await
        .expect("all-instance cleanup task panicked")
    }
}
