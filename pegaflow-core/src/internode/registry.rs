//! Instance registry for discovered PegaFlow instances.
//!
//! This module provides a thread-safe registry for managing discovered
//! PegaFlow instances from Kubernetes service discovery.

use dashmap::DashMap;

use super::types::PegaflowInstance;

/// Registry for discovered PegaFlow instances.
///
/// Thread-safe registry that maintains the current set of discovered instances.
/// Use `InstanceRegistry::global()` to access the global registry.
#[derive(Debug, Default)]
pub struct InstanceRegistry {
    /// Map of pod name to instance.
    instances: DashMap<String, PegaflowInstance>,
}

impl InstanceRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            instances: DashMap::new(),
        }
    }

    /// Get the global instance registry.
    pub fn global() -> &'static Self {
        static REGISTRY: std::sync::OnceLock<InstanceRegistry> = std::sync::OnceLock::new();
        REGISTRY.get_or_init(InstanceRegistry::new)
    }

    /// Insert or update an instance.
    pub fn upsert(&self, instance: PegaflowInstance) {
        self.instances.insert(instance.name.clone(), instance);
    }

    /// Remove an instance by name.
    pub fn remove(&self, name: &str) -> Option<PegaflowInstance> {
        self.instances.remove(name).map(|(_, v)| v)
    }

    /// Get an instance by name.
    pub fn get(&self, name: &str) -> Option<PegaflowInstance> {
        self.instances.get(name).map(|r| r.clone())
    }

    /// Get all healthy instances.
    pub fn healthy_instances(&self) -> Vec<PegaflowInstance> {
        self.instances
            .iter()
            .filter(|r| r.is_healthy())
            .map(|r| r.clone())
            .collect()
    }

    /// Get all instances.
    pub fn all_instances(&self) -> Vec<PegaflowInstance> {
        self.instances.iter().map(|r| r.clone()).collect()
    }

    /// Get the number of registered instances.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Clear all instances.
    pub fn clear(&self) {
        self.instances.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_instance_registry() {
        let registry = InstanceRegistry::new();

        let instance1 = PegaflowInstance {
            name: "pega-1".to_string(),
            ip: "10.0.0.1".to_string(),
            namespace: "default".to_string(),
            grpc_port: 50055,
            status: "Running".to_string(),
            is_ready: true,
            labels: HashMap::new(),
        };

        let instance2 = PegaflowInstance {
            name: "pega-2".to_string(),
            ip: "10.0.0.2".to_string(),
            namespace: "default".to_string(),
            grpc_port: 50055,
            status: "Pending".to_string(),
            is_ready: false,
            labels: HashMap::new(),
        };

        registry.upsert(instance1.clone());
        registry.upsert(instance2.clone());

        assert_eq!(registry.len(), 2);
        assert_eq!(registry.healthy_instances().len(), 1);
        assert_eq!(registry.healthy_instances()[0].name, "pega-1");

        let removed = registry.remove("pega-1");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_operations() {
        let registry = InstanceRegistry::new();

        assert!(registry.is_empty());

        let instance = PegaflowInstance {
            name: "test".to_string(),
            ip: "10.0.0.1".to_string(),
            namespace: "default".to_string(),
            grpc_port: 50055,
            status: "Running".to_string(),
            is_ready: true,
            labels: HashMap::new(),
        };

        registry.upsert(instance.clone());
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        let got = registry.get("test");
        assert!(got.is_some());
        assert_eq!(got.unwrap().ip, "10.0.0.1");

        registry.clear();
        assert!(registry.is_empty());
    }
}
