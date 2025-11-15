use std::collections::HashMap;

/// PegaEngine is a simple key-value storage engine backed by a HashMap
pub struct PegaEngine {
    store: HashMap<String, String>,
}

impl PegaEngine {
    /// Create a new PegaEngine instance
    pub fn new() -> Self {
        PegaEngine {
            store: HashMap::new(),
        }
    }

    /// Get a value by key
    /// Returns Some(value) if the key exists, None otherwise
    pub fn get(&self, key: &str) -> Option<String> {
        self.store.get(key).cloned()
    }

    /// Put a key-value pair into the store
    /// If the key already exists, the value is updated
    pub fn put(&mut self, key: String, value: String) {
        self.store.insert(key, value);
    }

    /// Remove a key-value pair from the store
    /// Returns Some(value) if the key existed, None otherwise
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.store.remove(key)
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_and_get() {
        let mut engine = PegaEngine::new();
        engine.put("key1".to_string(), "value1".to_string());
        assert_eq!(engine.get("key1"), Some("value1".to_string()));
    }

    #[test]
    fn test_get_nonexistent() {
        let engine = PegaEngine::new();
        assert_eq!(engine.get("nonexistent"), None);
    }

    #[test]
    fn test_remove() {
        let mut engine = PegaEngine::new();
        engine.put("key1".to_string(), "value1".to_string());
        assert_eq!(engine.remove("key1"), Some("value1".to_string()));
        assert_eq!(engine.get("key1"), None);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut engine = PegaEngine::new();
        assert_eq!(engine.remove("nonexistent"), None);
    }

    #[test]
    fn test_update_value() {
        let mut engine = PegaEngine::new();
        engine.put("key1".to_string(), "value1".to_string());
        engine.put("key1".to_string(), "value2".to_string());
        assert_eq!(engine.get("key1"), Some("value2".to_string()));
    }
}

