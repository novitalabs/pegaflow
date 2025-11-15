use pega_core::PegaEngine as CoreEngine;
use pyo3::prelude::*;

/// Python wrapper for PegaEngine
#[pyclass]
struct PegaEngine {
    engine: CoreEngine,
}

#[pymethods]
impl PegaEngine {
    /// Create a new PegaEngine instance
    #[new]
    fn new() -> Self {
        PegaEngine {
            engine: CoreEngine::new(),
        }
    }

    /// Get a value by key
    /// Returns the value if the key exists, None otherwise
    fn get(&self, key: &str) -> Option<String> {
        self.engine.get(key)
    }

    /// Put a key-value pair into the store
    /// If the key already exists, the value is updated
    fn put(&mut self, key: String, value: String) {
        self.engine.put(key, value);
    }

    /// Remove a key-value pair from the store
    /// Returns the value if the key existed, None otherwise
    fn remove(&mut self, key: &str) -> Option<String> {
        self.engine.remove(key)
    }
}

/// A Python module implemented in Rust.
/// This module is named "pegaflow" and will be imported as: from pegaflow import PegaEngine
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PegaEngine>()?;
    Ok(())
}
