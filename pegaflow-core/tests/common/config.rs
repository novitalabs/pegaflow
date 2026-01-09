//! Test configuration matrix for running tests with different storage configurations.

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;

use pegaflow_core::ssd_cache::SsdCacheConfig;
use pegaflow_core::storage::{StorageConfig, StorageEngine, SealNotification};
use tokio::sync::mpsc::UnboundedReceiver;

/// Storage mode for test configuration.
#[derive(Debug, Clone)]
pub enum StorageMode {
    /// Memory-only mode (no SSD cache).
    MemoryOnly,
    /// SSD cache enabled (Linux only).
    #[cfg(target_os = "linux")]
    WithSsd(SsdCacheConfig),
}

/// Test configuration builder.
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Pinned memory pool size in bytes.
    pub pool_size: usize,
    /// Whether to use huge pages.
    pub use_hugepages: bool,
    /// Storage mode (memory-only or with SSD).
    pub storage_mode: StorageMode,
    /// Whether to enable TinyLFU admission control.
    pub enable_lfu_admission: bool,
    /// Optional value size hint for cache and allocator.
    pub hint_value_size_bytes: Option<usize>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self::memory_only()
    }
}

impl TestConfig {
    /// Default memory-only configuration (64MB pool).
    pub fn memory_only() -> Self {
        Self {
            pool_size: 64 * 1024 * 1024, // 64MB
            use_hugepages: false,
            storage_mode: StorageMode::MemoryOnly,
            enable_lfu_admission: true,
            hint_value_size_bytes: None,
        }
    }

    /// Memory-only configuration without LFU admission.
    pub fn memory_no_lfu() -> Self {
        Self {
            enable_lfu_admission: false,
            ..Self::memory_only()
        }
    }

    /// Small memory pool for testing eviction (8MB).
    pub fn small_pool() -> Self {
        Self {
            pool_size: 8 * 1024 * 1024, // 8MB
            ..Self::memory_only()
        }
    }

    /// SSD-enabled configuration (Linux only).
    #[cfg(target_os = "linux")]
    pub fn with_ssd() -> Self {
        let temp_dir = std::env::temp_dir();
        let cache_path = temp_dir.join(format!(
            "pegaflow_test_ssd_{}.bin",
            std::process::id()
        ));

        Self {
            pool_size: 32 * 1024 * 1024, // 32MB
            use_hugepages: false,
            storage_mode: StorageMode::WithSsd(SsdCacheConfig {
                cache_path,
                capacity_bytes: 128 * 1024 * 1024, // 128MB
                write_queue_depth: 256,
                prefetch_io_depth: 64,
            }),
            enable_lfu_admission: true,
            hint_value_size_bytes: None,
        }
    }

    /// Build a StorageEngine from this configuration.
    pub fn build_storage(&self) -> (Arc<StorageEngine>, UnboundedReceiver<SealNotification>) {
        let ssd_config = match &self.storage_mode {
            StorageMode::MemoryOnly => None,
            #[cfg(target_os = "linux")]
            StorageMode::WithSsd(config) => Some(config.clone()),
        };

        let storage_config = StorageConfig {
            enable_lfu_admission: self.enable_lfu_admission,
            hint_value_size_bytes: self.hint_value_size_bytes,
            ssd_cache_config: ssd_config,
        };

        StorageEngine::new_with_config(self.pool_size, self.use_hugepages, storage_config)
    }

    /// Get a descriptive name for this configuration.
    pub fn name(&self) -> &'static str {
        match &self.storage_mode {
            StorageMode::MemoryOnly => {
                if self.enable_lfu_admission {
                    "memory_only"
                } else {
                    "memory_no_lfu"
                }
            }
            #[cfg(target_os = "linux")]
            StorageMode::WithSsd(_) => "with_ssd",
        }
    }
}

/// Helper struct for cleanup after SSD tests.
#[cfg(target_os = "linux")]
pub struct SsdTestCleanup {
    path: PathBuf,
}

#[cfg(target_os = "linux")]
impl SsdTestCleanup {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[cfg(target_os = "linux")]
impl Drop for SsdTestCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Macro to generate tests that run with multiple configurations.
///
/// Usage:
/// ```ignore
/// test_with_configs!(test_name, |config: TestConfig| {
///     let (storage, _rx) = config.build_storage();
///     // ... test body
/// });
/// ```
#[macro_export]
macro_rules! test_with_configs {
    ($test_name:ident, $body:expr) => {
        paste::paste! {
            #[test]
            fn [<$test_name _memory_only>]() {
                let config = $crate::common::config::TestConfig::memory_only();
                $body(config);
            }

            #[test]
            fn [<$test_name _memory_no_lfu>]() {
                let config = $crate::common::config::TestConfig::memory_no_lfu();
                $body(config);
            }

            #[test]
            #[cfg(target_os = "linux")]
            fn [<$test_name _with_ssd>]() {
                let config = $crate::common::config::TestConfig::with_ssd();
                $body(config);
            }
        }
    };
}

/// Macro to generate async tests that run with multiple configurations.
#[macro_export]
macro_rules! async_test_with_configs {
    ($test_name:ident, $body:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$test_name _memory_only>]() {
                let config = $crate::common::config::TestConfig::memory_only();
                $body(config).await;
            }

            #[tokio::test]
            async fn [<$test_name _memory_no_lfu>]() {
                let config = $crate::common::config::TestConfig::memory_no_lfu();
                $body(config).await;
            }

            #[tokio::test]
            #[cfg(target_os = "linux")]
            async fn [<$test_name _with_ssd>]() {
                let config = $crate::common::config::TestConfig::with_ssd();
                $body(config).await;
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_memory_only() {
        let config = TestConfig::memory_only();
        assert_eq!(config.pool_size, 64 * 1024 * 1024);
        assert!(config.enable_lfu_admission);
        assert!(matches!(config.storage_mode, StorageMode::MemoryOnly));
    }

    #[test]
    fn test_config_small_pool() {
        let config = TestConfig::small_pool();
        assert_eq!(config.pool_size, 8 * 1024 * 1024);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_config_with_ssd() {
        let config = TestConfig::with_ssd();
        assert!(matches!(config.storage_mode, StorageMode::WithSsd(_)));
    }
}
