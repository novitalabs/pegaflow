// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA (Non-Uniform Memory Access) utilities
//!
//! Re-exports primitives from `pegaflow-numa` and adds GPU-specific topology
//! detection and thread-pinning helpers used only within pegaflow-core.

use std::collections::HashMap;
use std::mem;
use std::process::Command;

// Re-export shared primitives so existing `pegaflow_core::numa::*` paths keep working.
pub use pegaflow_numa::{NumaNode, format_cpu_list, read_cpu_topology_from_sysfs};

/// Get the NUMA node for a GPU device
///
/// Uses `nvidia-smi topo --get-numa-id-of-nearby-cpu` to query the NUMA affinity
/// of the specified GPU. This returns the NUMA node closest to the GPU's PCIe bus.
///
/// If nvidia-smi is not available or fails, returns `NumaNode::UNKNOWN`.
///
/// # Arguments
/// * `device_id` - The CUDA device ID (e.g., 0 for GPU 0)
fn get_device_numa_node(device_id: u32) -> NumaNode {
    // Use nvidia-smi topo to get NUMA ID of nearest CPU
    let output = match Command::new("nvidia-smi")
        .args([
            "topo",
            "--get-numa-id-of-nearby-cpu",
            "-i",
            &device_id.to_string(),
        ])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            return NumaNode::UNKNOWN;
        }
    };

    if let Ok(stdout) = std::str::from_utf8(&output.stdout)
        && let Some(line) = stdout.lines().next()
        && let Some(numa_str) = line.split(':').nth(1)
        && let Ok(node) = numa_str.trim().parse::<u32>()
    {
        return NumaNode(node);
    }

    NumaNode::UNKNOWN
}

/// Pin the current thread to CPUs on a specific NUMA node
///
/// This sets the CPU affinity of the calling thread to only run on CPUs
/// belonging to the specified NUMA node. This is critical for ensuring
/// that memory allocations follow the first-touch policy on the correct node.
///
/// # Arguments
/// * `node` - The target NUMA node
///
/// # Errors
/// Returns an error if:
/// - The NUMA topology cannot be read
/// - The node ID is invalid
/// - The sched_setaffinity syscall fails
pub(crate) fn pin_thread_to_numa_node(node: NumaNode) -> Result<(), String> {
    if node.is_unknown() {
        return Err("Cannot pin to unknown NUMA node".to_string());
    }

    let node_to_cpus = read_cpu_topology_from_sysfs()
        .map_err(|e| format!("Failed to get NUMA topology: {}", e))?;

    let cpus = node_to_cpus
        .get(&node.0)
        .ok_or_else(|| format!("No CPUs found for NUMA node {}", node.0))?;

    if cpus.is_empty() {
        return Err(format!("CPU list is empty for NUMA node {}", node.0));
    }

    unsafe {
        let mut cpu_set: libc::cpu_set_t = mem::zeroed();

        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut cpu_set);
        }

        let result = libc::sched_setaffinity(
            0, // current thread
            mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );

        if result != 0 {
            let err = std::io::Error::last_os_error();
            return Err(format!("sched_setaffinity failed: {}", err));
        }
    }

    Ok(())
}

/// Run a closure on a thread pinned to a specific NUMA node.
///
/// This spawns a temporary thread, pins it to the specified NUMA node,
/// runs the closure, and returns the result. Useful for first-touch
/// memory allocation policy where memory should be allocated on a
/// specific NUMA node.
///
/// # Arguments
/// * `node` - The target NUMA node
/// * `f` - The closure to run
///
/// # Returns
/// The result of the closure, or an error if pinning failed
///
/// # Example
/// ```ignore
/// let pool = run_on_numa(NumaNode(0), || {
///     PinnedMemoryPool::new(size, true, None)
/// })?;
/// ```
pub(crate) fn run_on_numa<T, F>(node: NumaNode, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    if node.is_unknown() {
        return Err("Cannot run on unknown NUMA node".to_string());
    }

    let (tx, rx) = std::sync::mpsc::channel();

    let handle = std::thread::Builder::new()
        .name(format!("numa{}-init", node.0))
        .spawn(move || {
            // Pin thread to NUMA node before running closure
            if let Err(e) = pin_thread_to_numa_node(node) {
                let _ = tx.send(Err(e));
                return;
            }

            // Run the closure and send result
            let result = f();
            let _ = tx.send(Ok(result));
        })
        .map_err(|e| format!("Failed to spawn NUMA thread: {}", e))?;

    // Wait for result
    let result = rx
        .recv()
        .map_err(|_| "NUMA thread panicked or closed channel".to_string())?;

    // Wait for thread to finish
    handle
        .join()
        .map_err(|_| "NUMA thread panicked".to_string())?;

    result
}

/// Get NUMA affinity information for all available GPUs
///
/// Returns a vector of (device_id, numa_node) pairs for all GPUs
/// that can be detected. If nvidia-smi is not available, returns
/// an empty vector.
fn get_gpu_numa_affinity() -> Vec<(u32, NumaNode)> {
    // First, try to get the number of GPUs
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=count", "--format=csv,noheader"])
        .output()
    {
        Ok(out) if out.status.success() => out,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            log::warn!("nvidia-smi failed: {}", stderr);
            return Vec::new();
        }
        Err(e) => {
            log::warn!("nvidia-smi not found or failed to execute: {}", e);
            return Vec::new();
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    // nvidia-smi may return multiple lines, take the first non-empty line
    let count_str = stdout.lines().next().map(|s| s.trim()).unwrap_or("");
    let count: u32 = match count_str.parse::<u32>() {
        Ok(n) => n,
        Err(e) => {
            log::warn!("Failed to parse GPU count '{}': {}", count_str, e);
            return Vec::new();
        }
    };

    (0..count)
        .map(|device_id| (device_id, get_device_numa_node(device_id)))
        .collect()
}

// ============================================================================
// NumaTopology - Unified topology for GPU and CPU NUMA affinity
// ============================================================================

/// GPU-to-NUMA topology for the system.
///
/// This structure is built once during engine initialization and provides
/// efficient lookup of NUMA affinity for GPU devices.
#[derive(Debug, Clone)]
pub(crate) struct NumaTopology {
    /// Maps CUDA device ID to its preferred NUMA node.
    gpu_numa_map: HashMap<i32, NumaNode>,
    /// All NUMA nodes detected on the system.
    numa_nodes: Vec<NumaNode>,
}

impl NumaTopology {
    /// Detect and build the GPU-NUMA topology.
    ///
    /// This queries nvidia-smi for GPU NUMA affinity and reads system NUMA topology.
    /// Safe to call multiple times (idempotent).
    pub(crate) fn detect() -> Self {
        // Get GPU NUMA affinity
        let gpu_affinity = get_gpu_numa_affinity();
        let gpu_numa_map: HashMap<i32, NumaNode> = gpu_affinity
            .into_iter()
            .map(|(dev, node)| (dev as i32, node))
            .collect();

        // Get system NUMA nodes
        let numa_nodes = match read_cpu_topology_from_sysfs() {
            Ok(node_to_cpus) => {
                let mut node_ids: Vec<u32> = node_to_cpus.keys().copied().collect();
                node_ids.sort_unstable();
                node_ids.into_iter().map(NumaNode).collect()
            }
            Err(_) => {
                // Single node fallback
                vec![NumaNode(0)]
            }
        };

        Self {
            gpu_numa_map,
            numa_nodes,
        }
    }

    /// Get the preferred NUMA node for a GPU device.
    ///
    /// Returns `NumaNode::UNKNOWN` if the device is not found in the topology.
    pub(crate) fn numa_for_gpu(&self, device_id: i32) -> NumaNode {
        self.gpu_numa_map
            .get(&device_id)
            .copied()
            .unwrap_or(NumaNode::UNKNOWN)
    }

    /// Get all NUMA nodes in the system.
    pub(crate) fn numa_nodes(&self) -> &[NumaNode] {
        &self.numa_nodes
    }

    /// Get the number of NUMA nodes.
    pub(crate) fn num_nodes(&self) -> usize {
        self.numa_nodes.len()
    }

    /// Check if this is a multi-NUMA system.
    pub(crate) fn is_multi_numa(&self) -> bool {
        self.numa_nodes.len() > 1
    }

    /// Log the detected topology.
    pub(crate) fn log_summary(&self) {
        log::info!("=== GPU-NUMA Topology ===");
        log::info!("NUMA nodes: {}", self.num_nodes());

        if self.gpu_numa_map.is_empty() {
            log::warn!("No GPU NUMA affinity detected (nvidia-smi unavailable?)");
        } else {
            let mut devices: Vec<_> = self.gpu_numa_map.iter().collect();
            devices.sort_by_key(|(dev, _)| *dev);
            for (dev, node) in devices {
                log::info!("  GPU {} -> {}", dev, node);
            }
        }
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_unknown_node_fails() {
        let result = pin_thread_to_numa_node(NumaNode::UNKNOWN);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown"));
    }
}
