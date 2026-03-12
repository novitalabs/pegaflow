// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA (Non-Uniform Memory Access) primitives
//!
//! Lightweight crate providing:
//! - `NumaNode` identifier
//! - CPU topology detection from sysfs
//! - CPU-list formatting helpers
//!
//! Shared by `pegaflow-core` and `pegaflow-transfer` so neither depends on
//! the other for these primitives.

use std::collections::HashMap;
use std::fs;

/// Represents a NUMA node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumaNode(pub u32);

impl NumaNode {
    /// Represents an unknown or invalid NUMA node
    pub const UNKNOWN: NumaNode = NumaNode(u32::MAX);

    /// Check if this is the unknown node
    pub fn is_unknown(&self) -> bool {
        self.0 == u32::MAX
    }

    /// Check if this is a valid NUMA node
    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl std::fmt::Display for NumaNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            write!(f, "UNKNOWN")
        } else {
            write!(f, "NUMA{}", self.0)
        }
    }
}

/// Format a list of CPUs into a compact range representation
///
/// Example: [0, 1, 2, 3, 8, 9, 10] -> "0-3,8-10"
pub fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }

    let mut result = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];

    for &cpu in &cpus[1..] {
        if cpu == prev + 1 {
            prev = cpu;
        } else {
            // End current range
            if start == prev {
                result.push(format!("{}", start));
            } else {
                result.push(format!("{}-{}", start, prev));
            }
            start = cpu;
            prev = cpu;
        }
    }

    // Add final range
    if start == prev {
        result.push(format!("{}", start));
    } else {
        result.push(format!("{}-{}", start, prev));
    }

    result.join(",")
}

// ============================================================================
// CPU Topology from sysfs
// ============================================================================

/// Read CPU-to-NUMA mapping from sysfs
///
/// Returns a map of NUMA node ID -> list of CPU IDs.
pub fn read_cpu_topology_from_sysfs() -> Result<HashMap<u32, Vec<usize>>, String> {
    let mut node_to_cpus: HashMap<u32, Vec<usize>> = HashMap::new();

    let node_dir = std::path::Path::new("/sys/devices/system/node");
    if !node_dir.exists() {
        return Err("NUMA not supported: /sys/devices/system/node not found".to_string());
    }

    let entries =
        fs::read_dir(node_dir).map_err(|e| format!("Failed to read node directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Only process "nodeN" directories
        if !name.starts_with("node") {
            continue;
        }

        // Extract node number
        let node_id: u32 = name[4..]
            .parse()
            .map_err(|_| format!("Invalid node directory name: {}", name))?;

        // Read cpulist file
        let cpulist_path = path.join("cpulist");
        if !cpulist_path.exists() {
            continue;
        }

        let cpulist = fs::read_to_string(&cpulist_path)
            .map_err(|e| format!("Failed to read {}: {}", cpulist_path.display(), e))?;

        let cpus = parse_cpulist(cpulist.trim())?;
        node_to_cpus.insert(node_id, cpus);
    }

    if node_to_cpus.is_empty() {
        return Err("No NUMA nodes found".to_string());
    }

    Ok(node_to_cpus)
}

/// Parse Linux cpulist format
///
/// Examples:
/// - "0-15" -> [0,1,2,...,15]
/// - "0,4,8" -> [0,4,8]
/// - "0-3,8-11" -> [0,1,2,3,8,9,10,11]
/// - "0-15,32-47" (hyperthreading) -> [0,1,...,15,32,...,47]
fn parse_cpulist(cpulist: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();

    // Handle empty string
    if cpulist.is_empty() {
        return Ok(cpus);
    }

    for part in cpulist.split(',') {
        if part.contains('-') {
            // Range: "0-15"
            let range: Vec<&str> = part.split('-').collect();
            if range.len() != 2 {
                return Err(format!("Invalid CPU range format: {}", part));
            }

            let start: usize = range[0]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[0]))?;
            let end: usize = range[1]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[1]))?;

            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            // Single CPU
            let cpu: usize = part
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", part))?;
            cpus.push(cpu);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();

    Ok(cpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_node_display() {
        assert_eq!(format!("{}", NumaNode(0)), "NUMA0");
        assert_eq!(format!("{}", NumaNode(7)), "NUMA7");
        assert_eq!(format!("{}", NumaNode::UNKNOWN), "UNKNOWN");
    }

    #[test]
    fn test_format_cpu_list() {
        assert_eq!(format_cpu_list(&[]), "");
        assert_eq!(format_cpu_list(&[0]), "0");
        assert_eq!(format_cpu_list(&[0, 1, 2, 3]), "0-3");
        assert_eq!(format_cpu_list(&[0, 2, 4]), "0,2,4");
        assert_eq!(format_cpu_list(&[0, 1, 2, 4, 5]), "0-2,4-5");
        assert_eq!(format_cpu_list(&[0, 1, 2, 4, 6, 7, 8]), "0-2,4,6-8");
    }

    #[test]
    fn test_parse_cpulist_range() {
        let cpus = parse_cpulist("0-3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_list() {
        let cpus = parse_cpulist("0,4,8").unwrap();
        assert_eq!(cpus, vec![0, 4, 8]);
    }

    #[test]
    fn test_parse_cpulist_mixed() {
        let cpus = parse_cpulist("0-2,8,16-17").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 8, 16, 17]);
    }

    #[test]
    fn test_parse_cpulist_hyperthreading() {
        let cpus = parse_cpulist("0-15,32-47").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[15], 15);
        assert_eq!(cpus[16], 32);
        assert_eq!(cpus[31], 47);
    }

    #[test]
    fn test_parse_cpulist_empty() {
        let cpus = parse_cpulist("").unwrap();
        assert!(cpus.is_empty());
    }

    #[test]
    fn test_parse_cpulist_single_cpu() {
        let cpus = parse_cpulist("5").unwrap();
        assert_eq!(cpus, vec![5]);
    }
}
