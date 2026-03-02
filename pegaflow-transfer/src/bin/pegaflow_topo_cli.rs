// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PegaFlow System Topology CLI
//!
//! Displays the complete GPU + RDMA NIC + CPU topology grouped by NUMA node.
//!
//! Usage:
//!   cargo run --bin pegaflow_topo_cli

use pegaflow_core::logging;
use pegaflow_transfer::rdma_topo::SystemTopology;

fn main() {
    logging::init_stdout_colored("info");
    SystemTopology::detect().log_summary();
}
