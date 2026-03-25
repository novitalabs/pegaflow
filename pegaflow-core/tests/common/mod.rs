//! Shared test infrastructure for PegaEngine integration tests.
//!
//! - [`gpu_buffer`]: CUDA device memory wrapper for test data.
//! - [`helpers`]: Engine construction, registration/save helpers, poll waiters.
//! - [`harness`]: `RoundtripHarness` — one-call setup for save→query→load tests.

#![allow(dead_code, unused_imports, unreachable_pub)]

mod gpu_buffer;
mod harness;
mod helpers;

pub use gpu_buffer::*;
pub use harness::*;
pub use helpers::*;
