//! Tracing helper macros for conditional fastrace instrumentation.
//!
//! When the `tracing` feature is disabled, all macros expand to nothing.
//!
//! - Function-level: use `#[cfg_attr(feature = "tracing", fastrace::trace)]`
//! - Sub-spans within a function: use these macros.

use std::{
    future::Future,
    sync::atomic::{AtomicU32, Ordering},
};

#[cfg(feature = "tracing")]
use std::sync::Arc;

#[cfg(feature = "tracing")]
use fastrace::prelude::{Span, SpanContext};
#[cfg(feature = "tracing")]
use parking_lot::Mutex;

/// Sample rate in permille (0–1000). 1000 = 100%, 10 = 1%, 0 = off.
static SAMPLE_RATE_PERMILLE: AtomicU32 = AtomicU32::new(1000);

/// Set the trace sampling rate.
///
/// `rate` is a fraction in `[0.0, 1.0]`. E.g. `0.01` = 1%.
pub fn set_trace_sample_rate(rate: f64) {
    let v = (rate * 1000.0).clamp(0.0, 1000.0) as u32;
    SAMPLE_RATE_PERMILLE.store(v, Ordering::Relaxed);
}

/// Returns `true` if this request should be traced.
#[inline]
pub fn should_sample() -> bool {
    let rate = SAMPLE_RATE_PERMILLE.load(Ordering::Relaxed);
    rate >= 1000 || rand::random_range(0..1000) < rate
}

/// Trace handle for one PegaFlow-side request lifecycle.
///
/// Cloning this handle keeps the root span alive as the lifecycle moves from
/// prepare-load into the later async GPU load task. The final clone dropping is
/// what reports the full trace.
#[derive(Clone)]
pub struct RequestTrace {
    #[cfg(feature = "tracing")]
    root: Arc<Mutex<Span>>,
    #[cfg(feature = "tracing")]
    request_id: Arc<str>,
    #[cfg(feature = "tracing")]
    load_plan_id: Option<u64>,
    #[cfg(feature = "tracing")]
    load_request_blocks: Option<usize>,
    #[cfg(feature = "tracing")]
    load_submitted_at: Option<std::time::Instant>,
    #[cfg(feature = "tracing")]
    active_span: Option<Arc<Mutex<Span>>>,
}

impl RequestTrace {
    pub fn load_lifecycle(
        instance_id: &str,
        request_id: &str,
        block_hashes: usize,
        num_prompt_tokens: u64,
        num_computed_tokens: u64,
    ) -> Self {
        #[cfg(feature = "tracing")]
        {
            let root = if should_sample() {
                Span::root("pegaflow.load_lifecycle", SpanContext::random()).with_properties(|| {
                    [
                        ("instance_id", instance_id.to_string()),
                        ("request_id", request_id.to_string()),
                        ("block_hashes", block_hashes.to_string()),
                        ("prompt_tokens", num_prompt_tokens.to_string()),
                        ("computed_tokens", num_computed_tokens.to_string()),
                    ]
                })
            } else {
                Span::noop()
            };
            Self {
                root: Arc::new(Mutex::new(root)),
                request_id: Arc::from(request_id),
                load_plan_id: None,
                load_request_blocks: None,
                load_submitted_at: None,
                active_span: None,
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = (
                instance_id,
                request_id,
                block_hashes,
                num_prompt_tokens,
                num_computed_tokens,
            );
            Self {}
        }
    }

    pub async fn in_span<F>(&self, name: &'static str, future: F) -> F::Output
    where
        F: Future,
    {
        #[cfg(feature = "tracing")]
        {
            use fastrace::future::FutureExt as _;

            future.in_span(self.child_span(name)).await
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = name;
            future.await
        }
    }

    pub fn add_property(&self, key: &'static str, value: impl Into<String>) {
        #[cfg(feature = "tracing")]
        {
            let value = value.into();
            self.root.lock().add_property(|| (key, value));
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = (key, value);
        }
    }

    pub(crate) fn add_optional_property(
        trace: &Option<Self>,
        key: &'static str,
        value: impl Into<String>,
    ) {
        if let Some(trace) = trace {
            trace.add_property(key, value);
        }
    }

    pub(crate) fn push_load_plan(
        traces: &mut Vec<Self>,
        trace: Option<Self>,
        plan_id: u64,
        request_blocks: usize,
    ) {
        if let Some(trace) = trace {
            traces.push(trace.with_load_plan(plan_id, request_blocks));
        }
    }

    pub(crate) fn begin_load_wait_done_batch(
        traces: Vec<Self>,
        batch_blocks: usize,
        layer_count: usize,
        item_count: usize,
        submitted_at: std::time::Instant,
    ) -> Vec<Self> {
        traces
            .into_iter()
            .map(|trace| {
                trace.begin_load_wait_done(batch_blocks, layer_count, item_count, submitted_at)
            })
            .collect()
    }

    pub(crate) fn record_load_queue_wait_all(traces: &[Self]) {
        for trace in traces {
            trace.record_load_queue_wait();
        }
    }

    pub(crate) fn record_load_success_all(
        traces: &[Self],
        elapsed: std::time::Duration,
        total_bytes: usize,
        memcpy_calls: usize,
    ) {
        for trace in traces {
            trace.record_load_success(elapsed, total_bytes, memcpy_calls);
        }
    }

    pub(crate) fn record_load_error_all(traces: &[Self]) {
        for trace in traces {
            trace.record_load_error();
        }
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn standalone_load_root_if_empty(traces: &[Self]) -> Option<Span> {
        if !traces.is_empty() {
            return None;
        }
        if should_sample() {
            Some(Span::root("gpu.load_task", SpanContext::random()))
        } else {
            Some(Span::noop())
        }
    }

    fn with_load_plan(self, plan_id: u64, request_blocks: usize) -> Self {
        #[cfg(feature = "tracing")]
        {
            Self {
                load_plan_id: Some(plan_id),
                load_request_blocks: Some(request_blocks),
                ..self
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = (plan_id, request_blocks);
            self
        }
    }

    fn begin_load_wait_done(
        self,
        batch_blocks: usize,
        layer_count: usize,
        item_count: usize,
        submitted_at: std::time::Instant,
    ) -> Self {
        #[cfg(feature = "tracing")]
        {
            let request_id = self.request_id.to_string();
            let plan_id = self.load_plan_id.unwrap_or(0);
            let request_blocks = self.load_request_blocks.unwrap_or(0);
            let span = self.child_span("load.wait_done").with_properties(|| {
                [
                    ("request_id", request_id),
                    ("plan_id", plan_id.to_string()),
                    ("request_blocks", request_blocks.to_string()),
                    ("batch_blocks", batch_blocks.to_string()),
                    ("layers", layer_count.to_string()),
                    ("batch_items", item_count.to_string()),
                ]
            });
            Self {
                load_submitted_at: Some(submitted_at),
                active_span: Some(Arc::new(Mutex::new(span))),
                ..self
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = (batch_blocks, layer_count, item_count, submitted_at);
            self
        }
    }

    fn record_load_queue_wait(&self) {
        #[cfg(feature = "tracing")]
        {
            if let Some(submitted_at) = self.load_submitted_at.as_ref() {
                let queue_us = submitted_at.elapsed().as_secs_f64() * 1e6;
                self.add_active_property("queue_us", format!("{queue_us:.0}"));
            }
        }
    }

    fn record_load_success(
        &self,
        elapsed: std::time::Duration,
        total_bytes: usize,
        memcpy_calls: usize,
    ) {
        #[cfg(feature = "tracing")]
        {
            let copy_us = elapsed.as_secs_f64() * 1e6;
            self.add_active_property("copy_us", format!("{copy_us:.0}"));
            self.add_active_property("bytes", total_bytes.to_string());
            self.add_active_property("memcpy_calls", memcpy_calls.to_string());
            self.add_active_property("status", "ok");
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = (elapsed, total_bytes, memcpy_calls);
        }
    }

    fn record_load_error(&self) {
        self.add_active_property("status", "error");
    }

    fn add_active_property(&self, key: &'static str, value: impl Into<String>) {
        #[cfg(feature = "tracing")]
        {
            let value = value.into();
            if let Some(span) = &self.active_span {
                span.lock().add_property(|| (key, value));
            } else {
                self.root.lock().add_property(|| (key, value));
            }
        }

        #[cfg(not(feature = "tracing"))]
        {
            let _ = (key, value);
        }
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn child_span(&self, name: &'static str) -> Span {
        let root = self.root.lock();
        Span::enter_with_parent(name, &root)
    }
}

// ── Cross-thread context propagation ──

/// Create a child span from a captured `Option<SpanContext>`. No-op if context
/// is `None` or tracing is disabled.
///
/// Pair with `#[cfg(feature = "tracing")]` fields on task structs:
/// ```ignore
/// pub struct SaveTask {
///     #[cfg(feature = "tracing")]
///     pub trace_ctx: Option<SpanContext>,
/// }
/// // Worker thread:
/// trace_child!("gpu.save_task", task.trace_ctx);
/// ```
#[macro_export]
macro_rules! trace_child {
    ($name:expr, $ctx:expr) => {
        #[cfg(feature = "tracing")]
        let _trace_child = $ctx.map(|ctx| ::fastrace::prelude::Span::root($name, ctx));
    };
    ($name:expr, $ctx:expr, $guard:ident) => {
        #[cfg(feature = "tracing")]
        let $guard = $ctx.map(|ctx| ::fastrace::prelude::Span::root($name, ctx));
    };
}

// ── Scope spans ──

/// Create a [`LocalSpan`] that lives for the enclosing scope.
///
/// ```ignore
/// trace_scope!("phase_name");           // anonymous guard, drops at scope end
/// trace_scope!("phase_name", guard);    // named guard for manual drop / with_properties
/// ```
#[macro_export]
macro_rules! trace_scope {
    ($name:expr) => {
        #[cfg(feature = "tracing")]
        let _trace_guard = ::fastrace::prelude::LocalSpan::enter_with_local_parent($name);
    };
    ($name:expr, $guard:ident) => {
        #[cfg(feature = "tracing")]
        let $guard = ::fastrace::prelude::LocalSpan::enter_with_local_parent($name);
    };
}

/// Drop a named trace guard, optionally attaching properties.
///
/// ```ignore
/// trace_scope!("phase", span);
/// // ... work ...
/// trace_drop!(span);
/// trace_drop!(span, || [("key", val.to_string())]);
/// ```
#[macro_export]
macro_rules! trace_drop {
    ($guard:ident) => {
        #[cfg(feature = "tracing")]
        drop($guard);
    };
    ($guard:ident, $props:expr) => {
        #[cfg(feature = "tracing")]
        drop($guard.with_properties($props));
    };
}

// ── Async spans ──

/// Wrap an async future in a child [`Span`]. No-op when tracing is disabled.
///
/// ```ignore
/// trace_future!("operation", some_future).await?;
/// ```
#[macro_export]
macro_rules! trace_future {
    ($name:expr, $fut:expr) => {{
        #[cfg(feature = "tracing")]
        {
            use ::fastrace::future::FutureExt as _;
            $fut.in_span(::fastrace::prelude::Span::enter_with_local_parent($name))
        }
        #[cfg(not(feature = "tracing"))]
        {
            $fut
        }
    }};
}

// ── Root spans (RPC entry points) ──

/// Create a root [`Span`] for RPC or top-level operations.
///
/// Respects the global sampling rate set via [`set_trace_sample_rate`].
///
/// ```ignore
/// trace_root!("rpc.save", root);
/// trace_root!("rpc.save", root, || [("key", val.to_string())]);
/// ```
#[macro_export]
macro_rules! trace_root {
    ($name:expr, $guard:ident) => {
        #[cfg(feature = "tracing")]
        let $guard = if $crate::should_sample() {
            ::fastrace::prelude::Span::root($name, ::fastrace::prelude::SpanContext::random())
        } else {
            ::fastrace::prelude::Span::noop()
        };
    };
    ($name:expr, $guard:ident, $props:expr) => {
        #[cfg(feature = "tracing")]
        let $guard = if $crate::should_sample() {
            ::fastrace::prelude::Span::root($name, ::fastrace::prelude::SpanContext::random())
                .with_properties($props)
        } else {
            ::fastrace::prelude::Span::noop()
        };
    };
}

/// Await a future inside an existing span. No-op when tracing is disabled.
///
/// ```ignore
/// trace_root!("rpc.save", root, || [("key", val)]);
/// let fut = async { ... };
/// let result = trace_in_span!(root, fut).await;
/// ```
#[macro_export]
macro_rules! trace_in_span {
    ($guard:ident, $fut:expr) => {{
        #[cfg(feature = "tracing")]
        {
            use ::fastrace::future::FutureExt as _;
            $fut.in_span($guard)
        }
        #[cfg(not(feature = "tracing"))]
        {
            $fut
        }
    }};
}
