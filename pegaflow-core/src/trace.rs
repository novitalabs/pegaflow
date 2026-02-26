//! Tracing helper macros for conditional fastrace instrumentation.
//!
//! When the `tracing` feature is disabled, all macros expand to nothing.
//!
//! - Function-level: use `#[cfg_attr(feature = "tracing", fastrace::trace)]`
//! - Sub-spans within a function: use these macros.

use std::sync::atomic::{AtomicU32, Ordering};

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
