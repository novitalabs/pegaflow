use std::sync::Once;

use log::LevelFilter;
use logforth::diagnostic::ThreadLocalDiagnostic;

static INIT: Once = Once::new();

pub fn ensure_initialized() {
    if log::max_level() != LevelFilter::Off {
        return;
    }

    INIT.call_once(|| {
        let filter_str = std::env::var("RUST_LOG")
            .unwrap_or_else(|_| "info,pegaflow_transfer=debug".to_string());
        let filter: logforth::filter::EnvFilter =
            filter_str.parse().unwrap_or_else(|_| "info".into());

        let mut builder = logforth::starter_log::builder();
        builder = builder.dispatch(|d| {
            d.filter(filter)
                .diagnostic(ThreadLocalDiagnostic::default())
                .append(logforth::append::Stderr::default())
        });
        builder.apply();
    });
}
