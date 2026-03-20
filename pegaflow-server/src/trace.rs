use fastrace::collector::{Config, Reporter, SpanRecord};

pub fn init() {
    // Full collection by default; reporting interval controls batching cadence.
    fastrace::set_reporter(LogReporter, Config::default());
}

pub fn flush() {
    fastrace::flush();
}

#[derive(Default)]
struct LogReporter;

impl Reporter for LogReporter {
    fn report(&mut self, spans: Vec<SpanRecord>) {
        use std::collections::HashMap;
        use std::fmt::Write;

        // Group spans by trace_id for single-line output.
        let mut traces: HashMap<u128, Vec<SpanRecord>> = HashMap::new();
        for span in spans {
            traces.entry(span.trace_id.0).or_default().push(span);
        }

        for (_, mut group) in traces {
            // Sort: root span (parent_id == 0) first; children by span_id (creation order).
            group.sort_by_key(|s| (s.parent_id.0 != 0, s.span_id.0));

            // Find the root span to lead the line.
            let root_idx = group.iter().position(|s| s.parent_id.0 == 0);
            let mut line = String::with_capacity(256);

            if let Some(idx) = root_idx {
                let root = &group[idx];
                let _ = write!(
                    line,
                    "trace {} trace_id={} dur_us={}",
                    root.name,
                    root.trace_id,
                    root.duration_ns / 1_000,
                );
                for (k, v) in &root.properties {
                    let _ = write!(line, " {}={}", k, v);
                }

                // Append child spans inline.
                for (i, span) in group.iter().enumerate() {
                    if i == idx {
                        continue;
                    }
                    let _ = write!(line, " | {}={}us", span.name, span.duration_ns / 1_000);
                    for (k, v) in &span.properties {
                        let _ = write!(line, " {}={}", k, v);
                    }
                }
            } else {
                // No root found — emit each span individually (shouldn't happen).
                for (i, span) in group.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(line, " | ");
                    }
                    let _ = write!(line, "{}={}us", span.name, span.duration_ns / 1_000);
                    for (k, v) in &span.properties {
                        let _ = write!(line, " {}={}", k, v);
                    }
                }
            }

            log::info!(target: "pegaflow_trace", "{}", line);
        }
    }
}
