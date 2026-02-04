use fastrace::collector::{Config, Reporter, SpanRecord};

pub fn init() {
    // Full collection by default; reporting interval controls batching cadence.
    fastrace::set_reporter(LogReporter::default(), Config::default());
}

pub fn flush() {
    fastrace::flush();
}

#[derive(Default)]
struct LogReporter;

impl Reporter for LogReporter {
    fn report(&mut self, spans: Vec<SpanRecord>) {
        for span in spans {
            let duration_us = span.duration_ns / 1_000;
            log::info!(
                target: "pegaflow_trace",
                "trace span name={} dur_us={} trace_id={} span_id={} parent_id={} props={} events={}",
                span.name,
                duration_us,
                span.trace_id,
                span.span_id,
                span.parent_id,
                span.properties.len(),
                span.events.len(),
            );

            if !span.properties.is_empty() {
                log::info!(target: "pegaflow_trace", "trace span props: {:?}", span.properties);
            }

            if !span.events.is_empty() {
                log::info!(target: "pegaflow_trace", "trace span events: {:?}", span.events);
            }
        }
    }
}
