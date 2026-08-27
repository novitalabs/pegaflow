use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pegaflow_common::hll::{
    HllWindowSnapshot, HyperLogLog, MAX_BUCKET_BITS, MAX_HLL_REPORT_BYTES, MAX_HLL_WINDOWS,
    MIN_BUCKET_BITS,
};
use pegaflow_common::hll_config::{
    DEFAULT_HLL_BUCKET_BITS, DEFAULT_HLL_WINDOWS, parse_hll_windows,
};

const AGGREGATE_CACHE_LIFETIME: Duration = Duration::from_millis(250);

#[derive(Debug, Clone)]
pub struct HllNodeReport {
    pub windows: Vec<HllWindowSnapshot>,
}

#[derive(Debug, Clone, Default)]
pub struct ClusterHllSnapshot {
    pub windows: Vec<ClusterHllWindowSnapshot>,
}

#[derive(Debug, Clone)]
pub struct ClusterHllWindowSnapshot {
    pub window: String,
    pub cardinality: f64,
    pub total_requests: u64,
    pub estimated_hit_rate: f64,
    pub active_nodes: u64,
    pub snapshot_age_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HllWindowSchema {
    window: String,
    window_secs: u64,
    bucket_bits: u8,
    register_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HllSchema {
    windows: Vec<HllWindowSchema>,
}

impl HllSchema {
    pub fn new(windows: Vec<(String, Duration)>, bucket_bits: u8) -> Result<Self, String> {
        if windows.is_empty() || windows.len() > MAX_HLL_WINDOWS {
            return Err(format!(
                "HLL schema window count must be in 1..={MAX_HLL_WINDOWS}, got {}",
                windows.len()
            ));
        }
        if !(MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits) {
            return Err(format!(
                "HLL bucket_bits must be in {MIN_BUCKET_BITS}..={MAX_BUCKET_BITS}, got {bucket_bits}"
            ));
        }

        let mut labels = HashSet::with_capacity(windows.len());
        let mut durations = HashSet::with_capacity(windows.len());
        for (window, duration) in &windows {
            if window.is_empty() {
                return Err("HLL window label must not be empty".into());
            }
            if *duration < Duration::from_secs(60) {
                return Err(format!("HLL window {window} must be at least 1 minute"));
            }
            if !labels.insert(window) || !durations.insert(duration) {
                return Err(format!(
                    "duplicate HLL window label or duration: {window} ({}s)",
                    duration.as_secs()
                ));
            }
        }

        let register_count = 1usize << bucket_bits;
        let payload_bytes = register_count.saturating_mul(windows.len());
        if payload_bytes > MAX_HLL_REPORT_BYTES {
            return Err(format!(
                "HLL register payload exceeds {MAX_HLL_REPORT_BYTES} bytes: {payload_bytes}"
            ));
        }

        let windows = windows
            .into_iter()
            .map(|(window, duration)| HllWindowSchema {
                window,
                window_secs: duration.as_secs(),
                bucket_bits,
                register_count,
            })
            .collect();
        Ok(Self { windows })
    }

    fn validate(&self, report: &HllNodeReport) -> Result<(), String> {
        if report.windows.len() != self.windows.len() {
            return Err(format!(
                "HLL report has {} windows, expected {}",
                report.windows.len(),
                self.windows.len()
            ));
        }

        for (index, (actual, expected)) in
            report.windows.iter().zip(self.windows.iter()).enumerate()
        {
            if actual.window != expected.window
                || actual.window_secs != expected.window_secs
                || actual.bucket_bits != expected.bucket_bits
            {
                return Err(format!(
                    "HLL window {index} schema mismatch: got {} ({}s, {} bits), expected {} ({}s, {} bits)",
                    actual.window,
                    actual.window_secs,
                    actual.bucket_bits,
                    expected.window,
                    expected.window_secs,
                    expected.bucket_bits
                ));
            }
            if actual.registers.len() != expected.register_count {
                return Err(format!(
                    "HLL window {} registers length must be {}, got {}",
                    actual.window,
                    expected.register_count,
                    actual.registers.len()
                ));
            }
            let max_register = 65 - actual.bucket_bits;
            if let Some(register) = actual
                .registers
                .iter()
                .copied()
                .find(|register| *register > max_register)
            {
                return Err(format!(
                    "HLL window {} register value {register} exceeds 64-bit hash maximum {max_register}",
                    actual.window
                ));
            }
            if actual.total_requests == 0 && actual.registers.iter().any(|register| *register != 0)
            {
                return Err(format!(
                    "HLL window {} has registers but zero total_requests",
                    actual.window
                ));
            }
        }
        Ok(())
    }
}

impl Default for HllSchema {
    fn default() -> Self {
        let windows = parse_hll_windows(DEFAULT_HLL_WINDOWS)
            .expect("default HLL window configuration must be valid");
        Self::new(windows, DEFAULT_HLL_BUCKET_BITS)
            .expect("default HLL schema must fit the report payload limit")
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReceivedHllReport {
    received_at: Instant,
    report: Arc<HllNodeReport>,
}

#[derive(Debug, Clone)]
struct AggregatedWindow {
    window: String,
    cardinality: f64,
    total_requests: u64,
    estimated_hit_rate: f64,
    active_nodes: u64,
}

#[derive(Debug, Clone, Default)]
struct AggregatedSnapshot {
    windows: Vec<AggregatedWindow>,
    oldest_received_at: Option<Instant>,
}

impl AggregatedSnapshot {
    fn materialize(&self, now: Instant) -> ClusterHllSnapshot {
        let snapshot_age_seconds = self
            .oldest_received_at
            .map(|received_at| now.saturating_duration_since(received_at).as_secs_f64())
            .unwrap_or(0.0);
        ClusterHllSnapshot {
            windows: self
                .windows
                .iter()
                .map(|window| ClusterHllWindowSnapshot {
                    window: window.window.clone(),
                    cardinality: window.cardinality,
                    total_requests: window.total_requests,
                    estimated_hit_rate: window.estimated_hit_rate,
                    active_nodes: window.active_nodes,
                    snapshot_age_seconds,
                })
                .collect(),
        }
    }
}

struct CachedAggregate {
    generation: u64,
    built_at: Instant,
    expires_at: Option<Instant>,
    snapshot: AggregatedSnapshot,
}

pub(crate) struct HllState {
    schema: HllSchema,
    stale_after: Duration,
    generation: AtomicU64,
    cache: Mutex<Option<CachedAggregate>>,
}

impl HllState {
    pub(crate) fn new(schema: HllSchema, stale_after: Duration) -> Self {
        Self {
            schema,
            stale_after,
            generation: AtomicU64::new(0),
            cache: Mutex::new(None),
        }
    }

    pub(crate) fn receive(
        &self,
        report: HllNodeReport,
        received_at: Instant,
    ) -> Result<ReceivedHllReport, String> {
        self.schema.validate(&report)?;
        Ok(ReceivedHllReport {
            received_at,
            report: Arc::new(report),
        })
    }

    pub(crate) fn mark_changed(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn snapshot<F>(&self, collect_reports: F) -> ClusterHllSnapshot
    where
        F: FnOnce() -> Vec<ReceivedHllReport>,
    {
        let now = Instant::now();
        let generation = self.generation.load(Ordering::Acquire);
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.as_ref() {
            let report_set_is_current = cached.generation == generation;
            let refresh_is_coalesced =
                now.saturating_duration_since(cached.built_at) < AGGREGATE_CACHE_LIFETIME;
            let reports_are_fresh = cached.expires_at.is_none_or(|expires_at| now <= expires_at);
            if reports_are_fresh && (report_set_is_current || refresh_is_coalesced) {
                return cached.snapshot.materialize(now);
            }
        }

        let reports: Vec<_> = collect_reports()
            .into_iter()
            .filter(|report| now.saturating_duration_since(report.received_at) <= self.stale_after)
            .collect();
        let snapshot = aggregate(&self.schema, &reports).unwrap_or_else(|error| {
            log::warn!("MetaServer HLL aggregation skipped invalid report: {error}");
            AggregatedSnapshot::default()
        });
        let expires_at = reports
            .iter()
            .filter_map(|report| report.received_at.checked_add(self.stale_after))
            .min();
        let materialized = snapshot.materialize(now);
        *cache = Some(CachedAggregate {
            generation,
            built_at: now,
            expires_at,
            snapshot,
        });
        materialized
    }
}

fn aggregate(
    schema: &HllSchema,
    reports: &[ReceivedHllReport],
) -> Result<AggregatedSnapshot, String> {
    if reports.is_empty() {
        return Ok(AggregatedSnapshot::default());
    }

    let oldest_received_at = reports.iter().map(|report| report.received_at).min();
    let mut windows = Vec::with_capacity(schema.windows.len());
    for (index, expected) in schema.windows.iter().enumerate() {
        let mut union = HyperLogLog::new(expected.bucket_bits);
        let mut total_requests = 0u64;
        for report in reports {
            let actual = report.report.windows.get(index).ok_or_else(|| {
                format!(
                    "HLL report is missing configured window {}",
                    expected.window
                )
            })?;
            if actual.registers.len() != union.registers().len() {
                return Err(format!(
                    "cannot merge HLL window {} with {} registers into {} registers",
                    expected.window,
                    actual.registers.len(),
                    union.registers().len()
                ));
            }
            for (target, source) in union
                .registers_mut()
                .iter_mut()
                .zip(actual.registers.iter())
            {
                *target = (*target).max(*source);
            }
            total_requests = total_requests.saturating_add(actual.total_requests);
        }
        let cardinality = union.cardinality();
        let estimated_hit_rate = if total_requests == 0 {
            0.0
        } else {
            1.0 - cardinality.min(total_requests as f64) / total_requests as f64
        };
        windows.push(AggregatedWindow {
            window: expected.window.clone(),
            cardinality,
            total_requests,
            estimated_hit_rate,
            active_nodes: reports.len() as u64,
        });
    }

    Ok(AggregatedSnapshot {
        windows,
        oldest_received_at,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    fn schema(bucket_bits: u8) -> HllSchema {
        HllSchema::new(
            vec![("1m".to_string(), Duration::from_secs(60))],
            bucket_bits,
        )
        .unwrap()
    }

    fn report(bucket_bits: u8, registers: Vec<u8>, total_requests: u64) -> HllNodeReport {
        HllNodeReport {
            windows: vec![HllWindowSnapshot {
                window: "1m".to_string(),
                window_secs: 60,
                bucket_bits,
                registers,
                total_requests,
            }],
        }
    }

    #[test]
    fn schema_requires_an_exact_match() {
        let schema = schema(4);
        assert!(schema.validate(&report(4, vec![0; 16], 0)).is_ok());

        let mut wrong_duration = report(4, vec![0; 16], 0);
        wrong_duration.windows[0].window_secs = 120;
        assert!(schema.validate(&wrong_duration).is_err());
        assert!(schema.validate(&report(5, vec![0; 32], 0)).is_err());
    }

    #[test]
    fn schema_rejects_reordered_windows() {
        let schema = HllSchema::new(
            vec![
                ("1m".to_string(), Duration::from_secs(60)),
                ("2m".to_string(), Duration::from_secs(120)),
            ],
            4,
        )
        .unwrap();
        let report = HllNodeReport {
            windows: vec![
                HllWindowSnapshot {
                    window: "2m".to_string(),
                    window_secs: 120,
                    bucket_bits: 4,
                    registers: vec![0; 16],
                    total_requests: 0,
                },
                HllWindowSnapshot {
                    window: "1m".to_string(),
                    window_secs: 60,
                    bucket_bits: 4,
                    registers: vec![0; 16],
                    total_requests: 0,
                },
            ],
        };

        assert!(schema.validate(&report).is_err());
    }

    #[test]
    fn schema_rejects_invalid_or_duplicate_windows() {
        assert!(HllSchema::new(vec![(String::new(), Duration::from_secs(60))], 4).is_err());
        assert!(HllSchema::new(vec![("1s".to_string(), Duration::from_secs(1))], 4).is_err());
        assert!(
            HllSchema::new(
                vec![
                    ("1m".to_string(), Duration::from_secs(60)),
                    ("one-minute".to_string(), Duration::from_secs(60)),
                ],
                4,
            )
            .is_err()
        );
    }

    #[test]
    fn impossible_64_bit_register_value_is_rejected() {
        let schema = schema(4);
        let mut registers = vec![0; 16];
        registers[3] = 255;

        let error = schema.validate(&report(4, registers, 1)).unwrap_err();
        assert!(error.contains("64-bit hash maximum"), "{error}");
    }

    #[test]
    fn aggregation_rejects_unequal_register_lengths() {
        let schema = schema(4);
        let received = ReceivedHllReport {
            received_at: Instant::now(),
            report: Arc::new(report(4, vec![0; 15], 0)),
        };

        let error = aggregate(&schema, &[received]).unwrap_err();
        assert!(error.contains("cannot merge"), "{error}");
    }

    #[test]
    fn cache_coalesces_generation_changes_within_collection_interval() {
        let state = HllState::new(schema(4), Duration::from_secs(30));
        let received = state
            .receive(report(4, vec![0; 16], 0), Instant::now())
            .unwrap();
        state.mark_changed();
        let collections = AtomicUsize::new(0);

        for _ in 0..5 {
            state.snapshot(|| {
                collections.fetch_add(1, Ordering::Relaxed);
                vec![received.clone()]
            });
        }
        assert_eq!(collections.load(Ordering::Relaxed), 1);

        state.mark_changed();
        state.snapshot(|| {
            collections.fetch_add(1, Ordering::Relaxed);
            vec![received.clone()]
        });
        assert_eq!(collections.load(Ordering::Relaxed), 1);

        std::thread::sleep(AGGREGATE_CACHE_LIFETIME + Duration::from_millis(10));
        state.snapshot(|| {
            collections.fetch_add(1, Ordering::Relaxed);
            vec![received]
        });
        assert_eq!(collections.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn cache_does_not_keep_an_expired_report_active() {
        let state = HllState::new(schema(4), Duration::from_millis(5));
        let received = state
            .receive(report(4, vec![0; 16], 0), Instant::now())
            .unwrap();
        state.mark_changed();
        assert_eq!(state.snapshot(|| vec![received.clone()]).windows.len(), 1);

        std::thread::sleep(Duration::from_millis(10));
        assert!(state.snapshot(|| vec![received]).windows.is_empty());
    }
}
