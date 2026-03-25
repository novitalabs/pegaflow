//! HyperLogLog-based cache hit rate estimation.
//!
//! Provides a minimal HyperLogLog implementation and a sliding-window tracker
//! for estimating the theoretical maximum cache hit rate over a time window.
//!
//! Hash inputs must be at least 4 bytes and should have good uniformity
//! (e.g. SHA-256, xxHash). Longer hashes give more leading-zero headroom.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Allowed range for `bucket_bits` (register index width).
pub const MIN_BUCKET_BITS: u8 = 4;
pub const MAX_BUCKET_BITS: u8 = 18;

// ============================================================================
// HyperLogLog core
// ============================================================================

/// Minimal HyperLogLog for cardinality estimation.
///
/// Input hashes are expected to have good uniformity (e.g. SHA-256).
/// The full hash is used: top `bucket_bits` bits select the register,
/// remaining bits are scanned for leading zeros.
pub struct HyperLogLog {
    registers: Vec<u8>,
    bucket_bits: u8,
    /// Mask that zeros out the top `bucket_bits` bits in a big-endian u32
    /// read from hash\[0..4\]: `(1u32 << (32 - bucket_bits)) - 1`.
    lz_mask: u32,
}

impl HyperLogLog {
    /// Create a new HyperLogLog with the given bucket bits.
    ///
    /// `bucket_bits` determines the number of buckets (2^bucket_bits) and estimation
    /// accuracy (~1.04 / sqrt(2^bucket_bits)).  14 gives 16384 buckets
    /// and ~0.8% standard error.
    pub fn new(bucket_bits: u8) -> Self {
        assert!(
            (MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits),
            "HLL bucket_bits must be in {MIN_BUCKET_BITS}..={MAX_BUCKET_BITS}, got {bucket_bits}"
        );
        Self {
            registers: vec![0u8; 1 << bucket_bits],
            bucket_bits,
            lz_mask: (1u32 << (32 - bucket_bits)) - 1,
        }
    }

    /// Insert a block hash.
    ///
    /// Treats the hash as a big-endian bit stream:
    /// - Top `bucket_bits` bits → register index
    /// - Remaining bits → count leading zeros (ρ)
    ///
    /// Shorter hashes are implicitly zero-padded; longer hashes give
    /// more leading-zero headroom and better accuracy.
    pub fn insert(&mut self, hash: &[u8]) {
        let index = bucket_index(hash, self.bucket_bits);
        let rho = count_leading_zeros(hash, self.bucket_bits, self.lz_mask) + 1;

        let reg = &mut self.registers[index];
        if rho > *reg {
            *reg = rho;
        }
    }

    /// Estimate the cardinality (number of distinct elements).
    pub fn cardinality(&self) -> f64 {
        estimate_cardinality(&self.registers)
    }

    /// Merge another HLL into this one (element-wise max of registers).
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(
            self.bucket_bits, other.bucket_bits,
            "cannot merge HLLs with different bucket_bits"
        );
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *b > *a {
                *a = *b;
            }
        }
    }

    /// Reset all registers to zero.
    pub fn clear(&mut self) {
        self.registers.fill(0);
    }

    /// Number of bits used for bucket indexing.
    pub fn bucket_bits(&self) -> u8 {
        self.bucket_bits
    }
}

// ============================================================================
// Free helpers
// ============================================================================

/// Read byte at `index`, returning 0 for out-of-bounds positions.
fn byte_or_zero(hash: &[u8], index: usize) -> u8 {
    hash.get(index).copied().unwrap_or(0)
}

/// Top `bucket_bits` bits of the hash as a register index (bucket_bits ≤ 18 → 3 bytes suffice).
fn bucket_index(hash: &[u8], bucket_bits: u8) -> usize {
    let val = u32::from_be_bytes([
        0,
        byte_or_zero(hash, 0),
        byte_or_zero(hash, 1),
        byte_or_zero(hash, 2),
    ]);
    (val >> (24 - bucket_bits as u32)) as usize
}

/// Leading zeros in the remaining bits after the bucket index.
fn count_leading_zeros(hash: &[u8], bucket_bits: u8, lz_mask: u32) -> u8 {
    let head = u32::from_be_bytes([
        byte_or_zero(hash, 0),
        byte_or_zero(hash, 1),
        byte_or_zero(hash, 2),
        byte_or_zero(hash, 3),
    ]);
    let masked = head & lz_mask;
    if masked != 0 {
        return masked.leading_zeros() as u8 - bucket_bits;
    }
    let mut count = 32 - bucket_bits;
    for &byte in hash.get(4..).unwrap_or(&[]) {
        if byte != 0 {
            return count + byte.leading_zeros() as u8;
        }
        count += 8;
    }
    count
}

/// Cardinality of the union of two HLLs without mutating either.
fn union_cardinality(a: &HyperLogLog, b: &HyperLogLog) -> f64 {
    let merged: Vec<u8> = a
        .registers
        .iter()
        .zip(b.registers.iter())
        .map(|(&x, &y)| x.max(y))
        .collect();
    estimate_cardinality(&merged)
}

/// Standard HLL estimation with small-range correction (linear counting).
///
/// Large-range correction is omitted: with SHA-256 inputs (242+ remaining bits),
/// the threshold (~2^242) is unreachable in practice. This follows HyperLogLog++
/// (Google, 2013) which also dropped the large-range correction.
fn estimate_cardinality(registers: &[u8]) -> f64 {
    let m = registers.len() as f64;
    let alpha = alpha_m(registers.len());

    // E = α_m × m² / Σ 2^(-M[j])
    //   - 2^(-M[j]) = 1/2^M[j], the reciprocal for harmonic mean
    //   - m / Σ 2^(-M[j]) = harmonic mean of per-bucket estimates 2^M[j]
    //   - × m to scale from per-bucket to total cardinality
    //   - × α_m to correct systematic upward bias
    let sum: f64 = registers.iter().map(|&r| 2.0f64.powi(-(r as i32))).sum();
    let raw_estimate = alpha * m * m / sum;

    if raw_estimate <= 2.5 * m {
        // Small range correction: when cardinality << m, many registers are still 0
        // and the harmonic mean formula loses accuracy. Switch to Linear Counting:
        //
        //   n ≈ m × ln(m / V),  where V = number of zero registers
        //
        // Derivation: m buckets, n distinct balls → empty fraction V/m ≈ e^(-n/m),
        // solving gives n = m × ln(m/V).
        //
        // Threshold 2.5*m is empirically determined (Flajolet et al.).
        // When V=0, Linear Counting diverges (ln(∞)), but the cardinality is no
        // longer "small" so the harmonic mean estimate is accurate enough.
        let zeros = registers.iter().filter(|&&r| r == 0).count() as f64;
        if zeros > 0.0 {
            m * (m / zeros).ln()
        } else {
            raw_estimate
        }
    } else {
        raw_estimate
    }
}

/// Bias-correction constant α_m for HLL, where `m` = number of registers = 2^bucket_bits.
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

// ============================================================================
// Sliding-window HLL tracker
// ============================================================================

/// Metric snapshot returned by [`HllTracker::metric`].
#[derive(Debug, Clone)]
pub struct HllMetric {
    /// Estimated number of distinct block hashes in the window.
    pub cardinality: f64,
    /// Total block requests (including duplicates) in the window.
    pub total_requests: u64,
    /// Estimated hit rate assuming infinite cache: `(total - cardinality) / total`.
    pub estimated_hit_rate: f64,
    /// Number of active time slots in the sliding window.
    pub window_slot_count: usize,
}

struct WindowSlot {
    hll: HyperLogLog,
    start: Instant,
    request_count: u64,
}

/// Sliding-window HyperLogLog tracker for cache hit rate estimation.
///
/// Divides time into fixed-duration slots and maintains a ring of HLLs.
/// The merged cardinality across all active slots approximates the number
/// of distinct blocks requested in the window. From this we derive:
///
/// ```text
/// hit_rate = (total_requests - cardinality) / total_requests
/// ```
///
/// Thread safety: wrap in `Mutex<HllTracker>` at the call site.
pub struct HllTracker {
    slots: VecDeque<WindowSlot>,
    /// Merge of all finalized slots (everything except the active back slot).
    /// Incrementally updated on slot rotation; full recompute only after expiry.
    merged: HyperLogLog,
    /// True when expired slots invalidated `merged` (needs recompute from scratch).
    merged_dirty: bool,
    slot_duration: Duration,
    window_duration: Duration,
    bucket_bits: u8,
}

impl HllTracker {
    /// Create a new tracker.
    ///
    /// - `slot_duration`: how long each time slot lasts (e.g. 1 hour)
    /// - `window_duration`: total sliding window (e.g. 24 hours)
    /// - `bucket_bits`: HLL bucket index bits (4..=18, default 14)
    pub fn new(slot_duration: Duration, window_duration: Duration, bucket_bits: u8) -> Self {
        Self {
            slots: VecDeque::new(),
            merged: HyperLogLog::new(bucket_bits),
            merged_dirty: false,
            slot_duration,
            window_duration,
            bucket_bits,
        }
    }

    /// Record a block hash request.
    ///
    /// Lazily creates/rotates slots. Slot boundaries are aligned to multiples of
    /// `slot_duration` from the first slot, so gaps without requests don't cause
    /// time drift. For example with 1h slots: if the first slot starts at 0:00
    /// and the next request arrives at 1:30, the new slot starts at 1:00 (not 1:30).
    pub fn record(&mut self, hash: &[u8]) {
        let now = Instant::now();

        let need_new_slot = match self.slots.back() {
            None => true,
            Some(s) => now.duration_since(s.start) >= self.slot_duration,
        };

        if need_new_slot {
            self.merged_dirty = true;

            // Align to slot boundary: advance from last slot's start by N × slot_duration
            let aligned_start = match self.slots.back() {
                Some(last) => {
                    let elapsed = now.duration_since(last.start);
                    let periods = elapsed.as_nanos() / self.slot_duration.as_nanos();
                    last.start + self.slot_duration * periods as u32
                }
                None => now,
            };
            self.slots.push_back(WindowSlot {
                hll: HyperLogLog::new(self.bucket_bits),
                start: aligned_start,
                request_count: 0,
            });
        }

        let slot = self.slots.back_mut().unwrap();
        slot.hll.insert(hash);
        slot.request_count += 1;
    }

    /// Record a batch of block hashes from a gRPC request.
    pub fn record_hashes(&mut self, hashes: &[Vec<u8>]) {
        for hash in hashes {
            self.record(hash);
        }
    }

    /// Compute and return the current metric snapshot.
    ///
    /// Triggers slot expiry and merged recomputation if needed.
    pub fn metric(&mut self) -> HllMetric {
        self.expire_old_slots(Instant::now());
        self.ensure_merged();

        // Cardinality = union(merged finalized slots, active back slot)
        let cardinality = match self.slots.back() {
            Some(back) => union_cardinality(&self.merged, &back.hll),
            None => 0.0,
        };
        let total: u64 = self.slots.iter().map(|s| s.request_count).sum();
        let hit_rate = if total > 0 {
            let c = cardinality.min(total as f64);
            (total as f64 - c) / total as f64
        } else {
            0.0
        };

        HllMetric {
            cardinality,
            total_requests: total,
            estimated_hit_rate: hit_rate,
            window_slot_count: self.slots.len(),
        }
    }

    fn expire_old_slots(&mut self, now: Instant) {
        while let Some(front) = self.slots.front() {
            if now.duration_since(front.start) >= self.window_duration {
                self.slots.pop_front();
                self.merged_dirty = true; // can't un-merge, need full recompute
            } else {
                break;
            }
        }
    }

    /// Recompute `merged` from all finalized slots (all except back).
    /// Triggered by slot rotation or expiry; only runs on `metric()` calls.
    fn ensure_merged(&mut self) {
        if !self.merged_dirty {
            return;
        }
        self.merged.clear();
        let finalized = self.slots.len().saturating_sub(1);
        for slot in self.slots.iter().take(finalized) {
            self.merged.merge(&slot.hll);
        }
        self.merged_dirty = false;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn hll_empty_cardinality_is_zero() {
        let hll = HyperLogLog::new(14);
        assert_eq!(hll.cardinality(), 0.0);
    }

    #[test]
    fn hll_single_insert() {
        let mut hll = HyperLogLog::new(14);
        hll.insert(&sha256_like(42));
        assert!(hll.cardinality() >= 0.5); // Should be ~1
    }

    #[test]
    fn hll_accuracy_1000_distinct() {
        let mut hll = HyperLogLog::new(14);
        for i in 0u32..1000 {
            hll.insert(&sha256_like(i));
        }
        let est = hll.cardinality();
        assert!((900.0..1100.0).contains(&est), "expected ~1000, got {est}");
    }

    #[test]
    fn hll_accuracy_10000_distinct() {
        let mut hll = HyperLogLog::new(14);
        for i in 0u32..10_000 {
            hll.insert(&sha256_like(i));
        }
        let est = hll.cardinality();
        assert!(
            (9000.0..11000.0).contains(&est),
            "expected ~10000, got {est}"
        );
    }

    #[test]
    fn hll_duplicates_dont_increase_cardinality() {
        let mut hll = HyperLogLog::new(14);
        let hash = sha256_like(42);
        for _ in 0..1000 {
            hll.insert(&hash);
        }
        assert!(
            hll.cardinality() < 5.0,
            "cardinality should be ~1 for repeated inserts, got {}",
            hll.cardinality()
        );
    }

    #[test]
    fn hll_merge() {
        let mut a = HyperLogLog::new(10);
        let mut b = HyperLogLog::new(10);

        for i in 0u32..500 {
            a.insert(&sha256_like(i));
        }
        for i in 500u32..1000 {
            b.insert(&sha256_like(i));
        }

        let card_a = a.cardinality();
        let card_b = b.cardinality();
        a.merge(&b);
        let card_merged = a.cardinality();

        assert!(card_merged > card_a);
        assert!(card_merged > card_b);
        assert!(
            (800.0..1200.0).contains(&card_merged),
            "expected ~1000, got {card_merged}"
        );
    }

    #[test]
    fn hll_clear() {
        let mut hll = HyperLogLog::new(14);
        for i in 0u32..100 {
            hll.insert(&sha256_like(i));
        }
        assert!(hll.cardinality() > 50.0);
        hll.clear();
        assert_eq!(hll.cardinality(), 0.0);
    }

    #[test]
    #[should_panic(expected = "HLL bucket_bits must be in")]
    fn hll_bucket_bits_too_low() {
        HyperLogLog::new(MIN_BUCKET_BITS - 1);
    }

    #[test]
    #[should_panic(expected = "HLL bucket_bits must be in")]
    fn hll_bucket_bits_too_high() {
        HyperLogLog::new(MAX_BUCKET_BITS + 1);
    }

    // ---- Bit-level helper tests ----

    fn make_hash(first3: [u8; 3]) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = first3[0];
        h[1] = first3[1];
        h[2] = first3[2];
        h
    }

    #[test]
    fn bucket_index_basic() {
        assert_eq!(bucket_index(&make_hash([0xAB, 0, 0]), 4), 0b1010);
        assert_eq!(bucket_index(&make_hash([0xAB, 0xCD, 0]), 8), 0xAB);
        assert_eq!(bucket_index(&make_hash([0xAB, 0xCD, 0]), 12), 0xABC);
        assert_eq!(
            bucket_index(&make_hash([0xAB, 0xCD, 0xEF]), 14),
            0b10_1010_1111_0011
        );
    }

    fn lz(hash: &[u8], bits: u8) -> u8 {
        count_leading_zeros(hash, bits, (1u32 << (32 - bits)) - 1)
    }

    #[test]
    fn count_leading_zeros_all_ones() {
        assert_eq!(lz(&[0xFF; 32], 4), 0);
    }

    #[test]
    fn count_leading_zeros_all_zero() {
        // 32-14=18 from head, then 28 zero bytes → 18 + 224 = 242
        assert_eq!(lz(&[0u8; 32], 14), 242);
    }

    #[test]
    fn count_leading_zeros_hit_in_head() {
        let mut h = [0u8; 32];
        h[1] = 0x01; // head=0x00010000, masked lz=15 - 14 = 1
        assert_eq!(lz(&h, 14), 1);
    }

    #[test]
    fn count_leading_zeros_hit_in_tail() {
        let mut h = [0u8; 32];
        h[5] = 0x80; // 18 from head + (5-4)*8 + 0 = 26
        assert_eq!(lz(&h, 14), 26);
    }

    #[test]
    fn count_leading_zeros_byte_aligned() {
        let mut h = [0u8; 32];
        h[1] = 0x01; // head masked lz=15 - 8 = 7
        assert_eq!(lz(&h, 8), 7);
    }

    // ---- HllTracker tests ----

    #[test]
    fn tracker_empty_metric() {
        let mut tracker =
            HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);
        let m = tracker.metric();
        assert_eq!(m.cardinality, 0.0);
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.estimated_hit_rate, 0.0);
        assert_eq!(m.window_slot_count, 0);
    }

    #[test]
    fn tracker_records_and_reports() {
        let mut tracker =
            HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

        // Insert 100 distinct hashes, each 10 times
        for i in 0u32..100 {
            let hash = sha256_like(i);
            for _ in 0..10 {
                tracker.record(&hash);
            }
        }

        let m = tracker.metric();
        assert_eq!(m.total_requests, 1000);
        assert!(
            m.estimated_hit_rate > 0.80,
            "expected high hit rate, got {}",
            m.estimated_hit_rate
        );
        assert!(
            (80.0..120.0).contains(&m.cardinality),
            "expected ~100 cardinality, got {}",
            m.cardinality
        );
        assert_eq!(m.window_slot_count, 1);
    }

    #[test]
    fn tracker_all_unique_low_hit_rate() {
        let mut tracker =
            HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

        for i in 0u32..1000 {
            tracker.record(&sha256_like(i));
        }

        let m = tracker.metric();
        assert_eq!(m.total_requests, 1000);
        assert!(
            m.estimated_hit_rate < 0.1,
            "expected low hit rate for all unique, got {}",
            m.estimated_hit_rate
        );
    }

    #[test]
    fn tracker_bucket_rotation() {
        let mut tracker = HllTracker::new(Duration::from_millis(1), Duration::from_secs(86400), 10);

        tracker.record(&sha256_like(0));
        std::thread::sleep(Duration::from_millis(5));
        tracker.record(&sha256_like(1));

        let m = tracker.metric();
        assert_eq!(m.total_requests, 2);
        assert!(
            m.window_slot_count >= 2,
            "expected >= 2 slots, got {}",
            m.window_slot_count
        );
    }

    #[test]
    fn tracker_bucket_expiry() {
        let mut tracker = HllTracker::new(Duration::from_millis(1), Duration::from_millis(10), 10);

        for i in 0u32..10 {
            tracker.record(&sha256_like(i));
        }

        std::thread::sleep(Duration::from_millis(20));
        tracker.record(&sha256_like(100));

        let m = tracker.metric();
        assert_eq!(m.total_requests, 1);
        assert_eq!(m.window_slot_count, 1);
    }

    #[test]
    fn tracker_hit_rate_50_percent() {
        let mut tracker =
            HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

        // 10000 distinct hashes, each inserted twice → total 20000, cardinality ~10000
        // Expected hit rate ≈ (20000 - 10000) / 20000 = 0.50
        for i in 0u32..10_000 {
            let hash = sha256_like(i);
            tracker.record(&hash);
            tracker.record(&hash);
        }

        let m = tracker.metric();
        println!(
            "tracker_hit_rate_50_percent: cardinality={:.2}, total={}, hit_rate={:.4}",
            m.cardinality, m.total_requests, m.estimated_hit_rate
        );
        assert_eq!(m.total_requests, 20_000);
        assert!(
            (0.49..0.51).contains(&m.estimated_hit_rate),
            "expected ~0.50 hit rate, got {:.4}",
            m.estimated_hit_rate
        );
    }

    #[test]
    fn tracker_hit_rate_66_percent() {
        let mut tracker =
            HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

        // 10000 distinct hashes, each inserted 3 times → total 30000, cardinality ~10000
        // Expected hit rate ≈ (30000 - 10000) / 30000 = 0.6667
        for i in 0u32..10_000 {
            let hash = sha256_like(i);
            tracker.record(&hash);
            tracker.record(&hash);
            tracker.record(&hash);
        }

        let m = tracker.metric();
        assert_eq!(m.total_requests, 30_000);
        assert!(
            (0.656..0.676).contains(&m.estimated_hit_rate),
            "expected ~0.6667 hit rate, got {:.4}",
            m.estimated_hit_rate
        );
    }

    #[test]
    fn hll_distinct_count_scaling() {
        for &n in &[100u32, 1_000, 5_000] {
            let mut hll = HyperLogLog::new(14);
            let mut seen = HashSet::new();
            for i in 0..n {
                let hash = sha256_like(i);
                hll.insert(&hash);
                seen.insert(hash);
            }
            let est = hll.cardinality();
            let actual = seen.len() as f64;
            let error = (est - actual).abs() / actual;
            assert!(
                error < 0.10,
                "n={n}: estimated={est:.0}, actual={actual:.0}, error={error:.4}"
            );
        }
    }

    /// Generate a pseudo-SHA256 hash from an integer for testing.
    fn sha256_like(n: u32) -> [u8; 32] {
        let mut hash = [0u8; 32];
        let m0 = splitmix64(n as u64);
        hash[..8].copy_from_slice(&m0.to_le_bytes());
        let m1 = splitmix64(m0);
        hash[8..16].copy_from_slice(&m1.to_le_bytes());
        let m2 = splitmix64(m1);
        hash[16..24].copy_from_slice(&m2.to_le_bytes());
        let m3 = splitmix64(m2);
        hash[24..32].copy_from_slice(&m3.to_le_bytes());
        hash
    }

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}
