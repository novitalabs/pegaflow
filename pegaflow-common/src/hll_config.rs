//! Shared HLL deployment configuration parsing.

use std::collections::HashSet;
use std::time::Duration;

use crate::hll::{MAX_BUCKET_BITS, MAX_HLL_WINDOWS, MIN_BUCKET_BITS};

pub const DEFAULT_HLL_WINDOWS: &str = "15m,1h,1d";
pub const DEFAULT_HLL_BUCKET_BITS: u8 = 16;

pub fn parse_hll_bucket_bits(value: &str) -> Result<u8, String> {
    let bucket_bits: u8 = value.parse().map_err(|error| format!("{error}"))?;
    if !(MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits) {
        return Err(format!(
            "HLL bucket_bits must be in {MIN_BUCKET_BITS}..={MAX_BUCKET_BITS}, got {bucket_bits}"
        ));
    }
    Ok(bucket_bits)
}

pub fn parse_hll_windows_arg(value: &str) -> Result<String, String> {
    parse_hll_windows(value)?;
    Ok(value.to_string())
}

/// Parse comma-separated `<number><unit>` windows and canonicalize their labels.
pub fn parse_hll_windows(value: &str) -> Result<Vec<(String, Duration)>, String> {
    let mut windows = Vec::new();
    let mut seen = HashSet::new();
    for (index, token) in value.split(',').map(str::trim).enumerate() {
        if token.is_empty() {
            return Err(format!(
                "--metric-hll-windows contains an empty window at position {}",
                index + 1
            ));
        }
        let duration = parse_duration(token)?;
        if duration < Duration::from_secs(60) {
            return Err(format!("HLL window {token} must be at least 1 minute"));
        }
        if !seen.insert(duration) {
            return Err(format!(
                "duplicate HLL window duration: {token} ({})",
                format_window_label(duration)
            ));
        }
        windows.push((format_window_label(duration), duration));
    }
    if windows.is_empty() {
        return Err("--metric-hll-windows must list at least one window".into());
    }
    if windows.len() > MAX_HLL_WINDOWS {
        return Err(format!(
            "--metric-hll-windows supports at most {MAX_HLL_WINDOWS} windows, got {}",
            windows.len()
        ));
    }
    Ok(windows)
}

fn parse_duration(value: &str) -> Result<Duration, String> {
    let (number, unit) = value.split_at(
        value
            .find(|character: char| !character.is_ascii_digit())
            .ok_or_else(|| format!("missing time unit in {value}"))?,
    );
    let number: u64 = number
        .parse()
        .map_err(|error| format!("invalid number in {value}: {error}"))?;
    let seconds = match unit {
        "s" => number,
        "m" => number
            .checked_mul(60)
            .ok_or_else(|| format!("duration overflows seconds in {value}"))?,
        "h" => number
            .checked_mul(3_600)
            .ok_or_else(|| format!("duration overflows seconds in {value}"))?,
        "d" => number
            .checked_mul(86_400)
            .ok_or_else(|| format!("duration overflows seconds in {value}"))?,
        _ => {
            return Err(format!(
                "unknown time unit {unit:?} in {value}; use s/m/h/d"
            ));
        }
    };
    Ok(Duration::from_secs(seconds))
}

fn format_window_label(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds.is_multiple_of(86_400) {
        format!("{}d", seconds / 86_400)
    } else if seconds.is_multiple_of(3_600) {
        format!("{}h", seconds / 3_600)
    } else if seconds.is_multiple_of(60) {
        format!("{}m", seconds / 60)
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_canonicalized() {
        assert_eq!(
            parse_hll_windows("15m,60m,24h").unwrap(),
            vec![
                ("15m".to_string(), Duration::from_secs(15 * 60)),
                ("1h".to_string(), Duration::from_secs(60 * 60)),
                ("1d".to_string(), Duration::from_secs(24 * 60 * 60)),
            ]
        );
    }

    #[test]
    fn windows_reject_empty_and_duplicate_entries() {
        assert!(
            parse_hll_windows("15m,,1h")
                .unwrap_err()
                .contains("empty window")
        );
        assert!(
            parse_hll_windows("1h,60m")
                .unwrap_err()
                .contains("duplicate HLL window duration")
        );
    }

    #[test]
    fn bucket_bits_use_hll_bounds() {
        assert_eq!(parse_hll_bucket_bits("16").unwrap(), 16);
        assert!(parse_hll_bucket_bits("3").is_err());
        assert!(parse_hll_bucket_bits("19").is_err());
    }
}
