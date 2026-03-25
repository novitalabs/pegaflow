/// Parse memory size with optional unit suffix (kb, mb, gb, tb)
/// Examples: "10gb", "500MB", "1TB", "512kb", "1073741824"
pub(crate) fn parse_memory_size(s: &str) -> Result<usize, String> {
    let s = s.trim();

    // Find the boundary between number and unit
    let boundary = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());

    let (num_str, unit_str) = s.split_at(boundary);

    // Parse the number
    let value: f64 = num_str
        .trim()
        .parse()
        .map_err(|_| format!("Invalid number: '{}'", num_str))?;

    if value < 0.0 {
        return Err("Memory size cannot be negative".to_string());
    }

    // Apply unit multiplier
    let unit = unit_str.trim().to_lowercase();
    let multiplier: u64 = match unit.as_str() {
        "" | "b" => 1,
        "kb" => 1024,
        "mb" => 1024 * 1024,
        "gb" => 1024 * 1024 * 1024,
        "tb" => 1024 * 1024 * 1024 * 1024,
        _ => return Err(format!("Unknown unit: '{}'. Use kb, mb, gb, or tb", unit)),
    };

    let bytes = (value * multiplier as f64) as u64;

    usize::try_from(bytes)
        .map_err(|_| format!("Memory size {} bytes exceeds maximum supported size", bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_memory_size_basic() {
        assert_eq!(parse_memory_size("1024").unwrap(), 1024);
        assert_eq!(parse_memory_size("1gb").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(
            parse_memory_size("1.5gb").unwrap(),
            (1.5 * 1024.0 * 1024.0 * 1024.0) as usize
        );
    }

    #[test]
    fn test_parse_memory_size_invalid() {
        assert!(parse_memory_size("").is_err());
        assert!(parse_memory_size("abc").is_err());
        assert!(parse_memory_size("-10gb").is_err());
    }
}
