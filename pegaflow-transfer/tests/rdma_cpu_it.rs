use std::{env, path::Path, time::Instant};

use pegaflow_transfer::MooncakeTransferEngine;

const ENV_RDMA_NIC: &str = "PEGAFLOW_TRANSFER_IT_NIC";
const ENV_BASE_PORT: &str = "PEGAFLOW_TRANSFER_IT_BASE_PORT";
const ENV_TRANSFER_BYTES: &str = "PEGAFLOW_TRANSFER_IT_BYTES";

const DEFAULT_RDMA_NIC: &str = "mlx5_1";
const DEFAULT_BASE_PORT: u16 = 56080;
const DEFAULT_TRANSFER_BYTES: usize = 4 << 20;
const HANDSHAKE_BYTES: usize = 4096;

fn read_env(name: &str) -> Option<String> {
    env::var(name).ok().and_then(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

fn fill_pattern(buf: &mut [u8], seed: u8) {
    for (idx, byte) in buf.iter_mut().enumerate() {
        *byte = seed.wrapping_add((idx % 251) as u8);
    }
}

#[test]
#[ignore = "requires RDMA NIC; defaults to mlx5_1 or set PEGAFLOW_TRANSFER_IT_NIC"]
fn it_rdma_cpu_ud_handshake_rc_write_read() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();

    let nic_from_env = read_env(ENV_RDMA_NIC);
    let nic_name = nic_from_env
        .clone()
        .unwrap_or_else(|| DEFAULT_RDMA_NIC.to_string());
    if !Path::new("/sys/class/infiniband").join(&nic_name).exists() {
        if nic_from_env.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{ENV_RDMA_NIC}='{}' does not exist", nic_name),
            )
            .into());
        }
        eprintln!(
            "skip: default RDMA NIC '{}' does not exist; set {} to a valid NIC",
            nic_name, ENV_RDMA_NIC
        );
        return Ok(());
    }

    let base_port = read_env(ENV_BASE_PORT)
        .map(|value| value.parse::<u16>())
        .transpose()?
        .unwrap_or(DEFAULT_BASE_PORT);
    let bytes = read_env(ENV_TRANSFER_BYTES)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_TRANSFER_BYTES);
    let peer_port = base_port.checked_add(1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{ENV_BASE_PORT} must be <= {}", u16::MAX - 1),
        )
    })?;
    if bytes == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{ENV_TRANSFER_BYTES} must be > 0"),
        )
        .into());
    }

    eprintln!(
        "[it] start: nic={} base_port={} bytes={}",
        nic_name, base_port, bytes
    );

    let mut a_src = vec![0_u8; bytes];
    let mut a_dst = vec![0_u8; bytes];
    let mut b_src = vec![0_u8; bytes];
    let mut b_dst = vec![0_u8; bytes];
    fill_pattern(&mut a_src, 0x11);
    fill_pattern(&mut b_src, 0x55);
    a_dst.fill(0xA1);
    b_dst.fill(0xB2);
    let expected_a_src = a_src.clone();
    let expected_b_src = b_src.clone();

    let a_src_ptr = a_src.as_ptr() as u64;
    let a_dst_ptr = a_dst.as_mut_ptr() as u64;
    let b_src_ptr = b_src.as_ptr() as u64;
    let b_dst_ptr = b_dst.as_mut_ptr() as u64;

    let setup_start = Instant::now();
    let mut engine_a = MooncakeTransferEngine::new();
    engine_a.initialize(nic_name.clone(), base_port)?;
    engine_a.batch_register_memory(&[a_src_ptr, a_dst_ptr], &[bytes, bytes])?;

    let mut engine_b = MooncakeTransferEngine::new();
    engine_b.initialize(nic_name, peer_port)?;
    engine_b.batch_register_memory(&[b_src_ptr, b_dst_ptr], &[bytes, bytes])?;
    let b_session = engine_b.get_session_id();
    eprintln!(
        "[it] setup done: session={} elapsed_ms={:.3}",
        b_session,
        setup_start.elapsed().as_secs_f64() * 1_000.0
    );

    // First transfer triggers UD control-plane handshake and RC connection setup.
    let handshake_bytes = bytes.min(HANDSHAKE_BYTES);
    let handshake_written =
        engine_a.transfer_sync_write(&b_session, a_src_ptr, b_dst_ptr, handshake_bytes)?;
    assert_eq!(handshake_written, handshake_bytes);
    assert_eq!(
        &b_dst[..handshake_bytes],
        &expected_a_src[..handshake_bytes]
    );

    let write_start = Instant::now();
    let write_bytes = engine_a.transfer_sync_write(&b_session, a_src_ptr, b_dst_ptr, bytes)?;
    assert_eq!(write_bytes, bytes);
    eprintln!(
        "[it] rc write done: bytes={} elapsed_ms={:.3}",
        write_bytes,
        write_start.elapsed().as_secs_f64() * 1_000.0
    );
    assert_eq!(b_dst, expected_a_src, "A->B RC write payload mismatch");

    let read_start = Instant::now();
    let read_bytes = engine_a.transfer_sync_read(&b_session, a_dst_ptr, b_src_ptr, bytes)?;
    assert_eq!(read_bytes, bytes);
    eprintln!(
        "[it] rc read done: bytes={} elapsed_ms={:.3}",
        read_bytes,
        read_start.elapsed().as_secs_f64() * 1_000.0
    );
    assert_eq!(a_dst, expected_b_src, "B->A RC read payload mismatch");

    engine_a.unregister_memory(a_src_ptr)?;
    engine_a.unregister_memory(a_dst_ptr)?;
    engine_b.unregister_memory(b_src_ptr)?;
    engine_b.unregister_memory(b_dst_ptr)?;

    eprintln!(
        "[it] done: total_elapsed_ms={:.3}",
        total_start.elapsed().as_secs_f64() * 1_000.0
    );
    Ok(())
}
