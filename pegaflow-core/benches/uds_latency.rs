use std::path::Path;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

#[tokio::main]
async fn main() {
    let path = "/tmp/uds_latency.sock";

    if Path::new(path).exists() {
        std::fs::remove_file(path).unwrap();
    }

    let listener = UnixListener::bind(path).unwrap();

    // spawn server
    tokio::spawn(async move {
        let mut stream = listener.accept().await.unwrap().0;
        let mut buf = [0u8; 8];

        loop {
            if stream.read_exact(&mut buf).await.is_err() {
                break;
            }
            // echo back
            if stream.write_all(&buf).await.is_err() {
                break;
            }
        }
    });

    // client
    let mut client = UnixStream::connect(path).await.unwrap();

    const N: usize = 20_000;
    let mut latencies = Vec::with_capacity(N);

    for _ in 0..N {
        let payload = 12345678u64.to_ne_bytes();

        let t0 = Instant::now();

        client.write_all(&payload).await.unwrap();

        let mut resp = [0u8; 8];
        client.read_exact(&mut resp).await.unwrap();

        let dt = t0.elapsed();
        latencies.push(dt.as_nanos());
    }

    latencies.sort();

    let p50 = latencies[N / 2] as f64 / 1000.0;
    let p99 = latencies[(N as f64 * 0.99) as usize] as f64 / 1000.0;

    let avg: f64 = latencies.iter().map(|x| *x as f64).sum::<f64>() / N as f64 / 1000.0;

    println!("--- UDS latency ---");
    println!("avg  = {:.3} µs", avg);
    println!("p50  = {:.3} µs", p50);
    println!("p99  = {:.3} µs", p99);
}
