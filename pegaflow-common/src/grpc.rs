use std::time::Duration;

pub const GRPC_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
pub const GRPC_RPC_TIMEOUT: Duration = Duration::from_secs(3);
pub const GRPC_HTTP2_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
pub const GRPC_HTTP2_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);
