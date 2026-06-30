use std::time::Duration;

// Neon storage_broker client defaults.
pub const GRPC_CONNECT_TIMEOUT: Duration = Duration::from_millis(5000);
pub const GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL: Duration = Duration::from_millis(5000);

// Neon pageserver gRPC server defaults.
pub const GRPC_SERVER_HTTP2_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
pub const GRPC_SERVER_HTTP2_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(20);
