pub mod proto {
    #[allow(
        clippy::allow_attributes_without_reason,
        reason = "prost/tonic generated modules emit allow attributes"
    )]
    pub mod engine {
        tonic::include_proto!("pegaflow");
    }
}

pub const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
