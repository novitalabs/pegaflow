pub mod proto {
    #[allow(
        clippy::allow_attributes_without_reason,
        reason = "prost/tonic generated modules emit allow attributes"
    )]
    pub mod engine {
        tonic::include_proto!("pegaflow");
    }
}

pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+native-vmm-v1");
