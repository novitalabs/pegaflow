#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerConfig {
    pub(crate) nic_name: String,
    pub(crate) rpc_port: u16,
}
