#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    pub nic_name: String,
    pub rpc_port: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredMemory {
    pub ptr: u64,
    pub len: usize,
}
