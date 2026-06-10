mod verbs_address;
mod verbs_devinfo;
mod verbs_domain;
mod verbs_qp;
mod verbs_rdma_op;

pub(crate) use verbs_devinfo::{VerbsDeviceInfo, VerbsDeviceList};
pub(crate) use verbs_domain::VerbsDomain;

pub(super) unsafe fn zeroed<T>() -> T {
    unsafe { std::mem::zeroed() }
}
