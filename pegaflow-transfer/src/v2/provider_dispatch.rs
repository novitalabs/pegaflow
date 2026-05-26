use std::borrow::Cow;

use crate::v2::{provider::RdmaDomainInfo, verbs::VerbsDeviceInfo};

// NOTE: This enum was originally a dispatch point between multiple RDMA
// providers (EFA + Verbs upstream). The non-Verbs provider was removed during
// the port, but we keep the enum so that future fabric providers can be added
// without breaking the public API surface.
#[derive(Clone)]
enum DomainInfoInner {
    Verbs(VerbsDeviceInfo),
}

#[derive(Clone)]
pub struct DomainInfo {
    inner: DomainInfoInner,
}

impl DomainInfo {
    pub(crate) fn verbs(info: VerbsDeviceInfo) -> Self {
        Self {
            inner: DomainInfoInner::Verbs(info),
        }
    }

    pub fn name(&self) -> Cow<'_, str> {
        match &self.inner {
            DomainInfoInner::Verbs(info) => info.name(),
        }
    }

    pub fn link_speed(&self) -> u64 {
        match &self.inner {
            DomainInfoInner::Verbs(info) => info.link_speed(),
        }
    }

    pub(crate) fn as_verbs(&self) -> &VerbsDeviceInfo {
        match &self.inner {
            DomainInfoInner::Verbs(info) => info,
        }
    }

    pub(crate) fn into_verbs(self) -> VerbsDeviceInfo {
        match self.inner {
            DomainInfoInner::Verbs(info) => info,
        }
    }
}

impl RdmaDomainInfo for DomainInfo {
    fn name(&self) -> Cow<'_, str> {
        self.name()
    }

    fn link_speed(&self) -> u64 {
        self.link_speed()
    }
}
