use std::{
    collections::HashMap,
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering::Relaxed},
    },
};

use parking_lot::RwLock;

use crate::v2::api::ImmCounter;

pub enum ImmCount {
    Expected {
        counter: Arc<AtomicI64>,
        expected: NonZeroU32,
    },
    Imm {
        counter: Arc<AtomicI64>,
    },
}

impl ImmCount {
    pub fn consume(self) -> (u32, Option<NonZeroU32>) {
        match self {
            ImmCount::Expected { counter, expected } => {
                (counter.load(Relaxed) as u32, Some(expected))
            }
            ImmCount::Imm { counter } => (counter.load(Relaxed) as u32, None),
        }
    }

    pub fn inc(&self) -> bool {
        match &self {
            ImmCount::Expected { counter, expected } => {
                let prev = counter.fetch_add(1, Relaxed);
                let reached = prev as u32 + 1 == expected.get();
                if reached {
                    counter.fetch_sub(expected.get() as i64, Relaxed);
                }
                reached
            }
            ImmCount::Imm { counter } => {
                counter.fetch_add(1, Relaxed);
                false
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImmCountStatus {
    Vacant,
    NotReached,
    Reached,
}

pub struct ImmCountMap {
    map: RwLock<HashMap<u32, ImmCount>>,
}

impl ImmCountMap {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    pub fn set_expected(&self, imm: u32, expected: NonZeroU32) -> Option<ImmCount> {
        self.map.write().insert(
            imm,
            ImmCount::Expected {
                counter: Arc::new(AtomicI64::new(0)),
                expected,
            },
        )
    }

    pub fn get_imm_counter(&self, imm: u32) -> ImmCounter {
        let mut map = self.map.write();
        match map.get(&imm) {
            Some(ImmCount::Imm { counter }) => ImmCounter::new(counter.clone()),
            _ => {
                let counter = Arc::new(AtomicI64::new(0));
                map.insert(
                    imm,
                    ImmCount::Imm {
                        counter: counter.clone(),
                    },
                );
                ImmCounter::new(counter)
            }
        }
    }

    pub fn remove(&self, imm: u32) -> Option<ImmCount> {
        self.map.write().remove(&imm)
    }

    pub fn get(&self, imm: u32) -> Option<u32> {
        self.map.read().get(&imm).map(|v| match v {
            ImmCount::Expected { counter, .. } => counter.load(Relaxed) as u32,
            ImmCount::Imm { counter } => counter.load(Relaxed) as u32,
        })
    }

    pub fn get_expected(&self, imm: u32) -> Option<NonZeroU32> {
        let counters = self.map.read();
        match counters.get(&imm)? {
            ImmCount::Expected { expected, .. } => Some(*expected),
            ImmCount::Imm { .. } => None,
        }
    }

    pub fn inc(&self, imm: u32) -> ImmCountStatus {
        if let Some(imm_count) = self.map.read().get(&imm) {
            if imm_count.inc() {
                ImmCountStatus::Reached
            } else {
                ImmCountStatus::NotReached
            }
        } else {
            ImmCountStatus::Vacant
        }
    }
}

impl Default for ImmCountMap {
    fn default() -> Self {
        Self::new()
    }
}
