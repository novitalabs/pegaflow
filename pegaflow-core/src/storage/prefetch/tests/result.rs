use std::sync::Arc;

use crate::storage::backing_tier::TierSource;
use crate::storage::prefetch::task::{TierAttempt, assemble};

use super::{block, key};

#[test]
fn preserves_order() {
    let local = block();
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let b1 = block();
    let b2 = block();
    let b3 = block();

    let result = assemble(
        vec![Arc::clone(&local)],
        vec![k1.clone(), k2.clone(), k3.clone()],
        Some(TierAttempt {
            source: TierSource::Ssd,
            committed: 3,
            blocks: vec![
                (k2, Arc::clone(&b2)),
                (k1, Arc::clone(&b1)),
                (k3, Arc::clone(&b3)),
            ],
        }),
    );

    assert_eq!(result.ready_blocks.len(), 4);
    assert!(Arc::ptr_eq(&result.ready_blocks[0], &local));
    assert!(Arc::ptr_eq(&result.ready_blocks[1], &b1));
    assert!(Arc::ptr_eq(&result.ready_blocks[2], &b2));
    assert!(Arc::ptr_eq(&result.ready_blocks[3], &b3));
    assert_eq!(result.missing, 0);
}

#[test]
fn stops_at_gap() {
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let b1 = block();
    let b3 = block();

    let result = assemble(
        Vec::new(),
        vec![k1.clone(), k2, k3.clone()],
        Some(TierAttempt {
            source: TierSource::Ssd,
            committed: 3,
            blocks: vec![(k3, b3), (k1, Arc::clone(&b1))],
        }),
    );

    assert_eq!(result.ready_blocks.len(), 1);
    assert!(Arc::ptr_eq(&result.ready_blocks[0], &b1));
    assert_eq!(result.missing, 2);
}
