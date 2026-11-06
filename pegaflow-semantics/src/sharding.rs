//! Sharding declarations and the seal plan derived from them.
//!
//! Each worker rank declares, per layer it holds, which part of the layer's
//! per-block data it owns. [`seal_plan`] merges every rank's declaration into
//! the list of fragments a block of each group must collect before it is
//! complete ("sealed"), and rejects declarations with gaps or overlaps.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Which part of a layer's per-block data one rank holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shard {
    /// Every declaring rank holds identical bytes (e.g. MLA latent under TP).
    Replicated,
    /// Heads split into `count` equal parts; this rank holds part `index`.
    /// Several ranks may declare the same part (kv heads < TP ranks); they
    /// hold identical bytes.
    Heads { index: u32, count: u32 },
    /// Tokens of a block dealt round-robin in runs of `interleave` across
    /// `count` ranks; this rank holds residue `index` (DCP). Whether the
    /// group's `block_tokens` is the per-rank or the logical block size is
    /// pinned down when a DCP model is onboarded.
    Tokens {
        index: u32,
        count: u32,
        interleave: u32,
    },
}

impl Shard {
    /// Part id within the layer: identical parts hold identical bytes.
    fn part(&self) -> u32 {
        match *self {
            Shard::Replicated => 0,
            Shard::Heads { index, .. } | Shard::Tokens { index, .. } => index,
        }
    }

    /// Number of distinct parts the layer is split into.
    fn parts(&self) -> u32 {
        match *self {
            Shard::Replicated => 1,
            Shard::Heads { count, .. } | Shard::Tokens { count, .. } => count,
        }
    }

    /// Same sharding scheme, ignoring which part this rank holds.
    fn same_scheme(&self, other: &Shard) -> bool {
        match (*self, *other) {
            (Shard::Replicated, Shard::Replicated) => true,
            (Shard::Heads { count: a, .. }, Shard::Heads { count: b, .. }) => a == b,
            (
                Shard::Tokens {
                    count: a,
                    interleave: ia,
                    ..
                },
                Shard::Tokens {
                    count: b,
                    interleave: ib,
                    ..
                },
            ) => a == b && ia == ib,
            _ => false,
        }
    }
}

/// One layer as held by one rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub layer: String,
    pub group: u32,
    /// Byte size of each segment of one block on this rank (e.g. `[K, V]`,
    /// or `[conv, ssm]`). Every part of a layer has the same sizes: shards
    /// split evenly, so a mismatch means a rank declared the wrong share.
    pub segments: Vec<u64>,
    pub shard: Shard,
}

/// Everything one rank holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankDecl {
    /// Dense worker id within the instance (0..world_size).
    pub rank: u32,
    pub regions: Vec<Region>,
}

/// One piece of a sealed block: one part of one layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    pub layer: String,
    pub part: u32,
    pub segments: Vec<u64>,
    /// Ranks holding this fragment, ascending. All of them may load it. The
    /// first is the designated writer so replicas are copied out only once;
    /// this is a de-duplication hint, and the store must accept the copy from
    /// any holder (a writer that skips a save must not block sealing).
    pub holders: Vec<u32>,
}

impl Fragment {
    pub fn writer(&self) -> u32 {
        self.holders[0]
    }

    pub fn bytes(&self) -> u64 {
        self.segments.iter().sum()
    }
}

/// Fragments a block of one group needs, in a fixed order (layer name, then
/// part). The position of a fragment is its slot within the block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPlan {
    pub group: u32,
    pub fragments: Vec<Fragment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealPlan {
    /// Ascending by group id.
    pub groups: Vec<GroupPlan>,
}

impl SealPlan {
    pub fn group(&self, group: u32) -> Option<&GroupPlan> {
        self.groups.iter().find(|g| g.group == group)
    }

    /// `(group, slot)` of the fragment `rank` holds for `layer`.
    pub fn slot_of(&self, rank: u32, layer: &str) -> Option<(u32, usize)> {
        self.groups.iter().find_map(|g| {
            g.fragments
                .iter()
                .position(|f| f.layer == layer && f.holders.contains(&rank))
                .map(|slot| (g.group, slot))
        })
    }

    /// True if `rank` is the designated writer of `(group, slot)`.
    pub fn is_writer(&self, rank: u32, group: u32, slot: usize) -> bool {
        self.group(group)
            .and_then(|g| g.fragments.get(slot))
            .is_some_and(|f| f.writer() == rank)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SealError {
    Empty,
    DuplicateRank(u32),
    DuplicateLayer {
        rank: u32,
        layer: String,
    },
    InvalidShard {
        rank: u32,
        layer: String,
        shard: Shard,
    },
    EmptySegment {
        rank: u32,
        layer: String,
    },
    GroupMismatch {
        layer: String,
        groups: (u32, u32),
    },
    SchemeMismatch {
        layer: String,
        shards: (Shard, Shard),
    },
    SegmentMismatch {
        layer: String,
        part: u32,
    },
    MissingPart {
        layer: String,
        part: u32,
    },
}

impl fmt::Display for SealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SealError::Empty => write!(f, "no rank declared any layer"),
            SealError::DuplicateRank(rank) => write!(f, "rank {rank} declared twice"),
            SealError::DuplicateLayer { rank, layer } => {
                write!(f, "rank {rank} declared layer {layer} twice")
            }
            SealError::InvalidShard { rank, layer, shard } => {
                write!(f, "rank {rank} layer {layer}: invalid shard {shard:?}")
            }
            SealError::EmptySegment { rank, layer } => {
                write!(
                    f,
                    "rank {rank} layer {layer}: no segments or a zero-byte segment"
                )
            }
            SealError::GroupMismatch { layer, groups } => write!(
                f,
                "layer {layer} declared in group {} and group {}",
                groups.0, groups.1
            ),
            SealError::SchemeMismatch { layer, shards } => write!(
                f,
                "layer {layer} declared with incompatible shards {:?} and {:?}",
                shards.0, shards.1
            ),
            SealError::SegmentMismatch { layer, part } => write!(
                f,
                "layer {layer} part {part}: segment sizes differ from other parts or ranks"
            ),
            SealError::MissingPart { layer, part } => {
                write!(f, "layer {layer} part {part}: no rank holds it")
            }
        }
    }
}

impl std::error::Error for SealError {}

/// Merge every rank's declaration into the per-group fragment lists.
///
/// Identical parts collapse into one fragment with several holders. Fails
/// if a layer's parts do not cover the whole layer, if ranks disagree on a
/// layer's group, scheme, or segment sizes, or if declarations are malformed.
/// The result depends only on the set of declarations, not their order, so
/// every process computing it agrees on slot numbers.
pub fn seal_plan(ranks: &[RankDecl]) -> Result<SealPlan, SealError> {
    struct LayerAcc<'a> {
        group: u32,
        scheme: Shard,
        segments: &'a [u64],
        // part -> holders
        parts: BTreeMap<u32, BTreeSet<u32>>,
    }

    let mut seen_ranks = BTreeSet::new();
    let mut layers: BTreeMap<&str, LayerAcc<'_>> = BTreeMap::new();

    for decl in ranks {
        if !seen_ranks.insert(decl.rank) {
            return Err(SealError::DuplicateRank(decl.rank));
        }
        let mut own_layers = BTreeSet::new();
        for region in &decl.regions {
            if !own_layers.insert(region.layer.as_str()) {
                return Err(SealError::DuplicateLayer {
                    rank: decl.rank,
                    layer: region.layer.clone(),
                });
            }
            let shard = region.shard;
            let valid = match shard {
                Shard::Replicated => true,
                Shard::Heads { index, count } => count > 0 && index < count,
                Shard::Tokens {
                    index,
                    count,
                    interleave,
                } => count > 0 && index < count && interleave > 0,
            };
            if !valid {
                return Err(SealError::InvalidShard {
                    rank: decl.rank,
                    layer: region.layer.clone(),
                    shard,
                });
            }
            if region.segments.is_empty() || region.segments.contains(&0) {
                return Err(SealError::EmptySegment {
                    rank: decl.rank,
                    layer: region.layer.clone(),
                });
            }

            let acc = layers
                .entry(region.layer.as_str())
                .or_insert_with(|| LayerAcc {
                    group: region.group,
                    scheme: shard,
                    segments: &region.segments,
                    parts: BTreeMap::new(),
                });
            if acc.group != region.group {
                return Err(SealError::GroupMismatch {
                    layer: region.layer.clone(),
                    groups: (acc.group, region.group),
                });
            }
            if !acc.scheme.same_scheme(&shard) {
                return Err(SealError::SchemeMismatch {
                    layer: region.layer.clone(),
                    shards: (acc.scheme, shard),
                });
            }
            if acc.segments != region.segments.as_slice() {
                return Err(SealError::SegmentMismatch {
                    layer: region.layer.clone(),
                    part: shard.part(),
                });
            }
            acc.parts.entry(shard.part()).or_default().insert(decl.rank);
        }
    }

    if layers.is_empty() {
        return Err(SealError::Empty);
    }

    let mut groups: BTreeMap<u32, Vec<Fragment>> = BTreeMap::new();
    for (layer, acc) in layers {
        for part in 0..acc.scheme.parts() {
            let Some(holders) = acc.parts.get(&part) else {
                return Err(SealError::MissingPart {
                    layer: layer.to_string(),
                    part,
                });
            };
            groups.entry(acc.group).or_default().push(Fragment {
                layer: layer.to_string(),
                part,
                segments: acc.segments.to_vec(),
                holders: holders.iter().copied().collect(),
            });
        }
    }

    Ok(SealPlan {
        groups: groups
            .into_iter()
            .map(|(group, fragments)| GroupPlan { group, fragments })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(layer: &str, group: u32, bytes: u64, shard: Shard) -> Region {
        Region {
            layer: layer.to_string(),
            group,
            segments: vec![bytes, bytes],
            shard,
        }
    }

    #[test]
    fn mla_replicas_collapse_to_one_fragment_with_rank0_writer() {
        let ranks: Vec<RankDecl> = (0..4)
            .map(|rank| RankDecl {
                rank,
                regions: vec![region("l0", 0, 64, Shard::Replicated)],
            })
            .collect();
        let plan = seal_plan(&ranks).unwrap();
        let g = plan.group(0).unwrap();
        assert_eq!(g.fragments.len(), 1);
        assert_eq!(g.fragments[0].holders, vec![0, 1, 2, 3]);
        assert!(plan.is_writer(0, 0, 0));
        assert!(!plan.is_writer(2, 0, 0));
        assert_eq!(plan.slot_of(3, "l0"), Some((0, 0)));
    }

    #[test]
    fn heads_with_fewer_kv_heads_than_ranks_dedupe() {
        // 2 kv heads on TP4: ranks {0,1} hold part 0, {2,3} hold part 1.
        let ranks: Vec<RankDecl> = (0..4)
            .map(|rank| RankDecl {
                rank,
                regions: vec![region(
                    "l0",
                    0,
                    32,
                    Shard::Heads {
                        index: rank / 2,
                        count: 2,
                    },
                )],
            })
            .collect();
        let plan = seal_plan(&ranks).unwrap();
        let frags = &plan.group(0).unwrap().fragments;
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[1].holders, vec![2, 3]);
        assert_eq!(plan.slot_of(3, "l0"), Some((0, 1)));
    }

    #[test]
    fn missing_head_part_is_rejected() {
        let ranks = [RankDecl {
            rank: 0,
            regions: vec![region("l0", 0, 32, Shard::Heads { index: 0, count: 2 })],
        }];
        assert_eq!(
            seal_plan(&ranks),
            Err(SealError::MissingPart {
                layer: "l0".into(),
                part: 1
            })
        );
    }

    #[test]
    fn mixed_mla_and_heads_layers_in_one_model() {
        // TP2: MLA layer replicated, GDN layer split by heads, two groups.
        let ranks: Vec<RankDecl> = (0..2)
            .map(|rank| RankDecl {
                rank,
                regions: vec![
                    region("attn.0", 0, 64, Shard::Replicated),
                    region(
                        "gdn.1",
                        1,
                        16,
                        Shard::Heads {
                            index: rank,
                            count: 2,
                        },
                    ),
                ],
            })
            .collect();
        let plan = seal_plan(&ranks).unwrap();
        assert_eq!(plan.group(0).unwrap().fragments.len(), 1);
        assert_eq!(plan.group(1).unwrap().fragments.len(), 2);
    }

    #[test]
    fn pipeline_ranks_contribute_their_own_layers() {
        let ranks = [
            RankDecl {
                rank: 0,
                regions: vec![region("a", 0, 8, Shard::Replicated)],
            },
            RankDecl {
                rank: 1,
                regions: vec![region("b", 0, 8, Shard::Replicated)],
            },
        ];
        let plan = seal_plan(&ranks).unwrap();
        let frags = &plan.group(0).unwrap().fragments;
        assert_eq!(frags.len(), 2);
        assert_eq!(plan.slot_of(1, "b"), Some((0, 1)));
        assert_eq!(plan.slot_of(1, "a"), None);
    }

    #[test]
    fn inconsistent_declarations_are_rejected() {
        let two = |a: Region, b: Region| {
            seal_plan(&[
                RankDecl {
                    rank: 0,
                    regions: vec![a],
                },
                RankDecl {
                    rank: 1,
                    regions: vec![b],
                },
            ])
        };
        assert!(matches!(
            two(
                region("l", 0, 8, Shard::Replicated),
                region("l", 1, 8, Shard::Replicated)
            ),
            Err(SealError::GroupMismatch { .. })
        ));
        assert!(matches!(
            two(
                region("l", 0, 8, Shard::Replicated),
                region("l", 0, 8, Shard::Heads { index: 0, count: 1 })
            ),
            Err(SealError::SchemeMismatch { .. })
        ));
        assert!(matches!(
            two(
                region("l", 0, 8, Shard::Replicated),
                region("l", 0, 16, Shard::Replicated)
            ),
            Err(SealError::SegmentMismatch { .. })
        ));
        assert!(matches!(
            two(
                region("l", 0, 8, Shard::Heads { index: 2, count: 2 }),
                region("l", 0, 8, Shard::Heads { index: 1, count: 2 })
            ),
            Err(SealError::InvalidShard { .. })
        ));
        // Different parts of one layer must still have equal sizes.
        assert!(matches!(
            two(
                region("l", 0, 8, Shard::Heads { index: 0, count: 2 }),
                region("l", 0, 16, Shard::Heads { index: 1, count: 2 })
            ),
            Err(SealError::SegmentMismatch { .. })
        ));
        assert!(matches!(
            two(
                region("l", 0, 0, Shard::Replicated),
                region("l", 0, 0, Shard::Replicated)
            ),
            Err(SealError::EmptySegment { .. })
        ));
    }

    #[test]
    fn token_shards_need_every_residue_and_matching_interleave() {
        let tokens = |index, interleave| Shard::Tokens {
            index,
            count: 2,
            interleave,
        };
        let decl = |rank, shard| RankDecl {
            rank,
            regions: vec![region("l", 0, 8, shard)],
        };
        let plan = seal_plan(&[decl(1, tokens(1, 1)), decl(0, tokens(0, 1))]).unwrap();
        let holders: Vec<_> = plan.groups[0]
            .fragments
            .iter()
            .map(|f| f.holders.clone())
            .collect();
        assert_eq!(holders, vec![vec![0], vec![1]]);
        assert!(matches!(
            seal_plan(&[decl(0, tokens(0, 1)), decl(1, tokens(1, 2))]),
            Err(SealError::SchemeMismatch { .. })
        ));
        assert!(matches!(
            seal_plan(&[decl(0, tokens(0, 1))]),
            Err(SealError::MissingPart { part: 1, .. })
        ));
    }
}
