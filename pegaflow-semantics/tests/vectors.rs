//! Shared semantics vectors. The same JSON files are replayed by the Python
//! bindings, so both sides agree on every case.

use pegaflow_semantics::{
    Group, Kind, RankDecl, Region, SealError, Shard, key, resolve, saveable, seal_plan,
};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum KindVec {
    Full,
    Window { tokens: u32 },
    Checkpoint,
    Scratch,
}

#[derive(Deserialize)]
struct GroupVec {
    #[serde(flatten)]
    kind: KindVec,
    block_tokens: u32,
}

#[derive(Deserialize)]
struct SaveVec {
    before: u32,
    after: u32,
    ranges: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
struct ResolveVec {
    id: String,
    groups: Vec<GroupVec>,
    present: Vec<Vec<u32>>,
    prompt_len: u32,
    resolve: u32,
    #[serde(default)]
    saveable: Vec<SaveVec>,
}

#[derive(Deserialize)]
struct KeyVec {
    id: String,
    hash: Vec<u8>,
    group: u32,
    key: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ShardVec {
    Replicated,
    Heads {
        index: u32,
        count: u32,
    },
    Tokens {
        index: u32,
        count: u32,
        interleave: u32,
    },
}

#[derive(Deserialize)]
struct RegionVec {
    layer: String,
    group: u32,
    segments: Vec<u64>,
    shard: ShardVec,
}

#[derive(Deserialize)]
struct RankVec {
    rank: u32,
    regions: Vec<RegionVec>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct FragmentVec {
    layer: String,
    part: u32,
    holders: Vec<u32>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct GroupPlanVec {
    group: u32,
    fragments: Vec<FragmentVec>,
}

#[derive(Deserialize)]
struct SealVec {
    id: String,
    ranks: Vec<RankVec>,
    #[serde(default)]
    groups: Vec<GroupPlanVec>,
    error: Option<String>,
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> Vec<T> {
    let path = format!("{}/vectors/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn group(v: &GroupVec) -> Group {
    let kind = match v.kind {
        KindVec::Full => Kind::Full,
        KindVec::Window { tokens } => Kind::Window { tokens },
        KindVec::Checkpoint => Kind::Checkpoint,
        KindVec::Scratch => Kind::Scratch,
    };
    Group::new(kind, v.block_tokens).unwrap()
}

#[test]
fn resolve_and_saveable_vectors() {
    for case in load::<ResolveVec>("resolve.json") {
        let groups: Vec<Group> = case.groups.iter().map(group).collect();
        assert_eq!(
            resolve(&groups, &case.present, case.prompt_len),
            case.resolve,
            "resolve: {}",
            case.id
        );
        for s in &case.saveable {
            assert_eq!(s.ranges.len(), groups.len(), "saveable arity: {}", case.id);
            for (g, [lo, hi]) in groups.iter().zip(&s.ranges) {
                let got = saveable(g, s.before, s.after);
                let want = *lo..*hi;
                // Empty ranges compare by emptiness, not bounds.
                assert!(
                    got == want || (got.is_empty() && want.is_empty()),
                    "saveable {}..{} {:?}: {} got {got:?} want {want:?}",
                    s.before,
                    s.after,
                    g.kind(),
                    case.id
                );
            }
        }
    }
}

#[test]
fn key_vectors() {
    for case in load::<KeyVec>("key.json") {
        assert_eq!(key(&case.hash, case.group), case.key, "key: {}", case.id);
    }
}

fn error_name(e: &SealError) -> &'static str {
    match e {
        SealError::Empty => "empty",
        SealError::DuplicateRank(_) => "duplicate_rank",
        SealError::DuplicateLayer { .. } => "duplicate_layer",
        SealError::InvalidShard { .. } => "invalid_shard",
        SealError::EmptySegment { .. } => "empty_segment",
        SealError::GroupMismatch { .. } => "group_mismatch",
        SealError::SchemeMismatch { .. } => "scheme_mismatch",
        SealError::SegmentMismatch { .. } => "segment_mismatch",
        SealError::MissingPart { .. } => "missing_part",
    }
}

#[test]
fn seal_plan_vectors() {
    for case in load::<SealVec>("seal_plan.json") {
        let decls: Vec<RankDecl> = case
            .ranks
            .into_iter()
            .map(|r| RankDecl {
                rank: r.rank,
                regions: r
                    .regions
                    .into_iter()
                    .map(|g| Region {
                        layer: g.layer,
                        group: g.group,
                        segments: g.segments,
                        shard: match g.shard {
                            ShardVec::Replicated => Shard::Replicated,
                            ShardVec::Heads { index, count } => Shard::Heads { index, count },
                            ShardVec::Tokens {
                                index,
                                count,
                                interleave,
                            } => Shard::Tokens {
                                index,
                                count,
                                interleave,
                            },
                        },
                    })
                    .collect(),
            })
            .collect();
        match (seal_plan(&decls), case.error) {
            (Ok(plan), None) => {
                let got: Vec<GroupPlanVec> = plan
                    .groups
                    .into_iter()
                    .map(|g| GroupPlanVec {
                        group: g.group,
                        fragments: g
                            .fragments
                            .into_iter()
                            .map(|f| FragmentVec {
                                layer: f.layer,
                                part: f.part,
                                holders: f.holders,
                            })
                            .collect(),
                    })
                    .collect();
                assert_eq!(got, case.groups, "seal_plan: {}", case.id);
            }
            (Err(e), Some(want)) => assert_eq!(error_name(&e), want, "seal_plan: {}", case.id),
            (got, want) => panic!("seal_plan {}: got {got:?}, want error {want:?}", case.id),
        }
    }
}
