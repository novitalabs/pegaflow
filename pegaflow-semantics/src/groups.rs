//! Cache groups: what a group needs to resume, and what it may store.

use std::fmt;
use std::ops::Range;

/// How a cache group depends on earlier tokens.
///
/// The only question that matters for offloading is: to continue computing
/// from token `p`, which stored blocks of this group are required?
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Every block before `p` (standard attention, MLA).
    Full,
    /// Only the blocks covering the last `tokens` tokens before `p`
    /// (sliding-window attention). `tokens` is the number of trailing cached
    /// tokens the next token may still attend to. For vLLM's `sliding_window`
    /// W (which counts the current token) the exact value is `W - 1`; passing
    /// W is safe and costs at most one extra block. Must be positive.
    Window { tokens: u32 },
    /// The state captured exactly at `p` (Mamba, GDN, other recurrent state).
    /// The state at `p` is stored under block index `p / block_tokens - 1`,
    /// i.e. keyed by the last block it has absorbed.
    Checkpoint,
    /// Nothing; the group is rebuilt from scratch on resume.
    Scratch,
}

/// One cache group: its dependency kind and the token granularity of a block.
///
/// Fields are private so every `Group` went through [`Group::new`]; the rest of
/// this module divides by `block_tokens` and relies on that check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Group {
    kind: Kind,
    block_tokens: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidGroup {
    ZeroBlockTokens,
    ZeroWindow,
}

impl fmt::Display for InvalidGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidGroup::ZeroBlockTokens => write!(f, "block_tokens must be positive"),
            InvalidGroup::ZeroWindow => write!(f, "window tokens must be positive"),
        }
    }
}

impl std::error::Error for InvalidGroup {}

impl Group {
    pub fn new(kind: Kind, block_tokens: u32) -> Result<Self, InvalidGroup> {
        if block_tokens == 0 {
            return Err(InvalidGroup::ZeroBlockTokens);
        }
        if kind == (Kind::Window { tokens: 0 }) {
            return Err(InvalidGroup::ZeroWindow);
        }
        Ok(Self { kind, block_tokens })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn block_tokens(&self) -> u32 {
        self.block_tokens
    }
}

/// Block indices `g` needs to resume at `p` tokens.
///
/// `p` must be a multiple of `g.block_tokens`; [`resolve`] only probes such
/// positions. Off-boundary positions round down to the last full block.
pub fn needed(g: &Group, p: u32) -> Range<u32> {
    let bt = g.block_tokens;
    let n = p / bt;
    match g.kind {
        Kind::Full => 0..n,
        Kind::Window { tokens } => p.saturating_sub(tokens) / bt..n,
        Kind::Checkpoint => n.saturating_sub(1)..n,
        Kind::Scratch => 0..0,
    }
}

/// Longest `p` at which every group can resume, given the block indices the
/// store holds for each group (`present[i]` belongs to `groups[i]`, any order,
/// duplicates allowed).
///
/// `p` is a multiple of the lcm of the block sizes of all non-scratch groups
/// (scratch groups load nothing, so they do not constrain alignment) and at
/// most `prompt_len - 1`: the engine must compute at least one token itself
/// to produce logits. Returns 0 when nothing is reusable, including when no
/// group needs any stored data (all-scratch layouts have nothing to load, so
/// claiming a hit would skip real compute).
pub fn resolve<P: AsRef<[u32]>>(groups: &[Group], present: &[P], prompt_len: u32) -> u32 {
    assert_eq!(
        groups.len(),
        present.len(),
        "present must have one entry per group"
    );
    let cap = u64::from(prompt_len.saturating_sub(1));
    let mut step = 0u64;
    for g in groups.iter().filter(|g| g.kind != Kind::Scratch) {
        let bt = u64::from(g.block_tokens);
        // Both operands are <= u32::MAX here, so the lcm fits in u64; stop as
        // soon as it exceeds `cap`, before a further step could overflow.
        step = if step == 0 { bt } else { lcm(step, bt) };
        if step > cap {
            return 0;
        }
    }
    if step == 0 {
        return 0;
    }
    let top = (cap / step * step) as u32;
    let step = step as u32;

    let runs: Vec<RunIndex> = groups
        .iter()
        .zip(present)
        .map(|(g, have)| RunIndex::new(have.as_ref(), top / g.block_tokens))
        .collect();

    let mut p = top;
    while p > 0 {
        if groups
            .iter()
            .zip(&runs)
            .all(|(g, run)| run.covers(needed(g, p)))
        {
            return p;
        }
        p -= step;
    }
    0
}

/// Block indices of `g` that became storable when compute advanced from
/// `before` to `after` tokens.
///
/// Full and window groups may store every block completed in between.
/// A checkpoint group holds only its latest state, so it can store just the
/// state at `after`, and only when `after` sits on a block boundary.
///
/// `before` must be the computed-token count at the start of the step that
/// reached `after`. For window groups the engine may free blocks that slid
/// out of the window once compute moves on, so saving a range computed from a
/// stale `before` can read recycled GPU memory.
pub fn saveable(g: &Group, before: u32, after: u32) -> Range<u32> {
    if after <= before {
        return 0..0;
    }
    let bt = g.block_tokens;
    match g.kind {
        Kind::Full | Kind::Window { .. } => before / bt..after / bt,
        Kind::Checkpoint if after.is_multiple_of(bt) => after / bt - 1..after / bt,
        Kind::Checkpoint | Kind::Scratch => 0..0,
    }
}

/// Length of the contiguous run of present blocks ending at each index, so
/// "is every block in `a..b` present" is O(1).
struct RunIndex {
    run_ending_at: Vec<u32>,
}

impl RunIndex {
    fn new(have: &[u32], limit: u32) -> Self {
        // Indices past the last present block are never covered; do not
        // allocate for them.
        let limit = have
            .iter()
            .filter(|&&b| b < limit)
            .max()
            .map_or(0, |&b| b + 1);
        let mut present = vec![false; limit as usize];
        for &b in have {
            if b < limit {
                present[b as usize] = true;
            }
        }
        let mut run_ending_at = Vec::with_capacity(present.len());
        let mut run = 0u32;
        for hit in present {
            run = if hit { run + 1 } else { 0 };
            run_ending_at.push(run);
        }
        Self { run_ending_at }
    }

    fn covers(&self, range: Range<u32>) -> bool {
        if range.is_empty() {
            return true;
        }
        match self.run_ending_at.get(range.end as usize - 1) {
            Some(&run) => run >= range.end - range.start,
            None => false,
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(bt: u32) -> Group {
        Group::new(Kind::Full, bt).unwrap()
    }

    fn checkpoint(bt: u32) -> Group {
        Group::new(Kind::Checkpoint, bt).unwrap()
    }

    #[test]
    fn worked_example_from_design_doc() {
        let groups = [full(4), checkpoint(4)];
        let present: [&[u32]; 2] = [&[0, 1, 2, 3], &[1, 2]];
        assert_eq!(resolve(&groups, &present, 21), 12);
        assert_eq!(saveable(&groups[0], 12, 20), 3..5);
        assert_eq!(saveable(&groups[1], 12, 20), 4..5);
    }

    /// Reference resolve: probe every boundary with set lookups.
    fn brute_resolve(groups: &[Group], present: &[Vec<u32>], prompt_len: u32) -> u32 {
        if groups.iter().all(|g| g.kind == Kind::Scratch) {
            return 0;
        }
        let step = groups
            .iter()
            .filter(|g| g.kind != Kind::Scratch)
            .fold(1u64, |a, g| lcm(a, u64::from(g.block_tokens))) as u32;
        let mut best = 0;
        let mut p = step;
        while p < prompt_len {
            let ok = groups
                .iter()
                .zip(present)
                .all(|(g, have)| needed(g, p).all(|b| have.contains(&b)));
            if ok {
                best = p;
            }
            p += step;
        }
        best
    }

    #[test]
    fn run_index_matches_brute_force() {
        // Deterministic pseudo-random sweep (no rand dependency needed).
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let kinds = [
            Kind::Full,
            Kind::Window { tokens: 7 },
            Kind::Window { tokens: 32 },
            Kind::Checkpoint,
            Kind::Scratch,
        ];
        for _ in 0..2000 {
            let n_groups = 1 + (next() % 3) as usize;
            let groups: Vec<Group> = (0..n_groups)
                .map(|_| {
                    let kind = kinds[(next() % kinds.len() as u64) as usize];
                    let bt = [2, 4, 6, 8][(next() % 4) as usize];
                    Group::new(kind, bt).unwrap()
                })
                .collect();
            let prompt_len = (next() % 120) as u32;
            let present: Vec<Vec<u32>> = groups
                .iter()
                .map(|_| (0..64).filter(|_| next() % 4 != 0).collect())
                .collect();
            assert_eq!(
                resolve(&groups, &present, prompt_len),
                brute_resolve(&groups, &present, prompt_len),
                "groups={groups:?} prompt_len={prompt_len} present={present:?}"
            );
        }
    }

    #[test]
    fn invalid_groups_are_rejected() {
        assert_eq!(
            Group::new(Kind::Full, 0),
            Err(InvalidGroup::ZeroBlockTokens)
        );
        assert_eq!(
            Group::new(Kind::Window { tokens: 0 }, 4),
            Err(InvalidGroup::ZeroWindow)
        );
    }

    #[test]
    fn huge_coprime_block_sizes_do_not_overflow() {
        let groups = [
            full(4_294_967_291),
            checkpoint(4_294_967_279),
            full(4_294_967_231),
        ];
        let present: [&[u32]; 3] = [&[0], &[0], &[0]];
        assert_eq!(resolve(&groups, &present, u32::MAX), 0);
    }
}
