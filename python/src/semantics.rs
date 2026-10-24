//! Python bindings for `pegaflow-semantics`, exposed as `pegaflow.semantics`.
//!
//! Thin wrappers only: every rule lives in the Rust crate so the connector
//! and the server cannot drift.

use pegaflow_semantics as sem;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

#[pyclass(frozen, eq, hash, from_py_object, name = "Group")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PyGroup(sem::Group);

fn checked_group(kind: sem::Kind, block_tokens: u32) -> PyResult<PyGroup> {
    sem::Group::new(kind, block_tokens)
        .map(PyGroup)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl PyGroup {
    #[staticmethod]
    fn full(block_tokens: u32) -> PyResult<Self> {
        checked_group(sem::Kind::Full, block_tokens)
    }

    /// `tokens`: trailing cached tokens the next token attends to (vLLM's
    /// `sliding_window - 1` is exact; `sliding_window` costs <= 1 extra block).
    #[staticmethod]
    fn window(block_tokens: u32, tokens: u32) -> PyResult<Self> {
        checked_group(sem::Kind::Window { tokens }, block_tokens)
    }

    #[staticmethod]
    fn checkpoint(block_tokens: u32) -> PyResult<Self> {
        checked_group(sem::Kind::Checkpoint, block_tokens)
    }

    #[staticmethod]
    fn scratch(block_tokens: u32) -> PyResult<Self> {
        checked_group(sem::Kind::Scratch, block_tokens)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.0.kind() {
            sem::Kind::Full => "full",
            sem::Kind::Window { .. } => "window",
            sem::Kind::Checkpoint => "checkpoint",
            sem::Kind::Scratch => "scratch",
        }
    }

    #[getter]
    fn block_tokens(&self) -> u32 {
        self.0.block_tokens()
    }

    /// Window length in tokens, or None for non-window groups.
    #[getter]
    fn window_tokens(&self) -> Option<u32> {
        match self.0.kind() {
            sem::Kind::Window { tokens } => Some(tokens),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.0.kind() {
            sem::Kind::Window { tokens } => {
                format!("Group.window({}, {tokens})", self.0.block_tokens())
            }
            _ => format!("Group.{}({})", self.kind(), self.0.block_tokens()),
        }
    }
}

fn groups_of(groups: &[PyGroup]) -> Vec<sem::Group> {
    groups.iter().map(|g| g.0).collect()
}

/// Half-open range as a tuple; every empty range is `(0, 0)`.
fn as_tuple(r: std::ops::Range<u32>) -> (u32, u32) {
    if r.is_empty() {
        (0, 0)
    } else {
        (r.start, r.end)
    }
}

#[pyfunction]
fn needed(group: PyGroup, p: u32) -> (u32, u32) {
    as_tuple(sem::needed(&group.0, p))
}

#[pyfunction]
fn resolve(groups: Vec<PyGroup>, present: Vec<Vec<u32>>, prompt_len: u32) -> PyResult<u32> {
    if groups.len() != present.len() {
        return Err(PyValueError::new_err(format!(
            "present has {} entries for {} groups",
            present.len(),
            groups.len()
        )));
    }
    Ok(sem::resolve(&groups_of(&groups), &present, prompt_len))
}

#[pyfunction]
fn saveable(group: PyGroup, before: u32, after: u32) -> (u32, u32) {
    as_tuple(sem::saveable(&group.0, before, after))
}

#[pyfunction]
fn key<'py>(py: Python<'py>, block_hash: &[u8], group: u32) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &sem::key(block_hash, group))
}

#[pyclass(frozen, eq, hash, from_py_object, name = "Shard")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PyShard(sem::Shard);

#[pymethods]
impl PyShard {
    #[staticmethod]
    fn replicated() -> Self {
        Self(sem::Shard::Replicated)
    }

    #[staticmethod]
    fn heads(index: u32, count: u32) -> Self {
        Self(sem::Shard::Heads { index, count })
    }

    #[staticmethod]
    fn tokens(index: u32, count: u32, interleave: u32) -> Self {
        Self(sem::Shard::Tokens {
            index,
            count,
            interleave,
        })
    }

    fn __repr__(&self) -> String {
        match self.0 {
            sem::Shard::Replicated => "Shard.replicated()".to_string(),
            sem::Shard::Heads { index, count } => format!("Shard.heads({index}, {count})"),
            sem::Shard::Tokens {
                index,
                count,
                interleave,
            } => format!("Shard.tokens({index}, {count}, {interleave})"),
        }
    }
}

#[pyclass(frozen, from_py_object, name = "Region")]
#[derive(Clone)]
struct PyRegion(sem::Region);

#[pymethods]
impl PyRegion {
    #[new]
    fn new(layer: String, group: u32, segments: Vec<u64>, shard: PyShard) -> Self {
        Self(sem::Region {
            layer,
            group,
            segments,
            shard: shard.0,
        })
    }

    #[getter]
    fn layer(&self) -> &str {
        &self.0.layer
    }

    #[getter]
    fn group(&self) -> u32 {
        self.0.group
    }

    #[getter]
    fn segments(&self) -> Vec<u64> {
        self.0.segments.clone()
    }

    #[getter]
    fn shard(&self) -> PyShard {
        PyShard(self.0.shard)
    }
}

#[pyclass(frozen, name = "Fragment")]
struct PyFragment(sem::Fragment);

#[pymethods]
impl PyFragment {
    #[getter]
    fn layer(&self) -> &str {
        &self.0.layer
    }

    #[getter]
    fn part(&self) -> u32 {
        self.0.part
    }

    #[getter]
    fn segments(&self) -> Vec<u64> {
        self.0.segments.clone()
    }

    #[getter]
    fn holders(&self) -> Vec<u32> {
        self.0.holders.clone()
    }

    #[getter]
    fn writer(&self) -> u32 {
        self.0.writer()
    }

    #[getter]
    fn bytes(&self) -> u64 {
        self.0.bytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "Fragment(layer={:?}, part={}, segments={:?}, holders={:?})",
            self.0.layer, self.0.part, self.0.segments, self.0.holders
        )
    }
}

#[pyclass(frozen, name = "SealPlan")]
struct PySealPlan(sem::SealPlan);

#[pymethods]
impl PySealPlan {
    /// Group ids, ascending.
    #[getter]
    fn groups(&self) -> Vec<u32> {
        self.0.groups.iter().map(|g| g.group).collect()
    }

    /// Fragments of `group` in slot order.
    fn fragments(&self, group: u32) -> PyResult<Vec<PyFragment>> {
        let g = self
            .0
            .group(group)
            .ok_or_else(|| PyValueError::new_err(format!("unknown group {group}")))?;
        Ok(g.fragments.iter().cloned().map(PyFragment).collect())
    }

    /// `(group, slot)` of the fragment `rank` holds for `layer`, or None.
    fn slot_of(&self, rank: u32, layer: &str) -> Option<(u32, usize)> {
        self.0.slot_of(rank, layer)
    }

    fn is_writer(&self, rank: u32, group: u32, slot: usize) -> bool {
        self.0.is_writer(rank, group, slot)
    }
}

/// Build the seal plan from `(rank, regions)` pairs; raises ValueError on
/// gaps, overlaps, or inconsistent declarations.
#[pyfunction]
fn seal_plan(ranks: Vec<(u32, Vec<PyRegion>)>) -> PyResult<PySealPlan> {
    let decls: Vec<sem::RankDecl> = ranks
        .into_iter()
        .map(|(rank, regions)| sem::RankDecl {
            rank,
            regions: regions.into_iter().map(|r| r.0).collect(),
        })
        .collect();
    sem::seal_plan(&decls)
        .map(PySealPlan)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn add_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "semantics")?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyShard>()?;
    m.add_class::<PyRegion>()?;
    m.add_class::<PyFragment>()?;
    m.add_class::<PySealPlan>()?;
    m.add_function(wrap_pyfunction!(needed, &m)?)?;
    m.add_function(wrap_pyfunction!(resolve, &m)?)?;
    m.add_function(wrap_pyfunction!(saveable, &m)?)?;
    m.add_function(wrap_pyfunction!(key, &m)?)?;
    m.add_function(wrap_pyfunction!(seal_plan, &m)?)?;
    parent.add_submodule(&m)
}
