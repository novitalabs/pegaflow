---
name: version-bump
description: Bump pegaflow-llm package version using commitizen. Use when the user asks to bump version, release a new version, increment version, or update package version.
---

# Version Bump

Bump the `pegaflow-llm` Python package version using commitizen.

## Prerequisites

Ensure commitizen is installed:

```bash
pip install commitizen
```

## Bump PATCH Version

Run from the `python/` directory:

```bash
cd python && cz bump --increment PATCH
```

This will:
1. Increment version in `pyproject.toml` (e.g., `0.0.10` → `0.0.11`)
2. Update `[tool.commitizen]` version field
3. Create a git commit with the version bump
4. Create a git tag `v0.0.11`

## Current Configuration

Version is tracked in `python/pyproject.toml`:

| Setting | Value |
|---------|-------|
| Scheme | `cz_conventional_commits` |
| Tag format | `v$version` |
| Version file | `pyproject.toml:^version` |

## Verify Before Bumping

Check the current version:

```bash
grep -E "^version" python/pyproject.toml
```

## After Bumping

Push the commit and tag:

```bash
git push && git push --tags
```
