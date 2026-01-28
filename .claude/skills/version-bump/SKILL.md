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

**Important:** Create a branch first since pre-commit hooks forbid commits to master.

```bash
git checkout -b release/v<new_version>
cd python && cz bump --increment PATCH --no-tag
```

This will:
1. Increment version in `pyproject.toml` (e.g., `0.0.10` → `0.0.11`)
2. Update `[tool.commitizen]` version field
3. Create a git commit with the version bump
4. **NOT** create a git tag (tags are managed via GitHub releases)

### Alternative: With Tag

If you want to create a git tag locally:

```bash
cd python && cz bump --increment PATCH
```

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

## Workflow

1. **Push the release branch:**

```bash
git push -u origin release/v<version>
```

2. **Create a pull request:**

```bash
gh pr create --title "chore: bump version to <version>" --body "Release v<version>"
```

3. **After PR review and merge to master:**

User will manually create the GitHub release via the web UI or CLI.
