---
name: version-bump
description: Bump pegaflow-llm package version. Use when the user asks to bump version, release a new version, increment version, or update package version.
---

# Version Bump

Bump the `pegaflow-llm` Python package version by directly editing `python/pyproject.toml`.

## Bump PATCH Version

**Important:** Create a branch first since pre-commit hooks forbid commits to master.

1. **Check current version:**

```bash
grep -E "^version" python/pyproject.toml
```

2. **Create release branch:**

```bash
git checkout -b release/v<new_version>
```

3. **Edit `python/pyproject.toml`:**

Update both version fields:
- `version = "<new_version>"` (under `[project]`)
- `version = "<new_version>"` (under `[tool.commitizen]`)

Example: `0.0.10` → `0.0.11`

4. **Commit the change:**

```bash
git add python/pyproject.toml
git commit -m "chore: bump version to <new_version>"
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
