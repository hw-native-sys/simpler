# Developer Guidelines

See [docs/developer-guide.md](docs/developer-guide.md) for full directory structure, compilation pipeline, and conventions.

## Directory Ownership

| Role | Working directory |
| ---- | ----------------- |
| Platform Developer | `src/{arch}/platform/` |
| Runtime Developer | `src/{arch}/runtime/` |
| Codegen Developer | `examples/` |

## Common Commands

See [docs/testing.md](docs/testing.md) for the full testing guide (st, pyut, cpput) and [docs/ci.md](docs/ci.md) for CI pipeline details.

### Python environment

This repo requires `pip install .` to run. **Always use a project-local venv created with `--system-site-packages`** — never install into the user/global site. Applies equally to git worktrees (create `.venv` inside the worktree). See [.claude/rules/venv-isolation.md](.claude/rules/venv-isolation.md).

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install .
```

### Format C++ code

```bash
clang-format -i <file>
```

## Important Rules

1. **Consult `.claude/rules/` for coding conventions** (architecture, codestyle, terminology) — these are always-loaded guidelines. **Consult `.claude/skills/` for task-specific workflows** (e.g., `git-commit/` when committing, `testing/` when running tests)
2. **Do not modify directories outside your assigned area** unless the user explicitly requests it
3. Create new subdirectories under your assigned directory as needed
4. When in doubt, ask the user before making changes to other areas
5. **Avoid including private information in documentation or code** such as usernames, absolute paths with usernames, or other personally identifiable information. Use relative paths or generic placeholders instead
