# Developer Guidelines

See [docs/developer-guide.md](docs/developer-guide.md) for full directory structure, compilation pipeline, and conventions.

## Directory Ownership

| Role | Working directory |
|------|-------------------|
| Platform Developer | `src/{arch}/platform/` |
| Runtime Developer | `src/{arch}/runtime/` |
| Codegen Developer | `examples/` |

## Common Commands

### Simulation tests (no hardware required)
```bash
./ci.sh -p a2a3sim
```

### Hardware tests (requires Ascend device)
```bash
./ci.sh -p a2a3 -d 4-7 --parallel
```

### Run a single example
```bash
python examples/scripts/run_example.py \
    -k examples/a2a3/host_build_graph/vector_example/kernels \
    -g examples/a2a3/host_build_graph/vector_example/golden.py \
    -p a2a3sim
```

### Python unit tests
```bash
pytest tests -v
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
