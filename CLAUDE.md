# Developer Guidelines

## Directory Ownership

Each developer role has a designated working directory. Stay within your assigned area unless explicitly requested by the user.

### Platform Developer
- **Working directory**: `src/platform/`
- Write platform-specific logic and abstractions here

### Runtime Developer
- **Working directory**: `src/runtime/`
- Write runtime logic including host, aicpu, aicore, and common modules here

### Codegen Developer
- **Working directory**: `examples/`
- Write code generation examples and kernel implementations here

## Architecture

PTO Runtime compiles three independent programs (Host `.so`, AICPU `.so`, AICore `.o`) that communicate through handshake buffers on Ascend NPU devices. Three runtime variants live under `src/runtime/` (`host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`), two platform backends under `src/platform/` (`a2a3` = hardware, `a2a3sim` = simulation). See `README.md` for the full architecture diagram.

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
    -k examples/host_build_graph/vector_example/kernels \
    -g examples/host_build_graph/vector_example/golden.py \
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
