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

> The repo name **simpler** stands for Simple/Simpler Runtime.

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

1. **Consult `.ai-instructions/` for task-specific rules.** The directory contains topic-based guides — read the ones relevant to your current task (e.g., `coding/` when writing code, `git-commit/` when committing, `terminologies/` when unsure about domain terms). You do not need to read all files upfront
2. **Do not modify directories outside your assigned area** unless the user explicitly requests it
3. Create new subdirectories under your assigned directory as needed
4. When in doubt, ask the user before making changes to other areas
5. **Avoid including private information in documentation or code** such as usernames, absolute paths with usernames, or other personally identifiable information. Use relative paths or generic placeholders instead

## Reference Index

Each working directory has its own `CLAUDE.md` with localized rules:
- `src/platform/CLAUDE.md` — Platform backends (a2a3, a2a3sim), interface contracts
- `src/runtime/CLAUDE.md` — Runtime variants, graph-building strategies
- `examples/CLAUDE.md` — Example layout, how to add new examples

Task-specific guides in `.ai-instructions/`:
- `coding/architecture.md` — Three-program model deep-dive
- `coding/codestyle.md` — C++ style rules (enum class, volatile, offsetof)
- `coding/testing.md` — Golden test pattern, kernel_config reference
- `coding/debugging.md` — Debugging techniques for simulation and hardware
- `coding/performance.md` — Profiling workflow and optimization strategies
- `coding/platform-porting.md` — How to add a new platform backend
- `ci/pipeline.md` — CI pipeline structure and modification guide
- `git-commit/commit-message.md` — Commit message format
- `terminologies/ascend-device.md` — AIC/AIV/AICPU terminology

Project documentation in `docs/`:
- `docs/CONTRIBUTING.md` — Contribution guidelines, code style, PR process
- `docs/troubleshooting.md` — Common errors and their solutions

On-demand skills in `skills/`:
- `skills/kernel-development/SKILL.md` — Kernel writing patterns and templates
- `skills/profiling/SKILL.md` — Profiling workflow and visualization tools
