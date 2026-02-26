# Contributing to Simpler (Simple/Simpler Runtime)

## Getting Started

1. Clone the repository and run a simulation test to verify your setup:
   ```bash
   git clone <repo-url>
   cd simpler
   ./ci.sh -p a2a3sim
   ```

2. Read the [README.md](../README.md) for architecture overview and setup details.

3. Read the [CLAUDE.md](../CLAUDE.md) for developer guidelines and directory ownership.

## Directory Ownership

Each contributor role has a designated working directory:

| Role | Directory | Scope |
|------|-----------|-------|
| Platform Developer | `src/platform/` | Platform abstractions, a2a3/a2a3sim backends |
| Runtime Developer | `src/runtime/` | Runtime variants (host_build_graph, aicpu_build_graph, tensormap_and_ringbuffer) |
| Codegen Developer | `examples/` | Kernel implementations and example programs |

Stay within your assigned directory unless coordinating cross-cutting changes.

## Code Style

- Format C++ code with `clang-format -i <file>` before committing
- Use `enum class` for basic enums (not plain `enum`)
- Use `volatile` on shared-memory struct members
- Use `offsetof` instead of hardcoded byte offsets
- Avoid plan-specific comments (Phase 1, Step 1, Gap #3)

See `.ai-instructions/coding/codestyle.md` for the full style guide.

## Commit Messages

Use the format: `Type: concise description`

| Type | When to use |
|------|-------------|
| **Add** | New feature or file |
| **Fix** | Bug fix |
| **Refactor** | Restructuring without behavior change |
| **Update** | Enhancement to existing feature |
| **Support** | Tooling, profiling, CI infrastructure |
| **Sim** | Simulation-specific changes |
| **CI** | CI/CD pipeline changes |

Rules:
- Keep subject line under 72 characters
- Use imperative mood ("Add X" not "Added X")
- Do not add `Co-Authored-By:` lines

## Adding a New Example

1. Create a directory under the appropriate runtime variant:
   ```
   examples/<runtime_variant>/<example_name>/
     golden.py
     kernels/
       kernel_config.py
       aiv/          # or aic/
       orchestration/
   ```

2. Implement `generate_inputs(params)` and `compute_golden(tensors, params)` in `golden.py`

3. Define `KERNELS`, `ORCHESTRATION`, and `RUNTIME_CONFIG` in `kernel_config.py`

4. Test locally:
   ```bash
   python examples/scripts/run_example.py \
       -k examples/<runtime>/<name>/kernels \
       -g examples/<runtime>/<name>/golden.py \
       -p a2a3sim
   ```

5. The CI script auto-discovers new examples — no registration needed.

See `.ai-instructions/coding/testing.md` for the full golden test pattern reference.

## Testing

### Before submitting

```bash
# Run all simulation tests
./ci.sh -p a2a3sim

# Run Python unit tests
pytest tests -v
```

### On hardware (if available)

```bash
./ci.sh -p a2a3 -d 4-7 --parallel
```

## Pull Requests

- Keep PRs focused — one logical change per PR
- Ensure all simulation tests pass before requesting review
- Include a description of what changed and why
- If adding a new example, include sample output showing the test passes

## License

This project is licensed under the [CANN Open Software License Agreement Version 2.0](../LICENSE).
