---
name: testing
description: Testing guide and pre-commit testing strategy for PTO Runtime. Use when running tests, adding tests, or deciding what to test before committing.
---

# Testing

## Test Types

1. **Python unit tests (ut-py)** (`tests/ut/py/`): Standard pytest tests for the Python compilation pipeline and nanobind bindings. Run with `pytest tests/ut/py`. Tests declaring `@pytest.mark.requires_hardware[("<platform>")]` auto-skip unless `--platform` points to a matching device.
2. **C++ unit tests (ut-cpp)** (`tests/ut/cpp/`): GoogleTest-based tests for pure C++ modules. Run with `cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build && ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure`. Hardware-required tests carry a `requires_hardware` or `requires_hardware_<platform>` ctest label and are filtered via `-LE`.
3. **Scene tests** (`examples/{arch}/*/`, `tests/st/{arch}/*/`): End-to-end `@scene_test` classes declared inside `test_*.py`. Sim variants run cross-platform (Linux/macOS); hardware variants require the CANN toolkit and an Ascend device. Discovery is by pytest (batch) or `python test_*.py` (standalone); `#591`'s parallel orchestrator handles device bin-packing and ChipWorker reuse automatically.

## Running Tests

**Important**: Always read `.github/workflows/ci.yml` first to extract the current `--pto-isa-commit` and `--pto-session-timeout` values. These ensure reproducible builds by pinning the PTO-ISA dependency to a known-good commit.

### Runtime rebuild decision

Before running tests, determine whether runtime binaries need recompilation:

| What changed | Rebuild needed? | How |
| ------------ | --------------- | --- |
| Runtime/platform C++ (`src/{arch}/runtime/`, `src/{arch}/platform/`) | Yes | Pass `--build` to the pytest or `python test_*.py` invocation |
| Nanobind bindings (`python/bindings/`) | Yes | Re-run `pip install -e .` |
| Python-only code, examples, kernels | No | Just re-run the test |

In CI, `pip install .` pre-builds all runtimes, so `--build` is never needed on CI runners.

```bash
# Python unit tests (no hardware)
pytest tests/ut

# Python unit tests (a2a3 hardware)
pytest tests/ut --platform a2a3

# C++ unit tests (no hardware)
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure

# C++ unit tests (a2a3 hardware)
ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$" --output-on-failure

# All simulation scene tests (extract --pto-isa-commit, --pto-session-timeout from ci.yml)
pytest examples tests/st --platform a2a3sim \
    --clone-protocol https --pto-isa-commit <commit> --pto-session-timeout <timeout>

# All hardware scene tests (extract --pto-isa-commit, --pto-session-timeout from ci.yml, auto-detect idle devices)
pytest examples tests/st --platform a2a3 --device <range> \
    --clone-protocol https --pto-isa-commit <commit> --pto-session-timeout <timeout>

# Single runtime
pytest examples tests/st --platform a2a3sim --runtime host_build_graph \
    --clone-protocol https --pto-isa-commit <commit>

# Single example (pytest, uses pre-built binaries)
pytest examples/a2a3/host_build_graph/vector_example --platform a2a3sim \
    --clone-protocol https --pto-isa-commit <commit>

# Single example (standalone, recompile runtime from source after changing runtime C++)
python examples/a2a3/host_build_graph/vector_example/test_vector_example.py \
    -p a2a3sim --build --clone-protocol https --pto-isa-commit <commit>
```

## Pre-Commit Testing Strategy

When changed files require testing (C++, Python, or CMake), follow these steps to decide **what** to test and **how**.

### Step 1 — Platform Availability and Detection

```bash
command -v npu-smi &>/dev/null
```

| Result | Platforms to test |
| ------ | ----------------- |
| Found | `<arch>sim` (simulation) **and** `<arch>` (hardware) |
| Not found | Simulation only (default `a2a3sim`) |

**When `npu-smi` is found**, detect the platform by parsing chip name from `npu-smi info` output:

| Chip name contains | Platform |
| ------------------ | -------- |
| `910B` or `910C` | `a2a3` (sim: `a2a3sim`) |
| `950` | `a5` (sim: `a5sim`) |

Use the detected platform for all subsequent `--platform` flags. If the chip name is unrecognized, warn and default to `a2a3`.

### Step 2 — Test Scope

Run `git diff --name-only` (or `git diff --cached --name-only` for staged changes) and match the **first** applicable rule:

| Changed paths | Scope | Command pattern |
| ------------- | ----- | --------------- |
| `src/{arch}/platform/*` | Full (all runtimes) | `pytest examples tests/st --platform <platform>` |
| `src/{arch}/runtime/<rt>/*` | Single runtime | `pytest examples tests/st --platform <platform> --runtime <rt>` |
| `examples/{arch}/<rt>/<ex>/*` | Single example | `python <ex>/test_*.py -p <platform> --build` (or `pytest <ex> --platform <platform> --build`) |
| `tests/ut/*` (Python) | Python UT only | `pytest tests/ut` (add `--platform <platform>` on a device runner) |
| `tests/ut/cpp/*` | C++ UT only | `cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build && ctest --test-dir tests/ut/cpp/build -LE requires_hardware` |
| Mixed (spans multiple categories) | Escalate to the **widest** matching scope | — |

> **Note on `--build`**: When changed paths include `src/{arch}/runtime/` or `src/{arch}/platform/`, pass `--build` so the runtime C++ is recompiled incrementally. Pytest batch runs accept `--build` at the CLI (applies to the whole session).

### Step 3 — Parallel Strategy

Parallelism is handled by the `#591` scheduler (`simpler_setup/parallel_scheduler.py`) based on `--device` and `--max-parallel`:

**Simulation (`a2a3sim`)**: `--max-parallel auto` = `min(nproc, len(--device))`. Pass `--device 0-15` for a big virtual pool; `auto` caps in-flight at the CPU count. Override with `--max-parallel N` on CPU-constrained runners.

**Hardware (`a2a3`)**: `--max-parallel auto` = `len(--device)`. One in-flight subprocess per physical device — each device runs a dedicated ChipWorker (see `docs/ci.md`).

### Step 4 — Device Detection (hardware only)

When testing on `a2a3`, detect idle devices:

```bash
npu-smi info
```

Pick devices whose **HBM-Usage is 0** and find the **longest consecutive sub-range** (at most 4). Pass as `--device <start>-<end>` (or `--device <id>` if only one idle device). If no idle device is found, skip hardware testing and warn.

### Decision Tree

```text
git diff --name-only
  │
  ├─ Only docs/config? ──→ SKIP tests
  │
  └─ Code changed?
       │
       ├─ Determine SCOPE (Step 2)
       │    ├─ platform   → full (pytest --platform ...)
       │    ├─ runtime    → single runtime (--runtime ...)
       │    └─ example    → single example (standalone test_*.py or pytest <ex>)
       │
       ├─ Runtime C++ changed (src/{arch}/)? ──→ add --build
       │
       └─ npu-smi found?
            ├─ Yes → sim + hardware (idle devs, max 4)
            └─ No  → sim only
```

## Adding a New Scene Test

1. Create a directory under the appropriate arch and runtime:
   - Examples: `examples/{arch}/<runtime>/<name>/`
   - Device-only scene tests: `tests/st/{arch}/<runtime>/<name>/`
2. Add `test_<name>.py` with a `@scene_test`-decorated class (see [docs/testing.md](../../docs/testing.md) for the full template: `CALLABLE`, `CASES`, `generate_args`, `compute_golden`). End with `if __name__ == "__main__": SceneTestCase.run_module(__name__)` so the file runs standalone.
3. Add kernel source files under `kernels/aic/`, `kernels/aiv/`, and/or `kernels/orchestration/` — referenced by `CALLABLE["orchestration"]["source"]` / `CALLABLE["incores"][*]["source"]` as paths relative to the test file.
4. Pytest auto-discovers any `test_*.py` under `examples/` and `tests/st/`; no registration needed.

## Related Skills

- **`git-commit`** — Complete commit workflow (runs testing as a prerequisite)
