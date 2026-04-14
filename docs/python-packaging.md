# Python Packaging

How the Python wheel is laid out, what each module is for, and which install modes are supported. **When you change the package structure (move/rename/delete modules, edit `pyproject.toml`'s `wheel.packages`, change CMake install rules), you must re-verify the full install-mode × entry-point matrix at the bottom of this document.**

## Package layout

The wheel ships two top-level Python packages plus one nanobind extension:

```text
simpler/                  ← public runtime API (python/simpler/ in source)
  __init__.py
  env_manager.py          environment variable validation
  task_interface.py       wraps _task_interface nanobind module
  worker.py               Worker (level=2 single-chip, level=3 distributed)
  orchestrator.py         L3 orchestration helpers

simpler_setup/            ← test framework + build/runtime assembly (simpler_setup/ in source)
  __init__.py
  environment.py          PROJECT_ROOT auto-resolves wheel vs source-tree layout
  pto_isa.py              PTO-ISA dependency management
  platform_info.py        platform/runtime discovery, PROJECT_ROOT re-export
  toolchain.py            compiler toolchain abstraction
  elf_parser.py           text-section extraction for AICore .o
  kernel_compiler.py      single-kernel compilation
  runtime_compiler.py     full runtime (host/aicpu/aicore) compilation
  runtime_builder.py      RuntimeBuilder: discovers + builds runtime binaries
  scene_test.py           SceneTestCase framework + @scene_test decorator
  code_runner.py          CodeRunner engine for run_example.py + ci.py
  goldens/                shared golden reference compute
    __init__.py
    paged_attention.py    attention reference (used by multiple paged_attention tests)
  _assets/                (wheel-only, populated by CMake install)
    src/                  source tree mirror
    build/lib/            pre-built per-arch/platform/runtime .so/.o

_task_interface.*.so      nanobind extension at site-packages root
```

### Why two packages

| Concern | `simpler` | `simpler_setup` |
| ------- | --------- | --------------- |
| What it is | Stable user-facing runtime API | Test/build infrastructure |
| Imported by user code? | Yes — `from simpler.worker import Worker` | Sometimes — test framework uses it |
| Imported by other packages? | Yes — `simpler_setup` imports `simpler.env_manager` | No — leaf consumer |
| Lifecycle | Slow-changing public API | Fast-changing internal helpers |

Internal coupling: `simpler_setup.toolchain`, `simpler_setup.kernel_compiler`, and `simpler_setup.runtime_compiler` import `simpler.env_manager`. This is the only direction allowed (`simpler_setup → simpler`); never the reverse. `simpler` must not depend on `simpler_setup`.

### `PROJECT_ROOT` resolution

`simpler_setup.environment.PROJECT_ROOT` auto-detects between:

- **Wheel install**: `simpler_setup/_assets/` exists → `PROJECT_ROOT = .../site-packages/simpler_setup/_assets`. The wheel's bundled `_assets/src/` and `_assets/build/lib/` provide everything needed at runtime.
- **Source tree / editable install**: `_assets/` doesn't exist → `PROJECT_ROOT = repo root`. Live `src/` and `build/lib/` are used.

Anything that needs to find `src/`, `build/lib/`, or `build/cache/` MUST go through `simpler_setup.environment.PROJECT_ROOT` — never `Path(__file__).parent.parent...`.

## Import rules

1. **No bare imports of internal modules.** Everything goes through `simpler.X` or `simpler_setup.X`. The following are forbidden anywhere in the repo:
   - `from runtime_builder import ...`
   - `from platform_info import ...`
   - `from code_runner import ...`
   - `from paged_attention_golden import ...`
2. **No `sys.path.insert` to make bare imports work.** The single allowed exception is `examples/scripts/build_runtimes.py`, which is invoked by CMake **before** the package is installed and therefore must bootstrap the source tree onto `sys.path`. That bootstrap is documented inline in the file.
3. **Build/runtime assembly lives in `simpler_setup`, not `simpler`.** If a new module is shared by the test framework, `code_runner`, or `ci.py`, it goes under `simpler_setup/`. If it's part of the user-facing runtime API, it goes under `python/simpler/`.
4. **Goldens** (shared reference compute used by multiple test sites) live in `simpler_setup/goldens/`. New goldens are added as `simpler_setup/goldens/<name>.py`. Per-test ad-hoc goldens stay next to their `kernel_config.py` and aren't packaged.
5. **`pyproject.toml` `wheel.packages`** must list every directory that needs to ship in the wheel. Currently: `["simpler_setup", "python/simpler"]`. Subpackages (e.g., `simpler_setup/goldens/`) ship automatically as long as they have an `__init__.py`.

## CLI entry points

Four supported user-facing entry points:

| Command | Purpose | Module location |
| ------- | ------- | --------------- |
| `pytest` | Unit + scene tests | imports `simpler` and `simpler_setup` from installed package |
| `python <test_*>.py` | Standalone scene test (uses `SceneTestCase.run_module`) | reads `simpler_setup` from installed package |
| `python ci.py` | Batch CI runner | imports `simpler` + `simpler_setup` from installed package |
| `python examples/scripts/run_example.py` | Single-example runner | imports `simpler_setup.code_runner` from installed package |

Plus one build-time entry point invoked by CMake during `pip install`:

| Command | Purpose |
| ------- | ------- |
| `python examples/scripts/build_runtimes.py` | Pre-build all runtime variants. Must work before the package is installed → uses an explicit `sys.path.insert` to point at the source tree. |

## Install modes

Five install paths × four entry points = the verification matrix. CI enforces the matrix on macOS and Ubuntu via `.github/workflows/ci.yml::packaging-matrix`.

### Mode-by-mode

| Install command | Status | Notes |
| --------------- | ------ | ----- |
| `pip install .` | ✅ | pip creates an isolated build env, installs scikit-build-core there, builds wheel, installs wheel. Slower (extra build deps download), but works. |
| `pip install --no-build-isolation .` | ✅ | Uses the venv's already-installed `scikit-build-core`, `nanobind`, `cmake`. Fastest. Requires those packages pre-installed in the venv. |
| `pip install -e .` | ✅ | Works because we set `editable.rebuild = false` in `pyproject.toml`. C++ changes require an explicit `pip install ...` to recompile. |
| `pip install --no-build-isolation -e .` | ✅ | **Recommended for development** (per `.claude/rules/venv-isolation.md`). |
| `cmake + PYTHONPATH` | ✅ | No-pip workflow. See "cmake direct" below. |

The reason `editable.rebuild = false` is mandatory: with `editable.rebuild = true`, scikit-build-core's rebuild-on-import hook executes `cmake --build`, which re-runs the cmake binary path baked into `build.ninja`. When the install was done with build isolation, that path points into pip's ephemeral build env (`/var/folders/.../pip-build-env-XXX/`) which pip deletes after install — the next import fails with `cmake: No such file or directory`. Disabling rebuild-on-import sidesteps this entirely; users who change C++ re-run `pip install` (which is the same workflow as for `pip install .` anyway).

### cmake direct (no pip install)

For developers who don't want to use pip at all:

```bash
. .venv/bin/activate
pip install scikit-build-core nanobind cmake pytest numpy ml_dtypes torch  # one-time
cmake -S . -B build/cmake_only -Dnanobind_DIR=$(python -c 'import nanobind; print(nanobind.cmake_dir())')
cmake --build build/cmake_only
export PYTHONPATH=$(pwd):$(pwd)/python
# Now all four entry points work
```

The cmake build places `_task_interface.cpython-XYZ.so` directly into `python/` (controlled by `python/bindings/CMakeLists.txt::LIBRARY_OUTPUT_DIRECTORY` when not in `SKBUILD_MODE`). The `add_custom_target(build_runtimes)` in the top-level `CMakeLists.txt` populates `build/lib/...` with runtime binaries. With `PYTHONPATH=repo:repo/python`, `simpler` and `simpler_setup` resolve from the source tree, the nanobind extension resolves from `python/`, and `simpler_setup.environment.PROJECT_ROOT` resolves to the repo root (so `build/lib/` is found).

### Entry-point compatibility

| Entry point | `pip install .` | `pip install --no-build-isolation .` | `pip install -e .` | `pip install --no-build-isolation -e .` | cmake + PYTHONPATH |
| ----------- | --------------- | ------------------------------------ | ------------------ | --------------------------------------- | ------------------ |
| `pytest` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `python <test>.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `python ci.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `python examples/scripts/run_example.py` | ✅ | ✅ | ✅ | ✅ | ✅ |

On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` to silence the homebrew-numpy + pip-torch libomp collision (also documented in `ci.py`).

## Verification protocol when changing package structure

Any change that touches:

- `python/simpler/*.py` (additions, removals, renames)
- `simpler_setup/**/*.py` (additions, removals, renames)
- `pyproject.toml` `[tool.scikit-build]` section
- `pyproject.toml` `[tool.pytest.ini_options]` section
- `CMakeLists.txt` install rules
- `examples/scripts/build_runtimes.py` (the pre-install bootstrap)
- `python/bindings/CMakeLists.txt` (nanobind module placement)

…must keep the **5 install modes × 4 entry points = 20 combinations** green. CI enforces this on macOS + Ubuntu via the `packaging-matrix` job in `.github/workflows/ci.yml`, which calls a single shared script:

```bash
# Locally — same script CI runs.
source .venv/bin/activate
pip install scikit-build-core nanobind cmake pytest numpy ml_dtypes torch  # one-time
bash tools/verify_packaging.sh
```

The script wipes `build/`, uninstalls `simpler`, and re-runs the install + smoke check from scratch for every mode, so a previous mode's cached binaries cannot mask a regression in the next. Slow (~25–45 min for all 5 modes) but reliable. Both CI and the local invocation use the same script — there's no second source of truth that can drift.

The smoke check itself only verifies imports and each entry point's `--help`. Functional tests (real runtime compilation, scene tests) live in `ut-py`, `ut-cpp`, `st-sim-*`, and the hardware self-hosted jobs.

## See also

- [`.claude/rules/venv-isolation.md`](../.claude/rules/venv-isolation.md) — venv setup and `--no-build-isolation` rationale
- [`.claude/rules/architecture.md`](../.claude/rules/architecture.md) — codebase-level architecture map
- [`docs/developer-guide.md`](developer-guide.md) — full directory structure and build workflow
