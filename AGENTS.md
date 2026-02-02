# Repository Guidelines

## Project Structure & Module Organization

- `src/platform/`: Platform implementations.
  - `src/platform/a2a3/`: Ascend hardware (AICPU/AICore + host runtime).
  - `src/platform/a2a3sim/`: Thread-based host simulation (same public C API).
- `src/runtime/`: Runtime implementations (e.g. `src/runtime/host_build_graph/` with `build_config.py`).
- `python/`: Python bindings and build tooling (`bindings.py`, `runtime_builder.py`, `binary_compiler.py`).
- `examples/`: End-to-end examples and the runner (`examples/scripts/run_example.py`).
- `tests/`: Python unit tests (pytest).

## Build, Test, and Development Commands

- Install minimal Python deps: `python -m pip install numpy pytest`
- Run unit tests: `pytest -v`
- Run the simulation example (recommended for local dev):
  - `python examples/scripts/run_example.py -k examples/host_build_graph_sim_example/kernels -g examples/host_build_graph_sim_example/golden.py -p a2a3sim`
- Run the full local CI script (pytest + examples): `./ci.sh`
- Optional (simulation platform CMake build): `cmake -S src/platform/a2a3sim -B build && cmake --build build`

## Coding Style & Naming Conventions

- C/C++: format with `clang-format` using `.clang-format` (4-space indent, 120 column limit).
- Python: 4-space indent, keep modules importable from repo root (tests add `python/` to `sys.path`).
- Naming: prefer `snake_case` for files, functions, and variables; keep platform/runtime identifiers consistent (`a2a3`, `a2a3sim`, `host_build_graph`).

## Testing Guidelines

- Framework: `pytest` (`tests/test_*.py`).
- When adding a new runtime or platform feature:
  - Add/extend unit tests in `tests/`.
  - Add an example under `examples/<name>/` with `kernels/kernel_config.py` and a `golden.py` implementing `generate_inputs()` and `compute_golden()`.

## Commit & Pull Request Guidelines

- Commit messages commonly use a typed prefix (e.g. `Add: ...`, `Fix: ...`, `Refactor: ...`) and an imperative summary.
- PRs should include: what changed, which platform(s) you ran (`a2a3sim`/`a2a3`), and the command(s) used (e.g. `pytest -v`, `./ci.sh`, `run_example.py ...`).

## Security & Configuration Tips

- Default to `a2a3sim` for development; hardware builds typically require `ASCEND_HOME_PATH` and the Ascend toolchain.
- PTO ISA headers are located via `PTO_ISA_ROOT` (the example runner can auto-clone into `examples/scripts/_deps/pto-isa`).
- Device selection: set `PTO_DEVICE_ID` (fallback: `TILE_FWK_DEVICE_ID`).

## Deeper Docs

- Repository code walkthrough: `docs_contrib/repo-analysis.md`
- Design discussion for AICPU build+execute split: `docs_contrib/issues/aicpu-orch.md`
