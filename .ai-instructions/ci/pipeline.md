# CI Pipeline Guide

## Overview

CI is driven by two files:
- `ci.sh` — Main test runner script (local and CI)
- `.github/workflows/ci.yml` — GitHub Actions workflow definition

## CI Jobs

### Simulation (`run-example-on-sim`)

Runs on: `ubuntu-latest`, `macos-latest`

```bash
./ci.sh -p a2a3sim
```

This job:
1. Sets up C++ compiler (g++-15 on Ubuntu, g++ on macOS)
2. Installs Python 3.10, numpy, ml_dtypes, pytest, torch (CPU-only)
3. Auto-discovers and runs all examples under `examples/`
4. Runs `pytest tests -v` for Python unit tests

### Hardware (`run-example-on-device`)

Runs on: self-hosted runner with Ascend NPU

```bash
./ci.sh -p a2a3 -d 4-7 --parallel
```

This job:
1. Requires CANN toolkit and Ascend driver pre-installed on runner
2. Auto-discovers and runs examples + device tests under `tests/device_tests/`
3. Uses `--parallel` to run tests across multiple devices simultaneously

## ci.sh Usage

```bash
# Simulation (all examples)
./ci.sh -p a2a3sim --parallel

# Hardware (all examples)
./ci.sh -p a2a3 -d 4-7 --parallel

# Simulation and Hardware (all examples)
./ci.sh -d 4-7 --parallel
```

### How auto-discovery works

`ci.sh` finds test directories by looking for `kernels/kernel_config.py` files:
- `examples/*/` — Simulation examples (run with `-p a2a3sim`)
- `tests/device_tests/*/` — Device-only tests (run with `-p a2a3`)

No registration is needed — just create the directory with the correct structure and CI picks it up.

## Modifying CI

### Adding a new CI job

Edit `.github/workflows/ci.yml`. Follow the existing patterns:
- Use the same Python/compiler setup steps
- Call `./ci.sh` with appropriate platform and flags

### Adding pre-commit checks

If you need checks before committing (formatting, linting), configure hooks in `.claude/settings.json` rather than modifying CI. CI is for validation; hooks are for prevention.

### Adding a new test type

1. Create the test under `examples/` (simulation) or `tests/device_tests/` (hardware)
2. Follow the standard directory layout with `golden.py` and `kernels/kernel_config.py`
3. CI auto-discovers it — no changes to `ci.sh` or workflow files needed

## Debugging CI Failures

### Reproduce locally

```bash
# Reproduce simulation CI
./ci.sh -p a2a3sim

# Reproduce a single failing test
python examples/scripts/run_example.py \
    -k <failing_test>/kernels \
    -g <failing_test>/golden.py \
    -p a2a3sim
```

### Common CI-specific issues

- **macOS vs Linux differences** — `g++` may resolve to different compilers
- **Python package version mismatches** — CI pins specific versions; check `ci.yml`
- **Device runner state** — Self-hosted runners may have stale CANN toolkit versions
