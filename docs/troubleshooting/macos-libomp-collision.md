# macOS libomp Collision in Single-Process CI

## TL;DR

On macOS, sim CI crashes with `OMP: Error #15 ... libomp.dylib already initialized` (SIGABRT, every task fails before any runtime code runs) because two different `libomp.dylib` copies get loaded into the same Python process — one via `numpy → openblas`, one via `torch`. We work around this at the top of the root `conftest.py` by setting `KMP_DUPLICATE_LIB_OK=TRUE` before pytest collects any golden. This doc exists so the next person who touches sim CI does not re-investigate the same rabbit hole.

## Symptom

Running `pytest examples tests/st --platform a2a3sim` (or `a5sim`) on macOS produces, for **every** task:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
--- FAIL: example:a2a3/... (dev0, attempt 1) ---
```

Exit code of the spawned worker is `134` (SIGABRT). The failure happens during golden `import`, so no DeviceRunner, no `pto_runtime_c_api.cpp`, no aicpu/aicore thread ever executes.

## Root Cause

Two distinct `libomp.dylib` copies get mapped into the Python process that pytest (or its xdist worker) uses to collect and execute scene tests:

1. **Homebrew's libomp** — `/opt/homebrew/opt/libomp/lib/libomp.dylib`, pulled in by the chain:
   `numpy → openblas (/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib) → libomp`

   `numpy` is loaded from the homebrew-managed system Python because our venv is created with `--system-site-packages` (required by `.claude/rules/venv-isolation.md`). Homebrew's numpy links against homebrew's openblas, which links against homebrew's libomp.

2. **PyTorch's bundled libomp** — `.venv/lib/python3.14/site-packages/torch/lib/libomp.dylib`, pulled in by:
   `torch → torch/_C → libtorch_python → libomp`

   pip's torch wheel ships its own libomp with install name `/opt/llvm-openmp/lib/libomp.dylib`.

The two dylibs have **different `LC_ID_DYLIB` install names** (verified with `otool -D`), so `dyld` loads them as completely separate images even though they expose the identical `__kmpc_*` / `GOMP_*` symbol set. When the second libomp initializes, Intel's OMP runtime detects a prior active libomp and calls `abort()`.

`DYLD_INSERT_LIBRARIES` and `ctypes.CDLL(..., RTLD_GLOBAL)` **do not fix this** — dyld resolves the dependency chain by install name, not by symbol matching against already-loaded libraries.

Reproducer (no CI code required):

```console
$ source .venv/bin/activate
$ python -c "import numpy; import torch"
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
[1]    12345 abort      python -c "import numpy; import torch"
```

Any golden importing `torch` after another golden has already imported `numpy` (or vice versa) is enough — and `import torch` transitively imports numpy, so even "all goldens use torch" does not avoid it.

## Why It Surfaces

Sim CI collects every scene test into a single pytest process (or, post xdist introduction, a handful of long-lived xdist workers). Each scene test's golden issues `import numpy` and/or `import torch` at collection or execution time, and those imports accumulate inside the same interpreter. Once both a numpy-triggered libomp and a torch-triggered libomp end up resident, Intel's OMP runtime aborts the process.

## Why Linux Does Not Hit This

On Linux, homebrew is not typical; `numpy` and `torch` are usually both pip-installed into the venv, and the wheels share the same `libgomp` / `libomp` from a single location. Even when they do not, glibc's dynamic linker uses symbol versioning + the `STB_GNU_UNIQUE` model, and the OpenMP runtime is more permissive about duplicate loads. We have never reproduced OMP Error #15 on Linux for this repo.

## Mitigation

Near the top of the repo-root `conftest.py` — after the small stdlib imports, but **before** `import pytest` and before pytest begins collecting test files (which transitively `import numpy` / `import torch` through goldens) — we set:

```python
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

`KMP_DUPLICATE_LIB_OK=TRUE` is Intel's documented escape hatch. It instructs the libomp runtime to proceed when a duplicate load is detected rather than aborting. Intel labels it "unsafe, unsupported, undocumented"; the concrete risks are:

- If both libomp copies actually run parallel regions concurrently, thread pool counts can double-count and performance degrades.
- Mixing thread-local storage between the two runtimes can misbehave in pathological cases.

Neither risk applies to our workload: the goldens use numpy/torch only for random-input generation and reference computation (single-threaded in all practical paths), and the real parallel execution happens in our own C++ DeviceRunner threads, not inside libomp. In practice the two libomps sit side-by-side and nothing bad happens.

## What NOT to Do

- **Do not** try to fix this by `dlopen`-preloading one libomp with `ctypes.CDLL(..., RTLD_GLOBAL)`. It doesn't work — dyld resolves subsequent libomp references by install name, not by symbol, so the second copy still loads.
- **Do not** try `DYLD_INSERT_LIBRARIES=.../torch/lib/libomp.dylib`. Same reason: different install names.
- **Do not** drop `--system-site-packages` from the venv to try to get a pip-installed numpy — `.claude/rules/venv-isolation.md` requires `--system-site-packages` so system-level driver bindings remain accessible.
- **Do not** "fix" it by removing `numpy` or `torch` imports from goldens. `import torch` transitively imports numpy, and writing golden reference math in pure Python is painful. Converting all goldens to torch does **not** make the conflict go away.
- **Do not** interpret OMP Error #15 as evidence of a sim-parallel threading bug, dlopen/dlclose ordering issue, or pthread TSD race. The crash happens during Python import, well before any C++ DeviceRunner code executes. A significant amount of debugging effort was wasted in commit `5cc0814` ("fix: in progress sim parallel") chasing this misdiagnosis.

## If You Need to Debug This Again

1. Check the failure message: if it contains `OMP: Error #15`, it is this issue. If not, look elsewhere.
2. Confirm two libomps are loading with:

   ```console
   DYLD_PRINT_LIBRARIES=1 python -c "import numpy, torch" 2>&1 | grep libomp
   ```

   You should see two different `libomp.dylib` paths.
3. Verify install names:

   ```console
   otool -D /opt/homebrew/opt/libomp/lib/libomp.dylib
   otool -D .venv/lib/python3.14/site-packages/torch/lib/libomp.dylib
   ```

4. Confirm the root `conftest.py` still sets `KMP_DUPLICATE_LIB_OK` *before* `import pytest` and before any test file gets collected — someone refactoring imports may accidentally put a numpy/torch-pulling import above the `os.environ.setdefault` line. Standalone `python test_*.py` entry points bypass conftest, so they rely on the same env var being set by the user or the parent shell on macOS.

## References

- Intel OMP `KMP_DUPLICATE_LIB_OK`: <https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-macos/current/avoiding-conflicts-in-the-linkage-symbol-names.html>
- OpenMP Error #15 FAQ: <https://openmp.llvm.org/>
