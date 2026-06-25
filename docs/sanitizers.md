# Compiler Sanitizers (ASAN / UBSan / TSAN)

Opt-in `-fsanitize` instrumentation of **host-compiled** code, driven by a
single `--sanitizer` selection. This page is the design + usage reference. The
nightly CI job lives in [ci.md](ci.md#sanitizer-sim); the *scoping* rationale
(why macOS is excluded, why TSAN is report-only, why LSan is off) lives in
[investigations/2026-06-sanitizer-scope.md](investigations/2026-06-sanitizer-scope.md).

## What it instruments

Sanitizers instrument **host-compiled** code only — on sim that is the runtime
(`host`/`aicpu`/`aicore`), the per-test kernels, and orchestration; on onboard
only the host runtime. Device code (ccec AICore `.o`, aarch64 AICPU) cannot
carry a host sanitizer, and device custom arenas (`DeviceArena`/`HeapRing`)
bypass ASAN redzones, so device-side heap bugs are **not** caught.

## Architecture

It is a **two-part flag** — a build-time half and a run-time half — wired
through one single source of truth.

- **Build-time** — `SIMPLER_SANITIZER=<preset>` (a cmake define on
  `pip install`) flows to `build_runtimes.py`, which sets
  `RuntimeCompiler._sanitizers`. Only **host** targets honor it:
  `BuildTarget.gen_cmake_args` gates the `-fsanitize` injection on
  `toolchain.is_host`, and `cmake/sanitizers.cmake::simpler_apply_sanitizers`
  applies the flags. Device toolchains (ccec / aarch64) never see it.
- **Run-time** — `--sanitizer <preset>` (pytest `conftest.py` or standalone
  `scene_test.py`) compiles the per-test kernels/orchestration to match, and a
  **fail-fast guard** refuses to run unless the matching runtime is preloaded,
  printing the exact `LD_PRELOAD` command.

**Why preload is mandatory.** The instrumented `.so` are `dlopen`'d into a
**vanilla, un-instrumented Python**. A sanitizer runtime must be the very first
thing mapped into the process, so it cannot be pulled in late via `dlopen` — it
must be `LD_PRELOAD`'d (`DYLD_INSERT_LIBRARIES` on macOS) ahead of the
interpreter.

**Single source of truth.** [`simpler_setup/sanitizers.py`](../simpler_setup/sanitizers.py)
owns the preset table, the mutual-exclusion rule, and the runtime-library
mapping. The build, the per-test kernel compile, and the preload glue all read
from it, so adding a sanitizer is one dict entry (see [Extending](#extending)).

## Load-bearing invariants

Violating any of these produces a confusing failure that looks like a code bug
but is a build/ABI mismatch. They are enforced in code; this is the *why*.

1. **One sanitizer runtime per process (ABI unification).** Under a sanitizer,
   **every host artifact must link the same compiler's runtime, and the
   preload must be that same compiler's `lib*san`.** Sim unifies on **g++-15**
   (`GxxToolchain(prefer_g15=True)`), because the sanitizer runtime is
   ABI-versioned: mixing g++ and g++-15 runtimes, or preloading a different one
   than the `.so` link, fails at `dlopen` with **"cannot allocate memory in
   static TLS block"**. This is exactly the #949 bug — env `CC`/`CXX` (exported
   by scikit-build-core during `pip install`) silently overrode the g++-15 pin,
   so the `.so` linked `libtsan.so.0` while the run preloaded `libtsan.so.2`.
   The pin in `_host_compiler_cmake_args` now keeps it consistent.
2. **Host-only instrumentation.** Only `is_host` toolchains receive
   `-fsanitize`; device toolchains never. Device-side heap bugs are out of
   scope (custom arenas bypass redzones).
3. **ASAN and TSAN are mutually exclusive** — separate, incompatible builds.
   `validate()` rejects `thread` combined with `address`/`leak`/`memory`, and
   rejects `thread` on any non-Linux platform (no macOS `libtsan`).
4. **Run-time `--sanitizer` must match build-time `SIMPLER_SANITIZER`.** The
   fail-fast guard enforces that the matching runtime is preloaded; a mismatch
   stops before any test runs.

## Usage

Two parts — install the runtime instrumented, then run with the matching
runtime preloaded.

```bash
# 1. Build the runtime with the sanitizer (ASAN bundles UBSan).
pip install --no-build-isolation --config-settings=cmake.define.SIMPLER_SANITIZER=asan .

# 2. Run, preloading the matching runtime. Sim unifies on g++-15, so preload
#    g++-15's runtime — plain g++'s would mismatch the kernels' ABI (see
#    invariant 1) and fail at load. Preload g++-15's libstdc++ too, or ASan
#    can't resolve its __cxa_throw interceptor (libstdc++ isn't mapped into the
#    plain C Python when ASan inits) and the first C++ throw from the runtime
#    aborts with "CHECK failed: ... real___cxa_throw != 0".
LD_PRELOAD="$(g++-15 -print-file-name=libasan.so) $(g++-15 -print-file-name=libstdc++.so)" \
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
pytest examples tests/st --platform a2a3sim --sanitizer asan -v
```

**TSAN is a separate, mutually-exclusive build** and is **Linux-only**:

```bash
pip install --no-build-isolation --config-settings=cmake.define.SIMPLER_SANITIZER=tsan .
LD_PRELOAD=$(g++-15 -print-file-name=libtsan.so) TSAN_OPTIONS=halt_on_error=1 \
pytest examples tests/st --platform a2a3sim --sanitizer tsan -v
```

`detect_leaks=0` is recommended initially — LSan false-positives on the device
custom arenas until suppressions are added (see the scope investigation).

Capturing the report through an abort: pytest's default fd-capture buffers a
test's output in memory and prints it only at teardown. A sanitizer abort
(`abort_on_error=1`) kills the process mid-test, so teardown never runs and the
buffer is discarded — the job log shows only `Fatal Python error: Aborted` with
no sanitizer stack. Two ways to recover it, used together:

- `log_path` writes each process's report to a per-pid file at report time,
  independent of fd capture, so it survives the abort. Forked chip children
  (`worker.py` `os.fork`) inherit the env and get their own files.
- pytest `-s` (`--capture=no`) sends output straight to the real stderr fd, so
  the report — and any non-sanitizer abort/assert message, which never reaches
  `log_path` — also appears inline.

```bash
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1:log_path=/tmp/asan \
... pytest ... -s        # report lands in /tmp/asan.$PID (per process) and on the console
```

The nightly `Sanitizers` workflow does both and uploads the files as a
`sanitizer-logs-<sanitizer>-<platform>` artifact.

## Presets

`--sanitizer` (and `SIMPLER_SANITIZER`) take a preset name or a raw
`-fsanitize` token list. Presets expand in `sanitizers.py::SANITIZER_PRESETS`:

| Preset | `-fsanitize` tokens | Notes |
| ------ | ------------------- | ----- |
| `none` | *(empty)* | No instrumentation. |
| `asan` | `address,undefined` | ASAN bundles UBSan (compatible, cheap). |
| `ubsan` | `undefined` | UBSan alone. |
| `tsan` | `thread` | Separate build; Linux-only. |

## Extending

- **Add a sanitizer** — one entry in `SANITIZER_PRESETS`; if it needs a new
  runtime library, add it to `preload_lib()`. `cmake/sanitizers.cmake` does not
  change.
- **Add a platform** — `host_cxx()` decides which compiler's runtime gets
  preloaded (sim → g++-15, onboard → g++).

## See also

- [ci.md](ci.md#sanitizer-sim) — the nightly `sanitizer-sim` job (matrix,
  scope, gating).
- [investigations/2026-06-sanitizer-scope.md](investigations/2026-06-sanitizer-scope.md)
  — why macOS is excluded, why TSAN is report-only, why LSan is off, and the
  condition to re-open each.
