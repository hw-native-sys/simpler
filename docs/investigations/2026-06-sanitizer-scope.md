# Sanitizer rollout scope: macOS, TSAN gating, LSan

**Date**: 2026-06-01
**Verdict**: scoped — ASAN/UBSan gate on Linux CI only; TSAN
deferred-to-report-only; LSan measured-clean, no gate (see §3)

## Question

The opt-in sanitizer feature (`--sanitizer`, #915) instruments
host-compiled sim code (runtime + per-test kernels + orchestration) and runs
nightly. Three scoping questions recur — a future contributor will reasonably
ask "why not just turn this on too?" This entry records why each landed where
it did and the condition to re-open it. The "what currently happens" lives in
`docs/sanitizers.md` (design + usage) and `docs/ci.md` § sanitizer-sim; this is
the *why*.

## 1. Why ASAN/UBSan run only on Linux CI (not macOS)

**The pull**: the infra already has a darwin preload path
(`sanitizers.py::preload_command` emits `DYLD_INSERT_LIBRARIES` on macOS), so
adding a macOS leg to the nightly looks free, and most contributors develop on
macOS.

**Why not (now)**:

- Sim host artifacts unify on Homebrew **g++-15**. Linking `-fsanitize=address`
  through Apple's `ld` fails with `ld: library 'asan' not found` — Apple's
  linker does not resolve GCC's `libasan`. The sanitizer build does not link
  cleanly on macOS.
- `DYLD_INSERT_LIBRARIES` is stripped under **SIP** for protected binaries, so
  even a working build would have unreliable runtime preloading.
- The deployment target is Linux; that is where the toolchain is consistent and
  where the bugs that matter reproduce.

TSAN is a harder "no" than ASAN: there is no macOS `libtsan` at all, so
`validate()` rejects `thread` on any non-Linux platform outright.

**When to reconsider**: if the sim host build gains an Apple-clang path on macOS
(clang ships its own `libasan`/`ld` integration), or a macOS-only host bug needs
catching that the Linux leg cannot reproduce.

## 2. Why TSAN is report-only (not a hard gate)

**The pull**: ASAN gates the job (any finding fails CI); TSAN should too,
otherwise a real data race could land unnoticed.

**What was found**: ran TSAN on the real build in Linux/aarch64 docker and
triaged every reported race (see #947, #949).

**Result**: all races sit on the **device-shared region** — register MMIO
(`platform_regs.cpp`), the AICPU↔AICore handshake (`scheduler_*.cpp`,
`aicpu_executor.cpp`, `aicore_executor.cpp`), and the shared device arena /
tensors. The sim models the chip's **lock-free hardware handshake** (`volatile`
MMIO ordered by `wmb()`/`rmb()` barriers), which TSAN treats as
non-synchronizing — so it flags every cross-thread access. None are genuine
software bugs. The one host-side residual (the `device_runner.cpp` wall-clock
stamp) faithfully mirrors onboard `kernel.cpp`'s documented
*"last-writer-wins — wall measurements are µs-scale tolerant."*

**Why not (now)**: a hard gate needs the by-design races eliminated, not just
tolerated. Two routes, both deferred:

- A **suppressions file** — but JIT-compiled kernel frames symbolize as
  `<null>` (no function/file), so file-scoped `race:` suppressions cannot match
  them precisely without over-suppressing the host dispatch hub.
- **`__tsan_acquire`/`__tsan_release` annotations** at the handshake
  publish/observe points — the correct fix (TSAN would then *understand* the
  ordering and the false races vanish), but a large, invasive change across two
  arches' hot paths.

So the cell runs `TSAN_OPTIONS=halt_on_error=0:exitcode=0`: races go to the log,
the job gates only on hang / crash / test failure.

**When to reconsider**: when the handshake gains TSAN happens-before annotations
(the false races disappear and the gate becomes meaningful), or if a genuine
host-side race appears in the residual that is worth catching.

## 3. Why LSan (leak detection) is disabled

**The pull**: ASAN bundles LSan and defaults to `detect_leaks=1` on Linux —
leak detection looks like a free add-on. The nightly explicitly sets
`detect_leaks=0`, i.e. turns it *off*.

**What was tried** (2026-06-01): rebuilt the sim with ASAN and ran the
worker-reused L2 suites (`prepared_callable` + `dynamic_register`) under
`detect_leaks=1`. These share a **session-scoped `ChipWorker`**
(`conftest.py::_l2_worker_pool`), so many `register → run → unregister` cycles
run in one long-lived process — exactly the condition under which a per-case
host leak accumulates and becomes visible.

**Result**: **zero** leaks in our code. `libhost_runtime` appears in **no**
leak stack across any process; the host process's leak set is a *fixed* 560
objects (~862 KB) that does not grow with case count, and is **entirely**
PyTorch/Python import-time bindings (`torch::jit::init...` via `import torch`).
The rest of the report is structural noise: `cc1plus` compiler subprocesses
(the `LD_PRELOAD=libasan` is inherited by the per-test kernel compiles, so each
dumps ~13 MB of its own AST) and ~100 forked sim device processes (the
by-design device arenas).

**Why not (now)**: a standing gate would have to suppress `cc1plus`, every
forked device arena, `torch`, and the interpreter — a large, fragile
suppressions file guarding a **currently-empty** signal. The worker-reuse path
is verified leak-clean today, so there is nothing to protect.

**When to reconsider**: if a leak-prone host feature lands (new dlopen/unregister
churn, host-side caches). Re-verify with a *one-shot* pass — not a standing gate:
pre-compile kernels first (so no `cc1plus` runs under the preload), gate leak
reporting to the main process, and suppress `torch`/device-arena allocations.

## References

- PRs: #915 (feature), #931 (nightly pass — ASAN gate, TSAN informational),
  #947 (TSAN cell scope + report-only + `exitcode=0`), #949 (g++-15 ABI pin —
  the build bug that kept TSAN from running at all: env `CC`/`CXX` overrode the
  pin so the `.so` linked `libtsan.so.0` while the run preloaded `libtsan.so.2`,
  failing at `dlopen` with "cannot allocate memory in static TLS block").
- `docs/sanitizers.md` (design + usage), `docs/ci.md` § sanitizer-sim.
- `simpler_setup/sanitizers.py` — `validate()` platform gate, `preload_lib` /
  `host_cxx` runtime mapping.
