# Sanitizer rollout scope: macOS, TSAN gating, LSan

**Date**: 2026-06-01
**Verdict**: scoped — ASAN/UBSan gate on Linux CI only; TSAN
deferred-to-report-only; LSan deferred-pending-arena-suppressions

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

**Why not (now)**: the sim allocates large device **custom arenas**
(`DeviceArena` / `HeapRing`, ~1 GB) that are intentionally long-lived (released
at process exit) and bypass ASAN redzones. LSan reports these as leaks → mass
false-positives that would redden the ASAN job before any real leak is visible.

**When to reconsider**: after an `lsan.supp` triages the device-arena and static
allocations as known non-leaks. Then flip `detect_leaks=1` and let genuine leaks
surface — the same shape of follow-up as the TSAN suppressions above.

## References

- PRs: #915 (feature), #931 (nightly pass — ASAN gate, TSAN informational),
  #947 (TSAN cell scope + report-only + `exitcode=0`), #949 (g++-15 ABI pin —
  the build bug that kept TSAN from running at all: env `CC`/`CXX` overrode the
  pin so the `.so` linked `libtsan.so.0` while the run preloaded `libtsan.so.2`,
  failing at `dlopen` with "cannot allocate memory in static TLS block").
- `docs/sanitizers.md` (design + usage), `docs/ci.md` § sanitizer-sim.
- `simpler_setup/sanitizers.py` — `validate()` platform gate, `preload_lib` /
  `host_cxx` runtime mapping.
