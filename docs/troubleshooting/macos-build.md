# macOS Build & Lint Toolchain

Notes on macOS-specific toolchain quirks for the simpler build pipeline.
Read this before changing compilers / SDK paths in your dev environment, or
when a fresh checkout fails to compile or lint cleanly on macOS.

## TL;DR

- The build is **three independent phases** that may use different
  compilers; they communicate via C-style ABI only.
- Default macOS setup uses **Apple Clang** for phase 1 and orchestration,
  **Homebrew GCC 15** for sim kernels. This works out of the box.
- `pre-commit`'s `clang-tidy` hook uses **Homebrew LLVM** (Apple
  Command Line Tools does not ship `clang-tidy`). Homebrew Clang has a
  hard-coded default sysroot that often does not match the installed
  SDK, so `tests/lint/clang_tidy.py` injects `-isysroot` and
  `-isystem $SDK/usr/include/c++/v1` defensively — works regardless of
  whether phase 1 was built with Apple Clang or Homebrew Clang.
- Switching phase 1 to Homebrew Clang end-to-end is supported but
  optional. It needs `LDFLAGS` pointing at Homebrew's libc++ runtime and
  source-installs of `nanobind` / `pybind11`. See "Optional: Homebrew
  Clang for phase 1".

## Three-phase build, three toolchains

Three groups of `.so` files get produced at different times by different
compilers. They get loaded into the same Python process at test time.

| Phase | Driver | Compiler |
| ----- | ------ | -------- |
| **1. simpler install** | `pip install -e .` → scikit-build-core + `simpler_setup/build_runtimes.py` → CMake | Apple Clang (`/usr/bin/clang++`) by default; honors `CC`/`CXX` |
| **2a. test orch** | `simpler_setup/kernel_compiler.py` → `ToolchainType.HOST_GXX` (`g++`) | Apple Clang (`/usr/bin/g++` is a symlink to clang++) |
| **2b. test sim kernel** | `simpler_setup/kernel_compiler.py` → `ToolchainType.HOST_GXX_15` (`g++-15`) | **Homebrew GCC 15** (`/opt/homebrew/bin/g++-15`, real GCC, *not* Clang) |

Phase 1 produces the long-lived runtime libs (`libhost_runtime.so`,
`libaicpu_runtime.so`, `libaicore_runtime.so`, `libsim_context.so`,
`libsimpler_log.so`) plus the nanobind Python extension
(`_task_interface.cpython-*.so`). Phase 2a produces a
per-test `libexample_orchestration.so`, dlopened by phase-1 runtime
libs. Phase 2b produces per-task AIC / AIV / AICPU `.so` files for sim
runs, also dlopened by the runtime libs.

The toolchain pinning is in `simpler_setup/toolchain.py` — see
`HostGxx15Toolchain` and `GxxToolchain`.

### Why this is safe

These three artefacts come from three different C++ standard library
implementations:

- Apple Clang: Apple's libc++ (`/usr/lib/libc++.1.dylib`)
- Homebrew GCC 15: Homebrew's libstdc++
  (`/opt/homebrew/lib/gcc/15/libstdc++.dylib`)
- (Optional) Homebrew Clang: Homebrew's libc++
  (`/opt/homebrew/opt/llvm/lib/c++/libc++.1.dylib`)

They coexist in the same process because **the inter-`.so` boundaries
are C-style** (`extern "C"` functions, POD types). No `std::string` /
`std::vector` / template instantiations cross between phase-1 libs and
phase-2 `.so` files. Each `.so` carries its own C++ stdlib symbols and
they never need to agree.

**Implication**: changing phase 1's compiler does not force phase 2 to
follow. Decide per phase based on what each one needs.

## Issue: `clang-tidy` reports `'algorithm' file not found`

### Symptom

`pre-commit run` (or a `git commit` that triggers the hook) fails with:

```text
src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h:37:10:
  error: 'algorithm' file not found [clang-diagnostic-error]
   37 | #include <algorithm>
      |          ^~~~~~~~~~~
```

### Root cause

The error blames `<algorithm>` because it is the top-level include, but
`<algorithm>` itself is found fine — Homebrew LLVM ships its own libc++
headers at `/opt/homebrew/opt/llvm/include/c++/v1/`. What actually
breaks is the libc cascade triggered by libc++ internals:

```text
<algorithm>                  ← Homebrew libc++  (found)
  └─ #include <wchar.h>
     ├─ libc++ wrapper        ← Homebrew libc++  (found)
     └─ #include_next <wchar.h>
        └─ libc real wchar.h  ← needs SDK     (NOT found)
```

The libc++ wrapper headers `#include_next` to libc, which lives in the
macOS SDK at `<sdk>/usr/include/`. Homebrew LLVM is built with a
**hard-coded default sysroot path** (e.g.
`/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`). On a fresh
checkout the actually-installed SDK is typically a slightly different
version (`MacOSX26.2.sdk`, etc.), so the baked-in default points at a
non-existent directory and `#include_next` cannot resolve. The error
bubbles up as the original `<algorithm>` line.

### Why Apple Clang does not hit this

Apple Clang (`/usr/bin/clang++`) is integrated with `xcrun` and
**resolves the SDK at runtime**, not at LLVM build time. Even with no
flags it picks up the right SDK from the Command Line Tools install.
Homebrew Clang has no such integration; the only way to give it the
right SDK is an explicit `-isysroot`. So:

| Compiler | Where SDK comes from | Needs `-isysroot`? |
| -------- | -------------------- | ------------------ |
| Apple Clang | `xcrun` runtime resolution | No |
| Homebrew Clang | Hard-coded default at LLVM build time (often stale) | **Yes** |
| Homebrew clang-tidy | Same as Homebrew Clang | **Yes** |

### Fix

`tests/lint/clang_tidy.py` injects two `--extra-arg=` flags on macOS:

```python
def _macos_isysroot_args() -> list[str]:
    if platform.system() != "Darwin":
        return []
    try:
        sdk_path = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    if not sdk_path:
        return []
    return [
        f"--extra-arg=-isysroot{sdk_path}",
        f"--extra-arg=-isystem{sdk_path}/usr/include/c++/v1",
    ]
```

Each flag does a different job:

- **`-isysroot $SDK`** — points at the SDK so the libc cascade
  (`#include_next <wchar.h>` etc.) can resolve. Without this clang-tidy
  cannot lint any C++ file at all.
- **`-isystem $SDK/usr/include/c++/v1`** — explicitly adds Apple SDK's
  libc++ headers to the system include path *ahead* of Homebrew's
  bundled libc++. Without this, lint sees Homebrew's libc++ while the
  Apple-Clang build sees Apple SDK's libc++ — two different libc++
  releases. Forcing both to the same headers eliminates "lint clean,
  build fails" surprises caused by libc++ version drift.

`xcrun --show-sdk-path` returns the symlink
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` which auto-resolves
to the installed major.minor SDK — robust to SDK upgrades.

### Two independent paths to giving clang-tidy a usable sysroot

clang-tidy reads `compile_commands.json` and analyses each translation
unit using its recorded compile command. There are two places the
sysroot can come from:

```text
(a) Build side: pass -isysroot to compile_commands
        export CXXFLAGS="-isysroot $(xcrun --show-sdk-path)"
        ↓ cmake records it in compile_commands.json
        ↓ clang-tidy reads compile_commands.json, gets -isysroot for free

(b) Lint side: clang-tidy injects -isysroot itself
        tests/lint/clang_tidy.py --extra-arg=-isysroot... (this fix)
        ↓ works regardless of what compile_commands says
```

Apple Clang invocations omit `-isysroot` in `compile_commands.json`
(driver auto-resolves it), so (a) does not apply. The `-isysroot` patch
in (b) is what makes the hook usable in the default Apple-Clang build
configuration. If you switch phase 1 to Homebrew Clang you must export
`CXXFLAGS=-isysroot...` for the build itself anyway, in which case (a)
also kicks in and the (b) injection becomes redundant but harmless.

### Why not just match clang-tidy to phase-1 build's compiler?

Apple Command Line Tools does not ship `clang-tidy`, so on macOS the
only realistic clang-tidy is the Homebrew one. The hook must work
even when phase 1 is built with Apple Clang. The `-isysroot` +
`-isystem` injection makes Homebrew clang-tidy work against either
build.

## Optional: Homebrew Clang for phase 1

Phase 1 defaults to Apple Clang. You can switch it to Homebrew Clang if
you want phase 1 to align with the lint toolchain. **This is not
required**; the `-isysroot` patch above already aligns lint and build
sufficiently. Use this only when you specifically want
`compile_commands.json` to be produced by the same compiler that
clang-tidy runs.

### Steps

```bash
# 1. Source-install nanobind / pybind11. The wheels distributed on PyPI
#    are noarch, so this matters only for ABI flags during simpler's
#    own link step. (See "Why source-install" below.)
pip uninstall -y simpler nanobind pybind11
pip install --no-binary nanobind --no-binary pybind11 nanobind pybind11

# 2. Wipe build/ to drop any cached compile_commands.json or .a files
#    that were produced with the previous compiler.
rm -rf build/cache build/lib build/cp* _skbuild

# 3. Set env, install simpler.
export SDKROOT=$(xcrun --show-sdk-path)
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export CFLAGS="-isysroot $(xcrun --show-sdk-path)"
export CXXFLAGS="-isysroot $(xcrun --show-sdk-path)"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ \
                -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++"

pip install --no-build-isolation -e .
```

Verify with:

```bash
otool -L build/lib/a5/sim/tensormap_and_ringbuffer/libhost_runtime.so | head -5
# Should show: /opt/homebrew/opt/llvm/lib/c++/libc++.1.dylib (not /usr/lib/libc++.1.dylib)
```

### Why `LDFLAGS` is required

When linking with Homebrew Clang, the produced `.so` references
libc++ symbols emitted out-of-line by Homebrew's libc++ headers
(e.g. `std::__1::__hash_memory`). Apple's system
`/usr/lib/libc++.1.dylib` does not export those symbols (older libc++
release), so without `LDFLAGS` the link step fails:

```text
Undefined symbols for architecture arm64:
  "std::__1::__hash_memory(void const*, unsigned long)", referenced from:
      libnanobind-static.a[3](nb_func.cpp.o)
ld: symbol(s) not found for architecture arm64
```

`-L/opt/homebrew/opt/llvm/lib/c++ -lc++` plus the matching `-rpath`
points the linker — and the runtime loader — at the Homebrew libc++ that
matches the headers used at compile time.

### Why source-install nanobind / pybind11

Both ship as noarch wheels containing C++ headers and sources but no
prebuilt `.a` / `.dylib`. Source-installing them is **not strictly
required** for ABI; however it ensures the source tree under
`.venv/lib/.../{nanobind,pybind11}/` is what scikit-build-core compiles
against. If you skip this step and the venv still has a prior
binary-extracted copy, scikit-build-core will use whichever copy pip put
in place. The source-install is defensive; it costs little and avoids
state confusion when iterating.

### Phase 2 stays as-is

Phase 2a (orchestration) keeps using `g++` (Apple Clang) and phase 2b
(sim kernels) keeps using `g++-15` (Homebrew GCC), regardless of phase
1's compiler choice. Tests pass because the C-ABI boundary isolates the
three C++ stdlibs.

## Apple SDK version drift

Always read the SDK path through `xcrun --show-sdk-path`, never hardcode
a number:

```bash
# Good — survives SDK upgrades
$ xcrun --show-sdk-path
/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

# Bad — breaks the next time Apple ships an SDK update
-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk
```

The `MacOSX.sdk` path is a symlink that Apple maintains across point
releases. The numeric paths (e.g. `MacOSX26.2.sdk`) are real
directories that come and go.

## Related

- [`macos-libomp-collision.md`](macos-libomp-collision.md) — a
  separate macOS-only issue (libomp double-load in pytest workers,
  worked around in `conftest.py`).
- [`.claude/rules/venv-isolation.md`](../../.claude/rules/venv-isolation.md)
  — required setup for the `.venv` per worktree.
- `simpler_setup/toolchain.py` — authoritative source for which compiler
  each phase uses.
- `tests/lint/clang_tidy.py` — the `-isysroot` + `-isystem` defensive
  injection lives in `_macos_isysroot_args()`.
