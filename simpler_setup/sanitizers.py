# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single source of truth for compiler-sanitizer selection.

The same `--sanitizer` value drives three places, so the preset table, the
mutual-exclusion rule, and the runtime-library mapping all live here:

- runtime build (`build_runtimes.py` → cmake `-DSIMPLER_SANITIZERS=`)
- per-test kernel / orchestration compile (`kernel_compiler.py`)
- the `LD_PRELOAD` glue at test time (`conftest.py`)

A *preset* (what users type) expands to a comma-separated `-fsanitize` *token*
list (what the compiler and `cmake/sanitizers.cmake` consume). Adding a
sanitizer is one dict entry here; the cmake helper never changes.
"""

from __future__ import annotations

import sys

# Preset name -> comma-separated `-fsanitize=` tokens. Empty = no instrumentation.
# ASAN bundles UBSan (compatible, cheap). TSAN is its own (mutually exclusive)
# build. Raw token strings (e.g. "address,leak") are also accepted as-is.
SANITIZER_PRESETS: dict[str, str] = {
    "none": "",
    "asan": "address,undefined",
    "ubsan": "undefined",
    "tsan": "thread",
}

# `-fsanitize=thread` cannot coexist with these in one binary.
_THREAD_INCOMPATIBLE = {"address", "leak", "memory"}


def resolve(selection: str | None) -> str:
    """Expand a preset name to its `-fsanitize` token list (raw tokens pass through).

    Returns "" for ``None`` / "none" / "" — i.e. no instrumentation.
    """
    if not selection:
        return ""
    if selection in SANITIZER_PRESETS:
        return SANITIZER_PRESETS[selection]
    return selection  # already a raw token list like "address,undefined"


def validate(tokens: str) -> None:
    """Raise ValueError if the token set is an illegal or unsupported combination.

    This is the shared gate for the runtime build, the pytest path, and the
    standalone path, so the platform check lives here too.
    """
    toks = {t.strip() for t in tokens.split(",") if t.strip()}
    if "thread" in toks and (toks & _THREAD_INCOMPATIBLE):
        raise ValueError(
            f"sanitizer 'thread' cannot combine with {sorted(toks & _THREAD_INCOMPATIBLE)} "
            "in one binary — build them separately"
        )
    if "thread" in toks and sys.platform != "linux":
        raise ValueError("thread sanitizer (TSAN) is Linux-only — no libtsan on this platform")


def preload_lib(tokens: str) -> str | None:
    """Runtime library to `LD_PRELOAD` for a dlopen'd, sanitizer-built `.so`.

    The sanitizer runtime must be the first thing in the process; since the
    instrumented `.so` files are dlopen'd into a vanilla Python, preloading the
    matching runtime is required. Returns None when nothing needs preloading.
    """
    toks = {t.strip() for t in tokens.split(",") if t.strip()}
    if "thread" in toks:
        return "libtsan.so"
    if "address" in toks:  # ubsan/leak fold into the asan runtime
        return "libasan.so"
    if "leak" in toks:  # standalone LSan (no address)
        return "liblsan.so"
    if "undefined" in toks:
        return "libubsan.so"
    return None


def host_cxx(platform: str) -> str:
    """Host compiler whose sanitizer runtime must be preloaded.

    Sim unifies on g++-15 (matching the kernels) under a sanitizer; onboard host
    uses plain g++. The `LD_PRELOAD`'d runtime must come from the same compiler
    that built the `.so`, or the versioned ASan/TSan ABI check fails at load.
    """
    return "g++-15" if platform.endswith("sim") else "g++"


def is_runtime_loaded(lib: str) -> bool:
    """Whether `lib` (e.g. libasan.so) is already mapped into this process.

    Best-effort: only Linux exposes /proc/self/maps — elsewhere (macOS) trust
    the caller. The instrumented `.so` are dlopen'd into a vanilla Python, so
    the sanitizer runtime must be preloaded before the interpreter starts.
    """
    try:
        with open("/proc/self/maps") as f:
            return lib.split(".so")[0] in f.read()
    except OSError:
        return True


def preload_command(tokens: str, platform: str) -> str | None:
    """The `LD_PRELOAD=$(... -print-file-name=...)` hint for an error message."""
    lib = preload_lib(tokens)
    if not lib:
        return None
    var = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
    return f"{var}=$({host_cxx(platform)} -print-file-name={lib})"
