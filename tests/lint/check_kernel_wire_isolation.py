# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Keep the L3+ wire argument surface off the kernel and orchestration include paths.

Each runtime's `runtime/tensor.h` exports its working tensor as the unqualified name
`Tensor`, which kernels and cross-runtime orchestration sources use. That spelling is
only legal while nothing those translation units include reaches
`task_interface/buffer.h`, because buffer.h declares a *different* `Tensor` — the
address-free L3+ wire form — at **global scope**. One new include and the alias
becomes a redeclaration, and the error surfaces in whichever kernel happens to pick
the header up rather than in the file that added the edge.

Two changes established the separation: #2032 moved `EntryArgsStorage` out of
`src/common/{host_build_graph,tensormap_and_ringbuffer}/tensor.h`, and #2038 split
`TaskArgs` and the blob codec out of `task_args.h` into `task_args_wire.h`. Nothing
kept it after that, which is what this hook is for — the invariant was held by two
commits' worth of care, and care is not a mechanism.

## What is checked, and why it is the edge rather than the reachability

Reachability would be the direct statement of the invariant, but pre-commit passes
only the staged files, and the file that breaks reachability is usually *not* under a
kernel path — it is a shared header such as `runtime/types.h` or
`common/*/entry_args.h`. So this checks the **edge**: which files may include
`task_interface/buffer.h` or `task_interface/task_args_wire.h` at all. Every new
includer is itself a staged change, so the check cannot be evaded by partial staging,
and the allowlist doubles as the list of places that legitimately handle the wire
form — the host orchestrator, the Python bindings, the platform host code that decodes
a blob, and their tests.

Adding an entry is not forbidden; it is a reviewable diff. Adding one to a header that
kernels or orchestration reach is the defect this exists to surface.
"""

import argparse
import re
import sys
from pathlib import Path

# `#include "buffer.h"` / `#include "../task_interface/buffer.h"`, and the same for
# task_args_wire.h. The leading quote is required so `ring_buffer.h`,
# `aligned_buffer.h` and `kernel_pop_stack_buffer.h` do not match.
WIRE_INCLUDE = re.compile(r'#\s*include\s*"(?:[^"]*/)?(buffer\.h|task_args_wire\.h)"')

# Files that legitimately handle the L3+ wire form. Repo-relative, matched exactly.
ALLOWED = frozenset(
    {
        # The wire header itself: it is what buffer.h is for.
        "src/common/task_interface/task_args_wire.h",
        # The host orchestrator — L3+ is where a Tensor is built and validated.
        "src/common/hierarchical/types.h",
        "src/common/hierarchical/orchestrator.h",
        "src/common/hierarchical/remote_wire.h",
        "src/common/hierarchical/worker_manager.h",
        # Python bindings expose Tensor and TaskArgs to the L3+ caller.
        "python/bindings/task_interface.cpp",
        # Platform host code decodes a mailbox blob into ChipStorageTaskArgs.
        "src/common/platform/onboard/host/c_api_shared.cpp",
        "src/common/platform/onboard/host/device_runner_base.cpp",
        "src/common/platform/sim/host/c_api_shared.cpp",
        "src/common/platform/sim/host/device_runner_base.cpp",
        # Their unit tests.
        "tests/ut/cpp/common/task_interface/test_buffer.cpp",
        "tests/ut/cpp/common/task_interface/test_child_memory.cpp",
    }
)

SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".cc", ".cxx"})


def offending_lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [(lineno, line.strip()) for lineno, line in enumerate(text.splitlines(), 1) if WIRE_INCLUDE.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="files to check (pre-commit passes the staged set)")
    args = parser.parse_args()

    failures = []
    for name in args.files:
        if name in ALLOWED or Path(name).suffix not in SUFFIXES:
            continue
        for lineno, line in offending_lines(Path(name)):
            failures.append(f"{name}:{lineno}: {line}")

    if not failures:
        return 0

    print("This file includes the L3+ wire argument surface, which kernels and")
    print("orchestration sources must not reach: task_interface/buffer.h declares a")
    print("global `Tensor` that collides with each runtime's unqualified `Tensor` alias.")
    print()
    print("If the file only needs the argument container, include task_args.h — it")
    print("carries TaskArgsTpl and ChipStorageTaskArgs and reaches no wire type. If it")
    print("genuinely handles the L3+ form, add it to ALLOWED in this hook and say why.")
    print()
    for failure in failures:
        print(f"  {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
