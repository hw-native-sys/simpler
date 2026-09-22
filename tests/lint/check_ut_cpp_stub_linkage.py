# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
A stub the support archive provides must be the one a test binary got.

tests/ut/cpp/support is a static archive, so a member is pulled only to resolve
a symbol nothing else has defined. That is what lets a case compile the real
unified_log_host.cpp or device_time.cpp and simply not take the stub — it says
nothing about the stub at all.

A weak definition breaks that silently. The linker treats the symbol as
defined, never searches the archive, and the binary keeps the weak one. Product
code carries such fallbacks on purpose: the tmr runtime defines

    __attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }

in orchestrator.cpp and runtime_core.cpp, so that a host build linking neither
device_time.cpp nor the AICPU side still links. A test that takes that fallback
compiles and links and then hangs: the runtime's 500 ms reclaim backstop counts
ticks that never advance, so the timeout it is waiting for never arrives. The
fix is to compile the stub straight into whichever library also compiles the
weak definition — the archive copy is then simply not pulled there.

This check is what makes the next one loud. For every symbol the archive
defines strongly, no test binary may have resolved it to a weak definition. A
symbol the archive itself defines weakly — assert_impl and the two link
placeholders, which are meant to be overridable — is not the archive's to win,
and is not checked.

Needs a built tree, so it runs in CI's C++ unit test job beside the axis check:

    cmake --build tests/ut/cpp/build --parallel 4
    python tests/lint/check_ut_cpp_stub_linkage.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UT_CPP = REPO_ROOT / "tests" / "ut" / "cpp"


def defined_symbols(path: Path, kinds: str) -> set[str]:
    """Mangled names this file defines with one of the given nm type letters.

    A failed nm yields no names, and no names reads as "the archive provides
    nothing" — under which every binary passes. So the exit status decides,
    never the output.
    """
    result = subprocess.run(["nm", "--defined-only", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"nm --defined-only {path} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    found = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[-2] in kinds:
            found.add(fields[-1])
    return found


def demangle(names: list[str]) -> list[str]:
    result = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=False)
    return result.stdout.splitlines() or names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=UT_CPP / "build", help="a built tests/ut/cpp tree")
    args = parser.parse_args()

    archives = sorted(args.build_dir.rglob("lib*_ut_support.a"))
    if not archives:
        raise SystemExit(f"No support archive under {args.build_dir}. Build the tree before running this check.")

    # T only: a symbol the archive itself leaves weak is meant to be overridden.
    provided: set[str] = set()
    for archive in archives:
        provided |= defined_symbols(archive, "T")

    binaries = sorted(p for p in args.build_dir.glob("test_*") if p.is_file() and p.stat().st_mode & 0o111)
    if not binaries:
        raise SystemExit(f"No test binaries under {args.build_dir}. Build the tree before running this check.")

    shadowed: dict[str, list[str]] = {}
    for binary in binaries:
        lost = defined_symbols(binary, "WV") & provided
        for symbol in lost:
            shadowed.setdefault(symbol, []).append(binary.name)

    if not shadowed:
        return 0

    print("These binaries resolved a stub to a weak fallback instead of the archive's")
    print("definition. The archive was never searched, because the weak definition")
    print("already satisfied the reference:")
    print()
    for symbol, names in sorted(shadowed.items()):
        print(f"  {demangle([symbol])[0]}")
        print(f"      {len(names)} target(s), e.g. {', '.join(sorted(names)[:3])}")
    print()
    print("Compile the stub straight into whichever library also compiles the weak")
    print("definition. Leaving it in the archive too is correct and costs nothing:")
    print("the archive copy is not pulled where the symbol is already defined.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
