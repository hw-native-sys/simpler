# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Keep percounter's AICore program off the host_build_graph headers it has forked.

percounter owns its AICore program and shares host_build_graph's AICPU and host
sides -- build_config.py compiles host_build_graph sources straight into the
`aicpu` and `host` targets. So this is about the `aicore` target alone, and about
one hazard in it: three host_build_graph headers have a percounter counterpart,
and only one of each pair may be reachable from a lane.

- `runtime/scheduler/gm.h` replaces `scheduler/scheduler_memory.h`, because a
  lane's GM access is one instruction rather than a traced call.
- `runtime/scheduler/image_read.h` replaces `scheduler/scheduler_ready.h`, whose
  remainder is a ready-queue protocol a lane does not have.
- `runtime/scheduler/async_completion.h` replaces
  `scheduler/scheduler_async_control.h`, whose remainder is leases, a directory
  and a resolver.

Both spellings compiling in one translation unit is not an error a compiler can
raise: each pair declares different names, so the original simply comes back along
with everything it pulls in, while the lane keeps calling the copy. What it costs
is a dependency and object size, and neither of those fails a build.

## Reachability, not the edge

check_kernel_wire_isolation.py checks which files may name a header at all, because
its hazard surface is the whole repository and pre-commit sees only staged files.
Here the surface is percounter's AICore sources and their closure, so this walks the
closure outright. That is the invariant itself, and it catches the case an edge check
cannot: a host_build_graph header percounter legitimately includes acquiring an
include of a forked original. That edge lives in host_build_graph, which is why this
hook runs on changes to either side.

The reverse direction is an edge check, having no closure to walk: nothing outside
src/a5/runtime/percounter/ may name a percounter header. host_build_graph has
src/a5/runtime/ on its include path, so `percounter/...` resolves from there, and a
sibling runtime reaching into this one would compile.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERCOUNTER = REPO / "src/a5/runtime/percounter"

# The AICore compile's search path, in order: BUILD_CONFIG["aicore"]["include_dirs"]
# from src/a5/runtime/percounter/build_config.py, then the platform directories
# src/a5/platform/onboard/aicore/CMakeLists.txt prepends to every AICore target.
#
# This is a hand-maintained mirror, and a mirror that silently loses a directory
# would make this hook pass over a closure it cannot see. `unresolved_includes`
# is what keeps it honest: a quoted include that resolves nowhere is reported
# rather than dropped, so a missing directory here fails the hook instead of
# shrinking what it checks.
AICORE_INCLUDE_DIRS = (
    "src/a5/runtime/percounter/aicore",
    "src/a5/runtime/percounter/runtime",
    "src/a5/runtime/host_build_graph/runtime",
    "src/a5/runtime/host_build_graph/common",
    "src/a5/runtime",
    "src/common/host_build_graph",
    "src/a5/platform/onboard/aicore",
    "src/a5/platform/include",
    "src/common/platform/include",
    "src/common/task_interface",
    "src/common/log/include",
    "src/common",
)

# Each key is a header percounter's AICore program has its own version of.
FORKED = {
    "src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h": (
        "src/a5/runtime/percounter/runtime/scheduler/gm.h"
    ),
    "src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h": (
        "src/a5/runtime/percounter/runtime/scheduler/image_read.h"
    ),
    "src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_async_control.h": (
        "src/a5/runtime/percounter/runtime/scheduler/async_completion.h"
    ),
}

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
# An include the source itself marks as optional. `scheduler_types.h` reaches
# `spin_hint.h` this way: the header exists once per platform backend and the
# AICore build adds neither, which is a resolution failure this hook must not
# report and the compiler never sees.
OPTIONAL = re.compile(r'^[^\S\n]*#[^\S\n]*if[^\S\n]+__has_include\s*\(\s*"([^"]+)"\s*\)', re.M)
NAMES_PERCOUNTER = re.compile(r'^\s*#\s*include\s*["<]([^">]*percounter/[^">]*)[">]', re.M)
SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".cc", ".cxx"})
# The roots that hold C++ anyone compiles; build output and 3rdparty are not searched.
SOURCE_ROOTS = ("src", "tests", "examples", "python", "simpler_setup")


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def resolve(including: Path, name: str, search: list[Path]) -> Path | None:
    candidate = including.parent / name
    if candidate.is_file():
        return candidate.resolve()
    for directory in search:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def closure(roots: list[Path], search: list[Path]) -> tuple[dict[Path, Path | None], dict[str, str]]:
    """Every file reachable from roots, mapped to the file that first included it.

    The second return is the quoted includes that resolved nowhere, mapped to a
    file that names them. Those are reported rather than dropped: each one is a
    subtree this walk cannot see, so dropping them would shrink what the hook
    checks without saying so.
    """
    came_from: dict[Path, Path | None] = {root.resolve(): None for root in roots}
    unresolved: dict[str, str] = {}
    pending = [root.resolve() for root in roots]
    while pending:
        current = pending.pop()
        try:
            text = current.read_text(errors="replace")
        except OSError:
            continue
        optional = set(OPTIONAL.findall(text))
        for name in INCLUDE.findall(text):
            target = resolve(current, name, search)
            if target is None:
                if name not in optional:
                    unresolved.setdefault(name, rel(current))
                continue
            if target in came_from:
                continue
            came_from[target] = current
            pending.append(target)
    return came_from, unresolved


def chain(came_from: dict[Path, Path | None], leaf: Path) -> str:
    steps: list[str] = []
    node: Path | None = leaf
    while node is not None:
        steps.append(rel(node))
        node = came_from[node]
    return " -> ".join(reversed(steps))


def check_forks() -> tuple[list[str], list[str]]:
    """Forked originals the closure reaches, and includes the closure could not follow."""
    roots = sorted(p for p in (PERCOUNTER / "aicore").rglob("*") if p.suffix in SUFFIXES)
    if not roots:
        return [], ["src/a5/runtime/percounter/aicore holds no sources, so this hook is checking nothing"]
    search = [REPO / d for d in AICORE_INCLUDE_DIRS]
    absent = [rel(d) for d in search if not d.is_dir()]
    if absent:
        return [], [f"an include dir named by this hook no longer exists: {d}" for d in absent]

    came_from, unresolved = closure(roots, search)
    reached = {rel(path): path for path in came_from}
    forks = [
        f"{original}\n      reached by: {chain(came_from, reached[original])}"
        f"\n      percounter's own version is {replacement}"
        for original, replacement in FORKED.items()
        if original in reached
    ]
    blind = [f'"{name}", included by {includer}' for name, includer in sorted(unresolved.items())]
    return forks, blind


def check_reverse() -> list[str]:
    failures = []
    for root in SOURCE_ROOTS:
        for path in sorted((REPO / root).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            name = rel(path)
            if name.startswith("src/a5/runtime/percounter/"):
                continue
            for include in NAMES_PERCOUNTER.findall(path.read_text(errors="replace")):
                failures.append(f'{name} includes "{include}"')
    return failures


def main() -> int:
    (forks, blind), reverse = check_forks(), check_reverse()
    if not forks and not blind and not reverse:
        return 0
    if forks:
        print("percounter's AICore program reaches a header it has its own version of:\n")
        for failure in forks:
            print(f"  - {failure}\n")
        print("Include percounter's version. If the fork is being retired, retire it in this hook too.\n")
    if blind:
        print("This hook cannot follow every include, so it is not checking the whole closure:\n")
        for failure in blind:
            print(f"  - {failure}")
        print(
            "\nEach one stops the walk, leaving whatever it includes unchecked. Add the directory"
            "\nthe AICore build finds it in to AICORE_INCLUDE_DIRS, or, if the source marks the"
            "\ninclude optional with __has_include, nothing needs doing here.\n"
        )
    if reverse:
        print("percounter is a private runtime; nothing outside it may include its headers:\n")
        for failure in reverse:
            print(f"  - {failure}")
        print("\nPut what is genuinely shared in host_build_graph or src/common instead.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
