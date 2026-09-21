# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
A C++ unit test must be built for every combination its code varies over.

Most trees under src/ are compiled more than once by the product. Each arch's
platform tree is its own code, and the arch-independent trees compile into every
runtime's image, resolving their bare-name headers to that runtime's copy. A
test covering such a tree is therefore covering one configuration of it, and a
test built once covers one configuration and says nothing about the rest.

Nothing about a case's source says which configuration it got — that is decided
by the include path its declaration composes — so a case built for one arch
looks exactly like a case that does not depend on the arch. This check tells
them apart by measurement rather than by reading the declaration:

1. **Which axes can apply** comes from the case's mirror directory. A case under
   a2a3/ is a2a3's by construction and the arch axis is not open to it; a case
   under common/tensormap_and_ringbuffer/ likewise belongs to that runtime.
2. **Whether the case is sensitive to an axis** comes from preprocessing each of
   its translation units with `-H` and reading which files were opened. If any
   of them has a sibling copy under the axis's other value that is not the same
   file, the case's code differs along that axis.
3. **Which combinations it is built for** comes from the same reading: a build
   that opens files under a5/ is an a5 build, and one that opens a runtime's
   tree is that runtime's.

The two axes are checked together, as cells of their product rather than
independently. A case built for a2a3/tmr, a2a3/hbg and a5/hbg uses both arches
and both runtimes, so neither axis looks short on its own — while the a5/tmr
cell it never builds is exactly the configuration nothing covers.

A combination that is wanted and never built is a gap. The ones that cannot
exist are listed in WAIVERS with the reason; a gap that is not listed fails,
and so does a waiver whose gap no longer exists.

This needs a configured build, so it runs in CI's C++ unit test job rather than
in pre-commit:

    cmake -B tests/ut/cpp/build -S tests/ut/cpp -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build tests/ut/cpp/build --parallel 4
    python tests/lint/check_ut_cpp_axis.py
"""

from __future__ import annotations

import argparse
import collections
import filecmp
import json
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
UT_CPP = REPO_ROOT / "tests" / "ut" / "cpp"

ARCHES = ("a2a3", "a5")


def discovered_runtimes() -> tuple[str, ...]:
    """The runtimes, by the rule the rest of the tree uses.

    A directory under src/<arch>/runtime/ is a runtime exactly when it carries a
    build_config.py — the same criterion as
    simpler_setup.platform_info.discover_runtimes and simpler_discover_runtimes()
    in tests/ut/cpp/cmake/runtimes.cmake. Naming them here instead would be a
    list that has to be edited for a runtime the build discovers on its own, and
    until it was, this check would report the new runtime's cases as a directory
    it cannot classify.
    """
    found = {path.parent.name for arch in ARCHES for path in (SRC / arch / "runtime").glob("*/build_config.py")}
    if not found:
        raise SystemExit(f"No runtime found under {SRC}/<arch>/runtime — a build_config.py names one.")
    return tuple(sorted(found))


RUNTIMES = discovered_runtimes()

# How a directory under tests/ut/cpp/common/ mirrors src/, and therefore which
# axes its cases are open to. A directory named in neither set is unclassified
# and fails the check rather than defaulting, for the same reason a case may not
# default its axis: an unjudged tree must not pass as a judged one.
#
# PER_RUNTIME_TREES compile into every runtime's image and resolve their
# bare-name headers to that runtime's copy, so both axes apply.
PER_RUNTIME_TREES = frozenset({"platform", "log", "worker", "platform_comm", "task_interface", "aicpu_loader"})
# These reach no arch or runtime tree at all: header-only utilities, and the
# host orchestrator, which only the nanobind extension compiles.
SINGLE_CONTEXT_TREES = frozenset({"utils", "hierarchical", "runtime_status"})

# Combinations a case is deliberately not built for, each with the reason. Most
# cannot exist at all; one is a case whose own golden covers a single arch, which
# is a gap to close rather than an absent combination, and its reason says so. A
# stale entry is an error, so the table only shrinks.
WAIVERS: dict[tuple[str, str], str] = {
    ("common/platform/test_kernel_args_helper.cpp", "a5/tensormap_and_ringbuffer"): (
        "both cases assert the uploaded prefix of a Runtime image ends at "
        "offsetof(DeviceRuntimeLaunchDesc, teardown_gates). a5's "
        "tensormap_and_ringbuffer Runtime has no device-initialized tail — its "
        "header states extent equals copy size — so it declares no "
        "teardown_gates and the boundary has no counterpart there"
    ),
    ("common/platform/test_kernel_persistent_args.cpp", "a5/tensormap_and_ringbuffer"): (
        "same absent teardown_gates tail as test_kernel_args_helper"
    ),
    ("common/platform/test_chip_swimlane_run_export.cpp", "a5/host_build_graph"): (
        "the case compares against a golden captured on a2a3, carrying that "
        "capture's own clock_freq_hz and platform — both values the writer takes "
        "from the arch it was built for. The writer is correct on a5: an a5 build "
        "differs from the golden in exactly those two fields and nothing else, and "
        "the five sibling cases on the same two sides all cover a5. So this is a "
        "golden nobody has captured yet, not a combination that cannot exist; "
        "capturing one closes it"
    ),
    ("common/platform/test_chip_swimlane_run_export.cpp", "a5/tensormap_and_ringbuffer"): (
        "same uncaptured a5 golden as the host_build_graph cell above"
    ),
}


def applicable_axes(case: str) -> tuple[bool, bool]:
    """(arch axis open, runtime axis open) for a case, from its mirror directory."""
    parts = Path(case).parts
    if parts[0] in ARCHES:
        # The arch is fixed by the directory. Which runtimes the case is built
        # for is decided one level down, and "not platform" is not the same as
        # "one runtime": <arch>/runtime/ holds the cases covering the parts
        # every runtime carries, and its CMakeLists builds each of them once per
        # runtime. Only <arch>/runtime/<runtime>/ fixes the runtime too.
        if len(parts) > 1 and parts[1] == "platform":
            return (False, True)
        if len(parts) > 1 and parts[1] == "runtime":
            return (False, not (len(parts) > 2 and parts[2] in RUNTIMES))
        return (False, False)
    tree = parts[1] if len(parts) > 1 else ""
    if tree in RUNTIMES:
        return (True, False)
    if tree in SINGLE_CONTEXT_TREES:
        return (False, False)
    if tree in PER_RUNTIME_TREES:
        return (True, True)
    raise KeyError(tree)


def siblings(relative: str, axis: str) -> list[str]:
    """The same file under the axis's other values, empty when the axis misses it."""
    if axis == "arch":
        for here in ARCHES:
            if relative.startswith(here + "/"):
                return [there + relative[len(here) :] for there in ARCHES if there != here]
        return []
    for here in RUNTIMES:
        if f"/{here}/" in relative:
            return [relative.replace(f"/{here}/", f"/{there}/", 1) for there in RUNTIMES if there != here]
    return []


def differs_across(relative: str, axis: str) -> bool:
    """True when this file's counterpart on some other side of the axis is another file."""
    return any(
        (SRC / other).exists() and not filecmp.cmp(SRC / relative, SRC / other, shallow=False)
        for other in siblings(relative, axis)
    )


def axis_value(opened: set[str], axis: str) -> str | None:
    """Which value of the axis a build compiled against, from the trees it opened."""
    if axis == "arch":
        for arch in ARCHES:
            if any(f.startswith(arch + "/") for f in opened):
                return arch
        return None
    for runtime in RUNTIMES:
        if any(f"/{runtime}/" in f for f in opened):
            return runtime
    return None


def cell_name(cell: tuple[str | None, str | None]) -> str:
    """A combination's name, carrying only the axes the case varies over."""
    return "/".join(value for value in cell if value) or "single"


def command_argv(entry: dict) -> list[str]:
    """One compile entry's argv.

    A compilation database carries the command either already split, in
    `arguments`, or as a shell line in `command`. The split form is taken when
    present because the shell line has to be un-quoted to be replayed, and
    splitting it on whitespace is not that: a legal `-I"/path/with space"`
    becomes two wrong arguments, and `-DNAME=\\"x\\"` keeps backslashes the
    shell would have removed. Both produce a preprocessor run that does not
    match the compile it stands for.
    """
    arguments = entry.get("arguments")
    if arguments:
        return list(arguments)
    return shlex.split(entry["command"])


def preprocess_command(entry: dict) -> list[str]:
    """The compile line, rewritten to preprocess and list every file it opens."""
    argv, tokens, index = [], command_argv(entry), 0
    while index < len(tokens):
        if tokens[index] == "-c":
            index += 1
            continue
        if tokens[index] == "-o":
            index += 2
            continue
        argv.append(tokens[index])
        index += 1
    return argv + ["-E", "-H", "-o", "/dev/null"]


def files_opened(entry: dict) -> set[str]:
    """Paths under src/ this translation unit compiles, its own source included.

    `-H` lists what the preprocessor opened, which is every header and not the
    .cpp it was handed — so a tree whose two copies differ only in a .cpp body
    would be invisible here, and the axis it varies over would read as
    collapsed. The unit's own source is therefore added explicitly.

    A run that fails opens an unknown subset, and an unknown subset reads as a
    case that reaches neither tree — which is what SINGLE asserts, so the
    coverage gap this check exists to find would be reported as absent. The
    same standard as read_targets(): understating coverage is an error, never a
    silent pass.
    """
    argv = preprocess_command(entry)
    result = subprocess.run(
        argv,
        cwd=entry["directory"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Preprocessing {entry['file']} failed (exit {result.returncode}), so which trees it\n"
            f"reaches cannot be read. The compile line was rewritten to preprocess:\n\n"
            f"  {shlex.join(argv)}\n\n{result.stderr.strip()}"
        )
    opened = set()
    own = Path(entry["file"])
    if not own.is_absolute():
        own = Path(entry["directory"]) / own
    own = own.resolve()
    if own.is_relative_to(SRC):
        opened.add(str(own.relative_to(SRC)))
    for line in result.stderr.splitlines():
        # -H writes one line per opened file, prefixed by dots for nesting depth.
        if not line.startswith("."):
            continue
        try:
            path = Path(line.split(" ", 1)[1].strip()).resolve()
        except (IndexError, OSError):
            continue
        if path.is_relative_to(SRC):
            opened.add(str(path.relative_to(SRC)))
    return opened


def read_targets(build_dir: Path) -> dict[str, list[dict]]:
    """Compile entries grouped by the target that owns them.

    The object path names the target, and it is reachable two ways: the
    `output` field, which only newer CMake writes, and the `-o` argument, which
    every entry carries. Taking whichever is present keeps this working across
    the CMake a contributor happens to have on PATH — the venv's and the
    system's are not the same version here, and depending on the field alone
    made the check die rather than report.
    """
    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.exists():
        raise SystemExit(
            f"{compile_commands} is missing. Configure with "
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON and build before running this check."
        )
    entries = json.loads(compile_commands.read_text())
    grouped = collections.defaultdict(list)
    unattributed = 0
    for entry in entries:
        object_path = entry.get("output") or ""
        if not object_path:
            argument = re.search(r"-o\s+(\S+)", entry.get("command", ""))
            object_path = argument.group(1) if argument else ""
        match = re.search(r"CMakeFiles/([^/]+)\.dir/", object_path)
        if match:
            grouped[match.group(1)].append(entry)
        else:
            unattributed += 1
    if not grouped:
        raise SystemExit(
            f"No target could be read out of {compile_commands} ({len(entries)} entries, none carrying an object path)."
        )
    if unattributed:
        raise SystemExit(
            f"{unattributed} of {len(entries)} entries in {compile_commands} name no "
            "target. Silently skipping them would understate coverage, so this is an error."
        )
    return grouped


def case_of(entries: list[dict]) -> str | None:
    """The tests/ut/cpp-relative case source a target is built from, if any."""
    for entry in entries:
        path = Path(entry["file"])
        if path.name.startswith("test_") and path.is_relative_to(UT_CPP):
            return str(path.relative_to(UT_CPP))
    return None


def find_gaps(build_dir: Path, jobs: int) -> tuple[list[tuple[str, str, list[str]]], list[str]]:
    by_target = read_targets(build_dir)
    cases = {name: case_of(entries) for name, entries in by_target.items()}
    covered = [name for name, case in cases.items() if case]

    # A case's own target only. The object libraries it links — <arch>_<runtime>_objs
    # and friends — are shared by most cases in their directory, and their
    # contents differ along the arch axis whatever the case does, so charging
    # those differences to a case would make every one of them look arch-varying:
    # a case whose whole subject is the arch-agnostic src/common/host_build_graph
    # handle would be required to build for both arches because something it
    # links but never calls differs. What a case compiles into its own target is
    # what it chose to cover; what it links to resolve symbols is not.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        per_target = dict(
            zip(
                covered,
                pool.map(
                    lambda name: set().union(*(files_opened(e) for e in by_target[name])),
                    covered,
                ),
            )
        )

    by_case = collections.defaultdict(list)
    for name in covered:
        by_case[cases[name]].append(name)

    gaps, unclassified = [], []
    for case, targets in sorted(by_case.items()):
        try:
            arch_axis, runtime_axis = applicable_axes(case)
        except KeyError as unknown:
            unclassified.append(f"{case} (directory {unknown.args[0]!r})")
            continue
        opened = set().union(*(per_target[t] for t in targets))

        # An axis needs covering when the directory leaves it open and the code
        # actually differs along it. Otherwise it collapses to whatever the
        # builds happen to use, and contributes one value rather than all.
        wanted = {}
        for axis, values, applicable in (
            ("arch", ARCHES, arch_axis),
            ("runtime", RUNTIMES, runtime_axis),
        ):
            if applicable and any(differs_across(f, axis) for f in opened):
                wanted[axis] = set(values)
            else:
                wanted[axis] = {None}

        # The combinations built, read the same way: a build that opens files
        # under a5/ is an a5 build, one that opens a runtime's tree is that
        # runtime's. Both axes at once, so a missing cell of the product is
        # visible even when every value appears somewhere.
        built = set()
        for target in targets:
            files = per_target[target]
            built.add(
                (
                    axis_value(files, "arch") if wanted["arch"] != {None} else None,
                    axis_value(files, "runtime") if wanted["runtime"] != {None} else None,
                )
            )
        missing = sorted(
            (arch, runtime) for arch in wanted["arch"] for runtime in wanted["runtime"] if (arch, runtime) not in built
        )
        for cell in missing:
            gaps.append((case, cell_name(cell), sorted(cell_name(b) for b in built)))
    return gaps, unclassified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=UT_CPP / "build",
        help="a configured tests/ut/cpp build carrying compile_commands.json",
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel preprocessor runs")
    args = parser.parse_args()

    gaps, unclassified = find_gaps(args.build_dir, args.jobs)
    found = {(case, cell) for case, cell, _ in gaps}

    unwaived = [(case, cell, built) for case, cell, built in gaps if (case, cell) not in WAIVERS]
    stale = sorted(key for key in WAIVERS if key not in found)

    if not unwaived and not stale and not unclassified:
        return 0

    if unclassified:
        print("These cases live in a directory this check cannot classify, so which")
        print("axes apply to them is unknown:")
        print()
        for name in unclassified:
            print(f"  {name}")
        print()
        print("Add the directory to PER_RUNTIME_TREES or SINGLE_CONTEXT_TREES in")
        print(f"{Path(__file__).relative_to(REPO_ROOT)}, per what src/ compiles it into.")

    if unwaived:
        if unclassified:
            print()
        print("These cases compile code that differs along an axis but are not built")
        print("for every combination of it, so they cover some configurations and")
        print("report on all:")
        print()
        for case, cell, built in unwaived:
            print(f"  {case}\n      missing: {cell}    built: {', '.join(built)}")
        print()
        print("Give the case the axis it is missing — PER_ARCH / PER_RUNTIME for a")
        print("platform case, ARCHS for a runtime case. Where the combination")
        print("genuinely cannot exist, add it to WAIVERS with the reason.")

    if stale:
        if unwaived or unclassified:
            print()
        print("These WAIVERS entries no longer describe a gap. Delete them, so the")
        print("table keeps counting only what is still missing:")
        print()
        for case, axis in stale:
            print(f"  {case} ({axis})")

    return 1


if __name__ == "__main__":
    sys.exit(main())
