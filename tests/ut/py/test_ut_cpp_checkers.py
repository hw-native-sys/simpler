# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The coverage checkers' own failure paths.

`tests/lint/check_ut_cpp_axis.py` reports a case that covers one configuration
of a tree the product compiles several times. Running it against the real tree
proves it passes; it does not prove it would still fail when it should — and a
checker that silently stops finding gaps is indistinguishable from a tree that
has none, which is the whole class of defect it exists to catch.

So each test here builds a synthetic src/ tree and compilation database with
one planted gap, and asserts the checker finds exactly it. The tree is
synthetic on purpose: a fixture carved out of the real one would drift with it,
and the properties under test — "the difference is only in a .cpp body", "the
argument has a space in it" — are hard to guarantee in real code and trivial
to state here.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LINT = _REPO_ROOT / "tests" / "lint"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _LINT / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"{name}.py is not importable from {_LINT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


axis = _load("check_ut_cpp_axis")

# The checkers replay real compile lines, so these need a real compiler. Every
# platform this repo builds on has one; skipping rather than failing keeps the
# suite runnable where it is only Python being tested.
_FOUND_CXX = shutil.which("c++") or shutil.which("g++")
pytestmark = pytest.mark.skipif(_FOUND_CXX is None, reason="no c++ on PATH to replay compile lines with")
# Narrowed for the tests, which the marker above has already gated.
_CXX = _FOUND_CXX or "c++"


class Tree:
    """A synthetic src/ + tests/ut/cpp pair with a compilation database."""

    def __init__(self, root: Path):
        self.root = root
        self.src = root / "src"
        self.ut = root / "tests" / "ut" / "cpp"
        self.build = root / "build"
        self.build.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def compile_entry(self, target: str, source: str, includes: tuple[str, ...] = ()) -> None:
        """Record a compile line for one translation unit of `target`.

        `command` is written the way a compilation database writes it — one
        shell line, each argument quoted if it needs to be — because that
        quoting is itself under test.
        """
        argv = [_CXX]
        argv += [f"-I{self.root / directory}" for directory in includes]
        argv += ["-std=c++17", "-o", f"{self.build}/CMakeFiles/{target}.dir/{Path(source).name}.o", "-c"]
        argv += [str(self.root / source)]
        self.entries.append(
            {"directory": str(self.build), "command": shlex.join(argv), "file": str(self.root / source)}
        )

    def install(self, monkeypatch) -> Path:
        (self.build / "compile_commands.json").write_text(json.dumps(self.entries))
        monkeypatch.setattr(axis, "SRC", self.src)
        monkeypatch.setattr(axis, "UT_CPP", self.ut)
        return self.build


def _gaps(tree: Tree, monkeypatch) -> list[tuple[str, str]]:
    build = tree.install(monkeypatch)
    found, unclassified = axis.find_gaps(build, jobs=4)
    assert not unclassified, f"the synthetic tree should classify cleanly: {unclassified}"
    return [(case, cell) for case, cell, _ in found]


# --- the axis a difference lives on ----------------------------------------------------------


def test_a_difference_only_in_a_cpp_body_is_still_a_gap(tmp_path, monkeypatch):
    """`-H` lists headers, not the source it was handed.

    A tree whose two arch copies differ only in a .cpp body is the case this
    misses: every header the case opens is identical across the axis, so the
    axis reads as collapsed and one build looks like every build.
    """
    tree = Tree(tmp_path)
    tree.write("src/common/platform/shared/host/knob.h", "#pragma once\nint knob();\n")
    # Same declaration both sides, different body — the header cannot tell them apart.
    tree.write("src/a2a3/platform/knob.cpp", '#include "shared/host/knob.h"\nint knob() { return 2; }\n')
    tree.write("src/a5/platform/knob.cpp", '#include "shared/host/knob.h"\nint knob() { return 5; }\n')
    tree.write(
        "tests/ut/cpp/common/platform/test_knob.cpp", '#include "shared/host/knob.h"\nint main() { return 0; }\n'
    )

    tree.compile_entry("test_a2a3_knob", "tests/ut/cpp/common/platform/test_knob.cpp", ("src/common/platform",))
    tree.compile_entry("test_a2a3_knob", "src/a2a3/platform/knob.cpp", ("src/common/platform",))

    assert _gaps(tree, monkeypatch) == [("common/platform/test_knob.cpp", "a5")]


def test_a_difference_in_a_linked_object_library_is_not_the_cases_gap(tmp_path, monkeypatch):
    """An object library is shared, so its arch differences are nobody's coverage.

    Charging them to each case that links it would require every case in the
    directory to build for both arches — including one whose whole subject is
    an arch-agnostic header — because something it links but never calls
    differs.
    """
    tree = Tree(tmp_path)
    tree.write("src/common/handle.h", "#pragma once\ninline int handle() { return 1; }\n")
    tree.write("src/a2a3/platform/regs.cpp", "int regs() { return 2; }\n")
    tree.write("src/a5/platform/regs.cpp", "int regs() { return 5; }\n")
    tree.write("tests/ut/cpp/common/platform/test_handle.cpp", '#include "handle.h"\nint main() { return handle(); }\n')

    tree.compile_entry("test_handle", "tests/ut/cpp/common/platform/test_handle.cpp", ("src/common",))
    # A target of its own, carrying no test_*.cpp — the shape of <arch>_<runtime>_objs.
    tree.compile_entry("a2a3_fake_objs", "src/a2a3/platform/regs.cpp", ("src/common",))

    assert _gaps(tree, monkeypatch) == []


def test_a_case_directly_under_arch_runtime_keeps_its_runtime_axis(tmp_path, monkeypatch):
    """`<arch>/runtime/` holds the cases every runtime carries, built once each.

    Only `<arch>/runtime/<runtime>/` fixes the runtime, so treating "not
    platform" as "one runtime" drops the axis for exactly the cases that do
    repeat along it.
    """
    tree = Tree(tmp_path)
    tree.write("src/common/host_build_graph/task_id.h", "#pragma once\ninline int space() { return 1; }\n")
    tree.write("src/common/tensormap_and_ringbuffer/task_id.h", "#pragma once\ninline int space() { return 2; }\n")
    tree.write(
        "tests/ut/cpp/a2a3/runtime/test_executor.cpp",
        '#include "task_id.h"\nint main() { return space(); }\n',
    )

    # Built for one runtime only: the include path names host_build_graph.
    tree.compile_entry(
        "test_a2a3_hbg_executor",
        "tests/ut/cpp/a2a3/runtime/test_executor.cpp",
        ("src/common/host_build_graph",),
    )

    assert _gaps(tree, monkeypatch) == [
        ("a2a3/runtime/test_executor.cpp", "tensormap_and_ringbuffer"),
    ]


def test_a_case_under_arch_runtime_runtime_has_no_runtime_axis(tmp_path, monkeypatch):
    """The directory below names the runtime, so one build is the whole story."""
    tree = Tree(tmp_path)
    tree.write("src/common/host_build_graph/task_id.h", "#pragma once\ninline int space() { return 1; }\n")
    tree.write("src/common/tensormap_and_ringbuffer/task_id.h", "#pragma once\ninline int space() { return 2; }\n")
    tree.write(
        "tests/ut/cpp/a2a3/runtime/host_build_graph/test_slot.cpp",
        '#include "task_id.h"\nint main() { return space(); }\n',
    )

    tree.compile_entry(
        "test_a2a3_hbg_slot",
        "tests/ut/cpp/a2a3/runtime/host_build_graph/test_slot.cpp",
        ("src/common/host_build_graph",),
    )

    assert _gaps(tree, monkeypatch) == []


# --- replaying the compile line ---------------------------------------------------------------


def test_an_argument_with_a_space_survives_the_replay(tmp_path, monkeypatch):
    """The compile line is a shell line, so replaying it means un-quoting it.

    Splitting on whitespace turns a legal `-I"/path/with space"` into two wrong
    arguments. Since a failed replay is now an error rather than an empty
    result, that breaks a legal build instead of under-reporting it.
    """
    tree = Tree(tmp_path)
    tree.write("src/common/dir with space/knob.h", "#pragma once\ninline int knob() { return 1; }\n")
    tree.write(
        "tests/ut/cpp/common/utils/test_spaced.cpp",
        '#include "knob.h"\nint main() { return knob(); }\n',
    )
    tree.compile_entry(
        "test_spaced",
        "tests/ut/cpp/common/utils/test_spaced.cpp",
        ("src/common/dir with space",),
    )
    tree.install(monkeypatch)

    opened = axis.files_opened(tree.entries[0])
    assert "common/dir with space/knob.h" in opened, "the quoted include directory did not survive"


def test_a_define_keeps_the_quotes_the_shell_would_have_removed():
    r"""`-DNAME=\"x\"` in a shell line means `-DNAME="x"` to the compiler.

    Asserted on the argv rather than by replaying it, because `-E` does not
    parse: a macro body left as `\"x\"` preprocesses without complaint and
    only breaks a real compile, so a replay proves nothing here.
    """
    entry = {
        "directory": "/nowhere",
        "command": r'/usr/bin/c++ -DNAME=\"a2a3sim\" -I"/dir with space" -std=c++17 -c a.cpp',
        "file": "a.cpp",
    }
    assert axis.command_argv(entry) == [
        "/usr/bin/c++",
        '-DNAME="a2a3sim"',
        "-I/dir with space",
        "-std=c++17",
        "-c",
        "a.cpp",
    ]


def test_the_arguments_field_is_preferred_when_present(tmp_path, monkeypatch):
    """A database carrying the already-split form needs no un-quoting at all."""
    tree = Tree(tmp_path)
    tree.write("src/common/dir with space/knob.h", "#pragma once\ninline int knob() { return 1; }\n")
    source = tree.write(
        "tests/ut/cpp/common/utils/test_argv.cpp",
        '#include "knob.h"\nint main() { return knob(); }\n',
    )
    entry = {
        "directory": str(tree.build),
        "arguments": [
            _CXX,
            f"-I{tree.root / 'src/common/dir with space'}",
            "-std=c++17",
            "-o",
            f"{tree.build}/CMakeFiles/test_argv.dir/test_argv.cpp.o",
            "-c",
            str(source),
        ],
        "command": "this field must not be read when arguments is present",
        "file": str(source),
    }
    tree.entries.append(entry)
    tree.install(monkeypatch)

    assert "common/dir with space/knob.h" in axis.files_opened(entry)


def test_a_failed_replay_is_an_error_not_an_empty_result(tmp_path, monkeypatch):
    """An unknown set of opened files reads as a case reaching neither tree.

    That is what SINGLE asserts, so a failed replay would report the gap this
    check looks for as absent.
    """
    tree = Tree(tmp_path)
    tree.write("tests/ut/cpp/common/utils/test_broken.cpp", '#include "nowhere/missing.h"\nint main() { return 0; }\n')
    tree.compile_entry("test_broken", "tests/ut/cpp/common/utils/test_broken.cpp")
    tree.install(monkeypatch)

    with pytest.raises(SystemExit, match="Preprocessing"):
        axis.files_opened(tree.entries[0])


# --- the database itself ----------------------------------------------------------------------


def test_an_entry_naming_no_target_is_an_error(tmp_path, monkeypatch):
    """Skipping it would understate coverage by however much it held.

    Mixed with attributable entries on purpose: a database where *nothing*
    resolves is a different fault, reported separately, and would not exercise
    the per-entry path.
    """
    tree = Tree(tmp_path)
    tree.write("tests/ut/cpp/common/utils/test_found.cpp", "int main() { return 0; }\n")
    tree.compile_entry("test_found", "tests/ut/cpp/common/utils/test_found.cpp")
    loose = tree.write("tests/ut/cpp/common/utils/test_loose.cpp", "int main() { return 0; }\n")
    tree.entries.append(
        {
            "directory": str(tree.build),
            "command": f"{_CXX} -std=c++17 -o loose.o -c {loose}",
            "file": str(loose),
        }
    )
    build = tree.install(monkeypatch)

    with pytest.raises(SystemExit, match="name no"):
        axis.read_targets(build)


def test_a_database_where_nothing_resolves_is_an_error(tmp_path, monkeypatch):
    """Reported apart from the mixed case: there is no target to report against."""
    tree = Tree(tmp_path)
    loose = tree.write("tests/ut/cpp/common/utils/test_loose.cpp", "int main() { return 0; }\n")
    tree.entries.append(
        {
            "directory": str(tree.build),
            "command": f"{_CXX} -std=c++17 -o loose.o -c {loose}",
            "file": str(loose),
        }
    )
    build = tree.install(monkeypatch)

    with pytest.raises(SystemExit, match="No target could be read"):
        axis.read_targets(build)


def test_runtimes_come_from_the_tree_rather_than_a_list():
    """The same rule platform_info and runtimes.cmake use, so a third runtime
    needs no edit here to be classified."""
    discovered = axis.discovered_runtimes()
    assert discovered, "no runtime discovered from src/"
    for runtime in discovered:
        assert (_REPO_ROOT / "src" / "a2a3" / "runtime" / runtime / "build_config.py").exists()
