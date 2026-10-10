# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build-time Git provenance written into wheel assets."""

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIN_SCRIPT = PROJECT_ROOT / "cmake" / "write_simpler_pin.cmake"


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603


def _repo(path: Path) -> str:
    path.mkdir()
    _run("git", "init", "-q", str(path))
    (path / ".gitignore").write_text("build/\n*.bak\n__pycache__/\ncompile_commands.json\n")
    (path / "source.txt").write_text("original\n")
    _run("git", "-C", str(path), "add", ".gitignore", "source.txt")
    _run(
        "git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"
    )
    return _run("git", "-C", str(path), "rev-parse", "HEAD")


def _write_pin(source: Path, output: Path) -> Path:
    _run(
        "cmake",
        f"-DSIMPLER_PIN_SOURCE_DIR={source}",
        f"-DSIMPLER_PIN_OUTPUT_DIR={output}",
        "-P",
        str(PIN_SCRIPT),
    )
    return output / "simpler.pin"


def test_clean_repo_writes_revision_without_counting_ignored_build_files(tmp_path: Path):
    source = tmp_path / "source"
    revision = _repo(source)
    (source / "build").mkdir()
    (source / "build" / "generated.o").write_text("ignored\n")

    pin = _write_pin(source, tmp_path / "output")

    assert pin.read_text() == f"revision={revision}\nclean=true\n"


@pytest.mark.parametrize("input_dir", ["src", "cmake", "simpler_setup", "python/simpler", "python/bindings"])
def test_ignored_packaging_input_marks_repo_dirty(tmp_path: Path, input_dir: str):
    source = tmp_path / "source"
    revision = _repo(source)
    directory = source / input_dir
    directory.mkdir(parents=True)
    (directory / "local.bak").write_text("ignored input\n")

    pin = _write_pin(source, tmp_path / "output")

    assert pin.read_text() == f"revision={revision}\nclean=false\n"


def test_generated_ignored_files_do_not_mark_repo_dirty(tmp_path: Path):
    source = tmp_path / "source"
    revision = _repo(source)
    (source / "src" / "runtime").mkdir(parents=True)
    (source / "src" / "runtime" / "compile_commands.json").write_text("[]\n")
    (source / "simpler_setup" / "__pycache__").mkdir(parents=True)
    (source / "simpler_setup" / "__pycache__" / "cache.pyc").write_text("ignored\n")
    (source / "docs").mkdir()
    (source / "docs" / "notes.bak").write_text("not a package input\n")

    pin = _write_pin(source, tmp_path / "output")

    assert pin.read_text() == f"revision={revision}\nclean=true\n"


@pytest.mark.parametrize("change", ["tracked", "untracked", "staged"])
def test_dirty_repo_writes_false(tmp_path: Path, change: str):
    source = tmp_path / "source"
    revision = _repo(source)
    if change == "tracked":
        (source / "source.txt").write_text("modified\n")
    else:
        (source / "new.txt").write_text("new\n")
        if change == "staged":
            _run("git", "-C", str(source), "add", "new.txt")

    pin = _write_pin(source, tmp_path / "output")

    assert pin.read_text() == f"revision={revision}\nclean=false\n"


def test_non_git_source_removes_stale_pin(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    pin = output / "simpler.pin"
    pin.write_text("stale\n")

    _write_pin(source, output)

    assert not pin.exists()


def test_nested_tarball_does_not_inherit_parent_revision(tmp_path: Path):
    parent = tmp_path / "parent"
    _repo(parent)
    source = parent / "tarball"
    source.mkdir()

    pin = _write_pin(source, tmp_path / "output")

    assert not pin.exists()


def test_repo_without_commit_has_no_revision(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _run("git", "init", "-q", str(source))

    pin = _write_pin(source, tmp_path / "output")

    assert not pin.exists()
