# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime snapshots and preparation preflight need no device reservation."""

import importlib
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
PRODUCTION = Path(__file__).resolve().parents[1] / "production"
COMPONENTS = (
    "libhost_runtime.so",
    "libhost_runtime_impl.so",
    "libaicpu_kernel.so",
    "aicore_kernel.o",
    "device_start_gate.o",
)


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(PRODUCTION))
    return importlib.import_module("device_start_gate")


def snapshot(root, gate, variant="host_build_graph"):
    runtime = root / "runtime/a5/onboard"
    files = {}
    for relative in [
        *(f"a5/onboard/{variant}/{name}" for name in COMPONENTS),
        "a5/dispatcher/libsimpler_aicpu_dispatcher.so",
        "pto_isa_build.json",
    ]:
        path = root / "runtime" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"build-A")
        files[relative] = path.stat().st_size
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_dir": str(runtime),
                "test_only": True,
                "status": "build_passed",
                "wrappers": {variant: {"executable_sections_unchanged": True}},
                "gate": {"entry": gate.ENTRY},
                "snapshot_files": files,
            }
        )
    )
    return runtime


def test_snapshot_survives_source_rebuild_including_source_symlinks(tmp_path, gate):
    source = tmp_path / "source"
    source.mkdir()
    kernel = source / "aicore_kernel.o"
    kernel.write_bytes(b"build-A")
    (source / "alias.o").symlink_to(kernel)
    copied = tmp_path / "snapshot"
    gate.copy_tree(source, copied, {}, Path("a5/onboard"))
    kernel.write_bytes(b"build-B")
    for name in ("aicore_kernel.o", "alias.o"):
        assert not (copied / name).is_symlink()
        assert (copied / name).read_bytes() == b"build-A"


@pytest.mark.parametrize("variant", ["host_build_graph", "percounter"])
def test_completed_snapshot_accepts_both_variants(tmp_path, gate, variant):
    runtime = snapshot(tmp_path / variant, gate, variant)
    gate.validate_runtime_gate(runtime, variant)


@pytest.mark.parametrize("component", [*COMPONENTS, "../dispatcher/libsimpler_aicpu_dispatcher.so", "pin"])
def test_selected_runtime_cannot_follow_rebuilt_sources(tmp_path, gate, component):
    runtime = snapshot(tmp_path / "wrapper", gate)
    if component == "pin":
        path = runtime.parents[1] / "pto_isa_build.json"
    elif component.startswith("../"):
        path = runtime / component
    else:
        path = runtime / "host_build_graph" / component
    original = tmp_path / "mutable-original"
    original.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(original)
    with pytest.raises(ValueError, match="missing or linked"):
        gate.validate_runtime_gate(runtime, "host_build_graph")


def test_truncated_snapshot_is_rejected(tmp_path, gate):
    runtime = snapshot(tmp_path / "wrapper", gate)
    (runtime / "host_build_graph/aicore_kernel.o").write_bytes(b"A")
    with pytest.raises(ValueError, match="size changed"):
        gate.validate_runtime_gate(runtime, "host_build_graph")


def test_prepare_rejects_unwrapped_runtime_before_creating_output(tmp_path, gate, monkeypatch):
    prepare = importlib.import_module("prepare")
    runtime = snapshot(tmp_path / "wrapper", gate)
    (tmp_path / "wrapper/manifest.json").unlink()
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare.py", str(runtime), str(output), "--flavor", "pypto_lib_latest"])
    with pytest.raises(SystemExit) as error:
        prepare.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("selection", [[], ["--flavor", "all"], ["--flavor", "pypto_lib_latest"]])
def test_prepare_passes_valid_snapshot_to_capture(tmp_path, gate, monkeypatch, selection):
    prepare = importlib.import_module("prepare")
    runtime = snapshot(tmp_path / "wrapper", gate)
    output = tmp_path / "prepared"
    calls = []
    monkeypatch.setattr(sys, "argv", ["prepare.py", str(runtime), str(output), *selection])
    monkeypatch.setattr(prepare.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(prepare, "export", lambda *args, **kwargs: [])

    def build_diamond(variants, *args):
        (variants / "balanced").mkdir(parents=True)

    monkeypatch.setattr(prepare, "build_diamond", build_diamond)
    monkeypatch.setattr(prepare, "build_bundle", lambda path, *args: path.mkdir(parents=True))
    prepare.main()
    if "pypto_lib_latest" not in selection:
        assert (output / "cases/synthetic/mpmd_diamond").is_dir()
    assert len(calls) == 1 and calls[0][3] == str(runtime)
    assert Path(calls[0][2]).name == "pypto_lib_latest"
    assert json.loads((output / "runtime.json").read_text())["path"] == str(runtime)
