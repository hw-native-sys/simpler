# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Artifact provenance checks for the A5 HBG performance collector."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def collector():
    path = Path(__file__).resolve().parents[3] / "tools/benchmarks/a5_hbg_graph_task/collect.py"
    spec = importlib.util.spec_from_file_location("hbg_perf_collector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_runtime(root):
    directory = root / "build/lib/a5/onboard/host_build_graph"
    directory.mkdir(parents=True)
    for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o", "aicpu_cpu_topo_fallback.json"):
        (directory / name).write_bytes(name.encode())
    return directory


def test_fingerprint_follows_runtime_bytes_across_snapshot_locations(collector, tmp_path):
    left, right = tmp_path / "old_snapshot", tmp_path / "new_snapshot"
    make_runtime(left)
    changed = make_runtime(right)
    before = collector.runtime_identity(left)
    assert before == collector.runtime_identity(right)
    assert before["revision"].startswith("runtime-sha256:")
    (tmp_path / "HEAD").write_text("an unrelated enclosing checkout revision")
    assert collector.runtime_identity(right) == before
    (changed / "aicore_kernel.o").write_bytes(b"different resident scheduler")
    assert collector.runtime_identity(right)["revision"] != before["revision"]


def test_missing_runtime_cannot_be_labeled_with_a_checkout_revision(collector, tmp_path):
    directory = make_runtime(tmp_path)
    (directory / "libaicpu_kernel.so").unlink()
    with pytest.raises(FileNotFoundError, match="libaicpu_kernel.so"):
        collector.runtime_identity(tmp_path)


def test_runtime_rebuild_during_measurement_refuses_the_result(collector, tmp_path, monkeypatch):
    directory = make_runtime(tmp_path)
    monkeypatch.setattr(collector, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setenv("TASK_DEVICE", "7")
    manifest = tmp_path / "cases.json"
    collector.write_json(manifest, {"scenes": [{"measure": True, "class": "Scene", "case": "Case", "path": "test.py"}]})

    def rebuild(*args, **kwargs):
        (directory / "libhost_runtime.so").write_bytes(b"rebuilt while the test ran")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(collector.subprocess, "run", rebuild)
    args = SimpleNamespace(output=tmp_path / "outputs", manifest=manifest, batches=3, rounds=100, warmup=3, timeout=60)
    with pytest.raises(SystemExit, match="runtime artifacts changed"):
        collector.run(args)
    assert not (args.output / "results.json").exists()


@pytest.mark.parametrize("changed_field", [None, "params", "config", "case"])
def test_manifest_is_checked_in_the_executing_pytest_process(collector, tmp_path, changed_field):
    source = tmp_path / "test_manifest_scene.py"
    marker = tmp_path / "executed"
    source.write_text(
        "from pathlib import Path\n"
        "class TestScene:\n"
        "    CASES = [{'name': 'Case', 'params': {'width': 24}, "
        "'config': {'runtime_env': {'ring_task_window': 4096}}}]\n"
        "    def _matching_cases(self, platform, request):\n"
        "        return self.CASES\n"
        "    def test_run(self):\n"
        f"        Path({str(marker)!r}).write_text('ran')\n"
    )
    case = {
        "path": str(source),
        "class": "TestScene",
        "case": "Case",
        "params": {"width": 24},
        "config": {"runtime_env": {"ring_task_window": 4096}},
    }
    if changed_field:
        case[changed_field] = "Other" if changed_field == "case" else {}
    result = subprocess.run(
        collector.pytest_command(case) + [f"{source}::TestScene::test_run", f"--confcutdir={tmp_path}", "-q"],
        cwd=collector.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (pytest.ExitCode.USAGE_ERROR if changed_field else pytest.ExitCode.OK), (
        result.stdout + result.stderr
    )
    assert marker.exists() is (changed_field is None)


def test_external_output_keeps_the_completed_case_result(collector, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    make_runtime(root)
    monkeypatch.setattr(collector, "PROJECT_ROOT", root)
    monkeypatch.setattr(collector, "ROOT", root)
    monkeypatch.setenv("TASK_DEVICE", "7")
    monkeypatch.setattr(collector.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        collector,
        "extract_metrics",
        lambda *args: {"rows": [{"host_us": 3, "device_us": 2}], "summary": {}},
    )
    manifest = root / "cases.json"
    collector.write_json(manifest, {"scenes": [{"measure": True, "class": "Scene", "case": "Case", "path": "test.py"}]})
    output = tmp_path / "external-results"
    args = SimpleNamespace(output=output, manifest=manifest, batches=1, rounds=1, warmup=0, timeout=60)
    assert collector.run(args) == 0
    records = json.loads((output / "results.json").read_text())
    assert len(records) == 1 and records[0]["batches"][0]["valid"]
    assert (root / records[0]["functional"]["log"]).is_file()


@pytest.mark.parametrize("first_invocation", [0, 1])
def test_unscoped_completion_proofs_do_not_shift_warmup(collector, tmp_path, first_invocation):
    lines = []

    def emit(inv, name, timestamp, duration):
        lines.append(f"[STRACE] v=1 pid=1 tid=1 inv={inv} hid=0 depth=0 name={name} ts={timestamp} dur={duration}")

    for name in ("proof.observe_begin", "borrow_release", "claim_release"):
        emit(0, f"chip.run.{name}", 1, 0)
    for index, duration in enumerate((3000, 2000, 1000)):
        inv = first_invocation + index
        emit(inv, "chip.run", 10000 * (index + 1), duration + 1000)
        emit(inv, "chip.run.runner_run.device_wall", 10000 * (index + 1), duration)
    log = tmp_path / "run.log"
    log.write_text("\n".join(lines) + "\n")
    timing = collector.extract_metrics(log, 1)
    assert [row["epoch"] for row in timing["rows"]] == list(range(first_invocation, first_invocation + 3))
    assert timing["summary"]["device_us"]["count"] == 2
    assert timing["summary"]["device_us"]["median"] == 1.5


def test_unscoped_proof_filter_does_not_hide_missing_root(collector, tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[STRACE] v=1 pid=1 tid=1 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1 dur=0\n"
        "[STRACE] v=1 pid=1 tid=1 inv=0 hid=0 depth=1 name=chip.run.runner_run.device_wall ts=10 dur=1000\n"
    )
    timing = collector.extract_metrics(log, 0)
    assert len(timing["rows"]) == 1
    assert timing["rows"][0]["host_us"] is None
