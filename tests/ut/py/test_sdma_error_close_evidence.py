# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for isolated SDMA failure evidence; no device is initialized."""

import json
import subprocess
import sys

import pytest

from tests.st.a5.host_build_graph.sdma_defer_wait._error_close_evidence import (
    check_fault_captures,
    run_error_close_probe,
)


@pytest.fixture
def snapshot():
    # Native A5 HBG protocol: task 0 returned after accepting SDMA, then overflowed.
    return {
        "chip_swimlane_level": 4,
        "aicore_protocol": {
            "schema": "hbg-native-protocol",
            "producer": "aicore",
            "run_epoch": 2,
            "scheduler_error": 12,
            "error_claimed": 1,
            "tasks": [{"task_ref": 0, "active_mask": 2, "block_num": 1, "kernel_ids": [65535, 0, 65535]}],
            "executions": [
                {"task_ref": 0, "run_epoch": 2, "worker_index": 46, "entry": [100, 2], "returned": [200, 3]}
            ],
            "first_error": {"task_ref": 0, "run_epoch": 2, "worker_index": 46, "block_idx": 0},
            "task_controls": [{"task_ref": 0, "state": 0, "completed_subtasks": 0}],
            "resolved_task_count": 0,
        },
    }


def write_capture(output, snapshot, rank=0, **identity):
    capture = output / "fault-capture" / f"rank{rank}" / "d0"
    capture.mkdir(parents=True)
    (capture / "chip_swimlane_records.json").write_text(json.dumps(snapshot))
    (capture / "dispatch_identity.json").write_text(json.dumps({"chip_rank": rank, "run_id": 2, **identity}))


def probe(tmp_path, snapshot, *, ranks=2, returncode=0, retention=True):
    script = tmp_path / "probe.py"
    script.write_text(
        "import json, os, pathlib, sys\n"
        "root = pathlib.Path(os.environ['ASCEND_PROCESS_LOG_PATH'])\n"
        "assert root.is_absolute() and root.is_dir()\n"
        "assert os.environ['ASCEND_GLOBAL_LOG_LEVEL'] == '3'\n"
        "assert os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] == '0'\n"
        "print('healthy SDMA verified')\n"
        "print('fault evidence', file=sys.stderr)\n"
        f"for rank in range({ranks}):\n"
        "    capture = root.parent / 'fault-capture' / f'rank{rank}' / 'd0'\n"
        "    capture.mkdir(parents=True)\n"
        f"    (capture / 'chip_swimlane_records.json').write_text({json.dumps(snapshot)!r})\n"
        "    (capture / 'dispatch_identity.json').write_text(json.dumps({'chip_rank': rank, 'run_id': 2}))\n"
        f"if {retention!r}:\n"
        f"    (root.parent / 'retention.txt').write_text('{ranks} ranks: failed SDMA run retained its domain\\n')\n"
        f"sys.exit({returncode})\n"
    )
    return [sys.executable, str(script)]


@pytest.mark.parametrize("devices", [(4, 7), (4, 5, 6, 7)])
def test_environment_and_per_rank_evidence_survive_without_device_logs(tmp_path, monkeypatch, snapshot, devices):
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(tmp_path / "old-logs"))
    monkeypatch.setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    output = tmp_path / "case"
    run_error_close_probe(probe(tmp_path, snapshot, ranks=len(devices)), output, devices)
    assert (output / "stdout.log").read_text() == "healthy SDMA verified\n"
    assert (output / "stderr.log").read_text() == "fault evidence\n"
    for rank, device in enumerate(devices):
        evidence = json.loads((output / f"device-{device}-error.json").read_text())
        assert evidence["identity"] == {"chip_rank": rank, "run_id": 2}
        assert evidence["protocol"]["scheduler_error"] == 12
    assert not (tmp_path / "old-logs").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("scheduler_error", 10),
        ("run_epoch", 1),
        ("executions", []),
        ("first_error", {"task_ref": 0, "run_epoch": 2, "worker_index": 47, "block_idx": 0}),
        ("task_controls", [{"task_ref": 0, "state": 2, "completed_subtasks": 1}]),
        ("resolved_task_count", 1),
    ],
)
def test_wrong_fault_or_completed_work_is_rejected(tmp_path, snapshot, field, value):
    snapshot["aicore_protocol"][field] = value
    write_capture(tmp_path, snapshot)
    with pytest.raises(AssertionError):
        check_fault_captures(tmp_path, [4])


@pytest.mark.parametrize("identity", [{"chip_rank": 1}, {"run_id": 1}])
def test_foreign_rank_or_healthy_run_is_rejected(tmp_path, snapshot, identity):
    write_capture(tmp_path, snapshot, **identity)
    with pytest.raises(AssertionError, match="identity|injected run"):
        check_fault_captures(tmp_path, [4])


def test_each_requested_rank_needs_its_own_capture(tmp_path, snapshot):
    write_capture(tmp_path, snapshot)
    with pytest.raises(AssertionError, match="fault capture for rank 1"):
        check_fault_captures(tmp_path, [4, 7])


def test_native_fault_does_not_replace_domain_retention_proof(tmp_path, snapshot):
    with pytest.raises(AssertionError, match="domain-retention proof"):
        run_error_close_probe(probe(tmp_path, snapshot, retention=False), tmp_path / "case", [4, 7])


def test_child_failure_is_not_hidden_by_expected_native_error(tmp_path, snapshot):
    output = tmp_path / "case"
    with pytest.raises(AssertionError, match="SDMA error-close probe exited 9"):
        run_error_close_probe(probe(tmp_path, snapshot, returncode=9), output, [4, 7])
    assert "fault evidence" in (output / "stderr.log").read_text()


def test_timeout_keeps_partial_output(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        kwargs["stdout"].write("healthy pass\n")
        kwargs["stderr"].write("fault pending\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    output = tmp_path / "case"
    with pytest.raises(subprocess.TimeoutExpired):
        run_error_close_probe(["probe"], output, [4, 7])
    assert (output / "stdout.log").read_text() == "healthy pass\n"
    assert (output / "stderr.log").read_text() == "fault pending\n"
