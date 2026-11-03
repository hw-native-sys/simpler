# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bad-record checks for the resident-timeout fixture, not device failure evidence."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def native(request):
    request.config.addinivalue_line("markers", "scheduler_mode(kind): required scheduler selection")
    path = Path(__file__).resolve().parents[3] / "st/a5/host_build_graph/resident_failure/test_resident_failure.py"
    spec = importlib.util.spec_from_file_location("_resident_failure_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def test_failed_completion_proof_prints_owned_borrow_release(native, monkeypatch, tmp_path, capsys):
    # Controlled file/log leaves exercise the real failure helper, not device proof.
    logs = tmp_path / "owned-host-logs"
    logs.mkdir()
    records = [
        "[TIMING] name=chip.run.proof.query_return e=0x10 rc=0 status_valid=1 status=1 fresh=1",
        "[TIMING] name=chip.run.proof.consume ep=2 gen=2 disp=2 slot=0 role=0 cache=0 owner=1",
        "[TIMING] name=chip.run.borrow_release ctx=0x20 borrow=0x30 ep=2 gen=2 disp=2 slot=0 run=2 proof=1 keep=0",
        "[TIMING] name=chip.run.borrow_release.invalid truncated=1",
    ]
    raw = ("\n".join(records) + "\n").encode()
    (logs / f"host.{os.getpid()}.log").write_bytes(raw)
    monkeypatch.setattr(native._task_interface, "_native_host_log_directory", lambda: str(logs))
    monkeypatch.setattr(native._task_interface, "_flush_host_log", lambda timeout: True)
    monkeypatch.setattr(native._task_interface, "_host_log_pending_records", lambda: 0)
    monkeypatch.setattr(native._task_interface, "_host_log_dropped_records", lambda: 1)

    row = native._retain_completion_proof(tmp_path, "controlled-failure", [], failed=True)

    stderr = capsys.readouterr().err
    for record in records:
        assert record in stderr
    assert row["proof_records"] == len(records)
    assert row["host_log_complete"] is False
    assert row["dropped"] == 1
    assert row["printed_records_truncated"] is False
    assert (tmp_path / row["log"]).read_bytes() == raw
    assert json.loads((tmp_path / "proof-controlled-failure.json").read_text()) == row
    assert json.loads(stderr.splitlines()[0].removeprefix("RESIDENT_COMPLETION_PROOF ")) == row


def test_resident_routes_registration_to_the_owned_log_before_worker_creation(native, tmp_path):
    # A fresh process isolates the real logger's first-nonempty directory binding.
    script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        import sys
        from pathlib import Path
        from simpler import task_interface as api

        spec = importlib.util.spec_from_file_location("resident_log_route", sys.argv[1])
        scene = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scene
        spec.loader.exec_module(scene)
        artifacts = Path(sys.argv[2])
        assert not api._native_host_log_directory()
        api._initialize_host_log(25)
        registration = "Callable registration: device=5 runner=0x1234 cid=0"
        events = {}

        class RegistrationComplete(Exception):
            pass

        class LoggedWorker:
            def __init__(self, **kwargs):
                events["bound_at_construction"] = api._native_host_log_directory()

            def register(self, target):
                assert api._native_emit_host_log(25, "controlled_registration", registration)
                assert api._flush_host_log(1000)
                return object()

            def init(self):
                api._initialize_host_log(25)
                raise RegistrationComplete()

            def close(self):
                events["closed"] = True

        scene._compile = lambda platform, *, native_graph=False: object()
        scene.Worker = LoggedWorker
        try:
            scene._run_pass("a5", 5, artifacts, 1, 0, 1, 2500, recovery=False)
        except RegistrationComplete:
            pass
        else:
            raise AssertionError("the controlled Worker did not reach registration/init")
        # Binding here models the later output-prefix binding if registration was missed.
        directory = api._bind_host_log_session_directory()
        assert api._native_emit_host_log(25, "controlled_tail", "registration route finished")
        row = scene._retain_completion_proof(artifacts, "registration-route", [], failed=False)
        raw = (artifacts / row["log"]).read_text()
        events.update(pid=os.getpid(), snapshot=row, raw=raw)
        print(json.dumps(events), flush=True)
        assert registration in raw, "the real native logger left registration only on stderr"
        assert events["bound_at_construction"] == directory
        assert events["closed"]
        assert row["host_log_complete"] and row["pid"] == os.getpid()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(native.__file__), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Callable registration: device=5 runner=0x1234 cid=0" not in result.stderr


def test_resident_log_bind_failure_precedes_worker_creation(native, monkeypatch, tmp_path):
    events = []

    def refuse_binding():
        events.append("bind")
        raise RuntimeError("controlled host-log binding refused")

    def create_worker(**kwargs):
        pytest.fail("a Worker was created before the diagnostic sink was bound")

    monkeypatch.setattr(native, "_compile", lambda platform, *, native_graph=False: events.append("compile"))
    monkeypatch.setattr(native._task_interface, "_bind_host_log_session_directory", refuse_binding)
    monkeypatch.setattr(native, "Worker", create_worker)
    with pytest.raises(RuntimeError, match="controlled host-log binding refused"):
        native._run_pass("a5", 5, tmp_path, 1, 0, 1, 2500, recovery=False)
    assert events == ["compile", "bind"]


def _capture():
    return {
        "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 3},
        "aicore_protocol": {
            "schema": "hbg-native-protocol-v2",
            "producer": "aicore",
            "scheduler_error": 10,
            "first_error": {"site": 95},
            "worker_count": 3,
            "active_worker_count": 3,
            "cluster_count": 1,
            "workers": [{"worker_index": worker, "finished": 2, "error_flags": 0} for worker in range(3)],
            "tasks": [
                {"task_ref": 0, "active_mask": 3, "kernel_ids": [0, 1, 65535], "block_num": 2, "dependencies": []},
                {"task_ref": 1, "active_mask": 2, "kernel_ids": [65535, 3, 65535], "block_num": 1, "dependencies": [0]},
            ],
            "executions": [
                {
                    "task_ref": 0,
                    "block_idx": 0,
                    "subtask_slot": 0,
                    "worker_index": 0,
                    "entry": [100, 1],
                    "returned": [2_500_000_100, 2],
                    "visible": [2_500_000_101, 3],
                }
            ],
            "task_controls": [
                {"task_ref": 0, "state": 0, "completed_blocks": 1, "next_block": 2, "claim_tickets": 0},
                {"task_ref": 1, "state": 0, "completed_blocks": 0, "next_block": 0, "claim_tickets": 0},
            ],
            "resolved_task_count": 0,
        },
        "aicpu_lifecycle_records": [
            {
                "aicpu_thread_id": 0,
                "exit_signal_start_cycles": 2_000_000_100,
                "exit_signal_end_cycles": 2_000_000_150,
                "exit_wait_start_cycles": 2_000_000_200,
                "exit_wait_end_cycles": 2_500_000_200,
            },
            {
                "aicpu_thread_id": 1,
                "exit_signal_start_cycles": 2_000_000_101,
                "exit_signal_end_cycles": 2_000_000_151,
                "exit_wait_start_cycles": 2_000_000_201,
                "exit_wait_end_cycles": 2_000_000_300,
            },
        ],
    }


@pytest.mark.parametrize("site", (95, 102))
def test_both_real_timeout_writers_have_the_same_containment_requirements(native, site):
    raw = _capture()
    raw["aicore_protocol"]["first_error"]["site"] = site
    native._check_failed_capture(raw, 2, 3, 0)


def _large_ticket_capture(mask, lane, blocks=257):
    raw = _capture()
    protocol = raw["aicore_protocol"]
    slot = 1 if mask == 4 else lane
    effective_mask = 2 if mask == 4 else mask
    protocol["tasks"][0].update(
        active_mask=effective_mask,
        kernel_ids=[index if effective_mask & (1 << index) else 65535 for index in range(3)],
        block_num=blocks,
    )
    protocol["executions"][0].update(subtask_slot=slot, worker_index=slot)
    protocol["task_controls"][0].update(next_block=blocks, claim_tickets=blocks, completed_blocks=min(1, blocks - 1))
    # Either lifecycle partition may own the delayed lane.
    for record in raw["aicpu_lifecycle_records"]:
        record["exit_wait_end_cycles"] = 2_500_000_200
    return raw


@pytest.mark.parametrize("mask,lane", ((1, 0), (2, 1), (4, 2), (6, 1), (7, 0), (7, 1), (7, 2)))
@pytest.mark.parametrize("version", (2, 3, 4, 5))
@pytest.mark.parametrize("blocks", (1, 2, 3))
@pytest.mark.parametrize("tail", (False, True))
def test_small_fault_capture_requires_the_versioned_claim_policy(native, mask, lane, version, blocks, tail):
    raw = _large_ticket_capture(mask, lane, blocks)
    protocol = raw["aicore_protocol"]
    protocol.update(schema=f"hbg-native-protocol-v{version}", cluster_count=2, worker_count=6, active_worker_count=6)
    raw["metadata"]["num_cores"] = 6
    protocol["workers"] = [{"worker_index": worker, "finished": 2, "error_flags": 0} for worker in range(6)]
    positions = 4 if mask in (2, 4) else 2
    uses_tickets = version >= 4 and blocks > 1
    if version == 5:
        for control in protocol["task_controls"]:
            control["admission_claim_lock"] = 0
    protocol["task_controls"][0]["claim_tickets"] = blocks + (positions if tail else 0) if uses_tickets else 0
    native._check_failed_capture(raw, blocks, mask, lane)
    protocol["task_controls"][0]["claim_tickets"] = 0 if uses_tickets else blocks
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, blocks, mask, lane)


@pytest.mark.parametrize("mask", (1, 2, 4, 6, 7))
def test_two_wave_fault_asks_the_device_for_its_own_claimant_population(native, mask):
    # The host cannot read the cluster count before the run, so it asks with a
    # negative sentinel and the orchestration launches two waves against the
    # device's own topology. The host only sizes buffers, for the widest
    # topology the platform admits.
    assert native._TWO_WAVE_REQUEST < 0
    assert native._two_wave_capacity(mask) == native._MAX_CLUSTERS * (4 if mask in (2, 4) else 2)


@pytest.mark.parametrize("mask", (1, 2, 4, 6, 7))
@pytest.mark.parametrize("fault", ("schema", "clusters", "blocks"))
@pytest.mark.parametrize("version", (3, 4, 5))
def test_two_wave_fault_cannot_silently_run_with_a_different_topology(native, mask, fault, version):
    protocol = {"schema": f"hbg-native-protocol-v{version}", "cluster_count": 32}
    blocks = 128 if mask in (2, 4) else 64
    native._check_two_wave_topology(protocol, blocks, mask, True)
    if fault == "schema":
        protocol["schema"] = "hbg-native-protocol-v2"
    elif fault == "clusters":
        protocol["cluster_count"] = 8
    else:
        blocks += 1
    with pytest.raises(AssertionError):
        native._check_two_wave_topology(protocol, blocks, mask, True)


@pytest.mark.parametrize("mask,lane", ((1, 0), (2, 1), (4, 2), (6, 1), (7, 0), (7, 1), (7, 2)))
@pytest.mark.parametrize("version", (2, 3, 4, 5))
def test_fault_capture_keeps_versioned_two_wave_claim_policy(native, mask, lane, version):
    positions = 2 if mask in (2, 4) else 1
    blocks = positions * 2
    raw = _large_ticket_capture(mask, lane, blocks)
    protocol = raw["aicore_protocol"]
    protocol["schema"] = f"hbg-native-protocol-v{version}"
    if version == 5:
        for control in protocol["task_controls"]:
            control["admission_claim_lock"] = 0
    protocol["task_controls"][0]["claim_tickets"] = blocks if version >= 3 else 0
    native._check_failed_capture(raw, blocks, mask, lane)
    protocol["task_controls"][0]["claim_tickets"] = 0 if version >= 3 else blocks
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, blocks, mask, lane)


@pytest.mark.parametrize("task", (0, 1))
@pytest.mark.parametrize("value", (None, False, True, 1, -1, (1 << 64) - 1))
def test_v5_kernel_timeout_cannot_hide_a_held_or_missing_admission_claim_lock(native, task, value):
    raw = _large_ticket_capture(7, 0)
    protocol = raw["aicore_protocol"]
    protocol["schema"] = "hbg-native-protocol-v5"
    for control in protocol["task_controls"]:
        control["admission_claim_lock"] = 0
    native._check_failed_capture(raw, 257, 7, 0)
    if value is None:
        del protocol["task_controls"][task]["admission_claim_lock"]
    else:
        protocol["task_controls"][task]["admission_claim_lock"] = value
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 257, 7, 0)


@pytest.mark.parametrize("mask,lane", ((1, 0), (2, 1), (4, 2), (6, 1), (7, 0), (7, 1), (7, 2)))
@pytest.mark.parametrize("tail_tickets", (False, True))
def test_large_ticket_failure_accepts_valid_claims_and_bounded_tail(native, mask, lane, tail_tickets):
    raw = _large_ticket_capture(mask, lane)
    protocol = raw["aicore_protocol"]
    positions = protocol["cluster_count"] * (2 if mask in (2, 4) else 1)
    if tail_tickets:
        protocol["task_controls"][0]["claim_tickets"] += positions
    native._check_failed_capture(raw, 257, mask, lane)


@pytest.mark.parametrize("mask,lane", ((1, 0), (2, 1), (4, 2), (6, 1), (7, 0)))
@pytest.mark.parametrize(
    "corruption", ("old_schema", "no_ticket", "unbounded_tail", "claim_past_n", "claim_past_ticket", "consumer_ticket")
)
def test_large_ticket_failure_rejects_inconsistent_claim_accounting(native, mask, lane, corruption):
    raw = _large_ticket_capture(mask, lane)
    protocol = raw["aicore_protocol"]
    root, consumer = protocol["task_controls"]
    if corruption == "old_schema":
        protocol["schema"] = "hbg-native-protocol-v1"
    elif corruption == "no_ticket":
        root["claim_tickets"] = 0
    elif corruption == "unbounded_tail":
        positions = protocol["cluster_count"] * (2 if mask in (2, 4) else 1)
        root["claim_tickets"] = 257 + positions + 1
    elif corruption == "claim_past_n":
        root.update(next_block=258, claim_tickets=258)
    elif corruption == "claim_past_ticket":
        root["claim_tickets"] = 256
    else:
        consumer["claim_tickets"] = 1
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 257, mask, lane)


def test_failed_capture_requires_explicit_cluster_topology(native):
    raw = _capture()
    del raw["aicore_protocol"]["cluster_count"]
    with pytest.raises(KeyError, match="cluster_count"):
        native._check_failed_capture(raw, 2, 3, 0)


def test_nonreturning_probe_is_hardware_only_and_covers_each_executing_lane(native):
    test = native.test_nonreturning_kernel_timeout_keeps_failed_run_contained
    platforms = [mark.args[0] for mark in test.pytestmark if mark.name == "platforms"]
    assert platforms == [["a5"]]
    shapes = [mark.args[1] for mark in test.pytestmark if mark.name == "parametrize"]
    assert len(shapes) == 1
    entries = shapes[0]
    assert sorted(entry.values for entry in entries) == [
        (mask, lane) for mask in range(1, 8) for lane in range(3) if mask & (1 << lane)
    ]
    # The sweep keeps every shape; what the tiering decides is which of them a
    # PR waits for. An unmarked entry is Per-PR, and every other one carries
    # exactly the a5 manual mark that defers it to the daily sweep.
    assert {entry.values for entry in entries if not entry.marks} == native._PER_PR_SHAPES
    for entry in entries:
        if entry.marks:
            assert [(mark.name, mark.args[0]) for mark in entry.marks] == [("manual", ["a5"])]


def _nonreturning_capture():
    raw = _capture()
    protocol = raw["aicore_protocol"]
    protocol["workers"][0]["finished"] = 0
    execution = protocol["executions"][0]
    for field in ("returned", "visible", "completion_consumed", "trace_published"):
        execution[field] = [0, 0]
    raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] = 3_000_000_200
    return raw


def test_nonreturning_capture_proves_missing_return_and_exit_ack_timeout(native):
    native._check_failed_capture(_nonreturning_capture(), 2, 3, 0, nonreturning=True)


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_entry",
        "return",
        "visible",
        "consumed",
        "trace",
        "fake_ack",
        "extra_unfinished",
        "wrong_partition",
        "short_exit_wait",
        "premature_exit",
        "late_signal_end",
        "consumer_entered",
        "task_done",
        "completed_blocks",
        "resolved_count",
    ),
)
def test_nonreturning_capture_rejects_fabricated_completion_or_unrelated_failure(native, corruption):
    raw = _nonreturning_capture()
    protocol = raw["aicore_protocol"]
    execution = protocol["executions"][0]
    if corruption == "missing_entry":
        execution["entry"] = [0, 0]
    elif corruption in ("return", "visible", "consumed", "trace"):
        field = {
            "return": "returned",
            "visible": "visible",
            "consumed": "completion_consumed",
            "trace": "trace_published",
        }[corruption]
        execution[field] = [2_500_000_100, 2]
    elif corruption == "fake_ack":
        protocol["workers"][0]["finished"] = 2
    elif corruption == "extra_unfinished":
        protocol["workers"][1]["finished"] = 0
    elif corruption == "wrong_partition":
        execution["worker_index"] = 2
        protocol["workers"][0]["finished"] = 2
        protocol["workers"][2]["finished"] = 0
    elif corruption == "short_exit_wait":
        raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] -= 1
    elif corruption == "late_signal_end":
        raw["aicpu_lifecycle_records"][0]["exit_signal_end_cycles"] = 3_100_000_000
    else:
        _corrupt_shutdown(raw, corruption)
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 2, 3, 0, nonreturning=True)


@pytest.mark.parametrize(
    "corruption",
    (
        "no_error",
        "wrong_site",
        "unfinished_worker",
        "missing_worker",
        "duplicate_worker",
        "truncated_topology",
        "worker_error",
        "dependency",
        "wrong_mask",
        "wrong_block",
        "missing_return",
        "return_sequence",
        "short_stall",
        "premature_exit",
        "late_exit",
        "late_signal_end",
        "premature_retirement",
        "wrong_partition",
        "duplicate_partition",
        "consumer_entered",
        "task_done",
        "completed_blocks",
        "duplicate_control",
        "missing_control",
        "resolved_count",
    ),
)
def test_failed_capture_cannot_hide_missing_exit_or_successful_dependency_release(native, corruption):
    raw = _capture()
    protocol = raw["aicore_protocol"]
    execution = protocol["executions"][0]
    if corruption == "no_error":
        protocol["scheduler_error"] = 0
    elif corruption == "wrong_site":
        protocol["first_error"]["site"] = 93
    elif corruption == "unfinished_worker":
        protocol["workers"][0]["finished"] = 0
    elif corruption == "missing_worker":
        protocol["workers"].pop()
    elif corruption == "duplicate_worker":
        protocol["workers"][1] = copy.deepcopy(protocol["workers"][0])
    elif corruption == "truncated_topology":
        protocol["workers"].pop()
        protocol["active_worker_count"] = protocol["worker_count"] = 2
    elif corruption == "worker_error":
        protocol["workers"][0]["error_flags"] = 1
    elif corruption == "dependency":
        protocol["tasks"][1]["dependencies"] = []
    elif corruption == "wrong_mask":
        protocol["tasks"][0]["active_mask"] = 1
    elif corruption == "wrong_block":
        execution["block_idx"] = 1
    elif corruption == "missing_return":
        execution["returned"] = [0, 0]
    elif corruption == "return_sequence":
        execution["returned"][1] = execution["entry"][1]
    elif corruption == "short_stall":
        execution["returned"][0] -= 1
    else:
        _corrupt_shutdown(raw, corruption)
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 2, 3, 0)


def _corrupt_shutdown(raw, corruption):
    protocol = raw["aicore_protocol"]
    execution = protocol["executions"][0]
    if corruption == "premature_exit":
        raw["aicpu_lifecycle_records"][0]["exit_signal_start_cycles"] = 99
    elif corruption == "late_exit":
        for record in raw["aicpu_lifecycle_records"]:
            record["exit_signal_start_cycles"] = execution["returned"][0] + 1
    elif corruption == "late_signal_end":
        raw["aicpu_lifecycle_records"][0]["exit_signal_end_cycles"] = execution["returned"][0] + 1
    elif corruption == "premature_retirement":
        for record in raw["aicpu_lifecycle_records"]:
            record["exit_wait_end_cycles"] = execution["returned"][0] - 1
    elif corruption == "wrong_partition":
        raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] = execution["returned"][0] - 1
        raw["aicpu_lifecycle_records"][1]["exit_wait_end_cycles"] = execution["returned"][0] + 100
    elif corruption == "duplicate_partition":
        raw["aicpu_lifecycle_records"][1]["aicpu_thread_id"] = 0
    elif corruption == "consumer_entered":
        protocol["executions"].append({**execution, "task_ref": 1})
    elif corruption == "task_done":
        protocol["task_controls"][0]["state"] = 2
    elif corruption == "completed_blocks":
        protocol["task_controls"][0]["completed_blocks"] = 2
    elif corruption == "duplicate_control":
        protocol["task_controls"].append(copy.deepcopy(protocol["task_controls"][0]))
    elif corruption == "missing_control":
        protocol["task_controls"].pop()
    else:
        protocol["resolved_task_count"] = 1


@pytest.mark.parametrize(
    "corruption", ("duplicate", "missing_return", "wrong_n", "guard", "consumer", "inactive", "padding")
)
def test_failed_output_checks_return_writes_and_untouched_successor(native, corruption):
    records = torch.zeros((4, 16), dtype=torch.int64)
    for lane in (0, 1):
        records[lane, :2] = 1
        records[lane, 8:14] = torch.tensor([1, 1, lane + 1, 1, 100, 200])
    values = [records.flatten(), torch.zeros(16, dtype=torch.int64)]
    native._check_output(values, 1, 1, 3, failed=True)
    if corruption == "duplicate":
        records[0, 0] = 2
    elif corruption == "missing_return":
        records[0, 1] = 0
    elif corruption == "wrong_n":
        records[0, 9] = 2
    elif corruption == "guard":
        records[3, 0] = 1
    elif corruption == "consumer":
        values[1][0] = 2
    elif corruption == "inactive":
        records[2, 0] = 1
    else:
        records[0, 14] = 1
    with pytest.raises(AssertionError):
        native._check_output(values, 1, 1, 3, failed=True)


@pytest.mark.parametrize("recovery", (False, True))
@pytest.mark.parametrize("native_graph", (False, True))
def test_unexpected_healthy_failure_preserves_error_and_closes_worker(
    native, monkeypatch, tmp_path, recovery, native_graph
):
    class FailingWorker:
        closed = False

        def __init__(self, **kwargs):
            pass

        def register(self, callable_obj):
            return object()

        def init(self):
            pass

        def malloc(self, size):
            return object()

        scheduler_mode_publish_count = 0

        def run(self, *args):
            raise RuntimeError("unexpected healthy-pass timeout")

        def free(self, buffer):
            pytest.fail("failed run still borrows the buffer")

        def close(self):
            FailingWorker.closed = True

    monkeypatch.setattr(native, "Worker", FailingWorker)
    monkeypatch.setattr(native, "_compile", lambda platform, **kwargs: object())
    monkeypatch.setattr(native, "_args", lambda *args: object())
    with pytest.raises(RuntimeError, match="unexpected healthy-pass timeout"):
        native._run_pass("a5sim", 0, tmp_path, 7, 0, 257, 2500, recovery=recovery, native_graph=native_graph)
    assert FailingWorker.closed


@pytest.mark.parametrize("platform", ("a5sim", "a5"))
@pytest.mark.parametrize("stall_ms", (2500, -1))
def test_failure_budget_keeps_simulation_headroom_and_fault_injection(
    native, monkeypatch, tmp_path, platform, stall_ms
):
    calls = []

    def run_pass(platform, device_id, artifacts, mask, lane, blocks, stall_ms, *args, **kwargs):
        calls.append((int(os.environ["SIMPLER_SCHEDULER_TIMEOUT_MS"]), stall_ms))
        return []

    monkeypatch.setattr(native, "_ROOT", tmp_path)
    monkeypatch.setattr(native, "_run_pass", run_pass)
    monkeypatch.setattr(native, "_recovery_pass", run_pass)
    native._exercise_failure(platform, [0], 7, 0, 257, monkeypatch, stall_ms)
    assert len(calls) == 2 and calls[0] == calls[1]
    timeout_ms, duration = calls[0]
    if platform == "a5sim":
        assert timeout_ms >= 10000
    else:
        assert timeout_ms == 2000
    if stall_ms < 0:
        assert duration == -1
    else:
        assert duration >= timeout_ms * 5 // 4


@pytest.mark.parametrize("platform", ("a5sim", "a5"))
def test_native_graph_failure_budget_covers_failure_and_recovery(native, monkeypatch, tmp_path, platform):
    calls = []

    def run_pass(platform, device_id, artifacts, mask, lane, blocks, stall_ms, *, recovery, native_graph):
        assert native_graph
        calls.append((int(os.environ["SIMPLER_SCHEDULER_TIMEOUT_MS"]), stall_ms, recovery))
        return []

    def recover_in_subprocess(command, **kwargs):
        run_pass(**json.loads(command[-1]))
        return SimpleNamespace(returncode=0, stdout=native._RECOVERY_MARKER + "[]", stderr="")

    monkeypatch.setattr(native, "_ROOT", tmp_path)
    monkeypatch.setattr(native, "_run_pass", run_pass)
    monkeypatch.setattr(native.subprocess, "run", recover_in_subprocess)
    native.test_native_graph_failure_does_not_release_exit(platform, [0], 7, 2, monkeypatch)
    assert len(calls) == 2 and calls[0][:2] == calls[1][:2]
    assert [call[2] for call in calls] == [False, True]
    timeout_ms, duration, _ = calls[0]
    assert timeout_ms >= 10000 if platform == "a5sim" else timeout_ms == 2000
    assert duration >= timeout_ms * 5 // 4


def test_failed_capture_checks_the_actual_injected_stall(native):
    raw = _capture()
    execution = raw["aicore_protocol"]["executions"][0]
    execution["returned"][0] = 12_500_000_100
    execution["visible"][0] = 12_500_000_101
    raw["aicpu_lifecycle_records"][0]["exit_wait_end_cycles"] = 12_500_000_200
    native._check_failed_capture(raw, 2, 3, 0, stall_ms=12500)
    execution["returned"][0] -= 1
    with pytest.raises(AssertionError):
        native._check_failed_capture(raw, 2, 3, 0, stall_ms=12500)


@pytest.mark.parametrize("platform,stall_ms", [("a5sim", 2500), ("a5", 2500), ("a5sim", -1), ("a5", -1)])
def test_returned_sim_fault_releases_only_physically_finished_buffers(
    native, monkeypatch, tmp_path, platform, stall_ms
):
    _check_fault_buffer_policy(native, monkeypatch, tmp_path, platform, stall_ms)


def test_partial_buffer_release_does_not_retry_a_confirmed_free(native, monkeypatch, tmp_path):
    _check_fault_buffer_policy(native, monkeypatch, tmp_path, "a5sim", 2500, fail_second_free=True)


@pytest.mark.parametrize(
    "platform,code,damage,expected_error",
    [
        ("a5", 507018, None, None),
        ("a5sim", -100, None, None),
        ("a5", -100, None, AssertionError),
        ("a5", 507011, None, AssertionError),
        ("a5", 507015, None, AssertionError),
        ("a5", 0, None, pytest.fail.Exception),
        ("a5sim", 507018, None, AssertionError),
        ("a5", 507018, "missing", FileNotFoundError),
        ("a5", 507018, "epoch", AssertionError),
        ("a5", 507018, "scheduler", AssertionError),
        ("a5", 507018, "site", AssertionError),
        ("a5", 507018, "returned", AssertionError),
        ("a5", 507018, "output", AssertionError),
    ],
)
def test_bounded_fault_requires_strong_error_and_owned_records(
    native, monkeypatch, tmp_path, platform, code, damage, expected_error
):
    # The real scene body and validators run on controlled Worker/file leaves.
    # These records are CPU fixture data, not device fault/stop evidence.
    if expected_error is None:
        _check_fault_buffer_policy(
            native, monkeypatch, tmp_path, platform, 2500, reported_code=code, validate_records=True, damage=damage
        )
    else:
        with pytest.raises(expected_error):
            _check_fault_buffer_policy(
                native,
                monkeypatch,
                tmp_path,
                platform,
                2500,
                reported_code=code,
                validate_records=True,
                damage=damage,
            )


def _check_fault_buffer_policy(
    native,
    monkeypatch,
    tmp_path,
    platform,
    stall_ms,
    *,
    fail_second_free=False,
    reported_code=None,
    validate_records=False,
    damage=None,
    owned_wait_case=None,
):
    events = []
    freed = []
    free_calls = []
    owners = []
    allocations = []
    close_snapshots = []
    original_retain_proof = native._retain_completion_proof

    def retain_proof(artifacts, stage, buffers, **kwargs):
        if stage == "before-close":
            close_snapshots.append(tuple(buffers))
        return original_retain_proof(artifacts, stage, buffers, **kwargs)

    monkeypatch.setattr(native, "_retain_completion_proof", retain_proof)

    class ControlledWorker:
        scheduler_mode_publish_count = 0
        scheduler_runtime_mode = 1
        poisoned = False

        def __init__(self, **kwargs):
            owners.append(self)

        def register(self, callable_obj):
            return object()

        def init(self):
            pass

        def malloc(self, size):
            if owned_wait_case is not None:
                buffer = SimpleNamespace(identity=("controlled-owner", size), base=size * 4096, nbytes=size)
            else:
                buffer = object()
            allocations.append(buffer)
            return buffer

        def run(self, handle, duration, config):
            if self.poisoned:
                events.append("poison-refusal")
                raise RuntimeError("poisoned")
            self.scheduler_mode_publish_count += 1
            self.poisoned = duration != 0
            if self.poisoned:
                events.append("run-error")
                code = reported_code
                if code is None:
                    code = 507018 if platform == "a5" and stall_ms > 0 else -100
                if code != 0:
                    if owned_wait_case is not None:
                        raise RuntimeError(owned_wait_case["expected_error"])
                    raise RuntimeError(f"finalize_native_run failed with code {code}")
                return
            events.append("healthy")

        def copy_from(self, value, buffer):
            if validate_records:
                if value.numel() == 16:
                    value[0] = 0 if self.poisoned else 3
                else:
                    records = value.view(-1, 16)
                    for lane in range(3):
                        records[lane, :2] = 1
                        records[lane, 8:14] = torch.tensor([1, 1, lane + 1, 1, 100, 200])
                    if self.poisoned and damage == "output":
                        records[-1, 0] = 1
            elif self.poisoned:
                value[0] = 1

        def free(self, buffer):
            assert "failed-capture" in events and "poison-refusal" in events
            events.append("free")
            free_calls.append(buffer)
            if (owned_wait_case is None and (platform == "a5" or stall_ms < 0)) or (
                owned_wait_case is not None and not owned_wait_case.get("allow_free", True)
            ):
                raise RuntimeError("still in use by a submitted run")
            if fail_second_free and len(free_calls) == 2:
                raise RuntimeError("controlled free refused")
            freed.append(buffer)

        def close(self):
            events.append("close")

    def capture(prefix):
        owner = owners[0]
        if validate_records:
            if owner.poisoned and damage == "missing":
                raise FileNotFoundError("controlled missing owned capture")
            raw = _large_ticket_capture(7, 0, 1)
            protocol = raw["aicore_protocol"]
            protocol.update(schema="hbg-native-protocol-v5", run_epoch=owner.scheduler_mode_publish_count)
            for control in protocol["task_controls"]:
                control["admission_claim_lock"] = 0
                control["claim_tickets"] = 0
            if not owner.poisoned or damage == "scheduler":
                protocol["scheduler_error"] = 0
            if owner.poisoned:
                if damage == "epoch":
                    protocol["run_epoch"] -= 1
                elif damage == "site":
                    protocol["first_error"]["site"] = 93
                elif damage == "returned":
                    protocol["executions"][0]["returned"] = [0, 0]
            return prefix / "capture.bin", raw
        return prefix / "capture.bin", {
            "aicore_protocol": {
                "schema": "hbg-native-protocol-v5",
                "run_epoch": owner.scheduler_mode_publish_count,
                "scheduler_error": 100 if owner.poisoned else 0,
                "first_error": {"site": 95 if owner.poisoned else 0},
            }
        }

    if owned_wait_case is not None:
        logs = tmp_path / "owned-host-logs"
        logs.mkdir()
        (logs / f"host.{owned_wait_case['pid']}.log").write_text(owned_wait_case["rows"])
        controlled_os = SimpleNamespace(**vars(os))
        controlled_os.getpid = lambda: owned_wait_case["pid"]
        monkeypatch.setattr(native, "os", controlled_os)
        monkeypatch.setattr(native._task_interface, "_native_host_log_directory", lambda: str(logs))
        monkeypatch.setattr(native._task_interface, "_flush_host_log", lambda _: owned_wait_case.get("flush", True))
        monkeypatch.setattr(
            native._task_interface, "_host_log_pending_records", lambda: owned_wait_case.get("pending", 0)
        )
        monkeypatch.setattr(
            native._task_interface, "_host_log_dropped_records", lambda: owned_wait_case.get("dropped", 0)
        )

    original_capture_check = native._check_failed_capture
    original_output_check = native._check_output

    def failed_capture(*args, **kwargs):
        if validate_records:
            original_capture_check(*args, **kwargs)
        events.append("failed-capture")

    def check_output(*args, **kwargs):
        if validate_records:
            original_output_check(*args, **kwargs)
        events.append("output-checked")

    monkeypatch.setattr(native, "Worker", ControlledWorker)
    monkeypatch.setattr(native, "_compile", lambda platform, *, native_graph=False: object())
    monkeypatch.setattr(native, "_args", lambda *args: args[-1])
    monkeypatch.setattr(native, "_capture", capture)
    monkeypatch.setattr(native, "_root_block_num", lambda protocol: 1)
    monkeypatch.setattr(native, "_check_output", check_output)
    monkeypatch.setattr(native, "_check_failed_capture", failed_capture)
    monkeypatch.setattr(native, "read_perf_data", lambda path: {"protocol_audit": {"protocol_verified": True}})
    if fail_second_free:
        with pytest.raises(RuntimeError, match="controlled free refused"):
            native._run_pass(
                platform,
                owned_wait_case["device"] if owned_wait_case else 0,
                tmp_path,
                7,
                0,
                1,
                stall_ms,
                recovery=False,
            )
        assert len(freed) == 1
    elif (
        reported_code != 0
        and not (platform == "a5sim" and stall_ms > 0)
        and (owned_wait_case is None or owned_wait_case.get("unavailable", False))
    ):
        with pytest.raises(pytest.fail.Exception, match="owned RTS last-use evidence unavailable"):
            native._run_pass(
                platform,
                owned_wait_case["device"] if owned_wait_case else 0,
                tmp_path,
                7,
                0,
                1,
                stall_ms,
                recovery=False,
            )
        assert free_calls == freed == []
        assert close_snapshots == [tuple(allocations)]
        assert len(allocations) == 2
        assert events[-1] == "close"
        assert events.index("run-error") < events.index("failed-capture") < events.index("poison-refusal")
        return
    else:
        rows = native._run_pass(
            platform, owned_wait_case["device"] if owned_wait_case else 0, tmp_path, 7, 0, 1, stall_ms, recovery=False
        )
        released = (platform == "a5sim" and stall_ms > 0) or (
            owned_wait_case is not None and owned_wait_case.get("allow_free", True)
        )
        assert rows[-1]["borrowed_buffers_kept"] is not released
        assert len(freed) == (2 if released else 0)
    assert len(free_calls) == 2 and free_calls[0] is not free_calls[1]
    assert events[-1] == "close"
    assert (
        events.index("run-error")
        < events.index("failed-capture")
        < events.index("poison-refusal")
        < events.index("free")
    )


# Actual CI console proof rows; file/flush leaves and the expected error argument are controlled.
_OWNED_WAIT_CASES = [
    {
        "pid": 3352729,
        "device": 3,
        "stall_ms": 2500,
        "rows": "[mono_ns=1730405513072910][T0x7c3efc7ccb80][TIMING] record_host_orch_callable: "
        "[device_runner_base.cpp:2622] Callable registration: device=3 runner=0xffd9f40 cid=0 "
        "chip_hash=0xe872f4cfc3275442 aicore_image_hash=0xf2a664cc7ea838a0 host_dlopen_count=1\n"
        "[mono_ns=1730405606649755][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=1 hid=0 depth=0 name=chip.run.proof.record ts=1730405606646650 dur=0 "
        "f=0xcbbbe40 e=0xcea1e28 s=0x1399a8e8 role=0 ep=1 gen=1 disp=0 slot=0\n"
        "[mono_ns=1730405606669845][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=1 hid=0 depth=0 name=chip.run.proof.record ts=1730405606669144 dur=0 "
        "f=0xcbbbe40 e=0xce8e1a8 s=0x13897908 role=1 ep=1 gen=1 disp=0 slot=0\n"
        "[mono_ns=1730405607216846][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730405607215794 dur=0 "
        "e=0xce8e1a8 rc=0 status_valid=0 status=0 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730405607219890][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730405607219259 dur=0 "
        "f=0xcbbbe40 e=0xce8e1a8 role=1 ep=1 gen=1 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730405607222043][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730405607221583 dur=0 "
        "e=0xcea1e28 rc=0 status_valid=0 status=0 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730405607224908][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730405607224467 dur=0 "
        "f=0xcbbbe40 e=0xcea1e28 role=0 ep=1 gen=1 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730405609679581][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=1 hid=0 depth=1 name=chip.run.claim_release ts=1730405609678519 dur=501 \n"
        "[mono_ns=1730405609688735][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.borrow_release ts=1730405609685560 dur=0 "
        "ctx=0xffd9f40 borrow=1 ep=1 gen=1 disp=0 slot=0 run=0 proof=1 keep=0\n"
        "[mono_ns=1730405609704368][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.observe_begin ts=1730405609703587 dur=0 "
        "ep=1 gen=1 disp=0 slot=0 aicpu=0x13897908 aicore=0x1399a8e8\n"
        "[mono_ns=1730405609712070][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.stream_wait ts=1730405609711439 dur=0 "
        "role=AICPU s=0x13897908 rc=0 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730405609716116][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.stream_wait ts=1730405609715615 dur=0 "
        "role=AICore s=0x1399a8e8 rc=0 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730405609718019][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.observe_return ts=1730405609717518 dur=0 "
        "ep=1 gen=1 disp=0 slot=0 rc=0 cached=1 physical=1 polled=0 poll=0 retained=1\n"
        "[mono_ns=1730405609731879][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=1 hid=0 depth=0 name=chip.run ts=1730405515303647 dur=94427551 run_id=0 "
        "dispatch_id=0 slot_id=0 generation=1 run_epoch=1\n"
        "[mono_ns=1730405626233771][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=2 hid=0 depth=0 name=chip.run.proof.record ts=1730405626232739 dur=0 "
        "f=0xcbbbe40 e=0xcea1e28 s=0x1399a8e8 role=0 ep=2 gen=2 disp=0 slot=0\n"
        "[mono_ns=1730405626249515][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=2 hid=0 depth=0 name=chip.run.proof.record ts=1730405626249024 dur=0 "
        "f=0xcbbbe40 e=0xce8e1a8 s=0x13897908 role=1 ep=2 gen=2 disp=0 slot=0\n"
        "[mono_ns=1730408134689467][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730408134686643 dur=0 "
        "e=0xce8e1a8 rc=0 status_valid=0 status=0 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730408134696277][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730408134695356 dur=0 "
        "f=0xcbbbe40 e=0xce8e1a8 role=1 ep=2 gen=2 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730408134732231][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730408134731680 dur=0 "
        "e=0xcea1e28 rc=0 status_valid=0 status=0 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730408134734094][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730408134733623 dur=0 "
        "f=0xcbbbe40 e=0xcea1e28 role=0 ep=2 gen=2 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730408137302107][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=2 hid=0 depth=1 name=chip.run.claim_release ts=1730408137300925 dur=571 \n"
        "[mono_ns=1730408137309929][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.borrow_release ts=1730408137307375 dur=0 "
        "ctx=0xffd9f40 borrow=1 ep=2 gen=2 disp=0 slot=0 run=0 proof=1 keep=0\n"
        "[mono_ns=1730408137324671][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.observe_begin ts=1730408137323910 dur=0 "
        "ep=2 gen=2 disp=0 slot=0 aicpu=0x13897908 aicore=0x1399a8e8\n"
        "[mono_ns=1730408137493594][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.stream_wait ts=1730408137492623 dur=0 "
        "role=AICPU s=0x13897908 rc=507018 timeout_ms=6000 fresh=1\n"
        "[mono_ns=1730408137620765][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=0 hid=0 depth=0 name=chip.run.proof.observe_return ts=1730408137620114 dur=0 "
        "ep=2 gen=2 disp=0 slot=0 rc=507018 cached=1 physical=1 polled=0 poll=0 retained=1\n"
        "[mono_ns=1730408137667656][T0x7c3efc7ccb80][TIMING] emit_host_span: [STRACE] v=1 pid=3352729 "
        "tid=3352729 inv=2 hid=0 depth=0 name=chip.run ts=1730405625725568 dur=2511941537 run_id=0 "
        "dispatch_id=0 slot_id=0 generation=2 run_epoch=2\n",
        "expected_error": "finalize_native_run failed with code 507018 (run_id=0 slot=0 generation=2 "
        "dispatch_id=0 run_epoch=2)",
    },
    {
        "pid": 3358189,
        "device": 1,
        "stall_ms": -1,
        "rows": "[mono_ns=1730444803091589][T0x77d7a085ab80][TIMING] record_host_orch_callable: "
        "[device_runner_base.cpp:2622] Callable registration: device=1 runner=0x1fbee240 cid=0 "
        "chip_hash=0xe872f4cfc3275442 aicore_image_hash=0xf2a664cc7ea838a0 host_dlopen_count=1\n"
        "[mono_ns=1730444905612605][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=1 hid=0 depth=0 name=chip.run.proof.record ts=1730444905609200 dur=0 "
        "f=0x1c67e0f0 e=0x1c92f138 s=0x235af758 role=0 ep=1 gen=1 disp=0 slot=0\n"
        "[mono_ns=1730444905637212][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=1 hid=0 depth=0 name=chip.run.proof.record ts=1730444905636561 dur=0 "
        "f=0x1c67e0f0 e=0x1c91a138 s=0x234ac778 role=1 ep=1 gen=1 disp=0 slot=0\n"
        "[mono_ns=1730444912023284][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730444912017065 dur=0 "
        "e=0x1c91a138 rc=0 status_valid=0 status=0 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730444912038707][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730444912037976 dur=0 "
        "f=0x1c67e0f0 e=0x1c91a138 role=1 ep=1 gen=1 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730444912045888][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730444912045388 dur=0 "
        "e=0x1c92f138 rc=0 status_valid=0 status=0 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730444912049323][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730444912048833 dur=0 "
        "f=0x1c67e0f0 e=0x1c92f138 role=0 ep=1 gen=1 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730444915522545][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=1 hid=0 depth=1 name=chip.run.claim_release ts=1730444915521333 dur=641 \n"
        "[mono_ns=1730444915537538][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.borrow_release ts=1730444915534553 dur=0 "
        "ctx=0x1fbee240 borrow=1 ep=1 gen=1 disp=0 slot=0 run=0 proof=1 keep=0\n"
        "[mono_ns=1730444915572811][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.observe_begin ts=1730444915571909 dur=0 "
        "ep=1 gen=1 disp=0 slot=0 aicpu=0x234ac778 aicore=0x235af758\n"
        "[mono_ns=1730444915581824][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.stream_wait ts=1730444915581213 dur=0 "
        "role=AICPU s=0x234ac778 rc=0 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730444915586221][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.stream_wait ts=1730444915585780 dur=0 "
        "role=AICore s=0x235af758 rc=0 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730444915588444][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.observe_return ts=1730444915587943 dur=0 "
        "ep=1 gen=1 disp=0 slot=0 rc=0 cached=1 physical=1 polled=0 poll=0 retained=1\n"
        "[mono_ns=1730444915607573][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=1 hid=0 depth=0 name=chip.run ts=1730444804476037 dur=111130775 run_id=0 "
        "dispatch_id=0 slot_id=0 generation=1 run_epoch=1\n"
        "[mono_ns=1730444926986287][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=2 hid=0 depth=0 name=chip.run.proof.record ts=1730444926985015 dur=0 "
        "f=0x1c67e0f0 e=0x1c92f138 s=0x235af758 role=0 ep=2 gen=2 disp=0 slot=0\n"
        "[mono_ns=1730444927014679][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=2 hid=0 depth=0 name=chip.run.proof.record ts=1730444927014118 dur=0 "
        "f=0x1c67e0f0 e=0x1c91a138 s=0x234ac778 role=1 ep=2 gen=2 disp=0 slot=0\n"
        "[mono_ns=1730447935142016][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730447935134244 dur=0 "
        "e=0x1c91a138 rc=0 status_valid=0 status=0 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730447935153984][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730447935153033 dur=0 "
        "f=0x1c67e0f0 e=0x1c91a138 role=1 ep=2 gen=2 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730449225709528][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.event_wait ts=1730449225704801 dur=0 "
        "e=0x1c92f138 rc=0 status_valid=0 status=0 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730449225722577][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.consume ts=1730449225721776 dur=0 "
        "f=0x1c67e0f0 e=0x1c92f138 role=0 ep=2 gen=2 disp=0 slot=0 op=wait rc=0 c=1 cache=0 owner=1\n"
        "[mono_ns=1730449229187036][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=2 hid=0 depth=1 name=chip.run.claim_release ts=1730449229185343 dur=982 \n"
        "[mono_ns=1730449229201057][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.borrow_release ts=1730449229197672 dur=0 "
        "ctx=0x1fbee240 borrow=1 ep=2 gen=2 disp=0 slot=0 run=0 proof=1 keep=0\n"
        "[mono_ns=1730449229235609][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.observe_begin ts=1730449229234797 dur=0 "
        "ep=2 gen=2 disp=0 slot=0 aicpu=0x234ac778 aicore=0x235af758\n"
        "[mono_ns=1730449229406655][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.stream_wait ts=1730449229405534 dur=0 "
        "role=AICPU s=0x234ac778 rc=507018 timeout_ms=4000 fresh=1\n"
        "[mono_ns=1730449229561027][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=0 hid=0 depth=0 name=chip.run.proof.observe_return ts=1730449229560346 dur=0 "
        "ep=2 gen=2 disp=0 slot=0 rc=507018 cached=1 physical=1 polled=0 poll=0 retained=1\n"
        "[mono_ns=1730449229697151][T0x77d7a085ab80][TIMING] emit_host_span: [STRACE] v=1 pid=3358189 "
        "tid=3358189 inv=2 hid=0 depth=0 name=chip.run ts=1730444926051113 dur=4303645347 run_id=0 "
        "dispatch_id=0 slot_id=0 generation=2 run_epoch=2\n",
        "expected_error": "finalize_native_run failed with code 507018 (run_id=0 slot=0 generation=2 "
        "dispatch_id=0 run_epoch=2)",
    },
]


@pytest.mark.parametrize("case", _OWNED_WAIT_CASES, ids=["bounded", "nonreturning"])
def test_owned_rts_wait_proof_releases_fault_buffers(native, monkeypatch, tmp_path, case):
    _check_fault_buffer_policy(native, monkeypatch, tmp_path, "a5", case["stall_ms"], owned_wait_case=case)


def _damaged_wait_case(damage):
    case = copy.deepcopy(_OWNED_WAIT_CASES[0])
    case["allow_free"] = False
    case["unavailable"] = True
    rows = case["rows"].splitlines()
    index = next(i for i, row in enumerate(rows) if "name=chip.run.proof.consume " in row and "ep=2 " in row)
    if damage == "missing":
        rows.pop(index)
    elif damage == "pending":
        rows[index - 1] = rows[index - 1].replace("rc=0 ", "rc=507018 ")
    elif damage == "foreign-pid":
        rows[index] = rows[index].replace(f"pid={case['pid']} ", f"pid={case['pid'] + 1} ")
    elif damage == "foreign-generation":
        rows[index] = rows[index].replace("gen=2 ", "gen=99 ")
    elif damage == "foreign-fence":
        fields = dict(word.split("=", 1) for word in rows[index].split() if "=" in word)
        rows[index] = rows[index].replace("f=" + fields["f"], "f=0x1")
    elif damage == "foreign-context":
        i = next(i for i, row in enumerate(rows) if "name=chip.run.borrow_release " in row and "ep=2 " in row)
        fields = dict(word.split("=", 1) for word in rows[i].split() if "=" in word)
        rows[i] = rows[i].replace("ctx=" + fields["ctx"], "ctx=0x1")
    elif damage == "owner-query":
        rows[index] = rows[index].replace("owner=1", "owner=0")
    elif damage == "cached":
        rows[index] = rows[index].replace("cache=0 ", "cache=3 ")
    elif damage == "not-fresh":
        rows[index - 1] = rows[index - 1].replace("fresh=1", "fresh=0")
    elif damage == "duplicate":
        rows.insert(index + 1, rows[index])
    elif damage == "registration":
        rows = [row for row in rows if "Callable registration:" not in row]
    elif damage == "invalid":
        rows[index] = rows[index].replace("name=chip.run.proof.consume", "name=chip.run.proof.invalid")
    elif damage == "rebound":
        record = next(row for row in rows if "name=chip.run.proof.record " in row and "ep=2 " in row)
        i = next(i for i, row in enumerate(rows) if "name=chip.run.borrow_release " in row and "ep=2 " in row)
        rows.insert(i, record.replace("gen=2 ", "gen=99 "))
    elif damage in ("dropped", "pending-records", "flush"):
        case[{"dropped": "dropped", "pending-records": "pending", "flush": "flush"}[damage]] = (
            False if damage == "flush" else 1
        )
    else:
        raise AssertionError(damage)
    case["rows"] = "\n".join(rows) + "\n"
    return case


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "pending",
        "foreign-pid",
        "foreign-generation",
        "foreign-fence",
        "foreign-context",
        "owner-query",
        "cached",
        "not-fresh",
        "duplicate",
        "registration",
        "invalid",
        "rebound",
        "dropped",
        "pending-records",
        "flush",
    ],
)
def test_unavailable_rts_wait_evidence_fails_qualification_before_free(native, monkeypatch, tmp_path, damage):
    case = _damaged_wait_case(damage)
    _check_fault_buffer_policy(native, monkeypatch, tmp_path, "a5", case["stall_ms"], owned_wait_case=case)


@pytest.mark.parametrize("allow_free", [False, True])
def test_unavailable_logs_do_not_determine_runtime_free_policy(native, monkeypatch, tmp_path, allow_free):
    case = _damaged_wait_case("missing")
    case["allow_free"] = allow_free
    _check_fault_buffer_policy(native, monkeypatch, tmp_path, "a5", case["stall_ms"], owned_wait_case=case)


def test_qualified_proof_still_requires_free_success(native, monkeypatch, tmp_path):
    case = copy.deepcopy(_OWNED_WAIT_CASES[0])
    case["allow_free"] = False
    with pytest.raises(RuntimeError, match="still in use"):
        _check_fault_buffer_policy(native, monkeypatch, tmp_path, "a5", case["stall_ms"], owned_wait_case=case)


def test_owned_proof_partial_free_does_not_retry_a_confirmed_free(native, monkeypatch, tmp_path):
    case = _OWNED_WAIT_CASES[0]
    _check_fault_buffer_policy(
        native, monkeypatch, tmp_path, "a5", case["stall_ms"], owned_wait_case=case, fail_second_free=True
    )
