# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real resident timeouts, retained caller buffers and failed-lane containment."""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
from simpler import task_interface as _task_interface
from simpler.task_interface import ArgDirection, CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec
from simpler_setup.tools.hbg.native_protocol import has_native_graph, protocol_version
from simpler_setup.tools.swimlane_converter import read_perf_data

_ROOT = Path(__file__).resolve().parents[5]
_KERNELS = _ROOT / "tests/fdwic/pa_hbg/spmd/kernels"
_STALL_MS = 2500
_TIMEOUT_MS = 2000
_SIM_TIMEOUT_MS = 10000
_SHAPES = [(mask, lane) for mask in (1, 2, 4, 3, 5, 6, 7) for lane in range(3) if mask & (1 << lane)]
_TICKET_SHAPES = ((1, 0), (2, 1), (4, 2), (6, 1), (7, 0), (7, 1), (7, 2))
# The exhaustive mask/lane sweep is about silicon behaviour, and each case here
# holds a device for its own process because a failed run poisons it. A
# simulator tiers therefore cover two shapes: the lane the protocol reports in
# slot 1, which the Per-PR capture check uses, and all three lanes together in
# Daily.
_SIM_SHAPES = frozenset({(4, 2), (7, 0)})
# On silicon, Per-PR keeps one shape per failed-lane position for every timeout
# mode. The remaining mask/lane combinations run in Daily.
_PER_PR_SHAPES = frozenset({(1, 0), (4, 2), (7, 1)})


def _onboard_shape_params(shapes):
    """Shapes for a case the simulator cannot carry: only the Per-PR tier varies."""
    return [
        pytest.param(mask, lane, marks=[] if (mask, lane) in _PER_PR_SHAPES else [pytest.mark.manual(["a5"])])
        for mask, lane in shapes
    ]


def _shape_params(shapes):
    params = []
    for mask, lane in shapes:
        marks = [] if (mask, lane) in _SIM_SHAPES else [pytest.mark.platforms(["a5"])]
        manual_platforms = []
        if (mask, lane) == (7, 0):
            manual_platforms.append("a5sim")
        if (mask, lane) not in _PER_PR_SHAPES:
            manual_platforms.append("a5")
        if manual_platforms:
            marks.append(pytest.mark.manual(manual_platforms))
        params.append(pytest.param(mask, lane, marks=marks))
    return params


pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.device_count(1),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.scheduler_mode("resident"),
    pytest.mark.timeout(120),
]


def _compile(platform, *, native_graph=False):
    compiler = KernelCompiler(platform=platform)
    variant = "sim" if platform.endswith("sim") else "onboard"
    includes = [str(compiler.platform_dir / variant / "aicore")]
    spec = {
        "orchestration": {
            "source": str(
                _KERNELS
                / ("runtime_graph_failure_orchestration.cpp" if native_graph else "runtime_failure_orchestration.cpp")
            ),
            "function_name": "aicpu_orchestration_entry",
            "signature": [ArgDirection.INOUT, ArgDirection.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"runtime_stall_{lane}",
                "source": str(_KERNELS / f"runtime_failure_{lane}.cpp"),
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [ArgDirection.INOUT],
                "extra_include_dirs": includes,
            }
            for lane in range(3)
        ]
        + [
            {
                "func_id": 3,
                "name": "verify_fanin",
                "source": str(_KERNELS / "verify_fanin.cpp"),
                "core_type": "aiv",
                "signature": [ArgDirection.IN, ArgDirection.INOUT],
                "extra_include_dirs": includes,
            }
        ],
    }
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (__name__, platform, native_graph, _pto_isa_compile_cache_token())
    )


def _args(worker, buffers, values, blocks, mask, lane, stall_ms):
    args = TaskArgs()
    for buffer, value in zip(buffers, values, strict=True):
        value.zero_()
        worker.copy_to(buffer, value)
        args.add_tensor(buffer.tensor(shapes=tuple(value.shape), dtype=DataType.INT64), TensorArgType.INOUT)
    for scalar in (blocks, mask, lane, stall_ms):
        args.add_scalar(scalar)
    return args


def _capture(prefix):
    paths = list(prefix.rglob("chip_swimlane_records.json"))
    assert len(paths) == 1, paths
    return paths[0], json.loads(paths[0].read_text())


def _check_output(values, capacity, blocks, mask, *, failed):
    records = values[0].view(capacity * 3 + 1, 16)
    expected = torch.zeros_like(records)
    for block in range(blocks):
        for lane in range(3):
            row = records[block * 3 + lane]
            if not mask & (1 << lane) or (failed and row[0] == 0):
                continue
            expected[block * 3 + lane, :2] = 1
            expected[block * 3 + lane, 8:12] = torch.tensor([block + 1, blocks, lane + 1, 1])
            assert row[12] > 0 and row[13] >= row[12]
            expected[block * 3 + lane, 12:14] = row[12:14]
    torch.testing.assert_close(records, expected, rtol=0, atol=0)
    summary = torch.zeros_like(values[1])
    if not failed:
        summary[0] = blocks * mask.bit_count()
    torch.testing.assert_close(values[1], summary, rtol=0, atol=0)


def _check_failed_capture(raw, blocks, mask, lane, *, nonreturning=False, stall_ms=_STALL_MS):
    protocol = raw["aicore_protocol"]
    assert protocol["producer"] == "aicore"
    assert protocol["scheduler_error"] == 10
    assert protocol["first_error"]["site"] in (95, 102)
    worker_count = protocol["active_worker_count"]
    assert worker_count > 0 and protocol["worker_count"] == worker_count
    assert worker_count == raw["metadata"]["num_cores"]
    assert len(protocol["workers"]) == worker_count
    assert {worker["worker_index"] for worker in protocol["workers"]} == set(range(worker_count))
    assert all(worker["error_flags"] == 0 for worker in protocol["workers"])
    roots = [task for task in protocol["tasks"] if task["active_mask"] and 3 not in task["kernel_ids"]]
    consumers = [task for task in protocol["tasks"] if 3 in task["kernel_ids"]]
    assert len(roots) == len(consumers) == 1
    root, consumer = roots[0], consumers[0]
    assert root["task_ref"] in consumer["dependencies"]
    assert root["block_num"] == blocks and root["active_mask"] == (2 if mask == 4 else mask)
    expected_slot = 1 if mask == 4 else lane
    stalled = [
        entry
        for entry in protocol["executions"]
        if entry["task_ref"] == root["task_ref"] and entry["block_idx"] == 0 and entry["subtask_slot"] == expected_slot
    ]
    assert len(stalled) == 1
    execution = stalled[0]
    frequency = raw["metadata"]["clock_freq_hz"]
    if nonreturning:
        assert all(value > 0 for value in execution["entry"])
        for field in ("returned", "visible", "completion_consumed", "trace_published"):
            assert execution[field] == [0, 0]
        assert all(
            worker["finished"] == (0 if worker["worker_index"] == execution["worker_index"] else 2)
            for worker in protocol["workers"]
        )
    else:
        assert all(worker["finished"] == 2 for worker in protocol["workers"])
        assert 0 < execution["entry"][1] < execution["returned"][1] < execution["visible"][1]
        assert execution["entry"][0] <= execution["returned"][0] <= execution["visible"][0]
        assert execution["returned"][0] - execution["entry"][0] >= stall_ms * frequency // 1000
    lifecycle = raw["aicpu_lifecycle_records"]
    assert len(lifecycle) == 2
    assert {record["aicpu_thread_id"] for record in lifecycle} == {0, 1}
    owners = [
        record
        for record in lifecycle
        if record["aicpu_thread_id"] * worker_count // 2
        <= execution["worker_index"]
        < (record["aicpu_thread_id"] + 1) * worker_count // 2
    ]
    assert len(owners) == 1
    owner = owners[0]
    assert execution["entry"][0] < owner["exit_signal_start_cycles"] <= owner["exit_signal_end_cycles"]
    assert owner["exit_signal_end_cycles"] <= owner["exit_wait_start_cycles"]
    if nonreturning:
        # The onboard A5 EXIT ACK budget is one second. No ACK/return may be
        # invented when STARS, rather than the kernel, ends the operation.
        assert owner["exit_wait_end_cycles"] - owner["exit_wait_start_cycles"] >= frequency
    else:
        assert owner["exit_signal_end_cycles"] < execution["returned"][0]
        assert execution["visible"][0] <= owner["exit_wait_end_cycles"]
    assert all(record["exit_wait_end_cycles"] >= record["exit_wait_start_cycles"] > 0 for record in lifecycle)
    assert not any(entry["task_ref"] == consumer["task_ref"] for entry in protocol["executions"])
    control_rows = protocol["task_controls"]
    controls = {control["task_ref"]: control for control in control_rows}
    assert len(controls) == len(control_rows) == len(protocol["tasks"])
    assert set(controls) == {task["task_ref"] for task in protocol["tasks"]}
    total_subtasks = blocks * root["active_mask"].bit_count()
    assert controls[root["task_ref"]]["completed_subtasks"] < total_subtasks
    assert controls[consumer["task_ref"]]["completed_subtasks"] == 0
    positions = protocol["cluster_count"] * (2 if mask in (2, 4) else 1)
    assert protocol_version(protocol)
    assert type(protocol.get("close_seal_stride")) is int and protocol["close_seal_stride"] == 64
    for task in protocol["tasks"]:
        control = controls[task["task_ref"]]
        seal = control.get("claim_seal")
        assert type(seal) is int and seal >= 0
        cooperative = bool(task.get("flags", 0) & 16)
        assert seal == 0 or (cooperative and seal == control["next_block"] == task["block_num"])
    minimum_blocks = 2
    sync = bool(root.get("flags", 0) & 16)
    uses_tickets = sync and minimum_blocks <= blocks < positions
    tickets = controls[root["task_ref"]]["claim_tickets"]
    assert type(tickets) is int
    assert 0 <= controls[root["task_ref"]]["next_block"] <= blocks
    if uses_tickets:
        assert 1 <= tickets <= positions
        assert controls[root["task_ref"]]["next_block"] <= min(blocks, tickets)
    else:
        assert tickets == 0
    assert controls[consumer["task_ref"]]["claim_tickets"] == 0
    if has_native_graph(protocol):
        enters = [node["node_ref"] for node in protocol["node_map"] if node["kind"] == 3]
        exits = [node["node_ref"] for node in protocol["node_map"] if node["kind"] == 4]
        assert len(enters) == len(exits) == 1
        assert controls[enters[0]]["state"] == 2 and controls[exits[0]]["state"] == 0
        assert protocol["resolved_task_count"] == 1
    else:
        assert protocol["resolved_task_count"] == 0
    assert all(controls[ref]["state"] == 0 for ref in (root["task_ref"], consumer["task_ref"]))


# Compilation, healthy/fault/recovery runs and diagnostic decoding share this case budget.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("mask,lane", _shape_params(_SHAPES))
@pytest.mark.parametrize("blocks", (1, 2))
def test_bounded_kernel_timeout_keeps_failed_run_contained(st_platform, st_device_ids, mask, lane, blocks, monkeypatch):
    _exercise_failure(st_platform, st_device_ids, mask, lane, blocks, monkeypatch, _STALL_MS)


@pytest.mark.parametrize("mask,lane", [(2, 1), (7, 2)])
def test_native_graph_failure_does_not_release_exit(st_platform, st_device_ids, mask, lane, monkeypatch):
    timeout_ms = _SIM_TIMEOUT_MS if st_platform.endswith("sim") else _TIMEOUT_MS
    stall_ms = _STALL_MS * timeout_ms // _TIMEOUT_MS
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", str(timeout_ms))
    output_root = _ROOT / "outputs"
    output_root.mkdir(exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native-graph-failure-", dir=output_root))
    device = int(st_device_ids[0])
    evidence = _run_pass(st_platform, device, artifacts, mask, lane, 2, stall_ms, recovery=False, native_graph=True)
    if st_platform.endswith("sim"):
        evidence += _run_pass(st_platform, device, artifacts, mask, lane, 2, stall_ms, recovery=True, native_graph=True)
    else:
        code = (
            "import runpy,sys,json; from pathlib import Path; "
            "ns=runpy.run_path(sys.argv[1]); p=json.loads(sys.argv[2]); p['artifacts']=Path(p['artifacts']); "
            "print(ns['_RECOVERY_MARKER']+json.dumps(ns['_run_pass'](**p)))"
        )
        parameters = dict(
            platform=st_platform,
            device_id=device,
            artifacts=str(artifacts),
            mask=mask,
            lane=lane,
            blocks=2,
            stall_ms=stall_ms,
            recovery=True,
            native_graph=True,
        )
        result = subprocess.run(
            [sys.executable, "-u", "-c", code, str(Path(__file__).resolve()), json.dumps(parameters)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        rows = [
            json.loads(line.removeprefix(_RECOVERY_MARKER))
            for line in result.stdout.splitlines()
            if line.startswith(_RECOVERY_MARKER)
        ]
        assert len(rows) == 1, result.stdout + result.stderr
        evidence += rows[0]
    (artifacts / "result.json").write_text(json.dumps(evidence, indent=2))


@pytest.mark.platforms(["a5"])
@pytest.mark.timeout(180)
@pytest.mark.parametrize("mask,lane", _onboard_shape_params(_SHAPES))
def test_nonreturning_kernel_timeout_keeps_failed_run_contained(st_platform, st_device_ids, mask, lane, monkeypatch):
    # The simulator has no STARS watchdog; a non-returning kernel would wedge it.
    assert st_platform == "a5"
    monkeypatch.setenv("SIMPLER_OP_EXECUTE_TIMEOUT_US", "3000000")
    monkeypatch.setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "4000")
    _exercise_failure(st_platform, st_device_ids, mask, lane, 2, monkeypatch, -1)


@pytest.mark.parametrize("mask,lane", _shape_params(_TICKET_SHAPES))
def test_large_ticket_claim_timeout_keeps_failed_run_contained(st_platform, st_device_ids, mask, lane, monkeypatch):
    # 257 exceeds twice every supported single-chip claimant population.
    _exercise_failure(st_platform, st_device_ids, mask, lane, 257, monkeypatch, _STALL_MS)


# The orchestration reads this device's own cluster count and launches two
# full waves against it. The host cannot know that count before the run, so it
# asks with this sentinel and sizes its buffers for the widest topology the
# platform admits.
_MAX_CLUSTERS = 36
_TWO_WAVE_REQUEST = -1


def _lane_positions_per_cluster(mask):
    # A cluster offers two execution positions when a single AIV lane carries
    # the task, and one otherwise.
    return 2 if mask in (2, 4) else 1


def _two_wave_capacity(mask):
    return _MAX_CLUSTERS * _lane_positions_per_cluster(mask) * 2


def _root_block_num(protocol):
    roots = [task for task in protocol["tasks"] if task["active_mask"] and 3 not in task["kernel_ids"]]
    assert len(roots) == 1, roots
    return int(roots[0]["block_num"])


def _check_two_wave_topology(protocol, blocks, mask, required):
    if not required:
        return
    assert protocol_version(protocol)
    positions = protocol["cluster_count"] * _lane_positions_per_cluster(mask)
    assert blocks == positions * 2, "two-wave fault must use the actual device topology"


@pytest.mark.parametrize("mask,lane", _shape_params(_TICKET_SHAPES))
def test_two_wave_ticket_claim_timeout_keeps_failed_run_contained(st_platform, st_device_ids, mask, lane, monkeypatch):
    _exercise_failure(
        st_platform,
        st_device_ids,
        mask,
        lane,
        _TWO_WAVE_REQUEST,
        monkeypatch,
        _STALL_MS,
        two_wave=True,
    )


_RECOVERY_MARKER = "RESIDENT_RECOVERY_EVIDENCE "


def _owned_wait_release(raw, snapshot, native_error):
    """Qualify the fixture's free assertion from this call's owned RTS wait records."""
    result = {"complete": False, "reason": "unqualified", "kernel_stop_mechanism": "unknown"}

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    def fields(line):
        row = {}
        for word in line.split("[STRACE] ", 1)[1].split():
            if "=" in word:
                key, value = word.split("=", 1)
                require(key not in row, "duplicate field")
                row[key] = value
        return row

    try:
        require(snapshot.get("host_log_complete") and not snapshot.get("printed_records_truncated"), "incomplete log")
        require(snapshot["pid"] == os.getpid(), "foreign snapshot PID")
        require(
            len(raw) == snapshot["bytes"] and hashlib.sha256(raw).hexdigest() == snapshot["sha256"], "snapshot bytes"
        )
        matches = re.findall(
            r"\(run_id=(\d+) slot=(\d+) generation=(\d+) dispatch_id=(\d+) run_epoch=(\d+)\)",
            native_error,
        )
        require(len(matches) == 1, "missing or ambiguous caught native identity")
        run, slot, generation, dispatch, epoch = map(int, matches[0])
        require(epoch > 0 and epoch == snapshot["call"]["capture_run_epoch"], "foreign capture epoch")
        identity = {"ep": str(epoch), "gen": str(generation), "disp": str(dispatch), "slot": str(slot)}
        lines = raw.decode("utf-8").splitlines()
        anchors = {
            (int(device), runner)
            for line in lines
            for device, runner in re.findall(r"Callable registration: device=(\d+) runner=(0x[0-9a-fA-F]+)", line)
            if int(device) == snapshot["call"]["device"]
        }
        require(len(anchors) == 1, "missing or ambiguous owned registration")
        device, context = anchors.pop()
        require(int(context, 16) != 0, "null registration context")
        bindings, pending, completed, roots, releases = {}, {}, {}, [], []
        pid = str(snapshot["pid"])
        for line in lines:
            if "[STRACE] " not in line:
                continue
            name_match = re.search(r"\bname=([^ ]+)", line)
            if name_match is None or not (
                name_match[1].startswith("chip.run.proof.")
                or name_match[1].startswith("chip.run.borrow_release")
                or name_match[1] == "chip.run"
            ):
                continue
            row = fields(line)
            require(row["pid"] == pid, "foreign record PID")
            name = row["name"]
            require(not name.endswith(".invalid"), "invalid diagnostic record")
            thread = row["pid"], row["tid"]
            own = all(row.get(key) == value for key, value in identity.items())
            if name == "chip.run.proof.record":
                if not releases:
                    require(
                        all(proof["event"] != row["e"] for proof in completed.values()),
                        "event rebound before borrow release",
                    )
                bindings[row["e"]] = row
                pending.pop(thread, None)
            elif name == "chip.run.proof.event_wait":
                require(thread not in pending, "unpaired raw wait")
                pending[thread] = row
            elif name == "chip.run.proof.consume":
                sdk = pending.pop(thread, None)
                if not own:
                    continue
                bound = bindings.get(row["e"])
                require(
                    bound is not None and all(bound.get(key) == value for key, value in identity.items()),
                    "foreign event binding",
                )
                require(sdk is not None and sdk["e"] == row["e"], "missing raw wait")
                require(row["op"] == "wait" and sdk["fresh"] == "1" and sdk["status_valid"] == "0", "not a fresh wait")
                require(
                    sdk["rc"] == row["rc"] == "0" and row["c"] == row["owner"] == "1", "pending or unowned boundary"
                )
                require(bound["f"] == row["f"] and bound["role"] == row["role"], "foreign fence or role")
                role = int(row["role"])
                require(role in (0, 1) and role not in completed, "duplicate boundary role")
                cache = int(row["cache"])
                mask = sum(1 << done for done in completed)
                require(0 <= cache <= 3 and not cache & (1 << role) and not cache & ~mask, "foreign cache proof")
                completed[role] = {"event": row["e"], "fence": row["f"], "stream": bound["s"], "inv": bound["inv"]}
            elif name == "chip.run.borrow_release" and own:
                require(set(completed) == {0, 1}, "release preceded both boundaries")
                require(row["ctx"] == context and int(row["borrow"]) == slot + 1, "foreign borrow context")
                require(int(row["run"]) == run and row["proof"] == "1" and row["keep"] == "0", "borrow retained")
                releases.append(row)
            elif name == "chip.run" and row.get("run_epoch") == str(epoch):
                require(
                    row.get("run_id") == str(run)
                    and row.get("slot_id") == str(slot)
                    and row.get("generation") == str(generation)
                    and row.get("dispatch_id") == str(dispatch),
                    "foreign native root",
                )
                roots.append(row)
        require(set(completed) == {0, 1} and len(releases) == len(roots) == 1, "missing own proof or release")
        require(completed[0]["event"] != completed[1]["event"], "aliased event roles")
        require(completed[0]["fence"] == completed[1]["fence"], "different fences")
        require(all(proof["inv"] == roots[0]["inv"] for proof in completed.values()), "foreign invocation")
        result.update(
            complete=True,
            reason="owned two-boundary wait and borrow release",
            identity=identity,
            run_id=run,
            device=device,
            context=context,
            boundaries=completed,
            context_anchor="same owned log registration; no native failure record is assumed",
        )
    except (KeyError, TypeError, ValueError) as error:
        result["reason"] = str(error)[:256]
    return result


def _retain_completion_proof(
    artifacts, stage, buffers, *, failed, call=None, free_outcome=None, close_returned=None, native_error=None
):
    """Take a best-effort snapshot of this process's existing completion diagnostics."""
    row = {
        "stage": stage,
        "pid": os.getpid(),
        "mono_ns": time.monotonic_ns(),
        "free_outcome": free_outcome,
        "close_returned": close_returned,
    }
    records = []
    try:
        row["call"] = dict(call or {})
        row["buffers"] = [
            {"identity": repr(buffer.identity), "base": int(buffer.base), "bytes": int(buffer.nbytes)}
            for buffer in buffers
        ]
        row["flushed"] = _task_interface._flush_host_log(1000)
        row["pending"] = _task_interface._host_log_pending_records()
        row["dropped"] = _task_interface._host_log_dropped_records()
        directory = _task_interface._native_host_log_directory()
        if not directory:
            raise RuntimeError("this process has no bound host-log directory")
        source = Path(directory) / f"host.{os.getpid()}.log"
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise RuntimeError("owned host log is not a regular file")
            raw = stream.read()
        target = Path(artifacts) / f"proof-host-{stage}.log"
        target.write_bytes(raw)
        row.update(log=target.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        records = [
            line
            for line in raw.decode("utf-8", errors="replace").splitlines()
            if any(
                marker in line
                for marker in (
                    "name=chip.run.proof.",
                    "name=chip.run.borrow_release",
                    "name=chip.run.claim_release ",
                    "name=chip.run ",
                )
            )
        ]
        row["proof_records"] = len(records)
        row["printed_records_truncated"] = len(records) > 64 or any(len(line) > 1024 for line in records[-64:])
        row["host_log_complete"] = bool(row["flushed"] and row["pending"] == 0 and row["dropped"] == 0)
        if native_error is not None:
            row["last_use"] = _owned_wait_release(raw, row, native_error)
    except BaseException as error:  # noqa: BLE001 -- diagnostics cannot replace the original run/free/close error
        row.update(host_log_complete=False, diagnostic_error=f"{type(error).__name__}: {error}"[:512])
    try:
        (Path(artifacts) / f"proof-{stage}.json").write_text(json.dumps(row, indent=2) + "\n")
    except BaseException as error:  # noqa: BLE001 -- stderr is the remaining diagnostic destination
        row["snapshot_error"] = f"{type(error).__name__}: {error}"[:512]
    if failed:
        # CI's existing failure capture prints stderr; the process's transient
        # host-log spool is removed on exit and the scene workflow has no upload.
        try:
            print("RESIDENT_COMPLETION_PROOF " + json.dumps(row), file=sys.stderr, flush=True)
            for line in records[-64:]:
                print(line[:1024], file=sys.stderr, flush=True)
        except BaseException as error:  # noqa: BLE001 -- failed stderr cannot replace the original exception
            row["stderr_error"] = f"{type(error).__name__}: {error}"[:512]
    return row


def _run_pass(  # noqa: PLR0912 -- one Worker owns healthy, failed, and cleanup paths
    platform, device_id, artifacts, mask, lane, blocks, stall_ms, *, recovery, two_wave=False, native_graph=False
):
    """Drive one Worker over this case and return its evidence rows.

    Returned bounded SIM faults use their joined-producer release check.
    This test qualifies successful free from this call's owned RTS evidence.
    Unavailable diagnostics fail qualification without deciding runtime policy.
    Onboard recovery uses the fresh process `_recovery_pass` spawns.
    """
    nonreturning = stall_ms < 0
    capacity = _two_wave_capacity(mask) if blocks < 0 else blocks
    callable_obj = _compile(platform, native_graph=native_graph)
    _task_interface._bind_host_log_session_directory()
    worker = Worker(
        level=2,
        platform=platform,
        runtime="host_build_graph",
        device_id=int(device_id),
        enable_sdma=False,
    )
    buffers = []
    failed = False
    run_in_flight = False
    evidence = []
    proof_call = {
        "platform": platform,
        "device": int(device_id),
        "mask": mask,
        "lane": lane,
        "requested_blocks": blocks,
        "recovery": recovery,
    }
    try:
        handle = worker.register(callable_obj)
        worker.init()
        values = [torch.zeros(size, dtype=torch.int64) for size in ((capacity * 3 + 1) * 16, 16)]
        for value in values:
            buffers.append(worker.malloc(value.nbytes))
        for duration in (0,) if recovery else (0, stall_ms):
            args = _args(worker, buffers, values, blocks, mask, lane, duration)
            prefix = artifacts / ("recovered" if recovery else "failed" if duration else "healthy")
            prefix.mkdir()
            config = CallConfig()
            # The public runner requires at least two threads; either watchdog can publish first.
            config.aicpu_thread_num = 2
            config.enable_chip_swimlane = 4
            config.output_prefix = str(prefix)
            before = worker.scheduler_mode_publish_count
            proof_call.update(stall_ms=duration, capture=str(prefix), capture_run_epoch=None)
            started = time.monotonic()
            run_in_flight = True
            if duration:
                failed = True
                if nonreturning:
                    codes = r"(-100|507(046|018|000))"
                else:
                    codes = "-100" if platform.endswith("sim") else "507018"
                with pytest.raises(RuntimeError, match=rf"finalize_native_run failed with code {codes}\b") as error:
                    worker.run(handle, args, config)
            else:
                worker.run(handle, args, config)
                run_in_flight = False
            elapsed = time.monotonic() - started
            assert elapsed < (60 if platform.endswith("sim") else 20)
            assert worker.scheduler_mode_publish_count > before
            assert worker.scheduler_runtime_mode in (1, 2)
            path, raw = _capture(prefix)
            proof_call.update(capture=str(path), capture_run_epoch=raw["aicore_protocol"]["run_epoch"])
            assert protocol_version(raw["aicore_protocol"])
            assert has_native_graph(raw["aicore_protocol"]) == native_graph
            launched = _root_block_num(raw["aicore_protocol"])
            assert blocks < 0 or launched == blocks
            # A STARS-killed op can poison device access; its D2H output is
            # not readable evidence. Host-exported failure records remain
            # mandatory and must not masquerade as a successful capture.
            if not (failed and nonreturning):
                for buffer, value in zip(buffers, values, strict=True):
                    worker.copy_from(value, buffer)
                _check_output(values, capacity, launched, mask, failed=failed)
            _check_two_wave_topology(raw["aicore_protocol"], launched, mask, two_wave)
            if failed:
                assert raw["aicore_protocol"]["run_epoch"] > evidence[-1]["run_epoch"]
                _check_failed_capture(raw, launched, mask, lane, nonreturning=nonreturning, stall_ms=stall_ms)
                if not nonreturning:
                    assert values[0].view(-1, 16)[lane, 0] == 1
                rejected_count = worker.scheduler_mode_publish_count
                with pytest.raises(RuntimeError, match="poisoned"):
                    worker.run(handle, args, config)
                assert worker.scheduler_mode_publish_count == rejected_count
                joined_sim = platform.endswith("sim") and not nonreturning
                own_wait_complete = False
                if not joined_sim:
                    own_wait_complete = (
                        _retain_completion_proof(
                            artifacts,
                            "before-free",
                            buffers,
                            failed=True,
                            call=proof_call,
                            native_error=str(error.value),
                        )
                        .get("last_use", {})
                        .get("complete", False)
                    )
                if joined_sim or own_wait_complete:
                    while buffers:
                        worker.free(buffers[-1])
                        buffers.pop()
                    run_in_flight = False
                else:
                    pytest.fail("owned RTS last-use evidence unavailable; buffers retained until close")
            else:
                assert read_perf_data(str(path))["protocol_audit"]["protocol_verified"]
            evidence.append(
                {
                    "capture": str(path.relative_to(artifacts)),
                    "failed": failed,
                    "scheduler_error": raw["aicore_protocol"]["scheduler_error"],
                    "error_site": raw["aicore_protocol"]["first_error"]["site"],
                    "run_epoch": raw["aicore_protocol"]["run_epoch"],
                    "output_verified": not (failed and nonreturning),
                    "native_error": str(error.value) if failed else None,
                    "run_seconds": elapsed,
                    "runtime_mode": worker.scheduler_runtime_mode,
                    "resubmit_refused": failed,
                    "borrowed_buffers_kept": failed and bool(buffers),
                }
            )
    finally:
        try:
            if not failed and not run_in_flight:
                for buffer in buffers:
                    worker.free(buffer)
        finally:
            _retain_completion_proof(artifacts, "before-close", buffers, failed=failed, call=proof_call)
            closing = time.monotonic()
            close_returned = False
            try:
                worker.close()
                close_returned = True
            finally:
                close_elapsed = time.monotonic() - closing
                _retain_completion_proof(
                    artifacts,
                    "after-close",
                    buffers,
                    failed=failed or not close_returned,
                    call=proof_call,
                    close_returned=close_returned,
                )
            assert close_elapsed < (60 if nonreturning and failed else 20)
            if evidence:
                evidence[-1]["worker_close_seconds"] = close_elapsed
    return evidence


def _recovery_pass(platform, device_id, artifacts, mask, lane, blocks, stall_ms, two_wave):
    """Run the recovery pass in a fresh process and return its evidence rows."""
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--platform",
            platform,
            "--device",
            str(device_id),
            "--artifacts",
            str(artifacts),
            "--mask",
            str(mask),
            "--lane",
            str(lane),
            # A sentinel block count and a non-returning stall are negative;
            # the inline form keeps argparse from reading them as options.
            f"--blocks={blocks}",
            f"--stall-ms={stall_ms}",
            *(["--two-wave"] if two_wave else []),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [
        json.loads(line.removeprefix(_RECOVERY_MARKER))
        for line in result.stdout.splitlines()
        if line.startswith(_RECOVERY_MARKER)
    ]
    assert len(rows) == 1, result.stdout + result.stderr
    return rows[0]


def _exercise_failure(st_platform, st_device_ids, mask, lane, blocks, monkeypatch, stall_ms, *, two_wave=False):
    # A simulator's watchdog includes host scheduling delays in healthy and
    # recovery passes. The bounded fault must outlast that budget.
    timeout_ms = _SIM_TIMEOUT_MS if st_platform.endswith("sim") else _TIMEOUT_MS
    if stall_ms > 0:
        stall_ms = stall_ms * timeout_ms // _TIMEOUT_MS
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", str(timeout_ms))
    failure_kind = "nonreturning" if stall_ms < 0 else "bounded"
    output_root = _ROOT / "outputs"
    output_root.mkdir(exist_ok=True)
    artifacts = Path(
        tempfile.mkdtemp(
            prefix=f"resident-failure-{failure_kind}-{st_platform}-m{mask}-l{lane}-n{blocks}-", dir=output_root
        )
    )
    device_id = int(st_device_ids[0])
    evidence = _run_pass(
        st_platform, device_id, artifacts, mask, lane, blocks, stall_ms, recovery=False, two_wave=two_wave
    )
    # Only an onboard runtime binds the releasable hook that holds a failed
    # run's device reservation, so only there must the recovery pass pay for a
    # fresh process. A simulator admits the second Worker in this one.
    if st_platform.endswith("sim"):
        evidence += _run_pass(
            st_platform, device_id, artifacts, mask, lane, blocks, stall_ms, recovery=True, two_wave=two_wave
        )
    else:
        evidence += _recovery_pass(st_platform, device_id, artifacts, mask, lane, blocks, stall_ms, two_wave)
    (artifacts / "result.json").write_text(
        json.dumps(
            {"mask": mask, "lane": lane, "blocks": blocks, "failure_kind": failure_kind, "runs": evidence}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run this case's recovery pass in its own process.")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--mask", type=int, required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--stall-ms", type=int, required=True)
    parser.add_argument("--two-wave", action="store_true")
    options = parser.parse_args()
    print(
        _RECOVERY_MARKER
        + json.dumps(
            _run_pass(
                options.platform,
                options.device,
                Path(options.artifacts),
                options.mask,
                options.lane,
                options.blocks,
                options.stall_ms,
                recovery=True,
                two_wave=options.two_wave,
            )
        ),
        flush=True,
    )
