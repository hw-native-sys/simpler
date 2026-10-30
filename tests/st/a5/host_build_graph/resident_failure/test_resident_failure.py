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
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection, CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec
from simpler_setup.tools.swimlane_converter import read_perf_data

_ROOT = Path(__file__).resolve().parents[5]
_KERNELS = _ROOT / "tests/fdwic/pa_hbg/spmd/kernels"
_STALL_MS = 2500
_TIMEOUT_MS = 2000
_SHAPES = [(mask, lane) for mask in (1, 2, 4, 3, 5, 6, 7) for lane in range(3) if mask & (1 << lane)]
_TICKET_SHAPES = ((1, 0), (2, 1), (4, 2), (6, 1), (7, 0), (7, 1), (7, 2))
# The exhaustive mask/lane sweep is about silicon behaviour, and each case here
# holds a device for its own process because a failed run poisons it. A
# simulator run therefore carries two shapes: the lane the protocol reports in
# slot 1, which is the one every capture check special-cases, and all three
# lanes together.
_SIM_SHAPES = frozenset({(4, 2), (7, 0)})


def _shape_params(shapes):
    return [
        pytest.param(mask, lane, marks=[] if (mask, lane) in _SIM_SHAPES else [pytest.mark.platforms(["a5"])])
        for mask, lane in shapes
    ]


pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.device_count(1),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.scheduler_mode("resident"),
    pytest.mark.timeout(120),
]


def _compile(platform):
    compiler = KernelCompiler(platform=platform)
    variant = "sim" if platform.endswith("sim") else "onboard"
    includes = [str(compiler.platform_dir / variant / "aicore")]
    spec = {
        "orchestration": {
            "source": str(_KERNELS / "runtime_failure_orchestration.cpp"),
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
        spec, platform, "host_build_graph", (__name__, platform, _pto_isa_compile_cache_token())
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


def _check_failed_capture(raw, blocks, mask, lane, *, nonreturning=False):
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
        assert execution["returned"][0] - execution["entry"][0] >= _STALL_MS * frequency // 1000
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
    assert controls[root["task_ref"]]["completed_blocks"] < blocks
    assert controls[consumer["task_ref"]]["completed_blocks"] == 0
    positions = protocol["cluster_count"] * (2 if mask in (2, 4) else 1)
    assert protocol["schema"] in (
        "hbg-native-protocol-v2",
        "hbg-native-protocol-v3",
        "hbg-native-protocol-v4",
        "hbg-native-protocol-v5",
    )
    if protocol["schema"] == "hbg-native-protocol-v5":
        assert all(
            type(control.get("admission_claim_lock")) is int and control["admission_claim_lock"] == 0
            for control in control_rows
        )
    minimum_blocks = (
        2
        if protocol["schema"] in ("hbg-native-protocol-v4", "hbg-native-protocol-v5")
        else positions * 2 + (1 if protocol["schema"] == "hbg-native-protocol-v2" else 0)
    )
    uses_tickets = blocks >= minimum_blocks
    tickets = controls[root["task_ref"]]["claim_tickets"]
    if uses_tickets:
        assert 1 <= tickets <= blocks + positions
        assert controls[root["task_ref"]]["next_block"] <= min(blocks, tickets)
    else:
        assert tickets == 0
    assert controls[consumer["task_ref"]]["claim_tickets"] == 0
    assert protocol["resolved_task_count"] == 0
    assert all(controls[ref]["state"] == 0 for ref in (root["task_ref"], consumer["task_ref"]))


@pytest.mark.parametrize("mask,lane", _shape_params(_SHAPES))
@pytest.mark.parametrize("blocks", (1, 2))
def test_bounded_kernel_timeout_keeps_failed_run_contained(st_platform, st_device_ids, mask, lane, blocks, monkeypatch):
    _exercise_failure(st_platform, st_device_ids, mask, lane, blocks, monkeypatch, _STALL_MS)


@pytest.mark.platforms(["a5"])
@pytest.mark.timeout(180)
@pytest.mark.parametrize("mask,lane", _SHAPES)
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
    assert protocol["schema"] in ("hbg-native-protocol-v3", "hbg-native-protocol-v4", "hbg-native-protocol-v5")
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


def _run_pass(platform, device_id, artifacts, mask, lane, blocks, stall_ms, *, recovery, two_wave=False):
    """Drive one Worker over this case and return its evidence rows.

    A failed run keeps its buffers borrowed, so the device holds its
    process-wide execution reservation until this process exits — a second
    Worker on that device is refused here. The recovery pass therefore runs in
    a fresh process, which is what `_recovery_pass` spawns.
    """
    nonreturning = stall_ms < 0
    capacity = _two_wave_capacity(mask) if blocks < 0 else blocks
    callable_obj = _compile(platform)
    worker = Worker(
        level=2,
        platform=platform,
        runtime="host_build_graph",
        device_id=int(device_id),
        enable_sdma=False,
    )
    buffers = []
    failed = False
    evidence = []
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
            started = time.monotonic()
            if duration:
                failed = True
                codes = r"(-100|507(046|018|000))" if nonreturning else "-100"
                with pytest.raises(RuntimeError, match=rf"finalize_native_run failed with code {codes}\b") as error:
                    worker.run(handle, args, config)
            else:
                worker.run(handle, args, config)
            elapsed = time.monotonic() - started
            assert elapsed < 20
            assert worker.scheduler_mode_publish_count > before
            assert worker.scheduler_runtime_mode in (1, 2)
            path, raw = _capture(prefix)
            assert raw["aicore_protocol"]["schema"] == "hbg-native-protocol-v5"
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
                _check_failed_capture(raw, launched, mask, lane, nonreturning=nonreturning)
                if not nonreturning:
                    assert values[0].view(-1, 16)[lane, 0] == 1
                rejected_count = worker.scheduler_mode_publish_count
                with pytest.raises(RuntimeError, match="poisoned"):
                    worker.run(handle, args, config)
                assert worker.scheduler_mode_publish_count == rejected_count
                for buffer in buffers:
                    with pytest.raises(RuntimeError, match="still in use by a submitted run"):
                        worker.free(buffer)
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
                    "borrowed_buffers_kept": failed,
                }
            )
    finally:
        if not failed:
            for buffer in buffers:
                worker.free(buffer)
        closing = time.monotonic()
        worker.close()
        close_elapsed = time.monotonic() - closing
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
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", str(_TIMEOUT_MS))
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
