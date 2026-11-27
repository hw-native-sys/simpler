# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A kernel return hint preserves the real defer completion dependency."""

import json
from array import array
from contextlib import contextmanager
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

HERE = Path(__file__).resolve().parent
WORDS = 48

pytestmark = [
    pytest.mark.platforms(["a5", "a5sim"]),
    pytest.mark.runtime("percounter"),
    pytest.mark.device_count(1),
]


def build_callable(platform):
    spec = {
        "orchestration": {
            "source": str(HERE / "orchestration.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.SCALAR],
        },
        "incores": [
            {"func_id": 0, "source": str(HERE / "producer.cpp"), "core_type": "aiv", "signature": [D.INOUT, D.SCALAR]},
            {"func_id": 1, "source": str(HERE / "consumer.cpp"), "core_type": "aic", "signature": [D.INOUT]},
            {"func_id": 2, "source": str(HERE / "publisher.cpp"), "core_type": "aiv", "signature": [D.INOUT]},
        ],
    }
    return compile_chip_callable_spec(spec, platform, "percounter", (__file__, platform))


@contextmanager
def initialized_worker(platform, device, callable_):
    worker = Worker(level=2, platform=platform, runtime="percounter", device_id=device)
    try:
        handle = worker.register(callable_)
        worker.init()
        state = worker.malloc(array("i", [0] * WORDS).itemsize * WORDS)
        try:
            yield worker, handle, state
        finally:
            worker.free(state)
    finally:
        worker.close()


def run_case(worker, handle, state, deferred, level, output):
    data = array("i", [0] * WORDS)
    worker.copy_to(state, data)
    args = TaskArgs()
    args.add_tensor(state.tensor(shapes=(WORDS,), dtype=DataType.INT32), TensorArgType.INOUT)
    args.add_scalar(int(deferred))
    config = CallConfig()
    config.enable_chip_swimlane = level
    config.output_prefix = str(output)
    worker.run(handle, args, config)
    worker.copy_from(data, state)
    expected = array("i", [0] * WORDS)
    expected[0], expected[16], expected[32], expected[33] = 41, int(deferred), 42, int(deferred)
    assert data == expected, (list(data), list(expected))
    if not level:
        return None
    (capture,) = output.rglob("chip_swimlane_records.json")
    return json.loads(capture.read_text())


def _event(rows, task, kind):
    (event,) = (row for row in rows if row["task_id"] == task and row["kind"] == kind)
    return event


def check_trace(capture, deferred, peer_dmb):
    events = capture["percounter_task_events"]
    assert events["dropped"] == events["unmatched_deferred_updates"] == 0, events
    rows = events["records"]
    producer, consumer = (_event(rows, task, "kernel_end") for task in (0, 1))
    assert producer["worker_id"] != consumer["worker_id"]
    assert producer["cycles"] <= consumer["kernel_start_cycles"]

    # One return hint reaches the consumer whether the producer defers or completes directly.
    notifications = [row for row in rows if row["task_id"] == 0 and row["kind"] == "peer_dmb_published"]
    assert len(notifications) == int(peer_dmb), notifications
    if notifications:
        notification = notifications[0]
        assert notification["cycles"] >= producer["cycles"]
        assert notification["target_worker"] == consumer["worker_id"]

    registrations = [row for row in rows if row["task_id"] == 0 and row["kind"] == "defer_registered"]
    assert len(registrations) == int(deferred), registrations
    if deferred:
        ready = _event(rows, 0, "defer_signal_ready")
        done = _event(rows, 0, "deferred_completion_issue")
        assert producer["cycles"] <= registrations[0]["cycles"] <= ready["cycles"]
        assert ready["cycles"] <= done["cycles"] <= consumer["kernel_start_cycles"]
        if notifications:
            initial = _event(rows, 1, "peer_dmb_initial_proof")
            assert initial["ready_inputs"] == 0
            assert initial["cycles"] < producer["cycles"]
            (observed,) = (
                row
                for row in rows
                if row["task_id"] == 1
                and row["kind"] == "peer_dmb_observed"
                and row["worker_id"] == notification["target_worker"]
                and row["token"] == notification["token"]
            )
            assert observed["ready_inputs"] == 0
            assert registrations[0]["cycles"] <= notification["cycles"] <= observed["cycles"] < ready["cycles"]


def test_same_kernel_return_hint_with_and_without_defer(st_platform, st_device_ids, tmp_path):
    callable_ = build_callable(st_platform)
    with initialized_worker(st_platform, int(st_device_ids[0]), callable_) as (worker, handle, state):
        # Replay on the same Worker so a previous run's events can still occupy the same buffer.
        for run, deferred in enumerate((False, True, False, True)):
            capture = run_case(worker, handle, state, deferred, 2, tmp_path / str(run))
            check_trace(capture, deferred, st_platform == "a5")
