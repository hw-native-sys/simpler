#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The PMU window counts fixed work on both sides of a utilization query."""

import csv
import ctypes
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l2_ref_args
from simpler_setup.tools.hbg.task_id import TaskId
from tests.st.a2a3.host_build_graph.worker_async_fifo.device_gate import DeviceGate, device_gate_environment

_KERNELS = "../../worker_async_fifo/kernels"
_SIZE = 128 * 128
_CHAIN_LENGTH = 512  # pipelined_vector_orch.cpp
_EVENT_GROUP = 1  # A2/A3 arithmetic events, including vec_fp32_exec (0x4b)
# One FP32 repeat processes 256 bytes. Both the gated vector add and each
# scalar-add step execute exactly one tile's worth of repeats.
_FP32_REPEATS_PER_TASK = _SIZE // (256 // torch.float32.itemsize)


@pytest.fixture(scope="class", autouse=True)
def _device_gate_failure_budgets():
    with device_gate_environment():
        yield


def _utilization_query_for_device(device_id):
    mapping = subprocess.run(["npu-smi", "info", "-m"], capture_output=True, text=True, timeout=5, check=True)
    matches = []
    for line in mapping.stdout.splitlines():
        columns = line.split()[:3]  # Card, chip, and logical ID precede the optional physical ID.
        if len(columns) == 3 and all(col.isdecimal() for col in columns) and int(columns[2]) == device_id:
            matches.append(columns[:2])
    assert len(matches) == 1, f"cannot map logical NPU {device_id} to a unique npu-smi card/chip: {mapping.stdout}"
    card, chip = matches[0]
    return ["npu-smi", "info", "-t", "usages", "-i", card, "-c", chip]


def _assert_every_task_counted(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    expected = {TaskId.make_global(i): int(i != 0) for i in range(_CHAIN_LENGTH + 1)}
    rows_by_id = {TaskId(int(row["task_id"], 16)): row for row in rows}
    assert len(rows) == len(rows_by_id) == len(expected), "PMU CSV has missing or duplicate task rows"
    assert rows_by_id.keys() == expected.keys(), "PMU CSV task IDs do not match the submitted chain"
    epochs = {int(row["run_epoch"]) for row in rows}
    assert len(epochs) == 1 and min(epochs) > 0, f"PMU CSV mixes run identities: {epochs}"

    for task_id, func_id in expected.items():
        row = rows_by_id[task_id]
        assert int(row["func_id"]) == func_id and int(row["event_type"]) == _EVENT_GROUP, row
        observed = int(row["vec_fp32_exec"])
        assert observed == _FP32_REPEATS_PER_TASK, (
            f"{task_id.display()} counted {observed} FP32 repeats; expected {_FP32_REPEATS_PER_TASK}: {row}"
        )
        assert int(row["pmu_total_cycles"]) > 0, f"{task_id.display()} reported zero total cycles: {row}"


@scene_test(level=2, runtime="host_build_graph")
@pytest.mark.coverage_skip(
    platforms=["a2a3"],
    reason="the counting-window case requires --enable-pmu 1",
)
class TestPmuCountingWindow(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": f"{_KERNELS}/orchestration/pipelined_vector_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_KERNELS}/aiv/gated_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }
    CASES = [
        {
            "name": "utilization_query",
            "platforms": ["a2a3"],
            # The per-PR onboard DFX action selects this case with event group 1.
            "manual": True,
            "config": {"device_count": 1},
            "params": {},
        },
    ]

    def _run_and_validate_l2(self, worker, callable_obj, case, enable_pmu=0, output_prefix="", **_kwargs):
        if not enable_pmu:
            pytest.skip("the counting-window case requires --enable-pmu 1")
        assert enable_pmu == _EVENT_GROUP, "the counting-window case requires PMU arithmetic events (--enable-pmu 1)"
        query = _utilization_query_for_device(int(worker._config["device_id"]))
        # Pass a fresh path to the collector and read that exact artifact after
        # completion, even if another invocation has the same case label.
        run_dir = Path(tempfile.mkdtemp(prefix="window-", dir=output_prefix))
        config = self._build_config(case["config"], enable_pmu=enable_pmu, output_prefix=run_dir)
        callable_handle = worker.register(callable_obj)
        gate = DeviceGate(worker)
        run = None
        try:
            host_args = TaskArgsBuilder(
                TensorArg("a", torch.full((_SIZE,), 2.0, dtype=torch.float32)),
                TensorArg("b", torch.full((_SIZE,), 3.0, dtype=torch.float32)),
                TensorArg("out", torch.zeros(_SIZE, dtype=torch.float32)),
                # The host orchestration reads the scalar addend from this input.
                TensorArg("control", torch.ones(_SIZE, dtype=torch.float32), transfer=TensorTransfer.NONE),
                Scalar("gate", ctypes.c_int64(gate.address)),
            )
            args, _ = _build_l2_ref_args(host_args, self.CALLABLE["orchestration"]["signature"], worker)
            run = worker.submit(callable_handle, args, config=config)
            try:
                gate.wait_entered()
                result = subprocess.run(query, capture_output=True, text=True, timeout=6, check=True)
                assert "Aicore Usage Rate" in result.stdout, f"query did not report AICore utilization: {result.stdout}"
                gate.wait_entered()
            finally:
                try:
                    gate.release()
                finally:
                    run.wait(15.0)
            _assert_every_task_counted(run_dir / "pmu.csv")
            torch.testing.assert_close(host_args.out, host_args.a + host_args.b + _CHAIN_LENGTH * host_args.control[0])
        finally:
            if run is None or run.done:
                gate.close()


if __name__ == "__main__":
    with device_gate_environment():
        SceneTestCase.run_module(__name__)
