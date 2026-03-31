# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the batch CI runner."""

import sys
from pathlib import Path
from types import SimpleNamespace

import tools.ci as ci


class _FakePath:
    def __init__(self, value: str):
        self._value = value

    def read_bytes(self) -> bytes:
        return self._value.encode()

    def __str__(self) -> str:
        return self._value


class _FakeWorker:
    instances = []

    def __init__(self):
        self.inits = []
        self.reset_count = 0
        _FakeWorker.instances.append(self)

    def init(self, device_id, host_path, aicpu_binary, aicore_binary):
        self.inits.append((device_id, host_path, aicpu_binary, aicore_binary))

    def reset(self):
        self.reset_count += 1


def _make_compiled_task(name: str, runtime_name: str):
    return ci.CompiledTask(
        spec=ci.TaskSpec(
            name=name,
            task_dir=Path("/tmp") / name,
            kernels_dir=Path("/tmp") / name / "kernels",
            golden_path=Path("/tmp") / name / "golden.py",
            platform="a2a3sim",
            runtime_name=runtime_name,
        ),
        chip_callable=object(),
        cases=[],
        runtime_bins=SimpleNamespace(
            host_path=_FakePath(f"/tmp/{runtime_name}/host.so"),
            aicpu_path=_FakePath(f"/tmp/{runtime_name}/aicpu.so"),
            aicore_path=_FakePath(f"/tmp/{runtime_name}/aicore.so"),
        ),
        golden_module=object(),
        kernel_config=object(),
    )


def test_run_sim_tasks_reuses_one_worker_per_runtime(monkeypatch):
    tasks = [
        _make_compiled_task("example:rt1_case1", "host_build_graph"),
        _make_compiled_task("example:rt1_case2", "host_build_graph"),
        _make_compiled_task("example:rt2_case1", "tensormap_and_ringbuffer"),
    ]
    run_calls = []

    monkeypatch.setattr("task_interface.ChipWorker", _FakeWorker)

    def fake_run_single_task(task, worker, device_id):
        run_calls.append((task.spec.name, worker, device_id))
        return True

    monkeypatch.setattr(ci, "run_single_task", fake_run_single_task)

    results = ci.run_sim_tasks(tasks, parallel=False)

    assert [r.name for r in results] == [t.spec.name for t in tasks]
    assert all(r.passed for r in results)
    assert len(_FakeWorker.instances) == 2
    assert all(worker.reset_count == 1 for worker in _FakeWorker.instances)

    host_group_worker = run_calls[0][1]
    assert run_calls[1][1] is host_group_worker
    assert run_calls[2][1] is not host_group_worker
    assert [call[2] for call in run_calls] == [0, 0, 0]


def test_parse_args_accepts_build_runtime(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ci.py", "-p", "a2a3sim", "--build-runtime"])
    args = ci.parse_args()

    assert args.platform == "a2a3sim"
    assert args.build_runtime is True
