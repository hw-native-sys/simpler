#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Task-time Program payloads: add writes 5, then INOUT reads 5 and writes 6.

The same backing is reused by both tasks, so a run-boundary snapshot cannot
satisfy both stage values. PARTIAL and HYBRID reuse the orchestration's marks.
"""

import json
import struct

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename, dump_args_level
from simpler_setup.tools.hbg.task_id import Space, TaskId


class _ArgumentDumpScene(SceneTestCase):
    def test_run(self, st_platform, st_worker, request):
        mode = request.config.getoption("--dump-args", default="off")
        level = dump_args_level(mode)
        name = type(self).__name__
        before = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for folder in _outputs_dir().glob(f"{name}_*")
            for path in folder.rglob("args_dump.json")
        }
        super().test_run(st_platform, st_worker, request)
        if not level:
            return
        matched = self._matching_cases(st_platform, request)
        assert len(matched) == 1
        label = _sanitize_for_filename(f"{name}_{matched[0]['name']}")
        current = {
            path
            for folder in _outputs_dir().glob(f"{label}_*")
            for path in folder.rglob("args_dump.json")
            if before.get(path) != (path.stat().st_mtime_ns, path.read_bytes())
        }
        assert len(current) == 1, current
        manifest_path = current.pop()
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        payload = b"" if manifest["bin_file"] is None else (folder / manifest["bin_file"]).read_bytes()
        self._check_dump(manifest, payload, level)


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestDumpArgsExampleA5(_ArgumentDumpScene):
    """f = (a + b) + 1, where a=2.0, b=3.0 -> f=6.0."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/dump_args_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_add_scalar_inplace.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a5sim", "a5"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b) + 1

    def _check_dump(self, manifest, payload, level):
        _check_task_payloads(manifest, payload, level)


def _check_task_payloads(manifest, payload, level, *, native=False):
    assert manifest["dump_args_level"] == level
    assert manifest["dropped_records_complete"] is True
    assert manifest["dropped_records"] == 0 and not manifest.get("counts_unknown", False)
    assert not manifest.get("host_discarded_args", 0) and not manifest.get("metadata_discarded_records", 0)
    records = manifest["args"]
    epochs = {record["run_epoch"] for record in records}
    assert len(epochs) == 1 and all(type(epoch) is int and epoch > 0 for epoch in epochs)
    expected = {
        (0, 0, "before_dispatch"): 2.0,
        (0, 1, "before_dispatch"): 3.0,
        (0, 2, "after_completion"): 5.0,
        (1, 0, "before_dispatch"): 5.0,
        (1, 0, "after_completion"): 6.0,
    }
    if level == 1:
        expected = {key: value for key, value in expected.items() if key[0] == 1}
    tensors = [record for record in records if record["kind"] == "tensor"]
    assert len(tensors) == len(expected)
    seen = set()
    intervals = []
    for record in tensors:
        identity = TaskId(int(record["task_id"], 16))
        assert identity.space() is (Space.SUB_TASK if native else Space.GLOBAL)
        if native:
            assert identity.parent_id() == 0
        task = identity.local_id()
        key = task, record["arg_index"], record["stage"]
        assert key in expected and key not in seen, record
        seen.add(key)
        assert record["func_id"] == [task]
        assert record["dtype"] == "FLOAT32" and record["shape"] == [128 * 128]
        assert record["role"] == ("inout" if task == 1 else "output" if key[2] == "after_completion" else "input")
        assert record["strides"] == [1] and record["start_offset"] == 0
        assert record["is_contiguous"] is True and record["numel"] == 128 * 128
        assert not record.get("truncated", False)
        size = record["bin_size"]
        selected = level != 2 or task == 1
        assert size == (128 * 128 * 4 if selected else 0), record
        if selected:
            start = record["bin_offset"]
            assert type(start) is int and 0 <= start <= len(payload) - size
            assert payload[start : start + size] == struct.pack("<f", expected[key]) * (128 * 128)
            intervals.append((start, start + size))
    assert seen == set(expected)
    scalars = [record for record in records if record["kind"] == "scalar"]
    assert len(scalars) == 1
    scalar = scalars[0]
    expected_scalar_task = TaskId.make_sub_task(0, 1).raw if native else TaskId.make_global(1).raw
    assert int(scalar["task_id"], 16) == expected_scalar_task and scalar["arg_index"] == 1
    assert scalar["stage"] == "before_dispatch" and scalar["dtype"] == "FLOAT32"
    assert scalar["role"] == "input"
    assert scalar["func_id"] == [1] and scalar["value"] == 1.0 and scalar["bin_size"] == 0
    assert manifest["total_args"] == len(expected) + 1
    ordered = sorted(intervals)
    assert ordered[0][0] == 0 and ordered[-1][1] == len(payload)
    assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestNativeGraphArgumentDumpA5(TestDumpArgsExampleA5):
    """Native child task identities retain task-time snapshots of one backing."""

    CALLABLE = {
        **TestDumpArgsExampleA5.CALLABLE,
        "orchestration": {
            **TestDumpArgsExampleA5.CALLABLE["orchestration"],
            "source": "kernels/orchestration/native_args_orch.cpp",
        },
    }

    def _check_dump(self, manifest, payload, level):
        _check_task_payloads(manifest, payload, level, native=True)


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestEmptyArgumentDumpA5(_ArgumentDumpScene):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/empty_args_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../dfx/chip_swimlane/kernels/aiv/kernel_empty.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.INOUT],
            }
        ],
    }
    CASES = [{"name": "ranked_zero_extent", "platforms": ["a5sim", "a5"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros((1,), dtype=torch.float32)),
            TensorArg("inout", torch.zeros((1,), dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        assert args.output.numel() == args.inout.numel() == 1

    def _check_dump(self, manifest, payload, level):
        assert manifest["dump_args_level"] == level
        assert manifest["dropped_records_complete"] is True and manifest["dropped_records"] == 0
        assert not manifest.get("counts_unknown", False)
        assert not manifest.get("host_discarded_args", 0) and not manifest.get("metadata_discarded_records", 0)
        assert payload == b"" and manifest["total_args"] == len(manifest["args"]) == 3
        assert {(arg["arg_index"], arg["stage"], arg["role"]) for arg in manifest["args"]} == {
            (0, "after_completion", "output"),
            (1, "before_dispatch", "inout"),
            (1, "after_completion", "inout"),
        }
        epochs = {arg["run_epoch"] for arg in manifest["args"]}
        assert len(epochs) == 1 and all(type(epoch) is int and epoch > 0 for epoch in epochs)
        for arg in manifest["args"]:
            assert int(arg["task_id"], 16) == 0 and arg["func_id"] == [0]
            assert arg["kind"] == "tensor" and arg["dtype"] == "FLOAT32"
            assert arg["shape"] == [0] and arg["numel"] == 0 and arg["bin_size"] == 0
            assert not arg.get("truncated", False)


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestNativeGraphWaveArgumentDumpA5(_ArgumentDumpScene):
    """One 73-block Mix child exceeds A5's 36-cluster residency capacity."""

    BLOCKS = 73
    ELEMENTS = BLOCKS * 3 * 16
    RTOL = ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/native_wave_args_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "source": f"../graph_execution/kernels/{'aic' if lane == 0 else 'aiv'}/kernel_spmd_mix.cpp",
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [D.INOUT],
            }
            for lane in range(3)
        ],
    }
    CASES = [{"name": "mix_73_blocks", "platforms": ["a5sim", "a5"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("output", torch.zeros(self.ELEMENTS, dtype=torch.float32)))

    def compute_golden(self, args, params):
        args.output.reshape(self.BLOCKS, 3, 16)[:, :, 0] = torch.arange(self.BLOCKS)[:, None]

    def _check_dump(self, manifest, payload, level):
        assert manifest["dump_args_level"] == level
        assert manifest["dropped_records_complete"] is True and manifest["dropped_records"] == 0
        assert not manifest.get("counts_unknown", False)
        assert not manifest.get("host_discarded_args", 0) and not manifest.get("metadata_discarded_records", 0)
        records = manifest["args"]
        assert len(records) == manifest["total_args"] == 3
        assert len({row["run_epoch"] for row in records}) == 1
        assert all(
            row["run_epoch"] > 0 and int(row["task_id"], 16) == TaskId.make_sub_task(0, 0).raw for row in records
        )
        assert all(row["func_id"] == [0, 1, 2] for row in records)
        tensors = {row["stage"]: row for row in records if row["kind"] == "tensor"}
        assert set(tensors) == {"before_dispatch", "after_completion"}
        expected = torch.zeros(self.ELEMENTS, dtype=torch.float32)
        intervals = []
        for stage in ("before_dispatch", "after_completion"):
            row = tensors[stage]
            assert row["arg_index"] == 0 and row["role"] == "inout" and row["dtype"] == "FLOAT32"
            assert row["shape"] == [self.ELEMENTS] and row["strides"] == [1] and row["start_offset"] == 0
            assert row["numel"] == self.ELEMENTS and row["is_contiguous"] is True
            assert not row.get("truncated", False) and row["bin_size"] == self.ELEMENTS * 4
            if stage == "after_completion":
                expected.reshape(self.BLOCKS, 3, 16)[:, :, 0] = torch.arange(self.BLOCKS)[:, None]
            start, size = row["bin_offset"], row["bin_size"]
            assert type(start) is int and 0 <= start <= len(payload) - size
            actual = torch.frombuffer(bytearray(payload[start : start + size]), dtype=torch.float32)
            assert torch.equal(actual, expected)
            intervals.append((start, start + size))
        scalar = next(row for row in records if row["kind"] == "scalar")
        assert scalar["arg_index"] == 1 and scalar["stage"] == "before_dispatch"
        assert scalar["dtype"] == "INT64" and scalar["value"] == 0 and scalar["bin_size"] == 0
        assert sorted(intervals) == [(0, self.ELEMENTS * 4), (self.ELEMENTS * 4, len(payload))]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
