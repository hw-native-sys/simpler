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

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename, dump_args_level


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


def _check_task_payloads(manifest, payload, level):
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
        task = int(record["task_id"], 16)
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
    assert int(scalar["task_id"], 16) == 1 and scalar["arg_index"] == 1
    assert scalar["stage"] == "before_dispatch" and scalar["dtype"] == "FLOAT32"
    assert scalar["role"] == "input"
    assert scalar["func_id"] == [1] and scalar["value"] == 1.0 and scalar["bin_size"] == 0
    assert manifest["total_args"] == len(expected) + 1
    ordered = sorted(intervals)
    assert ordered[0][0] == 0 and ordered[-1][1] == len(payload)
    assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))


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


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
