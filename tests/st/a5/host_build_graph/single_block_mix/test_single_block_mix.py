#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-block Mix scheduling coverage for dependent and independent DAGs."""

import contextlib
import ctypes
import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename, dump_args_level

MATMUL_SIZE = 128
TILE_ELEMS = MATMUL_SIZE * MATMUL_SIZE
TASK_COUNT = 16


def _retain_failure_host_logs(case_name):
    folder = None
    try:
        from simpler_setup.scene_test import _bound_host_logs  # noqa: PLC0415

        folder = _outputs_dir() / "onboard-dfx-ci/failed" / case_name
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for source in _bound_host_logs()[:8]:
            size = source.stat().st_size
            limit = 4 * 1024 * 1024
            with source.open("rb") as stream:
                stream.seek(max(0, size - limit))
                body = stream.read(limit)
            (folder / source.name).write_bytes(body)
            records.append({"file": source.name, "source_bytes": size, "retained_bytes": len(body)})
        (folder / "logs.json").write_text(json.dumps(records, indent=2))
    except Exception as error:
        # Evidence collection must neither touch an unknown device owner nor
        # replace the actual execution failure with a spool/permissions error.
        print(f"Could not retain native failure log: {error}", flush=True)
        if folder is not None:
            with contextlib.suppress(OSError):
                (folder / "capture-error.txt").write_text(str(error)[:4096])


class _MixScene(SceneTestCase):
    def test_run(self, st_platform, st_worker, request):
        try:
            super().test_run(st_platform, st_worker, request)
        except Exception:
            _retain_failure_host_logs(type(self).__name__)
            raise


@scene_test(level=2, runtime="host_build_graph")
class TestSingleBlockMixHostBuildGraphA5(_MixScene):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_block_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.INOUT, D.IN, D.IN, D.INOUT, D.IN, D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MATMUL",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "MUL",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {"name": f"{kind}16_mask{mask}", "platforms": ["a5sim", "a5"], "params": {"graph_case": case, "mask": mask}}
        for kind, case in [("chain", 0), ("burst", 1)]
        for mask in [3, 5, 6, 7]
    ] + [{"name": "ordinary_mix_interleaved", "platforms": ["a5sim", "a5"], "params": {"graph_case": 2, "mask": 7}}]

    def generate_args(self, params):
        torch.manual_seed(42)
        a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        d = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        e = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        g = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        h = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

        def output():
            return torch.zeros(TASK_COUNT * TILE_ELEMS, dtype=torch.float32)

        return TaskArgsBuilder(
            TensorArg("a", a.flatten()),
            TensorArg("b", b.flatten()),
            TensorArg("c", output()),
            TensorArg("d", d),
            TensorArg("e", e),
            TensorArg("f", output()),
            TensorArg("g", g),
            TensorArg("h", h),
            TensorArg("i", output()),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("mask", ctypes.c_int64(params["mask"])),
        )

    def compute_golden(self, args, params):
        c = args.c.reshape(TASK_COUNT, TILE_ELEMS)
        f = args.f.reshape(TASK_COUNT, TILE_ELEMS)
        i = args.i.reshape(TASK_COUNT, TILE_ELEMS)
        for task in range(TASK_COUNT):
            mask = params["mask"]
            if params["graph_case"] == 2 and task % 2:
                mask = 1 if task % 4 == 1 else 2
            chained = params["graph_case"] == 0 and task != 0
            if mask & 1:
                a = c[task - 1] if chained else args.a
                c[task] = torch.matmul(
                    a.reshape(MATMUL_SIZE, MATMUL_SIZE), args.b.reshape(MATMUL_SIZE, MATMUL_SIZE)
                ).flatten()
            if mask & 2:
                f[task] = (f[task - 1] if chained else args.d) + args.e
            if mask & 4:
                i[task] = (i[task - 1] if chained else args.g) * args.h

    def test_run(self, st_platform, st_worker, request):
        level = dump_args_level(request.config.getoption("--dump-args", default="off"))
        before = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for folder in _outputs_dir().glob("TestSingleBlockMixHostBuildGraphA5_*")
            for path in folder.rglob("args_dump.json")
        }
        super().test_run(st_platform, st_worker, request)
        if not level:
            return
        for case in self._matching_cases(st_platform, request):
            label = _sanitize_for_filename(f"TestSingleBlockMixHostBuildGraphA5_{case['name']}")
            current = [
                path
                for folder in _outputs_dir().glob(f"{label}_*")
                for path in folder.rglob("args_dump.json")
                if before.get(path) != (path.stat().st_mtime_ns, path.read_bytes())
            ]
            assert len(current) == 1, current
            path = current[0]
            manifest = json.loads(path.read_text())
            args = self.generate_args(case["params"])
            self.compute_golden(args, case["params"])
            payload = b"" if manifest["bin_file"] is None else (path.parent / manifest["bin_file"]).read_bytes()
            _check_mix_payloads(manifest, payload, level, args, case["params"])


def _check_mix_payloads(manifest, payload, level, args, params):
    assert manifest["dump_args_level"] == level
    assert manifest["dropped_records_complete"] is True and manifest["dropped_records"] == 0
    assert not manifest.get("counts_unknown", False)
    assert not manifest.get("host_discarded_args", 0) and not manifest.get("metadata_discarded_records", 0)
    records = manifest["args"]
    # This graph has no explicit marks. PARTIAL selects no tasks, whereas
    # HYBRID emits every task's metadata without copying its payload.
    if level == 1:
        assert records == [] and manifest["total_args"] == 0 and payload == b""
        return
    assert len(records) == manifest["total_args"] == TASK_COUNT * 9
    epochs = {record["run_epoch"] for record in records}
    assert len(epochs) == 1 and all(type(epoch) is int and epoch > 0 for epoch in epochs)
    inputs = (args.a, args.b, None, args.d, args.e, None, args.g, args.h, None)
    outputs = {2: args.c, 5: args.f, 8: args.i}
    seen, intervals = set(), []
    for record in records:
        task, arg = int(record["task_id"], 16), record["arg_index"]
        assert 0 <= task < TASK_COUNT and 0 <= arg < 9
        key = task, arg
        assert key not in seen, key
        seen.add(key)
        mask = params["mask"]
        if params["graph_case"] == 2 and task % 2:
            mask = 1 if task % 4 == 1 else 2
        assert record["func_id"] == [lane for lane in range(3) if mask & (1 << lane)]
        output = arg in outputs
        assert record["stage"] == ("after_completion" if output else "before_dispatch")
        assert record["role"] == ("output" if output else "input")
        assert record["kind"] == "tensor" and record["dtype"] == "FLOAT32"
        assert record["shape"] == [TILE_ELEMS] and record["strides"] == [1]
        assert record["numel"] == TILE_ELEMS and record["is_contiguous"] is True
        assert not record.get("truncated", False)
        chained = params["graph_case"] == 0 and task > 0 and arg in (0, 3, 6) and mask & (1 << (arg // 3))
        expected_offset = task * TILE_ELEMS if output else (task - 1) * TILE_ELEMS if chained else 0
        assert record["start_offset"] == expected_offset
        size = record["bin_size"]
        assert size == (TILE_ELEMS * 4 if level == 3 else 0)
        if level != 3:
            continue
        start = record["bin_offset"]
        assert type(start) is int and 0 <= start <= len(payload) - size
        actual = torch.frombuffer(bytearray(payload[start : start + size]), dtype=torch.float32)
        if output:
            expected = outputs[arg][task * TILE_ELEMS : (task + 1) * TILE_ELEMS]
        elif chained:
            expected = outputs[arg + 2][(task - 1) * TILE_ELEMS : task * TILE_ELEMS]
        else:
            expected = inputs[arg]
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        intervals.append((start, start + size))
    assert seen == {(task, arg) for task in range(TASK_COUNT) for arg in range(9)}
    if level == 3:
        ordered = sorted(intervals)
        assert ordered[0][0] == 0 and ordered[-1][1] == len(payload)
        assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))
    else:
        assert payload == b""


@scene_test(level=2, runtime="host_build_graph")
class TestMixKernelRendezvous(_MixScene):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/rendezvous_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"RENDEZVOUS_{lane}",
                "source": f"kernels/rendezvous_{lane}.cpp",
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [D.INOUT],
            }
            for lane in range(3)
        ],
    }
    CASES = [
        {"name": f"burst256_mask{mask}", "platforms": ["a5sim", "a5"], "params": {"mask": mask, "task_count": 256}}
        for mask in [6, 7]
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros(params["task_count"] * 48, dtype=torch.int64)),
            Scalar("mask", ctypes.c_int64(params["mask"])),
            Scalar("task_count", ctypes.c_int64(params["task_count"])),
        )

    def compute_golden(self, args, params):
        for lane in range(3):
            if params["mask"] & (1 << lane):
                args.state.reshape(params["task_count"], 48)[:, lane * 16] = 2


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
