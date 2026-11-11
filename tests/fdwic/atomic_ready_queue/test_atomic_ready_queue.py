# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Onboard ticket primitives: explicit publication gates and exact consumption."""

import copy
import functools
import json
import operator
import tempfile
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec

MAX_WORKERS = 96
PER_PRODUCER = 16
MAX_TASKS = MAX_WORKERS // 3 * PER_PRODUCER
STATE_LINES = 6 + MAX_TASKS + 1 + 2 + 6 * MAX_WORKERS + 12 + 2 * MAX_TASKS
STATE_BYTES = STATE_LINES * 64
RECORD_WORDS = 24


def validate_records(records, params):
    workers = params["blocks"] * (3 if params["mask"] == 7 else 1)
    producers = workers // 3
    tasks = producers * PER_PRODUCER
    rows = records.view(params["epochs"], workers + 1, RECORD_WORDS)
    expected_sum = tasks * (tasks - 1) // 2
    expected_xor = functools.reduce(operator.xor, range(tasks), 0)
    for epoch_index, epoch_rows in enumerate(rows):
        epoch = epoch_index + 1
        assert torch.all(epoch_rows[:, 0] == 1).item(), "missing or duplicate invocation/summary"
        assert torch.count_nonzero(epoch_rows[:, 1:8]).item() == 0, "atomic record guard overwritten"
        values = epoch_rows[:, 8:]
        members, summary = values[:workers], values[workers]
        assert torch.count_nonzero(members[:, 0]).item() == 0, f"device failure codes: {members[:, 0].tolist()}"
        assert members[:, 1].tolist() == list(range(workers)), "wrong worker rank"
        assert torch.all(members[:, 2] == epoch).item(), "stale invocation epoch"
        assert torch.count_nonzero(members[:producers, 3:6]).item() == 0, "producer consumed stress entries"
        assert members[:, 3].sum().item() == tasks, "consumed count mismatch"
        assert members[:, 4].sum().item() == expected_sum, "consumed reference sum mismatch"
        assert functools.reduce(operator.xor, members[:, 5].tolist(), 0) == expected_xor, "reference XOR mismatch"
        expected_holes = [int(rank == producers) for rank in range(workers)]
        assert members[:, 6].tolist() == expected_holes, "publication-hole proof missing or duplicated"
        assert members[:, 7].tolist() == [6 * value for value in expected_holes], "hole FIFO sum mismatch"
        assert torch.count_nonzero(members[:, 8]).item() == 0, "hole FIFO XOR mismatch"
        assert members[:, 9].sum().item() == 1, "last entry has missing/duplicate winner"
        expected_lanes = [
            rank % 3 if params["mask"] == 7 else (0 if params["mask"] == 1 else 1) for rank in range(workers)
        ]
        assert members[:, 11].tolist() == expected_lanes, "wrong callable lane"
        cores = members[:, 10].tolist()
        assert len(expected_lanes) == len(cores), "lane and core counts differ"
        physical = {(int(lane == 0), core) for lane, core in zip(expected_lanes, cores)}
        assert len(physical) == workers, "physical execution positions were reused concurrently"
        assert torch.all(members[:, 12] > 0).item(), "missing entry clock"
        assert torch.all(members[:, 13] >= members[:, 12]).item(), "negative invocation interval"
        assert members[:, 12].max().item() < members[:, 13].min().item(), (
            "all-member barrier did not overlap invocations"
        )
        assert torch.all(members[:, 14] == params["mask"]).item(), "wrong task mask"
        assert torch.all(members[:, 15] == workers).item(), "wrong participant count"
        expected_summary = [0, epoch, tasks, tasks, tasks, tasks, 4, 4, 1, 2, 15, 64, 128, STATE_BYTES, -1, workers]
        assert summary.tolist() == expected_summary, f"queue/guard/summary mismatch: {summary.tolist()}"
    return {
        "workers": workers,
        "producers": producers,
        "tasks_per_epoch": tasks,
        "epochs": params["epochs"],
        "sum_per_epoch": expected_sum,
        "xor_per_epoch": expected_xor,
    }


@scene_test(level=2, runtime="host_build_graph")
class TestAtomicReadyQueue(SceneTestCase):
    RTOL = ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"atomic_queue_{name}",
                "source": f"kernels/{name}.cpp",
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [D.INOUT, D.INOUT],
            }
            for lane, name in enumerate(("aic", "aiv0", "aiv1"))
        ],
    }
    CASES = [
        {
            "name": f"mask{mask}_blocks{blocks}_reuse3",
            "platforms": ["a5"],
            "params": {"blocks": blocks, "mask": mask, "epochs": 3},
        }
        for mask, blocks in ((1, 8), (2, 16), (7, 8), (7, 32))
    ]

    @classmethod
    def compile_chip_callable(cls, platform):
        assert platform == "a5", "this acceptance probe requires real A5 hardware"
        compiler = KernelCompiler(platform=platform)
        spec = copy.deepcopy(cls.CALLABLE)
        for kernel in spec["incores"]:
            kernel["extra_include_dirs"] = [str(compiler.platform_dir / "onboard" / "aicore")]
        key = (cls.__module__, cls.__qualname__, platform, cls._st_runtime, _pto_isa_compile_cache_token())
        return compile_chip_callable_spec(spec, platform, cls._st_runtime, key)

    def generate_args(self, params):
        workers = params["blocks"] * (3 if params["mask"] == 7 else 1)
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros(STATE_BYTES // 8, dtype=torch.int64)),
            TensorArg("records", torch.zeros(params["epochs"] * (workers + 1) * RECORD_WORDS, dtype=torch.int64)),
            *(Scalar(name, params[name]) for name in ("blocks", "mask", "epochs")),
        )

    def compute_golden(self, args, params):
        # Queue publication order and the winning consumer vary; validation checks
        # per-reference ownership plus exact aggregates in device-produced records.
        pass

    def _run_and_validate(self, worker, *args, **kwargs):
        case = kwargs.get("case", args[1] if len(args) > 1 else {})
        assert not kwargs.get("skip_golden", False), "the queue probe requires exact validation"
        assert not case.get("skip_golden", self.SKIP_GOLDEN), "the queue probe requires exact validation"
        self._probe_worker = worker
        self._probe_publish_count = worker.scheduler_mode_publish_count
        return super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        count = self._probe_worker.scheduler_mode_publish_count
        mode = self._probe_worker.scheduler_runtime_mode
        assert count > self._probe_publish_count and mode in (1, 2), "fresh AICore resident invocation required"
        self._probe_publish_count = count
        summary: dict[str, object] = dict(mask=params["mask"], blocks=params["blocks"], mode=mode, publish_count=count)
        output = Path("outputs/atomic_ready_queue")
        output.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"mask{params['mask']}_blocks{params['blocks']}_", dir=output))
        for name in ("state", "records"):
            raw = getattr(test_args, name).detach().cpu().contiguous().numpy().tobytes()
            (output / f"{name}.i64.bin").write_bytes(raw)
        summary["artifact"] = str(output)
        try:
            summary.update(validate_records(test_args.records, params))
        except AssertionError as failure:
            summary.update(status="FAIL", error=str(failure))
            raise
        else:
            summary["status"] = "PASS"
        finally:
            (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("[ATOMIC_READY_QUEUE] " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
