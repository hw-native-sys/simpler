#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One root fanning out to many independent consumers — a deliberately unbalanced graph.

Every consumer becomes ready inside one root-completion wake-list walk. The
ready pool must make that work globally reachable by eligible cores, regardless
of which Scheduler publishes it. Each consumer burns `repeats` iterations so
the backlog outlives the moment it is created.

This differs from `single_core_dag::mixed_fanout_63`, which is the same graph
shape, in the two things that make the imbalance observable: scale (up to 1024
consumers instead of 63) and per-task work. The shape alone is already covered
there; what is not covered anywhere is how widely the fanout actually spreads.

`TestFanoutImbalance` is the regression guard and doubles as a performance
baseline case. The spread assertions live in `test_fanout_spread_is_wide`
below, which turns the chip swimlane on itself rather than inheriting a
command-line flag — an assertion that evaporates when a flag is absent guards
nothing.
"""

import ctypes
import functools
import json
import os
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "host_build_graph"
ORCHESTRATION_SOURCE = os.path.join(HERE, "kernels", "orchestration", "fanout_imbalance_orch.cpp")
WORKER_SOURCE = os.path.join(HERE, "kernels", "fanout_worker.cpp")

# Enough iterations that a consumer is not finished before its siblings are even
# published, and few enough that 1024 of them stay well inside the op timeout.
REPEATS = 2048

# The spread assertion runs a smaller graph than the functional cases. Level-3
# swimlane records cost the Scheduler per task, and at 1024 tasks that cost is
# itself enough to trip the device scheduler timeout
# (SIMPLER_ERROR_SCHEDULER_TIMEOUT, surfaced as finalize code -100) — the
# profiling would destroy the very run it was added to observe. 256 consumers
# oversubscribe every core several times on any supported topology, which is
# all the distribution claim needs.
SPREAD_COUNT = 256

# Level 3 is SCHED_PHASES: the level that carries the per-Scheduler records the
# spread assertion reads.
SWIMLANE_SCHED_PHASES = 3


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestFanoutImbalance(SceneTestCase):
    """Functional guard: every consumer observes the root and runs exactly once."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/fanout_imbalance_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [ArgDirection.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "FANOUT_AIC",
                "source": "kernels/fanout_worker.cpp",
                "core_type": "aic",
                "signature": [ArgDirection.INOUT],
            },
            {
                "func_id": 1,
                "name": "FANOUT_AIV",
                "source": "kernels/fanout_worker.cpp",
                "core_type": "aiv",
                "signature": [ArgDirection.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": f"fanout_{count}",
            "platforms": ["a5sim", "a5"],
            "params": {"task_count": count},
            **({} if count <= 256 else {"manual": ["a5"]}),
        }
        for count in (256, 1024)
    ]

    def generate_args(self, params):
        count = params["task_count"]
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros(count * 8, dtype=torch.int64)),
            Scalar("task_count", ctypes.c_int64(count)),
            Scalar("repeats", ctypes.c_int64(REPEATS)),
        )

    def compute_golden(self, args, params):
        count = params["task_count"]
        # A negative slot means the consumer ran before the root published.
        args.state[: count * 8 : 8] = torch.arange(1, count + 1, dtype=torch.int64)


@functools.cache
def _build_callable(platform: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = list(compiler.get_orchestration_include_dirs(RUNTIME)) + [
        str(compiler.project_root / "src" / "common")
    ]

    def incore(core_type):
        binary = compiler.compile_incore(
            source_path=WORKER_SOURCE,
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        # Onboard loads raw code, so the ELF wrapper must come off first;
        # sim consumes the ELF as-is. scene_test.py:1852 makes the same split.
        if not platform.endswith("sim"):
            binary = extract_text_section(binary)
        return CoreCallable.build(signature=[ArgDirection.INOUT], binary=binary)

    orchestration = compiler.compile_orchestration(runtime_name=RUNTIME, source_path=ORCHESTRATION_SOURCE)
    return ChipCallable.build(
        signature=[ArgDirection.INOUT],
        func_name="aicpu_orchestration_entry",
        binary=orchestration,
        children=[(0, incore("aic")), (1, incore("aiv"))],
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_fanout_spread_is_wide(st_platform, st_device_ids, tmp_path):
    """A wide fanout reaches at least half of the participating cores."""
    count = SPREAD_COUNT
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    buffer = None
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()

        host_state = torch.zeros(count * 8, dtype=torch.int64)
        buffer = worker.malloc(host_state.nbytes)
        worker.copy_to(buffer, host_state)

        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(count * 8,), dtype=DataType.INT64), TensorArgType.INOUT)
        args.add_scalar(count)
        args.add_scalar(REPEATS)

        output_prefix = tmp_path / "fanout_spread"
        output_prefix.mkdir(parents=True, exist_ok=True)
        config = CallConfig()
        config.enable_chip_swimlane = SWIMLANE_SCHED_PHASES
        config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)

        worker.copy_from(host_state, buffer)
        expected = torch.arange(1, count + 1, dtype=torch.int64)
        assert torch.equal(host_state[: count * 8 : 8], expected), "fanout consumers did not all observe the root"

        records_path = _find_swimlane_records(output_prefix)
        raw = json.loads(records_path.read_text())

        _assert_fanout_spread(raw, count)
    finally:
        # A poisoned run still owns its device memory, so free() would raise and
        # bury whatever actually failed. Closing the worker releases it.
        if buffer is not None:
            try:
                worker.free(buffer)
            except RuntimeError:
                pass
        worker.close()


def _assert_fanout_spread(raw: dict, count: int) -> None:
    rows = raw["aicore_tasks"]
    assert len(rows) == count, f"expected {count} task records, got {len(rows)}"
    active = int(raw["metadata"]["num_cores"])
    assert active > 0, f"swimlane metadata reports no cores: {raw['metadata']}"
    cores = {int(row[0]) for row in rows}
    # The graph oversubscribes every supported topology. This floor detects
    # concentration in a publishing cluster or directory shard without requiring
    # a particular queue topology or a worksteal event for global claims.
    minimum = min(count, (active + 1) // 2)
    assert len(cores) >= minimum, (
        f"fanout reached only {len(cores)} of {active} cores (floor {minimum}): {sorted(cores)}"
    )
    streams = raw["scheduler_records"]["streams"]
    assert streams, "no Scheduler records"
    kinds = {record["kind"] for stream in streams for record in stream["records"]}
    assert {"dispatch", "complete"} <= kinds, f"missing dispatch/completion records: {sorted(kinds)}"


def _find_swimlane_records(output_prefix: Path) -> Path:
    matches = sorted(output_prefix.rglob("chip_swimlane_records.json"))
    assert matches, f"chip swimlane produced no records under {output_prefix}"
    return matches[-1]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
