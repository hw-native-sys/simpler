# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST asserting Worker.run returns a usable RunTiming.

Reuses the vector_add example's kernel + ChipCallable build so this test
doesn't drag in its own kernel sources. The contract being verified:

    * `worker.run(...)` returns a `RunTiming` instance (not None).
    * `host_wall_us` is strictly positive — there's no way a real dispatch
      took zero steady-clock time, so this would catch a regression where
      the C ABI stopped writing the timing back to the out-param.
    * `device_wall_us` is non-negative. When the runtime was built with
      PTO2_PROFILING AND the call had enable_l2_swimlane=1 the value is
      typically > 0; without those, the contract is "0, not garbage".

Why not assert device_wall > 0 unconditionally: the default sim build does
not have PTO2_PROFILING wired through the L2 perf shared region for every
sim variant, so requiring it here would make the test flaky across
platforms. The dedicated assertion lives in the swimlane-enabled benchmark
flow (`tools/benchmark_rounds.sh --enable-l2-swimlane`).
"""

import pytest
from _task_interface import RunTiming  # pyright: ignore[reportMissingImports]
from simpler.task_interface import CallConfig, ChipStorageTaskArgs, ContinuousTensor, DataType
from simpler.worker import Worker

from .main import N_COLS, N_ELEMS, N_ROWS, NBYTES, build_chip_callable


def _drive_one_run(platform: str, device_id: int, *, enable_l2_swimlane: bool = False):
    import torch  # noqa: PLC0415

    worker = Worker(
        level=2,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_id=device_id,
    )
    chip_callable = build_chip_callable(platform)
    chip_cid = worker.register(chip_callable)
    worker.init()
    try:
        # Use deterministic inputs so the run never accidentally hits a
        # degenerate kernel path that fails to publish a perf record.
        host_a = torch.full((N_ROWS, N_COLS), 1.0, dtype=torch.float32)
        host_b = torch.full((N_ROWS, N_COLS), 2.0, dtype=torch.float32)

        dev_a = worker.malloc(NBYTES)
        dev_b = worker.malloc(NBYTES)
        dev_out = worker.malloc(NBYTES)
        worker.copy_to(dev_a, host_a.data_ptr(), NBYTES)
        worker.copy_to(dev_b, host_b.data_ptr(), NBYTES)

        args = ChipStorageTaskArgs()
        args.add_tensor(ContinuousTensor.make(dev_a, (N_ROWS, N_COLS), DataType.FLOAT32))
        args.add_tensor(ContinuousTensor.make(dev_b, (N_ROWS, N_COLS), DataType.FLOAT32))
        args.add_tensor(ContinuousTensor.make(dev_out, (N_ROWS, N_COLS), DataType.FLOAT32))

        config = CallConfig()
        config.enable_l2_swimlane = enable_l2_swimlane

        timing = worker.run(chip_cid, args, config)

        # Verify the output is sane (so we know the kernel actually ran and
        # the timing isn't from an early-error path).
        host_out = torch.zeros(N_ROWS, N_COLS, dtype=torch.float32)
        worker.copy_from(host_out.data_ptr(), dev_out, NBYTES)
        worker.free(dev_a)
        worker.free(dev_b)
        worker.free(dev_out)
        assert torch.allclose(host_out, host_a + host_b, rtol=1e-5, atol=1e-5), (
            f"vector_add kernel output diverged; max |a+b - out| = "
            f"{float(torch.max(torch.abs(host_out - (host_a + host_b)))):.3e} "
            f"(N_ELEMS={N_ELEMS})"
        )
        return timing
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_worker_run_returns_run_timing(st_platform, st_device_ids):
    timing = _drive_one_run(st_platform, int(st_device_ids[0]))

    assert isinstance(timing, RunTiming), (
        f"Worker.run must return a RunTiming (got {type(timing).__name__}); "
        f"a None return means the ChipWorker Python wrapper dropped the C++ return value."
    )
    # Host wall is measured with steady_clock around the dispatch; even on
    # sim it covers thread + IPC overhead so the only way to see 0 is a
    # bug in the out-param plumbing.
    assert timing.host_wall_us > 0.0, f"host_wall_us must be > 0, got {timing.host_wall_us}"
    assert timing.host_wall_ns > 0, f"host_wall_ns must be > 0, got {timing.host_wall_ns}"
    # device_wall is allowed to be 0 (see module docstring) but must never
    # be negative — it's a uint64 in C++ so this would only fail on
    # serialization corruption.
    assert timing.device_wall_us >= 0.0
    assert timing.device_wall_ns >= 0
