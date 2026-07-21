# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST asserting simpler_run emits ``[STRACE]`` host-trace markers.

``Worker.run`` no longer returns a RunTiming — per-stage timing is emitted by
the platform as ``[STRACE]`` log markers (see docs/dfx/host-trace.md). Reuses
the vector_add example's kernel + ChipCallable build so this test doesn't drag
in its own kernel sources. The contract being verified:

    * a real dispatch emits an ``[STRACE] ... name=simpler_run ... dur=<ns>``
      marker (the host wall around the dispatch) with a strictly positive
      duration — there's no way a real run took zero steady-clock time;
    * the device-domain wall marker
      (``name=simpler_run.runner_run.device_wall``) is present with a positive
      duration on the default SIMPLER_HOST_STRACE build.

Markers go to stderr via the unified host logger, captured here with ``capfd``.
"""

import re

import pytest
from simpler.task_interface import CallConfig, ChipStorageTaskArgs, DataType, Tensor
from simpler.worker import Worker

from .main import N_COLS, N_ELEMS, N_ROWS, NBYTES, build_chip_callable

_STRACE_RE = re.compile(r"\[STRACE\] .*\bname=(?P<name>\S+)\b.*\bdur=(?P<dur>\d+)")


def _strace_durs(captured: str, name: str) -> list:
    """Return the dur (ns) of every [STRACE] marker with the given span name."""
    out = []
    for line in captured.splitlines():
        m = _STRACE_RE.search(line)
        if m and m["name"] == name:
            out.append(int(m["dur"]))
    return out


def _drive_one_run(platform: str, device_id: int, *, enable_l2_swimlane: bool = False):
    import torch  # noqa: PLC0415

    worker = Worker(
        level=2,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_id=device_id,
    )
    chip_callable = build_chip_callable(platform)
    chip_handle = worker.register(chip_callable)
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
        args.add_tensor(Tensor.make(dev_a, (N_ROWS, N_COLS), DataType.FLOAT32))
        args.add_tensor(Tensor.make(dev_b, (N_ROWS, N_COLS), DataType.FLOAT32))
        args.add_tensor(Tensor.make(dev_out, (N_ROWS, N_COLS), DataType.FLOAT32))

        config = CallConfig()
        config.enable_l2_swimlane = enable_l2_swimlane

        # run() returns None; timing is emitted as [STRACE] markers on stderr.
        assert worker.run(chip_handle, args, config) is None

        # Verify the output is sane (so we know the kernel actually ran and
        # the markers aren't from an early-error path).
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
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_simpler_run_emits_strace_markers(st_platform, st_device_ids, capfd):
    _drive_one_run(st_platform, int(st_device_ids[0]))
    err = capfd.readouterr().err

    # Host wall: the simpler_run root span. A real dispatch can't take 0 ns.
    host_durs = _strace_durs(err, "simpler_run")
    assert host_durs, (
        "no `[STRACE] ... name=simpler_run` marker found on stderr; "
        "simpler_run stopped emitting host-trace markers (SIMPLER_HOST_STRACE off, "
        "or the host logger V9 tier was suppressed)."
    )
    assert max(host_durs) > 0, f"simpler_run marker dur must be > 0 ns, got {host_durs}"

    # Device-domain wall: the on-NPU AICPU wall, emitted after readback.
    dev_durs = _strace_durs(err, "simpler_run.runner_run.device_wall")
    assert dev_durs, (
        "no device_wall [STRACE] marker found; the AICPU phase buffer readback "
        "or marker emission regressed on the default SIMPLER_HOST_STRACE build."
    )
    assert max(dev_durs) > 0, f"device_wall marker dur must be > 0 ns, got {dev_durs}"
