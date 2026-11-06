# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual native TMR cold replay publishes PMU samples and honest aggregate quality."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import CallConfig, _binaries, _CallerContexts, _load, _reset_caller_device
from test_tmr_native_capture import _bind, _Model, _Recording

from simpler_setup.tools.pmu_samples import read_pmu_samples

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = "tensormap_and_ringbuffer"
_EPOCH = 810
pytestmark = pytest.mark.runtime(_RUNTIME)


def _flush(lib, ctx):
    message = ctypes.create_string_buffer(4096)
    rc = lib.simpler_flush_diagnostics_ctx(ctx, 60000, message, len(message))
    assert rc == 0, (rc, message.value.decode(errors="replace"))


def _run(device, root, replays):
    chip = _callable("a2a3", _RUNTIME)
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib = _load("a2a3", "onboard", _RUNTIME)
    _bind(lib)
    lib.simpler_wait_run.argtypes = [ctypes.c_void_p] * 2
    lib.simpler_flush_diagnostics_ctx.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t]
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    ctx, caller = lib.create_device_context(), ctypes.c_void_p()
    assert ctx
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    cpu, core, dispatcher = _binaries("a2a3", _RUNTIME)
    config = CallConfig(aicpu_thread_num=2, enable_pmu=2, output_prefix=str(root).encode())
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, cpu, len(cpu), core, len(core), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    prepare_rc = lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image))
    assert prepare_rc == 0, f"prepare_callable failed with rc={prepare_rc}"
    model = _Model(lib, ctx, caller, config, (_EPOCH,))
    assert not (root / "pmu.collection.json").exists(), "capture published execution samples"
    assert lib.simpler_unregister_callable(ctx, 0) != 0
    for value in (2, 7, 19)[:replays]:
        model.replay(value)
    model.close()
    _flush(lib, ctx)
    recorded = read_pmu_samples(root / "pmu.csv", recorded=True)
    assert recorded["collection"]["event_type"] == 2
    assert recorded["collection"]["run_epoch"] == _EPOCH
    assert all(int(row["pmu_total_cycles"]) > 0 for row in recorded["records"]), recorded
    if replays:
        counts = Counter(row["task_id"] for row in recorded["records"])
        assert max(counts.values()) >= replays, counts
    else:
        assert not recorded["records"], recorded

    eager_root = root / "eager"
    eager_root.mkdir()
    config.output_prefix = str(eager_root).encode()
    config.enable_pmu = 4
    eager = _Recording(lib, ctx, caller, config, _EPOCH + 1, 0, capture=False)
    eager.set_inputs(23)
    eager.record()
    assert lib.aclrtSynchronizeStreamWithTimeout(caller, 60000) == 0
    eager.check_output(23)
    assert lib.simpler_wait_run(ctx, eager.run) == 0
    eager.retire_original_run()
    eager.close()
    _flush(lib, ctx)
    ordinary = read_pmu_samples(eager_root / "pmu.csv")
    assert ordinary["records"] and {int(row["event_type"]) for row in ordinary["records"]} == {4}, ordinary
    assert all(int(row["pmu_total_cycles"]) > 0 for row in ordinary["records"]), ordinary
    # This fixed fixture executes the same DAG each time; the assertion does not
    # turn fixed-window PMU counters into a general completeness guarantee.
    assert len(recorded["records"]) == replays * len(ordinary["records"]), (recorded, ordinary)
    assert len({row["task_id"] for row in ordinary["records"]}) == 5
    assert Counter(int(row["func_id"]) for row in ordinary["records"]) == Counter({0: 2, 1: 2, 2: 1})
    assert Counter((row["task_id"], row["func_id"]) for row in recorded["records"]) == Counter(
        {
            key: count * replays
            for key, count in Counter((row["task_id"], row["func_id"]) for row in ordinary["records"]).items()
        }
    )
    assert lib.simpler_unregister_callable(ctx, 0) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    assert lib.aclrtDestroyStream(caller) == 0
    contexts.close()
    _reset_caller_device(lib, device)
    print(f"TMR_PMU_CAPTURE_PASS replays={replays} samples={len(recorded['records'])}", flush=True)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.parametrize("replays", [0, 3])
def test_tmr_pmu_capture_exports_actual_replay_samples(replays, tmp_path, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), device, str(tmp_path), str(replays)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    (tmp_path / "native.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TMR_PMU_CAPTURE_PASS replays={replays}" in result.stdout


if __name__ == "__main__":
    _run(int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
