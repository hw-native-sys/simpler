# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real native TMR replay exports unpaired Swimlane observations."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import CallConfig, _binaries, _CallerContexts, _load, _reset_caller_device
from test_tmr_native_capture import _bind, _Model, _Recording

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = "tensormap_and_ringbuffer"
_EPOCH = 910
pytestmark = pytest.mark.runtime(_RUNTIME)


def _flush(lib, ctx):
    message = ctypes.create_string_buffer(4096)
    rc = lib.simpler_flush_diagnostics_ctx(ctx, 60000, message, len(message))
    assert rc == 0, (rc, message.value.decode(errors="replace"))


def _check_trace(path, raw):
    before = path.read_bytes()
    output = path.with_name("observations.json")
    converted = subprocess.run(
        [sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(path), "-o", str(output)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (path.parent / "converter.log").write_text(converted.stdout + converted.stderr)
    assert converted.returncode == 0, converted.stdout + converted.stderr
    assert path.read_bytes() == before
    trace = json.loads(output.read_text())
    assert trace["metadata"]["clock_domain"] == "device_syscnt_cycles"
    assert trace["metadata"]["clock_alignment"]["status"] == "unavailable"
    assert trace["metadata"]["collection"] == raw["metadata"]["collection"]
    events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert all(event["ph"] in ("M", "X") for event in trace["traceEvents"])
    for population, records in (
        ("aicore_tasks", raw["aicore_tasks"]),
        ("scheduler_tasks", raw.get("scheduler_tasks", {}).get("records", [])),
    ):
        observed = [event for event in events if event["args"]["population"] == population]
        assert len(observed) == len(records)
        assert [event["args"]["record_index"] for event in observed] == list(range(len(records)))
    if "host_preparation_file" in trace["metadata"]:
        host = json.loads((path.parent / trace["metadata"]["host_preparation_file"]).read_text())
        assert host["metadata"]["clock_domain"] == "host_monotonic_ns"
        assert host["metadata"]["preparation_only"] is True


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
    config = CallConfig(aicpu_thread_num=2, enable_chip_swimlane=4, output_prefix=str(root).encode())
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, cpu, len(cpu), core, len(core), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    prepare_rc = lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image))
    assert prepare_rc == 0, f"prepare_callable failed with rc={prepare_rc}"
    model = _Model(lib, ctx, caller, config, (_EPOCH,))
    assert not list(root.glob("swimlane-*/records_e*.json")), "capture published execution samples"
    assert lib.simpler_unregister_callable(ctx, 0) != 0
    # Mutating this source config cannot replace the model's frozen selection/destination.
    config.enable_chip_swimlane = 1
    config.output_prefix = str(root / "unused").encode()
    for value in (2, 7, 19)[:replays]:
        model.replay(value)
    model.close()
    _flush(lib, ctx)
    [path] = root.glob("swimlane-*/records_e*.json")
    raw = json.loads(path.read_text())
    collection = raw["metadata"]["collection"]
    assert raw["chip_swimlane_level"] == 4
    assert collection["recorded_window"] is True and collection["counts_unknown"] is True
    assert collection["processing_complete"] and collection["metadata_complete"], collection
    assert all(collection[key] is None for key in ("device_total", "device_dropped", "fatal", "reconciliation"))
    assert collection["run_epoch"] == _EPOCH
    assert not (root / "unused").exists()
    _check_trace(path, raw)

    eager_root = root / "eager"
    config.output_prefix = str(eager_root).encode()
    config.enable_chip_swimlane = 4
    eager = _Recording(lib, ctx, caller, config, _EPOCH + 1, 0, capture=False)
    eager.set_inputs(23)
    eager.record()
    assert lib.aclrtSynchronizeStreamWithTimeout(caller, 60000) == 0
    eager.check_output(23)
    assert lib.simpler_wait_run(ctx, eager.run) == 0
    eager.retire_original_run()
    eager.close()
    _flush(lib, ctx)
    ordinary = json.loads((eager_root / "chip_swimlane_records.json").read_text())
    assert not ordinary["metadata"].get("collection", {}).get("recorded_window")
    assert ordinary["aicore_tasks"], ordinary
    # Counts are a property of this fixed DAG, not a runtime completeness denominator.
    assert len(raw["aicore_tasks"]) == replays * len(ordinary["aicore_tasks"])
    assert Counter(row[1] for row in raw["aicore_tasks"]) == +Counter(
        {key: count * replays for key, count in Counter(row[1] for row in ordinary["aicore_tasks"]).items()}
    )
    assert lib.simpler_unregister_callable(ctx, 0) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    assert lib.aclrtDestroyStream(caller) == 0
    contexts.close()
    _reset_caller_device(lib, device)
    print(f"TMR_SWIMLANE_CAPTURE_PASS replays={replays} samples={len(raw['aicore_tasks'])}", flush=True)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.parametrize("replays", [0, 3])
def test_tmr_swimlane_capture_exports_actual_replay_samples(replays, tmp_path, request):
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
    assert f"TMR_SWIMLANE_CAPTURE_PASS replays={replays}" in result.stdout


if __name__ == "__main__":
    _run(int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
