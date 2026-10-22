# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public DFX selections follow each invocation through owned and borrowed execution."""

import array
import csv
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import _CallerContexts, _load


def _check_swimlane(folder, level):
    artifact = folder / "chip_swimlane_records.json"
    data = json.loads(artifact.read_text())
    assert data["chip_swimlane_level"] == level
    assert data["aicore_tasks"]
    if level >= 3:
        assert any(data["scheduler_records"]), (level, data["scheduler_records"])
    else:
        assert "scheduler_records" not in data


def _check_pmu(folder, level):
    artifact = folder / "pmu.csv"
    with artifact.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        assert rows, (level, artifact)
        expected = int(os.environ.get("SIMPLER_PMU_EVENT_TYPE", str(level)))
        assert {int(row["event_type"]) for row in rows} == {expected}
        assert all(int(row["pmu_total_cycles"]) > 0 for row in rows), (level, rows)
        assert reader.fieldnames and "run_epoch" in reader.fieldnames


def _run(arch, runtime, device, borrowed, root, diagnostic):
    from simpler.task_interface import (  # noqa: PLC0415
        CallConfig,
        DataType,
        TaskArgs,
        Tensor,
        TensorArgType,
    )
    from simpler.worker import Worker  # noqa: PLC0415

    lib = _load(arch, "onboard", runtime)
    contexts = None
    worker = None
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    stream = ctypes.c_void_p()
    buffers = []
    try:
        if borrowed:
            lib.rtSetDevice.argtypes = [ctypes.c_int]
            assert lib.rtSetDevice(device) == 0
            contexts = _CallerContexts(lib, device)
        worker = Worker(level=2, device_id=device, platform=arch, runtime=runtime, output_root=root)
        handle = worker.register(_callable(arch, runtime))
        worker.init(borrowed_context_generation=77 if borrowed else None)
        assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        count = 128 * 128
        for _ in range(3):
            buffers.append(worker.malloc(count * 4))
        worker.copy_to(buffers[0], array.array("f", [2.0]) * count)
        worker.copy_to(buffers[1], array.array("f", [3.0]) * count)
        args = TaskArgs()
        for i, backing in enumerate(buffers):
            args.add_tensor(
                Tensor(backing, shapes=(count,), dtype=DataType.FLOAT32),
                TensorArgType.INPUT if i < 2 else TensorArgType.OUTPUT_EXISTING,
            )
        levels = (2, 4, 0, 1, 2) if diagnostic == "pmu" else (4, 1, 0, 3, 2)
        check_artifact = _check_pmu if diagnostic == "pmu" else _check_swimlane
        for level in levels:
            config = CallConfig()
            if diagnostic == "pmu":
                config.enable_pmu = level
            else:
                config.enable_chip_swimlane = level
            worker.submit(handle, args, config, caller_stream=stream.value).wait()
            capture = worker.capture_output_prefix
            result = array.array("f", [0.0]) * count
            worker.copy_from(result, buffers[2])
            expected = 47.0 if runtime == "tensormap_and_ringbuffer" else 42.0
            assert all(value == expected for value in result), (level, result[:8])
            if level == 0:
                assert capture is None
                continue
            assert capture is not None
            folder = Path(capture)
            check_artifact(folder, level)
    finally:
        # The caller stream may still hold a wait after the kernel boundaries.
        if stream.value:
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
        if worker is not None:
            for backing in buffers:
                worker.free(backing)
        if stream.value:
            assert lib.aclrtDestroyStream(stream) == 0
        if worker is not None:
            worker.close()
        if contexts is not None:
            contexts.close()


@pytest.mark.parametrize(("diagnostic", "event_override"), [("swimlane", None), ("pmu", None), ("pmu", 4)])
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
def test_worker_uses_each_invocations_selection(
    arch, runtime, borrowed, diagnostic, event_override, tmp_path, request, monkeypatch
):
    if event_override is None:
        monkeypatch.delenv("SIMPLER_PMU_EVENT_TYPE", raising=False)
    else:
        monkeypatch.setenv("SIMPLER_PMU_EVENT_TYPE", str(event_override))
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            arch,
            runtime,
            device,
            str(int(borrowed)),
            str(tmp_path),
            diagnostic,
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])), Path(sys.argv[5]), sys.argv[6])
