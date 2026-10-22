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
    if level == 0:
        assert not artifact.exists()
        return
    data = json.loads(artifact.read_text())
    assert data["chip_swimlane_level"] == level
    assert data["aicore_tasks"]
    if level >= 3:
        assert any(data["scheduler_records"]), (level, data["scheduler_records"])
    else:
        assert "scheduler_records" not in data


def _check_pmu(folder, level):
    artifact = folder / "pmu.csv"
    if level == 0:
        assert not artifact.exists()
        return
    with artifact.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        assert rows, (level, artifact)
        expected = int(os.environ.get("SIMPLER_PMU_EVENT_TYPE", str(level)))
        assert {int(row["event_type"]) for row in rows} == {expected}
        assert all(int(row["pmu_total_cycles"]) > 0 for row in rows), (level, rows)
        assert reader.fieldnames and "run_epoch" in reader.fieldnames


def _check_args_dump(folder, level):
    artifact = folder / "args_dump" / "args_dump.json"
    if level in (0, 1):
        # The vector callable has no dump-marked arguments.
        assert not artifact.exists()
        return
    data = json.loads(artifact.read_text())
    assert data["dump_args_level"] == level
    assert data["total_args"] > 0
    if level == 2:
        assert data["bin_file"] is None
    else:
        assert (artifact.parent / data["bin_file"]).stat().st_size > 0


def _check_scope_stats(folder, enabled, *, producer_expected=True):
    artifact = folder / "scope_stats" / "scope_stats.jsonl"
    if not enabled:
        assert not artifact.exists()
        return
    metadata, *records = [json.loads(line) for line in artifact.read_text().splitlines()]
    assert metadata["fatal"] is False, metadata
    assert metadata["dropped"] == 0, metadata
    assert metadata["total"] == len(records), (metadata, records)
    if not producer_expected:
        assert not records, records
        return
    assert len(records) >= 4, records
    epochs = {record["run_epoch"] for record in records}
    assert len(epochs) == 1 and 0 not in epochs, epochs
    return next(iter(epochs))


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
        worker = Worker(level=2, device_id=device, platform=arch, runtime=runtime)
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
        selections = {
            "pmu": (2, 4, 0, 1, 2),
            "swimlane": (4, 1, 0, 3, 2),
            "args_dump": (3, 1, 0, 2, 3),
            "scope_stats": (True, False, True, True),
        }
        levels = selections[diagnostic]
        scope_epochs = set()
        for invocation, level in enumerate(levels):
            folder = root / str(invocation)
            folder.mkdir()
            config = CallConfig()
            field = {
                "pmu": "enable_pmu",
                "swimlane": "enable_chip_swimlane",
                "args_dump": "enable_dump_args",
                "scope_stats": "enable_scope_stats",
            }
            setattr(config, field[diagnostic], level)
            config.output_prefix = str(folder)
            worker.submit(handle, args, config, caller_stream=stream.value).wait()
            result = array.array("f", [0.0]) * count
            worker.copy_from(result, buffers[2])
            expected = 47.0 if runtime == "tensormap_and_ringbuffer" else 42.0
            assert all(value == expected for value in result), (level, result[:8])
            checker = {
                "pmu": _check_pmu,
                "swimlane": _check_swimlane,
                "args_dump": _check_args_dump,
            }
            if diagnostic == "scope_stats":
                epoch = _check_scope_stats(folder, level, producer_expected=runtime == "tensormap_and_ringbuffer")
                if epoch is not None:
                    assert epoch not in scope_epochs, (invocation, epoch, scope_epochs)
                    scope_epochs.add(epoch)
            else:
                checker[diagnostic](folder, level)
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


@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch, runtime in (
            ("a2a3", "host_build_graph"),
            ("a2a3", "tensormap_and_ringbuffer"),
            ("a5", "tensormap_and_ringbuffer"),
        )
    ],
)
def test_worker_uses_each_invocations_args_dump_selection(arch, runtime, tmp_path, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, "1", str(tmp_path), "args_dump"],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.requires_hardware
@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=pytest.mark.platforms([arch]))
        for arch in ("a2a3", "a5")
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    ],
)
@pytest.mark.parametrize("borrowed", [False, True])
def test_worker_collects_scope_stats_each_invocation(arch, runtime, borrowed, tmp_path, request):
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
            "scope_stats",
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(int(sys.argv[4])), Path(sys.argv[5]), sys.argv[6])
