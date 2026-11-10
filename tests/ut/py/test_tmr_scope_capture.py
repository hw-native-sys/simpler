# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real TMR replay samples survive source retirement and export at model release."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_kernel_caller_stream import _callable
from test_kernel_mode_c_api import CallConfig, _binaries, _CallerContexts, _load, _reset_caller_device
from test_tmr_native_capture import _bind, _Model, _Recording

from simpler_setup.tools.tmr.scope_stats_plot import process

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = "tensormap_and_ringbuffer"
_EPOCH = 710
pytestmark = pytest.mark.runtime(_RUNTIME)


def _expected_boundaries():
    source = (
        _ROOT / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/orchestration/example_orchestration.cpp"
    )
    [line] = [
        i for i, text in enumerate(source.read_text().splitlines(), 1) if text.strip().startswith("SIMPLER_SCOPE()")
    ]
    site = f"{source.name[:31]}:{line}"
    return [("(unknown):0", "begin", 0), (site, "begin", 1), (site, "end", 1), ("(unknown):0", "end", 0)]


def _check_export(path, *, recorded, replays, epoch):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    metadata, records = rows[0], rows[1:]
    if recorded:
        assert metadata["recorded_window"] is True, metadata
        assert metadata["counts_unknown"] is True, metadata
        assert metadata["collection_verdict"] == "partial_cut_unknown", metadata
        assert all(metadata[key] is None for key in ("fatal", "dropped", "total")), metadata
        assert metadata["host_received_records"] == metadata["host_retained_records"] == len(records), metadata
    else:
        assert not metadata.get("recorded_window"), metadata
        assert metadata["fatal"] is False and metadata["dropped"] == 0, metadata
        assert metadata["total"] == len(records), metadata
    assert len(records) == 4 * replays, (metadata, records)
    assert [
        (record["site"], record["phase"], record["depth"]) for record in records
    ] == _expected_boundaries() * replays
    assert {record["run_epoch"] for record in records} == ({epoch} if replays else set()), records
    fields = (
        "task_window_start",
        "task_window_end",
        "heap_start",
        "heap_end",
        "dep_pool_start",
        "dep_pool_end",
        "tensormap",
        "buf_seq",
    )
    assert all(all(isinstance(record[field], int) and record[field] >= 0 for field in fields) for record in records)
    for resource in ("task_window", "heap", "dep_pool"):
        assert len(metadata[f"{resource}_max"]) == 4, metadata
        assert all(value > 0 for value in metadata[f"{resource}_max"]), metadata
    assert metadata["tensormap_max"] > 0, metadata
    if replays:
        ends = [record for record in records if record["phase"] == "end"]
        assert any(record["task_window_end"] > 0 for record in ends), records
        assert any(record["heap_end"] > 0 for record in ends), records
    [report] = process(path)
    html = report.read_text()
    if recorded:
        assert "Raw boundary samples" in html and "total paired scopes" not in html
    else:
        assert "2 total paired scopes" in html
    return metadata


def _fixture_export(path, *, recorded, replays):
    records = []
    for _ in range(replays):
        for site, phase, depth in _expected_boundaries():
            advanced = int(phase == "end")
            records.append(
                {
                    "site": site,
                    "phase": phase,
                    "depth": depth,
                    "ring": depth,
                    "run_epoch": _EPOCH,
                    "buf_seq": 0,
                    "task_window_start": 0,
                    "task_window_end": advanced,
                    "heap_start": 0,
                    "heap_end": advanced * 65536,
                    "dep_pool_start": 0,
                    "dep_pool_end": advanced,
                    "tensormap": 0,
                }
            )
    metadata = {
        "fatal": False,
        "dropped": 0,
        "total": len(records),
        "task_window_max": [64] * 4,
        "heap_max": [1048576] * 4,
        "dep_pool_max": [1024] * 4,
        "tensormap_max": 65536,
    }
    if recorded:
        metadata.update(
            recorded_window=True,
            fatal=None,
            dropped=None,
            total=None,
            counts_unknown=True,
            collection_verdict="partial_cut_unknown",
            host_received_records=len(records),
            host_retained_records=len(records),
        )
    path.write_text("\n".join(json.dumps(row) for row in [metadata, *records]) + "\n")


@pytest.mark.parametrize(("recorded", "replays"), [(True, 0), (True, 3), (False, 1)])
def test_scope_capture_artifact_checker_accepts_controlled_schema(tmp_path, recorded, replays):
    # Controlled rows exercise the checker, not the device producer or RTS.
    path = tmp_path / "scope_stats.jsonl"
    _fixture_export(path, recorded=recorded, replays=replays)
    _check_export(path, recorded=recorded, replays=replays, epoch=_EPOCH)


@pytest.mark.parametrize(
    "defect", ["missing_boundary", "different_epoch", "known_totals", "unknown_capacity", "no_work"]
)
def test_scope_capture_artifact_checker_rejects_incomplete_evidence(tmp_path, defect):
    path = tmp_path / "scope_stats.jsonl"
    _fixture_export(path, recorded=True, replays=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "missing_boundary":
        rows.pop()
    elif defect == "different_epoch":
        rows[-1]["run_epoch"] += 1
    elif defect == "known_totals":
        rows[0]["total"] = 4
    elif defect == "unknown_capacity":
        rows[0]["heap_max"] = None
    else:
        for row in rows[1:]:
            row["heap_end"] = 0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises((AssertionError, TypeError)):
        _check_export(path, recorded=True, replays=3, epoch=_EPOCH)


def _flush(lib, ctx):
    error = ctypes.create_string_buffer(2048)
    rc = lib.simpler_flush_diagnostics_ctx(ctx, 60000, error, len(error))
    assert rc == 0, (rc, error.value.decode(errors="replace"))


def _run(device, root, replays):
    arch = "a2a3"
    chip = _callable(arch, _RUNTIME)
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib = _load(arch, "onboard", _RUNTIME)
    _bind(lib)
    lib.simpler_wait_run.argtypes = [ctypes.c_void_p] * 2
    lib.simpler_flush_diagnostics_ctx.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t]
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    contexts = _CallerContexts(lib, device)
    ctx, caller = lib.create_device_context(), ctypes.c_void_p()
    assert ctx
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    aicpu, aicore, dispatcher = _binaries(arch, _RUNTIME)
    config = CallConfig(aicpu_thread_num=2, enable_scope_stats=1, output_prefix=str(root).encode())
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    model = _Model(lib, ctx, caller, config, (_EPOCH,))
    artifact = root / "scope_stats/scope_stats.jsonl"
    assert not artifact.exists(), "recording/source retirement published an execution artifact"
    assert lib.simpler_unregister_callable(ctx, 0) != 0
    for value in (2, 7, 19)[:replays]:
        model.replay(value)
    model.close()
    _flush(lib, ctx)
    _check_export(artifact, recorded=True, replays=replays, epoch=_EPOCH)

    eager_root = root / "eager"
    eager_root.mkdir()
    config.output_prefix = str(eager_root).encode()
    eager = _Recording(lib, ctx, caller, config, _EPOCH + 1, 0, capture=False)
    eager.set_inputs(23)
    eager.record()
    assert lib.aclrtSynchronizeStreamWithTimeout(caller, 60000) == 0
    eager.check_output(23)
    assert lib.simpler_wait_run(ctx, eager.run) == 0
    eager.retire_original_run()
    eager.close()
    _flush(lib, ctx)
    _check_export(eager_root / "scope_stats/scope_stats.jsonl", recorded=False, replays=1, epoch=_EPOCH + 1)
    assert lib.simpler_unregister_callable(ctx, 0) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    assert lib.aclrtDestroyStream(caller) == 0
    contexts.close()
    _reset_caller_device(lib, device)
    print(f"TMR_SCOPE_CAPTURE_PASS replays={replays} samples={4 * replays} eager_samples=4", flush=True)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.parametrize("replays", [0, 3])
def test_tmr_scope_capture_exports_real_replay_boundaries(replays, tmp_path, request):
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
    assert f"TMR_SCOPE_CAPTURE_PASS replays={replays}" in result.stdout


if __name__ == "__main__":
    _run(int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
