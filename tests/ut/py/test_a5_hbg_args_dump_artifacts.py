# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synthetic artifact controls for the actual A5 HBG Program scene checker."""

import copy
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tests/st/a5/host_build_graph/dump_args/test_dump_args_example.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("actual_a5_program_args_dump_scene", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module._check_task_payloads


def artifact(level) -> tuple[dict[str, Any], bytes]:
    records = []
    payload = bytearray()
    # Independent arithmetic oracle: 2+3=5; the next task reads that same
    # backing before changing it to 6. A run-end snapshot loses the 5.
    for task, arg, stage, value in (
        (0, 0, "before_dispatch", 2.0),
        (0, 1, "before_dispatch", 3.0),
        (0, 2, "after_completion", 5.0),
        (1, 0, "before_dispatch", 5.0),
        (1, 0, "after_completion", 6.0),
    ):
        if level == 1 and task == 0:
            continue
        selected = level != 2 or task == 1
        size = 128 * 128 * 4 if selected else 0
        records.append(
            dict(
                run_epoch=UINT64_EPOCH,
                task_id=f"0x{task:016x}",
                arg_index=arg,
                stage=stage,
                kind="tensor",
                dtype="FLOAT32",
                role="inout" if task == 1 else "output" if stage == "after_completion" else "input",
                func_id=[task],
                shape=[128 * 128],
                strides=[1],
                start_offset=0,
                is_contiguous=True,
                numel=128 * 128,
                truncated=False,
                bin_offset=len(payload),
                bin_size=size,
            )
        )
        if selected:
            payload.extend(struct.pack("<f", value) * (128 * 128))
    records.append(
        dict(
            run_epoch=UINT64_EPOCH,
            task_id="0x0000000000000001",
            arg_index=1,
            stage="before_dispatch",
            kind="scalar",
            role="input",
            dtype="FLOAT32",
            value=1.0,
            func_id=[1],
            bin_size=0,
        )
    )
    manifest = dict(
        dump_args_level=level, dropped_records_complete=True, dropped_records=0, args=records, total_args=len(records)
    )
    return manifest, bytes(payload)


UINT64_EPOCH = 0x1234567800000001


@pytest.mark.parametrize("level", [1, 2, 3])
def test_real_program_checker_accepts_the_two_actual_stage_values(check, level):
    manifest, payload = artifact(level)
    check(manifest, payload, level)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_real_program_checker_rejects_a_run_end_snapshot(check, level):
    manifest, payload = artifact(level)
    before = next(
        record
        for record in manifest["args"]
        if record["task_id"].endswith("1") and record["kind"] == "tensor" and record["stage"] == "before_dispatch"
    )
    corrupted = bytearray(payload)
    offset, size = before["bin_offset"], before["bin_size"]
    corrupted[offset : offset + size] = struct.pack("<f", 6.0) * (128 * 128)
    with pytest.raises(AssertionError):
        check(manifest, corrupted, level)


@pytest.mark.parametrize("defect", ["epoch", "task", "func", "role", "selection", "truncated", "unknown", "scalar"])
def test_real_program_checker_rejects_wrong_identity_or_incomplete_payloads(check, defect):
    manifest, payload = artifact(3)
    manifest = copy.deepcopy(manifest)
    record = manifest["args"][0]
    if defect == "epoch":
        record["run_epoch"] ^= 1 << 40
    elif defect == "task":
        record["task_id"] = "0x0000000100000000"
    elif defect == "func":
        record["func_id"] = [2]
    elif defect == "role":
        record["role"] = "output"
    elif defect == "selection":
        record["bin_size"] = 0
    elif defect == "truncated":
        record["truncated"] = True
    elif defect == "unknown":
        manifest["counts_unknown"] = True
    elif defect == "scalar":
        manifest["args"][-1]["value"] = 0.0
    with pytest.raises(AssertionError):
        check(manifest, payload, 3)


@pytest.fixture(scope="module")
def mix_scene():
    path = _PATH.parent.parent / "single_block_mix/test_single_block_mix.py"
    spec = importlib.util.spec_from_file_location("actual_a5_mix_args_dump_scene", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def mix_artifact(scene, level, params) -> tuple[dict[str, Any], bytearray, Any]:
    args = scene.TestSingleBlockMixHostBuildGraphA5().generate_args(params)
    scene.TestSingleBlockMixHostBuildGraphA5().compute_golden(args, params)
    records, payload = [], bytearray()
    for task in range(scene.TASK_COUNT if level != 1 else 0):
        mask = params["mask"]
        if params["graph_case"] == 2 and task % 2:
            mask = 1 if task % 4 == 1 else 2
        for arg, name in enumerate("abcdefghi"):
            output = arg in (2, 5, 8)
            chained = params["graph_case"] == 0 and task > 0 and arg in (0, 3, 6) and mask & (1 << (arg // 3))
            offset = task * scene.TILE_ELEMS if output else (task - 1) * scene.TILE_ELEMS if chained else 0
            source = getattr(args, chr(ord(name) + 2) if chained else name)
            selected = level == 3
            size = scene.TILE_ELEMS * 4 if selected else 0
            records.append(
                dict(
                    run_epoch=UINT64_EPOCH,
                    task_id=f"0x{task:016x}",
                    arg_index=arg,
                    func_id=[lane for lane in range(3) if mask & (1 << lane)],
                    stage="after_completion" if output else "before_dispatch",
                    role="output" if output else "input",
                    kind="tensor",
                    dtype="FLOAT32",
                    shape=[scene.TILE_ELEMS],
                    strides=[1],
                    numel=scene.TILE_ELEMS,
                    is_contiguous=True,
                    start_offset=offset,
                    truncated=False,
                    bin_offset=len(payload),
                    bin_size=size,
                )
            )
            if selected:
                payload.extend(source[offset : offset + scene.TILE_ELEMS].numpy().tobytes())
    return (
        dict(
            dump_args_level=level,
            dropped_records_complete=True,
            dropped_records=0,
            args=records,
            total_args=len(records),
        ),
        payload,
        args,
    )


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("params", [{"graph_case": 0, "mask": 7}, {"graph_case": 2, "mask": 7}])
def test_real_mix_checker_accepts_one_set_for_all_active_participants(mix_scene, level, params):
    manifest, payload, args = mix_artifact(mix_scene, level, params)
    mix_scene._check_mix_payloads(manifest, payload, level, args, params)


@pytest.mark.parametrize("defect", ["missing_lane", "duplicate_set", "wrong_stage", "stale_output", "wrong_epoch"])
def test_real_mix_checker_rejects_incomplete_or_aliased_task_capture(mix_scene, defect):
    params = {"graph_case": 0, "mask": 7}
    manifest, payload, args = mix_artifact(mix_scene, 3, params)
    record = manifest["args"][5]
    if defect == "missing_lane":
        record["func_id"] = [0, 1]
    elif defect == "duplicate_set":
        manifest["args"][14] = copy.deepcopy(record)
    elif defect == "wrong_stage":
        record["stage"] = "before_dispatch"
    elif defect == "stale_output":
        start = record["bin_offset"]
        payload[start : start + record["bin_size"]] = b"\0" * record["bin_size"]
    elif defect == "wrong_epoch":
        record["run_epoch"] ^= 1 << 40
    with pytest.raises(AssertionError):
        mix_scene._check_mix_payloads(manifest, payload, 3, args, params)


@pytest.mark.parametrize("capture_fails", [False, True])
@pytest.mark.parametrize("scene_name", ["TestSingleBlockMixHostBuildGraphA5", "TestMixKernelRendezvous"])
def test_failed_mix_preserves_native_log_and_original_error(
    mix_scene, tmp_path, monkeypatch, capture_fails, scene_name
):
    scene_module = importlib.import_module("simpler_setup.scene_test")

    original = RuntimeError("native run failed before completion")
    source = tmp_path / "native" / "host.123.log"
    source.parent.mkdir()
    source.write_text("[ERROR] first device error task=7 site=42\n")
    attempts = []

    def fail_run(*_args, **_kwargs):
        raise original

    def bound_logs():
        attempts.append(True)
        if capture_fails:
            raise OSError("log spool unavailable")
        return [source]

    class UnknownWorker:
        def __getattr__(self, name):
            raise AssertionError(f"failure evidence must not touch Worker.{name}")

    monkeypatch.setattr(scene_module, "_bound_host_logs", bound_logs)
    monkeypatch.setattr(mix_scene.SceneTestCase, "test_run", fail_run)
    monkeypatch.setattr(mix_scene, "_outputs_dir", lambda: tmp_path / "outputs")
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda *_args, **_kwargs: "full"))
    with pytest.raises(RuntimeError) as raised:
        getattr(mix_scene, scene_name)().test_run(None, UnknownWorker(), request)
    assert raised.value is original
    assert attempts == [True]
    evidence = tmp_path / "outputs/onboard-dfx-ci/failed" / scene_name
    if capture_fails:
        assert "log spool unavailable" in (evidence / "capture-error.txt").read_text()
    else:
        assert (evidence / source.name).read_bytes() == source.read_bytes()


def test_failed_mix_retains_bounded_log_tail_with_source_size(mix_scene, tmp_path, monkeypatch):
    scene_module = importlib.import_module("simpler_setup.scene_test")
    source = tmp_path / "host.123.log"
    source.write_bytes(b"x" * (4 * 1024 * 1024) + b"[ERROR] last device first-error\n")
    monkeypatch.setattr(scene_module, "_bound_host_logs", lambda: [source])
    monkeypatch.setattr(mix_scene, "_outputs_dir", lambda: tmp_path / "outputs")
    mix_scene._retain_failure_host_logs("bounded")
    evidence = tmp_path / "outputs/onboard-dfx-ci/failed/bounded"
    assert (evidence / source.name).stat().st_size == 4 * 1024 * 1024
    assert (evidence / source.name).read_bytes().endswith(b"[ERROR] last device first-error\n")
    assert json.loads((evidence / "logs.json").read_text())[0]["source_bytes"] == source.stat().st_size
