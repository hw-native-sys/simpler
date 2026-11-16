# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The benchmark runtime is isolated from production sources and artifacts."""

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
ROOT = Path(__file__).resolve().parents[4]
PRODUCTION = Path(__file__).resolve().parents[1] / "production"
PATHS = (
    "src/a5/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp",
    "src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp",
    "src/a5/runtime/tensormap_and_ringbuffer/runtime/runtime.h",
)


def module():
    return runpy.run_path(str(PRODUCTION / "tmr_runtime.py"))


def test_overlay_changes_only_the_source_copy(tmp_path):
    source = tmp_path / "source"
    for name in ("cmake", "simpler_setup", "python"):
        (source / name).mkdir(parents=True)
    for name in PATHS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    before = {name: (source / name).read_bytes() for name in PATHS}
    with pytest.raises(ValueError, match="outside production"):
        module()["stage_sources"](source, source / "src/nested-snapshot")
    assert not (source / "src/nested-snapshot").exists()
    snapshot = tmp_path / "build/source"
    report = module()["stage_sources"](source, snapshot)
    assert {name: (source / name).read_bytes() for name in PATHS} == before
    assert set(report["patched_sources"]) == set(PATHS)
    assert "work_start_cycle" not in (source / PATHS[0]).read_text()
    assert "work_start_cycle" in (snapshot / PATHS[0]).read_text()
    assert "cache_invalidate_range(runtime->dev.workers" in (snapshot / PATHS[1]).read_text()
    assert "work_end_cycle" in (snapshot / PATHS[2]).read_text()
    with pytest.raises(ValueError, match="new"):
        module()["stage_sources"](source, snapshot)


def test_l0_rejects_stock_runtime_and_mismatched_components(tmp_path):
    runtime = tmp_path / "runtime/a5/onboard"
    variant = runtime / "tensormap_and_ringbuffer"
    variant.mkdir(parents=True)
    validate = module()["validate_runtime"]
    with pytest.raises(ValueError, match="test-only"):
        validate(runtime)
    components = {}
    for name in ("aicore_kernel.o", "libaicpu_kernel.so"):
        data = name.encode()
        (variant / name).write_bytes(data)
        components[name] = hashlib.sha256(data).hexdigest()
    (variant / "dsv41_timing.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "test_only": True,
                "metric": "first_kernel_start_to_last_kernel_end",
                "components": components,
            }
        )
    )
    validate(runtime)
    # The marker travels with an ordinary copied Device-gate snapshot.
    copied = tmp_path / "gate/runtime/a5/onboard"
    shutil.copytree(runtime, copied)
    validate(copied)
    (copied / "tensormap_and_ringbuffer/aicore_kernel.o").write_bytes(b"stock")
    with pytest.raises(ValueError, match="mismatch"):
        validate(copied)


def test_stock_l0_is_rejected_before_output_or_reservation(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    runtime = tmp_path / "stock"
    runtime.mkdir()
    output = tmp_path / "run"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    requested = tmp_path / "reservation-requested"
    submit = fake_bin / "task-submit"
    submit.write_text(f"#!/bin/sh\ntouch '{requested}'\nexit 99\n")
    submit.chmod(0o755)
    env = dict(os.environ, ASCEND_HOME_PATH=str(tmp_path), TASK_DEVICE="", DSV41_PYTHON=sys.executable)
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [
            "bash",
            str(PRODUCTION / "run.sh"),
            str(bundle),
            str(runtime),
            str(output),
            "0",
            "1",
            "10",
            "--runtime=tensormap_and_ringbuffer",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "test-only TMR build" in result.stderr
    assert not output.exists()
    assert not requested.exists()
