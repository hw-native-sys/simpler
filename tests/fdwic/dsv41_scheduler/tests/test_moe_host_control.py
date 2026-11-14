# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native MoE must read a live Host control and cannot accept frozen replay bundles."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual

PRODUCTION = Path(__file__).resolve().parents[1] / "production"


def load(name, directory=PRODUCTION):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_read_requires_executable_orchestration(tmp_path):
    moe = load("moe")
    kernel = tmp_path / "kernels/aiv/kernel.cpp"
    kernel.parent.mkdir(parents=True)
    call = "int32_t n = get_tensor_data<int32_t>(ext_num_tokens, 1, indices);"
    kernel.write_text(call)
    with pytest.raises(ValueError, match="does not read"):
        moe.control_reads(tmp_path)
    orch = tmp_path / "next_levels/l2_moe/orchestration/l2_moe.cpp"
    orch.parent.mkdir(parents=True)
    orch.write_text("// " + call)
    with pytest.raises(ValueError, match="does not read"):
        moe.control_reads(tmp_path)
    orch.write_text(call)
    assert moe.control_reads(tmp_path) == [{"file": str(orch.relative_to(tmp_path)), "line": 1, "code": call}]


def test_native_moe_has_two_devices_without_replay_inputs(tmp_path):
    topology = load("topology")
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "flavor": "pypto_lib_latest",
                "case": "moe",
                "execution": "native_moe_host_control",
                "device_count": 2,
            }
        )
    )
    assert topology.device_count(tmp_path) == 2
    with pytest.raises(ValueError, match="without cut-input"):
        topology.device_count(tmp_path, materialize=True)


def test_frozen_pypto_lib_moe_is_rejected(tmp_path):
    (tmp_path / "case.json").write_text(json.dumps({"flavor": "pypto_lib_latest", "case": "moe", "device_count": 2}))
    with pytest.raises(ValueError, match="frozen MoE plans are unsupported"):
        load("topology").device_count(tmp_path)


def test_native_audit_rejects_compile_only_and_missing_count_case(tmp_path):
    audit = load("analyze", PRODUCTION.parent)
    case = {"case": "moe", "flavor": "pypto_lib_latest", "lib_commit": "source"}
    for level in (0, 1, 3):
        directory = tmp_path / "moe" / f"L{level}"
        directory.mkdir(parents=True)
        report = {
            "passed": True,
            "compile_only": level == 0,
            "profiling_level": level,
            "source_commit": "source",
            "control_reads": [{"file": "orchestration.cpp", "line": 1}],
            "performance": {"rounds": 10},
            "checks": [{"counts": counts, "passed": True} for counts in ([0, 6], [6, 0], [72, 72])],
        }
        (directory / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="all Host-control goldens"):
        audit.native_moe_reports(tmp_path, case)
    path = tmp_path / "moe/L0/result.json"
    report = json.loads(path.read_text())
    report["compile_only"] = False
    report["checks"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="all Host-control goldens"):
        audit.native_moe_reports(tmp_path, case)
