# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check native MoE and replay dispatch for the retained model."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
PRODUCTION = Path(__file__).resolve().parents[1] / "production"
SPEC = importlib.util.spec_from_file_location("native_export_cases", PRODUCTION / "export_cases.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("selected", [None, ("swa", "c2a_full", "c1a_reindex", "c1a_reuse", "engram")])
def test_export_keeps_native_plans_and_uses_host_entry_for_moe(monkeypatch, tmp_path, selected):
    flavor = "pypto_lib_latest"
    cases = json.loads((PRODUCTION.parent / "workloads" / flavor / "source.json").read_text())["cases"]
    by_first = {case["first_source_task"]: case for case in cases.values() if "first_source_task" in case}
    bundle, capture, output = (tmp_path / name for name in ("bundle", "capture", "output"))
    bundle.mkdir()
    for rank in range(2):
        path = capture / f"rank{rank}"
        path.mkdir(parents=True)
        (path / "submissions.bin").write_bytes(f"original plan for rank {rank}".encode())
        (bundle / f"rank{rank}.txt").write_text("DSV41_NATIVE_V1\n2 0\n0\n0\nVALIDATE_ZERO 0\n")

    def record_detail(command, **kwargs):
        case = by_first[command[-2]]
        assert command[-1] == case["last_source_task"]
        return json.dumps(
            {"tasks": case["tasks"], "executions": case["executions"], "external_bytes": [], "boundary_buffers": []}
        )

    native = []
    monkeypatch.setattr(MODULE.subprocess, "check_output", record_detail)
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE, "_export_native", lambda destination, *args: native.append(destination))
    paths = MODULE.export(bundle, capture, flavor, output, tmp_path / "tools", case_names=selected)
    assert {p.name for p in paths} == set(cases if selected is None else selected)
    assert native == ([output / flavor / "moe"] if selected is None else [])
    for case in (p for p in paths if p.name != "moe"):
        for rank in range(2):
            captured = (capture / f"rank{rank}/submissions.bin").read_bytes()
            assert (case / f"plan-rank{rank}.bin").read_bytes() == captured
