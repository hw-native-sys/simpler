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
import os
import shutil
import subprocess
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


@pytest.fixture(scope="module")
def native_runner(tmp_path_factory):
    runner = os.environ.get("DSV41_RUNNER")
    if runner:
        return runner
    if not shutil.which("cmake"):
        pytest.skip("cmake or a built DSV41_RUNNER is required")
    build = tmp_path_factory.mktemp("tools")
    subprocess.run(["cmake", "-S", PRODUCTION, "-B", build, f"-DSIMPLER_ROOT={PRODUCTION.parents[3]}"], check=True)
    subprocess.run(["cmake", "--build", build, "--target", "native-runner", "-j4"], check=True)
    return build / "native-runner"


def test_kernel_signatures_survive_packing_and_orchestration_replacement(tmp_path, native_runner):
    kernel = tmp_path / "kernel.bin"
    kernel.write_bytes(bytes(range(32)))
    host = tmp_path / "host.so"
    host.write_bytes(bytes(range(64)))
    manifest = tmp_path / "kernels.txt"
    manifest.write_text(f'1 3\n2\n0 1 3 "{kernel}"\n1 1 1 "{kernel}"\n')
    packed, repacked = tmp_path / "callable.bin", tmp_path / "replacement.bin"
    subprocess.run([native_runner, "--assemble-orchestration", manifest, host, packed], check=True)
    host.write_bytes(b"changed orchestration")
    subprocess.run([native_runner, "--repack-orchestration", packed, host, repacked], check=True)
    for path in (packed, repacked):
        description = json.loads(subprocess.check_output([native_runner, "--describe-callable", path], text=True))
        assert description["kernels"] == [{"id": 0, "signature": [3]}, {"id": 1, "signature": [1]}]


def test_native_assembly_rejects_trailing_kernel_fields(tmp_path, native_runner):
    kernel = tmp_path / "kernel.bin"
    kernel.write_bytes(bytes(range(32)))
    host = tmp_path / "host.so"
    host.write_bytes(bytes(range(64)))
    manifest = tmp_path / "kernels.txt"
    manifest.write_text(f'1 3\n1\n0 1 3 "{kernel}" extra\n')
    output = tmp_path / "callable.bin"
    result = subprocess.run(
        [native_runner, "--assemble-orchestration", manifest, host, output], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "Unexpected trailing fields in kernel manifest" in result.stderr
    assert not output.exists()
