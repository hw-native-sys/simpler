# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from simpler_setup.tools.urma_verify import ExecutionGate, verify_artifacts


@pytest.mark.parametrize("bad", ["empty", "missing", "skip", "no_call", "xfail", "fail", "teardown", None])
def test_required_urma_execution(bad):
    gate = ExecutionGate(("read", "consumer"))
    names = [] if bad == "empty" else (["read"] if bad == "missing" else ["read", "consumer"])
    session = NS(items=[NS(nodeid=name) for name in names], exitstatus=0)
    gate.pytest_collection_finish(session)
    for name in names:
        if bad == "no_call":
            continue
        report = NS(
            nodeid=name,
            when="teardown" if bad == "teardown" else "call",
            skipped=bad == "skip",
            failed=bad in {"fail", "teardown"},
            passed=bad not in {"skip", "fail", "teardown"},
            wasxfail="expected" if bad == "xfail" else None,
            outcome=bad or "passed",
        )
        gate.pytest_runtest_logreport(report)
    gate.pytest_sessionfinish(session)
    assert (session.exitstatus == 0) == (bad is None)


@pytest.mark.parametrize("mode", ["empty", "skip", "pass"])
def test_gate_with_real_pytest(tmp_path, mode):
    if mode != "empty":
        body = "pytest.skip('unavailable')" if mode == "skip" else "pass"
        (tmp_path / "test_cases.py").write_text(
            f"import pytest\ndef test_read():\n    pass\ndef test_consumer():\n    {body}\n"
        )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pytest; from simpler_setup.tools.urma_verify import ExecutionGate; "
            "raise SystemExit(pytest.main(['-q'], plugins=[ExecutionGate(('test_read', 'test_consumer'))]))",
        ],
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if mode == "pass" else 1), result.stdout + result.stderr


@pytest.mark.parametrize("urma,rdma", [("OFF", "OFF"), ("", "OFF"), ("ON", "ON"), ("ON", "")])
def test_explicit_urma_capability_required(monkeypatch, urma, rdma):
    monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", urma)
    monkeypatch.setenv("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", rdma)
    with pytest.raises(RuntimeError, match="requires explicit"):
        verify_artifacts()


@pytest.mark.parametrize(
    "path,non_code,urma_job",
    [
        (".claude/skills/onboard-arch-precheck/check.sh", False, True),
        (".claude/skills/onboard-arch-precheck/SKILL.md", True, False),
        ("docs/a5-urma.md", True, False),
        (".gitignore", True, False),
        ("src/a5/platform/onboard/host/comm_hccl.cpp", False, True),
        (".github/workflows/_st-urma-a5.yml", False, True),
        ("tests/ut/py/test_urma_contract.py", False, False),
    ],
)
def test_urma_ci_change_detection(tmp_path, path, non_code, urma_job):
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/_detect-changes.yml").read_text()
    # Execute the canonical workflow's decision body with its input file list.
    script = textwrap.dedent(workflow[workflow.index("          NON_CODE=") :])
    output = tmp_path / "github-output"
    subprocess.run(
        ["bash", "-e", "-c", script],
        env={**os.environ, "FILES": path, "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        timeout=15,
    )
    flags = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (flags["non_code_only"] == "true") == non_code
    assert (flags["a5_changed"] == "true" and flags["st_affected"] == "true") == urma_job
