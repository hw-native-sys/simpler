# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest

from tests.st.a2a3.host_build_graph.worker_async_fifo import test_worker_async_fifo as fifo

_ROOT = Path(__file__).resolve().parents[3]
_FIFO = "tests/st/a2a3/host_build_graph/worker_async_fifo"


def test_default_ci_sweep_collects_all_fifo_cases():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", _FIFO, "--collect-only", "--platform", "a2a3", "-q"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [line for line in result.stdout.splitlines() if line.startswith("tests/st/") and "::" in line]
    assert len(nodes) == 8
    assert sum("independent-input" in node for node in nodes) == 2
    assert sum("shared-read-only-input" in node for node in nodes) == 2
    assert any("::TestWorkerAsyncWholeRunFifoTmrMultiRank::test_run" in node for node in nodes)
    assert any("::test_incompatible_runtime_env_falls_back_to_depth_one" in node for node in nodes)
    assert sum("::test_a_run_whose_cleanup_touches_the_device_degrades_to_depth_one" in node for node in nodes) == 2


@pytest.mark.parametrize(
    ("cls", "extra_checks"),
    [(fifo.TestWorkerAsyncWholeRunFifo, []), (fifo.TestWorkerAsyncWholeRunFifoTmr, ["fallback"])],
)
def test_standalone_runs_fifo_checks(cls, extra_checks):
    scene = cls()
    calls = []
    scene.test_a_run_whose_cleanup_touches_the_device_degrades_to_depth_one = lambda *_args: calls.append("cleanup")
    scene.test_run = lambda _platform, _worker, shared: calls.append(shared)
    scene.test_incompatible_runtime_env_falls_back_to_depth_one = lambda *_args: calls.append("fallback")
    scene._run_and_validate_l3(SimpleNamespace(_config={"platform": "a2a3"}), {}, {}, cls.CASES[0])
    assert calls == ["cleanup", False, True, *extra_checks]


def test_default_subtask_callable_preserves_signals_in_fresh_process(tmp_path):
    payload = tmp_path / "subtask.pkl"
    payload.write_bytes(cloudpickle.dumps(fifo._wait_for_release))
    fifo._SUB_ENTERED.clear()
    fifo._SUB_RELEASE.set()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import cloudpickle, sys; from pathlib import Path; "
                "cloudpickle.loads(Path(sys.argv[1]).read_bytes())(None)",
                str(payload),
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert fifo._SUB_ENTERED.wait(0), "the child did not signal the parent's fence"
    finally:
        fifo._SUB_ENTERED.clear()
        fifo._SUB_RELEASE.clear()
