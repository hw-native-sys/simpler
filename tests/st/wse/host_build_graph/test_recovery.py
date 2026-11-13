# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host startup rollback before device bootstrap; execution fault recovery is deferred."""

import subprocess
from pathlib import Path

import pytest

from tests.ut.py.test_sim_startup_failure import _run_scene

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@pytest.mark.parametrize("mode,thread", [("create", "wse-0-0"), ("create", "wse-1-2"), ("bind", "wse-0-1")])
def test_partial_startup_drains_created_threads(tmp_path, mode, thread):
    root = Path(__file__).resolve().parents[4]
    template = (root / "tests/ut/py/support/sim_startup_fault.cpp").read_text()
    source = tmp_path / "fault.cpp"
    source.write_text(
        template.replace("@THREAD@", thread).replace(
            "constexpr bool fail_creation = false;", f"constexpr bool fail_creation = {str(mode == 'create').lower()};"
        )
    )
    library = tmp_path / "fault.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-pthread", "-ldl", "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    code, output = _run_scene("wse", "host_build_graph", library)
    (tmp_path / "startup.log").write_text(output)
    marker = "SIM_STARTUP_CREATE_EAGAIN" if mode == "create" else "SIM_STARTUP_BIND_ENOMEM"
    assert output.count(marker) == 1, output
    assert code == 1 and "TIMEOUT:" not in output, output
    assert "WSE startup rollback: joined" in output and "device events=0" in output, output
    assert "launch_native_run failed with code -1" in output, output
