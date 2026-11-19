# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native TMR replay uses the same task-time ArgsDump fixture and viewer as HBG."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3"]),
    pytest.mark.runtime("tensormap_and_ringbuffer"),
]


@pytest.fixture(scope="module")
def tmr_args_dump_probe(tmp_path_factory):
    path = _ROOT / "tests/st/a2a3/kernel_capture/test_hbg_capture.py"
    spec = importlib.util.spec_from_file_location("tmr_args_dump_native_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binaries = module.build_probe(
        tmp_path_factory.mktemp("tmr-argsdump-capture"),
        retained_models=True,
        args_dump=True,
        runtime_name="tensormap_and_ringbuffer",
    )
    return module, binaries


@pytest.mark.parametrize("replays", [0, 3])
def test_tmr_args_dump_capture_exports_task_time_values(tmr_args_dump_probe, tmp_path, request, replays):
    module, binaries = tmr_args_dump_probe
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    module.run_args_dump_probe(binaries, device, tmp_path / "dump", replays)
