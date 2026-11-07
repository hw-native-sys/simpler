#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST wrapper for Qwen3-14B decode on host_build_graph."""

import importlib.util
import sys
from pathlib import Path

import pytest

from simpler_setup.scene_test import standalone_pytest_options

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]

_DRIVER_NAME = "_qwen3_14b_a5_hbg_driver"


def _driver():
    cached = sys.modules.get(_DRIVER_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_DRIVER_NAME, Path(__file__).resolve().parent / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Qwen3-14B host_build_graph driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_DRIVER_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
def test_qwen3_14b_decode_host_build_graph(st_platform, st_device_ids, request):
    assert _driver().run(st_device_ids, st_platform, **standalone_pytest_options(request)) == 0


if __name__ == "__main__":
    sys.exit(_driver().main())
