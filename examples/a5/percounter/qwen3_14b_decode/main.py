#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode on percounter with device-resident parameters.

The graph is the one host_build_graph runs, down to the orchestration source:
percounter accepts it unchanged, and running the same shapes through both is
how the two schedulers are compared. Only the runtime name differs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMR_CASE_DIR = HERE.parents[1] / "tensormap_and_ringbuffer/qwen3_14b_decode"
HBG_CASE_DIR = HERE.parents[1] / "host_build_graph/qwen3_14b_decode"
RUNTIME = "percounter"
ORCHESTRATION_SOURCE = HBG_CASE_DIR / "kernels/orchestration/decode_fwd_layers.cpp"
CASE_NAME = "GraphExecutionBatch16Seq3500"


def _load_tmr_driver():
    module_name = "_qwen3_14b_a5_tmr_driver"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, TMR_CASE_DIR / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the TMR Qwen3-14B decode driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run(device_ids, platform: str, **kwargs) -> int:
    driver = _load_tmr_driver()
    kwargs.setdefault("runtime", RUNTIME)
    kwargs.setdefault("orchestration_source", ORCHESTRATION_SOURCE)
    kwargs.setdefault("runtime_env", {})
    return driver.run(device_ids, platform, **kwargs)


def main(argv=None) -> int:
    driver = _load_tmr_driver()
    return driver.main(
        argv,
        case_name=CASE_NAME,
        runtime=RUNTIME,
        orchestration_source=ORCHESTRATION_SOURCE,
        runtime_env={},
    )


if __name__ == "__main__":
    sys.exit(main())
