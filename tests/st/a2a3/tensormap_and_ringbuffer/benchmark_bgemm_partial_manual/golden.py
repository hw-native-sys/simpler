# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reuse the baseline benchmark_bgemm golden data for the partial-manual scene."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_base_module():
    base_path = Path(__file__).resolve().parents[1] / "benchmark_bgemm" / "golden.py"
    spec = spec_from_file_location("benchmark_bgemm_base_golden", base_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()

__outputs__ = _BASE.__outputs__
RTOL = _BASE.RTOL
ATOL = _BASE.ATOL
ALL_CASES = _BASE.ALL_CASES
DEFAULT_CASE = _BASE.DEFAULT_CASE
SUPPORTED_INCORE_DATA_SIZES = _BASE.SUPPORTED_INCORE_DATA_SIZES
generate_inputs = _BASE.generate_inputs
compute_golden = _BASE.compute_golden
