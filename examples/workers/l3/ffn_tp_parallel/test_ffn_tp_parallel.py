# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ST for examples/workers/l3/ffn_tp_parallel."""

import os
from importlib.machinery import SourceFileLoader

import pytest

_main = SourceFileLoader("ffn_tp_parallel_main", os.path.join(os.path.dirname(__file__), "main.py")).load_module()
run = _main.run


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_ffn_tp_parallel(st_platform, st_device_ids):
    rc = run(st_platform, [int(d) for d in st_device_ids])
    assert rc == 0
