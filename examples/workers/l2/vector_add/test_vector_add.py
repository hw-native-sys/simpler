# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST for examples/workers/l2/vector_add."""

import pytest

from .main import run


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_vector_add(st_platform, st_device_ids):
    rc = run(st_platform, int(st_device_ids[0]))
    assert rc == 0
