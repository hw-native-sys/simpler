# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST for examples/workers/l3/allreduce.

Covers the demo only. The allreduce *algorithm* corpus — ring, twophase, ibing
and bidirectional-ring — lives in tests/st/worker/collectives/allreduce.
"""

import pytest

from .main import run

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]


# a2a3 only, which is what this demo's own README and CLI default document.
# The algorithm itself runs on a5 — tests/st/worker/collectives/allreduce
# declares all four platforms — so widening this is likely safe, but nobody has
# exercised *this* demo there yet.
@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.manual(["a2a3sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_allreduce(st_platform, st_device_ids):
    # run() takes (device_ids, platform) — the reverse of the single-device
    # demos in this tree, which take (platform, device_id).
    rc = run([int(d) for d in st_device_ids], platform=st_platform)
    assert rc == 0
