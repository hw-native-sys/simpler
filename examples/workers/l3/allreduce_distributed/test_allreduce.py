# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ST for examples/workers/l3/allreduce_distributed."""

import pytest

from .main import run


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_allreduce_distributed(st_platform, st_device_ids):
    assert len(st_device_ids) == 2
    rc = run([int(d) for d in st_device_ids], platform=st_platform)
    assert rc == 0


# >2-rank cases live in a separate function so a5 can be dropped via the
# function-level platforms mark (the harness deselects by that mark, not by
# per-param marks). a5 onboard CI exposes only 2 NPUs, and a device_count(N>2)
# job aborts the whole resource phase — which would also take down the 2-rank
# case above. Still covered on a2a3 hardware and both sims.
@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.parametrize(
    "n_devices",
    [
        pytest.param(4, marks=pytest.mark.device_count(4)),
    ],
)
def test_allreduce_distributed_multi_rank(st_platform, st_device_ids, n_devices):
    assert len(st_device_ids) == n_devices
    rc = run([int(d) for d in st_device_ids], platform=st_platform)
    assert rc == 0
