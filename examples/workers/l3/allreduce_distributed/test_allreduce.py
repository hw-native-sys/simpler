# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ST for examples/workers/l3/allreduce_distributed — all five tested algorithm modes."""

import pytest

from .main import run


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.parametrize("mode", ["onephase", "twophase", "ring", "bidirectional_ring", "ibing"])
def test_allreduce_distributed(st_platform, st_device_ids, mode):
    """Test all allreduce modes with 2 devices."""
    assert len(st_device_ids) == 2
    rc = run([int(d) for d in st_device_ids], platform=st_platform, mode=mode)
    assert rc == 0


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.parametrize(
    "n_devices,mode",
    [
        pytest.param(4, "onephase", marks=pytest.mark.device_count(4)),
        pytest.param(4, "twophase", marks=pytest.mark.device_count(4)),
        pytest.param(4, "ring", marks=pytest.mark.device_count(4)),
        pytest.param(4, "bidirectional_ring", marks=pytest.mark.device_count(4)),
    ],
)
def test_allreduce_distributed_multi_rank(st_platform, st_device_ids, n_devices, mode):
    """Test all allreduce modes with 4 devices.

    a5 onboard CI exposes only 2 NPUs, so >2-rank tests run on a2a3 hardware
    and both sims only.
    """
    assert len(st_device_ids) == n_devices
    rc = run([int(d) for d in st_device_ids], platform=st_platform, mode=mode)
    assert rc == 0
