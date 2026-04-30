# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST for examples/workers/l3/moe_multi_chip_experts."""

import pytest

from .main import run


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_moe_multi_chip_2_experts(st_platform, st_device_ids):
    """Test multi-chip MoE with 2 experts (1 per chip).

    This should produce the SAME results as moe_single_chip with 2 experts,
    just executed in parallel across 2 chips instead of sequentially on 1 chip.
    """
    rc = run(st_platform, [int(d) for d in st_device_ids])
    assert rc == 0


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(4)
def test_moe_multi_chip_4_experts(st_platform, st_device_ids):
    """Test multi-chip MoE with 4 experts (1 per chip).

    This should produce the SAME results as moe_single_chip with 4 experts,
    just executed in parallel across 4 chips instead of sequentially on 1 chip.
    """
    rc = run(st_platform, [int(d) for d in st_device_ids])
    assert rc == 0
