#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from .helpers import l3_case, run_sdma

pytestmark = [
    pytest.mark.manual,  # Acceptance contracts awaiting resident async integration; run with --manual include.
    pytest.mark.platforms(["a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(2),
    pytest.mark.sdma(worker_workspace=False),
]


@pytest.mark.parametrize("elements", [0, 8, 1024 * 1024], ids=["empty", "minimum_tile", "pending_after_return"])
@l3_case
def test_real_peer_sdma_completion(st_platform, st_device_ids, tmp_path, elements):
    run_sdma(st_platform, st_device_ids, tmp_path, elements)
