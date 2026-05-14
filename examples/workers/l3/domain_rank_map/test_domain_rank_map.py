# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ST for examples/workers/l3/domain_rank_map."""

import pytest

from .main import run


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(3)
def test_domain_rank_map(st_platform, st_device_ids, capsys):
    rc = run(st_platform, [int(d) for d in st_device_ids])
    out = capsys.readouterr().out
    assert rc == 0
    assert "even worker 0: max_diff=" in out
    assert "tail worker 2: max_diff=" in out
    assert "communication checks PASSED" in out
