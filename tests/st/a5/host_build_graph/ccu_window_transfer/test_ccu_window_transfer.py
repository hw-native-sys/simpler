# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HBG completion and dependency ordering with real CCU window transfers."""

import pytest

from tests.st.a5.tensormap_and_ringbuffer.ccu_window_transfer.test_ccu_window_transfer import _exercise_window_transfer


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(2)
def test_ccu_real_window_transfer(st_platform, st_device_ids, tmp_path, request):
    _exercise_window_transfer(st_platform, st_device_ids[:2], tmp_path, request, (0, 1), runtime="host_build_graph")
