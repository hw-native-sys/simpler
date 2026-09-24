# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 onboard counterparts for TMR report-epoch hard-boundary scenarios."""

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.test_kernel_mode_capture import (
    _run_onboard_scenario,
    build_capture_observer,
)


@pytest.fixture(scope="module")
def capture_observer(tmp_path_factory):
    return build_capture_observer("a5", tmp_path_factory)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.parametrize("scenario", ("delayed_aicore_report", "device_error_eager"))
def test_a5_kernel_mode_report_epoch(st_platform, st_device_ids, scenario, capture_observer):
    _run_onboard_scenario(st_platform, st_device_ids, scenario, capture_observer)
