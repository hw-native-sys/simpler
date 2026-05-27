# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware smoke coverage for HostDeviceMappedRegion on a2a3 onboard."""

from __future__ import annotations

import os

import pytest


@pytest.mark.requires_hardware("a2a3")
@pytest.mark.platforms(["a2a3"])
def test_a2a3_onboard_mapped_region_host_side_smoke(st_device_ids):
    from simpler.worker import Worker
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    _ = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=build)
    device_id = int(st_device_ids[0])

    worker = Worker(level=2, platform="a2a3", runtime="tensormap_and_ringbuffer", device_id=device_id, build=build)
    worker.init()
    try:
        region = worker.open_mapped_region(128, signal_count=2)
        try:
            info = worker.mapped_region_info(region)
            assert info.host_data_ptr == 0
            assert info.host_signal_ptr == 0
            assert info.device_data_ptr != 0
            assert info.device_signal_ptr != 0
            assert info.data_bytes == 128
            assert info.signal_count == 2

            payload = bytes((i * 7) % 251 for i in range(96))
            worker.mapped_region_datacopy_h2region(region, 16, payload)
            assert worker.mapped_region_datacopy_region2h(region, 16, len(payload)) == payload

            worker.mapped_region_wait(region, 0, 0, 0)
            with pytest.raises(TimeoutError):
                worker.mapped_region_wait(region, 0, 1, 0)
            worker.mapped_region_notify(region, 0, 3)
            worker.mapped_region_wait(region, 0, 3, 0)
        finally:
            worker.close_mapped_region(region)
    finally:
        worker.close()
