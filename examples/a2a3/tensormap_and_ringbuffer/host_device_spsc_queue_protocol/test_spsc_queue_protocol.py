# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import inspect
from pathlib import Path

import pytest

from .main import run

HERE = Path(__file__).resolve().parent


def test_run_uses_channel_protocol_helpers():
    source = inspect.getsource(run)
    assert "open_channel(" in source
    for forbidden in (
        "open_mapped_region",
        "mapped_region_info",
        "mapped_region_datacopy_h2region",
        "mapped_region_notify",
        "mapped_region_wait",
        "close_mapped_region",
    ):
        assert forbidden not in source


def test_kernel_constructs_response_metadata_field_by_field():
    source = (HERE / "kernels" / "aiv" / "host_device_spsc_queue_protocol.cpp").read_text()
    assert "*out = in" not in source
    for required in (
        "out->flags = in.flags",
        "out->seq = in.seq",
        "out->correlation_id = in.correlation_id",
        "out->route = in.route ^ 0x80000000U",
    ):
        assert required in source


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_spsc_queue_protocol(st_platform, st_device_ids):
    assert run(st_platform, int(st_device_ids[0])) == 0
