#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private L3→L2 single-hop delegated-region scene test."""

from __future__ import annotations

import pytest
from simpler.worker import Worker

from ._helpers import (
    _RUNTIME,
    SUCCESS_PLATFORMS,
    build_chip_callable,
    close_owned_workers,
    install_lifecycle_recorder,
    make_l3_submit,
    run_two_lifecycles,
)

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]


@pytest.mark.platforms(SUCCESS_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l3_single_hop_two_lifecycles(st_platform, st_device_ids):
    chip_callable = build_chip_callable(st_platform)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    recorder = None
    primary = None
    try:
        recorder = install_lifecycle_recorder(worker)
        chip_handle = worker.register(chip_callable)
        worker.init()
        run_two_lifecycles(
            worker,
            provider_path="L3/L2[0]",
            submit_chip=make_l3_submit(chip_handle),
            recorder=recorder,
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close_owned_workers(primary, worker)
        finally:
            if recorder is not None and recorder.restore is not None:
                recorder.restore()
