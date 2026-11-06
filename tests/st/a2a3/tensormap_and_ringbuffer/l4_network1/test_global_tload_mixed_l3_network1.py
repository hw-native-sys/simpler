# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Network1 collection sentinel for tests/st.

This intentionally re-runs the global_tload_mixed_l3 example path so network1 CI
proves both `examples/` and `tests/st/` network1 tests are collected by marker.
"""

import pytest

from examples.workers.l4.global_tload_mixed_l3.main import run
from simpler_setup import SceneTestLevel, scene_level

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]


def _device_spec(device_ids) -> str:
    return ",".join(str(device_id) for device_id in device_ids)


@scene_level(SceneTestLevel.NETWORK1)
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.network1_remote_device_count(1)
def test_global_tload_mixed_l3_network1(
    st_platform, st_device_ids, st_network1_peer, st_network1_remote_device_ids, st_network1_logs
):
    rc = run(
        remote=st_network1_peer.endpoint,
        local_devices=_device_spec(st_device_ids),
        remote_devices=_device_spec(st_network1_remote_device_ids),
        platform=st_platform,
        session_timeout=st_network1_peer.session_timeout_s,
        session_listen_host=st_network1_peer.session_listen_host,
    )
    assert rc == 0
