# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Peer-card notifications complete returned percounter kernels."""

from copy import deepcopy

from simpler_setup import SceneTestCase, scene_test
from tests.st.a5.host_build_graph.notify_defer_wait.test_notify_defer_wait import (
    TestNotifyDeferWaitTwoRanks as _PeerNotify,
)


@scene_test(level=3, runtime="percounter")
class TestPercounterPeerDeferWait(SceneTestCase):
    CALLABLE = deepcopy(_PeerNotify.CALLABLE)
    CASES = deepcopy(_PeerNotify.CASES)
    RTOL = _PeerNotify.RTOL
    ATOL = _PeerNotify.ATOL
    generate_args = _PeerNotify.generate_args
    compute_golden = _PeerNotify.compute_golden


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
