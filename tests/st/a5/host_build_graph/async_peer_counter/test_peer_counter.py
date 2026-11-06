# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Two peer COUNTER qualification cases; direct observer probes cannot substitute for these runs."""

import pytest

from .helpers import PeerCounterRunner, l3_case

pytestmark = [
    pytest.mark.manual,
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(2),
]


@l3_case
def test_peer_counter_epoch_boundaries(st_platform, st_device_ids, tmp_path):
    with PeerCounterRunner(st_platform, st_device_ids, tmp_path) as runner:
        for epoch, value_tag, reset in ((1, 101, True), (2, 102, False), (1, 103, True)):
            runner.run(epoch, value_tag, reset=reset)


@l3_case
def test_peer_counter_c_plus_one_progress(st_platform, st_device_ids, tmp_path):
    with PeerCounterRunner(st_platform, st_device_ids, tmp_path, pressure=True) as runner:
        runner.run(1, 201, reset=True)
