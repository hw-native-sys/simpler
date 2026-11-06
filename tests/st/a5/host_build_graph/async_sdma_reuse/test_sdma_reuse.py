# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from .helpers import SdmaReuseRunner, l3_case

pytestmark = [
    pytest.mark.manual,
    pytest.mark.platforms(["a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(2),
    pytest.mark.sdma(worker_workspace=False),
]


@l3_case
def test_public_post_ids_wrap_and_continue_in_retained_workspace(st_platform, st_device_ids, tmp_path):
    with SdmaReuseRunner(st_platform, st_device_ids, tmp_path) as runner:
        runner.run(66)
        runner.run(2)
