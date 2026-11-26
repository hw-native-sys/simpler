#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from tests.st.a5.host_build_graph.urma_queues import test_urma_queues
from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_domain_lifecycle import (
    exercise_retained_domains,
)


@pytest.mark.manual(["a5"])
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(3)
@pytest.mark.parametrize("members", [(1, 2), (2, 1)], ids=["subset", "reordered"])
def test_retained_domains(st_platform, st_device_ids, members):
    exercise_retained_domains(st_platform, st_device_ids, members, test_urma_queues.TestHbgUrmaSegments)


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
