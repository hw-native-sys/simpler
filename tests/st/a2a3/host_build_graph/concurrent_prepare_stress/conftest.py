# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated L2 worker for the concurrent-prepare white-box stress.

The default ``st_worker`` (root conftest) is shared across L2 ST classes in a
session-scoped pool — correct for ordinary business tests, but this stress
drives the ChipWorker native-run lane directly (registering a callable slot and
submitting overlapping runs), so it needs a fresh worker whose private slot
table and native-run lane start empty and are not left with residue for other
pooled tests.

Override ``st_worker`` here as class-scope, building a fresh L2 worker after
retiring any same-device pooled owner. Cost: one extra init/close per test class.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="class")
def st_worker(_isolated_l2_worker):
    """Use the shared isolated lifetime after any pooled owner has retired."""
    return _isolated_l2_worker
