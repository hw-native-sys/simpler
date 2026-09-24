# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for what only tensormap_and_ringbuffer does, mirroring ``simpler_setup/tools/tmr/``.

A test belongs here when its subject is this runtime's own vocabulary -- its TaskId
layout, its phase set, the markers only it records. A test of a tool both runtimes
share stays in the parent directory, even when its fixture carries this runtime's
data; the ids for that come from ``tests/ut/py/_task_ids.py``, which mints them
through this runtime's own mirror.
"""
