# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Raw task ids for a test to feed a decoder, minted through each runtime's own mirror.

A capture holds task ids as plain integers, so a test building one needs the raw word
rather than a ``TaskId``. That is the whole of this module: every id comes from the
owning runtime's ``make_*``, and each function hands back ``.raw``.

No bit position appears here. The layout lives in ``simpler_setup/tools/hbg/task_id.py``
and ``simpler_setup/tools/tmr/task_id.py``, which mint and decode from the same
constants, so a shift that drifted from its C++ header moves both directions together
and ``tests/ut/py/{hbg,tmr}/test_task_id_layout.py`` is what compares it against the
header.
"""

from __future__ import annotations

from simpler_setup.tools.hbg import TaskId as _HbgTaskId
from simpler_setup.tools.tmr import TaskId as _TmrTaskId


def hbg_global(local_id: int) -> int:
    """Raw id for an hbg task of the run itself."""
    return _HbgTaskId.make_global(local_id).raw


def hbg_sub_task(parent_id: int, local_id: int) -> int:
    """Raw id for an hbg sub-task materialized under ``parent_id``."""
    return _HbgTaskId.make_sub_task(parent_id, local_id).raw


def hbg_param(param_index: int) -> int:
    """Raw id for an hbg boundary parameter."""
    return _HbgTaskId.make_param(param_index).raw


def tmr_task(ring_id: int, local_id: int) -> int:
    """Raw id for a tmr task on ``ring_id``."""
    return _TmrTaskId.make(ring_id, local_id).raw
