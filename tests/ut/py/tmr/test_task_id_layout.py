# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Position-consistency check: Python ``tmr.TaskId``'s bit layout vs. the C++ header it mirrors.

Unlike hbg, tmr's ring field has no named ``static constexpr`` in task_id.h -- its
shift is inlined as a literal in both ``ring()`` and ``make()``, and its width is
stated only by the return type of ``ring()``. Both are extracted from the header
directly, so a change to either still fails here without a name to grep for.
"""

from __future__ import annotations

import re
from pathlib import Path

from simpler_setup.tools.tmr import TaskId

_HEADER_PATH = Path(__file__).resolve().parents[4] / "src" / "common" / "tensormap_and_ringbuffer" / "task_id.h"


def test_ring_shift_matches_the_cpp_header():
    header = _HEADER_PATH.read_text()

    ring_shift = re.search(r"ring\(\)\s*const\s*\{\s*return[^;]*raw\s*>>\s*(\d+)", header)
    make_shift = re.search(r"ring_id\)\s*<<\s*(\d+)", header)

    assert ring_shift is not None, f"ring() accessor's shift amount not found in {_HEADER_PATH}"
    assert make_shift is not None, f"make()'s ring shift amount not found in {_HEADER_PATH}"
    assert int(ring_shift.group(1)) == int(make_shift.group(1)) == TaskId.RING_SHIFT


def test_ring_width_matches_the_cpp_accessor_return_type():
    """``ring()``'s return type is what bounds the field, so it is what ``RING_BITS`` must match.

    The C++ accessor narrows with ``static_cast<uint8_t>``, so eight bits from
    ``RING_SHIFT`` up are the whole of a mintable ring. The mirror bounds the field to
    the same width and refuses a value above it, so a C++ type widened without
    widening ``RING_BITS`` leaves the mirror rejecting rings the runtime can mint --
    which is what this comparison catches.
    """
    header = _HEADER_PATH.read_text()

    return_type = re.search(r"constexpr\s+uint(\d+)_t\s+ring\(\)\s*const", header)

    assert return_type is not None, (
        f"ring() accessor's return type not found in {_HEADER_PATH} -- task_id.h's "
        f"declaration style changed and this test no longer recognises it"
    )
    assert int(return_type.group(1)) == TaskId.RING_BITS
    assert TaskId.RING_MASK == (1 << TaskId.RING_BITS) - 1
