# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Extract ``static constexpr`` literals and ``enum class`` members from C++ header text.

Used by the task_id layout sync tests (hbg/test_task_id_layout.py,
tmr/test_task_id_layout.py) to compare a runtime's C++ TaskId layout against its
Python mirror. Carries no knowledge of either runtime's layout -- it only knows how
to read this specific slice of C++ declaration syntax.
"""

from __future__ import annotations

import re

_CONSTEXPR_RE = re.compile(r"static\s+constexpr\s+[\w:]+\s+(\w+)\s*=\s*(-?\d+)\s*[uUlL]*\s*;")


def parse_constexpr_values(header_text, names):
    """``{name: int(value)}`` for each plain-integer ``static constexpr ... NAME = VALUE;`` in ``names``.

    Only matches a bare integer literal on the right-hand side (an optional
    ``u``/``U``/``l``/``L`` suffix is stripped). A constant defined as an expression
    (``1u << PARENT_BITS``) is not evaluated and is simply absent from the result --
    callers must treat a missing name as "not found", not "found as None".
    """
    found = {}
    for match in _CONSTEXPR_RE.finditer(header_text):
        name = match.group(1)
        if name in names:
            found[name] = int(match.group(2))
    return found


def parse_enum_class_members(header_text, enum_name):
    """``{member_name: int(value)}`` for a C++ ``enum class <enum_name> ... { ... };``."""
    body_match = re.search(rf"enum\s+class\s+{re.escape(enum_name)}\b[^{{]*\{{([^}}]*)\}}", header_text)
    if body_match is None:
        return {}
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"(\w+)\s*=\s*(\d+)", body_match.group(1))}
