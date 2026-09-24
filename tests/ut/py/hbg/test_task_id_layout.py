# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Position-consistency check: Python ``hbg.TaskId``'s bit layout vs. the C++ header it mirrors.

task_id.h is the single source of truth for the hbg TaskId layout; this test parses
it directly, so a shift or width change there fails here rather than surfacing later
as a mislabeled task id in a trace.
"""

from __future__ import annotations

from pathlib import Path

from _cpp_constant_parser import parse_constexpr_values, parse_enum_class_members

from simpler_setup.tools.hbg import Space, TaskId

_HEADER_PATH = Path(__file__).resolve().parents[4] / "src" / "common" / "host_build_graph" / "task_id.h"


def _read_header():
    return _HEADER_PATH.read_text()


def test_shift_and_width_constants_match_the_cpp_header():
    names = {"SPACE_SHIFT", "PARENT_SHIFT", "PARENT_BITS"}
    found = parse_constexpr_values(_read_header(), names)

    assert found.keys() == names, (
        f"expected to find {sorted(names)} as `static constexpr` in {_HEADER_PATH}, only "
        f"found {sorted(found)} -- task_id.h's declaration style changed and this parser "
        f"no longer recognises it"
    )
    assert found["SPACE_SHIFT"] == TaskId.SPACE_SHIFT
    assert found["PARENT_SHIFT"] == TaskId.PARENT_SHIFT
    assert found["PARENT_BITS"] == TaskId.PARENT_BITS


def test_space_enum_matches_the_cpp_header():
    members = parse_enum_class_members(_read_header(), "Space")

    assert members, f"expected to find `enum class Space` in {_HEADER_PATH}"
    assert members == {space.name: space.value for space in Space}
