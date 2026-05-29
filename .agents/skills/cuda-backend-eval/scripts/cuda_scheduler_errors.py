#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared CUDA persistent-device scheduler error labels."""

from __future__ import annotations

from typing import Any

SCHEDULER_ERROR_NAMES = {
    0: "none",
    1: "unsupported_func_id",
    2: "invalid_dependent_id",
    3: "invalid_dependent_range",
    4: "fanin_underflow",
    5: "initial_fanin_mismatch",
    6: "no_root_task",
    7: "unreachable_task",
    8: "duplicate_dependent",
    9: "self_dependent",
}


def scheduler_error_code_label(code: Any) -> str:
    if not isinstance(code, int):
        return str(code)
    name = SCHEDULER_ERROR_NAMES.get(code)
    if name is None:
        return str(code)
    if code == 0:
        return "0"
    return f"{code}({name})"
