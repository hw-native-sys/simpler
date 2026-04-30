# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler runtime — public Python surface.

CANN dlog bootstrap and host-side log filter setup are performed by the
platform SO's `simpler_init` C entry, called once from `ChipWorker::init()`
with the snapshot of the `simpler` Python logger level. No Python-side
ctypes / dlopen is needed here.
"""

# Importing _log auto-configures the simpler logger to V5 if unset.
from ._log import (
    DEFAULT_THRESHOLD,
    NUL,
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    get_current_config,
    get_logger,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "NUL",
    "V0",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "get_current_config",
    "get_logger",
]
