# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler runtime — public Python surface.

Host-side log filter setup happens in `ChipWorker.init` (see
`simpler.task_interface`): it `ctypes.CDLL`s libsimpler_log.so RTLD_GLOBAL,
calls its `simpler_log_init` C entry to seed the process-wide HostLogger, then
hands off to the C++ `_ChipWorker.init` which dlopens host_runtime.so (whose
`simpler_init` reads CANN dlog config off that same HostLogger, onboard only).
The level forwarded is a one-shot snapshot of the `simpler` Python logger.
Nothing log-related needs to happen at import time here.
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
