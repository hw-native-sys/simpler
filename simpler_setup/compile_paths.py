# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stable compiler-visible spellings for files under the simpler checkout."""

from __future__ import annotations

import os
from pathlib import Path

from .environment import PROJECT_ROOT

CPU_SIM_FORCED_HEADERS = ("pto/cpu/target_config.hpp", "cpu_sim_target.h")


def compiler_visible_path(path: str | Path) -> Path:
    """Return a stable path relative to the checkout, or an absolute external path."""
    absolute_path = Path(os.path.abspath(path))
    project_root = Path(os.path.abspath(PROJECT_ROOT))
    try:
        return absolute_path.relative_to(project_root)
    except ValueError:
        return absolute_path
