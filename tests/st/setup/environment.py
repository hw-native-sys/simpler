# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Centralized path management for scene test setup."""

import sys
from pathlib import Path

# Single source of truth: project root derived from this file's location
# tests/st/setup/environment.py → 4 parents up → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Key directories
PYTHON_DIR = PROJECT_ROOT / "python"
EXAMPLES_SCRIPTS_DIR = PROJECT_ROOT / "examples" / "scripts"
BUILD_CACHE_DIR = PROJECT_ROOT / "build" / "cache"
BUILD_LIB_DIR = PROJECT_ROOT / "build" / "lib"


def ensure_python_path() -> None:
    """Add python/ to sys.path for task_interface imports."""
    p = str(PYTHON_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
