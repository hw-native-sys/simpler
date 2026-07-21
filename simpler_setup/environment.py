# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Centralized path management.

PROJECT_ROOT auto-resolves between two layouts:
  - wheel install: simpler_setup/_assets/{src,build/lib} populated by CMakeLists install()
  - source tree / editable: repo root with src/ and build/lib/ in original positions
"""

from pathlib import Path


def _resolve_project_root() -> Path:
    assets = Path(__file__).resolve().parent / "_assets"
    if (assets / "src").is_dir():
        return assets
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _resolve_project_root()
