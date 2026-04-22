# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest configuration for Python unit tests (tests/ut/py/).

Adds project directories to sys.path so that:
- ``import simpler_setup`` works (PROJECT_ROOT on path)
- ``from simpler import env_manager`` works (python/ on path)
- legacy ``import env_manager`` works (python/simpler/ on path)
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent.parent

# Order matters: PROJECT_ROOT first (so ``import simpler_setup`` works as a
# package), then python/ so ``from simpler import env_manager`` resolves, then
# python/simpler/ so legacy ``import env_manager`` works.
for _d in [
    _ROOT,
    _ROOT / "python",
    _ROOT / "python" / "simpler",
]:
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return _ROOT
