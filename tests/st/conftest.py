# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conftest for scene tests (tests/st/).

Adds python/ and examples/scripts/ to sys.path for imports.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
for _d in [_ROOT / "python", _ROOT / "examples" / "scripts"]:
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)
