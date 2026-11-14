# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Allow explicit CPU-only checks without the repository's runtime bootstrap."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for directory in (ROOT, ROOT / "python"):
    sys.path.insert(0, str(directory))


def pytest_configure(config):
    config.addinivalue_line("markers", "manual: explicitly invoked benchmark check")
