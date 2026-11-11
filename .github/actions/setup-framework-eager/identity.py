# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Identity entry point for comment workflows loaded from the default branch."""

import runpy
from pathlib import Path


def main():
    identity = Path(__file__).resolve().parents[1] / "setup-framework" / "identity.py"
    runpy.run_path(str(identity))["main"](destination=Path("outputs/framework-eager-ci/identity.json"))


if __name__ == "__main__":
    main()
