#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""List the active variant/case paths for a complete scheduler benchmark run."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def selected_cases():
    for flavor in ("pypto_lib", "pypto_lib_dsl", "pypto_lib_latest"):
        cases = json.loads((ROOT / "workloads" / flavor / "source.json").read_text())["cases"]
        variants = (flavor, "pypto_lib_dsl_all_sync64") if flavor == "pypto_lib_dsl" else (flavor,)
        for variant in variants:
            for case in cases:
                yield f"{variant}/{case}"


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("\n".join(selected_cases()))
