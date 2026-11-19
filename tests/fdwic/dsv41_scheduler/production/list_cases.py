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
    manifest = ROOT / "workloads/pypto_lib_latest/source.json"
    for name in json.loads(manifest.read_text())["cases"]:
        yield f"pypto_lib_latest/{name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), default="host_build_graph")
    parser.parse_args()
    print("\n".join(selected_cases()))
