#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the case topology before reserving any devices."""

import json
import sys
from pathlib import Path


def device_count(bundle, materialize=False):
    case_path = bundle / "case.json"
    case = json.loads(case_path.read_text()) if case_path.exists() else {}
    if case.get("execution") == "native_moe_host_control":
        if materialize or case.get("flavor") != "pypto_lib" or case.get("case") != "moe":
            raise ValueError("Native MoE is an EP2 case without cut-input materialization")
        return 2
    if case.get("flavor") == "pypto_lib" and case.get("case") == "moe":
        raise ValueError("Regenerate PyPTO-lib MoE with production/moe.py; frozen MoE plans are unsupported")
    lines = (bundle / "rank0.txt").read_text().splitlines()
    if lines[0] != "DSV41_NATIVE_V1":
        raise ValueError("Invalid input manifest")
    ranks, window = map(int, lines[1].split())
    if ranks not in (1, 2) or window < 0:
        raise ValueError("Expected a local single-device case or a two-device EP2 case")
    expected = case.get("device_count", 2)
    if materialize:
        if expected != 1 or (bundle / "snapshot.txt").exists() or not (bundle / "slice.txt").is_file():
            raise ValueError("Only unprepared local slices can materialize inputs")
    elif ranks != expected:
        raise ValueError("Prepare single-device inputs with materialize.py first")
    if ranks == 1:
        if window or (not materialize and not (bundle / "snapshot.txt").is_file()):
            raise ValueError("Single-device cases require snapshots without communication windows")
        if any(line.startswith("COMM ") or line == "COMM" for line in lines[3:]):
            raise ValueError("Single-device manifest still contains communication arguments")
    for rank in range(1, ranks):
        other = (bundle / f"rank{rank}.txt").read_text().splitlines()
        if other[:2] != lines[:2]:
            raise ValueError("Rank manifests disagree on topology")
    return ranks


if __name__ == "__main__":
    try:
        print(device_count(Path(sys.argv[1]), len(sys.argv) > 2 and sys.argv[2] == "--materialize"))
    except (ValueError, OSError, IndexError) as error:
        sys.exit(str(error))
