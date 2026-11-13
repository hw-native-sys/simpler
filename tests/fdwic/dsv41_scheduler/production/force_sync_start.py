#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare a DSL sync-start variant, preserving tasks above the block limit."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def prepare(bundle, output, record_tool, max_blocks=64):
    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    metadata = json.loads((bundle / "case.json").read_text())
    if metadata["flavor"] != "pypto_lib_dsl":
        raise ValueError("This variant is defined for DSL cases")
    if output == bundle or bundle in output.parents:
        raise ValueError("Output must be outside the source bundle")
    if output.exists():
        raise FileExistsError(output)
    first, last = (bundle / "slice.txt").read_text().split()
    shutil.copytree(bundle, output, copy_function=os.link)
    reports = []
    for source in sorted(bundle.glob("plan-rank*.bin")):
        target = output / source.name
        target.unlink()
        reports.append(
            json.loads(
                subprocess.check_output(
                    [str(record_tool), "--force-sync-start", str(source), first, last, str(target), str(max_blocks)],
                    text=True,
                )
            )
        )
    target = output / "case.json"
    target.unlink()
    metadata["sync_start_variant"] = {
        "policy": "multi_block_compute_up_to_limit",
        "max_blocks": max_blocks,
        "plans": reports,
    }
    target.write_text(json.dumps(metadata, indent=2) + "\n")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--record-tool", type=Path, required=True)
    parser.add_argument("--max-blocks", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.bundle.resolve(), args.output.resolve(), args.record_tool.resolve(), args.max_blocks))
    )
