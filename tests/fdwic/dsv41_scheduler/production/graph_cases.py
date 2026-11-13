#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare matched Flat/native Graph cases using the original model kernels."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from export_cases import link

ATTENTION = {"swa", "c1a_reuse"}
MULTICARD = {"moe", "layers_19_21"}


def prepare(bundle, output, runner, replay, mode, repetitions=None):
    metadata = json.loads((bundle / "case.json").read_text())
    case = metadata["case"]
    if case not in ATTENTION | MULTICARD:
        raise ValueError("Graph benchmark supports two attention and two EP2 cases")
    if mode not in {"flat", "graph"}:
        raise ValueError("mode must be flat or graph")
    if repetitions is None:
        repetitions = 8 if case in ATTENTION else 1
    if not 1 <= repetitions <= 128 or (case in MULTICARD and repetitions != 1):
        raise ValueError("Attention supports 1..128 repetitions; EP2 uses one original segment")
    if output == bundle or bundle in output.parents:
        raise ValueError("Output must be outside the source bundle")
    if output.exists():
        raise FileExistsError(output)
    if (bundle / "graph.txt").exists():
        raise ValueError("Input must be an original single-segment bundle")
    shutil.copytree(bundle, output, copy_function=link)
    (output / "callable.bin").unlink()
    subprocess.run(
        [
            str(runner),
            "--repack-orchestration",
            str(bundle / "callable.bin"),
            str(replay),
            str(output / "callable.bin"),
        ],
        check=True,
    )
    (output / "graph.txt").write_text(f"{mode} {repetitions}\n")
    metadata["graph_benchmark"] = {
        "mode": mode,
        "repetitions": repetitions,
        "ordering": "serial logical completion between repeated segments",
        "input_policy": "reuse the fixed snapshot and backing; no token-position advancement",
    }
    (output / "case.json").unlink()
    (output / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata["graph_benchmark"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--mode", choices=["flat", "graph"], required=True)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.bundle.resolve(),
                args.output.resolve(),
                args.runner.resolve(),
                args.replay.resolve(),
                args.mode,
                args.repetitions,
            )
        )
    )
