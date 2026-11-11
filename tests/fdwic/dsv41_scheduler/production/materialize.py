#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Materialize original prefix inputs for independent single-device replay."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from export_cases import link
from topology import device_count

ROOT = Path(__file__).resolve().parents[1]


def finalize(bundle, capture):
    case = json.loads((bundle / "case.json").read_text())
    detail = case["ranks"][0]
    lines = (bundle / "rank0.txt").read_text().splitlines()
    count = int(lines[2])
    if detail["uses_comm_context"]:
        raise ValueError("The selected slice uses a communication context")
    for index, required in enumerate(detail["case_external_bytes"]):
        if lines[3 + index].startswith("COMM ") and required:
            raise ValueError("The selected slice accesses a communication window")
    for rank in range(int(lines[1].split()[0])):
        if not json.loads((capture / f"rank{rank}/result.json").read_text())["passed"]:
            raise ValueError("Original preparation did not pass validation")
    source = capture / "rank0"
    (bundle / "reference").mkdir()
    for path in (source / "reference-data").glob("*.bin"):
        link(path, bundle / "reference" / path.name)
    for row in (source / "inputs.txt").read_text().splitlines():
        index_text, mode = row.split()
        index = int(index_text)
        fields = lines[3 + index].split()
        filename = f"snapshot-input-{index}.bin"
        link(source / "inputs" / f"{index}.bin", bundle / filename)
        lines[3 + index] = " ".join([*fields[:-2], mode, filename])
    # Unused original argument slots remain in the signature without communication allocations.
    for index in range(count):
        if lines[3 + index].startswith("COMM "):
            lines[3 + index] = "DEVICE 5 1 1 1 REPEAT zero.bin"
            reference = bundle / "reference" / f"{index}.bin"
            if reference.exists():
                reference.unlink()
            link(bundle / "zero.bin", reference)
    lines = ["VALUE 0" if line == "COMM" else line for line in lines]
    lines[1] = "1 0"
    case["preparation"] = (
        "Original prefix inputs are materialized once; single-device runs replay only the selected slice."
    )
    case["validation"] = "Compare bytes with the original fused prefix-plus-case result."
    (bundle / "snapshot.txt").write_text("rank0\n")
    (bundle / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
    (bundle / "rank0.txt").write_text("\n".join(lines) + "\n")
    (bundle / "rank1.txt").unlink()


def materialize(bundle, runtime, capture, env=None):
    device_count(bundle, materialize=True)
    case = json.loads((bundle / "case.json").read_text())
    detail = case["ranks"][0]
    lines = (bundle / "rank0.txt").read_text().splitlines()
    if detail["uses_comm_context"] or any(
        required and lines[3 + index].startswith("COMM ")
        for index, required in enumerate(detail["case_external_bytes"])
    ):
        raise ValueError("Cannot materialize a single-device snapshot of a communicating slice")
    prefix_communicates = detail.get("prefix_uses_comm_context", True) or any(
        required and lines[3 + index].startswith("COMM ") for index, required in enumerate(detail["external_bytes"])
    )
    if not prefix_communicates:
        lines[1] = "1 0"
        for index in range(int(lines[2])):
            if lines[3 + index].startswith("COMM "):
                lines[3 + index] = "DEVICE 5 1 1 1 REPEAT zero.bin"
        lines = ["VALUE 0" if line == "COMM" else line for line in lines]
        (bundle / "rank0.txt").write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            "bash",
            str(ROOT / "production/run.sh"),
            str(bundle),
            str(runtime),
            str(capture),
            "0",
            "0",
            "1",
            "--materialize",
        ],
        env=env,
        check=True,
    )
    finalize(bundle, capture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    materialize(args.bundle.resolve(), args.runtime.resolve(), args.output.resolve(), os.environ.copy())
