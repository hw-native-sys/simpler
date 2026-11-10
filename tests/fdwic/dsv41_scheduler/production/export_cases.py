#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Assemble nine original-kernel cases from a production Host submission capture."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def export(bundle, capture, flavor, output, tools):
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "workloads" / flavor / "source.json").read_text())["cases"]
    for name, case in cases.items():
        destination = output / flavor / name
        destination.mkdir(parents=True, exist_ok=False)
        first, last = str(case["first_source_task"]), str(case["last_source_task"])
        metadata = []
        for rank in range(2):
            plan = capture / f"rank{rank}/submissions.bin"
            detail = json.loads(
                subprocess.check_output([str(tools / "record-tool"), str(plan), first, last], text=True)
            )
            if (detail["tasks"], detail["executions"]) != (case["tasks"], case["executions"]):
                raise ValueError(f"{flavor}/{name}: original task or physical execution count changed")
            metadata.append(detail)
            link(plan, destination / f"plan-rank{rank}.bin")
            lines = (bundle / f"rank{rank}.txt").read_text().splitlines()
            count = int(lines[2])
            # The replay entry takes backing buffers. Original kernel views,
            # offsets and strides come from the capture. Do not upload the
            # unreferenced tail of all 40 layers' weights for a small case.
            for index, required in enumerate(detail["external_bytes"]):
                fields = lines[3 + index].split()
                if fields[0] == "COMM":
                    continue
                source = bundle / fields[-1]
                value = source.read_bytes()
                repeat = fields[-2] == "REPEAT"
                size = max(required, len(value) if repeat else 1)
                if size > int(fields[2]):
                    raise ValueError("Captured kernel view exceeds original backing")
                if repeat and size % len(value):
                    size = (size + len(value) - 1) // len(value) * len(value)
                if not repeat:
                    value = value[:size]
                    if len(value) != size:
                        raise ValueError("Original initializer does not cover kernel view")
                filename = f"input-rank{rank}-{index}.bin"
                (destination / filename).write_bytes(value)
                lines[3 + index] = f"{fields[0]} 5 {size} 1 1 {fields[-2]} {filename}"
            extra = detail["boundary_buffers"]
            added = []
            for buffer in extra:
                if buffer["bytes"] >= 2**32:
                    raise ValueError("Boundary buffer exceeds a single byte-view extent")
                added.append(f"DEVICE 5 {buffer['bytes']} 1 {buffer['bytes']} REPEAT zero.bin")
            lines[2] = str(count + len(added))
            lines[3 + count : 3 + count] = added
            (destination / f"rank{rank}.txt").write_text("\n".join(lines) + "\n")
        if len(metadata[0]["boundary_buffers"]) != len(metadata[1]["boundary_buffers"]):
            raise ValueError("Ranks have different entry tensor counts")
        (destination / "zero.bin").write_bytes(b"\0")
        (destination / "slice.txt").write_text(f"{first} {last}\n")
        subprocess.run(
            [
                str(tools / "native-runner"),
                "--repack-orchestration",
                str(bundle / "original-callable.bin"),
                str(tools / "libreplay.so"),
                str(destination / "callable.bin"),
                str(len(metadata[0]["boundary_buffers"])),
            ],
            check=True,
        )
        (destination / "case.json").write_text(
            json.dumps(
                {
                    "flavor": flavor,
                    "case": name,
                    "source": case,
                    "ranks": metadata,
                    "device_count": 2 if name in ("moe", "layers_19_21") else 1,
                    "preparation": "Run the original prefix into persistent cut inputs; exclude it from measured runs.",
                    "validation": "Compare writable backing against a fused original-prefix-plus-case run.",
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"{flavor}/{name}: {case['tasks']} tasks, {case['executions']} executions, "
            f"{len(metadata[0]['boundary_buffers'])} boundary buffers"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flavor", choices=["pypto_lib", "pypto_lib_dsl"], required=True)
    parser.add_argument("--tools", type=Path, default=Path(__file__).resolve().parents[1] / "build/portable/host")
    args = parser.parse_args()
    export(args.bundle.resolve(), args.capture.resolve(), args.flavor, args.output.resolve(), args.tools.resolve())
