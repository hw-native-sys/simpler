#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Assemble PyPTO-lib, half-sync DSL and capacity-bounded all-sync DSL cases."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _export_native(destination, root, lib_root, native_moe_bundle):
    if lib_root is None:
        raise ValueError("Latest native MoE requires --pypto-lib-root pointing to dev/hbg")
    command = [
        sys.executable,
        str(root / "production/moe.py"),
        str(destination),
        str(destination / "compile"),
        "--compile-only",
        "--lib-root",
        str(lib_root),
    ]
    if native_moe_bundle is not None:
        command += ["--reuse-bundle", str(native_moe_bundle)]
    subprocess.run(command, check=True)


def _export_variant(bundle, capture, flavor, variant, policy, output, tools, lib_root=None, native_moe_bundle=None):
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "workloads" / flavor / "source.json").read_text())["cases"]
    for name, case in cases.items():
        destination = output / variant / name
        if case.get("execution") == "native_moe_host_control":
            _export_native(destination, root, lib_root, native_moe_bundle)
            continue
        destination.mkdir(parents=True, exist_ok=False)
        first, last = str(case["first_source_task"]), str(case["last_source_task"])
        metadata = []
        sync_reports = []
        for rank in range(2):
            plan = capture / f"rank{rank}/submissions.bin"
            detail = json.loads(
                subprocess.check_output([str(tools / "record-tool"), str(plan), first, last], text=True)
            )
            if (detail["tasks"], detail["executions"]) != (case["tasks"], case["executions"]):
                raise ValueError(f"{variant}/{name}: original task or physical execution count changed")
            metadata.append(detail)
            if flavor == "pypto_lib_dsl":
                sync_reports.append(
                    json.loads(
                        subprocess.check_output(
                            [
                                str(tools / "record-tool"),
                                "--all-sync-start-cap64" if policy == "all-cap64" else "--half-sync-start",
                                str(plan),
                                first,
                                last,
                                str(destination / f"plan-rank{rank}.bin"),
                            ],
                            text=True,
                        )
                    )
                )
            else:
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
                    "variant": variant,
                    "case": name,
                    "source": case,
                    **(
                        {
                            "sync_start_policy": {
                                "policy": "all_spmd_up_to_64"
                                if policy == "all-cap64"
                                else "half_spmd_preserve_existing",
                                "plans": sync_reports,
                            }
                        }
                        if sync_reports
                        else {}
                    ),
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
            f"{variant}/{name}: {case['tasks']} tasks, {case['executions']} executions, "
            f"{len(metadata[0]['boundary_buffers'])} boundary buffers"
        )


def export(bundle, capture, flavor, output, tools, sync_policy="both", lib_root=None, native_moe_bundle=None):
    """Export selected variants from the same unmodified source capture."""
    if sync_policy not in ("half", "all-cap64", "both"):
        raise ValueError(f"Unknown sync policy: {sync_policy}")
    policies = ("half", "all-cap64") if sync_policy == "both" else (sync_policy,)
    if flavor != "pypto_lib_dsl":
        policies = ("half",)
    variants = []
    for policy in policies:
        variant = "pypto_lib_dsl_all_sync64" if flavor == "pypto_lib_dsl" and policy == "all-cap64" else flavor
        _export_variant(bundle, capture, flavor, variant, policy, output, tools, lib_root, native_moe_bundle)
        variants.append(variant)
    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flavor", choices=["pypto_lib", "pypto_lib_dsl", "pypto_lib_latest"], required=True)
    parser.add_argument("--tools", type=Path, default=Path(__file__).resolve().parents[1] / "build/portable/host")
    parser.add_argument("--sync-policy", choices=["half", "all-cap64", "both"], default="both")
    parser.add_argument("--pypto-lib-root", type=Path)
    parser.add_argument("--native-moe-bundle", type=Path, help="Reuse an existing compatible native MoE bundle")
    args = parser.parse_args()
    export(
        args.bundle.resolve(),
        args.capture.resolve(),
        args.flavor,
        args.output.resolve(),
        args.tools.resolve(),
        args.sync_policy,
        args.pypto_lib_root,
        args.native_moe_bundle,
    )
