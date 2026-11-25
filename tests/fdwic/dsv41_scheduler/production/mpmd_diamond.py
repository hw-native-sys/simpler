#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build the single-block AIC -> {AIV, AIV} -> AIC diamond for production/run.sh."""

import argparse
import json
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
KERNELS = SUITE / "kernels/synthetic"
MASK = (1 << 64) - 1
# Each callable has its own entry and argument directions (1=IN, 3=INOUT).
STAGES = (
    ("A", "aic", "fork", (1, 3)),
    ("B", "aiv", "branch<true>", (1, 3)),
    ("C", "aiv", "branch<false>", (1, 3)),
    ("D", "aic", "join", (1, 1, 3)),
)


def checksum(value, steps):
    for _ in range(steps):
        value ^= (value << 13) & MASK
        value ^= value >> 7
        value ^= (value << 17) & MASK
    return value


def write_snapshot(dest, seed, steps):
    dest.mkdir()
    reference = dest / "reference"
    reference.mkdir()
    a = seed + 1
    b, c = a + 3, 2 * a + 1
    b_tag = checksum(b, steps if a % 2 else 0)
    c_tag = checksum(c, 0 if a % 2 else steps)
    values = [(seed, steps), (a, steps), (b, b_tag), (c, c_tag), (b + 2 * c, b_tag ^ c_tag)]
    poison = [MASK] * 16
    manifest = ["DSV41_NATIVE_V1", "1 0", str(len(values))]
    for i, pair in enumerate(values):
        expected = struct.pack("<16Q", *pair, *poison[2:])
        initial = expected if i == 0 else struct.pack("<16Q", *poison)
        (dest / f"input{i}.bin").write_bytes(initial)
        manifest.append(f"DEVICE 7 128 1 16 RAW input{i}.bin")
        if i:
            (reference / f"{i}.bin").write_bytes(expected)
    (dest / "rank0.txt").write_text("\n".join([*manifest, "0", "VALIDATE_ZERO 0", ""]))
    (dest / "slice.txt").write_text("0 3\n")
    (dest / "snapshot.txt").write_text("analytical\n")


def build(output, runner, runtime, pto_isa, slow_steps=4096, legacy=False):
    from simpler_setup import KernelCompiler, extract_text_section  # noqa: PLC0415

    output = output.resolve()
    runner = runner.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    compiler = KernelCompiler("a5")
    includes = compiler.get_orchestration_include_dirs(runtime)

    def compile_stage(stage):
        name, core, function, _ = stage
        entry = output / f"{name}.cpp"
        entry.write_text(
            '#include "mpmd_diamond_kernels.h"\n'
            'extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {\n'
            f"    diamond::{function}(args);\n"
            "}\n"
        )
        elf = compiler.compile_incore(
            str(entry), core_type=core, pto_isa_root=str(pto_isa), extra_include_dirs=[str(KERNELS), *includes]
        )
        (output / f"{name}.bin").write_bytes(extract_text_section(elf))

    with ThreadPoolExecutor(max_workers=5) as pool:
        host = pool.submit(compiler.compile_orchestration, runtime, str(KERNELS / "mpmd_diamond.cpp"))
        list(pool.map(compile_stage, STAGES))
        (output / "orchestration.so").write_bytes(host.result())
    manifest = ["5 1 3 3 3 3", "4"]
    for task, (name, _, _, dirs) in enumerate(STAGES):
        signature = " ".join(map(str, dirs))
        decl = " 1 1 0 0" if legacy else ""
        manifest.append(f'{task} {len(dirs)} {signature} "{name}.bin"{decl}')
    (output / "kernels.txt").write_text("\n".join(manifest) + "\n")
    subprocess.run(
        [runner, "--assemble-orchestration", "kernels.txt", "orchestration.so", "callable.bin"], cwd=output, check=True
    )
    names = {str(i): name for i, (name, _, _, _) in enumerate(STAGES)}
    for variant, seed, steps in (("balanced", 11, 0), ("b_last", 20, slow_steps), ("c_last", 31, slow_steps)):
        dest = output / variant
        write_snapshot(dest, seed, steps)
        shutil.copyfile(output / "callable.bin", dest / "callable.bin")
        (dest / "names.json").write_text(
            json.dumps({"level": 2, "orchestrator_name": "mpmd_diamond", "callable_id_to_name": names}, indent=2) + "\n"
        )
        (dest / "case.json").write_text(
            json.dumps(
                {
                    "flavor": "synthetic",
                    "case": "mpmd_diamond",
                    "execution": "mpmd_diamond",
                    "runtime": runtime,
                    "device_count": 1,
                    "variant": variant,
                    "seed": seed,
                    "branch_steps": steps,
                    "source": {
                        "tasks": 4,
                        "executions": 4,
                        "edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
                        "stages": [
                            {
                                "task_id": i,
                                "kernel_ids": [i, -1, -1] if core == "aic" else [-1, i, -1],
                                "block_num": 1,
                                "sync_start": False,
                            }
                            for i, (_, core, _, _) in enumerate(STAGES)
                        ],
                    },
                    "validation": "Compare four output buffers, including padding, against analytical references.",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory containing balanced, b_last and c_last bundles")
    parser.add_argument("--runner", required=True, type=Path, help="Matching native-runner used to pack the callable")
    parser.add_argument("--runtime", choices=("percounter", "host_build_graph"), default="percounter")
    parser.add_argument("--pto-isa", type=Path, default=SUITE.parents[2] / "build/pto-isa")
    parser.add_argument("--slow-steps", type=int, default=4096, help="Checksum iterations on the selected long branch")
    parser.add_argument("--legacy-sync", action="store_true", help="Declare synchronous kernels for the pre-PGO packer")
    args = parser.parse_args()
    if args.slow_steps <= 0:
        parser.error("--slow-steps must be positive")
    build(args.output, args.runner, args.runtime, args.pto_isa, args.slow_steps, args.legacy_sync)
