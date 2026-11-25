#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the six-stage native A5 scheduling case without a model capture."""

import argparse
import json
import subprocess
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASE = "full_wave_mix_chain"
ROOT = Path(__file__).resolve().parents[4]
KERNELS = Path(__file__).resolve().parents[1] / "kernels/synthetic"
# Kernel IDs name the AIC, AIV0 and AIV1 members in this order.
FULL_MIX = (0, 1, 1)
STAGES = (
    (32, (0, -1, -1)),
    (32, FULL_MIX),
    (64, (-1, 1, -1)),
    (32, FULL_MIX),
    (1, (0, 1, -1)),
    (32, FULL_MIX),
)
WORDS_PER_MEMBER = 16


def write_inputs(output: Path) -> None:
    # Each member owns a 128-byte line and writes only its first int64.
    seed = array("q", range(11, 11 + 32 * WORDS_PER_MEMBER))
    (output / "input0.bin").write_bytes(seed.tobytes())
    manifest = ["DSV41_NATIVE_V1", "1 0", "7", f"DEVICE 7 {len(seed) * 8} 1 {len(seed)} RAW input0.bin"]
    previous = seed[::WORDS_PER_MEMBER]
    for index, (blocks, kernels) in enumerate(STAGES, 1):
        members = blocks * sum(kernel >= 0 for kernel in kernels)
        words = array("q", [-1]) * (WORDS_PER_MEMBER * members)
        name = f"input{index}.bin"
        (output / name).write_bytes(words.tobytes())
        manifest.append(f"DEVICE 7 {len(words) * 8} 1 {len(words)} RAW {name}")
        # The snapshot runner appends phase=2 as the orchestration's scalar.
        previous = array("q", (previous[m % len(previous)] + 2 * index + m for m in range(members)))
        words[::WORDS_PER_MEMBER] = previous
        (output / "reference" / f"{index}.bin").write_bytes(words.tobytes())
    (output / "rank0.txt").write_text("\n".join([*manifest, "0", "VALIDATE_ZERO 0", ""]))
    (output / "slice.txt").write_text("0 5\n")
    (output / "snapshot.txt").write_text("analytical\n")


def build_bundle(output: Path, runtime: str, runner: Path) -> Path:
    """Build for this runtime's block-index ABI and write independent numerical references."""
    # Listing cases or reading an existing bundle does not require a compiler installation.
    from simpler_setup import KernelCompiler, extract_text_section  # noqa: PLC0415

    if runtime not in ("host_build_graph", "percounter"):
        raise ValueError(f"Unsupported full-wave runtime: {runtime}")
    output = output.resolve()
    runner = runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference").mkdir()
    compiler = KernelCompiler("a5")
    includes = compiler.get_orchestration_include_dirs(runtime)

    def compile_member(core):
        device_elf = compiler.compile_incore(
            str(KERNELS / "full_wave_mix_member.cpp"),
            core_type=core,
            pto_isa_root=str(ROOT / "build/pto-isa"),
            extra_include_dirs=includes,
        )
        (output / f"{core}.bin").write_bytes(extract_text_section(device_elf))

    with ThreadPoolExecutor(max_workers=3) as pool:
        host = pool.submit(compiler.compile_orchestration, runtime, str(KERNELS / "full_wave_mix_chain.cpp"))
        members = [pool.submit(compile_member, core) for core in ("aic", "aiv")]
        (output / "orchestration.so").write_bytes(host.result())
        for future in members:
            future.result()

    # Native runner wire format: seven tensor directions (IN then six INOUT),
    # two children with directions IN/INOUT.
    (output / "kernels.txt").write_text('7 1 3 3 3 3 3 3\n2\n0 2 1 3 "aic.bin"\n1 2 1 3 "aiv.bin"\n')
    subprocess.run(
        [runner, "--assemble-orchestration", "kernels.txt", "orchestration.so", "callable.bin"],
        cwd=output,
        check=True,
    )
    write_inputs(output)
    stages = [
        {"task_id": task, "block_num": blocks, "kernel_ids": list(kernels), "sync_start": blocks > 1}
        for task, (blocks, kernels) in enumerate(STAGES)
    ]
    executions = sum(blocks * sum(kernel >= 0 for kernel in kernels) for blocks, kernels in STAGES)
    case = {
        "flavor": "synthetic",
        "variant": "synthetic",
        "case": CASE,
        "execution": CASE,
        "runtime": runtime,
        "device_count": 1,
        "validation": "Compare all six writable tensors, including padding, byte-for-byte against analytical outputs.",
        "source": {"tasks": len(stages), "executions": executions, "stages": stages},
    }
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    names = {
        "level": 2,
        "orchestrator_name": CASE,
        "callable_id_to_name": {"0": "chain_member_C", "1": "chain_member_V"},
    }
    (output / "names.json").write_text(json.dumps(names, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the prepared bundle")
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), required=True)
    parser.add_argument("--runner", type=Path, required=True)
    args = parser.parse_args()
    print(build_bundle(args.output, args.runtime, args.runner))
