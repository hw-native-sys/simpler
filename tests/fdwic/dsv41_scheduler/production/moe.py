#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the original dev/hbg PyPTO-lib MoE with live Host control inputs."""

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def create_bundle(bundle, lib):
    lib = lib.resolve()
    if not (lib / "docs/models/hbg-compatibility.md").is_file():
        raise ValueError("Use the PyPTO-lib dev/hbg checkout for native MoE")
    commit = subprocess.check_output(["git", "-C", str(lib), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(lib), "diff", "--exit-code", "HEAD", "--", "models", "golden"], check=True)
    bundle.mkdir(parents=True, exist_ok=False)
    case = {
        "case": "moe",
        "flavor": "pypto_lib",
        "device_count": 2,
        "execution": "native_moe_host_control",
        "lib_root": str(lib),
        "lib_commit": commit,
    }
    (bundle / "case.json").write_text(json.dumps(case, indent=2) + "\n")


def control_reads(program):
    """Require an actual orchestration read of num_tokens in the compiled program."""
    reads = []
    for path in sorted(program.rglob("*.cpp")):
        if "orchestration" not in path.parts:
            continue
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if text.lstrip().startswith("//"):
                continue
            if re.search(r"get_tensor_data\s*<\s*int32_t\s*>\s*\([^,]*num_tokens\b", text):
                reads.append({"file": str(path.relative_to(program)), "line": line, "code": text.strip()})
    if not reads:
        raise ValueError("MoE orchestration does not read num_tokens with get_tensor_data<int32_t>")
    return reads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--lib-root", type=Path, help="Create a new bundle from this dev/hbg checkout")
    parser.add_argument("--level", type=int, choices=(0, 1, 3), default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--runtime-lib", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.rounds < 1:
        parser.error("Use at least one warmup and one measured round")
    bundle, output = args.bundle.resolve(), args.output.resolve()
    if args.lib_root:
        if not args.compile_only:
            parser.error("--lib-root requires --compile-only; compile before reserving devices")
        create_bundle(bundle, args.lib_root)
    case = json.loads((bundle / "case.json").read_text())
    if case.get("execution") != "native_moe_host_control":
        parser.error("Expected a native PyPTO-lib MoE bundle")
    lib = Path(case["lib_root"])
    model = lib / "models/deepseek_v4_1_flash"
    if not (model / "decode_moe.py").is_file():
        parser.error("The bundle's dev/hbg source checkout is unavailable")
    commit = subprocess.check_output(["git", "-C", str(lib), "rev-parse", "HEAD"], text=True).strip()
    if commit != case["lib_commit"]:
        parser.error("The Lib checkout has moved; regenerate the native MoE bundle")
    subprocess.run(["git", "-C", str(lib), "diff", "--exit-code", "HEAD", "--", "models", "golden"], check=True)
    if not args.compile_only and not os.environ.get("TASK_DEVICE"):
        parser.error("Run through production/run.sh to reserve two devices")
    devices = [0, 1] if args.compile_only else [int(x) for x in os.environ["TASK_DEVICE"].split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error("MoE requires an exclusive reservation of two distinct A5 devices")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    # The model's config module reads these topology arguments during import.
    sys.argv = [str(model / "decode_moe.py"), "--ep", "2", "--experts-per-rank", "64"]
    sys.path[:0] = [str(model), str(lib)]
    moe = importlib.import_module("decode_moe")
    golden = importlib.import_module("golden")
    backend = importlib.import_module("pypto.backend")
    ir = importlib.import_module("pypto.ir")
    passes = importlib.import_module("pypto.pypto_core.passes")
    runtime = importlib.import_module("pypto.runtime")
    environment = importlib.import_module("simpler_setup.environment")
    pin = importlib.import_module("pypto.runtime.runtime_pin").check_runtime_pin()

    runtime_lib = (environment.PROJECT_ROOT / "build/lib/a5/onboard").resolve()
    if args.runtime_lib and args.runtime_lib.resolve() != runtime_lib:
        parser.error(f"Native MoE uses the installed Simpler runtime at {runtime_lib}; select its Python environment")

    program = bundle / "program"
    config = runtime.RunConfig(
        backend_type=backend.BackendType.Ascend950,
        execution_mode=runtime.ExecutionMode.ONBOARD,
        distributed_config=ir.DistributedConfig(device_ids=devices, num_sub_workers=0),
        save_kernels=True,
        save_kernels_dir=str(program),
        enable_chip_swimlane=args.level,
        ring_heap=2 * 1024**3,
    )
    report = {
        "case": "moe",
        "flavor": "pypto_lib",
        "source_commit": case["lib_commit"],
        "runtime_lib": str(runtime_lib),
        "runtime_commit": pin.installed,
        "profiling_level": args.level,
        "checks": [],
    }
    counts_list = [(72, 72)] if args.compile_only else [(0, 6), (6, 0), (72, 72)]
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        for counts in counts_list:
            measure = not args.compile_only and counts == (72, 72)
            os.environ["PYPTO_BENCH"] = "1" if measure else "0"
            os.environ["PYPTO_BENCH_WARMUP"] = str(args.warmup)
            os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
            specs = moe.build_moe_specs(counts, initialize=not args.compile_only)
            result = golden.run(
                fn=moe.l3_moe,
                specs=specs,
                golden_fn=moe.golden_moe,
                save_data=False,
                compile_only=args.compile_only,
                runtime_dir=None if args.compile_only else str(program),
                config=config,
                rtol=1e-3,
                atol=1e-3,
                compare_fn={
                    "next_pre_mix": golden.ratio_allclose(atol=2.5e-5, rtol=5e-3),
                    "x_mixed": golden.ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                    "x_next": moe._local_hc_compare(counts),
                },
            )
            report["checks"].append({"counts": counts, "passed": result.passed, "error": result.error})
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            if not result.passed:
                raise RuntimeError(result.error or "MoE golden failed")
            report["control_reads"] = control_reads(Path(result.work_dir))
            if measure:
                if result.bench is None:
                    raise RuntimeError("MoE correctness passed, but performance samples are unavailable")
                samples = result.bench.device_wall_us
                if len(samples) != args.rounds or any(value <= 0 for value in samples):
                    raise RuntimeError("Missing or invalid measured Device samples")
                report["performance"] = {
                    "rounds": args.rounds,
                    "warmup": args.warmup,
                    "device_wall_us": samples,
                    "host_wall_us": result.bench.host_wall_us,
                    "device_us_median": result.bench.device_us_median,
                    "device_us_p95": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
                    "per_rank_device_us": result.bench.per_rank("device"),
                }
    report["compile_only"] = args.compile_only
    report["passed"] = True
    (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"MoE Host control {'compilation' if args.compile_only else 'validation'} passed: {output}")


if __name__ == "__main__":
    main()
