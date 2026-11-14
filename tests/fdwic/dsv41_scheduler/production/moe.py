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
import ctypes
import hashlib
import importlib
import json
import math
import os
import re
import runpy
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def create_bundle(bundle, lib, reuse=None):
    lib = lib.resolve()
    if not (lib / "docs/models/hbg-compatibility.md").is_file():
        raise ValueError("Use the PyPTO-lib dev/hbg checkout for native MoE")
    commit = subprocess.check_output(["git", "-C", str(lib), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(lib), "diff", "--exit-code", "HEAD", "--", "models", "golden"], check=True)
    bundle.mkdir(parents=True, exist_ok=False)
    case = {
        "case": "moe",
        "flavor": "pypto_lib_latest",
        "device_count": 2,
        "execution": "native_moe_host_control",
        "lib_root": str(lib),
        "lib_commit": commit,
    }
    (bundle / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    if reuse is not None:
        previous = json.loads((reuse / "case.json").read_text())
        if previous.get("execution") != "native_moe_host_control" or previous.get("lib_commit") != commit:
            raise ValueError("Reused native MoE bundle has a different model or execution contract")
        shutil.copytree(reuse / "program", bundle / "program")
        for name in ("native-callable.bin", "native-callable.json"):
            shutil.copy2(reuse / name, bundle / name)


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


def prepare_callable(bundle, compiled):
    """Store the native orchestration; execution never substitutes a submission replay."""
    _task_interface = importlib.import_module("_task_interface")
    _assemble_chip_callables = importlib.import_module("pypto.runtime.distributed_runner")._assemble_chip_callables

    callables, _, _ = _assemble_chip_callables(compiled)
    if set(callables) != {"l2_moe"}:
        raise ValueError("Expected the original l2_moe chip program")
    call = callables["l2_moe"]
    data = ctypes.string_at(int(call.buffer_ptr()), int(call.buffer_size()))
    program = bundle / "program"
    config = runpy.run_path(str(program / "next_levels/l2_moe/kernel_config.py"))
    source = Path(config["ORCHESTRATION"]["source"])
    (bundle / "native-callable.bin").write_bytes(data)
    (bundle / "native-callable.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(data).hexdigest(),
                "binding_commit": _task_interface.__build_commit__,
                "source_commit": json.loads((bundle / "case.json").read_text())["lib_commit"],
                "host_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "runtime_config": config["RUNTIME_CONFIG"],
            },
            indent=2,
        )
        + "\n"
    )


def install_callable(bundle, runtime):
    """Reuse one native callable on the explicitly selected compatible runtime."""
    _task_interface = importlib.import_module("_task_interface")
    distributed_runner = importlib.import_module("pypto.runtime.distributed_runner")
    ChipCallable = importlib.import_module("simpler.task_interface").ChipCallable

    manifest = json.loads((bundle / "native-callable.json").read_text())
    case = json.loads((bundle / "case.json").read_text())
    program = bundle / "program"
    data = (bundle / "native-callable.bin").read_bytes()
    source = program / "next_levels/l2_moe/orchestration/l2_moe.cpp"
    if (
        manifest["binding_commit"] != _task_interface.__build_commit__
        or manifest["source_commit"] != case["lib_commit"]
        or hashlib.sha256(data).hexdigest() != manifest["sha256"]
        or hashlib.sha256(source.read_bytes()).hexdigest() != manifest["host_source_sha256"]
    ):
        raise ValueError("Native MoE callable/source/binding changed; prepare the bundle again")
    control_reads(program)

    def assemble(compiled):
        if Path(compiled.output_dir).resolve() != program.resolve():
            raise ValueError("Native MoE loader received a different compiled program")
        callables = distributed_runner._ChipCallables()
        callables.add("l2_moe", ChipCallable.from_bytes(data), manifest["runtime_config"])
        return callables, runtime, False

    distributed_runner._assemble_chip_callables = assemble


def validate_report(report, case, level=None):
    """Require all PR3038 control cases and complete Host/Device measurements."""
    checks = report.get("checks", [])
    required = {(0, 6), (6, 0), (72, 72)}
    if (
        report.get("passed") is not True
        or report.get("compile_only") is not False
        or len(checks) != 3
        or any(check.get("passed") is not True for check in checks)
        or {tuple(check.get("counts", [])) for check in checks} != required
    ):
        raise ValueError("Native MoE has not passed all Host-control goldens")
    if report.get("source_commit") != case["lib_commit"] or (
        level is not None and report.get("profiling_level") != level
    ):
        raise ValueError("Native MoE report source or profiling level mismatch")
    if not report.get("control_reads"):
        raise ValueError("Native MoE is missing its executable control-read evidence")
    perf = report.get("performance", {})
    rounds = perf.get("rounds", 0)
    if not isinstance(rounds, int) or rounds < 1 or perf.get("warmup", 0) < 1:
        raise ValueError("Native MoE needs warmup and measured rounds")
    samples = perf.get("device_wall_us", [])
    host = perf.get("host_wall_us", [])
    ranks = list(perf.get("per_rank_device_us", {}).values())
    if len(ranks) != 2 or any(len(values) != rounds for values in [samples, host, *ranks]):
        raise ValueError("Native MoE is missing Host/Device or rank samples")
    if any(not math.isfinite(x) or x <= 0 for values in [samples, host, *ranks] for x in values):
        raise ValueError("Native MoE has invalid timing samples")
    if any(not math.isclose(samples[i], max(rank[i] for rank in ranks), abs_tol=1e-6) for i in range(rounds)):
        raise ValueError("Native MoE Device samples must use the per-round rank maximum")


def _create_requested_bundle(args, bundle, parser):
    if args.lib_root:
        if not args.compile_only:
            parser.error("--lib-root requires --compile-only; compile before reserving devices")
        create_bundle(bundle, args.lib_root, args.reuse_bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--lib-root", type=Path, help="Create a new bundle from this dev/hbg checkout")
    parser.add_argument("--reuse-bundle", type=Path, help="Reuse a prepared native bundle from the same model/binding")
    parser.add_argument("--level", type=int, choices=(0, 1, 3), default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--runtime-lib", type=Path)
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), default="host_build_graph")
    args = parser.parse_args()
    if args.warmup < 1 or args.rounds < 1:
        parser.error("Use at least one warmup and one measured round")
    bundle, output = args.bundle.resolve(), args.output.resolve()
    _create_requested_bundle(args, bundle, parser)
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

    runtime_lib = (args.runtime_lib or environment.PROJECT_ROOT / "build/lib/a5/onboard").resolve()
    RuntimeBuilder = importlib.import_module("simpler_setup.runtime_builder").RuntimeBuilder

    RuntimeBuilder._LIB_DIR = runtime_lib.parent.parent
    binaries = RuntimeBuilder("a5").get_binaries(args.runtime, build=False)
    runtime_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (binaries.host_path, binaries.aicpu_path, binaries.aicore_path, binaries.dispatcher_path)
        if path is not None
    }

    program = bundle / "program"
    if args.compile_only and (bundle / "native-callable.json").is_file():
        install_callable(bundle, args.runtime)
        (output / "result.json").write_text(
            json.dumps(
                {
                    "case": "moe",
                    "flavor": "pypto_lib_latest",
                    "source_commit": case["lib_commit"],
                    "compile_only": True,
                    "compilation_passed": True,
                    "passed": False,
                    "checks": [],
                    "control_reads": control_reads(program),
                    "reused_callable": True,
                },
                indent=2,
            )
            + "\n"
        )
        return
    if not args.compile_only:
        install_callable(bundle, args.runtime)
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
        "flavor": "pypto_lib_latest",
        "source_commit": case["lib_commit"],
        "runtime_lib": str(runtime_lib),
        "binding_commit": pin.installed,
        "runtime": args.runtime,
        "runtime_sha256": runtime_hashes,
        "devices": devices,
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
                    "device_us_mean": statistics.mean(samples),
                    "host_us_mean": statistics.mean(result.bench.host_wall_us),
                    "per_rank_device_us": result.bench.per_rank("device"),
                }
    report["compile_only"] = args.compile_only
    report["passed"] = not args.compile_only
    if args.compile_only:
        DistributedCompiledProgram = importlib.import_module(
            "pypto.ir.distributed_compiled_program"
        ).DistributedCompiledProgram

        compiled = DistributedCompiledProgram.from_dir(
            program,
            backend_type=config.backend_type,
            execution_mode=config.execution_mode,
            distributed_config=config.distributed_config,
        )
        prepare_callable(bundle, compiled)
        report["compilation_passed"] = True
    else:
        validate_report(report, case, args.level)
    (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"MoE Host control {'compilation' if args.compile_only else 'validation'} passed: {output}")


if __name__ == "__main__":
    main()
