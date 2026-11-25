#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the PyPTO-lib cases and the standalone full-wave MIX chain."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from device_start_gate import validate_runtime_gate
from export_cases import export
from full_wave_mix_chain import build_bundle
from list_cases import SYNTHETIC_CASE
from materialize import materialize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_dir", type=Path, help="Matching production a5/onboard runtime directory")
    parser.add_argument("output", type=Path, help="New directory for captures and cases")
    parser.add_argument(
        "--build-dir", type=Path, default=Path(os.environ.get("DSV41_BUILD_DIR", ROOT / "build/portable"))
    )
    parser.add_argument("--flavor", choices=("all", "pypto_lib_latest", "synthetic"), default="all")
    parser.add_argument("--native-moe-bundle", type=Path, help="Reuse an existing compatible native MoE bundle")
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), default="host_build_graph")
    args = parser.parse_args()
    output, build, runtime = args.output.resolve(), args.build_dir.resolve(), args.runtime_dir.resolve()
    runtime_files = (
        f"{args.runtime}/libhost_runtime.so",
        f"{args.runtime}/libaicpu_kernel.so",
        f"{args.runtime}/aicore_kernel.o",
        "../dispatcher/libsimpler_aicpu_dispatcher.so",
    )
    for name in runtime_files:
        if not (runtime / name).is_file():
            parser.error(f"Required production runtime file is missing: {runtime / name}")
    if args.flavor != "synthetic":
        try:
            validate_runtime_gate(runtime, args.runtime)
        except ValueError as exc:
            parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=False)
    runner = build / "host/native-runner"
    (output / "runtime.json").write_text(json.dumps({"path": str(runtime), "runtime": args.runtime}, indent=2) + "\n")
    if args.flavor in ("all", "pypto_lib_latest"):
        prepare_model_cases(output, build, runtime, args.runtime, args.native_moe_bundle)
    if args.flavor in ("all", "synthetic"):
        build_bundle(output / "cases" / SYNTHETIC_CASE, args.runtime, runner)
    print(f"Prepared cases: {output / 'cases'}")


def prepare_model_cases(output, build, runtime, runtime_name, native_moe_bundle):
    env = {**os.environ, "DSV41_RUNNER": str(build / "host/native-runner")}
    flavor = "pypto_lib_latest"
    bundle = build / "forward" / flavor
    capture = output / "capture" / flavor
    subprocess.run(
        [
            "bash",
            str(ROOT / "production/run.sh"),
            str(bundle),
            str(runtime),
            str(capture),
            "1",
            "0",
            "1",
            f"--runtime={runtime_name}",
        ],
        env=env,
        check=True,
    )
    cases = export(bundle, capture, flavor, output / "cases", build / "host", native_moe_bundle)
    for case in cases:
        if json.loads((case / "case.json").read_text())["device_count"] == 1:
            materialize(case, runtime, output / "preparation" / flavor / case.name, env, runtime_name)


if __name__ == "__main__":
    main()
