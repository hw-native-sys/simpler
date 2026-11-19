#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the selected 11-case matrix from the three original orchestrations."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from device_start_gate import validate_runtime_gate
from export_cases import export
from materialize import materialize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_dir", type=Path, help="Matching production a5/onboard runtime directory")
    parser.add_argument("output", type=Path, help="New directory for captures and cases")
    parser.add_argument(
        "--build-dir", type=Path, default=Path(os.environ.get("DSV41_BUILD_DIR", ROOT / "build/portable"))
    )
    parser.add_argument(
        "--flavor", choices=["pypto_lib", "pypto_lib_dsl", "pypto_lib_latest", "both", "all"], default="all"
    )
    parser.add_argument("--sync-policy", choices=["half", "all-cap64", "both"], default="both")
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
    try:
        validate_runtime_gate(runtime, args.runtime)
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "DSV41_RUNNER": str(build / "host/native-runner")}
    (output / "runtime.json").write_text(json.dumps({"path": str(runtime), "runtime": args.runtime}, indent=2) + "\n")
    flavors = (
        ("pypto_lib", "pypto_lib_dsl", "pypto_lib_latest")
        if args.flavor == "all"
        else ("pypto_lib", "pypto_lib_dsl")
        if args.flavor == "both"
        else (args.flavor,)
    )
    for flavor in flavors:
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
                f"--runtime={args.runtime}",
            ],
            env=env,
            check=True,
        )
        variants = export(
            bundle,
            capture,
            flavor,
            output / "cases",
            build / "host",
            args.sync_policy,
            args.native_moe_bundle,
        )
        for variant in variants:
            for case in sorted((output / "cases" / variant).iterdir()):
                if json.loads((case / "case.json").read_text())["device_count"] == 1:
                    materialize(case, runtime, output / "preparation" / variant / case.name, env, args.runtime)
    print(f"Prepared cases: {output / 'cases'}")


if __name__ == "__main__":
    main()
