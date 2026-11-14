#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the selected 16-case matrix from the three original orchestrations."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from export_cases import export
from materialize import materialize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path, help="Matching production a5/onboard runtime directory")
    parser.add_argument("output", type=Path, help="New directory for captures and cases")
    parser.add_argument(
        "--build-dir", type=Path, default=Path(os.environ.get("DSV41_BUILD_DIR", ROOT / "build/portable"))
    )
    parser.add_argument(
        "--flavor", choices=["pypto_lib", "pypto_lib_dsl", "pypto_lib_latest", "both", "all"], default="all"
    )
    parser.add_argument("--sync-policy", choices=["half", "all-cap64", "both"], default="both")
    parser.add_argument("--pypto-lib-root", type=Path, help="Clean dev/hbg checkout for native MoE goldens")
    parser.add_argument("--native-moe-bundle", type=Path, help="Reuse an existing compatible native MoE bundle")
    args = parser.parse_args()
    if args.flavor in ("all", "pypto_lib_latest") and args.pypto_lib_root is None:
        parser.error("The selected native MoE case requires --pypto-lib-root")
    output, build, runtime = args.output.resolve(), args.build_dir.resolve(), args.runtime.resolve()
    runtime_files = (
        "host_build_graph/libhost_runtime.so",
        "host_build_graph/libaicpu_kernel.so",
        "host_build_graph/aicore_kernel.o",
        "../dispatcher/libsimpler_aicpu_dispatcher.so",
    )
    for name in runtime_files:
        if not (runtime / name).is_file():
            parser.error(f"Required production runtime file is missing: {runtime / name}")
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "DSV41_RUNNER": str(build / "host/native-runner")}
    (output / "runtime.json").write_text(json.dumps({"path": str(runtime)}, indent=2) + "\n")
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
            ["bash", str(ROOT / "production/run.sh"), str(bundle), str(runtime), str(capture), "1", "0", "1"],
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
            args.pypto_lib_root,
            args.native_moe_bundle,
        )
        for variant in variants:
            for case in sorted((output / "cases" / variant).iterdir()):
                if json.loads((case / "case.json").read_text())["device_count"] == 1:
                    materialize(case, runtime, output / "preparation" / variant / case.name, env)
    print(f"Prepared cases: {output / 'cases'}")


if __name__ == "__main__":
    main()
