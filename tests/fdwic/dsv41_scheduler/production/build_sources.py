#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build the checked-in model sources with the existing Simpler compiler/cache."""

import argparse
import gzip
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simpler_setup.compile_pool import compile_slot, compile_worker_budget
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test_cache import compile_incore_artifact_key, get_or_compile_incore

ROOT = Path(__file__).resolve().parents[1]
FLAVORS = ("pypto_lib", "pypto_lib_dsl", "pypto_lib_latest")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(flavor, output, tools, pto, jobs):
    sources = ROOT / "kernels" / flavor
    workload = ROOT / "workloads" / flavor
    manifest = json.loads((sources / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    compiler = KernelCompiler("a5")
    common_includes = compiler.get_orchestration_include_dirs("host_build_graph")
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    # Generated DSL C++ includes impl/c_api/... relative to the ASC SDK root.
    asc_roots = [str(path) for path in (sdk / "asc", sdk / "ascendc") if path.is_dir()]
    sdk_versions = {
        name: (sdk / name).read_text()
        for name in ("compiler/version.info", "share/info/asc-devkit/version.info")
        if (sdk / name).is_file()
    }
    compiler_tokens = {core: compiler.incore_compile_cache_token(core) for core in ("aic", "aiv")}

    def compile_kernel(kernel):
        source = sources / kernel["source"]
        extra = common_includes + [str(sources / item) for item in kernel["include_dirs"]]
        dsl_entry = flavor == "pypto_lib_dsl" and "/external/" in kernel["source"]
        cache_extra = list(extra)
        if dsl_entry:
            extra += asc_roots
        includes = [
            str(pto / "include"),
            str(pto / "include/pto"),
            *compiler.get_incore_include_dirs(),
            # Match the normal scene-test policy for SDK headers: CANN's
            # legacy header symlinks contain ../ spellings which the lexical
            # include scanner cannot normalize. Compiler/SDK identity covers
            # those; source, PTO, Simpler and local headers remain in closure.
            *([] if dsl_entry else compiler.get_ascend_incore_include_dirs()),
            *cache_extra,
        ]
        metadata = {
            "platform": "a5",
            "host_platform": sys.platform,
            "host_machine": platform.machine(),
            "compiler": compiler_tokens[kernel["core_type"]],
        }
        if dsl_entry:
            metadata["sdk_include_roots"] = compiler.get_ascend_incore_include_dirs() + asc_roots
            metadata["sdk_versions"] = sdk_versions
        key = compile_incore_artifact_key(metadata, source, includes)
        compiled = False

        def compile_one():
            nonlocal compiled
            compiled = True
            with compile_slot():
                return extract_text_section(
                    compiler.compile_incore(
                        str(source),
                        core_type=kernel["core_type"],
                        pto_isa_root=str(pto),
                        extra_include_dirs=extra,
                    )
                )

        binary = get_or_compile_incore(key, compile_one)
        if binary.startswith(b"\x7fELF"):
            raise ValueError("CoreCallable requires raw instructions, not an ELF image")
        target = output / f"kernel-{kernel['func_id']}.bin"
        target.write_bytes(binary)
        print(f"{flavor}: {'compiled' if compiled else 'cached'} {kernel['func_id']} {kernel['name']}", flush=True)
        return {
            "func_id": kernel["func_id"],
            "name": kernel["name"],
            "core_type": kernel["core_type"],
            "file": target.name,
            "cache_key": key,
            "cache_hit": not compiled,
        }

    with compile_worker_budget(jobs), ThreadPoolExecutor(max_workers=jobs) as pool:
        records = list(pool.map(compile_kernel, manifest["kernels"]))

    source = output / "orchestration.cpp"
    source.write_bytes(gzip.decompress((workload / "vendor/orchestration.cpp.gz").read_bytes()))
    includes, support = compiler.get_orchestration_cache_inputs("host_build_graph")
    host_token = compiler.compile_cache_token("host_build_graph", [])
    # Cache the Host ELF as bytes without importing Simpler's Python bindings.
    # Hash every supporting translation unit and its include closure as well.
    support_keys = [compile_incore_artifact_key({"platform": "host"}, path, includes) for path in support]
    key = compile_incore_artifact_key(
        {"platform": "host", "kind": "dsv41-orchestration", "compiler": host_token, "support": support_keys},
        source,
        includes,
    )
    orchestration = get_or_compile_incore(key, lambda: compiler.compile_orchestration("host_build_graph", str(source)))
    (output / "orchestration.so").write_bytes(orchestration)
    shutil.copytree(workload / "inputs", output, dirs_exist_ok=True)
    rows = []
    for kernel, record in zip(manifest["kernels"], records, strict=True):
        signature = kernel["signature"]
        rows.append(
            f"{kernel['func_id']} {len(signature)} {' '.join(map(str, signature))} "
            + json.dumps(str(output / record["file"]))
        )
    signature = manifest["entry_signature"]
    (output / "kernels.txt").write_text(
        f"{len(signature)} {' '.join(map(str, signature))}\n{len(rows)}\n" + "\n".join(rows) + "\n"
    )
    subprocess.run(
        [
            str(tools / "native-runner"),
            "--assemble-orchestration",
            str(output / "kernels.txt"),
            str(output / "orchestration.so"),
            str(output / "original-callable.bin"),
        ],
        check=True,
    )
    # This callable captures the full original Host submissions; case execution
    # later uses the replay adapter with the same original core code placement.
    subprocess.run(
        [
            str(tools / "native-runner"),
            "--repack-orchestration",
            str(output / "original-callable.bin"),
            str(tools / "libcapture.so"),
            str(output / "callable.bin"),
        ],
        check=True,
    )
    save(
        output / "name_map.json",
        {
            "level": 2,
            "orchestrator_name": "l2_decode_fwd",
            "callable_id_to_name": {str(k["func_id"]): k["name"] for k in manifest["kernels"]},
        },
    )
    save(
        output / "build.json",
        {
            "flavor": flavor,
            "platform": "a5",
            "pto_isa": str(pto),
            "sdk_include_roots": asc_roots,
            "sdk_versions": sdk_versions,
            "compiler": {"host": host_token, "incore": compiler_tokens},
            "orchestration_cache_key": key,
            "kernels": records,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=[*FLAVORS, "both", "all"], default="all")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/portable")
    parser.add_argument("--pto-isa", type=Path, default=ROOT.parents[2] / "build/pto-isa")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or not (args.pto_isa / "include/pto").is_dir():
        parser.error("--jobs must be positive and --pto-isa must contain include/pto")
    logging.basicConfig(level=logging.DEBUG)
    flavors = FLAVORS if args.flavor == "all" else FLAVORS[:2] if args.flavor == "both" else [args.flavor]
    for flavor in flavors:
        build(
            flavor,
            args.build_dir.resolve() / "forward" / flavor,
            args.build_dir.resolve() / "host",
            args.pto_isa.resolve(),
            args.jobs,
        )


if __name__ == "__main__":
    main()
