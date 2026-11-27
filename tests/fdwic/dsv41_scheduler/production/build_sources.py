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
import hashlib
import json
import logging
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simpler_setup.a5_input_cache import annotate_a5_hbg_text
from simpler_setup.compile_pool import compile_slot, compile_worker_budget
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test_cache import compile_incore_artifact_key, get_or_compile_incore

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


# The model contains separately compiled vector functions. Their portable LLVM
# bodies are the output of the pinned PyPTO/PTOAS export.
VF_FLAGS = (
    "--cce-aicore-arch=dav-c310-vec",
    "--cce-aicore-only",
    "-O2",
    "--cce-generic-addrspace=off",
    "-cce-bitcode-is-aicore",
    "-Wno-override-module",
    "-dc",
    "--cce-long-scbz=true",
    "-mllvm",
    "-cce-dyn-kernel-stack-size=true",
)


def validate_device_image(data: bytes) -> None:
    """Reject linked images whose dependencies would be lost by .text extraction.

    The persistent loader starts at byte zero and retains no ELF sections or
    relocation metadata. This structural check does not prove arbitrary code
    position-independent; helper authors must still obey the device-call ABI.
    """
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("VPTO linking requires a little-endian ELF64 device image")
    elf_type, machine = struct.unpack_from("<HH", data, 16)
    if elf_type != 2 or machine != 0x1029:
        raise ValueError("VPTO linking requires an executable AI Core ELF, not a host fatobj")
    (entry,) = struct.unpack_from("<Q", data, 24)
    (offset,) = struct.unpack_from("<Q", data, 40)
    size, count, names_index = struct.unpack_from("<HHH", data, 58)
    if size != 64 or not count or names_index >= count or offset + count * size > len(data):
        raise ValueError("VPTO linked image has an invalid section table")
    sections: list[tuple[int, ...]] = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    names = sections[names_index]
    strings = data[names[4] : names[4] + names[5]]
    found_text = False
    for name_offset, kind, flags, address, start, length, _, _, _, stride in sections:
        name = strings[name_offset:].split(b"\x00", 1)[0]
        if kind != 8 and start + length > len(data):  # SHT_NOBITS has no file payload.
            raise ValueError("VPTO linked image has a truncated section")
        if name == b".text":
            found_text = bool(length)
            if entry != address:
                raise ValueError("VPTO kernel_entry must be at the start of .text")
        if flags & 2 and length and name != b".text":  # SHF_ALLOC
            raise ValueError(f"VPTO linked image needs unsupported allocated section {name!r}")
        if kind in (4, 9) and length:  # SHT_RELA / SHT_REL
            raise ValueError("VPTO linked image must have no remaining relocations")
        if kind in (2, 11):  # SHT_SYMTAB / SHT_DYNSYM
            if stride != 24 or length % stride:
                raise ValueError("VPTO linked image has an invalid symbol table")
            for position in range(start, start + length, stride):
                symbol, _, _, section = struct.unpack_from("<IBBH", data, position)
                if symbol and section == 0:
                    raise ValueError("VPTO linked image has an unresolved symbol")
    if not found_text:
        raise ValueError("VPTO linked image requires a nonempty .text section")


def compile_vector_functions(compiler, source, includes, helpers):
    toolchain = compiler.ccec
    bisheng = Path(toolchain.cxx_path).with_name("bisheng")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        caller = directory / "caller.o"
        subprocess.run(
            [
                toolchain.cxx_path,
                *toolchain.get_compile_flags(core_type="aiv"),
                *compiler._incore_feature_defines(),
                "-ffunction-sections",
                *(f"-I{path}" for path in includes),
                "-o",
                str(caller),
                str(source),
            ],
            check=True,
        )
        objects = []
        for i, helper in enumerate(helpers):
            obj = directory / f"helper{i}.o"
            subprocess.run([bisheng, *VF_FLAGS, "-c", "-x", "ir", helper, "-o", obj], check=True)
            objects.append(obj)
        order, elf = directory / "order.txt", directory / "kernel.elf"
        order.write_text("kernel_entry\n")
        subprocess.run(
            [
                toolchain.linker_path,
                "-e",
                "kernel_entry",
                f"--symbol-ordering-file={order}",
                "-o",
                elf,
                caller,
                *objects,
            ],
            check=True,
        )
        binary = elf.read_bytes()
        validate_device_image(binary)
        return extract_text_section(binary)


def build(flavor, output, tools, pto, jobs, runtime):
    sources = ROOT / "kernels" / flavor
    workload = ROOT / "workloads" / flavor
    manifest = json.loads((sources / "manifest.json").read_text())
    native = manifest.get("execution") == "native_moe_host_control"
    output.mkdir(parents=True, exist_ok=True)
    compiler = KernelCompiler("a5")
    common_includes = compiler.get_orchestration_include_dirs(runtime)
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    asc_roots = [str(path) for path in (sdk / "asc", sdk / "ascendc") if path.is_dir()]
    sdk_versions = {
        name: (sdk / name).read_text()
        for name in ("compiler/version.info", "share/info/asc-devkit/version.info")
        if (sdk / name).is_file()
    }
    compiler_tokens = {core: compiler.incore_compile_cache_token(core) for core in ("aic", "aiv")}

    vf_toolchain = None
    if any(kernel.get("vf_llvm") for kernel in manifest["kernels"]):
        if compiler.ccec is None:
            raise RuntimeError("A5 vector functions require the CCEC toolchain")
        bisheng = Path(compiler.ccec.cxx_path).with_name("bisheng")
        vf_toolchain = subprocess.check_output([bisheng, "--version"], text=True)

    def compile_kernel(kernel):
        source = sources / kernel["source"]
        extra = common_includes + [str(sources / item) for item in kernel["include_dirs"]]
        includes = [
            str(pto / "include"),
            str(pto / "include/pto"),
            *compiler.get_incore_include_dirs(),
            *compiler.get_ascend_incore_include_dirs(),
            *extra,
        ]
        metadata = {
            "platform": "a5",
            "host_platform": sys.platform,
            "host_machine": platform.machine(),
            "compiler": compiler_tokens[kernel["core_type"]],
        }
        helpers = [sources / path for path in kernel.get("vf_llvm", [])]
        if helpers:
            if kernel["core_type"] != "aiv":
                raise ValueError("Vector functions require an AIV caller")
            metadata["vf"] = {
                "compiler": vf_toolchain,
                "flags": VF_FLAGS,
                "caller": "-ffunction-sections",
                "entry": "kernel_entry",
                "helpers": [hashlib.sha256(path.read_bytes()).hexdigest() for path in helpers],
                "builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        key = compile_incore_artifact_key(metadata, source, includes)
        compiled = False

        def compile_one():
            nonlocal compiled
            compiled = True
            with compile_slot():
                if helpers:
                    return compile_vector_functions(compiler, source, includes, helpers)
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
        registration_file = target
        if runtime == "host_build_graph":
            annotated = annotate_a5_hbg_text(binary, kernel["core_type"])
            if annotated != binary:
                registration_file = target.with_suffix(".hbg")
                registration_file.write_bytes(annotated)
        print(f"{flavor}: {'compiled' if compiled else 'cached'} {kernel['func_id']} {kernel['name']}", flush=True)
        return {
            "func_id": kernel["func_id"],
            "name": kernel["name"],
            "core_type": kernel["core_type"],
            "file": target.name,
            "registration_file": registration_file.name,
            "cache_key": key,
            "cache_hit": not compiled,
        }

    with compile_worker_budget(jobs), ThreadPoolExecutor(max_workers=jobs) as pool:
        records = list(pool.map(compile_kernel, manifest["kernels"]))

    source = output / "orchestration.cpp"
    shutil.copyfile(workload / "vendor/orchestration.cpp", source)
    includes, support = compiler.get_orchestration_cache_inputs(runtime)
    host_token = compiler.compile_cache_token(runtime, [])
    # Cache the Host ELF as bytes without importing Simpler's Python bindings.
    # Hash every supporting translation unit and its include closure as well.
    support_keys = [compile_incore_artifact_key({"platform": "host"}, path, includes) for path in support]
    key = compile_incore_artifact_key(
        {"platform": "host", "kind": "dsv41-orchestration", "compiler": host_token, "support": support_keys},
        source,
        includes,
    )
    orchestration = get_or_compile_incore(key, lambda: compiler.compile_orchestration(runtime, str(source)))
    (output / "orchestration.so").write_bytes(orchestration)
    if not native:
        shutil.copytree(workload / "inputs", output, dirs_exist_ok=True)
    rows = []
    for kernel, record in zip(manifest["kernels"], records, strict=True):
        signature = kernel["signature"]
        rows.append(
            f"{kernel['func_id']} {len(signature)} {' '.join(map(str, signature))} "
            + json.dumps(str(output / record["registration_file"]))
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
    if native:
        shutil.copy2(output / "original-callable.bin", output / "callable.bin")
    else:
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
            "orchestrator_name": manifest.get("orchestrator_name", "l2_moe" if native else "l2_decode_fwd"),
            "callable_id_to_name": {str(k["func_id"]): k["name"] for k in manifest["kernels"]},
        },
    )
    save(
        output / "build.json",
        {
            "flavor": flavor,
            "platform": "a5",
            "runtime": runtime,
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
    parser.add_argument("--flavor", choices=("pypto_lib_latest", "all"), default="pypto_lib_latest")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/portable")
    parser.add_argument("--pto-isa", type=Path, default=ROOT.parents[2] / "build/pto-isa")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--runtime", choices=("host_build_graph", "percounter"), default="host_build_graph")
    args = parser.parse_args()
    if args.jobs < 1 or not (args.pto_isa / "include/pto").is_dir():
        parser.error("--jobs must be positive and --pto-isa must contain include/pto")
    logging.basicConfig(level=logging.DEBUG)
    build_dir = args.build_dir.resolve()
    for flavor, target in (
        ("pypto_lib_latest", build_dir / "forward/pypto_lib_latest"),
        ("native_moe", build_dir / "native-moe"),
    ):
        build(flavor, target, build_dir / "host", args.pto_isa.resolve(), args.jobs, args.runtime)


if __name__ == "__main__":
    main()
