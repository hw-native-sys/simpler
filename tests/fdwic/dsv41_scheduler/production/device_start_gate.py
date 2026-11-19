#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a test-only device-start-gate wrapper around existing A5 runtime binaries.

No device is opened. The original Host library is copied with a distinct SONAME;
all executable sections are compared unchanged. AICPU/AICore, dispatcher and
pin metadata are copied into a self-contained snapshot. The wrapper and its
standalone AIV kernel are the only newly compiled components.
"""

import argparse
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[3]
ENTRY = "dsv41_device_start_gate"


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def validate_runtime_gate(runtime_dir, variant):
    """Require a completed, self-contained gate snapshot without loading NPU libraries."""
    runtime = Path(runtime_dir).resolve()
    instruction = (
        "Before reserving devices, run production/device_start_gate.py "
        "--runtime-dir ORIGINAL_A5_ONBOARD --output-dir NEW_WRAPPER_DIR, "
        "then pass NEW_WRAPPER_DIR/runtime/a5/onboard as the runtime."
    )
    try:
        if variant not in ("host_build_graph", "percounter", "tensormap_and_ringbuffer"):
            raise ValueError(f"Unsupported runtime variant: {variant}")
        if runtime.name != "onboard" or runtime.parent.name != "a5":
            raise ValueError("Expected a wrapper runtime/a5/onboard directory")
        manifest = json.loads((runtime.parents[2] / "manifest.json").read_text())
        if manifest["test_only"] is not True or manifest["status"] != "build_passed":
            raise ValueError("The device-start-gate wrapper build is incomplete")
        if Path(manifest["runtime_dir"]).resolve() != runtime:
            raise ValueError("The wrapper manifest names a different runtime directory")
        wrapper = manifest["wrappers"][variant]
        if wrapper["executable_sections_unchanged"] is not True or manifest["gate"]["entry"] != ENTRY:
            raise ValueError("The manifest does not describe an unchanged runtime with the device-start gate")
        prefix = f"a5/onboard/{variant}/"
        shared = {"a5/dispatcher/libsimpler_aicpu_dispatcher.so", "pto_isa_build.json"}
        required = shared | {
            prefix + name
            for name in (
                "libhost_runtime.so",
                "libhost_runtime_impl.so",
                "libaicpu_kernel.so",
                "aicore_kernel.o",
                "device_start_gate.o",
            )
        }
        files = manifest["snapshot_files"]
        if not required <= files.keys():
            raise ValueError("The runtime snapshot is missing required components")
        for relative, size in files.items():
            if not relative.startswith(prefix) and relative not in shared:
                continue
            path = runtime.parents[1] / relative
            if path.resolve() != path or not path.is_file():
                raise ValueError(f"Runtime snapshot file is missing or linked: {relative}")
            if path.stat().st_size != size:
                raise ValueError(f"Runtime snapshot file size changed: {relative}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Two-device timing requires a valid device-start-gate wrapper: {exc}. {instruction}") from exc
    return manifest


def executable_sections(path):
    """Read ELF64 little-endian executable section bytes, independent of offsets."""
    data = Path(path).read_bytes()
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected an ELF64 little-endian image: {path}")
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", data, 58)
    if stride < 64 or not count or names_index >= count or offset + stride * count > len(data):
        raise ValueError(f"Invalid ELF section table: {path}")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride) for i in range(count)]
    names = sections[names_index]
    strings = data[names[4] : names[4] + names[5]]
    result = {}
    for section in sections:
        name_offset, _, flags, _, start, size, *_ = section
        if not flags & 4 or size == 0:
            continue
        end = strings.find(b"\0", name_offset)
        if end < 0 or start + size > len(data):
            raise ValueError(f"Invalid executable section: {path}")
        name = strings[name_offset:end].decode("utf-8")
        result[name] = data[start : start + size]
    if ".text" not in result:
        raise ValueError(f"Missing nonempty .text section: {path}")
    return result


def tool(name):
    found = shutil.which(str(name))
    if found is None:
        raise ValueError(f"Required build tool is unavailable: {name}")
    return str(Path(found).resolve())


def run(command, env, log):
    command = list(map(str, command))
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    with log.open("a") as stream:
        stream.write("$ " + shlex.join(command) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(f"Build command failed ({result.returncode}); see {log}")
    return result.stdout.strip()


def copy_tree(source, destination, originals, prefix):
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        target = destination / path.name
        relative = prefix / path.name
        if path.is_dir():
            copy_tree(path, target, originals, relative)
        elif path.is_file():
            before = file_stamp(path)
            shutil.copy2(path, target)
            if file_stamp(path) != before:
                raise ValueError(f"Runtime source changed while copying: {relative}")
            originals[str(relative)] = {"source": str(path.resolve()), "stamp": before}


def build(args):
    original = args.runtime_dir.resolve()
    output = args.output_dir.resolve()
    sdk = args.cann.resolve()
    pto = args.pto_isa.resolve()
    runtime_pin = original.parent.parent / "pto_isa_build.json"
    shim_source = SOURCE / "device_start_gate.cpp"
    gate_source = SOURCE / "device_start_gate_kernel.cpp"
    source_stamps = {str(path): file_stamp(path) for path in (Path(__file__).resolve(), shim_source, gate_source)}
    variants = [
        name
        for name in ("host_build_graph", "percounter", "tensormap_and_ringbuffer")
        if (original / name / "libhost_runtime.so").is_file()
    ]
    if (
        not variants
        or any(
            not (original / name / component).is_file()
            for name in variants
            for component in ("libaicpu_kernel.so", "aicore_kernel.o")
        )
        or not (original.parent / "dispatcher/libsimpler_aicpu_dispatcher.so").is_file()
        or not runtime_pin.is_file()
    ):
        raise ValueError(
            "--runtime-dir must name an A5 onboard build with HBG/percounter/TMR, its sibling dispatcher, "
            "and runtime-root pto_isa_build.json for pin validation"
        )
    if output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError("The output must be separate from the supplied runtime tree")
    if not shim_source.is_file() or not gate_source.is_file():
        raise ValueError("The checked-in gate Host and AIV sources must both exist")
    if not (sdk / "include/acl/acl.h").is_file() or not (pto / "include").is_dir():
        raise ValueError("A CANN SDK and PTO-ISA include directory are required")
    tools = {
        "cxx": tool(args.cxx),
        "ccec": tool(sdk / "bin/ccec"),
        "linker": tool(sdk / "bin/ld.lld"),
        "patchelf": tool("patchelf"),
        "readelf": tool("readelf"),
    }
    output.mkdir(parents=True, exist_ok=False)
    work = output / "build"
    work.mkdir()
    temporary = work / "tmp"
    temporary.mkdir()
    env = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    log = work / "build.log"
    versions = {name: run([path, "--version"], env, log) for name, path in tools.items()}
    gate = work / "device_start_gate.o"
    gate_flags = [
        tools["ccec"],
        "-O3",
        "-std=c++17",
        "-x",
        "cce",
        "--cce-aicore-only",
        "--cce-aicore-arch=dav-c310-vec",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-DPTO_NPU_ARCH_A5",
        "-I",
        pto / "include",
        "-I",
        REPO / "src/common",
    ]
    if args.gate_kernel is not None:
        shutil.copy2(args.gate_kernel.resolve(), gate)
    else:
        part = work / "device_start_gate_vec.o"
        run(gate_flags + ["-c", "-MD", "-MF", work / "device_start_gate.d", "-o", part, gate_source], env, log)
        run(
            [tools["linker"], "-m", "aicorelinux", "-Ttext=0", "-static", "-n", "-e", ENTRY, "-o", gate, part], env, log
        )
    # CCEC emits device IR through LTO; this is the gate TU with its actual flags.
    bitcode = work / "device_start_gate.bc"
    llvm_ir = work / "device_start_gate.ll"
    run(gate_flags + ["-c", "-flto", "-o", bitcode, gate_source], env, log)
    run(
        [
            tools["ccec"],
            "-cc1",
            "-triple",
            "hiipu64-hisilicon-cce",
            "-disable-llvm-passes",
            "-emit-llvm",
            "-x",
            "ir",
            bitcode,
            "-o",
            llvm_ir,
        ],
        env,
        log,
    )
    gate_sections = executable_sections(gate)
    symbols = run([tools["readelf"], "-sW", gate], env, log)
    if not any(line.split()[-1:] == [ENTRY] and " UND " not in line for line in symbols.splitlines()):
        raise ValueError(f"The gate image does not define {ENTRY}")

    runtime = output / "runtime/a5/onboard"
    originals = {}
    copy_tree(original, runtime, originals, Path("a5/onboard"))
    copy_tree(original.parent / "dispatcher", runtime.parent / "dispatcher", originals, Path("a5/dispatcher"))
    originals["pto_isa_build.json"] = {"source": str(runtime_pin.resolve()), "stamp": file_stamp(runtime_pin)}
    shutil.copy2(runtime_pin, output / "runtime/pto_isa_build.json")
    wrappers = {}
    includes = [
        sdk / "include",
        REPO / "src/common",
        REPO / "src/common/task_interface",
        REPO / "src/common/worker",
        REPO / "src/common/hierarchical",
        REPO / "src/common/platform/include",
        REPO / "src/common/platform/include/common",
        REPO / "src/common/platform/include/host",
        REPO / "src/common/log/include",
    ]
    for name in variants:
        directory = runtime / name
        host = directory / "libhost_runtime.so"
        source_host = original / name / host.name
        before = executable_sections(host)
        impl = directory / "libhost_runtime_impl.so"
        if impl.exists():
            raise ValueError("The supplied runtime already contains a test wrapper implementation")
        host.rename(impl)
        run([tools["patchelf"], "--set-soname", impl.name, impl], env, log)
        after = executable_sections(impl)
        if before != after:
            raise ValueError("Changing the implementation SONAME modified executable bytes")
        shutil.copy2(gate, directory / gate.name)
        command = [tools["cxx"], "-std=c++17", "-O3", "-fPIC", "-shared", "-fno-gnu-unique", "-fvisibility=hidden"]
        command += [item for include in includes for item in ("-I", include)]
        command += [
            "-MMD",
            "-MF",
            work / f"{name}-shim.d",
            shim_source,
            "-L",
            directory,
            "-L",
            sdk / "lib64",
            "-Wl,--no-as-needed",
            "-lhost_runtime_impl",
            "-lascendcl",
            "-ldl",
            "-pthread",
            f"-Wl,-rpath,$ORIGIN:{sdk / 'lib64'}",
            "-o",
            host,
        ]
        run(command, env, log)
        needed = run([tools["patchelf"], "--print-needed", host], env, log).splitlines()
        if impl.name not in needed or host.name in needed:
            raise ValueError("Wrapper must depend on the renamed implementation without depending on itself")
        wrappers[name] = {
            "original": str(source_host),
            "impl_soname": run([tools["patchelf"], "--print-soname", impl], env, log),
            "executable_sections_unchanged": True,
            "executable_sections": {section: len(data) for section, data in before.items()},
            "needed": needed,
            "rpath": run([tools["patchelf"], "--print-rpath", host], env, log),
        }
    for relative, record in originals.items():
        if file_stamp(record["source"]) != record["stamp"]:
            raise ValueError(f"The supplied binary changed during wrapping: {relative}")
    if any(file_stamp(path) != expected for path, expected in source_stamps.items()):
        raise ValueError("Gate sources changed during the build; discard this wrapper and rebuild")
    manifest = {
        "status": "build_passed",
        "test_only": True,
        "hardware_validated": False,
        "runtime_dir": str(runtime),
        "source_runtime_dir": str(original),
        "original_binaries": originals,
        "wrappers": wrappers,
        "tools": {name: {"path": path, "version": versions[name]} for name, path in tools.items()},
        "gate": {
            "file": str(gate),
            "entry": ENTRY,
            "sections": {section: len(data) for section, data in gate_sections.items()},
            "ir": {"file": str(llvm_ir), "scope": "gate single translation unit"},
            "bitcode": {"file": str(bitcode)},
            "reused_from": str(args.gate_kernel.resolve()) if args.gate_kernel is not None else None,
        },
        "sources": source_stamps,
        "snapshot_files": {
            str(path.relative_to(output / "runtime")): path.stat().st_size
            for path in sorted((output / "runtime").rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"runtime_dir": str(runtime), "manifest": str(output / "manifest.json"), "gate_kernel": str(gate)}, indent=2
        )
    )


def main():
    if sys.argv[1:2] == ["validate"]:
        parser = argparse.ArgumentParser(description="Validate a prebuilt device-start-gate wrapper; no compilation")
        parser.add_argument("runtime_dir", type=Path)
        parser.add_argument("variant", choices=("host_build_graph", "percounter", "tensormap_and_ringbuffer"))
        args = parser.parse_args(sys.argv[2:])
        try:
            validate_runtime_gate(args.runtime_dir, args.variant)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Device-start-gate wrapper validated: {args.runtime_dir.resolve()}/{args.variant}")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True, help="Existing a5/onboard build")
    parser.add_argument("--output-dir", type=Path, required=True, help="New wrapper artifact directory")
    parser.add_argument("--gate-kernel", type=Path, help="Reuse a previously compiled gate ELF")
    parser.add_argument(
        "--cann", type=Path, default=Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    )
    parser.add_argument("--pto-isa", type=Path, default=REPO / "build/pto-isa")
    parser.add_argument("--cxx", default=shutil.which("g++-15") or "g++")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
