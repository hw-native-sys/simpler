#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the eleven frozen workloads for AICPU orchestration and TMR scheduling."""

import argparse
import errno
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test_cache import compile_incore_artifact_key, get_or_compile_incore

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64) - 1
MAGIC = 0x44535634544D5231


def publish(path, data):
    """Publish complete cache artifacts when preparations share a build directory."""
    if path.is_file() and path.read_bytes() == data:
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def encode_plan(records, cut, tensor_count):
    """Resolve captured task/buffer identities without interpreting task-id bits."""
    first, last = cut
    begin = next(i for i, r in enumerate(records) if r["op"] in (2, 3) and r["id"] == first)
    end = next(i + 1 for i, r in enumerate(records) if r["op"] in (2, 3) and r["id"] == last)
    records = records[:end]
    ids = {r["id"]: i for i, r in enumerate(records) if r["op"] in (1, 2, 3)}
    slots = {}
    for r in records:
        for t in r["tensors"]:
            slots.setdefault(t["buffer"], len(slots))
    params = {t["buffer"]: i for i, t in enumerate(records[0]["tensors"])}
    allocated = {t["buffer"]: i for i, r in enumerate(records) for t in r["tensors"] if t["tag"] == 1}
    boundary = sorted(
        {
            t["buffer"]
            for r in records[begin:end]
            for t in r["tensors"]
            if t["buffer"] not in params and allocated[t["buffer"]] < begin
        }
    )
    original_tensors = len(records[0]["tensors"])
    for offset, address in enumerate(boundary):
        params[address] = original_tensors + offset
    if original_tensors + len(boundary) != tensor_count:
        raise ValueError("Captured boundary arguments differ from the prepared input manifest")
    scopes = []
    for r in records[1:begin]:
        if r["op"] == 4:
            scopes.append(r["scope"])
        elif r["op"] == 5:
            scopes.pop()
    scalar_count = len(records[0]["scalars"])
    words = [MAGIC, 0, tensor_count, scalar_count, len(slots), len(records), begin, end, len(scopes), 14, 0, 0, 5, 0]
    words += [params.get(address, MASK) for address in slots]
    words[10] = len(words)
    words += scopes
    words[11] = len(words)
    words += [0] * len(records)
    for i, r in enumerate(records):
        words[words[11] + i] = len(words)
        deps = [ids[d] for d in r["dependencies"] if d in ids]
        words += [
            r["op"],
            *r["kernels"],
            r["blocks"],
            r["sync"],
            r["early"],
            r["timing"],
            r["scope"],
            len(r["tensors"]),
            len(r["scalars"]),
            len(deps),
        ]
        for t in r["tensors"]:
            if t["ndims"] < 1 or t["ndims"] > 5:
                raise ValueError("TMR replay requires device tensors with valid rank")
            shapes = t["shapes"][: t["ndims"]] + [0] * (5 - t["ndims"])
            strides = t["strides"][: t["ndims"]] + [0] * (5 - t["ndims"])
            words += [
                slots[t["buffer"]],
                ids.get(t["owner"], MASK),
                t["bytes"],
                t["offset"],
                t["version"],
                t["ndims"],
                t["dtype"],
                t["manual"],
                t["contiguous"],
                t["space"],
                t["extent"],
                t["tag"],
                *shapes,
                *strides,
            ]
        for value in r["scalars"]:
            words += [value, int(scalar_count > 1 and value == records[0]["scalars"][1])]
        words += deps
    words[1] = len(words)
    return struct.pack(f"<{len(words)}Q", *(int(w) & MASK for w in words)), begin, end


def prepare(args):
    source = args.bundle.resolve()
    output = args.output.resolve()
    case = json.loads((source / "case.json").read_text())
    if case.get("execution") == "native_moe_host_control":
        raise ValueError("Native MoE is excluded from the TMR matrix")
    if output.exists():
        raise ValueError("Prepare TMR into a new output directory")
    compiler = KernelCompiler("a5")
    runtime = "tensormap_and_ringbuffer"
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    tools = args.tools.resolve()
    runner, record_tool = tools / "native-runner", tools / "record-tool"
    orchestration = build / "tmr-replay.so"
    orch_source = ROOT / "production/tmr_replay.cpp"
    includes, support = compiler.get_orchestration_cache_inputs(runtime)
    support_keys = [compile_incore_artifact_key({"platform": "a5"}, p, includes) for p in support]
    key = compile_incore_artifact_key(
        {"runtime": runtime, "compiler": compiler.compile_cache_token(runtime, []), "support": support_keys},
        orch_source,
        includes,
    )
    orchestration = build / f"tmr-replay-{key}.so"
    publish(
        orchestration, get_or_compile_incore(key, lambda: compiler.compile_orchestration(runtime, str(orch_source)))
    )
    info = json.loads(
        subprocess.check_output([str(runner), "--describe-callable", str(source / "callable.bin")], text=True)
    )
    cut = tuple(map(int, (source / "slice.txt").read_text().split()))

    def clone(src, dst):
        if Path(src).suffix in (".bin", ".so"):
            try:
                os.link(src, dst)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    shutil.copytree(source, output, copy_function=clone)
    (output / "callable.bin").unlink()
    used = set()
    for rank_path in sorted(source.glob("rank[0-9].txt")):
        rank = int(rank_path.stem[4:])
        records = json.loads(
            subprocess.check_output(
                [str(record_tool), "--records-json", str(source / f"plan-rank{rank}.bin")], text=True
            )
        )
        lines = rank_path.read_text().splitlines()
        tensor_count = int(lines[2])
        # Fixed replay has no Host control reads. Make its boundary fixtures
        # addressable by device orchestration without changing their bytes.
        lines[3 : 3 + tensor_count] = [
            line.replace("HOST ", "DEVICE ", 1) if line.startswith("HOST ") else line
            for line in lines[3 : 3 + tensor_count]
        ]
        blob, begin, end = encode_plan(records, cut, tensor_count)
        start = begin if (source / "snapshot.txt").exists() else 1
        used.update(k for r in records[start:end] for k in r["kernels"] if r["op"] == 2 and k >= 0)
        plan_name = f"tmr-plan-rank{rank}.bin"
        (output / plan_name).write_bytes(blob)
        lines[2] = str(tensor_count + 1)
        lines.insert(3 + tensor_count, f"DEVICE 5 {len(blob)} 1 {len(blob)} RAW {plan_name}")
        (output / rank_path.name).write_text("\n".join(lines) + "\n")
    flavor = case["flavor"]
    if flavor == "pypto_lib_dsl_all_sync64":
        flavor = "pypto_lib_dsl"
    sources = ROOT / "kernels" / flavor
    manifest = json.loads((sources / "manifest.json").read_text())
    kernels = {k["func_id"]: k for k in manifest["kernels"]}
    sdk = Path(os.environ["ASCEND_HOME_PATH"])

    def compile_one(func):
        k = kernels[func]
        kernel_source = sources / k["source"]
        extra = compiler.get_orchestration_include_dirs(runtime) + [str(sources / p) for p in k["include_dirs"]]
        dsl = flavor == "pypto_lib_dsl" and "/external/" in k["source"]
        if dsl:
            # Generated DSL entry glue names HBG's tensor explicitly; its body
            # remains unchanged, while this derived entry uses the TMR ABI.
            adapted = kernel_source.read_text().replace('"host_build_graph/tensor.h"', '"tensor.h"')
            adapted = adapted.replace("simpler::hbg::Tensor", "Tensor")
            extra.append(str(kernel_source.parent))
            kernel_source = build / f"{flavor}-{func}-entry.cpp"
            publish(kernel_source, adapted.encode())
            extra += [str(p) for p in (sdk / "asc", sdk / "ascendc") if p.is_dir()]
        includes = [
            str(args.pto_isa / "include"),
            str(args.pto_isa / "include/pto"),
            *compiler.get_incore_include_dirs(),
            *extra,
        ]
        meta = {"platform": "a5", "runtime": runtime, "compiler": compiler.incore_compile_cache_token(k["core_type"])}
        if dsl:
            meta["sdk_versions"] = {
                name: (sdk / name).read_text()
                for name in ("compiler/version.info", "share/info/asc-devkit/version.info")
                if (sdk / name).is_file()
            }
        key = compile_incore_artifact_key(meta, kernel_source, includes)
        binary = get_or_compile_incore(
            key,
            lambda: extract_text_section(
                compiler.compile_incore(
                    str(kernel_source),
                    core_type=k["core_type"],
                    pto_isa_root=str(args.pto_isa),
                    extra_include_dirs=extra,
                )
            ),
        )
        target = build / f"{flavor}-{func}-{key}.bin"
        publish(target, binary)
        print(f"TMR kernel ready: {flavor}/{func}", flush=True)
        return func, target

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        binaries = dict(pool.map(compile_one, sorted(used)))
    signature = info["signature"][:]
    signature.insert(tensor_count, 1)

    def sig(values):
        return str(len(values)) + " " + " ".join(map(str, values))

    rows = [sig(signature), str(len(used))]
    for k in info["kernels"]:
        if k["id"] in used:
            rows.append(f"{k['id']} {sig(k['signature'])} {json.dumps(str(binaries[k['id']]))}")
    pack = output / "assembly.txt"
    pack.write_text("\n".join(rows) + "\n")
    subprocess.run(
        [str(runner), "--assemble-orchestration", str(pack), str(orchestration), str(output / "callable.bin")],
        check=True,
    )
    (output / "tmr.json").write_text(
        json.dumps(
            {
                "runtime": runtime,
                "source_bundle": str(source),
                "kernels": sorted(used),
                "orchestration": "AICPU",
                "native_moe_excluded": True,
                "orchestration_sha256": hashlib.sha256(orchestration.read_bytes()).hexdigest(),
                "callable_sha256": hashlib.sha256((output / "callable.bin").read_bytes()).hexdigest(),
                "plans": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("tmr-plan-rank*.bin")
                },
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tools", type=Path, default=ROOT / "build/portable/host")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/tmr")
    parser.add_argument("--pto-isa", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
