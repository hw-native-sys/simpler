#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare a standalone native MoE bundle; numerical files remain local."""

import gzip
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workloads/native_moe/source.json"
INPUT_NAMES = [
    "x_hc",
    "pre_mix",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_weight",
    "gate_weight",
    "correction_bias",
    "routed_w1",
    "routed_w1_scale",
    "routed_w2",
    "routed_w2_scale",
    "routed_w3",
    "routed_w3_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w2",
    "shared_w2_scale",
    "shared_w3",
    "shared_w3_scale",
    "next_pre_mix",
    "x_mixed",
    "x_next",
]
COMM = [
    ("recv_meta", 2, [2, 64]),
    ("recv_x", 4, [10240, 5120]),
    ("recv_scale", 5, [10240, 160]),
    ("recv_weights", 0, [10240, 8]),
    ("recv_routes", 2, [10240, 8]),
    ("data_arrived", 2, [2, 64, 128]),
    ("routed_output", 6, [480, 5120]),
    ("combine_arrived", 2, [2, 64, 128]),
]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_bundle(bundle):
    bundle = Path(bundle)
    source = json.loads(SOURCE.read_text())
    meta = json.loads((bundle / "native.json").read_text())
    if meta["source_commit"] != source["source_commit"]:
        raise ValueError("Native bundle uses a different model fixture")
    if hashlib.sha256((bundle / "callable.bin").read_bytes()).hexdigest() != meta["callable_sha256"]:
        raise ValueError("Native callable changed; prepare the bundle again")
    for name, size in meta["files"].items():
        path = bundle / name
        if path.resolve().parent != bundle.resolve() or path.stat().st_size != size:
            raise ValueError(f"Invalid native input file: {name}")
    for rank in range(2):
        rows = (bundle / f"rank{rank}.txt").read_text().splitlines()
        if rows[0] != "DSV41_MOE_V1" or rows[2] != "33" or not rows[34].startswith("HOST 2 8 1 2 "):
            raise ValueError("Native MoE must retain the live HOST num_tokens argument")
    return meta


def prepare_bundle(bundle, build_dir, reuse=None):
    bundle = Path(bundle)
    if bundle.exists():
        raise ValueError("Prepare native MoE into a new bundle directory")
    if reuse is not None:
        validate_bundle(reuse)
        shutil.copytree(reuse, bundle)
        return validate_bundle(bundle)
    build_dir = Path(build_dir)
    callable_path = build_dir / "callable.bin"
    if not callable_path.is_file():
        raise ValueError("Build native MoE with build.sh --flavor pypto_lib_latest first")
    inputs = load_module("moe_inputs")
    cache = load_module("weight_cache")
    source = json.loads(SOURCE.read_text())
    specs, cache_info = cache.cached_weight_specs(
        inputs, (72, 72), source["source_commit"], ROOT / "build/native-weights"
    )
    torch = inputs.torch
    specs = {spec.name: spec for spec in specs if hasattr(spec, "shape")}
    dtype_ids = {
        torch.float32: 0,
        torch.float16: 1,
        torch.int32: 2,
        torch.int8: 4,
        torch.uint8: 5,
        torch.bfloat16: 6,
        torch.float8_e4m3fn: 12,
    }
    if hasattr(torch, "float8_e8m0fnu"):
        dtype_ids[torch.float8_e8m0fnu] = 13
    sizes = {0: 4, 2: 4, 4: 1, 5: 1, 6: 2}
    bundle.mkdir(parents=True)
    (bundle / "zero.bin").write_bytes(b"\0")
    manifests = [[], []]
    for name in INPUT_NAMES + ["num_tokens", "force_topk"]:
        spec = specs[name]
        value = spec.create_tensor()
        for rank in range(2):
            tensor = (value if name == "num_tokens" else value[rank]).contiguous()
            dtype, shape = dtype_ids[tensor.dtype], list(tensor.shape)
            if spec.init_value is None:
                mode, filename = "REPEAT", "zero.bin"
            else:
                mode, filename = "RAW", f"{name}-rank{rank}.bin"
                with (bundle / filename).open("wb") as stream:
                    stream.write(memoryview(tensor.view(torch.uint8).numpy()))
            kind = "HOST" if name == "num_tokens" else "DEVICE"
            line = f"{kind} {dtype} {tensor.numel() * tensor.element_size()} {len(shape)} "
            line += " ".join(map(str, shape)) + f" {mode} {filename}"
            manifests[rank].append(line)
    offset = 0
    comm_rows = []
    for _, dtype, shape in COMM:
        size = math.prod(shape) * sizes[dtype]
        comm_rows.append(f"COMM {dtype} {size} {len(shape)} " + " ".join(map(str, shape)) + f" {offset}")
        offset += (size + 63) // 64 * 64
    for rank, rows in enumerate(manifests):
        rows[23:23] = comm_rows
        text = f"DSV41_MOE_V1\n2 {offset}\n33\n" + "\n".join(rows)
        text += "\n10\nRANK\nVALUE 1\n" + "COMM\n" * 8 + "VALIDATE_ZERO 0\n"
        (bundle / f"rank{rank}.txt").write_text(text)
    shutil.copy2(callable_path, bundle / "callable.bin")
    original = gzip.decompress((ROOT / "workloads/native_moe/vendor/orchestration.cpp.gz").read_bytes()).decode()
    reads = [
        {"file": "workloads/native_moe/vendor/orchestration.cpp.gz", "line": i, "code": line.strip()}
        for i, line in enumerate(original.splitlines(), 1)
        if "get_tensor_data<int32_t>(ext_num_tokens" in line
    ]
    if not reads:
        raise ValueError("Native orchestration has no executable Host control read")
    meta = {
        "source_commit": source["source_commit"],
        "host_source_sha256": source["host_source_sha256"],
        "callable_sha256": hashlib.sha256(callable_path.read_bytes()).hexdigest(),
        "files": {p.name: p.stat().st_size for p in bundle.iterdir() if p.is_file()},
        "counts": [72, 72],
        "control_reads": reads,
        "weight_cache": cache_info,
    }
    (bundle / "native.json").write_text(json.dumps(meta, indent=2) + "\n")
    (bundle / "case.json").write_text(
        json.dumps(
            {
                "case": "moe",
                "flavor": "pypto_lib_latest",
                "device_count": 2,
                "execution": "native_moe_host_control",
                "source_commit": source["source_commit"],
            },
            indent=2,
        )
        + "\n"
    )
    return validate_bundle(bundle)
