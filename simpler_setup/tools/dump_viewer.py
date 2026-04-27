#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor dump viewer — extract tensors from tensors.bin to txt files.

Filters (freely combinable):
    --task   Filter by task_id (hex, e.g. 0x0000000200000a00)
    --func   Filter by func_id (int)
    --stage  Filter by stage (before / after)
    --role   Filter by role (input / output / inout)
    --arg    Filter by arg_index (int)

With no filters: lists all tensors.
With filters: lists matching tensors. Add --export to save them to txt.

Usage:
    # List all tensors (auto-picks latest outputs/*/tensor_dump dir under ./outputs/)
    python -m simpler_setup.tools.dump_viewer

    # List all tensors in a specific dump dir
    python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/tensor_dump/

    # List before-dispatch inputs of func_id=3 (latest dir)
    python -m simpler_setup.tools.dump_viewer --func 3 --stage before --role input

    # Export them to txt
    python -m simpler_setup.tools.dump_viewer outputs/<case>/tensor_dump/ --func 3 --stage before --export

    # Export a specific tensor by index
    python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/tensor_dump/ --index 42 --export
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

DTYPE_INFO = {
    "float32": ("f", 4),
    "float16": ("e", 2),
    "bfloat16": (None, 2),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint64": ("Q", 8),
    "int16": ("h", 2),
    "int8": ("b", 1),
    "uint8": ("B", 1),
}


def bfloat16_to_float32(raw: int) -> float:
    return struct.unpack("f", struct.pack("I", raw << 16))[0]


def read_tensor_data(bin_path: Path, offset: int, size: int) -> bytes:
    with open(bin_path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def decode_elements(data: bytes, dtype: str, count: int) -> list:
    dtype_lower = dtype.lower()
    fmt, elem_sz = DTYPE_INFO.get(dtype_lower, (None, 1))
    if dtype_lower == "bfloat16":
        return [bfloat16_to_float32(struct.unpack_from("H", data, i * 2)[0]) for i in range(count)]
    if fmt is None:
        return [f"0x{data[i]:02x}" for i in range(count)]
    return [struct.unpack_from(fmt, data, i * elem_sz)[0] for i in range(count)]


def format_element(val, dtype: str) -> str:
    dtype_lower = dtype.lower()
    if isinstance(val, float):
        if dtype_lower == "float32":
            return f"{val:.6g}"
        elif dtype_lower == "float16":
            return f"{val:.4g}"
        elif dtype_lower == "bfloat16":
            return f"{val:.3g}"
    return str(val)


def tensor_filename(t: dict) -> str:
    stage_map = {"before_dispatch": "before", "after_completion": "after"}
    role_map = {"input": "in", "output": "out", "inout": "inout"}
    stage_str = stage_map.get(t["stage"], t["stage"])
    role_str = role_map.get(t["role"], t["role"])
    return f"task_{t['task_id']}_s{t['subtask_id']}_{stage_str}_{role_str}{t['arg_index']}.txt"


def write_tensor(tensor: dict, bin_path: Path, out):
    t = tensor
    out.write(f"# task_id: {t['task_id']}\n")
    out.write(f"# subtask_id: {t['subtask_id']}\n")
    out.write(f"# role: {t['role']}\n")
    out.write(f"# stage: {t['stage']}\n")
    out.write(f"# arg_index: {t['arg_index']}\n")
    out.write(f"# func_id: {t['func_id']}\n")
    out.write(f"# dtype: {t['dtype']}\n")
    out.write(f"# is_contiguous: {t['is_contiguous']}\n")
    out.write(f"# shape: {t['shape']}\n")
    out.write(f"# raw_shape: {t['raw_shape']}\n")
    out.write(f"# offsets: {t['offsets']}\n")

    if t.get("overwritten"):
        out.write("# DATA OVERWRITTEN (host too slow)\n")
        return
    if t.get("truncated"):
        out.write("# DATA TRUNCATED (tensor too large for arena)\n")

    bin_size = t.get("bin_size", 0)
    if bin_size == 0:
        out.write("# (no data)\n")
        return

    data = read_tensor_data(bin_path, t["bin_offset"], bin_size)
    shape = t["shape"]
    numel = 1
    for d in shape:
        numel *= d
    if numel == 0:
        numel = 1

    _, elem_sz = DTYPE_INFO.get(t["dtype"].lower(), (None, 1))
    max_from_bytes = len(data) // elem_sz
    numel = min(numel, max_from_bytes)

    elements = decode_elements(data, t["dtype"], numel)
    formatted = [format_element(v, t["dtype"]) for v in elements]

    out.write("\n# Overview:\n")
    col_width = max(len(s) for s in formatted) if formatted else 1
    last_dim = shape[-1] if shape else numel
    if last_dim == 0:
        last_dim = numel

    for i, s in enumerate(formatted):
        if i > 0 and (i % last_dim) == 0:
            out.write("\n")
        elif i > 0:
            out.write(" ")
        out.write(f"{s:>{col_width}}")
    out.write("\n")

    out.write("\n# Detail:\n")
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]

    for i, s in enumerate(formatted):
        idx = []
        rem = i
        for d in range(len(shape)):
            idx.append(rem // strides[d])
            rem %= strides[d]
        idx_str = ", ".join(str(x) for x in idx)
        out.write(f"[{idx_str}] {s}\n")


def export_tensor(tensor: dict, bin_path: Path, dump_dir: Path):
    txt_dir = dump_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    fname = tensor_filename(tensor)
    txt_path = txt_dir / fname
    with open(txt_path, "w") as f:
        write_tensor(tensor, bin_path, f)
    return txt_path


def collect_valid_values(tensors: list, field: str) -> list:
    return sorted(set(str(t[field]) for t in tensors))


def list_tensors(tensors: list):
    print(
        f"{'idx':>6}  {'task_id':>18}  {'s':>1}  {'stage':>7}  {'role':>5}"
        f"  {'arg':>3}  {'func':>4}  {'dtype':>8}  {'shape':<20}  {'bytes':>10}"
    )
    print("-" * 100)
    for i, t in enumerate(tensors):
        stage_short = "before" if t["stage"] == "before_dispatch" else "after"
        print(
            f"{i:>6}  {t['task_id']:>18}  {t['subtask_id']:>1}  {stage_short:>7}  {t['role']:>5}  "
            f"{t['arg_index']:>3}  {t['func_id']:>4}  {t['dtype']:>8}  {str(t['shape']):<20}  {t['bin_size']:>10}"
        )


def _resolve_dump_dir(dump_dir_arg: str | None) -> Path:
    if dump_dir_arg is not None:
        return Path(dump_dir_arg)
    # Tests/runtime now write tensor_dump under outputs/<case>/tensor_dump/.
    candidates = sorted(Path("outputs").glob("*/tensor_dump"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("Error: no outputs/*/tensor_dump directory found", file=sys.stderr)
        sys.exit(1)
    print(f"Using latest dump directory: {candidates[-1]}")
    return candidates[-1]


def _apply_filters(tensors: list, args: argparse.Namespace) -> list:
    filtered = tensors

    if args.task:
        valid = collect_valid_values(tensors, "task_id")
        if args.task not in valid:
            print(f"Error: --task {args.task} not found.", file=sys.stderr)
            sample = valid[:5]
            print(f"  Valid task_ids (showing first {len(sample)}): {', '.join(sample)}", file=sys.stderr)
            sys.exit(1)
        filtered = [t for t in filtered if t["task_id"] == args.task]

    if args.func is not None:
        valid = collect_valid_values(filtered, "func_id")
        if str(args.func) not in valid:
            print(f"Error: --func {args.func} not found in current selection.", file=sys.stderr)
            print(f"  Valid func_ids: {', '.join(valid)}", file=sys.stderr)
            sys.exit(1)
        filtered = [t for t in filtered if t["func_id"] == args.func]

    if args.stage:
        stage_map = {"before": "before_dispatch", "after": "after_completion"}
        if args.stage not in stage_map:
            print(f"Error: --stage must be 'before' or 'after', got '{args.stage}'", file=sys.stderr)
            sys.exit(1)
        filtered = [t for t in filtered if t["stage"] == stage_map[args.stage]]

    if args.role:
        valid_roles = {"input", "output", "inout"}
        if args.role not in valid_roles:
            print(f"Error: --role must be one of {valid_roles}, got '{args.role}'", file=sys.stderr)
            sys.exit(1)
        filtered = [t for t in filtered if t["role"] == args.role]

    if args.arg is not None:
        valid = collect_valid_values(filtered, "arg_index")
        if str(args.arg) not in valid:
            print(f"Error: --arg {args.arg} not found in current selection.", file=sys.stderr)
            print(f"  Valid arg_indices: {', '.join(valid)}", file=sys.stderr)
            sys.exit(1)
        filtered = [t for t in filtered if t["arg_index"] == args.arg]

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Tensor dump viewer — extract tensors from tensors.bin to txt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dump_dir",
        nargs="?",
        default=None,
        help="Path to outputs/<case>_<ts>/tensor_dump directory (default: latest outputs/*/tensor_dump dir)",
    )
    parser.add_argument("--task", "-t", help="Filter by task_id (e.g. 0x0000000200000a00)")
    parser.add_argument("--func", "-f", type=int, help="Filter by func_id")
    parser.add_argument("--stage", "-s", help="Filter by stage (before / after)")
    parser.add_argument("--role", "-r", help="Filter by role (input / output / inout)")
    parser.add_argument("--arg", "-a", type=int, help="Filter by arg_index")
    parser.add_argument("--index", "-i", type=int, help="Select tensor by index in manifest")
    parser.add_argument("--export", "-e", action="store_true", help="Export filtered tensors to txt")
    args = parser.parse_args()

    dump_dir = _resolve_dump_dir(args.dump_dir)
    manifest_files = list(dump_dir.glob("*.json"))
    if not manifest_files:
        print(f"Error: no manifest JSON found in {dump_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_files[0]) as f:
        manifest = json.load(f)

    bin_path = dump_dir / manifest.get("bin_file", "tensors.bin")
    tensors = manifest["tensors"]

    filtered = _apply_filters(tensors, args)

    # --- Select by index ---
    if args.index is not None:
        if args.index < 0 or args.index >= len(tensors):
            print(f"Error: --index {args.index} out of range (0-{len(tensors) - 1})", file=sys.stderr)
            sys.exit(1)
        filtered = [tensors[args.index]]
        args.export = True  # --index always exports

    if not filtered:
        print("No tensors match the given filters.", file=sys.stderr)
        sys.exit(1)

    # --- Export or list ---
    has_filters = any([args.task, args.func is not None, args.stage, args.role, args.arg is not None])

    if args.export or args.index is not None:
        for t in filtered:
            txt_path = export_tensor(t, bin_path, dump_dir)
            print(f"Saved: {txt_path}")
        print(f"\nExported {len(filtered)} tensor(s) to {dump_dir / 'txt/'}")
    else:
        if has_filters:
            print(f"Filtered: {len(filtered)}/{len(tensors)} tensors")
            print(f"Add --export to save these tensors to {dump_dir / 'txt/'}\n")
        list_tensors(filtered)


if __name__ == "__main__":
    main()
