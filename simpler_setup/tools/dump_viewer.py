#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Args dump viewer — extract args from args.bin to txt files.

Filters (freely combinable):
    --task   Filter by task_id (hex, e.g. 0x0000000200000a00)
    --stage  Filter by stage (before / after)
    --role   Filter by role (input / output / inout)
    --arg    Filter by arg_index (int)

With no filters: lists all args.
With filters: lists matching args. Add --export to save them to txt.

Usage:
    # List all args (auto-picks latest outputs/*/args_dump dir under ./outputs/)
    python -m simpler_setup.tools.dump_viewer

    # List all args in a specific dump dir
    python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/args_dump/

    # List before-dispatch inputs of task_id 0x... (latest dir)
    python -m simpler_setup.tools.dump_viewer --task 0x0000000200000a00 --stage before --role input

    # Export them to txt
    python -m simpler_setup.tools.dump_viewer outputs/<case>/args_dump/ --stage before --export

    # Export a specific arg by index
    python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/args_dump/ --index 42 --export
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
    "uint16": ("H", 2),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "uint32": ("I", 4),
    "bool": ("?", 1),
}


def bfloat16_to_float32(raw: int) -> float:
    return struct.unpack("f", struct.pack("I", raw << 16))[0]


def read_arg_data(bin_path: Path, offset: int, size: int) -> bytes:
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


def arg_filename(arg: dict) -> str:
    stage_map = {"before_dispatch": "before", "after_completion": "after"}
    role_map = {"input": "in", "output": "out", "inout": "inout"}
    stage_str = stage_map.get(arg["stage"], arg["stage"])
    role_str = role_map.get(arg["role"], arg["role"])
    return f"task_{arg['task_id']}_{stage_str}_{role_str}{arg['arg_index']}.txt"


def write_arg(arg: dict, bin_path: Path | None, out):
    out.write(f"# task_id: {arg['task_id']}\n")
    out.write(f"# role: {arg['role']}\n")
    out.write(f"# stage: {arg['stage']}\n")
    out.write(f"# arg_index: {arg['arg_index']}\n")
    out.write(f"# dtype: {arg['dtype']}\n")
    out.write(f"# kind: {arg.get('kind', 'tensor')}\n")
    out.write(f"# is_contiguous: {arg['is_contiguous']}\n")
    out.write(f"# shape: {arg['shape']}\n")
    out.write(f"# strides: {arg['strides']}\n")
    out.write(f"# start_offset: {arg['start_offset']}\n")

    if arg.get("overwritten"):
        out.write("# DATA OVERWRITTEN (host too slow)\n")
        return
    if arg.get("truncated"):
        out.write("# DATA TRUNCATED (tensor too large for arena)\n")

    if arg.get("kind") == "scalar":
        val = arg.get("value")
        if arg.get("dtype", "").upper() == "BOOL":
            val = "true" if val else "false"
        out.write(f"# value: {val}\n")
        return

    bin_size = arg.get("bin_size", 0)
    if bin_size == 0:
        out.write("# (no data)\n")
        return

    if bin_path is None:
        raise ValueError("bin_path is None but bin_size > 0 (corrupt manifest?)")
    data = read_arg_data(bin_path, arg["bin_offset"], bin_size)
    shape = arg["shape"]
    numel = 1
    for d in shape:
        numel *= d
    if numel == 0:
        numel = 1

    _, elem_sz = DTYPE_INFO.get(arg["dtype"].lower(), (None, 1))
    max_from_bytes = len(data) // elem_sz
    numel = min(numel, max_from_bytes)

    elements = decode_elements(data, arg["dtype"], numel)
    formatted = [format_element(v, arg["dtype"]) for v in elements]

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


def export_arg(arg: dict, bin_path: Path | None, dump_dir: Path):
    txt_dir = dump_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    fname = arg_filename(arg)
    txt_path = txt_dir / fname
    with open(txt_path, "w") as f:
        write_arg(arg, bin_path, f)
    return txt_path


def collect_valid_values(tensors: list, field: str) -> list:
    return sorted(set(str(t[field]) for t in tensors))


def list_args(tensors: list):
    print(
        f"{'idx':>6}  {'task_id':>18}  {'stage':>7}  {'role':>5}"
        f"  {'arg':>3}  {'dtype':>8}  {'shape':<20}  {'bytes':>10}"
    )
    print("-" * 100)
    for i, t in enumerate(tensors):
        stage_short = "before" if t["stage"] == "before_dispatch" else "after"
        print(
            f"{i:>6}  {t['task_id']:>18}  {stage_short:>7}  {t['role']:>5}  "
            f"{t['arg_index']:>3}  {t['dtype']:>8}  {str(t['shape']):<20}  {t['bin_size']:>10}"
        )


def _resolve_dump_dir(dump_dir_arg: str | None) -> Path:
    if dump_dir_arg is not None:
        return Path(dump_dir_arg)
    # Tests/runtime now write args_dump under outputs/<case>/args_dump/.
    candidates = sorted(Path("outputs").glob("*/args_dump"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("Error: no outputs/*/args_dump directory found", file=sys.stderr)
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
        description="Args dump viewer — extract args from args.bin to txt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dump_dir",
        nargs="?",
        default=None,
        help="Path to outputs/<case>_<ts>/args_dump directory (default: latest outputs/*/args_dump dir)",
    )
    parser.add_argument("--task", "-t", help="Filter by task_id (e.g. 0x0000000200000a00)")
    parser.add_argument("--stage", "-s", help="Filter by stage (before / after)")
    parser.add_argument("--role", "-r", help="Filter by role (input / output / inout)")
    parser.add_argument("--arg", "-a", type=int, help="Filter by arg_index")
    parser.add_argument("--index", "-i", type=int, help="Select arg by index in manifest")
    parser.add_argument("--export", "-e", action="store_true", help="Export filtered args to txt")
    args = parser.parse_args()

    dump_dir = _resolve_dump_dir(args.dump_dir)
    manifest_path = dump_dir / "args_dump.json"
    if not manifest_path.exists():
        print(f"Error: args_dump.json not found in {dump_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # full_json_only dumps (level 3) carry no payload: bin_file is null and
    # there is no .bin to export from — listing metadata still works.
    bin_name = manifest.get("bin_file", "args.bin")
    bin_path = (dump_dir / bin_name) if bin_name else None
    args_data = manifest.get("args", manifest.get("tensors", []))

    filtered = _apply_filters(args_data, args)

    # --- Select by index ---
    if args.index is not None:
        if args.index < 0 or args.index >= len(args_data):
            print(f"Error: --index {args.index} out of range (0-{len(args_data) - 1})", file=sys.stderr)
            sys.exit(1)
        filtered = [args_data[args.index]]
        args.export = True  # --index always exports

    if not filtered:
        print("No args match the given filters.", file=sys.stderr)
        sys.exit(1)

    # --- Export or list ---
    has_filters = any([args.task, args.stage, args.role, args.arg is not None])

    if args.export or args.index is not None:
        for arg in filtered:
            txt_path = export_arg(arg, bin_path, dump_dir)
            print(f"Saved: {txt_path}")
        print(f"\nExported {len(filtered)} arg(s) to {dump_dir / 'txt/'}")
    else:
        if has_filters:
            print(f"Filtered: {len(filtered)}/{len(args_data)} args")
            print(f"Add --export to save these args to {dump_dir / 'txt/'}\n")
        list_args(filtered)


if __name__ == "__main__":
    main()
