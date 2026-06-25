#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
fully_distributed_within_core swimlane renderer.

Renders the per-core execution swimlane emitted by the distributed engine
(``dist_engine_dump_trace`` -> Chrome Trace Event JSON, written when the run is
launched with ``PTO_DIST_SWIMLANE=<path>``) into a Gantt-style PNG. Each row is
a physical lane (block x AIC/AIV0/AIV1); each bar is one executed (sub)task,
colored by ``func_id`` (kernel id).

This is the distributed-runtime counterpart to ``swimlane_converter`` (which
targets the centralized scheduler's L2 records and is empty for this runtime,
since orchestration/scheduling/execution all run on the AI cores). Perfetto
remains the authoritative interactive view — drag the same JSON into
https://ui.perfetto.dev/. This tool is for a quick static picture.

Usage:
    # Latest outputs/dist_swimlane/*.json (or outputs/**/*swimlane*.json):
    python -m simpler_setup.tools.dist_swimlane_render
    python -m simpler_setup.tools.dist_swimlane_render path/to/bgemm_swimlane.json
    python -m simpler_setup.tools.dist_swimlane_render in.json -o out.png
    python -m simpler_setup.tools.dist_swimlane_render in.json --names 0=GEMM,1=ADD
"""

import argparse
import json
import sys
from pathlib import Path

LANE_NAMES = ("AIC", "AIV0", "AIV1")

# Distinct, color-blind-friendly palette indexed by func_id (wraps if exceeded).
_PALETTE = [
    "#2c7fb8",  # blue
    "#de2d26",  # red
    "#31a354",  # green
    "#756bb1",  # purple
    "#e6550d",  # orange
    "#636363",  # gray
    "#c51b8a",  # magenta
    "#1c9099",  # teal
]


def _resolve_input(arg: str | None) -> Path | None:
    """Resolve the input JSON: explicit arg, else the most recent candidate
    under outputs/dist_swimlane/ then outputs/ (by mtime)."""
    if arg:
        p = Path(arg)
        if not p.is_file():
            print(f"Error: input file not found: {p}", file=sys.stderr)
            return None
        return p
    candidates: list[Path] = []
    d = Path("outputs/dist_swimlane")
    if d.is_dir():
        candidates += list(d.glob("*.json"))
    if not candidates:
        out = Path("outputs")
        if out.is_dir():
            candidates += list(out.glob("**/*swimlane*.json"))
    if not candidates:
        print(
            "Error: no input given and no outputs/dist_swimlane/*.json found. "
            "Run with PTO_DIST_SWIMLANE=<path> first, then pass that path.",
            file=sys.stderr,
        )
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_names(spec: str | None) -> dict[int, str]:
    """Parse a ``0=GEMM,1=ADD`` style func_id->name mapping."""
    names: dict[int, str] = {}
    if not spec:
        return names
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            names[int(k.strip())] = v.strip()
        except ValueError:
            continue
    return names


def _load_trace(path: Path) -> tuple[dict[tuple[int, int], str], list[dict]]:
    """Return (lane_name_by_(pid,tid), duration_events) from a Chrome trace."""
    data = json.loads(path.read_text())
    events = data.get("traceEvents", [])
    lane_names: dict[tuple[int, int], str] = {}
    durs: list[dict] = []
    for e in events:
        ph = e.get("ph")
        if ph == "M" and e.get("name") == "thread_name" and "tid" in e:
            lane_names[(e["pid"], e["tid"])] = e.get("args", {}).get("name", f'{e["pid"]}:{e["tid"]}')
        elif ph == "X":
            durs.append(e)
    return lane_names, durs


def render(input_path: Path, output_path: Path, names: dict[int, str], title: str | None, verbose: bool) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("Error: matplotlib is required (pip install matplotlib).", file=sys.stderr)
        return 1

    lane_names, durs = _load_trace(input_path)
    if not durs:
        print(f"Error: no execution (ph=X) events in {input_path}.", file=sys.stderr)
        return 1

    # Lane rows: prefer the metadata order; fall back to whatever events carry.
    lanes = sorted(lane_names.keys()) if lane_names else sorted({(e["pid"], e["tid"]) for e in durs})
    row = {lk: i for i, lk in enumerate(lanes)}

    def label_for(lk: tuple[int, int]) -> str:
        return lane_names.get(lk, f'{LANE_NAMES[lk[1]] if lk[1] < len(LANE_NAMES) else lk[1]} (blk{lk[0]})')

    func_ids = sorted({int(e.get("args", {}).get("func_id", -1)) for e in durs})

    # Resolve func_id -> name. Priority: --names (CLI) > args.name baked into the
    # JSON (scene_test injects the incore function name when capturing) > none.
    auto_names: dict[int, str] = {}
    for e in durs:
        a = e.get("args", {})
        nm = a.get("name")
        if nm:
            auto_names[int(a.get("func_id", -1))] = nm
    effective_names = {**auto_names, **names}

    def color_for(fid: int) -> str:
        if fid < 0:
            return "#999999"
        return _PALETTE[fid % len(_PALETTE)]

    fig_h = max(2.5, 0.5 * len(lanes) + 1.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    for e in durs:
        lk = (e["pid"], e["tid"])
        if lk not in row:
            continue
        fid = int(e.get("args", {}).get("func_id", -1))
        ax.barh(row[lk], e["dur"], left=e["ts"], height=0.7, color=color_for(fid), edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([label_for(lk) for lk in lanes], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("time (us, relative to run start)")
    ax.set_title(title or f"fully_distributed_within_core per-core execution swimlane\n{input_path.name}")
    ax.grid(axis="x", alpha=0.3)

    handles = [
        Patch(color=color_for(fid), label=effective_names.get(fid, f"func {fid}" if fid >= 0 else "unknown"))
        for fid in func_ids
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close(fig)

    if verbose:
        per_lane: dict[tuple[int, int], int] = {}
        for e in durs:
            per_lane[(e["pid"], e["tid"])] = per_lane.get((e["pid"], e["tid"]), 0) + 1
        print("tasks per lane:")
        for lk in lanes:
            print(f"  {label_for(lk):28s} n={per_lane.get(lk, 0)}")
    print(f"✓ Rendered {len(durs)} events across {len(lanes)} lanes")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"\nFor the interactive view, drag {input_path} into https://ui.perfetto.dev/")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a fully_distributed_within_core execution swimlane JSON to a PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Chrome-trace JSON from PTO_DIST_SWIMLANE. Default: latest outputs/dist_swimlane/*.json.",
    )
    p.add_argument("-o", "--output", default=None, help="Output PNG path. Default: <input>.png next to the input.")
    p.add_argument(
        "--names",
        default=None,
        help="func_id->name legend map, e.g. '0=GEMM,1=ADD'. Without it, lanes are labeled 'func N'.",
    )
    p.add_argument("--title", default=None, help="Override the plot title.")
    p.add_argument("-v", "--verbose", action="store_true", help="Also print per-lane task counts.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    input_path = _resolve_input(args.input)
    if input_path is None:
        return 1
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")
    return render(input_path, output_path, _parse_names(args.names), args.title, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
