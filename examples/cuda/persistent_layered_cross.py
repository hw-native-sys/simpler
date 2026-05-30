#!/usr/bin/env python3
"""Run the CUDA persistent-device layered-cross graph example."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / ".agents" / "skills" / "cuda-backend-eval" / "scripts" / "cuda_persistent_smoke.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--scheduler-blocks", type=int, default=3)
    parser.add_argument("--worker-blocks", type=int, default=4)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--stream-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    forwarded = [
        str(SMOKE),
        "--device",
        str(args.device),
        "--task-count",
        "9",
        "--n",
        str(args.n),
        "--arch",
        args.arch,
        "--mode",
        "dag",
        "--dag-shape",
        "graph_descriptor_layered_cross",
        "--block-dim",
        str(args.block_dim),
        "--scheduler-blocks",
        str(args.scheduler_blocks),
        "--worker-blocks",
        str(args.worker_blocks),
        "--repeat-runs",
        str(args.repeat_runs),
        "--stream-id",
        str(args.stream_id),
    ]
    if args.output_json is not None:
        forwarded.extend(["--output-json", str(args.output_json)])

    sys.argv = forwarded
    runpy.run_path(str(SMOKE), run_name="__main__")


if __name__ == "__main__":
    main()
