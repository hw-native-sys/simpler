#!/usr/bin/env python3
"""Run a CUDA host_schedule vector smoke example."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / ".agents" / "skills" / "cuda-backend-eval" / "scripts" / "cuda_smoke.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument(
        "--op",
        choices=(
            "add",
            "mul",
            "scale",
            "square",
            "axpy",
            "affine",
            "triad",
            "quad",
            "generic_args",
            "generic_args4",
        ),
        default="add",
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    forwarded = [
        str(SMOKE),
        "--runner",
        "worker",
        "--device",
        str(args.device),
        "--n",
        str(args.n),
        "--block-dim",
        str(args.block_dim),
        "--arch",
        args.arch,
        "--op",
        args.op,
    ]
    if not args.build:
        forwarded.append("--no-build")
    if args.output_json is not None:
        forwarded.extend(["--output-json", str(args.output_json)])

    sys.argv = forwarded
    runpy.run_path(str(SMOKE), run_name="__main__")


if __name__ == "__main__":
    main()
