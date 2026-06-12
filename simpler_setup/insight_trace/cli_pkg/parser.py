# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..models import TraceBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MindStudio Insight replay workspaces and final trace artifacts for an incore kernel"
    )
    parser.add_argument("test_module", nargs="?", type=Path)
    parser.add_argument("--backend", choices=[item.value for item in TraceBackend], default=TraceBackend.SIMPLER.value)
    parser.add_argument("--case")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--kernel")
    selector.add_argument("--func-id", type=int)
    selector.add_argument("--kernel-source")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Workspace directory; final Insight artifacts are exported under <output-dir>/insight_export",
    )
    parser.add_argument("--cann-home", type=Path, default=_default_cann_home())
    parser.add_argument("--pto-isa-root", type=Path)
    parser.add_argument("--soc-version")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--hw-block-num", type=int, default=1)
    parser.add_argument("--arg-spec", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--dispatch-id", type=int)  # deprecated, kept for backward compat
    parser.add_argument("--task-id", type=str, help="Task ID for PR792 tensor_dump.json (optional)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate replay workspace only; skip build/collect/export",
    )
    parser.add_argument("--ptoas-root", type=Path)
    parser.add_argument("--source-cpp", type=Path)
    parser.add_argument("--kernel-base-name")
    parser.add_argument("--aicore-arch")
    parser.add_argument("--kernel-symbol")
    return parser


def _default_cann_home() -> Path | None:
    value = os.environ.get("CANN_HOME") or os.environ.get("ASCEND_HOME_PATH")
    return Path(value) if value else None
