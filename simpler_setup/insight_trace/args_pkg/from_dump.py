# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load kernel args from tensor_dump.json (PR792 unified format)."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import TraceArg, TraceTensorArg


def load_kernel_dump_args(
    dump_dir: Path,
    task_id: str | None = None,
) -> tuple[TraceArg, ...]:
    """
    Load kernel replay args from tensor_dump.json (PR792 unified format).

    Args:
        dump_dir: outputs/<test>/tensor_dump/ or the tensor_dump.json parent directory
        task_id: task ID in hex string (optional, PR792 format)

    Returns:
        TraceArg tuple sorted by arg_index

    Note:
        PR792 unified format: no kind/value fields, scalar detected by shape=[] and bin_size=0.
        Scalar arg values must come from arg_spec or scene_test fallback.
    """
    manifest_path = _find_manifest(dump_dir)
    manifest = json.loads(manifest_path.read_text())

    tensors = manifest.get("tensors", [])
    result: dict[int, TraceArg] = {}

    for t in tensors:
        # Filter by task_id if specified
        if task_id is not None and t.get("task_id") != task_id:
            continue
        # Only before_dispatch stage has input args
        if t.get("stage") != "before_dispatch":
            continue

        index = t.get("arg_index", 0)
        if index in result:
            continue

        # Detect scalar by shape=[] and bin_size=0 (PR792 unified format)
        shape = t.get("shape", [])
        bin_size = t.get("bin_size", 0)
        is_scalar = shape == [] and bin_size == 0

        if is_scalar:
            # Scalar arg values not in JSON - must come from arg_spec or scene_test
            continue
        else:
            # Tensor args - get shape info from JSON
            result[index] = TraceTensorArg(
                index=index,
                name=f"arg{index}",
                dtype=t.get("dtype", "float32"),
                shape=tuple(int(d) for d in shape),
            )

    return tuple(result[index] for index in sorted(result.keys()))


def _find_manifest(dump_dir: Path) -> Path:
    """Find tensor_dump.json under dump_dir."""
    dump_dir = Path(dump_dir)
    candidates = [
        dump_dir / "tensor_dump.json",
        dump_dir / "tensor_dump" / "tensor_dump.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError(f"tensor_dump.json not found under {dump_dir}")
