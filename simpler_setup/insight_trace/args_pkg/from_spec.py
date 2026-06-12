# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import json
import struct
from pathlib import Path

from ..models import TraceArg, TraceScalarArg, TraceTensorArg


def _load_arg_spec(path: Path) -> tuple[TraceArg, ...]:
    raw = json.loads(path.read_text())
    result: list[TraceArg] = []
    # Handle both {"args": [...]} and direct [...] formats
    if isinstance(raw, dict) and "args" in raw:
        items = raw["args"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    for item in items:
        if item["kind"] == "tensor":
            result.append(
                TraceTensorArg(
                    index=int(item["index"]),
                    name=item["name"],
                    dtype=item["dtype"],
                    shape=tuple(int(dim) for dim in item["shape"]),
                    role=item.get("role", "input"),
                    fill=item.get("fill", "zero"),
                )
            )
        elif item["kind"] == "scalar":
            value = item["value"]
            pack_mode = item.get("pack_mode", "value")
            if (pack_mode == "bits" or item["dtype"] == "FLOAT32_BITS") and isinstance(value, float):
                value = _f32_bits(value)
            result.append(
                TraceScalarArg(
                    index=int(item["index"]),
                    name=item["name"],
                    dtype=item["dtype"],
                    value=value,
                    pack_mode=pack_mode,
                )
            )
        else:
            raise ValueError(f"Unknown arg kind: {item['kind']}")
    return tuple(sorted(result, key=lambda arg: arg.index))


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]
