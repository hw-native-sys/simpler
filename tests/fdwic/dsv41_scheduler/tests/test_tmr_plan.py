# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Captured aliases and dependency identities survive conversion to AICPU replay."""

import runpy
import struct
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
ENCODE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "production/tmr_prepare.py"))["encode_plan"]


def tensor(buffer, owner, tag):
    return dict(
        buffer=buffer,
        bytes=16,
        owner=owner,
        offset=0,
        version=0,
        ndims=1,
        dtype=0,
        manual=0,
        contiguous=1,
        space=1,
        extent=4,
        tag=tag,
        shapes=[4, 0, 0, 0, 0],
        strides=[1, 0, 0, 0, 0],
    )


def record(op, identity, tensors, dependencies=(), scalars=()):
    return dict(
        op=op,
        id=identity,
        tensors=tensors,
        dependencies=list(dependencies),
        scalars=list(scalars),
        kernels=[-1, 7, -1],
        blocks=64,
        sync=1,
        early=0,
        timing=-1,
        scope=0,
    )


def test_alias_boundary_and_opaque_task_identity():
    records = [
        record(0, 0, [tensor(100, -1, 0), tensor(100, -1, 0)], scalars=[0, 0x9000]),
        record(1, 700, [tensor(200, -1, 1)]),
        record(2, 800, [tensor(200, 700, 2)]),
        record(2, 900, [tensor(200, 800, 0)], dependencies=[800], scalars=[0x9000, 8]),
    ]
    blob, begin, end = ENCODE(records, (900, 900), 3)
    words = struct.unpack(f"<{len(blob) // 8}Q", blob)
    assert (begin, end) == (3, 4)
    assert words[2:9] == (3, 2, 2, 4, 3, 4, 0)
    # Aliased external arguments bind to the last supplied view; the live-in
    # temporary uses argument 2, after both external argument positions.
    assert words[words[9] : words[9] + 2] == (1, 2)
    row = words[words[words[11] + 3] :]
    assert row[:12] == (2, (1 << 64) - 1, 7, (1 << 64) - 1, 64, 1, 0, (1 << 64) - 1, 0, 1, 2, 1)
    assert row[12:14] == (1, 2)  # buffer slot 1, creator record 2
    assert row[34:] == (0x9000, 1, 8, 0, 2)  # relocated communication context; dependency record 2
    with pytest.raises(ValueError, match="boundary arguments"):
        ENCODE(records, (900, 900), 2)
