# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Encode a task-args blob from Python, for tests that exercise a receive path directly.

Production never encodes one here: `write_blob` in `task_args.h` is the only writer, and the bound
wire types expose their fields but not their bytes, which is what keeps `validate_tensor` the single
gate on the way in. A test that wants to hand `ImportRegistry` a blob therefore has to lay the bytes
out itself.

That is a feature rather than a workaround. These formats are written from the `static_assert`s in
`src/common/task_interface/buffer.h` and the layout comment in `task_args.h`, independently of the
C++ that reads them, so a layout change that updates only one side fails here instead of passing by
construction — which is exactly what a `pack()` on the bound type could not do.
"""

from __future__ import annotations

import struct

# CanonicalIdentity, 32 B: owner_instance_id[8], buffer_id u64 @8, generation u32 @16, _pad[12] @20.
_IDENTITY = struct.Struct("<8sQI12x")

# BufferDescriptor, 88 B: magic u16 @0, address_space @2, access @3, backend_kind @4, _pad0[3],
# identity @8, nbytes u64 @40, owner_worker_path_id u32 @48, body_len u16 @52, _pad1[2], body[32] @56.
_DESC_HEAD = struct.Struct("<HBBB3x")
_DESC_TAIL = struct.Struct("<QIH2x32s")

# Tensor, 144 B: buffer @0, byte_offset u64 @88, ndims u32 @96, shapes[5] @100, strides[5] @120,
# dtype u8 @140, transfer u8 @141, _pad[2].
_TENSOR_TAIL = struct.Struct("<QI5I5IBB2x")

# Leading sentinel of a BufferDescriptor, frozen by buffer.h and deliberately not exported to Python
# (no production caller needs it). Pinned here so a change to it fails a test rather than only a
# static_assert.
_BUFFER_DESCRIPTOR_MAGIC = 0x5342

_MAX_TENSOR_DIMS = 5


def encode_tensor(t) -> bytes:
    """The 144 wire bytes of a bound `Tensor`, rebuilt from its accessors."""
    d = t.buffer
    body = bytes(d.body)
    head = _DESC_HEAD.pack(_BUFFER_DESCRIPTOR_MAGIC, int(d.address_space), int(d.access), int(d.backend_kind))
    identity = _IDENTITY.pack(d.identity.owner_instance_id, d.identity.buffer_id, d.identity.generation)
    tail = _DESC_TAIL.pack(d.nbytes, d.owner_worker_path_id, len(body), body)

    ndims = t.ndims
    shapes = list(t.shapes) + [0] * (_MAX_TENSOR_DIMS - ndims)
    strides = list(t.strides) + [0] * (_MAX_TENSOR_DIMS - ndims)
    view = _TENSOR_TAIL.pack(t.byte_offset, ndims, *shapes, *strides, int(t.dtype), int(t.transfer))
    return head + identity + tail + view


def encode_blob(tensors=(), scalars=()) -> bytes:
    """A task-args blob in `write_blob` format: [i32 count][i32 count][Tensor...][u64 scalar...]."""
    return (
        struct.pack("=ii", len(tensors), len(scalars))
        + b"".join(encode_tensor(t) for t in tensors)
        + struct.pack(f"<{len(scalars)}Q", *scalars)
    )
