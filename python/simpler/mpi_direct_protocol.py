# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed MPI tag lanes for the direct-MPI transport."""

import enum
import json
import socket
import struct
import time

MPI_DIRECT_STARTUP_TOKEN_ENV = "SIMPLER_MPI_DIRECT_STARTUP_TOKEN"
MPI_DIRECT_GATE_MAX_BYTES = 64 * 1024


class MpiDirectTag(enum.IntEnum):
    COMMAND_REQUEST = 1
    COMMAND_REPLY = 2
    HEALTH = 3
    LIFECYCLE = 4


def _assert_native_wire_constants() -> None:
    import _task_interface as native  # noqa: PLC0415

    from .remote_l3_limits import FRAME_HEADER_BYTES, MAX_FRAME_PAYLOAD_BYTES  # noqa: PLC0415

    for tag in MpiDirectTag:
        if int(tag) != getattr(native._MpiDirectTag, tag.name).value:
            raise RuntimeError(f"MPI direct wire tag mismatch: {tag.name}")
    if (
        FRAME_HEADER_BYTES != native._REMOTE_FRAME_HEADER_BYTES
        or MAX_FRAME_PAYLOAD_BYTES != native._REMOTE_MAX_FRAME_PAYLOAD_BYTES
    ):
        raise RuntimeError("MPI direct wire frame limits mismatch")


def _gate_send(sock: socket.socket, message: dict[str, object]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MPI_DIRECT_GATE_MAX_BYTES:
        raise ValueError("MPI startup gate message is too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _gate_recv(sock: socket.socket, deadline: float) -> dict[str, object]:
    def read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MPI startup gate read deadline exceeded")
            sock.settimeout(remaining)
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("MPI startup gate peer closed")
            data.extend(chunk)
        return bytes(data)

    length = struct.unpack("!I", read_exact(4))[0]
    if length <= 0 or length > MPI_DIRECT_GATE_MAX_BYTES:
        raise ValueError("invalid MPI startup gate message length")
    message = json.loads(read_exact(length))
    if not isinstance(message, dict):
        raise ValueError("MPI startup gate message must be an object")
    return message
