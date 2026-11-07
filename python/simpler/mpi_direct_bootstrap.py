# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded byte-preserving bootstrap frames for the authenticated startup gate."""

import base64
import hashlib
import json
import struct
import time

MAX_STARTUP_BYTES = 256 * 1024 * 1024
_HEADER = struct.Struct("!Q32s")
_CHUNK_BYTES = 64 * 1024


def _pack(value):
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return ["dict", {key: _pack(item) for key, item in value.items()}]
    if isinstance(value, (tuple, list)):
        return ["list", [_pack(item) for item in value]]
    if value is None or type(value) in (str, int, float, bool):
        return ["value", value]
    raise ValueError("startup records require JSON values and bytes")


def _unpack(value):
    tag, item = value
    if tag == "bytes":
        return base64.b64decode(item, validate=True)
    if tag == "dict":
        return {key: _unpack(child) for key, child in item.items()}
    if tag == "list":
        return [_unpack(child) for child in item]
    if tag == "value":
        return item
    raise ValueError("invalid startup record type")


def encode_startup(record):
    if not isinstance(record, dict):
        raise ValueError("startup record must be an object")
    payload = json.dumps(_pack(record), allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_STARTUP_BYTES:
        raise ValueError("startup record length exceeds limit")
    return payload


def _timeout(sock, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("startup transfer deadline exceeded")
    sock.settimeout(remaining)


def send_startup(sock, payload, deadline):
    if not 0 < len(payload) <= MAX_STARTUP_BYTES:
        raise ValueError("invalid startup record length")
    _timeout(sock, deadline)
    sock.sendall(_HEADER.pack(len(payload), hashlib.sha256(payload).digest()))
    for offset in range(0, len(payload), _CHUNK_BYTES):
        _timeout(sock, deadline)
        sock.sendall(memoryview(payload)[offset : offset + _CHUNK_BYTES])


def receive_startup(sock, deadline):
    def read(size):
        data = bytearray()
        while len(data) < size:
            _timeout(sock, deadline)
            chunk = sock.recv(min(size - len(data), _CHUNK_BYTES))
            if not chunk:
                raise ConnectionError("startup transfer peer closed")
            data.extend(chunk)
        return data

    length, digest = _HEADER.unpack(read(_HEADER.size))
    if not 0 < length <= MAX_STARTUP_BYTES:
        raise ValueError("invalid startup record length")
    payload = read(length)
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("startup record digest mismatch")
    return decode_startup(payload)


def decode_startup(payload):
    """Read the same safe record format from a local launcher handoff file."""
    if not 0 < len(payload) <= MAX_STARTUP_BYTES:
        raise ValueError("invalid startup record length")
    result = _unpack(json.loads(payload))
    if not isinstance(result, dict):
        raise ValueError("startup record must be an object")
    return result
