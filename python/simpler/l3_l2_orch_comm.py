# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-L2 orchestrator communication facade."""

from __future__ import annotations

import ctypes
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from _task_interface import _mailbox_load_i32, _mailbox_store_i32  # pyright: ignore[reportMissingImports]


class L3L2OrchCommCmd(IntEnum):
    ALLOC_REGION = 1
    FREE_REGION = 2
    PAYLOAD_WRITE = 3
    PAYLOAD_READ = 4
    SIGNAL_NOTIFY = 5
    SIGNAL_WAIT = 6


class L3L2OrchCommSignalSlot(IntEnum):
    L3_TO_L2 = 0
    L2_TO_L3 = 1


_STATE_IDLE = 0
_STATE_READY = 1
_STATE_DONE = 3
_POLL_INTERVAL_S = 0.00005
_DEFAULT_SUBMIT_TIMEOUT_S = 5.0
_SIGNAL_TIMEOUT_MARGIN_S = 1.0

_REQUEST = struct.Struct("<IIQQQQQQQ")
_DESC = struct.Struct("<6Q")
_RESPONSE = struct.Struct("<iIQQ6Q256s")
_CONTROL_OFF_STATE = 0
_CONTROL_OFF_REQUEST = 8
_CONTROL_OFF_RESPONSE = _CONTROL_OFF_REQUEST + _REQUEST.size
CONTROL_BLOCK_SIZE = _CONTROL_OFF_RESPONSE + _RESPONSE.size
CONTROL_SHM_SIZE = max(4096, CONTROL_BLOCK_SIZE)
L3L2_ORCH_REGION_DESC_SCALAR_COUNT = 6


@dataclass(frozen=True)
class L3L2OrchRegionDesc:
    magic_version: int
    region_id: int
    payload_base: int
    payload_bytes: int
    l3_to_l2_signal_base: int
    l2_to_l3_signal_base: int

    def scalars(self) -> list[int]:
        return [
            int(self.magic_version),
            int(self.region_id),
            int(self.payload_base),
            int(self.payload_bytes),
            int(self.l3_to_l2_signal_base),
            int(self.l2_to_l3_signal_base),
        ]


@dataclass(frozen=True)
class L3L2OrchCommRequest:
    cmd: L3L2OrchCommCmd
    region_id: int = 0
    offset: int = 0
    host_ptr: int = 0
    nbytes: int = 0
    signal_slot: int = 0
    seq: int = 0
    timeout_ns: int = 0


@dataclass(frozen=True)
class L3L2OrchCommResponse:
    status: int
    error_kind: int
    region_id: int
    observed_signal: int
    desc: L3L2OrchRegionDesc | None
    message: str


class L3L2OrchCommClient:
    def __init__(self, shm: SharedMemory) -> None:
        self._shm = shm
        self._buf = shm.buf
        self._state_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf)) + _CONTROL_OFF_STATE
        self._mu = threading.Lock()
        _mailbox_store_i32(self._state_addr, _STATE_IDLE)

    def submit(self, request: L3L2OrchCommRequest, timeout_s: float = _DEFAULT_SUBMIT_TIMEOUT_S) -> L3L2OrchCommResponse:
        deadline = time.monotonic() + float(timeout_s)
        with self._mu:
            while _mailbox_load_i32(self._state_addr) != _STATE_IDLE:
                if time.monotonic() >= deadline:
                    raise TimeoutError("L3-L2 orch comm client timed out waiting for IDLE")
                time.sleep(_POLL_INTERVAL_S)

            _REQUEST.pack_into(
                self._buf,
                _CONTROL_OFF_REQUEST,
                int(request.cmd),
                0,
                int(request.region_id),
                int(request.offset),
                int(request.host_ptr),
                int(request.nbytes),
                int(request.signal_slot),
                int(request.seq),
                int(request.timeout_ns),
            )
            self._buf[_CONTROL_OFF_RESPONSE : _CONTROL_OFF_RESPONSE + _RESPONSE.size] = b"\x00" * _RESPONSE.size
            _mailbox_store_i32(self._state_addr, _STATE_READY)

            while _mailbox_load_i32(self._state_addr) != _STATE_DONE:
                if time.monotonic() >= deadline:
                    raise TimeoutError("L3-L2 orch comm client timed out waiting for DONE")
                time.sleep(_POLL_INTERVAL_S)

            response = self._read_response()
            _mailbox_store_i32(self._state_addr, _STATE_IDLE)
            return response

    def _read_response(self) -> L3L2OrchCommResponse:
        fields = _RESPONSE.unpack_from(self._buf, _CONTROL_OFF_RESPONSE)
        status, error_kind, region_id, observed_signal = fields[:4]
        desc_values = fields[4:10]
        raw_message = fields[10]
        desc = None
        if any(int(v) != 0 for v in desc_values):
            desc = L3L2OrchRegionDesc(*[int(v) for v in desc_values])
        message = raw_message.split(b"\x00", 1)[0].decode("utf-8", "replace")
        return L3L2OrchCommResponse(
            status=int(status),
            error_kind=int(error_kind),
            region_id=int(region_id),
            observed_signal=int(observed_signal),
            desc=desc,
            message=message,
        )


class _PinnedBuffer:
    def __init__(self, obj: Any, owner: Any) -> None:
        from .task_interface import ContinuousTensor  # noqa: PLC0415

        if not isinstance(obj, ContinuousTensor):
            raise ValueError("L3-L2 payload buffer must be a ContinuousTensor returned by orch.alloc(...)")
        owner._validate_l3_l2_orch_comm_host_buffer(obj)
        self.addr = int(obj.data)
        self.nbytes = int(obj.nbytes())

    def close(self) -> None:
        return None

    def __enter__(self) -> "_PinnedBuffer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class L3L2OrchRegion:
    def __init__(self, owner: Any, worker_id: int, desc: L3L2OrchRegionDesc, payload_bytes: int) -> None:
        self._owner = owner
        self._worker_id = int(worker_id)
        self._descriptor = desc
        self._payload_bytes = int(payload_bytes)
        self._released = False
        self._poisoned = False
        self._expired = False

    @property
    def region_id(self) -> int:
        return int(self._descriptor.region_id)

    def descriptor_scalars(self) -> list[int]:
        self._ensure_live()
        return self._descriptor.scalars()

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        with _PinnedBuffer(host_buffer, self._owner) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            self._submit(
                L3L2OrchCommRequest(
                    cmd=L3L2OrchCommCmd.PAYLOAD_WRITE,
                    region_id=self.region_id,
                    offset=int(offset),
                    host_ptr=pinned.addr,
                    nbytes=size,
                )
            )

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        with _PinnedBuffer(host_buffer, self._owner) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            self._submit(
                L3L2OrchCommRequest(
                    cmd=L3L2OrchCommCmd.PAYLOAD_READ,
                    region_id=self.region_id,
                    offset=int(offset),
                    host_ptr=pinned.addr,
                    nbytes=size,
                )
            )

    def notify(self, seq: int) -> None:
        self._ensure_live()
        seq = int(seq)
        if seq <= 0:
            raise ValueError("L3-L2 notify sequence must be positive")
        self._submit(
            L3L2OrchCommRequest(
                cmd=L3L2OrchCommCmd.SIGNAL_NOTIFY,
                region_id=self.region_id,
                signal_slot=int(L3L2OrchCommSignalSlot.L3_TO_L2),
                seq=seq,
            )
        )

    def wait(self, seq: int, timeout: float) -> None:
        self._ensure_live()
        seq = int(seq)
        if seq <= 0:
            raise ValueError("L3-L2 wait sequence must be positive")
        if timeout is None or float(timeout) <= 0:
            raise ValueError("L3-L2 wait requires a positive timeout")
        timeout_ns = int(float(timeout) * 1_000_000_000)
        self._submit(
            L3L2OrchCommRequest(
                cmd=L3L2OrchCommCmd.SIGNAL_WAIT,
                region_id=self.region_id,
                signal_slot=int(L3L2OrchCommSignalSlot.L2_TO_L3),
                seq=seq,
                timeout_ns=timeout_ns,
            ),
            timeout_s=float(timeout) + _SIGNAL_TIMEOUT_MARGIN_S,
        )

    def free(self) -> None:
        if self._released:
            return
        self._released = True

    def _expire(self) -> None:
        self._expired = True

    def _poison(self) -> None:
        self._poisoned = True

    def _ensure_live(self) -> None:
        if self._expired:
            raise RuntimeError(f"L3-L2 region {self.region_id} expired after orchestration run")
        if self._released:
            raise RuntimeError(f"L3-L2 region {self.region_id} has been released")
        if self._poisoned:
            raise RuntimeError(f"L3-L2 region {self.region_id} is poisoned")

    def _validate_payload_range(self, offset: int, nbytes: int, buffer_nbytes: int) -> None:
        offset = int(offset)
        nbytes = int(nbytes)
        if offset < 0 or nbytes <= 0:
            raise ValueError("L3-L2 payload offset must be non-negative and nbytes must be positive")
        if nbytes > int(buffer_nbytes):
            raise ValueError(f"L3-L2 payload nbytes={nbytes} exceeds host buffer size {buffer_nbytes}")
        if offset + nbytes > self._payload_bytes:
            raise ValueError(
                f"L3-L2 payload range [{offset}, {offset + nbytes}) exceeds region size {self._payload_bytes}"
            )

    def _submit(self, request: L3L2OrchCommRequest, timeout_s: float = _DEFAULT_SUBMIT_TIMEOUT_S) -> L3L2OrchCommResponse:
        try:
            response = self._owner._l3_l2_orch_comm_submit(self._worker_id, request, timeout_s)
        except Exception:
            self._poison()
            raise
        if response.status != 0:
            self._poison()
            msg = response.message or f"L3-L2 orch comm command {int(request.cmd)} failed"
            raise RuntimeError(msg)
        return response
