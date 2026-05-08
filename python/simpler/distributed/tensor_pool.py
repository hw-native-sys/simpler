"""Tensor byte pool used by distributed dispatch tensor references.

This is the Python MVP of the data-plane bridge described in
``L4_L3_data_plane_design.md``.  It is intentionally a byte pool rather than an
RDMA implementation: handles, leases, alloc/free, chunked pull/push, and inline
vs handle decisions are represented explicitly so the backend can later swap the
storage implementation for SHM/RDMA/Urma without changing the control protocol.
"""

from __future__ import annotations

import ctypes
import itertools
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import grpc

from .proto import dispatch_pb2, dispatch_pb2_grpc
from .transport_backend import GrpcTensorTransport, RegisteredRegion, TensorTransportBackend

DEFAULT_INLINE_THRESHOLD = 4 * 1024
DEFAULT_POOL_CAPACITY = 64 * 1024 * 1024
DEFAULT_LEASE_TTL_MS = 60_000


class TensorPoolError(RuntimeError):
    pass


class TensorPoolFull(TensorPoolError):
    pass


@dataclass
class _Entry:
    data: bytearray
    nbytes: int
    expires_at_ms: int
    shape: tuple[int, ...]
    dtype: int
    tag: int
    region: RegisteredRegion


class TensorPool:
    def __init__(
        self,
        *,
        node_id: Optional[str] = None,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
        capacity_bytes: int = DEFAULT_POOL_CAPACITY,
        default_ttl_ms: int = DEFAULT_LEASE_TTL_MS,
        transport_backend: Optional[TensorTransportBackend] = None,
    ) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.inline_threshold = int(inline_threshold)
        self.capacity_bytes = int(capacity_bytes)
        self.default_ttl_ms = int(default_ttl_ms)
        self.transport_backend = transport_backend or GrpcTensorTransport()
        self._next_id = itertools.count(1)
        self._entries: dict[int, _Entry] = {}
        self._used_bytes = 0

    @property
    def used_bytes(self) -> int:
        self.gc_expired()
        return self._used_bytes

    def alloc(
        self,
        nbytes: int,
        *,
        ttl_ms: Optional[int] = None,
        shape: Iterable[int] = (),
        dtype: int = 0,
        tag: int = 0,
    ) -> dispatch_pb2.TensorHandle:
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"nbytes must be non-negative, got {nbytes}")
        self.gc_expired()
        if self._used_bytes + nbytes > self.capacity_bytes:
            raise TensorPoolFull(
                f"tensor pool {self.node_id} is full: requested={nbytes}, "
                f"used={self._used_bytes}, capacity={self.capacity_bytes}"
            )
        handle_id = next(self._next_id)
        ttl = self.default_ttl_ms if ttl_ms is None or int(ttl_ms) == 0 else int(ttl_ms)
        data = bytearray(nbytes)
        region = self.transport_backend.register_region(data, tag=f"{self.node_id}:{handle_id}:{int(tag)}")
        entry = _Entry(
            data=data,
            nbytes=nbytes,
            expires_at_ms=_now_ms() + ttl,
            shape=tuple(int(x) for x in shape),
            dtype=int(dtype),
            tag=int(tag),
            region=region,
        )
        self._entries[handle_id] = entry
        self._used_bytes += nbytes
        return self._make_handle(handle_id, entry)

    def free(self, handle: dispatch_pb2.TensorHandle) -> None:
        handle_id = self._checked_handle_id(handle)
        entry = self._entries.pop(handle_id)
        self._used_bytes -= entry.nbytes
        self.transport_backend.unregister_region(entry.region)

    def refresh(self, handle: dispatch_pb2.TensorHandle, ttl_ms: Optional[int] = None) -> dispatch_pb2.TensorHandle:
        handle_id = self._checked_handle_id(handle)
        entry = self._entries[handle_id]
        ttl = self.default_ttl_ms if ttl_ms is None or int(ttl_ms) == 0 else int(ttl_ms)
        entry.expires_at_ms = _now_ms() + ttl
        refresh_region = getattr(self.transport_backend, "refresh_region", None)
        if refresh_region is not None:
            entry.region = refresh_region(entry.region, entry.data, tag=f"{self.node_id}:{handle_id}:{entry.tag}")
        return self._make_handle(handle_id, entry)

    def write_bytes(self, handle: dispatch_pb2.TensorHandle, data: bytes, *, offset: int = 0) -> None:
        handle_id = self._checked_handle_id(handle)
        entry = self._entries[handle_id]
        offset = int(offset)
        data = bytes(data)
        end = offset + len(data)
        if offset < 0 or end > entry.nbytes:
            raise ValueError(f"write out of range: offset={offset}, size={len(data)}, nbytes={entry.nbytes}")
        entry.data[offset:end] = data

    def read_bytes(self, handle: dispatch_pb2.TensorHandle, *, offset: int = 0, nbytes: Optional[int] = None) -> bytes:
        handle_id = self._checked_handle_id(handle)
        entry = self._entries[handle_id]
        offset = int(offset)
        size = entry.nbytes - offset if nbytes is None else int(nbytes)
        end = offset + size
        if offset < 0 or size < 0 or end > entry.nbytes:
            raise ValueError(f"read out of range: offset={offset}, size={size}, nbytes={entry.nbytes}")
        return bytes(entry.data[offset:end])

    def put_bytes(
        self,
        data: bytes,
        *,
        shape: Iterable[int] = (),
        dtype: int = 0,
        tag: int = 0,
        ttl_ms: Optional[int] = None,
        force_handle: bool = False,
    ) -> dispatch_pb2.TensorRef:
        data = bytes(data)
        ref = dispatch_pb2.TensorRef(shape=[int(x) for x in shape], dtype=int(dtype), tag=int(tag))
        if not force_handle and len(data) <= self.inline_threshold:
            ref.inline_data = data
            return ref
        handle = self.alloc(len(data), ttl_ms=ttl_ms, shape=shape, dtype=dtype, tag=tag)
        self.write_bytes(handle, data)
        ref.handle.CopyFrom(handle)
        return ref

    def get_bytes(self, handle: dispatch_pb2.TensorHandle) -> bytes:
        return self.read_bytes(handle)

    def materialize_ref(self, ref: dispatch_pb2.TensorRef) -> bytes:
        if ref.HasField("inline_data"):
            return bytes(ref.inline_data)
        if ref.HasField("handle"):
            return self.get_bytes(ref.handle)
        raise TensorPoolError("TensorRef has neither inline_data nor handle")

    def gc_expired(self) -> int:
        now = _now_ms()
        expired = [handle_id for handle_id, entry in self._entries.items() if entry.expires_at_ms <= now]
        for handle_id in expired:
            entry = self._entries.pop(handle_id)
            self._used_bytes -= entry.nbytes
            self.transport_backend.unregister_region(entry.region)
        return len(expired)

    def close(self) -> None:
        for handle_id in list(self._entries):
            entry = self._entries.pop(handle_id)
            self._used_bytes -= entry.nbytes
            self.transport_backend.unregister_region(entry.region)
        close = getattr(self.transport_backend, "close", None)
        if close is not None:
            close()

    def service(self) -> "TensorPoolService":
        return TensorPoolService(self)

    def _checked_handle_id(self, handle: dispatch_pb2.TensorHandle) -> int:
        self.gc_expired()
        if handle.node_id != self.node_id:
            raise KeyError(f"tensor handle belongs to node {handle.node_id!r}, not {self.node_id!r}")
        handle_id = int(handle.handle_id)
        if handle_id not in self._entries:
            raise KeyError(f"tensor handle {handle_id} is not allocated")
        return handle_id

    def _make_handle(self, handle_id: int, entry: _Entry) -> dispatch_pb2.TensorHandle:
        return dispatch_pb2.TensorHandle(
            node_id=self.node_id,
            handle_id=int(handle_id),
            remote_addr=int(entry.region.remote_addr),
            rkey=int(entry.region.rkey),
            nbytes=entry.nbytes,
            lease_deadline_unix_ms=entry.expires_at_ms,
            transport=entry.region.transport,
            transport_desc=entry.region.transport_desc,
        )


class TensorPoolService(dispatch_pb2_grpc.TensorPoolServicer):
    def __init__(self, pool: TensorPool, *, chunk_size: int = 1024 * 1024) -> None:
        self._pool = pool
        self._chunk_size = int(chunk_size)

    def AllocTensor(self, request, context):  # noqa: N802, ANN001
        try:
            return self._pool.alloc(
                request.nbytes,
                ttl_ms=request.ttl_ms,
                shape=request.shape,
                dtype=request.dtype,
                tag=request.tag,
            )
        except Exception as e:  # noqa: BLE001
            _abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))

    def FreeTensor(self, request, context):  # noqa: N802, ANN001
        try:
            self._pool.free(request.handle)
        except Exception as e:  # noqa: BLE001
            _abort(context, grpc.StatusCode.NOT_FOUND, str(e))
        return dispatch_pb2.Empty()

    def RefreshTensor(self, request, context):  # noqa: N802, ANN001
        try:
            return self._pool.refresh(request.handle, request.ttl_ms)
        except Exception as e:  # noqa: BLE001
            _abort(context, grpc.StatusCode.NOT_FOUND, str(e))

    def PullTensor(self, request, context):  # noqa: N802, ANN001
        try:
            data = self._pool.get_bytes(request)
        except Exception as e:  # noqa: BLE001
            _abort(context, grpc.StatusCode.NOT_FOUND, str(e))
        for offset in range(0, len(data), self._chunk_size):
            chunk = data[offset : offset + self._chunk_size]
            yield dispatch_pb2.TensorChunk(
                handle=request,
                offset=offset,
                data=chunk,
                last=offset + len(chunk) >= len(data),
            )
        if not data:
            yield dispatch_pb2.TensorChunk(handle=request, offset=0, data=b"", last=True)

    def PushTensor(self, request_iterator: Iterable[dispatch_pb2.TensorChunk], context):  # noqa: N802, ANN001
        chunks = list(request_iterator)
        if not chunks:
            return self._pool.alloc(0)
        handle = chunks[0].handle
        payload = _join_chunks(chunks)
        try:
            if handle.handle_id:
                self._pool.write_bytes(handle, payload)
                return self._pool.refresh(handle)
            ref = self._pool.put_bytes(payload, force_handle=True)
            return ref.handle
        except Exception as e:  # noqa: BLE001
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(e))


def _join_chunks(chunks: list[dispatch_pb2.TensorChunk]) -> bytes:
    total = 0
    for chunk in chunks:
        total = max(total, int(chunk.offset) + len(chunk.data))
    out = bytearray(total)
    for chunk in chunks:
        offset = int(chunk.offset)
        out[offset : offset + len(chunk.data)] = chunk.data
    return bytes(out)


def _buffer_addr(data: bytearray) -> int:
    if not data:
        return 0
    return ctypes.addressof(ctypes.c_char.from_buffer(data))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _abort(context, code: grpc.StatusCode, message: str):  # noqa: ANN001
    if context is None:
        raise TensorPoolError(message)
    context.abort(code, message)
