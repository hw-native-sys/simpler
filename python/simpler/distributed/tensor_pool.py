"""Tensor byte pool used by distributed dispatch tensor references."""

from __future__ import annotations

import itertools
import uuid
from collections.abc import Iterable
from typing import Optional

import grpc

from .proto import dispatch_pb2, dispatch_pb2_grpc


class TensorPool:
    def __init__(self, *, node_id: Optional[str] = None, inline_threshold: int = 1024 * 1024) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.inline_threshold = int(inline_threshold)
        self._next_id = itertools.count(1)
        self._data: dict[int, bytes] = {}

    def put_bytes(self, data: bytes) -> dispatch_pb2.TensorRef:
        data = bytes(data)
        if len(data) <= self.inline_threshold:
            return dispatch_pb2.TensorRef(inline_data=data)
        handle_id = next(self._next_id)
        self._data[handle_id] = data
        return dispatch_pb2.TensorRef(handle=dispatch_pb2.TensorHandle(node_id=self.node_id, handle_id=handle_id))

    def get_bytes(self, handle: dispatch_pb2.TensorHandle) -> bytes:
        if handle.node_id != self.node_id:
            raise KeyError(f"tensor handle belongs to node {handle.node_id!r}, not {self.node_id!r}")
        return self._data[int(handle.handle_id)]

    def service(self) -> "TensorPoolService":
        return TensorPoolService(self)


class TensorPoolService(dispatch_pb2_grpc.TensorPoolServicer):
    def __init__(self, pool: TensorPool, *, chunk_size: int = 1024 * 1024) -> None:
        self._pool = pool
        self._chunk_size = int(chunk_size)

    def PullTensor(self, request, context):  # noqa: N802, ANN001
        try:
            data = self._pool.get_bytes(request)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
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
        parts = []
        for chunk in request_iterator:
            parts.append(bytes(chunk.data))
        ref = self._pool.put_bytes(b"".join(parts))
        if not ref.HasField("handle"):
            handle_id = next(self._pool._next_id)
            self._pool._data[handle_id] = ref.inline_data
            return dispatch_pb2.TensorHandle(node_id=self._pool.node_id, handle_id=handle_id)
        return ref.handle
