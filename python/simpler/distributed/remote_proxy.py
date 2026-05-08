"""L4-side proxy for a remote L3 worker."""

from __future__ import annotations

import ctypes
import itertools
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import grpc

from simpler.task_interface import CallConfig, TaskArgs

from .catalog import Catalog
from .proto import dispatch_pb2
from .rpc import RpcClient, RpcError
from .serialization import encode_config, encode_task_args
from .tensor_pool import DEFAULT_INLINE_THRESHOLD
from .transport_backend import (
    HcommDataPlaneClient,
    RxeDataPlaneClient,
    RxeRuntime,
    TransportBackendError,
    TransportUnavailable,
    _encode_rxe_desc,
)


class RemoteUnavailable(RuntimeError):
    pass


@dataclass
class _LocalOutputRegion:
    handle: dispatch_pb2.TensorHandle
    runtime: RxeRuntime
    server_handle: int

    def close(self) -> None:
        self.runtime.server_stop(self.server_handle)
        self.server_handle = 0


class RemoteWorkerProxy:
    """Synchronous L4-side stub for one remote L3 worker."""

    def __init__(
        self,
        endpoint: str,
        l4_catalog: Catalog,
        *,
        timeout: float = 10.0,
        heartbeat_timeout: float = 1.0,
        heartbeat_interval: float = 5.0,
        heartbeat_failures: int = 3,
        tensor_inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
        tensor_chunk_size: int = 1024 * 1024,
        tensor_transport: Optional[str] = None,
        hcomm_client: Optional[HcommDataPlaneClient] = None,
        rxe_client: Optional[RxeDataPlaneClient] = None,
    ) -> None:
        self.endpoint = endpoint
        self._client = RpcClient(endpoint)
        self._catalog = l4_catalog
        self._timeout = float(timeout)
        self._heartbeat_timeout = float(heartbeat_timeout)
        self._heartbeat_interval = float(heartbeat_interval)
        self._heartbeat_failures = int(heartbeat_failures)
        self._tensor_inline_threshold = int(tensor_inline_threshold)
        self._tensor_chunk_size = int(tensor_chunk_size)
        self._tensor_transport = (tensor_transport or os.getenv("SIMPLER_TENSOR_TRANSPORT", "grpc")).lower()
        self._hcomm_client = hcomm_client
        self._rxe_client = rxe_client
        self._local_node_id = f"l4-rxe-{os.getpid()}-{uuid.uuid4().hex}"
        self._task_ids = itertools.count(1)
        self._available = True
        self._closed = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def handshake(self) -> None:
        self._check_heartbeat()
        for payload in self._catalog.payloads():
            self._client.call_unary("Catalog.PushCallable", payload, self._timeout)
        self._start_heartbeat()

    def heartbeat(self) -> None:
        self._check_heartbeat()

    def _check_heartbeat(self) -> None:
        try:
            health = self._heartbeat_rpc()
        except RemoteUnavailable:
            self._available = False
            raise
        if not health.ok:
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} unhealthy: {health.message}")
        self._available = True

    def _heartbeat_rpc(self):
        try:
            return self._client.heartbeat(self._heartbeat_timeout)
        except RpcError as e:
            raise RemoteUnavailable(f"remote {self.endpoint} heartbeat failed: {e}") from e

    def _start_heartbeat(self) -> None:
        if self._heartbeat_interval <= 0 or self._heartbeat_thread is not None:
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"simpler-remote-heartbeat-{self.endpoint}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        failures = 0
        while not self._closed.wait(self._heartbeat_interval):
            try:
                health = self._heartbeat_rpc()
                if not health.ok:
                    raise RemoteUnavailable(f"remote {self.endpoint} unhealthy: {health.message}")
                failures = 0
                self._available = True
            except RemoteUnavailable:
                failures += 1
                if failures >= self._heartbeat_failures:
                    self._available = False

    def dispatch(self, callable_id: int, args: Optional[TaskArgs], cfg: Optional[CallConfig]) -> None:
        if not self._available:
            raise RemoteUnavailable(f"remote {self.endpoint} is unavailable")
        config = cfg if cfg is not None else CallConfig()
        tensor_args, scalar_args = encode_task_args(args)
        tensor_refs, remote_handles, local_output_regions = self._stage_tensor_args(args)
        version = self._catalog.refs_by_id().get(int(callable_id), 0)
        req = dispatch_pb2.DispatchReq(
            task_id=next(self._task_ids),
            callable_id=int(callable_id),
            callable_version=int(version),
            config_blob=encode_config(config),
            scalar_args=scalar_args,
            tensor_args=[] if tensor_refs else tensor_args,
            tensor_refs=tensor_refs,
        )
        try:
            resp = self._client.dispatch(req, self._timeout)
        except RpcError as e:
            self._available = False
            self._free_remote_handles(remote_handles)
            self._close_local_output_regions(local_output_regions)
            raise RemoteUnavailable(f"remote {self.endpoint} dispatch RPC failed: {e}") from e
        if resp.error_code != 0:
            self._free_remote_handles(remote_handles)
            self._close_local_output_regions(local_output_regions)
            detail = resp.error_msg
            if resp.remote_traceback:
                detail = detail + "\nremote traceback:\n" + "\n".join(resp.remote_traceback)
            raise RuntimeError(f"remote dispatch failed on {self.endpoint}: {detail}")
        try:
            self._write_output_tensors(args, resp.output_tensors)
        finally:
            self._free_remote_handles(remote_handles)
            self._free_response_handles(resp.output_tensors)
            self._close_local_output_regions(local_output_regions)

    def _stage_tensor_args(
        self,
        args: Optional[TaskArgs],
    ) -> tuple[list[dispatch_pb2.TensorRef], list[dispatch_pb2.TensorHandle], list[_LocalOutputRegion]]:
        if args is None or args.tensor_count() == 0:
            return [], [], []
        refs = []
        remote_handles = []
        local_output_regions = []
        try:
            for i in range(args.tensor_count()):
                tensor = args.tensor(i)
                tag = args.tag(i)
                nbytes = _tensor_nbytes(tensor)
                shape = [int(x) for x in tensor.shapes[: int(tensor.ndims)]]
                dtype = int(tensor.dtype.value)
                tag_value = int(tag.value)
                if self._should_stage_local_output(tag, nbytes):
                    try:
                        ref, region = self._stage_local_output_tensor(tensor, nbytes, shape, dtype, tag_value)
                    except (TransportBackendError, TransportUnavailable):
                        if self._tensor_transport != "auto":
                            raise
                    else:
                        refs.append(ref)
                        local_output_regions.append(region)
                        continue
                data = ctypes.string_at(int(tensor.data), nbytes) if nbytes else b""
                if nbytes <= self._tensor_inline_threshold:
                    refs.append(
                        dispatch_pb2.TensorRef(
                            inline_data=data,
                            shape=shape,
                            dtype=dtype,
                            tag=tag_value,
                        )
                    )
                    continue
                handle = self._alloc_remote_tensor(nbytes, shape, dtype, tag_value)
                remote_handles.append(handle)
                self._push_remote_tensor(handle, data)
                refs.append(
                    dispatch_pb2.TensorRef(
                        handle=handle,
                        shape=shape,
                        dtype=dtype,
                        tag=tag_value,
                    )
                )
        except Exception:
            self._free_remote_handles(remote_handles)
            self._close_local_output_regions(local_output_regions)
            raise
        return refs, remote_handles, local_output_regions

    def _should_stage_local_output(self, tag, nbytes: int) -> bool:  # noqa: ANN001
        return (
            self._tensor_transport in {"rxe", "auto"}
            and getattr(tag, "name", "") in {"OUTPUT", "OUTPUT_EXISTING"}
            and int(nbytes) > self._tensor_inline_threshold
        )

    def _stage_local_output_tensor(
        self,
        tensor,  # noqa: ANN001
        nbytes: int,
        shape: list[int],
        dtype: int,
        tag: int,
    ) -> tuple[dispatch_pb2.TensorRef, _LocalOutputRegion]:
        runtime = RxeRuntime.from_env(required=True)
        desc, server_handle = runtime.server_start(int(tensor.data), int(nbytes))
        transport_desc = _encode_rxe_desc(desc, runtime.device or "", runtime.gid_index)
        handle = dispatch_pb2.TensorHandle(
            node_id=self._local_node_id,
            handle_id=0,
            remote_addr=int(desc.addr),
            rkey=int(desc.rkey),
            nbytes=int(nbytes),
            transport="rxe",
            transport_desc=transport_desc,
        )
        ref = dispatch_pb2.TensorRef(handle=handle, shape=shape, dtype=int(dtype), tag=int(tag))
        return ref, _LocalOutputRegion(handle=handle, runtime=runtime, server_handle=server_handle)

    def _alloc_remote_tensor(
        self,
        nbytes: int,
        shape: list[int],
        dtype: int,
        tag: int,
    ) -> dispatch_pb2.TensorHandle:
        req = dispatch_pb2.TensorAllocReq(nbytes=int(nbytes), shape=shape, dtype=int(dtype), tag=int(tag))
        try:
            return self._client.tensor_pool.AllocTensor(req, timeout=self._timeout)
        except grpc.RpcError as e:
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} tensor alloc failed: {e.details() or e}") from e

    def _push_remote_tensor(self, handle: dispatch_pb2.TensorHandle, data: bytes) -> None:
        if self._should_use_rxe(handle):
            self._push_remote_tensor_rxe(handle, data)
            return
        if self._should_use_hcomm(handle):
            self._push_remote_tensor_hcomm(handle, data)
            return
        self._push_remote_tensor_grpc(handle, data)

    def _should_use_rxe(self, handle: dispatch_pb2.TensorHandle) -> bool:
        return (
            self._tensor_transport in {"rxe", "auto"}
            and handle.transport == "rxe"
            and int(handle.nbytes) > 0
        )

    def _should_use_hcomm(self, handle: dispatch_pb2.TensorHandle) -> bool:
        return (
            self._tensor_transport in {"hcomm", "auto"}
            and handle.transport == "hcomm"
            and int(handle.nbytes) > 0
        )

    def _push_remote_tensor_rxe(self, handle: dispatch_pb2.TensorHandle, data: bytes) -> None:
        if len(data) != int(handle.nbytes):
            raise ValueError(f"RXE tensor push size mismatch: data={len(data)}, handle={handle.nbytes}")
        client = self._rxe_client or RxeDataPlaneClient.from_env()
        if self._rxe_client is None:
            self._rxe_client = client
        local = ctypes.create_string_buffer(data)
        local_addr = ctypes.addressof(local)
        try:
            client.write_handle(handle, local_addr, len(data))
            client.fence()
            self._client.tensor_pool.RefreshTensor(
                dispatch_pb2.TensorRefreshReq(handle=handle),
                timeout=self._timeout,
            )
        except (TransportBackendError, TransportUnavailable) as e:
            if self._tensor_transport == "auto":
                self._push_remote_tensor_grpc(handle, data)
                return
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} RXE tensor push unavailable: {e}") from e
        except grpc.RpcError as e:
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} tensor refresh failed: {e.details() or e}") from e

    def _push_remote_tensor_hcomm(self, handle: dispatch_pb2.TensorHandle, data: bytes) -> None:
        if len(data) != int(handle.nbytes):
            raise ValueError(f"HCOMM tensor push size mismatch: data={len(data)}, handle={handle.nbytes}")
        client = self._hcomm_client or HcommDataPlaneClient.from_env()
        if self._hcomm_client is None:
            self._hcomm_client = client
        local = ctypes.create_string_buffer(data)
        local_addr = ctypes.addressof(local)
        try:
            if hasattr(client, "write_handle"):
                client.write_handle(handle, local_addr, len(data))
            else:
                client.write_with_notify(int(handle.remote_addr), local_addr, len(data))
            client.fence()
            self._client.tensor_pool.RefreshTensor(
                dispatch_pb2.TensorRefreshReq(handle=handle),
                timeout=self._timeout,
            )
        except (TransportBackendError, TransportUnavailable) as e:
            if self._tensor_transport == "auto":
                self._push_remote_tensor_grpc(handle, data)
                return
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} HCOMM tensor push unavailable: {e}") from e
        except grpc.RpcError as e:
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} tensor refresh failed: {e.details() or e}") from e

    def _push_remote_tensor_grpc(self, handle: dispatch_pb2.TensorHandle, data: bytes) -> None:
        def chunks():
            if not data:
                yield dispatch_pb2.TensorChunk(handle=handle, offset=0, data=b"", last=True)
                return
            for offset in range(0, len(data), self._tensor_chunk_size):
                chunk = data[offset : offset + self._tensor_chunk_size]
                yield dispatch_pb2.TensorChunk(
                    handle=handle,
                    offset=offset,
                    data=chunk,
                    last=offset + len(chunk) >= len(data),
                )

        try:
            self._client.tensor_pool.PushTensor(chunks(), timeout=self._timeout)
        except grpc.RpcError as e:
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} tensor push failed: {e.details() or e}") from e

    def _free_remote_handles(self, handles: list[dispatch_pb2.TensorHandle]) -> None:
        for handle in handles:
            try:
                self._client.tensor_pool.FreeTensor(dispatch_pb2.TensorFreeReq(handle=handle), timeout=self._timeout)
            except grpc.RpcError:
                pass

    def _write_output_tensors(self, args: Optional[TaskArgs], refs) -> None:  # noqa: ANN001
        if args is None:
            return
        output_indexes = _output_tensor_indexes(args)
        if len(refs) != len(output_indexes):
            if refs:
                raise RuntimeError(
                    f"remote returned {len(refs)} output tensors for {len(output_indexes)} local output tensors"
                )
            return
        for ref, tensor_index in zip(refs, output_indexes):
            tensor = args.tensor(tensor_index)
            nbytes = _tensor_nbytes(tensor)
            if self._is_local_output_ack(ref, nbytes):
                continue
            data = self._read_tensor_ref(ref)
            if len(data) != nbytes:
                raise RuntimeError(
                    f"remote output tensor {tensor_index} has {len(data)} bytes, expected {nbytes}"
                )
            if nbytes:
                ctypes.memmove(int(tensor.data), data, nbytes)

    def _read_tensor_ref(self, ref: dispatch_pb2.TensorRef) -> bytes:
        if ref.HasField("inline_data"):
            return bytes(ref.inline_data)
        if ref.HasField("handle"):
            try:
                chunks = self._client.tensor_pool.PullTensor(ref.handle, timeout=self._timeout)
                return _join_chunks(chunks)
            except grpc.RpcError as e:
                self._available = False
                raise RemoteUnavailable(f"remote {self.endpoint} tensor pull failed: {e.details() or e}") from e
        raise RuntimeError("remote output tensor has neither inline_data nor handle")

    def _is_local_output_ack(self, ref: dispatch_pb2.TensorRef, nbytes: int) -> bool:
        return (
            ref.HasField("handle")
            and ref.handle.transport == "rxe"
            and ref.handle.node_id == self._local_node_id
            and int(ref.handle.nbytes) == int(nbytes)
        )

    def _free_response_handles(self, refs) -> None:  # noqa: ANN001
        handles = [
            ref.handle
            for ref in refs
            if ref.HasField("handle") and ref.handle.node_id != self._local_node_id
        ]
        self._free_remote_handles(handles)

    def _close_local_output_regions(self, regions: list[_LocalOutputRegion]) -> None:
        while regions:
            region = regions.pop()
            try:
                region.close()
            except Exception:
                pass

    def close(self) -> None:
        self._closed.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None
        if self._hcomm_client is not None and hasattr(self._hcomm_client, "close"):
            self._hcomm_client.close()
            self._hcomm_client = None
        if self._rxe_client is not None and hasattr(self._rxe_client, "close"):
            self._rxe_client.close()
            self._rxe_client = None
        self._client.close()


def sleep_poll_interval() -> None:
    time.sleep(0.0005)


def _tensor_nbytes(tensor) -> int:  # noqa: ANN001
    nbytes = tensor.nbytes
    return int(nbytes() if callable(nbytes) else nbytes)


def _output_tensor_indexes(args: TaskArgs) -> list[int]:
    return [
        i
        for i in range(args.tensor_count())
        if args.tag(i).name in {"OUTPUT", "INOUT", "OUTPUT_EXISTING"}
    ]


def _join_chunks(chunks) -> bytes:  # noqa: ANN001
    chunks = list(chunks)
    total = 0
    for chunk in chunks:
        total = max(total, int(chunk.offset) + len(chunk.data))
    out = bytearray(total)
    for chunk in chunks:
        offset = int(chunk.offset)
        out[offset : offset + len(chunk.data)] = chunk.data
    return bytes(out)
