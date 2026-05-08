"""Long-running L3 worker daemon for remote L4 dispatch."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import threading
import traceback
from collections.abc import Callable
from typing import Optional

import grpc

from simpler.worker import Worker

from .catalog import Catalog, CatalogService
from .proto import dispatch_pb2, dispatch_pb2_grpc
from .rpc import RpcServer
from .serialization import (
    decode_config,
    decode_task_args,
    decode_task_args_with_tensor_refs_and_writebacks,
    encode_output_tensor_refs,
)
from .tensor_pool import TensorPool
from .transport_backend import build_tensor_transport


class L3Daemon(dispatch_pb2_grpc.L3WorkerServicer):
    """RPC facade that delegates dispatches to a lazily initialized inner Worker."""

    def __init__(
        self,
        port: int = 0,
        worker_factory: Optional[Callable[[], Worker]] = None,
        *,
        tensor_transport: Optional[str] = None,
    ) -> None:
        self.port = int(port)
        self.catalog = Catalog()
        self.tensor_pool = TensorPool()
        self.tensor_transport = tensor_transport or os.getenv("SIMPLER_TENSOR_TRANSPORT", "grpc")
        self._worker_factory = worker_factory or (lambda: Worker(level=3, num_sub_workers=1))
        self._server: Optional[RpcServer] = None
        self._backend_proc = None
        self._backend_conn = None
        self._backend_lock = threading.Lock()

    def start(self, host: str = "127.0.0.1") -> int:
        self._start_backend()
        server = RpcServer()
        server.add_l3_worker(self)
        server.add_catalog(_BackendCatalogService(self.catalog, self._backend_call))
        server.add_tensor_pool(_BackendTensorPoolService(self._backend_call))
        self.port = server.start(self.port, host)
        self._server = server
        return self.port

    def serve_forever(self, host: str = "127.0.0.1") -> None:
        self.start(host)
        assert self._server is not None
        self._server.wait_for_termination()

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(0)
            self._server = None
        if self._backend_conn is not None:
            try:
                self._backend_call(("stop",))
            except Exception:  # noqa: BLE001
                pass
            self._backend_conn.close()
            self._backend_conn = None
        if self._backend_proc is not None:
            self._backend_proc.join(timeout=5.0)
            if self._backend_proc.is_alive():
                self._backend_proc.terminate()
                self._backend_proc.join(timeout=5.0)
            self._backend_proc = None

    def Dispatch(self, request, context):  # noqa: N802, ANN001
        try:
            return self._on_dispatch(request)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            return dispatch_pb2.DispatchResp(
                task_id=request.task_id,
                error_code=1,
                error_msg=f"{type(e).__name__}: {e}",
                remote_traceback=[tb],
            )

    def Heartbeat(self, request, context):  # noqa: N802, ANN001
        return dispatch_pb2.Health(ok=True, message="ok")

    def _on_dispatch(self, req: dispatch_pb2.DispatchReq) -> dispatch_pb2.DispatchResp:
        resp_bytes = self._backend_call(("dispatch", req.SerializeToString()))
        resp = dispatch_pb2.DispatchResp()
        resp.ParseFromString(resp_bytes)
        return resp

    def _start_backend(self) -> None:
        if self._backend_proc is not None:
            return
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_backend_loop,
            args=(
                child_conn,
                self._worker_factory,
                self.tensor_pool.node_id,
                self.tensor_pool.inline_threshold,
                self.tensor_pool.capacity_bytes,
                self.tensor_pool.default_ttl_ms,
                self.tensor_transport,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._backend_conn = parent_conn
        self._backend_proc = proc

    def _backend_call(self, msg):
        if self._backend_conn is None:
            raise RuntimeError("L3 daemon backend is not running")
        with self._backend_lock:
            self._backend_conn.send(msg)
            ok, payload = self._backend_conn.recv()
        if not ok:
            raise RuntimeError(payload)
        return payload


class _BackendCatalogService(CatalogService):
    def __init__(self, catalog: Catalog, backend_call) -> None:
        super().__init__(catalog)
        self._backend_call = backend_call

    def PushCallable(self, request, context):  # noqa: N802, ANN001
        super().PushCallable(request, context)
        self._backend_call(("push", request.callable_id, request.version, bytes(request.pickled)))
        return dispatch_pb2.Empty()


class _BackendTensorPoolService(dispatch_pb2_grpc.TensorPoolServicer):
    def __init__(self, backend_call) -> None:
        self._backend_call = backend_call

    def AllocTensor(self, request, context):  # noqa: N802, ANN001
        try:
            data = self._backend_call(("tensor_alloc", request.SerializeToString()))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
        handle = dispatch_pb2.TensorHandle()
        handle.ParseFromString(data)
        return handle

    def FreeTensor(self, request, context):  # noqa: N802, ANN001
        try:
            self._backend_call(("tensor_free", request.handle.SerializeToString()))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return dispatch_pb2.Empty()

    def RefreshTensor(self, request, context):  # noqa: N802, ANN001
        try:
            data = self._backend_call(("tensor_refresh", request.SerializeToString()))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        handle = dispatch_pb2.TensorHandle()
        handle.ParseFromString(data)
        return handle

    def PullTensor(self, request, context):  # noqa: N802, ANN001
        try:
            payload = self._backend_call(("tensor_pull", request.SerializeToString()))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        chunk_size = 1024 * 1024
        for offset in range(0, len(payload), chunk_size):
            chunk = payload[offset : offset + chunk_size]
            yield dispatch_pb2.TensorChunk(
                handle=request,
                offset=offset,
                data=chunk,
                last=offset + len(chunk) >= len(payload),
            )
        if not payload:
            yield dispatch_pb2.TensorChunk(handle=request, offset=0, data=b"", last=True)

    def PushTensor(self, request_iterator, context):  # noqa: N802, ANN001
        chunks = list(request_iterator)
        handle = chunks[0].handle if chunks else dispatch_pb2.TensorHandle()
        payload = _join_chunks(chunks)
        try:
            data = self._backend_call(("tensor_push", handle.SerializeToString(), payload))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        out = dispatch_pb2.TensorHandle()
        out.ParseFromString(data)
        return out


def _backend_loop(conn, worker_factory, node_id, inline_threshold, capacity_bytes, default_ttl_ms, tensor_transport) -> None:
    inner: Optional[Worker] = None
    try:
        catalog = Catalog()
        tensor_pool = TensorPool(
            node_id=node_id,
            inline_threshold=inline_threshold,
            capacity_bytes=capacity_bytes,
            default_ttl_ms=default_ttl_ms,
            transport_backend=build_tensor_transport(tensor_transport),
        )
        while True:
            msg = conn.recv()
            op = msg[0]
            if op == "stop":
                conn.send((True, None))
                break
            if op == "push":
                _, cid, version, payload = msg
                catalog.install_from_payload(cid, version, payload)
                conn.send((True, None))
                continue
            if op == "tensor_alloc":
                _, req_bytes = msg
                req = dispatch_pb2.TensorAllocReq()
                req.ParseFromString(req_bytes)
                handle = tensor_pool.alloc(
                    req.nbytes,
                    ttl_ms=req.ttl_ms,
                    shape=req.shape,
                    dtype=req.dtype,
                    tag=req.tag,
                )
                conn.send((True, handle.SerializeToString()))
                continue
            if op == "tensor_free":
                _, handle_bytes = msg
                handle = dispatch_pb2.TensorHandle()
                handle.ParseFromString(handle_bytes)
                tensor_pool.free(handle)
                conn.send((True, None))
                continue
            if op == "tensor_refresh":
                _, req_bytes = msg
                req = dispatch_pb2.TensorRefreshReq()
                req.ParseFromString(req_bytes)
                handle = tensor_pool.refresh(req.handle, req.ttl_ms)
                conn.send((True, handle.SerializeToString()))
                continue
            if op == "tensor_pull":
                _, handle_bytes = msg
                handle = dispatch_pb2.TensorHandle()
                handle.ParseFromString(handle_bytes)
                conn.send((True, tensor_pool.get_bytes(handle)))
                continue
            if op == "tensor_push":
                _, handle_bytes, payload = msg
                handle = dispatch_pb2.TensorHandle()
                handle.ParseFromString(handle_bytes)
                if handle.handle_id:
                    tensor_pool.write_bytes(handle, payload)
                    out = tensor_pool.refresh(handle)
                else:
                    out_ref = tensor_pool.put_bytes(payload, force_handle=True)
                    out = out_ref.handle
                conn.send((True, out.SerializeToString()))
                continue
            if op == "dispatch":
                _, req_bytes = msg
                req = dispatch_pb2.DispatchReq()
                req.ParseFromString(req_bytes)
                resp, inner = _backend_dispatch(req, catalog, tensor_pool, worker_factory, inner)
                conn.send((True, resp.SerializeToString()))
                continue
            raise RuntimeError(f"unknown backend op {op!r}")
    except EOFError:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            conn.send((False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        if "tensor_pool" in locals():
            tensor_pool.close()
        if inner is not None:
            inner.close()


def _backend_dispatch(
    req: dispatch_pb2.DispatchReq,
    catalog: Catalog,
    tensor_pool: TensorPool,
    worker_factory: Callable[[], Worker],
    inner: Optional[Worker],
) -> tuple[dispatch_pb2.DispatchResp, Optional[Worker]]:
    run_inner = inner
    ephemeral_inner = False
    try:
        orch_fn = catalog.lookup(req.callable_id, req.callable_version)
        if orch_fn is None:
            return (
                dispatch_pb2.DispatchResp(
                    task_id=req.task_id,
                    error_code=2,
                    error_msg=f"callable {req.callable_id} version {req.callable_version} not in catalog",
                ),
                inner,
            )
        cfg = decode_config(req.config_blob)
        keepalive = []
        writebacks = []
        if req.tensor_refs:
            args, keepalive, writebacks = decode_task_args_with_tensor_refs_and_writebacks(
                req.tensor_refs,
                req.scalar_args,
                tensor_pool,
            )
        else:
            args = decode_task_args(req.tensor_args, req.scalar_args)
        if req.tensor_refs:
            run_inner = worker_factory()
            ephemeral_inner = True
            _install_catalog(run_inner, catalog)
            run_inner.init()
        elif run_inner is None:
            run_inner = worker_factory()
            _install_catalog(run_inner, catalog)
            run_inner.init()
            inner = run_inner
        run_inner.run(orch_fn, args, cfg)
        output_tensors = encode_output_tensor_refs(args, tensor_pool, writebacks)
        keepalive.clear()
        return dispatch_pb2.DispatchResp(task_id=req.task_id, error_code=0, output_tensors=output_tensors), inner
    except Exception as e:  # noqa: BLE001
        return (
            dispatch_pb2.DispatchResp(
                task_id=req.task_id,
                error_code=1,
                error_msg=f"{type(e).__name__}: {e}",
                remote_traceback=[traceback.format_exc()],
            ),
            inner,
        )
    finally:
        if ephemeral_inner and run_inner is not None:
            run_inner.close()


def _install_catalog(worker: Worker, catalog: Catalog) -> None:
    for cid, version in catalog.refs():
        fn = catalog.lookup(cid, version)
        if fn is not None:
            worker._callable_registry[int(cid)] = fn


def _join_chunks(chunks: list[dispatch_pb2.TensorChunk]) -> bytes:
    total = 0
    for chunk in chunks:
        total = max(total, int(chunk.offset) + len(chunk.data))
    out = bytearray(total)
    for chunk in chunks:
        offset = int(chunk.offset)
        out[offset : offset + len(chunk.data)] = chunk.data
    return bytes(out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--num-sub-workers", type=int, default=1)
    parser.add_argument("--tensor-transport", default=None, choices=("grpc", "rxe", "hcomm", "auto"))
    args = parser.parse_args(argv)

    def make_worker() -> Worker:
        return Worker(level=3, num_sub_workers=args.num_sub_workers)

    daemon = L3Daemon(args.port, make_worker, tensor_transport=args.tensor_transport)
    try:
        daemon.serve_forever(args.host)
    except KeyboardInterrupt:
        daemon.stop()
        return 130
    except grpc.RpcError:
        daemon.stop()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
