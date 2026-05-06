"""Long-running L3 worker daemon for remote L4 dispatch."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import threading
import traceback
from collections.abc import Callable
from typing import Optional

import grpc

from simpler.worker import Worker

from .catalog import Catalog, CatalogService
from .proto import dispatch_pb2, dispatch_pb2_grpc
from .rpc import RpcServer
from .serialization import decode_config, decode_task_args
from .tensor_pool import TensorPool


class L3Daemon(dispatch_pb2_grpc.L3WorkerServicer):
    """RPC facade that delegates dispatches to a lazily initialized inner Worker."""

    def __init__(self, port: int = 0, worker_factory: Optional[Callable[[], Worker]] = None) -> None:
        self.port = int(port)
        self.catalog = Catalog()
        self.tensor_pool = TensorPool()
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
        server.add_tensor_pool(self.tensor_pool.service())
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
        proc = ctx.Process(target=_backend_loop, args=(child_conn, self._worker_factory), daemon=True)
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


def _backend_loop(conn, worker_factory) -> None:
    catalog = Catalog()
    inner: Optional[Worker] = None
    try:
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
            if op == "dispatch":
                _, req_bytes = msg
                req = dispatch_pb2.DispatchReq()
                req.ParseFromString(req_bytes)
                resp, inner = _backend_dispatch(req, catalog, worker_factory, inner)
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
        if inner is not None:
            inner.close()


def _backend_dispatch(
    req: dispatch_pb2.DispatchReq,
    catalog: Catalog,
    worker_factory: Callable[[], Worker],
    inner: Optional[Worker],
) -> tuple[dispatch_pb2.DispatchResp, Optional[Worker]]:
    try:
        if inner is None:
            inner = worker_factory()
            for cid, version in catalog.refs():
                fn = catalog.lookup(cid, version)
                if fn is not None:
                    inner._callable_registry[int(cid)] = fn
            inner.init()
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
        args = decode_task_args(req.tensor_args, req.scalar_args)
        inner.run(orch_fn, args, cfg)
        return dispatch_pb2.DispatchResp(task_id=req.task_id, error_code=0), inner
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--num-sub-workers", type=int, default=1)
    args = parser.parse_args(argv)

    def make_worker() -> Worker:
        return Worker(level=3, num_sub_workers=args.num_sub_workers)

    daemon = L3Daemon(args.port, make_worker)
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
