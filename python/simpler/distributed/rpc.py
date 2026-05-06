"""Small grpcio wrappers used by distributed dispatch."""

from __future__ import annotations

from concurrent import futures
from typing import Any, Callable, Optional

import grpc

from .proto import dispatch_pb2, dispatch_pb2_grpc

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
    ("grpc.so_reuseport", 0),
]


class RpcError(RuntimeError):
    """Raised when a gRPC call fails."""

    def __init__(self, message: str, *, code: Optional[grpc.StatusCode] = None, remote_traceback: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.remote_traceback = remote_traceback


class RpcServer:
    """Thin owner for a grpc.server instance."""

    def __init__(self, *, max_workers: int = 8) -> None:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=_CHANNEL_OPTIONS)
        self._port: Optional[int] = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("RpcServer has not been started")
        return self._port

    def add_l3_worker(self, impl: dispatch_pb2_grpc.L3WorkerServicer) -> None:
        dispatch_pb2_grpc.add_L3WorkerServicer_to_server(impl, self._server)

    def add_catalog(self, impl: dispatch_pb2_grpc.CatalogServicer) -> None:
        dispatch_pb2_grpc.add_CatalogServicer_to_server(impl, self._server)

    def add_tensor_pool(self, impl: dispatch_pb2_grpc.TensorPoolServicer) -> None:
        dispatch_pb2_grpc.add_TensorPoolServicer_to_server(impl, self._server)

    def add_handler(self, service: str, impl: Any) -> None:
        if service == "L3Worker":
            self.add_l3_worker(impl)
        elif service == "Catalog":
            self.add_catalog(impl)
        elif service == "TensorPool":
            self.add_tensor_pool(impl)
        else:
            raise ValueError(f"unknown service {service!r}")

    def start(self, port: int = 0, host: str = "127.0.0.1") -> int:
        try:
            bound = self._server.add_insecure_port(f"{host}:{int(port)}")
        except RuntimeError as e:
            raise RpcError(f"failed to bind gRPC server on {host}:{port}: {e}") from e
        if bound == 0:
            raise RpcError(f"failed to bind gRPC server on {host}:{port}")
        self._server.start()
        self._port = bound
        return bound

    def wait_for_termination(self) -> None:
        self._server.wait_for_termination()

    def stop(self, grace: Optional[float] = 0) -> None:
        self._server.stop(grace)


class RpcClient:
    """Typed client wrapper for the distributed proto services."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._channel = grpc.insecure_channel(endpoint, options=_CHANNEL_OPTIONS)
        self.l3_worker = dispatch_pb2_grpc.L3WorkerStub(self._channel)
        self.catalog = dispatch_pb2_grpc.CatalogStub(self._channel)
        self.tensor_pool = dispatch_pb2_grpc.TensorPoolStub(self._channel)

    def call_unary(self, method: str, req: Any, timeout: Optional[float] = None) -> Any:
        mapping: dict[str, Callable[..., Any]] = {
            "L3Worker.Dispatch": self.l3_worker.Dispatch,
            "L3Worker.Heartbeat": self.l3_worker.Heartbeat,
            "Catalog.PullCallable": self.catalog.PullCallable,
            "Catalog.PushCallable": self.catalog.PushCallable,
        }
        try:
            return mapping[method](req, timeout=timeout)
        except KeyError as e:
            raise ValueError(f"unknown unary method {method!r}") from e
        except grpc.RpcError as e:
            raise RpcError(str(e.details() or e), code=e.code()) from e

    def dispatch(self, req: dispatch_pb2.DispatchReq, timeout: Optional[float] = None) -> dispatch_pb2.DispatchResp:
        return self.call_unary("L3Worker.Dispatch", req, timeout)

    def heartbeat(self, timeout: Optional[float] = None) -> dispatch_pb2.Health:
        return self.call_unary("L3Worker.Heartbeat", dispatch_pb2.Empty(), timeout)

    def close(self) -> None:
        self._channel.close()
