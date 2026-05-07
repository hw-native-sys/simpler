"""L4-side proxy for a remote L3 worker."""

from __future__ import annotations

import itertools
import threading
import time
from typing import Optional

from simpler.task_interface import CallConfig, TaskArgs

from .catalog import Catalog
from .proto import dispatch_pb2
from .rpc import RpcClient, RpcError
from .serialization import encode_config, encode_task_args


class RemoteUnavailable(RuntimeError):
    pass


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
    ) -> None:
        self.endpoint = endpoint
        self._client = RpcClient(endpoint)
        self._catalog = l4_catalog
        self._timeout = float(timeout)
        self._heartbeat_timeout = float(heartbeat_timeout)
        self._heartbeat_interval = float(heartbeat_interval)
        self._heartbeat_failures = int(heartbeat_failures)
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
        version = self._catalog.refs_by_id().get(int(callable_id), 0)
        req = dispatch_pb2.DispatchReq(
            task_id=next(self._task_ids),
            callable_id=int(callable_id),
            callable_version=int(version),
            config_blob=encode_config(config),
            scalar_args=scalar_args,
            tensor_args=tensor_args,
        )
        try:
            resp = self._client.dispatch(req, self._timeout)
        except RpcError as e:
            self._available = False
            raise RemoteUnavailable(f"remote {self.endpoint} dispatch RPC failed: {e}") from e
        if resp.error_code != 0:
            detail = resp.error_msg
            if resp.remote_traceback:
                detail = detail + "\nremote traceback:\n" + "\n".join(resp.remote_traceback)
            raise RuntimeError(f"remote dispatch failed on {self.endpoint}: {detail}")

    def close(self) -> None:
        self._closed.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None
        self._client.close()


def sleep_poll_interval() -> None:
    time.sleep(0.0005)
