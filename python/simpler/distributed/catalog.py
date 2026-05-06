"""Callable catalog for cross-host Worker dispatch."""

from __future__ import annotations

import hashlib
import importlib
import pickle
from collections.abc import Callable
from typing import Optional, Tuple

import grpc

from .proto import dispatch_pb2, dispatch_pb2_grpc

try:
    import cloudpickle as _pickle_impl
except Exception:  # noqa: BLE001
    _pickle_impl = pickle


class CatalogError(RuntimeError):
    pass


class Catalog:
    """Stable callable ids plus versioned serialized payloads."""

    def __init__(self, *, allowed_modules: Optional[Tuple[str, ...]] = None) -> None:
        self._functions: dict[tuple[int, int], Callable] = {}
        self._payloads: dict[tuple[int, int], bytes] = {}
        self._latest: dict[int, int] = {}
        self._next_id = 0
        self._allowed_modules = allowed_modules

    def register(self, fn: Callable, callable_id: Optional[int] = None) -> tuple[int, int]:
        payload = _pickle_impl.dumps(fn)
        version = _version(payload)
        cid = self._next_id if callable_id is None else int(callable_id)
        self.install_from_payload(cid, version, payload)
        if callable_id is None:
            self._next_id = max(self._next_id, cid + 1)
        else:
            self._next_id = max(self._next_id, cid + 1)
        return cid, version

    def lookup(self, cid: int, version: Optional[int] = None) -> Optional[Callable]:
        cid = int(cid)
        if version is None or int(version) == 0:
            version = self._latest.get(cid)
            if version is None:
                return None
        return self._functions.get((cid, int(version)))

    def install_from_payload(self, cid: int, version: int, payload: bytes) -> None:
        cid = int(cid)
        version = int(version)
        actual = _version(payload)
        if version != actual:
            raise CatalogError(f"callable {cid} version mismatch: expected {version}, payload has {actual}")
        fn = _loads_with_allowlist(payload, self._allowed_modules)
        if not callable(fn):
            raise CatalogError(f"payload for callable {cid} did not deserialize to a callable")
        key = (cid, version)
        self._functions[key] = fn
        self._payloads[key] = bytes(payload)
        self._latest[cid] = version
        self._next_id = max(self._next_id, cid + 1)

    def export_payload(self, cid: int, version: Optional[int] = None) -> bytes:
        cid = int(cid)
        if version is None or int(version) == 0:
            version = self._latest.get(cid)
        key = (cid, int(version)) if version is not None else None
        if key not in self._payloads:
            raise CatalogError(f"callable {cid} version {version} not found")
        return self._payloads[key]

    def refs(self) -> list[tuple[int, int]]:
        return sorted(self._payloads)

    def refs_by_id(self) -> dict[int, int]:
        return dict(self._latest)

    def payloads(self) -> list[dispatch_pb2.CallablePayload]:
        return [
            dispatch_pb2.CallablePayload(callable_id=cid, version=version, pickled=payload)
            for (cid, version), payload in sorted(self._payloads.items())
        ]


def _version(payload: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _loads_with_allowlist(payload: bytes, allowed_modules: Optional[Tuple[str, ...]]) -> Callable:
    if allowed_modules is None:
        return pickle.loads(payload)

    class AllowlistUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):  # noqa: ANN001
            if not any(module == prefix or module.startswith(prefix + ".") for prefix in allowed_modules):
                raise CatalogError(f"module {module!r} is not allowed in callable payload")
            importlib.import_module(module)
            return super().find_class(module, name)

    import io

    return AllowlistUnpickler(io.BytesIO(payload)).load()


class CatalogService(dispatch_pb2_grpc.CatalogServicer):
    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def PullCallable(self, request, context):  # noqa: N802, ANN001
        try:
            payload = self._catalog.export_payload(request.callable_id, request.version)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return dispatch_pb2.CallablePayload(
            callable_id=request.callable_id,
            version=request.version,
            pickled=payload,
        )

    def PushCallable(self, request, context):  # noqa: N802, ANN001
        try:
            self._catalog.install_from_payload(request.callable_id, request.version, request.pickled)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return dispatch_pb2.Empty()
