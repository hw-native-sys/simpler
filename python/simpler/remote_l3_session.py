# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Remote L3 session runner.

The runner is started by ``simpler-remote-worker`` after the daemon validates a
bootstrap manifest. It reads the manifest before starting transport threads,
prestarts the embedded L3 Worker, then exposes the Remote L3 command lane.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Callable

from .callable_identity import (
    CallableHandle,
    build_chip_callable_descriptor,
    build_python_import_descriptor,
    compute_callable_hashid,
    hashid_to_digest,
    parse_python_import_target,
    validate_hashid,
)
from .remote_l3_protocol import (
    CallableKind,
    ChipCallableBlobLocation,
    ControlName,
    ExportBufferResult,
    FrameHeader,
    FrameType,
    HelloPayload,
    ImportBufferResult,
    PROTOCOL_VERSION,
    ReadyState,
    RemoteAddressSpace,
    RemoteRegistryTarget,
    RemoteTaskArgsWire,
    decode_control,
    decode_digest_callable_command,
    decode_export_buffer_request,
    decode_import_buffer_request,
    decode_register_callable_command,
    decode_release_import_request,
    decode_remote_chip_callable_payload,
    decode_task_payload,
    encode_completion,
    encode_control_reply,
    encode_export_buffer_result,
    encode_frame,
    encode_hello,
    encode_import_buffer_result,
    encode_register_callable_command,
    read_frame,
    send_frame,
)
from .task_interface import ChipCallable, TaskArgs, Tensor
from .worker import Worker

sys.modules.setdefault("simpler.remote_l3_session", sys.modules[__name__])


_INNER_HANDLE_LOCK = threading.Lock()
_INNER_HANDLES: dict[bytes, CallableHandle] = {}


def get_inner_handle(hashid: str) -> CallableHandle:
    normalized = hashid if hashid.startswith("sha256:") else f"sha256:{hashid}"
    validate_hashid(normalized)
    digest = hashid_to_digest(normalized)
    with _INNER_HANDLE_LOCK:
        handle = _INNER_HANDLES.get(digest)
    if handle is None:
        raise KeyError(f"remote inner callable {normalized} is not installed")
    return handle


def _publish_inner_handle(digest: bytes, handle: CallableHandle) -> None:
    with _INNER_HANDLE_LOCK:
        _INNER_HANDLES[digest] = handle


def _unpublish_inner_handle(digest: bytes) -> None:
    with _INNER_HANDLE_LOCK:
        _INNER_HANDLES.pop(digest, None)


@dataclass
class _RemoteBufferEntry:
    data: Any
    nbytes: int
    generation: int
    address_space: RemoteAddressSpace
    offset: int = 0
    released: bool = False

    @property
    def addr(self) -> int:
        if isinstance(self.data, shared_memory.SharedMemory):
            buf = self.data.buf
            assert buf is not None
            return ctypes.addressof(ctypes.c_char.from_buffer(buf))
        return ctypes.addressof(self.data)

    @property
    def shm_name(self) -> str:
        if not isinstance(self.data, shared_memory.SharedMemory):
            raise ValueError("remote buffer is not backed by SharedMemory")
        return self.data.name

    def close(self, *, unlink: bool = False) -> None:
        if not isinstance(self.data, shared_memory.SharedMemory):
            return
        self.data.close()
        if unlink:
            try:
                self.data.unlink()
            except FileNotFoundError:
                pass


def _load_import_target(target: str) -> Callable[..., Any]:
    module_name, qualname = parse_python_import_target(target)
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"remote callable {target!r} is not callable")
    return obj


def _bind_listener(host: str) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen(1)
    return sock


def _send_ready(fd: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
    os.write(fd, data)
    os.close(fd)


def _session_timeout_s(manifest: dict[str, Any]) -> float:
    timeout_s = float(manifest.get("session_timeout_s", 30.0))
    if not (timeout_s > 0 and math.isfinite(timeout_s)):
        raise ValueError("manifest session_timeout_s must be a positive finite number of seconds")
    return timeout_s


def _startup_remaining_s(manifest: dict[str, Any]) -> float:
    # The parent's remaining slice of the single root startup budget. Absent only
    # from a pre-P0.3 parent; fall back to the runtime command timeout then. The
    # fallback is evaluated only when the key is absent, so a valid
    # startup_remaining_s is not held hostage to an invalid session_timeout_s.
    if "startup_remaining_s" not in manifest:
        return _session_timeout_s(manifest)
    remaining_s = float(manifest["startup_remaining_s"])
    if not (remaining_s > 0 and math.isfinite(remaining_s)):
        raise ValueError("manifest startup_remaining_s must be a positive finite number of seconds")
    return remaining_s


def _health_loop(sock: socket.socket, stop: threading.Event, session_id: int, worker_id: int) -> None:
    conn: socket.socket | None = None
    sock.settimeout(0.2)
    try:
        sequence = 0
        while not stop.is_set():
            if conn is None:
                try:
                    conn, _addr = sock.accept()
                    conn.settimeout(0.2)
                except socket.timeout:
                    continue
            try:
                sequence += 1
                header = FrameHeader(FrameType.HEALTH, session_id, worker_id, sequence)
                conn.sendall(encode_frame(header, b""))
                stop.wait(0.2)
            except OSError:
                try:
                    conn.close()
                except OSError:
                    pass
                conn = None
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        try:
            sock.close()
        except OSError:
            pass


def _format_remote_error(prefix: str, exc: BaseException) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{prefix}: {text}"[:4096]


def _validate_python_import_command(command_payload: bytes) -> tuple[bytes, str, Callable[..., Any]]:
    command = decode_register_callable_command(command_payload)
    if command.callable_kind != CallableKind.PYTHON_IMPORT:
        raise ValueError("PYTHON_IMPORT is the required remote callable kind")
    if command.payload_version != 1:
        raise ValueError("PYTHON_IMPORT payload_version must be 1")
    target = command.payload.decode("utf-8")
    module, qualname = parse_python_import_target(target)
    rebuilt = hashid_to_digest(compute_callable_hashid(build_python_import_descriptor(module, qualname)))
    if rebuilt != command.digest:
        raise ValueError("PYTHON_IMPORT callable hashid does not match canonical descriptor")
    return command.digest, target, _load_import_target(target)


def _chip_callable_bytes(callable_obj: ChipCallable) -> bytes:
    return ctypes.string_at(int(callable_obj.buffer_ptr()), int(callable_obj.buffer_size()))


def _prepare_inner_chip_callable(command_payload: bytes, manifest: dict[str, Any]) -> tuple[bytes, ChipCallable]:
    command = decode_register_callable_command(command_payload)
    if command.callable_kind != CallableKind.CHIP_CALLABLE:
        raise ValueError("CHIP_CALLABLE command expected")
    if command.payload_version != 1:
        raise ValueError("CHIP_CALLABLE payload_version must be 1")
    payload = decode_remote_chip_callable_payload(command.payload)
    if payload.blob_location != ChipCallableBlobLocation.INLINE_BLOB:
        raise ValueError("CHIP_CALLABLE STAGED_BLOB is unsupported without a negotiated staged-blob adapter")
    if hashlib.sha256(payload.inline_blob).digest() != payload.blob_sha256:
        raise ValueError("CHIP_CALLABLE executable blob SHA-256 mismatch")
    callable_obj = ChipCallable.from_bytes(payload.inline_blob)
    platform = str(manifest.get("platform", ""))
    runtime = str(manifest.get("runtime", ""))
    rebuilt_descriptor = build_chip_callable_descriptor(target=callable_obj, platform=platform, runtime=runtime)
    if rebuilt_descriptor != payload.descriptor_bytes:
        raise ValueError("CHIP_CALLABLE descriptor does not match executable blob or endpoint context")
    rebuilt_digest = hashid_to_digest(compute_callable_hashid(rebuilt_descriptor))
    if rebuilt_digest != command.digest:
        raise ValueError("CHIP_CALLABLE callable hashid does not match canonical descriptor")
    return command.digest, callable_obj


def _prepare_register_callable(
    command_payload: bytes, manifest: dict[str, Any]
) -> tuple[bytes, CallableKind, RemoteRegistryTarget, Any]:
    command = decode_register_callable_command(command_payload)
    if command.callable_kind == CallableKind.PYTHON_SERIALIZED:
        raise ValueError("PYTHON_SERIALIZED is not negotiated for remote protocol v1")
    if command.target_registry == RemoteRegistryTarget.REMOTE_TASK_DISPATCHER:
        if command.callable_kind != CallableKind.PYTHON_IMPORT:
            raise ValueError("REMOTE_TASK_DISPATCHER only accepts PYTHON_IMPORT in protocol v1")
        digest, _target, target_callable = _validate_python_import_command(command_payload)
        return digest, command.callable_kind, command.target_registry, target_callable
    if command.target_registry == RemoteRegistryTarget.INNER_L3_WORKER:
        if command.callable_kind == CallableKind.PYTHON_IMPORT:
            digest, target, _target_callable = _validate_python_import_command(command_payload)
            return digest, command.callable_kind, command.target_registry, target
        if command.callable_kind == CallableKind.CHIP_CALLABLE:
            digest, callable_obj = _prepare_inner_chip_callable(command_payload, manifest)
            return digest, command.callable_kind, command.target_registry, callable_obj
    raise ValueError("unsupported remote callable target/kind combination")


def _decode_digest_command(command_payload: bytes) -> tuple[bytes, CallableKind, RemoteRegistryTarget]:
    command = decode_digest_callable_command(command_payload)
    return command.digest, command.callable_kind, command.target_registry


def _manifest_digest(entry: dict[str, Any], *, context: str = "inner_l3_worker") -> bytes:
    raw = entry.get("hashid")
    if not isinstance(raw, str):
        raise ValueError(f"{context} manifest entry requires string hashid")
    normalized = raw if raw.startswith("sha256:") else f"sha256:{raw}"
    validate_hashid(normalized)
    return hashid_to_digest(normalized)


def _manifest_kind(entry: dict[str, Any], *, context: str = "inner_l3_worker") -> CallableKind:
    raw = entry.get("kind")
    if not isinstance(raw, str):
        raise ValueError(f"{context} manifest entry requires string kind")
    try:
        return CallableKind[raw]
    except KeyError as exc:
        raise ValueError(f"{context} manifest entry has unsupported kind {raw!r}") from exc


def _manifest_dispatcher_register_command(entry: dict[str, Any]) -> bytes:
    target_registry = entry.get("target_registry", "REMOTE_TASK_DISPATCHER")
    if target_registry != "REMOTE_TASK_DISPATCHER":
        raise ValueError("remote_task_dispatcher manifest entry target_registry must be REMOTE_TASK_DISPATCHER")
    kind = _manifest_kind(entry, context="remote_task_dispatcher")
    if kind != CallableKind.PYTHON_IMPORT:
        raise ValueError("remote_task_dispatcher manifest entry kind must be PYTHON_IMPORT")
    digest = _manifest_digest(entry, context="remote_task_dispatcher")
    payload_version = int(entry.get("payload_version", 1))
    target = entry.get("target")
    if not isinstance(target, str):
        raise ValueError("remote_task_dispatcher PYTHON_IMPORT manifest entry requires string target")
    return encode_register_callable_command(
        RemoteRegistryTarget.REMOTE_TASK_DISPATCHER,
        kind,
        digest,
        payload_version,
        target.encode("utf-8"),
    )


def _manifest_inner_register_command(entry: dict[str, Any]) -> bytes:
    target_registry = entry.get("target_registry", "INNER_L3_WORKER")
    if target_registry != "INNER_L3_WORKER":
        raise ValueError("inner_l3_worker manifest entry target_registry must be INNER_L3_WORKER")
    kind = _manifest_kind(entry)
    digest = _manifest_digest(entry)
    payload_version = int(entry.get("payload_version", 1))
    if kind == CallableKind.PYTHON_IMPORT:
        target = entry.get("target")
        if not isinstance(target, str):
            raise ValueError("inner_l3_worker PYTHON_IMPORT manifest entry requires string target")
        payload = target.encode("utf-8")
    elif kind == CallableKind.CHIP_CALLABLE:
        payload_hex = entry.get("payload_hex")
        if not isinstance(payload_hex, str):
            raise ValueError("inner_l3_worker CHIP_CALLABLE manifest entry requires payload_hex")
        try:
            payload = bytes.fromhex(payload_hex)
        except ValueError as exc:
            raise ValueError("inner_l3_worker CHIP_CALLABLE manifest payload_hex is not valid hex") from exc
    else:
        payload = b""
    return encode_register_callable_command(
        RemoteRegistryTarget.INNER_L3_WORKER,
        kind,
        digest,
        payload_version,
        payload,
    )


def _install_manifest_dispatcher_registry(manifest: dict[str, Any]) -> dict[bytes, Callable[..., Any]]:
    entries = manifest.get("remote_task_dispatcher", [])
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ValueError("remote_task_dispatcher manifest registry must be a list")

    installed: dict[bytes, Callable[..., Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("remote_task_dispatcher manifest entries must be objects")
        command_payload = _manifest_dispatcher_register_command(entry)
        digest, kind, registry, target = _prepare_register_callable(command_payload, manifest)
        if registry != RemoteRegistryTarget.REMOTE_TASK_DISPATCHER or kind != CallableKind.PYTHON_IMPORT:
            raise ValueError("remote_task_dispatcher manifest entry decoded to the wrong registry")
        if digest in installed:
            raise ValueError(f"remote_task_dispatcher manifest contains duplicate hashid {digest.hex()}")
        installed[digest] = target
    return installed


def _install_manifest_inner_registry(
    manifest: dict[str, Any],
    inner_worker: Worker,
) -> dict[tuple[CallableKind, bytes], CallableHandle]:
    entries = manifest.get("inner_l3_worker", [])
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ValueError("inner_l3_worker manifest registry must be a list")

    installed: dict[tuple[CallableKind, bytes], CallableHandle] = {}
    seen_digests: set[bytes] = set()
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("inner_l3_worker manifest entries must be objects")
            command_payload = _manifest_inner_register_command(entry)
            digest, kind, registry, target = _prepare_register_callable(command_payload, manifest)
            if registry != RemoteRegistryTarget.INNER_L3_WORKER:
                raise ValueError("inner_l3_worker manifest entry decoded to the wrong registry")
            if digest in seen_digests:
                raise ValueError(f"inner_l3_worker manifest contains duplicate hashid {digest.hex()}")
            seen_digests.add(digest)

            if kind == CallableKind.PYTHON_IMPORT:
                handle = inner_worker._register_child_python_import(target, digest=digest)  # noqa: SLF001
            elif kind == CallableKind.CHIP_CALLABLE:
                handle = inner_worker._register_child_chip(  # noqa: SLF001
                    target,
                    digest=digest,
                    publish_handle=True,
                )
                if handle is None:
                    raise RuntimeError("inner_l3_worker CHIP_CALLABLE manifest install did not publish a handle")
            else:
                raise ValueError("inner_l3_worker manifest callable kind is not supported in protocol v1")

            key = (kind, digest)
            installed[key] = handle
            _publish_inner_handle(digest, handle)
    except BaseException:
        for _key, handle in reversed(list(installed.items())):
            try:
                inner_worker.unregister(handle)
            except BaseException:  # noqa: BLE001
                pass
            _unpublish_inner_handle(handle.digest)
        raise
    return installed


def _control_reply(
    conn: socket.socket,
    manifest: dict[str, Any],
    sequence: int,
    control_name: ControlName,
    version: int,
    error_code: int,
    error_message: str,
) -> None:
    session_id = int(manifest["session_id"])
    worker_id = int(manifest["worker_id"])
    payload = encode_control_reply(sequence, control_name, version, error_code, error_message)
    send_frame(conn, FrameHeader(FrameType.CONTROL_REPLY, session_id, worker_id, sequence), payload)


def _copy_command_header(data: bytes) -> tuple[int, int, int, int, int, bytes]:
    if len(data) < 36:
        raise ValueError("remote buffer copy command is truncated")
    worker_id, buffer_id, generation, offset, size = struct.unpack_from("<iQQQQ", data, 0)
    return int(worker_id), int(buffer_id), int(generation), int(offset), int(size), data[36:]


def _buffer_key(buffer_id: int, generation: int) -> tuple[int, int]:
    return int(buffer_id), int(generation)


def _tensor_with_data(tensor: Tensor, data: int) -> Tensor:
    return Tensor.make(int(data), tuple(tensor.shapes), tensor.dtype, bool(tensor.child_memory))


def _materialize_task_args(  # noqa: PLR0912
    args: RemoteTaskArgsWire, buffers: dict[tuple[int, ...], _RemoteBufferEntry], worker_id: int
) -> tuple[TaskArgs, list[Any]]:
    if len(args.remote_desc) != len(args.tensor_metadata):
        raise ValueError("remote TASK descriptor count does not match tensor metadata count")
    task_args = TaskArgs()
    keepalive: list[Any] = []

    for tensor, sidecar in zip(args.tensor_metadata, args.remote_desc):
        materialized = tensor
        if sidecar.present:
            desc = sidecar.desc
            if desc is None:
                raise ValueError("remote TASK descriptor is marked present but missing")
            if desc.nbytes != tensor.nbytes():
                raise ValueError("remote TASK descriptor nbytes does not match tensor metadata")
            if desc.address_space == RemoteAddressSpace.HOST_INLINE:
                start = int(desc.inline_payload_offset)
                end = start + int(desc.inline_payload_len)
                payload = args.inline_payload[start:end]
                if len(payload) != desc.inline_payload_len:
                    raise ValueError("HOST_INLINE payload range exceeds inline arena")
                buf = ctypes.create_string_buffer(payload, len(payload))
                keepalive.append(buf)
                materialized = _tensor_with_data(tensor, ctypes.addressof(buf))
            else:
                if desc.owner_worker_id == worker_id and desc.address_space == RemoteAddressSpace.REMOTE_DEVICE:
                    key = _buffer_key(desc.buffer_id, desc.generation)
                elif desc.address_space in (RemoteAddressSpace.REMOTE_WINDOW, RemoteAddressSpace.UB_LDST):
                    key = (desc.owner_worker_id, desc.buffer_id, desc.generation, desc.rkey_or_token)
                else:
                    raise ValueError(
                        "remote TASK descriptor names a different worker without an imported buffer handle"
                    )
                entry = buffers.get(key)
                if entry is None:
                    raise KeyError("remote TASK descriptor names an unknown or stale buffer/import generation")
                if entry.released:
                    raise ValueError("remote TASK descriptor references a released buffer/import")
                if desc.offset < entry.offset or desc.offset + desc.nbytes > entry.offset + entry.nbytes:
                    raise ValueError("remote TASK descriptor range exceeds buffer/import")
                materialized = _tensor_with_data(tensor, entry.addr + int(desc.offset - entry.offset))
        elif tensor.nbytes() != 0:
            raise ValueError("remote TASK tensor payload requires a RemoteTensorRef sidecar")
        task_args.add_tensor(materialized)

    for scalar in args.scalars:
        task_args.add_scalar(int(scalar))
    return task_args, keepalive


def _run_command_loop(  # noqa: PLR0912, PLR0915
    conn: socket.socket,
    manifest: dict[str, Any],
    inner_worker: Worker,
    manifest_inner_handles: dict[tuple[CallableKind, bytes], CallableHandle] | None = None,
    manifest_dispatch_registry: dict[bytes, Callable[..., Any]] | None = None,
) -> None:
    session_id = int(manifest["session_id"])
    worker_id = int(manifest["worker_id"])
    dispatch_registry: dict[bytes, Callable[..., Any]] = dict(manifest_dispatch_registry or {})
    prepared_dispatcher: dict[bytes, Callable[..., Any]] = {}
    prepared_inner: dict[tuple[CallableKind, bytes], Any] = {}
    inner_handles: dict[tuple[CallableKind, bytes], CallableHandle] = dict(manifest_inner_handles or {})
    next_buffer_id = 1
    next_export_id = 1
    next_import_id = 1
    buffers: dict[tuple[int, ...], _RemoteBufferEntry] = {}

    hello = HelloPayload(
        session_id=session_id,
        worker_id=worker_id,
        protocol_version=PROTOCOL_VERSION,
        comm_profile=str(manifest["transport"]),
        feature_flags=0,
        ready_state=ReadyState.READY,
    )
    send_frame(conn, FrameHeader(FrameType.HELLO, session_id, worker_id, 0), encode_hello(hello))

    try:
        while True:
            frame = read_frame(conn)
            header = frame.header
            if header.session_id != session_id or header.worker_id != worker_id:
                raise RuntimeError("remote session received mismatched session or worker frame")
            if header.frame_type == FrameType.SHUTDOWN:
                return
            if header.frame_type == FrameType.CONTROL:
                try:
                    control = decode_control(frame.payload)
                    if control.control_name == ControlName.PREPARE_REGISTER_CALLABLE:
                        digest, kind, registry, target = _prepare_register_callable(control.command_bytes, manifest)
                        if registry == RemoteRegistryTarget.REMOTE_TASK_DISPATCHER:
                            prepared_dispatcher[digest] = target
                        else:
                            prepared_inner[(kind, digest)] = target
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.COMMIT_REGISTER_CALLABLE:
                        digest, kind, registry = _decode_digest_command(control.command_bytes)
                        if registry == RemoteRegistryTarget.REMOTE_TASK_DISPATCHER:
                            if kind != CallableKind.PYTHON_IMPORT:
                                raise ValueError("COMMIT_REGISTER_CALLABLE target/kind mismatch")
                            if digest not in prepared_dispatcher:
                                raise KeyError("COMMIT_REGISTER_CALLABLE digest was not prepared")
                            dispatch_registry[digest] = prepared_dispatcher.pop(digest)
                        elif registry == RemoteRegistryTarget.INNER_L3_WORKER:
                            key = (kind, digest)
                            if key not in prepared_inner:
                                raise KeyError("COMMIT_REGISTER_CALLABLE inner digest was not prepared")
                            target = prepared_inner.pop(key)
                            if kind == CallableKind.PYTHON_IMPORT:
                                handle = inner_worker._register_child_python_import(target, digest=digest)  # noqa: SLF001
                            elif kind == CallableKind.CHIP_CALLABLE:
                                handle = inner_worker._register_child_chip(  # noqa: SLF001
                                    target,
                                    digest=digest,
                                    publish_handle=True,
                                )
                                if handle is None:
                                    raise RuntimeError("INNER_L3_WORKER CHIP_CALLABLE install did not publish a handle")
                            else:
                                raise ValueError("COMMIT_REGISTER_CALLABLE unsupported inner callable kind")
                            inner_handles[key] = handle
                            _publish_inner_handle(digest, handle)
                        else:
                            raise ValueError("COMMIT_REGISTER_CALLABLE target/kind mismatch")
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.ABORT_REGISTER_CALLABLE:
                        digest, kind, registry = _decode_digest_command(control.command_bytes)
                        if registry == RemoteRegistryTarget.REMOTE_TASK_DISPATCHER:
                            prepared_dispatcher.pop(digest, None)
                        elif registry == RemoteRegistryTarget.INNER_L3_WORKER:
                            prepared_inner.pop((kind, digest), None)
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.UNREGISTER_CALLABLE:
                        digest, kind, registry = _decode_digest_command(control.command_bytes)
                        if registry == RemoteRegistryTarget.REMOTE_TASK_DISPATCHER:
                            if kind != CallableKind.PYTHON_IMPORT:
                                raise ValueError("UNREGISTER_CALLABLE target/kind mismatch")
                            prepared_dispatcher.pop(digest, None)
                            dispatch_registry.pop(digest, None)
                        elif registry == RemoteRegistryTarget.INNER_L3_WORKER:
                            key = (kind, digest)
                            prepared_inner.pop(key, None)
                            handle = inner_handles.pop(key, None)
                            if handle is not None:
                                inner_worker.unregister(handle)
                                _unpublish_inner_handle(digest)
                        else:
                            raise ValueError("UNREGISTER_CALLABLE target/kind mismatch")
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.PREPARE_CALLABLE:
                        digest, kind, registry = _decode_digest_command(control.command_bytes)
                        if (
                            registry != RemoteRegistryTarget.REMOTE_TASK_DISPATCHER
                            or kind != CallableKind.PYTHON_IMPORT
                        ):
                            raise ValueError("PREPARE_CALLABLE target/kind mismatch")
                        if digest not in dispatch_registry:
                            raise KeyError("PREPARE_CALLABLE digest is not committed in dispatcher registry")
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.ALLOC_REMOTE_BUFFER:
                        if len(control.command_bytes) != 8:
                            raise ValueError("ALLOC_REMOTE_BUFFER payload must be uint64 nbytes")
                        (nbytes,) = struct.unpack("<Q", control.command_bytes)
                        if nbytes == 0:
                            raise ValueError("ALLOC_REMOTE_BUFFER nbytes must be non-zero")
                        buffer_id = next_buffer_id
                        next_buffer_id += 1
                        generation = 1
                        buf = shared_memory.SharedMemory(create=True, size=int(nbytes))
                        key = _buffer_key(buffer_id, generation)
                        entry = _RemoteBufferEntry(buf, int(nbytes), generation, RemoteAddressSpace.REMOTE_DEVICE)
                        buffers[key] = entry
                        remote_addr = entry.addr
                        result = struct.pack(
                            "<iQQiQQQQ",
                            worker_id,
                            buffer_id,
                            generation,
                            int(RemoteAddressSpace.REMOTE_DEVICE),
                            int(nbytes),
                            remote_addr,
                            0,
                            0,
                        )
                        payload = encode_control_reply(
                            header.sequence, control.control_name, control.control_version, 0, "", result
                        )
                        send_frame(
                            conn,
                            FrameHeader(FrameType.CONTROL_REPLY, session_id, worker_id, header.sequence),
                            payload,
                        )
                    elif control.control_name == ControlName.FREE_REMOTE_BUFFER:
                        if len(control.command_bytes) != 20:
                            raise ValueError("FREE_REMOTE_BUFFER payload must be worker_id, buffer_id, generation")
                        owner_worker_id, buffer_id, generation = struct.unpack("<iQQ", control.command_bytes)
                        if int(owner_worker_id) != worker_id:
                            raise ValueError("FREE_REMOTE_BUFFER worker mismatch")
                        key = _buffer_key(buffer_id, generation)
                        entry = buffers.get(key)
                        if entry is not None:
                            entry.released = True
                            buffers.pop(key, None)
                            entry.close(unlink=True)
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.COPY_TO_REMOTE:
                        owner_worker_id, buffer_id, generation, offset, size, data = _copy_command_header(
                            control.command_bytes
                        )
                        if owner_worker_id != worker_id:
                            raise ValueError("COPY_TO_REMOTE worker mismatch")
                        key = _buffer_key(buffer_id, generation)
                        entry = buffers.get(key)
                        if entry is None:
                            raise KeyError("COPY_TO_REMOTE names unknown buffer")
                        if entry.released:
                            raise ValueError("COPY_TO_REMOTE names released buffer")
                        if len(data) != size:
                            raise ValueError("COPY_TO_REMOTE payload size mismatch")
                        if offset + size > entry.nbytes:
                            raise ValueError("COPY_TO_REMOTE range exceeds buffer")
                        ctypes.memmove(entry.addr + offset, data, size)
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name == ControlName.COPY_FROM_REMOTE:
                        owner_worker_id, buffer_id, generation, offset, size, data = _copy_command_header(
                            control.command_bytes
                        )
                        if data:
                            raise ValueError("COPY_FROM_REMOTE request must not carry data bytes")
                        if owner_worker_id != worker_id:
                            raise ValueError("COPY_FROM_REMOTE worker mismatch")
                        key = _buffer_key(buffer_id, generation)
                        entry = buffers.get(key)
                        if entry is None:
                            raise KeyError("COPY_FROM_REMOTE names unknown buffer")
                        if entry.released:
                            raise ValueError("COPY_FROM_REMOTE names released buffer")
                        if offset + size > entry.nbytes:
                            raise ValueError("COPY_FROM_REMOTE range exceeds buffer")
                        result = ctypes.string_at(entry.addr + offset, size)
                        payload = encode_control_reply(
                            header.sequence, control.control_name, control.control_version, 0, "", result
                        )
                        send_frame(
                            conn,
                            FrameHeader(FrameType.CONTROL_REPLY, session_id, worker_id, header.sequence),
                            payload,
                        )
                    elif control.control_name == ControlName.EXPORT_BUFFER:
                        request = decode_export_buffer_request(control.command_bytes)
                        if request.owner_worker_id != worker_id:
                            raise ValueError("EXPORT_BUFFER worker mismatch")
                        key = _buffer_key(request.buffer_id, request.generation)
                        entry = buffers.get(key)
                        if entry is None:
                            raise KeyError("EXPORT_BUFFER names unknown buffer")
                        if entry.released:
                            raise ValueError("EXPORT_BUFFER names released buffer")
                        if request.offset + request.nbytes > entry.nbytes:
                            raise ValueError("EXPORT_BUFFER range exceeds buffer")
                        if request.transport_profile not in ("", "sim"):
                            raise ValueError("EXPORT_BUFFER transport_profile is not supported by sim")
                        export_id = next_export_id
                        next_export_id += 1
                        result = ExportBufferResult(
                            owner_worker_id=worker_id,
                            buffer_id=request.buffer_id,
                            generation=request.generation,
                            address_space=RemoteAddressSpace.REMOTE_WINDOW,
                            offset=request.offset,
                            nbytes=request.nbytes,
                            export_id=export_id,
                            remote_addr=entry.addr + request.offset,
                            rkey_or_token=export_id,
                            ub_ldst_va=0,
                            access_flags=request.access_flags,
                            transport_profile="sim",
                            transport_descriptor=entry.shm_name.encode("utf-8"),
                        )
                        payload = encode_control_reply(
                            header.sequence,
                            control.control_name,
                            control.control_version,
                            0,
                            "",
                            encode_export_buffer_result(result),
                        )
                        send_frame(
                            conn,
                            FrameHeader(FrameType.CONTROL_REPLY, session_id, worker_id, header.sequence),
                            payload,
                        )
                    elif control.control_name == ControlName.IMPORT_BUFFER:
                        request = decode_import_buffer_request(control.command_bytes)
                        if request.importer_worker_id != worker_id:
                            raise ValueError("IMPORT_BUFFER worker mismatch")
                        export_desc = request.export_desc
                        if export_desc.transport_profile != "sim":
                            raise ValueError("IMPORT_BUFFER transport_profile is not supported by sim")
                        shm_name = export_desc.transport_descriptor.decode("utf-8")
                        shm = shared_memory.SharedMemory(name=shm_name)
                        import_id = next_import_id
                        next_import_id += 1
                        key = (export_desc.owner_worker_id, export_desc.buffer_id, export_desc.generation, import_id)
                        buffers[key] = _RemoteBufferEntry(
                            shm,
                            int(export_desc.nbytes),
                            int(export_desc.generation),
                            export_desc.address_space,
                            offset=int(export_desc.offset),
                        )
                        result = ImportBufferResult(
                            importer_worker_id=worker_id,
                            owner_worker_id=export_desc.owner_worker_id,
                            buffer_id=export_desc.buffer_id,
                            generation=export_desc.generation,
                            import_id=import_id,
                            address_space=export_desc.address_space,
                            offset=export_desc.offset,
                            nbytes=export_desc.nbytes,
                            remote_addr=buffers[key].addr,
                            rkey_or_token=import_id,
                            ub_ldst_va=export_desc.ub_ldst_va,
                            access_flags=request.requested_access_flags,
                            transport_profile="sim",
                            import_descriptor=b"",
                        )
                        payload = encode_control_reply(
                            header.sequence,
                            control.control_name,
                            control.control_version,
                            0,
                            "",
                            encode_import_buffer_result(result),
                        )
                        send_frame(
                            conn,
                            FrameHeader(FrameType.CONTROL_REPLY, session_id, worker_id, header.sequence),
                            payload,
                        )
                    elif control.control_name == ControlName.RELEASE_IMPORT:
                        request = decode_release_import_request(control.command_bytes)
                        if request.importer_worker_id != worker_id:
                            raise ValueError("RELEASE_IMPORT worker mismatch")
                        key = (request.owner_worker_id, request.buffer_id, request.generation, request.import_id)
                        entry = buffers.get(key)
                        if entry is None:
                            raise KeyError("RELEASE_IMPORT names unknown import")
                        entry.released = True
                        buffers.pop(key, None)
                        entry.close(unlink=False)
                        _control_reply(
                            conn, manifest, header.sequence, control.control_name, control.control_version, 0, ""
                        )
                    elif control.control_name in (
                        ControlName.COMM_INIT,
                        ControlName.ALLOC_DOMAIN,
                        ControlName.RELEASE_DOMAIN,
                    ):
                        _control_reply(
                            conn,
                            manifest,
                            header.sequence,
                            control.control_name,
                            control.control_version,
                            1,
                            f"unsupported reserved remote domain control {control.control_name.name}",
                        )
                    else:
                        _control_reply(
                            conn,
                            manifest,
                            header.sequence,
                            control.control_name,
                            control.control_version,
                            1,
                            f"unsupported remote control {control.control_name.name}",
                        )
                except BaseException as exc:  # noqa: BLE001
                    try:
                        control = decode_control(frame.payload)
                        control_name = control.control_name
                        control_version = control.control_version
                    except BaseException:  # noqa: BLE001
                        control_name = ControlName.PREPARE_CALLABLE
                        control_version = 1
                    _control_reply(
                        conn,
                        manifest,
                        header.sequence,
                        control_name,
                        control_version,
                        1,
                        _format_remote_error(f"remote worker_id={worker_id} control sequence={header.sequence}", exc),
                    )
                continue
            if header.frame_type != FrameType.TASK:
                payload = encode_completion(
                    header.sequence, 1, f"unsupported remote frame type {int(header.frame_type)}"
                )
                send_frame(conn, FrameHeader(FrameType.COMPLETION, session_id, worker_id, header.sequence), payload)
                continue

            try:
                task = decode_task_payload(frame.payload)
                orch_fn = dispatch_registry.get(task.callable_digest)
                if orch_fn is None:
                    raise KeyError(f"remote TASK dispatcher has no callable hashid {task.callable_digest.hex()}")
                task_args, keepalive = _materialize_task_args(task.args, buffers, worker_id)
                try:
                    inner_worker.run(orch_fn, task_args, task.config)
                finally:
                    keepalive.clear()
                payload = encode_completion(header.sequence, 0, "")
            except BaseException as exc:  # noqa: BLE001
                payload = encode_completion(
                    header.sequence,
                    1,
                    _format_remote_error(
                        f"remote worker_id={worker_id} hashid={frame.payload[:32].hex()} sequence={header.sequence}",
                        exc,
                    ),
                )
            send_frame(conn, FrameHeader(FrameType.COMPLETION, session_id, worker_id, header.sequence), payload)
    finally:
        for key, entry in list(buffers.items()):
            entry.close(unlink=len(key) == 2)
        buffers.clear()
        with _INNER_HANDLE_LOCK:
            _INNER_HANDLES.clear()


def run_session(manifest: dict[str, Any], ready_fd: int) -> int:
    inner_worker = Worker(
        level=3,
        platform=str(manifest["platform"]),
        runtime=str(manifest.get("runtime", "tensormap_and_ringbuffer")),
        device_ids=tuple(int(x) for x in manifest.get("device_ids", [])),
        num_sub_workers=int(manifest.get("num_sub_workers", 0)),
        heap_ring_size=int(manifest["heap_ring_size"]) if manifest.get("heap_ring_size") is not None else None,
    )
    command_sock: socket.socket | None = None
    health_sock: socket.socket | None = None
    stop_health = threading.Event()
    health_thread: threading.Thread | None = None
    try:
        session_timeout_s = _session_timeout_s(manifest)
        startup_remaining_s = _startup_remaining_s(manifest)
        manifest_dispatch_registry = _install_manifest_dispatcher_registry(manifest)
        # Register the inner L3 callables before init() so they are frozen into
        # the eager startup snapshot and uploaded to the chip children before
        # those children publish INIT_READY. init() is the single, eager startup
        # point: it forks and readies the whole inner L3->L2 tree, so the runner
        # reports ready (below) only once that subtree is up.
        manifest_inner_handles = _install_manifest_inner_registry(manifest, inner_worker)
        # Bound the inner startup by the parent's remaining startup budget
        # (rebuilt against this host's own monotonic clock — the parent's
        # absolute deadline is not comparable across machines), not a fresh full
        # session_timeout_s. Mark it non-root so its chip/sub children inherit
        # this runner's process group (see start_new_session in the daemon)
        # rather than splitting into their own — the daemon reaps the whole
        # L3->L2 subtree with one killpg on the runner.
        inner_worker.init(_startup_deadline=time.monotonic() + startup_remaining_s)

        listen_host = str(manifest.get("listen_host", "127.0.0.1"))
        command_sock = _bind_listener(listen_host)
        health_sock = _bind_listener(listen_host)
        health_thread = threading.Thread(
            target=_health_loop,
            args=(health_sock, stop_health, int(manifest["session_id"]), int(manifest["worker_id"])),
            daemon=True,
        )
        health_thread.start()

        command_port = int(command_sock.getsockname()[1])
        health_port = int(health_sock.getsockname()[1])
        _send_ready(
            ready_fd,
            {
                "ok": True,
                "command_host": str(manifest.get("connect_host", listen_host)),
                "command_port": command_port,
                "health_host": str(manifest.get("connect_host", listen_host)),
                "health_port": health_port,
            },
        )

        command_sock.settimeout(session_timeout_s)
        conn, _addr = command_sock.accept()
        with conn:
            _run_command_loop(conn, manifest, inner_worker, manifest_inner_handles, manifest_dispatch_registry)
        return 0
    except BaseException as exc:  # noqa: BLE001
        try:
            _send_ready(ready_fd, {"ok": False, "error": _format_remote_error("remote session startup", exc)})
        except OSError:
            pass
        return 1
    finally:
        stop_health.set()
        if command_sock is not None:
            try:
                command_sock.close()
            except OSError:
                pass
        if health_thread is not None:
            health_thread.join(timeout=1.0)
        try:
            inner_worker.close()
        except BaseException:  # noqa: BLE001
            pass


def _raise_keyboard_interrupt(_signum, _frame):
    # A daemon cooperative-kill (SIGTERM) unwinds run_session so its finally
    # closes the inner Worker — reaping the L3->L2 subtree — before exit.
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ready-fd", required=True, type=int)
    ns = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    with open(ns.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    return run_session(manifest, ns.ready_fd)


if __name__ == "__main__":
    sys.exit(main())
