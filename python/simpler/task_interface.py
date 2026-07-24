# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLW0603, PLC0415
"""Public Python API for task_interface nanobind bindings.

Re-exports the canonical C++ types (DataType, Tensor, ChipStorageTaskArgs,
TaskArgs, TensorArgType) plus ``scalar_to_uint64``. Torch-aware helpers
(``make_tensor_arg``, ``torch_dtype_to_datatype``) live in
``simpler_setup.torch_interop`` — this module has no torch dependency.

Usage:
    from simpler.task_interface import DataType, Tensor, ChipStorageTaskArgs
    from simpler_setup.torch_interop import make_tensor_arg
"""

from __future__ import annotations

import ctypes
import os
import threading
import uuid
import weakref
from dataclasses import dataclass
from enum import IntEnum
from math import prod
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_OFF_PROTOCOL,
    MAILBOX_PROTOCOL_MAGIC_VERSION,
    MAILBOX_SIZE,
    MAILBOX_TASK_SLOT_COUNT,
    MAILBOX_TASK_SLOT_SIZE,
    MAX_REGISTERED_CALLABLE_IDS,
    MAX_TENSOR_DIMS,
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    CoreCallable,
    DataType,
    RuntimeEnv,
    TaskArgs,
    TaskState,
    Tensor,
    TensorArgType,
    WorkerType,
    _ChipWorker,
    _Worker,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
    read_args_from_blob,
)

__all__ = [
    "DataType",
    "get_element_size",
    "get_dtype_name",
    "MAX_TENSOR_DIMS",
    "Tensor",
    "ChipStorageTaskArgs",
    "TensorArgType",
    "TaskArgs",
    "RemoteAddressSpace",
    "RemoteBufferHandle",
    "RemoteBufferExport",
    "RemoteTensorRef",
    "ArgDirection",
    "CoreCallable",
    "ChipCallable",
    "CallConfig",
    "RuntimeEnv",
    "ChipWorker",
    "arg_direction_name",
    "scalar_to_uint64",
    # Distributed runtime
    "WorkerType",
    "TaskState",
    "_Worker",
    "MAILBOX_SIZE",
    "MAILBOX_TASK_SLOT_SIZE",
    "MAILBOX_TASK_SLOT_COUNT",
    "MAILBOX_OFF_ERROR_MSG",
    "MAILBOX_OFF_PROTOCOL",
    "MAILBOX_PROTOCOL_MAGIC_VERSION",
    "MAILBOX_ERROR_MSG_SIZE",
    "read_args_from_blob",
    # Dynamic CommDomain allocation (orch-only API)
    "CommBufferSpec",
    "ChipDomainContext",
    "CommDomainHandle",
]

COMM_MAX_RANK_NUM = 64


class RemoteAddressSpace(IntEnum):
    HOST_INLINE = 1
    REMOTE_DEVICE = 2
    REMOTE_WINDOW = 3
    UB_LDST = 4


_REMOTE_BUFFER_ACCESS_READ = 1 << 0
_REMOTE_BUFFER_ACCESS_WRITE = 1 << 1
_REMOTE_BUFFER_ACCESS_READ_WRITE = _REMOTE_BUFFER_ACCESS_READ | _REMOTE_BUFFER_ACCESS_WRITE
_REMOTE_BUFFER_HANDLE_TOKEN = object()
_REMOTE_BUFFER_EXPORT_TOKEN = object()


class RemoteBufferHandle:
    __slots__ = (
        "_worker_id",
        "_owner_worker_id",
        "_buffer_id",
        "_generation",
        "_import_id",
        "_address_space",
        "_nbytes",
        "_offset",
        "_remote_addr",
        "_rkey_or_token",
        "_ub_ldst_va",
        "_access_flags",
        "_released",
        "_live_slot_refs",
        "_live_import_refs",
        "_owner_handle_ref",
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        worker_id: int,
        owner_worker_id: int | None = None,
        buffer_id: int,
        generation: int,
        import_id: int = 0,
        address_space: RemoteAddressSpace = RemoteAddressSpace.REMOTE_DEVICE,
        nbytes: int = 0,
        offset: int = 0,
        remote_addr: int = 0,
        rkey_or_token: int = 0,
        ub_ldst_va: int = 0,
        access_flags: int = 3,
        released: bool = False,
        owner_handle_ref: RemoteBufferHandle | None = None,
        _internal_token: object | None = None,
    ) -> None:
        address_space = RemoteAddressSpace(int(address_space))
        if _internal_token is not _REMOTE_BUFFER_HANDLE_TOKEN:
            raise TypeError("RemoteBufferHandle values are returned by Worker.remote_malloc/import")

        self._worker_id = int(worker_id)
        self._owner_worker_id = int(worker_id if owner_worker_id is None else owner_worker_id)
        self._buffer_id = int(buffer_id)
        self._generation = int(generation)
        self._import_id = int(import_id)
        self._address_space = address_space
        self._nbytes = int(nbytes)
        self._offset = int(offset)
        self._remote_addr = int(remote_addr)
        self._rkey_or_token = int(rkey_or_token)
        self._ub_ldst_va = int(ub_ldst_va)
        self._access_flags = int(access_flags)
        self._released = bool(released)
        self._live_slot_refs = 0
        self._live_import_refs = 0
        self._owner_handle_ref = owner_handle_ref

        if self._worker_id < 0:
            raise ValueError("RemoteBufferHandle.worker_id must be non-negative")
        if self._owner_worker_id < 0:
            raise ValueError("RemoteBufferHandle.owner_worker_id must be non-negative")
        if self._buffer_id < 0 or self._generation < 0 or self._import_id < 0:
            raise ValueError("RemoteBufferHandle ids must be non-negative")
        if self._nbytes < 0:
            raise ValueError("RemoteBufferHandle.nbytes must be non-negative")
        if self._offset < 0:
            raise ValueError("RemoteBufferHandle.offset must be non-negative")
        if self._address_space != RemoteAddressSpace.HOST_INLINE and self._buffer_id == 0:
            raise ValueError("RemoteBufferHandle.buffer_id must be non-zero for remote buffers")
        if self._address_space == RemoteAddressSpace.REMOTE_DEVICE and self._worker_id != self._owner_worker_id:
            raise ValueError("REMOTE_DEVICE handles must be consumed on their owner worker")
        if (
            self._address_space in (RemoteAddressSpace.REMOTE_WINDOW, RemoteAddressSpace.UB_LDST)
            and self._import_id == 0
        ):
            raise ValueError("imported remote handles require a non-zero import_id")
        if self._access_flags & ~0x3:
            raise ValueError("RemoteBufferHandle.access_flags contains unknown bits")

    @classmethod
    def _from_remote_allocation(
        cls,
        *,
        worker_id: int,
        buffer_id: int,
        generation: int,
        address_space: RemoteAddressSpace,
        nbytes: int,
        remote_addr: int = 0,
        rkey_or_token: int = 0,
        ub_ldst_va: int = 0,
        released: bool = False,
    ) -> RemoteBufferHandle:
        return cls(
            worker_id=worker_id,
            owner_worker_id=worker_id,
            buffer_id=buffer_id,
            generation=generation,
            import_id=0,
            address_space=address_space,
            nbytes=nbytes,
            offset=0,
            remote_addr=remote_addr,
            rkey_or_token=rkey_or_token,
            ub_ldst_va=ub_ldst_va,
            access_flags=3,
            released=released,
            _internal_token=_REMOTE_BUFFER_HANDLE_TOKEN,
        )

    @classmethod
    def _from_imported_mapping(  # noqa: PLR0913
        cls,
        *,
        worker_id: int,
        owner_worker_id: int,
        buffer_id: int,
        generation: int,
        import_id: int,
        address_space: RemoteAddressSpace,
        nbytes: int,
        offset: int,
        remote_addr: int = 0,
        rkey_or_token: int = 0,
        ub_ldst_va: int = 0,
        access_flags: int = 0,
        released: bool = False,
        owner_handle_ref: RemoteBufferHandle | None = None,
    ) -> RemoteBufferHandle:
        return cls(
            worker_id=worker_id,
            owner_worker_id=owner_worker_id,
            buffer_id=buffer_id,
            generation=generation,
            import_id=import_id,
            address_space=address_space,
            nbytes=nbytes,
            offset=offset,
            remote_addr=remote_addr,
            rkey_or_token=rkey_or_token,
            ub_ldst_va=ub_ldst_va,
            access_flags=access_flags,
            released=released,
            owner_handle_ref=owner_handle_ref,
            _internal_token=_REMOTE_BUFFER_HANDLE_TOKEN,
        )

    @property
    def worker_id(self) -> int:
        return self._worker_id

    @property
    def owner_worker_id(self) -> int:
        return self._owner_worker_id

    @property
    def import_id(self) -> int:
        return self._import_id

    @property
    def address_space(self) -> RemoteAddressSpace:
        return self._address_space

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def released(self) -> bool:
        return self._released

    @property
    def access_flags(self) -> int:
        return self._access_flags

    @property
    def is_imported(self) -> bool:
        return self._import_id != 0

    def _mark_released(self) -> None:
        self._released = True

    def _acquire_slot_ref(self) -> None:
        if self._released:
            raise RuntimeError("RemoteBufferHandle has already been released")
        self._live_slot_refs += 1

    def _release_slot_ref(self) -> None:
        if self._live_slot_refs <= 0:
            raise RuntimeError("RemoteBufferHandle live slot refs underflow")
        self._live_slot_refs -= 1

    def _acquire_import_ref(self) -> None:
        if self._released:
            raise RuntimeError("RemoteBufferHandle has already been released")
        self._live_import_refs += 1

    def _release_import_ref(self) -> None:
        if self._live_import_refs <= 0:
            raise RuntimeError("RemoteBufferHandle live import refs underflow")
        self._live_import_refs -= 1

    def __repr__(self) -> str:
        return (
            "RemoteBufferHandle("
            f"worker_id={self.worker_id}, owner_worker_id={self.owner_worker_id}, "
            f"address_space={self.address_space.name}, nbytes={self.nbytes}, released={self.released})"
        )


class RemoteBufferExport:
    """Opaque descriptor returned by ``Worker.remote_export``.

    The transport fields are intentionally kept private so callers cannot forge
    or log remote keys by accidentally treating the export as a plain dataclass.
    """

    __slots__ = (
        "_owner_worker_id",
        "_buffer_id",
        "_generation",
        "_address_space",
        "_offset",
        "_nbytes",
        "_export_id",
        "_remote_addr",
        "_rkey_or_token",
        "_ub_ldst_va",
        "_access_flags",
        "_transport_profile",
        "_transport_descriptor",
        "_owner_handle",
        "_worker_owner_id",
        "_sealed",
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        owner_worker_id: int,
        buffer_id: int,
        generation: int,
        address_space: RemoteAddressSpace,
        offset: int,
        nbytes: int,
        export_id: int,
        remote_addr: int,
        rkey_or_token: int,
        ub_ldst_va: int,
        access_flags: int,
        transport_profile: str,
        transport_descriptor: bytes = b"",
        _owner_handle: RemoteBufferHandle | None = None,
        _worker_owner_id: str | None = None,
        _internal_token: object | None = None,
    ) -> None:
        if _internal_token is not _REMOTE_BUFFER_EXPORT_TOKEN:
            raise TypeError("RemoteBufferExport values are returned by Worker.remote_export")
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_owner_worker_id", int(owner_worker_id))
        object.__setattr__(self, "_buffer_id", int(buffer_id))
        object.__setattr__(self, "_generation", int(generation))
        object.__setattr__(self, "_address_space", RemoteAddressSpace(int(address_space)))
        object.__setattr__(self, "_offset", int(offset))
        object.__setattr__(self, "_nbytes", int(nbytes))
        object.__setattr__(self, "_export_id", int(export_id))
        object.__setattr__(self, "_remote_addr", int(remote_addr))
        object.__setattr__(self, "_rkey_or_token", int(rkey_or_token))
        object.__setattr__(self, "_ub_ldst_va", int(ub_ldst_va))
        object.__setattr__(self, "_access_flags", int(access_flags))
        object.__setattr__(self, "_transport_profile", str(transport_profile))
        object.__setattr__(self, "_transport_descriptor", bytes(transport_descriptor))
        object.__setattr__(self, "_owner_handle", _owner_handle)
        object.__setattr__(self, "_worker_owner_id", None if _worker_owner_id is None else str(_worker_owner_id))

        for name in (
            "_owner_worker_id",
            "_buffer_id",
            "_generation",
            "_offset",
            "_nbytes",
            "_export_id",
            "_remote_addr",
            "_rkey_or_token",
            "_ub_ldst_va",
            "_access_flags",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"RemoteBufferExport.{name[1:]} must be non-negative")
        if self._owner_worker_id < 0 or self._buffer_id == 0 or self._generation == 0 or self._export_id == 0:
            raise ValueError("RemoteBufferExport requires live owner buffer identity and export_id")
        if self._nbytes <= 0:
            raise ValueError("RemoteBufferExport.nbytes must be positive")
        if self._address_space not in (RemoteAddressSpace.REMOTE_WINDOW, RemoteAddressSpace.UB_LDST):
            raise ValueError("RemoteBufferExport address_space must be REMOTE_WINDOW or UB_LDST")
        if self._access_flags == 0 or self._access_flags & ~_REMOTE_BUFFER_ACCESS_READ_WRITE:
            raise ValueError("RemoteBufferExport.access_flags must use read/write bits")
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def _from_remote_export(  # noqa: PLR0913
        cls,
        *,
        owner_worker_id: int,
        buffer_id: int,
        generation: int,
        address_space: RemoteAddressSpace,
        offset: int,
        nbytes: int,
        export_id: int,
        remote_addr: int,
        rkey_or_token: int,
        ub_ldst_va: int,
        access_flags: int,
        transport_profile: str,
        transport_descriptor: bytes = b"",
        _owner_handle: RemoteBufferHandle | None = None,
        worker_owner_id: str | None = None,
    ) -> RemoteBufferExport:
        return cls(
            owner_worker_id=owner_worker_id,
            buffer_id=buffer_id,
            generation=generation,
            address_space=address_space,
            offset=offset,
            nbytes=nbytes,
            export_id=export_id,
            remote_addr=remote_addr,
            rkey_or_token=rkey_or_token,
            ub_ldst_va=ub_ldst_va,
            access_flags=access_flags,
            transport_profile=transport_profile,
            transport_descriptor=transport_descriptor,
            _owner_handle=_owner_handle,
            _worker_owner_id=worker_owner_id,
            _internal_token=_REMOTE_BUFFER_EXPORT_TOKEN,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("RemoteBufferExport is immutable")
        object.__setattr__(self, name, value)

    @property
    def owner_worker_id(self) -> int:
        return self._owner_worker_id

    @property
    def address_space(self) -> RemoteAddressSpace:
        return self._address_space

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def access_flags(self) -> int:
        return self._access_flags

    @property
    def transport_profile(self) -> str:
        return self._transport_profile

    def __repr__(self) -> str:
        return (
            "RemoteBufferExport("
            f"owner_worker_id={self.owner_worker_id}, address_space={self.address_space.name}, "
            f"offset={self.offset}, nbytes={self.nbytes}, access_flags={self.access_flags}, "
            f"transport_profile={self.transport_profile!r})"
        )


@dataclass(frozen=True)
class _RemoteTensorDesc:
    address_space: RemoteAddressSpace
    owner_worker_id: int = -1
    buffer_id: int = 0
    offset: int = 0
    nbytes: int = 0
    remote_addr: int = 0
    rkey_or_token: int = 0
    generation: int = 0
    inline_payload_offset: int = 0
    inline_payload_len: int = 0
    flags: int = 0


@dataclass(frozen=True)
class _RemoteTensorSidecar:
    present: bool
    desc: _RemoteTensorDesc
    handle: RemoteBufferHandle | None = None


@dataclass(frozen=True)
class _RemoteTaskArgsSidecar:
    tensors: tuple[_RemoteTensorSidecar | None, ...] = ()
    inline_payload: bytes = b""


@dataclass(frozen=True)
class RemoteTensorRef:
    handle: RemoteBufferHandle
    offset: int = 0
    shape: tuple[int, ...] = ()
    dtype: DataType = DataType.FLOAT32
    nbytes: int | None = None
    inline_payload: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.handle, RemoteBufferHandle):
            raise TypeError("RemoteTensorRef.handle must be a RemoteBufferHandle")
        shape = tuple(int(x) for x in self.shape)
        if any(x < 0 for x in shape):
            raise ValueError("RemoteTensorRef.shape entries must be non-negative")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "offset", int(self.offset))
        if self.offset < 0:
            raise ValueError("RemoteTensorRef.offset must be non-negative")
        nbytes = _remote_tensor_nbytes(shape, self.dtype) if self.nbytes is None else int(self.nbytes)
        object.__setattr__(self, "nbytes", nbytes)
        payload = bytes(self.inline_payload)
        object.__setattr__(self, "inline_payload", payload)
        if nbytes < 0:
            raise ValueError("RemoteTensorRef.nbytes must be non-negative")
        if self.handle.address_space == RemoteAddressSpace.HOST_INLINE:
            if len(payload) != nbytes:
                raise ValueError("HOST_INLINE payload length must match RemoteTensorRef.nbytes")
        elif payload:
            raise ValueError("inline_payload is only valid for HOST_INLINE RemoteTensorRef")
        if self.handle.nbytes and self.offset + nbytes > self.handle.nbytes:
            raise ValueError("RemoteTensorRef range exceeds RemoteBufferHandle.nbytes")
        if self.handle.released:
            raise ValueError("RemoteTensorRef cannot reference a released RemoteBufferHandle")

    @classmethod
    def host_inline(cls, payload: bytes, *, shape: tuple[int, ...], dtype: DataType) -> RemoteTensorRef:
        data = bytes(payload)
        shape_tuple = tuple(int(x) for x in shape)
        if any(x < 0 for x in shape_tuple):
            raise ValueError("RemoteTensorRef.shape entries must be non-negative")
        expected_nbytes = _remote_tensor_nbytes(shape_tuple, dtype)
        if len(data) != expected_nbytes:
            raise ValueError("HOST_INLINE payload length must match shape*dtype size")
        handle = RemoteBufferHandle(
            worker_id=0,
            owner_worker_id=0,
            buffer_id=0,
            generation=0,
            address_space=RemoteAddressSpace.HOST_INLINE,
            nbytes=expected_nbytes,
            _internal_token=_REMOTE_BUFFER_HANDLE_TOKEN,
        )
        return cls(handle=handle, offset=0, shape=shape_tuple, dtype=dtype, nbytes=expected_nbytes, inline_payload=data)


@dataclass
class _RemoteTaskArgsStorage:
    sidecars: list[_RemoteTensorSidecar | None]
    inline_payload: bytearray


_TASK_ARGS_ADD_TENSOR = TaskArgs.add_tensor
_TASK_ARGS_CLEAR = TaskArgs.clear
_REMOTE_TASK_ARGS_STORAGE: weakref.WeakKeyDictionary[TaskArgs, _RemoteTaskArgsStorage] = weakref.WeakKeyDictionary()
_REMOTE_TASK_ARGS_STORAGE_LOCK = threading.Lock()


def _sidecar_from_ref(storage: _RemoteTaskArgsStorage, ref: RemoteTensorRef) -> _RemoteTensorSidecar:
    handle = ref.handle
    inline_offset = 0
    inline_len = 0
    if handle.address_space == RemoteAddressSpace.HOST_INLINE:
        inline_offset = len(storage.inline_payload)
        inline_len = len(ref.inline_payload)
        storage.inline_payload.extend(ref.inline_payload)
    nbytes = ref.nbytes
    assert nbytes is not None

    desc = _RemoteTensorDesc(
        address_space=handle.address_space,
        owner_worker_id=0 if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle.owner_worker_id,
        buffer_id=0 if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle._buffer_id,
        offset=0 if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle._offset + ref.offset,
        nbytes=int(nbytes),
        remote_addr=0 if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle._remote_addr,
        rkey_or_token=0 if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle._rkey_or_token,
        generation=0 if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle._generation,
        inline_payload_offset=inline_offset,
        inline_payload_len=inline_len,
        flags=0,
    )
    handle_ref = None if handle.address_space == RemoteAddressSpace.HOST_INLINE else handle
    return _RemoteTensorSidecar(True, desc, handle_ref)


def _storage_for_remote_task_args(args: TaskArgs) -> _RemoteTaskArgsStorage:
    with _REMOTE_TASK_ARGS_STORAGE_LOCK:
        storage = _REMOTE_TASK_ARGS_STORAGE.get(args)
        if storage is None or len(storage.sidecars) != args.tensor_count():
            storage = _RemoteTaskArgsStorage([None for _ in range(args.tensor_count())], bytearray())
            _REMOTE_TASK_ARGS_STORAGE[args] = storage
        return storage


def _task_args_add_tensor(
    self: TaskArgs, tensor: Tensor | RemoteTensorRef, tag: TensorArgType = TensorArgType.INPUT
) -> None:
    if isinstance(tensor, RemoteTensorRef):
        storage = _storage_for_remote_task_args(self)
        metadata = Tensor.make(0, tensor.shape, tensor.dtype)
        _TASK_ARGS_ADD_TENSOR(self, metadata, tag)
        storage.sidecars.append(_sidecar_from_ref(storage, tensor))
        return
    if not isinstance(tensor, Tensor):
        raise TypeError("TaskArgs.add_tensor expects Tensor or RemoteTensorRef")
    _TASK_ARGS_ADD_TENSOR(self, tensor, tag)
    with _REMOTE_TASK_ARGS_STORAGE_LOCK:
        storage = _REMOTE_TASK_ARGS_STORAGE.get(self)
        if storage is not None:
            storage.sidecars.append(None)


def _task_args_clear(self: TaskArgs) -> None:
    _TASK_ARGS_CLEAR(self)
    with _REMOTE_TASK_ARGS_STORAGE_LOCK:
        _REMOTE_TASK_ARGS_STORAGE.pop(self, None)


TaskArgs.add_tensor = _task_args_add_tensor
TaskArgs.clear = _task_args_clear


def _remote_tensor_nbytes(shape: tuple[int, ...], dtype: DataType) -> int:
    element_count = int(prod(shape)) if shape else 1
    return element_count * int(get_element_size(dtype))


def _empty_remote_sidecar_for(args: TaskArgs) -> _RemoteTaskArgsSidecar:
    return _RemoteTaskArgsSidecar(tuple(None for _ in range(args.tensor_count())), b"")


def _remote_sidecar_for(args: TaskArgs) -> _RemoteTaskArgsSidecar | None:
    with _REMOTE_TASK_ARGS_STORAGE_LOCK:
        storage = _REMOTE_TASK_ARGS_STORAGE.get(args)
        if storage is None:
            return None
        if len(storage.sidecars) != args.tensor_count():
            _REMOTE_TASK_ARGS_STORAGE.pop(args, None)
            return None
        return _RemoteTaskArgsSidecar(tuple(storage.sidecars), bytes(storage.inline_payload))


def _remote_access_label(flags: int) -> str:
    flags = int(flags)
    if flags == _REMOTE_BUFFER_ACCESS_READ:
        return "read"
    if flags == _REMOTE_BUFFER_ACCESS_WRITE:
        return "write"
    if flags == _REMOTE_BUFFER_ACCESS_READ_WRITE:
        return "readwrite"
    return f"0x{flags:x}"


def _required_remote_access_for_tag(tag: TensorArgType) -> int:
    if tag == TensorArgType.INPUT:
        return _REMOTE_BUFFER_ACCESS_READ
    if tag in (TensorArgType.OUTPUT, TensorArgType.OUTPUT_EXISTING):
        return _REMOTE_BUFFER_ACCESS_WRITE
    if tag in (TensorArgType.INOUT, TensorArgType.NO_DEP):
        return _REMOTE_BUFFER_ACCESS_READ_WRITE
    raise ValueError(f"unsupported TensorArgType for remote tensor: {tag!r}")


def _validate_remote_sidecar_access(args: TaskArgs, remote_sidecar: _RemoteTaskArgsSidecar | None) -> None:
    if remote_sidecar is None:
        return
    tensor_count = int(args.tensor_count())
    if len(remote_sidecar.tensors) != tensor_count:
        raise ValueError("remote tensor sidecar count does not match TaskArgs tensor count")

    for idx, tensor_sidecar in enumerate(remote_sidecar.tensors):
        if tensor_sidecar is None or not tensor_sidecar.present:
            continue
        tag = args.tag(idx)
        required = _required_remote_access_for_tag(tag)
        desc = tensor_sidecar.desc
        if RemoteAddressSpace(int(desc.address_space)) == RemoteAddressSpace.HOST_INLINE:
            granted = _REMOTE_BUFFER_ACCESS_READ
        else:
            handle = tensor_sidecar.handle
            if not isinstance(handle, RemoteBufferHandle):
                raise TypeError(f"remote tensor {idx} sidecar handle must be a RemoteBufferHandle")
            if handle.released:
                raise ValueError(f"remote tensor {idx} references a released RemoteBufferHandle")
            granted = int(handle.access_flags)
        if required & ~granted:
            tag_name = getattr(tag, "name", str(tag))
            raise ValueError(
                f"remote tensor {idx} tag {tag_name} requires {_remote_access_label(required)} access; "
                f"handle grants {_remote_access_label(granted)}"
            )


class _CommContextStruct(ctypes.Structure):
    _fields_ = [
        ("workSpace", ctypes.c_uint64),
        ("workSpaceSize", ctypes.c_uint64),
        ("rankId", ctypes.c_uint32),
        ("rankNum", ctypes.c_uint32),
        ("winSize", ctypes.c_uint64),
        ("windowsIn", ctypes.c_uint64 * COMM_MAX_RANK_NUM),
        ("windowsOut", ctypes.c_uint64 * COMM_MAX_RANK_NUM),
    ]


assert ctypes.sizeof(_CommContextStruct) == 1056


def scalar_to_uint64(value) -> int:
    """Convert a scalar value to ``uint64``.

    *value* can be a Python int, float, a ctypes scalar (``c_int64``,
    ``c_float``, etc.), or any object convertible to ``int``.

    Python float values are converted to IEEE 754 single precision (32-bit)
    and their bit pattern is zero-extended to uint64. This may cause a loss of
    precision. For double precision, use ``ctypes.c_double``.
    """
    import struct as _struct

    if isinstance(value, float):
        bits = _struct.unpack("<I", _struct.pack("<f", value))[0]
        return bits
    import ctypes as _ct

    if isinstance(value, _ct._SimpleCData):
        if isinstance(value, (_ct.c_float, _ct.c_double)):
            uint_type = _ct.c_uint32 if isinstance(value, _ct.c_float) else _ct.c_uint64
            return uint_type.from_buffer_copy(value).value
        return int(value.value) & 0xFFFFFFFFFFFFFFFF
    return int(value) & 0xFFFFFFFFFFFFFFFF


@dataclass
class CommBufferSpec:
    """A named slice of the per-rank communicator window.

    Buffers are placed sequentially inside the window in declaration order —
    Buffers are placed sequentially inside the window in declaration order.
    The ``CommDomainHandle.contexts[chip_idx].buffer_ptrs`` dict returned by
    ``Orchestrator.allocate_domain`` is keyed by ``CommBufferSpec.name``.
    """

    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False


@dataclass
class ChipDomainContext:
    name: str
    domain_rank: int
    domain_size: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: dict[str, int]


class CommDomainHandle:
    """User-facing handle for one dynamically-allocated CommDomain.

    Returned by ``Orchestrator.allocate_domain(...)``.  Acts as a context
    manager: ``with`` exit *marks* the handle for release and prevents
    further use; the actual backend free runs **after** ``Worker.run`` has
    drained any tasks the orch function submitted using this domain.  This
    is required because ``submit_*`` only enqueues to the DAG — freeing
    before drain would create a use-after-free on the chip side.

    Lifecycle states::

        live           — allocated, indexable, can be passed to submit_*
        released       — release() called; further indexing raises;
                          backend memory still alive until Worker.run drain
        freed          — backend release_domain has executed, memory gone

    Most users only see ``released``; the ``live → released`` transition
    happens at ``with`` exit (or explicit ``release()``), and the
    ``released → freed`` transition is the runtime's job at end-of-run.
    """

    __slots__ = ("name", "workers", "contexts", "allocation_id", "_release_fn", "_released", "_freed")

    def __init__(
        self,
        *,
        name: str,
        workers: tuple[int, ...],
        contexts: dict[int, ChipDomainContext],
        allocation_id: int,
        _release_fn,
    ) -> None:
        self.name = name
        self.workers = tuple(workers)
        # Frozen dict-ish — we don't expose mutation
        self.contexts: dict[int, ChipDomainContext] = dict(contexts)
        self.allocation_id = int(allocation_id)
        self._release_fn = _release_fn
        self._released = False
        self._freed = False

    def __getitem__(self, chip_idx: int) -> ChipDomainContext:
        if self._released:
            raise RuntimeError(
                f"CommDomainHandle({self.name!r}) already released; do not pass it to submit_* "
                "after release(). Submitted tasks that captured device_ctx / buffer_ptrs before "
                "release will still see live memory until Worker.run drains."
            )
        return self.contexts[chip_idx]

    @property
    def released(self) -> bool:
        """True once ``release()`` (or ``with`` exit) has been called.

        Backend memory may still be alive — it is freed by the Worker after
        DAG drain at end-of-run.  Use this to gate further indexing /
        submission, not to assert physical teardown (use ``freed`` for that).
        """
        return self._released

    @property
    def freed(self) -> bool:
        """True once the backend ``comm_release_domain_windows`` has executed.

        Only flips after the owning ``Worker.run`` drains and processes the
        pending-release queue.  An ``orch_fn`` will never observe ``True``
        for a handle it released within the same ``run`` call.
        """
        return self._freed

    def release(self) -> None:
        """Mark this handle for collective release.  Idempotent.

        Inside an orch function, this is a non-blocking mark — the actual
        backend ``comm_release_domain_windows`` runs after
        ``Worker.run.drain()`` so that any tasks already submitted with
        this domain's ``device_ctx`` see live memory through execution.

        After this returns, the handle is treated as released for the
        user's purposes: ``__getitem__`` raises, repeated ``release()`` is
        a no-op, and the orch function must not pass it to further
        ``submit_*`` calls.
        """
        if self._released:
            return
        self._released = True
        # _release_fn is owned by Worker; it queues the actual backend
        # release and runs it after drain.  Worker also flips _freed.
        self._release_fn(self)

    def __enter__(self) -> CommDomainHandle:
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self) -> str:
        if self._freed:
            state = "freed"
        elif self._released:
            state = "released-pending-free"
        else:
            state = "live"
        return f"CommDomainHandle(name={self.name!r}, workers={self.workers}, {state})"


# Process-wide RTLD_GLOBAL preload registry. host_runtime.so resolves its
# undefined HostLogger / unified_log_* (and, on sim, sim_context_*) symbols
# against these globals, so they must be loaded — exactly once — before any
# host_runtime.so dlopen. Keyed by path; mirrors the C++ side's old
# std::once_flag semantics. Never closed.
_preloaded_globals: dict[str, ctypes.CDLL] = {}


def _preload_global(path: str) -> ctypes.CDLL:
    """dlopen `path` with RTLD_NOW | RTLD_GLOBAL, idempotently (one CDLL per path).

    Eager resolution (RTLD_NOW) mirrors the previous C++ dlopen flags and
    surfaces any missing-symbol problem at load time rather than first use.
    """
    handle = _preloaded_globals.get(path)
    if handle is None:
        handle = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        _preloaded_globals[path] = handle
    return handle


class ChipWorker:
    """Unified execution interface wrapping the host runtime C API.

    The runtime library and target device are bound once via init() and
    cannot be changed.
    Public dispatch uses opaque ``CallableHandle`` values. Integer execution
    slots are private to this wrapper and the runtime ABI.

    Usage::

        worker = ChipWorker()
        worker.init(device_id=0, bins=bins)
        handle = worker.register_callable(chip_callable)
        worker.run(handle, args=orch_args, config=CallConfig())  # block_dim defaults to 0 = auto
        worker.unregister_callable(handle)
        worker.finalize()
    """

    def __init__(self):
        self._impl = _ChipWorker()
        self._owner_id = uuid.uuid4().hex
        self._registry_lock = threading.Lock()
        self._callable_registry: dict[int, ChipCallable] = {}
        self._identity_registry: dict[bytes, Any] = {}
        self._live_handles: dict[int, bytes] = {}
        self._next_handle_id = 0

    def init(self, device_id, bins, log_level=None, log_info_v=None, prewarm_config=None, enable_sdma=False):
        """Attach the calling thread to ``device_id``, load the host runtime
        library, and cache platform binaries.

        Can only be called once — the runtime and device cannot be changed
        after init.

        Performs the process-wide RTLD_GLOBAL bootstrap (libsimpler_log.so,
        plus libcpu_sim_context.so on sim platforms) and seeds the HostLogger
        via ``simpler_log_init`` *before* the C++ ``_ChipWorker.init`` dlopens
        host_runtime.so — host_runtime.so resolves its undefined HostLogger /
        unified_log_* (and, on sim, sim_context_*) symbols against those
        globals, and any LOG_* macro firing during its dlopen-time
        constructors must already see the right filter.

        Args:
            device_id: NPU device ID to attach the calling thread to.
            bins: A `simpler_setup.runtime_builder.RuntimeBinaries` (or any
                object exposing host_path / aicpu_path / aicore_path /
                simpler_log_path / sim_context_path / dispatcher_path).
                ``dispatcher_path`` is required for onboard platforms and
                ignored on sim (set to None).
            log_level: Severity floor (0=DEBUG..4=NUL). Defaults to a snapshot
                of the simpler logger via `_log.get_current_config()`.
            log_info_v: INFO verbosity threshold (0..9). Same default.

        For tests that need to drive the binding directly with arbitrary path
        strings (e.g. to assert dlopen failure on `/nonexistent/foo.so`), call
        `_ChipWorker.init(...)` from `_task_interface` instead of going
        through this wrapper.
        """
        if log_level is None or log_info_v is None:
            from . import _log  # noqa: PLC0415

            sev, info_v = _log.get_current_config()
            if log_level is None:
                log_level = sev
            if log_info_v is None:
                log_info_v = info_v

        # 1. libsimpler_log.so — RTLD_GLOBAL singleton, before host_runtime.so.
        if not bins.simpler_log_path:
            raise ValueError("ChipWorker.init: bins.simpler_log_path is required")
        log_handle = _preload_global(str(bins.simpler_log_path))
        log_handle.simpler_log_init.argtypes = [ctypes.c_int, ctypes.c_int]
        log_handle.simpler_log_init.restype = ctypes.c_int
        rc = log_handle.simpler_log_init(int(log_level), int(log_info_v))
        if rc != 0:
            raise RuntimeError(f"simpler_log_init failed with code {rc}")

        # 2. libcpu_sim_context.so — sim platforms only (host_runtime.so's sim
        #    variant resolves sim_context_set_* / pto_sim_get_* against it).
        if bins.sim_context_path:
            _preload_global(str(bins.sim_context_path))

        # 3. host_runtime.so is dlopen'd RTLD_LOCAL inside _impl.init.
        #    dispatcher_path is passed as an empty string on sim (where bins
        #    has dispatcher_path=None); the onboard simpler_init reads it
        #    via LoadAicpuOp::BootstrapDispatcher, sim ignores it.
        dispatcher_path = getattr(bins, "dispatcher_path", None)
        self._impl.init(
            str(bins.host_path),
            str(bins.aicpu_path),
            str(bins.aicore_path),
            "" if dispatcher_path is None else str(dispatcher_path),
            int(device_id),
            prewarm_config,
            bool(enable_sdma),
        )
        for slot_id, callable_obj in list(self._callable_registry.items()):
            self._impl.register_callable(int(slot_id), callable_obj)

    def finalize(self):
        """Tear down everything: device resources and runtime library.

        Terminal operation — the object cannot be reused after this.
        """
        try:
            self._impl.finalize()
        finally:
            with self._registry_lock:
                self._callable_registry.clear()
                self._identity_registry.clear()
                self._live_handles.clear()

    @property
    def pipeline_slot_count(self) -> int:
        """Pipeline depth declared by the initialized runtime."""
        return int(self._impl.pipeline_slot_count)

    @property
    def arena_bank_count(self) -> int:
        """Number of independently filled runtime arena banks."""
        return int(self._impl.arena_bank_count)

    def _allocate_slot_locked(self) -> int:
        for slot_id in range(MAX_REGISTERED_CALLABLE_IDS):
            if slot_id not in self._callable_registry:
                return slot_id
        raise RuntimeError(
            "ChipWorker.register_callable: callable capacity exhausted "
            f"(MAX_REGISTERED_CALLABLE_IDS={MAX_REGISTERED_CALLABLE_IDS})"
        )

    def _make_handle_locked(self, state):
        from .callable_identity import CallableHandle  # noqa: PLC0415

        handle_id = self._next_handle_id
        self._next_handle_id += 1
        self._live_handles[handle_id] = state.digest
        return CallableHandle._from_registration(
            hashid=state.hashid,
            kind=state.kind,
            target_namespace=state.target_namespace,
            handle_id=handle_id,
            owner_id=self._owner_id,
        )

    def _rollback_handle_locked(self, handle) -> None:
        state = self._identity_registry.get(handle.digest)
        self._live_handles.pop(handle._handle_id, None)
        if state is None:
            return
        state.ref_count -= 1
        if state.ref_count > 0:
            return
        self._callable_registry.pop(state.slot_id, None)
        self._identity_registry.pop(state.digest, None)

    def _resolve_handle_locked(self, handle):
        from .callable_identity import CallableHandle  # noqa: PLC0415

        if not isinstance(handle, CallableHandle):
            raise TypeError("ChipWorker.run expects a CallableHandle returned by ChipWorker.register_callable")
        if handle._owner_id != self._owner_id:
            raise KeyError(f"CallableHandle {handle.hashid} does not belong to this ChipWorker")
        digest = self._live_handles.get(handle._handle_id)
        if digest is None or digest != handle.digest:
            raise KeyError(f"CallableHandle {handle.hashid} is not live on this ChipWorker")
        state = self._identity_registry.get(digest)
        if state is None:
            raise KeyError(f"CallableHandle {handle.hashid} is not registered")
        if (
            handle.hashid != state.hashid
            or handle.kind != state.kind
            or handle.target_namespace != state.target_namespace
        ):
            raise RuntimeError(f"CALLABLE_HANDLE_MUTATED: {handle.hashid}")
        return state

    def _resolve_handle(self, handle):
        with self._registry_lock:
            return self._resolve_handle_locked(handle)

    def register_callable(self, callable):
        """Prepare a ``ChipCallable`` and return an opaque handle.

        The runtime still uses an integer slot internally, but the caller never
        chooses or observes it.
        """
        if not isinstance(callable, ChipCallable):
            raise TypeError("ChipWorker.register_callable only supports ChipCallable targets")
        from .callable_identity import (  # noqa: PLC0415
            _CallableIdentityState,
            build_chip_callable_descriptor,
            compute_callable_hashid,
            hashid_to_digest,
        )

        descriptor = build_chip_callable_descriptor(target=callable)
        hashid = compute_callable_hashid(descriptor)
        digest = hashid_to_digest(hashid)
        with self._registry_lock:
            state = self._identity_registry.get(digest)
            if state is not None:
                if state.descriptor != descriptor or state.kind != "CHIP_CALLABLE":
                    raise RuntimeError(f"HASHID_DESCRIPTOR_MISMATCH: {hashid}")
                state.ref_count += 1
                return self._make_handle_locked(state)
            slot_id = self._allocate_slot_locked()
            state = _CallableIdentityState(
                hashid=hashid,
                digest=digest,
                kind="CHIP_CALLABLE",
                target_namespace="LOCAL_CHIP",
                descriptor=descriptor,
                payload_digest=descriptor,
                slot_id=slot_id,
                target=callable,
                ref_count=1,
            )
            self._identity_registry[digest] = state
            self._callable_registry[slot_id] = callable
            handle = self._make_handle_locked(state)

        if self.initialized:
            try:
                self._impl.register_callable(int(slot_id), callable)
            except Exception:
                with self._registry_lock:
                    self._rollback_handle_locked(handle)
                raise
        return handle

    def run(self, handle, args, config=None, **kwargs):
        """Launch a callable previously returned by ``register_callable``.

        Args:
            handle: ``CallableHandle`` returned by ``register_callable``.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional CallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g. ``block_dim=8`` to
                pin a smaller value than the default). Omit ``block_dim`` (or
                set it to 0) to have DeviceRunner auto-resolve it to the max
                the AICore stream allows (``aclrtGetStreamResLimit`` on
                onboard, ``PLATFORM_MAX_BLOCKDIM`` on sim).

        Returns ``None``. Per-stage run timing is emitted as ``[STRACE]`` log
        markers by the platform — see ``docs/dfx/host-trace.md``.
        """
        state = self._resolve_handle(handle)
        self._run_slot(state.slot_id, args, config, **kwargs)

    def unregister_callable(self, handle) -> None:
        """Drop one live callable handle and release its private resources when final."""
        with self._registry_lock:
            state = self._resolve_handle_locked(handle)
            self._live_handles.pop(handle._handle_id, None)
            state.ref_count -= 1
            if state.ref_count > 0:
                return
            slot_id = state.slot_id
            self._callable_registry.pop(slot_id, None)
            self._identity_registry.pop(state.digest, None)

        if self.initialized:
            self._impl.unregister_callable(int(slot_id))

    def _register_callable_at_slot(self, callable_id, callable):
        self._impl.register_callable(int(callable_id), callable)

    def _run_slot(self, callable_id, args, config=None, **kwargs):
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        # Returns None; per-stage timing is emitted as `[STRACE]` log markers.
        self._impl.run(int(callable_id), args, config)

    def _unregister_slot(self, callable_id):
        self._impl.unregister_callable(int(callable_id))

    @property
    def aicpu_dlopen_count(self):
        """Number of distinct callable identities the AICPU has dlopened for."""
        return self._impl.aicpu_dlopen_count

    @property
    def host_dlopen_count(self):
        """Number of host-side orch SO dlopens (host_build_graph variants)."""
        return self._impl.host_dlopen_count

    def malloc(self, size):
        """Allocate memory. Returns a pointer (uint64)."""
        return int(self._impl.malloc(int(size)))

    def free(self, ptr):
        """Free memory allocated by ``malloc()``."""
        self._impl.free(int(ptr))

    def copy_to(self, dst, src, size):
        """Copy *size* bytes from host *src* to worker *dst*."""
        self._impl.copy_to(int(dst), int(src), int(size))

    def copy_from(self, dst, src, size):
        """Copy *size* bytes from worker *src* to host *dst*."""
        self._impl.copy_from(int(dst), int(src), int(size))

    def comm_init(self, rank: int, nranks: int, rootinfo_path: str) -> int:
        """Initialize a distributed communicator for this rank.

        ChipWorker owns ACL bring-up and the aclrtStream internally, so
        callers never touch ``aclInit`` / ``aclrtSetDevice`` / stream
        lifetimes.  On sim, ACL / stream are not used.  Pair with
        ``comm_destroy`` for teardown.

        Args:
            rank: This process's rank (0-based).
            nranks: Total number of ranks.
            rootinfo_path: Filesystem path used for rank handshake.

        Returns:
            Opaque communicator handle (uint64) for the other ``comm_*`` calls.
        """
        return int(self._impl.comm_init(int(rank), int(nranks), str(rootinfo_path)))

    def comm_alloc_windows(self, comm_handle: int, win_size: int) -> int:
        """Allocate per-rank windows. Returns a device CommContext pointer (uint64)."""
        return int(self._impl.comm_alloc_windows(int(comm_handle), int(win_size)))

    def comm_get_local_window_base(self, comm_handle: int) -> int:
        """Return this rank's local window base address (uint64)."""
        return int(self._impl.comm_get_local_window_base(int(comm_handle)))

    def comm_get_window_size(self, comm_handle: int) -> int:
        """Return the actual per-rank window size in bytes."""
        return int(self._impl.comm_get_window_size(int(comm_handle)))

    def comm_derive_context(
        self,
        comm_handle: int,
        rank_ids: list[int],
        domain_rank: int,
        window_offset: int,
        window_size: int,
    ) -> int:
        """Derive a domain-local device CommContext from an allocated base communicator."""
        return int(
            self._impl.comm_derive_context(
                int(comm_handle),
                [int(x) for x in rank_ids],
                int(domain_rank),
                int(window_offset),
                int(window_size),
            )
        )

    def comm_barrier(self, comm_handle: int) -> None:
        """Synchronize all ranks."""
        self._impl.comm_barrier(int(comm_handle))

    def comm_destroy(self, comm_handle: int) -> None:
        """Destroy the communicator and release its resources."""
        self._impl.comm_destroy(int(comm_handle))

    def comm_destroy_all(self) -> None:
        """Destroy all communicators owned by this worker."""
        self._impl.comm_destroy_all()

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized
