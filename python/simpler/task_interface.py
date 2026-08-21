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

Re-exports the canonical C++ types (DataType, ChipTensor, ChipStorageTaskArgs, TaskArgs,
TensorArgType) plus ``scalar_to_uint64``, and re-exports the address-free ``Tensor`` — the task
argument users build — from ``simpler.buffer``. Torch-aware helpers (``make_chip_tensor_arg``,
``torch_dtype_to_datatype``) live in ``simpler_setup.torch_interop`` — this module has no torch
dependency.

``ChipTensor`` is the chip-only POD the runtime ABI expects, paired with
``ChipStorageTaskArgs`` on the direct ``ChipWorker`` path; it carries a
materialized address and never crosses a process boundary.

Usage:
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType
    from simpler_setup.torch_interop import make_chip_tensor_arg
"""

from __future__ import annotations

import ctypes
import threading
import uuid
import weakref
from dataclasses import dataclass
from enum import IntEnum
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only: `CallableHandle` is imported lazily at its use site, and
    # PEP 563 keeps these annotations as strings, so nothing is imported at
    # runtime.
    from .callable_identity import CallableHandle

import _task_interface as _ti_module  # pyright: ignore[reportMissingImports]
from _task_interface import (  # pyright: ignore[reportMissingImports]
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_FRAME_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_PREPARATION_DISPOSITION_VALUES,
    MAILBOX_SIZE,
    MAILBOX_STATE_VALUES,
    MAX_REGISTERED_CALLABLE_IDS,
    MAX_TENSOR_DIMS,
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipTensor,
    CoreCallable,
    DataType,
    RuntimeEnv,
    TaskArgs,
    TaskState,
    TensorArgType,
    WorkerType,
    _ChipWorker,
    _Worker,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
    read_args_from_blob,
)
from _task_interface import (
    _initialize_host_log as _native_initialize_host_log,
)

from .buffer import Buffer, Tensor


def _assert_bindings_match_source_tree() -> None:
    """Refuse a `_task_interface` built from a different revision of this tree.

    An editable install pins the compiled extension at install time
    (``editable.rebuild = false``) while this file is read live, so switching
    branches or rebasing moves the Python source out from under a fixed binary.
    Nothing then rebuilds, and a changed struct layout — `CallConfig` losing a
    field, say — makes attributes read as 0 with no error at all. That surfaces
    much later as a plausible-looking runtime rejection
    (``launch_aicpu_num (1) must be 0 (auto) or in range [2, 4]``) and reads as a product
    bug, so it is worth one git call at import to stop.

    Only source-tree installs are checked: a wheel has no ``.git`` to compare
    against, and a build with no git available carries an empty stamp.

    A *missing* stamp is not the same as an empty one. The attribute only
    disappears on an extension compiled before it existed, which in a checkout
    new enough to run this function is by definition a different revision — the
    exact case this guards, and the one every already-installed worktree is in
    the moment this lands. Treating it like "cannot tell" would let precisely
    those through.
    """
    import subprocess  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return
    if not hasattr(_ti_module, "__build_commit__"):
        raise ImportError(
            "_task_interface predates the build stamp, so it was compiled before the "
            "revision you are running and its struct layouts may not match the Python "
            "that drives them — fields can read as 0 with no error.\n"
            "Rebuild:  pip install --no-build-isolation -e ."
        )
    built_from: str = _ti_module.__build_commit__
    if not built_from:
        return
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return
    if result.returncode != 0:
        return
    head = result.stdout.strip()
    if not head or head == built_from:
        return
    raise ImportError(
        f"_task_interface was built from {built_from[:12]}, but this source tree is at "
        f"{head[:12]}. The compiled extension does not rebuild on import "
        f"(editable.rebuild = false), so its struct layouts may no longer match the "
        f"Python that drives them — fields can read as 0 with no error.\n"
        f"Rebuild:  pip install --no-build-isolation -e ."
    )


_assert_bindings_match_source_tree()

from .global_comm_domain import GlobalDomainAttachment, GlobalDomainBuffer, GlobalDomainMember  # noqa: E402

__all__ = [
    "DataType",
    "get_element_size",
    "get_dtype_name",
    "MAX_TENSOR_DIMS",
    "Tensor",
    "ChipTensor",
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
    "MAILBOX_FRAME_SIZE",
    "MAILBOX_OFF_ERROR_MSG",
    "MAILBOX_ERROR_MSG_SIZE",
    "MAILBOX_STATE_VALUES",
    "MAILBOX_PREPARATION_DISPOSITION_VALUES",
    "read_args_from_blob",
    # Dynamic CommDomain allocation (orch-only API)
    "CommBufferSpec",
    "ChipDomainContext",
    "CommDomainHandle",
    "GlobalCommDomainHandle",
    "GlobalCommDomainView",
]

COMM_MAX_RANK_NUM = 64


class RemoteAddressSpace(IntEnum):
    """How a remote buffer's bytes are reached.

    ``HOST_INLINE`` carries the payload in the message itself rather than
    naming remote memory. ``REMOTE_WINDOW`` and ``UB_LDST`` are protocol
    placeholders: the shipped host_tcp transport uses host-side session buffers.
    """

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
    """A reference to memory on a remote L3 worker.

    Returned by ``Worker.remote_malloc`` (an *owner* handle) or by
    ``Worker.remote_import`` (an *imported* handle, told apart by
    ``is_imported``). The two are not interchangeable: owner handles are freed
    with ``remote_free``, imported ones with ``remote_release_import``.

    ``RemoteTensorRef.host_inline`` produces a third form, with
    ``address_space`` of ``HOST_INLINE``: it carries its bytes in the message
    and names no remote allocation, so neither release call applies —
    ``remote_free`` rejects it outright and it is never ``is_imported``.

    Construct only through ``Worker`` or ``RemoteTensorRef.host_inline``; the
    constructor is token-guarded.
    """

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
        "_slot_ref_tokens",
        "_import_ref_tokens",
        "_owner_handle_ref",
        "_owner_import_ref_token",
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
        owner_import_ref_token: object | None = None,
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
        self._slot_ref_tokens: set[object] = set()
        self._import_ref_tokens: set[object] = set()
        self._owner_handle_ref = owner_handle_ref
        self._owner_import_ref_token = owner_import_ref_token

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
        owner_import_ref_token: object | None = None,
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
            owner_import_ref_token=owner_import_ref_token,
            _internal_token=_REMOTE_BUFFER_HANDLE_TOKEN,
        )

    @property
    def worker_id(self) -> int:
        """Worker holding this reference — the importer, for an imported handle."""
        return self._worker_id

    @property
    def owner_worker_id(self) -> int:
        """Worker that owns the underlying allocation."""
        return self._owner_worker_id

    @property
    def import_id(self) -> int:
        """Nonzero on an imported handle; ``0`` on an owner handle."""
        return self._import_id

    @property
    def address_space(self) -> RemoteAddressSpace:
        """How these bytes are reached; see ``RemoteAddressSpace``."""
        return self._address_space

    @property
    def nbytes(self) -> int:
        """Size of the allocation in bytes, or of the payload for ``HOST_INLINE``."""
        return self._nbytes

    @property
    def released(self) -> bool:
        """Whether the handle has been freed or released."""
        return self._released

    @property
    def access_flags(self) -> int:
        """Permitted access as a read/write bitmask; an export may only narrow it."""
        return self._access_flags

    @property
    def is_imported(self) -> bool:
        """Whether this came from ``remote_import`` rather than ``remote_malloc``."""
        return self._import_id != 0

    @property
    def _live_slot_refs(self) -> int:
        return len(self._slot_ref_tokens)

    @property
    def _live_import_refs(self) -> int:
        return len(self._import_ref_tokens)

    def _mark_released(self) -> None:
        self._released = True

    def _acquire_slot_ref(self, token: object | None = None) -> object:
        if self._released:
            raise RuntimeError("RemoteBufferHandle has already been released")
        if token is None:
            token = object()
        self._slot_ref_tokens.add(token)
        return token

    def _release_slot_ref(self, token: object | None = None) -> None:
        if token is not None:
            self._slot_ref_tokens.discard(token)
            return
        if not self._slot_ref_tokens:
            raise RuntimeError("RemoteBufferHandle live slot refs underflow")
        self._slot_ref_tokens.pop()

    def _acquire_import_ref(self, token: object | None = None) -> object:
        if self._released:
            raise RuntimeError("RemoteBufferHandle has already been released")
        if token is None:
            token = object()
        self._import_ref_tokens.add(token)
        return token

    def _release_import_ref(self, token: object | None = None) -> None:
        if token is not None:
            self._import_ref_tokens.discard(token)
            return
        if not self._import_ref_tokens:
            raise RuntimeError("RemoteBufferHandle live import refs underflow")
        self._import_ref_tokens.pop()

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
        """Worker that owns the exported allocation."""
        return self._owner_worker_id

    @property
    def address_space(self) -> RemoteAddressSpace:
        """How the exported bytes are reached."""
        return self._address_space

    @property
    def offset(self) -> int:
        """Start of the exported range within the owner buffer."""
        return self._offset

    @property
    def nbytes(self) -> int:
        """Length of the exported range in bytes."""
        return self._nbytes

    @property
    def access_flags(self) -> int:
        """Access granted here; a subset of the owner handle's flags."""
        return self._access_flags

    @property
    def transport_profile(self) -> str:
        """Transport this export was minted for."""
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
    """A tensor argument that lives on, or travels to, a remote worker."""

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
        """Build a reference whose payload travels inline, naming no remote memory.

        ``payload`` length must equal the byte size implied by ``shape`` and
        ``dtype``, and shape entries must be non-negative.
        """
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


def _task_args_add_tensor(self: TaskArgs, tensor, tag: TensorArgType = TensorArgType.INPUT) -> None:
    """Add a task arg. ``tensor`` is a ``simpler.buffer.Tensor`` (packable) or its packed
    bytes. A RemoteTensorRef (arg destined for a remote worker) is rewritten to a REMOTE_SIDECAR
    ``Tensor`` (no local backing) with its remote descriptor tracked in the sidecar."""
    if isinstance(tensor, RemoteTensorRef):
        from .buffer import AddressSpace, remote_sidecar_tensor

        storage = _storage_for_remote_task_args(self)
        handle = tensor.handle
        inline = handle.address_space == RemoteAddressSpace.HOST_INLINE
        nbytes = tensor.nbytes
        assert nbytes is not None
        placeholder = remote_sidecar_tensor(
            shapes=tuple(int(s) for s in tensor.shape),
            dtype=int(tensor.dtype.value),
            # A remote placeholder is still a normal Tensor record on the wire: its descriptor
            # spans the complete handle backing, while the sidecar records this view's nbytes.
            nbytes=int(handle.nbytes),
            owner_worker_id=0 if inline else int(handle.owner_worker_id),
            buffer_id=0 if inline else int(handle._buffer_id),
            generation=0 if inline else int(handle._generation),
            address_space=(
                AddressSpace.DEVICE if handle.address_space == RemoteAddressSpace.REMOTE_DEVICE else AddressSpace.HOST
            ),
            byte_offset=int(tensor.offset),
        )
        _TASK_ARGS_ADD_TENSOR(self, placeholder, tag)
        storage.sidecars.append(_sidecar_from_ref(storage, tensor))
        return
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

    Buffers are placed sequentially inside the window in declaration order.
    The ``CommDomainHandle.contexts[chip_idx].buffers`` dict returned by
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
    """Per-domain view handed to a chip worker: its rank within the domain and
    the local slice of the symmetric window.
    """

    name: str
    domain_rank: int
    domain_size: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    # Each named window slice as a device ``VMM_WINDOW`` Buffer owned by this chip. Name a task
    # arg with ``buffers[name].tensor(shapes, dtype)`` and dispatch it only to this chip (``domain_rank``).
    buffers: dict[str, Buffer]


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

    __slots__ = (
        "name",
        "workers",
        "contexts",
        "allocation_id",
        "_domain_size",
        "_domain_ranks",
        "_release_fn",
        "_released",
        "_freed",
    )

    def __init__(
        self,
        *,
        name: str,
        workers: tuple[int, ...],
        contexts: dict[int, ChipDomainContext],
        allocation_id: int,
        _release_fn,
        _domain_size: int | None = None,
        _domain_ranks: dict[int, int] | None = None,
    ) -> None:
        self.name = name
        self.workers = tuple(workers)
        # Frozen dict-ish — we don't expose mutation
        self.contexts: dict[int, ChipDomainContext] = dict(contexts)
        self.allocation_id = int(allocation_id)
        self._domain_size = len(self.workers) if _domain_size is None else int(_domain_size)
        self._domain_ranks = (
            {worker: rank for rank, worker in enumerate(self.workers)} if _domain_ranks is None else dict(_domain_ranks)
        )
        self._release_fn = _release_fn
        self._released = False
        self._freed = False

    def __getitem__(self, chip_idx: int) -> ChipDomainContext:
        if self._released:
            raise RuntimeError(
                f"CommDomainHandle({self.name!r}) already released; do not pass it to submit_* "
                "after release(). Submitted tasks that captured device_ctx / buffers before"
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

        Only flips after the owning ``Worker.run`` completes and processes the
        pending-release queue.  An ``orch_fn`` will never observe ``True``
        for a handle it released within the same ``run`` call.
        """
        return self._freed

    def release(self) -> None:
        """Mark this handle for collective release.  Idempotent.

        Inside an orch function, this is a non-blocking mark — the actual
        backend ``comm_release_domain_windows`` runs after
        the owning run's completion wait so that tasks already submitted with
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
        # release and runs it after the owning run completes. Worker also flips _freed.
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


class GlobalCommDomainHandle:
    """L4-owned handle for one CommDomain spanning local and/or remote L3 nodes.

    The handle contains stable topology, attachment metadata, and buffer
    offsets only. Device addresses remain in the L3/L2 process that imported
    the transport handles.
    """

    __slots__ = (
        "_freed",
        "_release_fn",
        "_released",
        "attachments",
        "buffers",
        "domain_id",
        "generation",
        "mapping_size",
        "members",
        "name",
        "retain_after_run",
    )

    def __init__(
        self,
        *,
        name: str,
        members: tuple[GlobalDomainMember, ...],
        buffers: tuple[GlobalDomainBuffer, ...],
        domain_id: int,
        generation: int,
        mapping_size: int,
        retain_after_run: bool,
        _release_fn,
        attachments: tuple[GlobalDomainAttachment, ...] = (),
    ) -> None:
        self.name = str(name)
        self.members = tuple(members)
        self.buffers = tuple(buffers)
        self.attachments = tuple(attachments)
        self.domain_id = int(domain_id)
        self.generation = int(generation)
        self.mapping_size = int(mapping_size)
        self.retain_after_run = bool(retain_after_run)
        self._release_fn = _release_fn
        self._released = False
        self._freed = False

    def member(self, domain_rank: int) -> GlobalDomainMember:
        if self._released:
            raise RuntimeError(f"GlobalCommDomainHandle({self.name!r}) is already released")
        rank = int(domain_rank)
        if rank < 0 or rank >= len(self.members):
            raise IndexError(f"global domain rank {rank} is out of range")
        member = self.members[rank]
        if member.domain_rank != rank:
            raise RuntimeError("global domain member table is not rank ordered")
        return member

    def buffer_range(self, name: str) -> tuple[int, int]:
        if self._released:
            raise RuntimeError(f"GlobalCommDomainHandle({self.name!r}) is already released")
        offset = 0
        for buffer in self.buffers:
            if buffer.name == name:
                return offset, buffer.nbytes
            offset += buffer.nbytes
        raise KeyError(f"global domain {self.name!r} has no buffer {name!r}")

    @property
    def released(self) -> bool:
        return self._released

    @property
    def freed(self) -> bool:
        return self._freed

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._release_fn(self)

    def __enter__(self) -> GlobalCommDomainHandle:
        return self

    def __exit__(self, *_):
        self.release()


class GlobalCommDomainView:
    """L3-local imported view and its receiving-node attachment row."""

    __slots__ = (
        "_committed",
        "attachments",
        "contexts",
        "domain_id",
        "generation",
        "mapping_size",
        "members",
        "name",
    )

    def __init__(
        self,
        *,
        name: str,
        members: tuple[GlobalDomainMember, ...],
        contexts: dict[int, ChipDomainContext],
        domain_id: int,
        generation: int,
        mapping_size: int,
        attachments: tuple[GlobalDomainAttachment, ...] = (),
    ) -> None:
        self.name = str(name)
        self.members = tuple(members)
        self.attachments = tuple(attachments)
        self.contexts = dict(contexts)
        self.domain_id = int(domain_id)
        self.generation = int(generation)
        self.mapping_size = int(mapping_size)
        self._committed = False

    def __getitem__(self, local_worker_id: int) -> ChipDomainContext:
        if not self._committed:
            raise RuntimeError(f"GlobalCommDomainView({self.name!r}) is not committed")
        return self.contexts[int(local_worker_id)]

    @property
    def committed(self) -> bool:
        return self._committed


def _initialize_host_log(log_level: int | None = None) -> None:
    """Seed the extension-owned host-log state before runtime use or fork."""
    if log_level is None:
        from . import _log  # noqa: PLC0415

        log_level = _log.get_current_config()
    if not _native_initialize_host_log(int(log_level)):
        raise ValueError(f"unsupported simpler log threshold: {log_level}")


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
        worker.run(handle, args=orch_args, config=CallConfig())
        worker.unregister_callable(handle)
        worker.finalize()
    """

    def __init__(self):
        self._impl = _ChipWorker()
        self._owner_id = uuid.uuid4().hex
        self._lifecycle_lock = threading.Lock()
        self._init_owner_thread: threading.Thread | None = None
        self._init_in_progress = False
        self._registry_lock = threading.Lock()
        self._callable_registry: dict[int, ChipCallable] = {}
        self._identity_registry: dict[bytes, Any] = {}
        self._live_handles: dict[int, bytes] = {}
        self._next_handle_id = 0

    def init(
        self,
        device_id: int,
        # Structurally typed: any object exposing the *_path attributes below.
        # Not RuntimeBinaries — that lives in simpler_setup, which this package
        # must not depend on.
        bins: Any,
        log_level: int | None = None,
        prewarm_config: CallConfig | None = None,
        enable_sdma: bool = False,
    ):
        """Attach the calling thread to ``device_id``, load the host runtime
        library, and cache platform binaries.

        Can only be called once — the runtime and device cannot be changed
        after init.

        Seeds the extension-owned HostLogger state before C++ loads any
        consumer. Each consumer contains its own logger implementation and
        receives that state pointer during module init. On sim, C++ retains
        libcpu_sim_context.so in a process-wide RTLD_GLOBAL registry so
        host_runtime.so can resolve the PTO simulator hooks.

        Args:
            device_id: NPU device ID to attach the calling thread to.
            bins: A `simpler_setup.runtime_builder.RuntimeBinaries` (or any
                object exposing host_path / aicpu_path / aicore_path /
                sim_context_path / dispatcher_path / sdma_warmup_path).
                ``dispatcher_path`` is required for onboard platforms and
                ignored on sim (set to None). ``sdma_warmup_path`` is optional
                everywhere: without it the first TPREFETCH_ASYNC pays the cold
                SDMA control path instead of init absorbing it.
            log_level: Threshold (10=DEBUG, 20=INFO, 25=TIMING, 30=WARN,
                40=ERROR, 60=NUL). Defaults to a snapshot of the simpler
                logger via `_log.get_current_config()`.

        For tests that need to drive the binding directly with arbitrary path
        strings (e.g. to assert dlopen failure on `/nonexistent/foo.so`), call
        `_ChipWorker.init(...)` from `_task_interface` instead of going
        through this wrapper.
        """
        with self._lifecycle_lock:
            if self._init_in_progress:
                raise RuntimeError("ChipWorker.init() is already in progress")
            if self._impl.initialized:
                raise RuntimeError("ChipWorker is already initialized")
            self._init_owner_thread = threading.current_thread()
            self._init_in_progress = True

        try:
            _initialize_host_log(log_level)

            # C++ retains libcpu_sim_context.so in the sim process registry,
            # loads host_runtime.so, and binds both private logger copies.
            # dispatcher_path is empty on sim; onboard consumes the real path.
            dispatcher_path = getattr(bins, "dispatcher_path", None)
            sim_context_path = getattr(bins, "sim_context_path", None)
            sdma_warmup_path = getattr(bins, "sdma_warmup_path", None)
            self._impl.init(
                str(bins.host_path),
                str(bins.aicpu_path),
                str(bins.aicore_path),
                "" if dispatcher_path is None else str(dispatcher_path),
                int(device_id),
                prewarm_config,
                bool(enable_sdma),
                "" if sim_context_path is None else str(sim_context_path),
                "" if sdma_warmup_path is None else str(sdma_warmup_path),
            )
            for slot_id, callable_obj in list(self._callable_registry.items()):
                self._impl.register_callable(int(slot_id), callable_obj)
        finally:
            with self._lifecycle_lock:
                self._init_in_progress = False

    def finalize(self):
        """Tear down everything: device resources and runtime library.

        Terminal operation — the object cannot be reused after this.
        """
        with self._lifecycle_lock:
            owner = self._init_owner_thread
            if owner is not None and owner is not threading.current_thread():
                raise RuntimeError("ChipWorker.finalize() must run on the thread that called ChipWorker.init()")
            if self._init_in_progress:
                raise RuntimeError("ChipWorker.finalize() cannot run while ChipWorker.init() is in progress")
        try:
            self._impl.finalize()
        finally:
            with self._registry_lock:
                self._callable_registry.clear()
                self._identity_registry.clear()
                self._live_handles.clear()

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

    def run(
        self,
        handle: CallableHandle,
        args: ChipStorageTaskArgs,
        config: CallConfig | None = None,
        **kwargs: Any,
    ):
        """Launch a callable previously returned by ``register_callable``.

        Args:
            handle: ``CallableHandle`` returned by ``register_callable``.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional CallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g.
                ``aicpu_thread_num=2``). A run always takes the whole device;
                orchestration reads the resulting width back through
                ``rt_available_cluster_count()``.

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

    def _run_slot_with_pipeline_lease(self, callable_id, args, slot_id, generation, config=None, **kwargs):
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        self._impl._run_with_pipeline_lease(int(callable_id), args, config, int(slot_id), int(generation))

    def _prepare_native_run_with_pipeline_lease(self, callable_id, args, slot_id, generation, config=None, **kwargs):
        """Prepare one native run without crossing its device launch fence.

        The lease generation is validated during admission. The returned
        token's unique prepare epoch is authoritative for subsequent
        launch/poll/wait/finalize calls on this ChipWorker.
        Keep every tensor backing buffer referenced by ``args`` alive until
        finalize returns.
        """
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        return self._impl._prepare_native_run_with_pipeline_lease(
            int(callable_id), args, config, int(slot_id), int(generation)
        )

    def _launch_native_run(self, run):
        self._impl._launch_native_run(run)

    def _poll_native_run(self, run):
        return bool(self._impl._poll_native_run(run))

    def _wait_native_run(self, run):
        self._impl._wait_native_run(run)

    def _finalize_native_run(self, run):
        self._impl._finalize_native_run(run)

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

    @property
    def run_stream_set_create_count(self):
        """Number of AICore run streams the bound runner has created."""
        return self._impl.run_stream_set_create_count

    @property
    def pipeline_depth(self):
        return self._impl.pipeline_depth

    @property
    def runtime_slot_count(self):
        return self._impl.runtime_slot_count

    @property
    def runtime_buffer_addrs(self):
        """Address of each opaque host native-run storage buffer, in slot order."""
        return list(self._impl.runtime_buffer_addrs)

    def arena_bank_gm_heap_base(self, bank_id):
        """Committed GM heap base of one arena bank, or 0 when uncommitted."""
        return int(self._impl.arena_bank_gm_heap_base(int(bank_id)))

    def retained_temp_addr(self, slot_id):
        """Retained temporary-buffer address for one slot, or 0 when unheld."""
        return int(self._impl.retained_temp_addr(int(slot_id)))

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
        """ACL device ordinal this worker is bound to."""
        return self._impl.device_id

    @property
    def initialized(self):
        """Whether the underlying native worker has completed init."""
        return self._impl.initialized

    @property
    def committed_device_memory(self) -> int:
        """Total device HBM (bytes) committed by this chip worker's MemoryAllocator."""
        return int(self._impl.committed_device_memory)
