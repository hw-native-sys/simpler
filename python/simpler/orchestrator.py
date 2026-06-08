# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestrator — DAG builder exposed to the user's orch function during Worker.run().

A thin Python facade over the C++ ``Orchestrator``. The Worker creates one
Orchestrator handle at init, retrieves the C++ object via ``Worker.get_orchestrator()``,
and passes the handle to the user's orch function::

    def my_orch(orch, args, cfg):
        # chip_handle/sub_handle come from Worker.register(...)
        # build the args object yourself; tags drive dependency inference
        a = TaskArgs()
        a.add_tensor(make_tensor_arg(input_tensor),  TensorArgType.INPUT)
        a.add_tensor(make_tensor_arg(output_tensor), TensorArgType.OUTPUT)
        orch.submit_next_level(chip_handle, a, cfg)  # handle from Worker.register(chip_callable)

        sub_args = TaskArgs()
        sub_args.add_tensor(make_tensor_arg(output_tensor), TensorArgType.INPUT)
        orch.submit_sub(sub_handle, sub_args)

    w.run(my_orch, my_args, my_config)

Scope/drain lifecycle is managed by ``Worker.run()``; users never call those
directly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

from _task_interface import _Orchestrator as _COrchestrator  # pyright: ignore[reportMissingImports]

from .callable_identity import CallableHandle
from .task_interface import (
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CommDomainHandle,
    ContinuousTensor,
    DataType,
    RemoteAddressSpace,
    TaskArgs,
    _empty_remote_sidecar_for,
    _remote_sidecar_for,
)


def _require_handle(
    callable_or_handle: Any,
    *,
    kind: str,
    worker: Any = None,
    expected_namespace: str | None = None,
) -> tuple[bytes, str, str, tuple[int, ...]]:
    """Validate a submit argument is a registered CallableHandle.

    Raises a clear migration error when the caller still passes a
    ``ChipCallable`` directly — every chip callable must be registered
    via ``Worker.register(callable)`` *before* ``init()`` so each chip
    child can pre-warm it on its own device.
    """
    if isinstance(callable_or_handle, ChipCallable) or hasattr(callable_or_handle, "buffer_ptr"):
        raise TypeError(
            f"{kind} now takes a CallableHandle, not a ChipCallable. "
            "Register the callable before init() via "
            "`handle = worker.register(chip_callable)` and pass `handle` here."
        )
    if not isinstance(callable_or_handle, CallableHandle):
        raise TypeError(f"{kind} expects a CallableHandle returned by Worker.register")
    if worker is not None:
        state = worker._resolve_handle(callable_or_handle, expected_namespace=expected_namespace)
        return state.digest, state.kind, state.target_namespace, state.eligible_endpoint_ids
    if expected_namespace is not None and callable_or_handle.target_namespace != expected_namespace:
        raise TypeError(
            f"{kind} cannot run {callable_or_handle.target_namespace}; expected {expected_namespace} "
            f"for {callable_or_handle.hashid}"
        )
    return callable_or_handle.digest, callable_or_handle.kind, callable_or_handle.target_namespace, ()


def _split_next_level_args(args: TaskArgs) -> tuple[TaskArgs, object | None]:
    if isinstance(args, TaskArgs):
        return args, _remote_sidecar_for(args)
    raise TypeError("NEXT_LEVEL submit expects TaskArgs")


def _remote_data_eligible_endpoint_ids(
    remote_sidecar: object | None,
    callable_endpoint_ids: tuple[int, ...],
) -> list[int]:
    endpoint_ids = [int(endpoint_id) for endpoint_id in callable_endpoint_ids]
    if remote_sidecar is None:
        return endpoint_ids

    allowed = set(endpoint_ids)
    for tensor_sidecar in getattr(remote_sidecar, "tensors", ()):
        if tensor_sidecar is None or not getattr(tensor_sidecar, "present", False):
            continue
        desc = tensor_sidecar.desc
        if RemoteAddressSpace(int(desc.address_space)) == RemoteAddressSpace.HOST_INLINE:
            continue
        handle = getattr(tensor_sidecar, "handle", None)
        consumable_endpoint_id = int(getattr(handle, "endpoint_id", desc.owner_endpoint_id))
        allowed.intersection_update({consumable_endpoint_id})

    final_endpoint_ids = [endpoint_id for endpoint_id in endpoint_ids if endpoint_id in allowed]
    if not final_endpoint_ids:
        raise ValueError("remote tensor sidecars leave no eligible remote endpoint")
    return final_endpoint_ids


class Orchestrator:
    """DAG builder. Valid only inside the orch function passed to Worker.run().

    Wraps a borrowed reference to the C++ Orchestrator owned by the parent
    Worker. The Python ``Worker`` keeps a strong reference to the parent
    C++ Worker for the entire orch-fn execution, so the borrowed reference
    stays valid.
    """

    def __init__(self, c_orchestrator: _COrchestrator, worker: Any | None = None) -> None:
        self._o = c_orchestrator
        # Back-reference to the Python Worker so dynamic-allocate APIs
        # (allocate_domain / release_domain) can dispatch CTRL_* through the
        # Worker's chip mailboxes.  None when the Orchestrator is constructed
        # in isolation for tests.
        self._worker = worker

    def _expected_next_level_namespace(self) -> str | None:
        if self._worker is None:
            return None
        if getattr(self._worker, "_next_level_workers", []):
            return "LOCAL_PYTHON"
        if getattr(self._worker, "_chip_shms", []):
            return "LOCAL_CHIP"
        return None

    # ------------------------------------------------------------------
    # User-facing submit API
    # ------------------------------------------------------------------

    def submit_next_level(
        self, callable_handle: Any, args: TaskArgs, config: CallConfig | None = None, *, worker: int = -1
    ):
        """Submit a NEXT_LEVEL task by registered callable handle.

        ``callable_handle`` must be returned by ``Worker.register``. Tags inside ``args`` drive deps.
        ``worker``: logical worker id for affinity (-1 = unconstrained).
        """
        cfg = config if config is not None else CallConfig()
        expected_namespace = (
            None
            if isinstance(callable_handle, CallableHandle)
            and callable_handle.target_namespace == "REMOTE_TASK_DISPATCHER"
            else self._expected_next_level_namespace()
        )
        digest, kind, target_namespace, eligible_endpoint_ids = _require_handle(
            callable_handle,
            kind="orch.submit_next_level",
            worker=self._worker,
            expected_namespace=expected_namespace,
        )
        c_args, explicit_remote_sidecar = _split_next_level_args(args)
        if target_namespace == "REMOTE_TASK_DISPATCHER":
            remote_sidecar = (
                explicit_remote_sidecar if explicit_remote_sidecar is not None else _empty_remote_sidecar_for(c_args)
            )
        else:
            if explicit_remote_sidecar is not None:
                raise TypeError("RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits")
            remote_sidecar = None
        final_endpoint_ids = _remote_data_eligible_endpoint_ids(remote_sidecar, eligible_endpoint_ids)
        captured_refs = self._worker._capture_remote_sidecar_refs(remote_sidecar) if self._worker is not None else []
        try:
            self._o.submit_next_level(
                digest, kind, target_namespace, c_args, cfg, int(worker), final_endpoint_ids, remote_sidecar
            )
        except BaseException:
            if self._worker is not None:
                self._worker._release_remote_slot_refs(captured_refs)
            raise
        if self._worker is not None:
            self._worker._adopt_remote_slot_refs(captured_refs)

    def submit_next_level_group(
        self,
        callable_handle: Any,
        args_list: list,
        config: CallConfig | None = None,
        *,
        workers: list | None = None,
    ):
        """Submit a group of NEXT_LEVEL tasks (N TaskArgs → N workers, 1 DAG node).

        ``workers``: per-args affinity list (None/empty = all unconstrained).
        """
        cfg = config if config is not None else CallConfig()
        w = [int(x) for x in workers] if workers else []
        expected_namespace = (
            None
            if isinstance(callable_handle, CallableHandle)
            and callable_handle.target_namespace == "REMOTE_TASK_DISPATCHER"
            else self._expected_next_level_namespace()
        )
        digest, kind, target_namespace, eligible_endpoint_ids = _require_handle(
            callable_handle,
            kind="orch.submit_next_level_group",
            worker=self._worker,
            expected_namespace=expected_namespace,
        )
        c_args_list = []
        explicit_remote_sidecars = []
        has_explicit_remote_sidecar = False
        for args in args_list:
            c_args, sidecar = _split_next_level_args(args)
            c_args_list.append(c_args)
            explicit_remote_sidecars.append(sidecar)
            has_explicit_remote_sidecar = has_explicit_remote_sidecar or sidecar is not None
        if target_namespace == "REMOTE_TASK_DISPATCHER":
            remote_sidecars = [
                sidecar if sidecar is not None else _empty_remote_sidecar_for(c_args)
                for c_args, sidecar in zip(c_args_list, explicit_remote_sidecars)
            ]
        else:
            if has_explicit_remote_sidecar:
                raise TypeError("RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits")
            remote_sidecars = None
        endpoint_sets = (
            [
                _remote_data_eligible_endpoint_ids(remote_sidecar, eligible_endpoint_ids)
                for remote_sidecar in remote_sidecars
            ]
            if remote_sidecars is not None
            else [list(eligible_endpoint_ids) for _ in args_list]
            if eligible_endpoint_ids
            else []
        )
        captured_refs: list[Any] = []
        if self._worker is not None and remote_sidecars is not None:
            for sidecar in remote_sidecars:
                captured_refs.extend(self._worker._capture_remote_sidecar_refs(sidecar))
        try:
            self._o.submit_next_level_group(
                digest, kind, target_namespace, c_args_list, cfg, w, endpoint_sets, remote_sidecars
            )
        except BaseException:
            if self._worker is not None:
                self._worker._release_remote_slot_refs(captured_refs)
            raise
        if self._worker is not None:
            self._worker._adopt_remote_slot_refs(captured_refs)

    def submit_sub(self, callable_handle: Any, args: TaskArgs | None = None):
        """Submit a SUB task by registered callable handle.

        ``args`` may be omitted for a tag-less task (no dependencies, no outputs).
        """
        if args is None:
            args = TaskArgs()
        digest, kind, target_namespace, _eligible_endpoint_ids = _require_handle(
            callable_handle,
            kind="orch.submit_sub",
            worker=self._worker,
            expected_namespace="LOCAL_PYTHON",
        )
        self._o.submit_sub(digest, kind, target_namespace, args)

    def submit_sub_group(self, callable_handle: Any, args_list: list):
        """Submit a group of SUB tasks (N TaskArgs → N workers, 1 DAG node)."""
        digest, kind, target_namespace, _eligible_endpoint_ids = _require_handle(
            callable_handle,
            kind="orch.submit_sub_group",
            worker=self._worker,
            expected_namespace="LOCAL_PYTHON",
        )
        self._o.submit_sub_group(digest, kind, target_namespace, args_list)

    # ------------------------------------------------------------------
    # Dynamic CommDomain allocation (collective; blocks orch_fn for the
    # duration of the alloc / release handshake)
    # ------------------------------------------------------------------

    def allocate_domain(
        self,
        *,
        name: str,
        workers: Sequence[int],
        window_size: int,
        buffers: Sequence[CommBufferSpec] = (),
    ) -> CommDomainHandle:
        """Collectively allocate a fresh CommDomain across `workers`.

        Driven from the orch thread.  Dispatches CTRL_ALLOC_DOMAIN to each
        participating chip in parallel and blocks until all have completed
        the IPC handshake (HCCL: aclrtMalloc + IPC import; sim: shm + ftruncate).
        Returns a ``CommDomainHandle`` whose ``contexts[chip_idx]`` exposes
        the per-chip ``ChipDomainContext`` (``device_ctx``, ``local_window_base``,
        ``buffer_ptrs`` by name).

        ``name`` is a local identifier (uniqueness checked against currently-live
        handles); peers do not need to agree on the string.  ``workers`` must be
        a subset of the Worker's ``device_ids`` indices; their order defines
        dense domain ranks.  ``buffers`` are carved sequentially inside the
        window in declaration order; their ``nbytes`` sum must fit within
        ``window_size`` — this is validated on the orch thread before any
        chip-side allocation is dispatched, so an oversized request raises
        ``ValueError`` here without leaking a backend allocation.

        Use the handle as a context manager for auto-release:

            with orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096) as tp:
                for chip_idx in tp.workers:
                    orch.submit_next_level(chip_handle, ..., worker=chip_idx)
        """
        if self._worker is None:
            raise RuntimeError("allocate_domain requires an Orchestrator bound to a Worker")
        return self._worker._allocate_domain(
            name=str(name),
            workers=tuple(int(w) for w in workers),
            window_size=int(window_size),
            buffers=list(buffers),
        )

    def release_domain(self, handle: CommDomainHandle) -> None:
        """Collective release.  Equivalent to ``handle.release()``."""
        handle.release()

    # ------------------------------------------------------------------
    # Nested scope (Strict-1 per-scope rings)
    # ------------------------------------------------------------------
    #
    # Tasks and allocations inside a nested ``with orch.scope():`` bind to a
    # deeper heap ring (``min(depth, MAX_RING_DEPTH-1)``) so their
    # memory reclaims independently of the outer scope. ``scope_end`` is
    # non-blocking — it releases scope refs and returns; call
    # ``Worker.run``/``drain`` for a synchronous wait.
    #
    # Usage::
    #
    #     def my_orch(orch, args):
    #         with orch.scope():
    #             orch.submit_next_level(a, ...)
    #             orch.submit_next_level(b, ...)
    #         orch.submit_next_level(c, ...)   # back on outer-scope ring

    def scope_begin(self) -> None:
        self._o.scope_begin()

    def scope_end(self) -> None:
        self._o.scope_end()

    @contextlib.contextmanager
    def scope(self) -> Iterator[Orchestrator]:
        """Open a nested scope for the ``with`` block.

        Tasks submitted inside the block use a deeper heap ring so they
        reclaim independently of the outer scope (see Strict-1 in
        ``.claude/plans/HIERARCHICAL_RUNTIME_REFACTOR.md``).
        """
        self._o.scope_begin()
        try:
            yield self
        finally:
            self._o.scope_end()

    def malloc(self, worker_id: int, size: int) -> int:
        """Allocate memory on next-level worker *worker_id*. Returns a pointer."""
        return int(self._o.malloc(int(worker_id), int(size)))

    def free(self, worker_id: int, ptr: int) -> None:
        """Free memory on next-level worker *worker_id*."""
        self._o.free(int(worker_id), int(ptr))

    def copy_to(self, worker_id: int, dst: int, src: int, size: int) -> None:
        """Copy *size* bytes from host *src* to worker *dst*."""
        self._o.copy_to(int(worker_id), int(dst), int(src), int(size))

    def copy_from(self, worker_id: int, dst: int, src: int, size: int) -> None:
        """Copy *size* bytes from worker *src* to host *dst*."""
        self._o.copy_from(int(worker_id), int(dst), int(src), int(size))

    def alloc(self, shape: Sequence[int], dtype: DataType) -> ContinuousTensor:
        """Allocate a runtime-managed intermediate buffer.

        Returns a ``ContinuousTensor`` whose backing memory comes from a
        per-allocation MAP_SHARED mmap (visible to forked child workers).
        Lifetime is bound to a synthetic task slot that the Orchestrator
        treats as the buffer's producer; the buffer is freed when all
        downstream consumers have completed and the run's scope ends.

        Use this for chip-A → chip-B intermediate buffers instead of
        pre-allocating with ``torch.share_memory_()`` — the runtime owns
        the lifecycle.
        """
        return self._o.alloc(list(shape), dtype)

    # ------------------------------------------------------------------
    # Internal (called by Worker.run)
    # ------------------------------------------------------------------

    def _scope_begin(self) -> None:
        self._o._scope_begin()

    def _scope_end(self) -> None:
        self._o._scope_end()

    def _drain(self) -> None:
        self._o._drain()

    def _clear_error(self) -> None:
        self._o._clear_error()
