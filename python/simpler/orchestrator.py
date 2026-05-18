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
        # build the args object yourself; tags drive dependency inference
        a = TaskArgs()
        a.add_tensor(make_tensor_arg(input_tensor),  TensorArgType.INPUT)
        a.add_tensor(make_tensor_arg(output_tensor), TensorArgType.OUTPUT)
        orch.submit_next_level(chip_cid, a, cfg)  # cid from Worker.register(chip_callable)

        sub_args = TaskArgs()
        sub_args.add_tensor(make_tensor_arg(output_tensor), TensorArgType.INPUT)
        orch.submit_sub(sub_cid, sub_args)

    w.run(my_orch, my_args, my_config)

Scope/drain lifecycle is managed by ``Worker.run()``; users never call those
directly.
"""

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from .task_interface import (
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CommDomainHandle,
    ContinuousTensor,
    DataType,
    TaskArgs,
)
from .task_interface import (
    _Orchestrator as _COrchestrator,
)


def _require_cid(callable_or_cid: Any, *, kind: str) -> int:
    """Coerce a submit argument to a registered cid.

    Raises a clear migration error when the caller still passes a
    ``ChipCallable`` directly — every chip callable must be registered
    via ``Worker.register(callable)`` *before* ``init()`` so each chip
    child can pre-warm it on its own device.
    """
    if isinstance(callable_or_cid, ChipCallable) or hasattr(callable_or_cid, "buffer_ptr"):
        raise TypeError(
            f"{kind} now takes a registered cid, not a ChipCallable. "
            "Register the callable before init() via "
            "`cid = worker.register(chip_callable)` and pass `cid` here."
        )
    return int(callable_or_cid)


class Orchestrator:
    """DAG builder. Valid only inside the orch function passed to Worker.run().

    Wraps a borrowed reference to the C++ Orchestrator owned by the parent
    Worker. The Python ``Worker`` keeps a strong reference to the parent
    C++ Worker for the entire orch-fn execution, so the borrowed reference
    stays valid.
    """

    def __init__(self, c_orchestrator: _COrchestrator, worker: Optional[Any] = None) -> None:
        self._o = c_orchestrator
        # Back-reference to the Python Worker so dynamic-allocate APIs
        # (allocate_domain / release_domain) can dispatch CTRL_* through the
        # Worker's chip mailboxes.  None when the Orchestrator is constructed
        # in isolation for tests.
        self._worker = worker

    # ------------------------------------------------------------------
    # User-facing submit API
    # ------------------------------------------------------------------

    def submit_next_level(
        self, callable_id: Any, args: TaskArgs, config: Optional[CallConfig] = None, *, worker: int = -1
    ):
        """Submit a NEXT_LEVEL (chip) task by registered callable id.

        ``callable_id`` must be the int returned by
        ``Worker.register(chip_callable)``. Tags inside ``args`` drive deps.
        ``worker``: logical worker id for affinity (-1 = unconstrained).
        """
        cfg = config if config is not None else CallConfig()
        cid = _require_cid(callable_id, kind="orch.submit_next_level")
        return self._o.submit_next_level(cid, args, cfg, int(worker))

    def submit_next_level_group(
        self,
        callable_id: Any,
        args_list: list,
        config: Optional[CallConfig] = None,
        *,
        workers: Optional[list] = None,
    ):
        """Submit a group of NEXT_LEVEL tasks (N TaskArgs → N workers, 1 DAG node).

        ``workers``: per-args affinity list (None/empty = all unconstrained).
        """
        cfg = config if config is not None else CallConfig()
        w = [int(x) for x in workers] if workers else []
        cid = _require_cid(callable_id, kind="orch.submit_next_level_group")
        return self._o.submit_next_level_group(cid, args_list, cfg, w)

    def submit_sub(self, callable_id: int, args: Optional[TaskArgs] = None):
        """Submit a SUB task by registered callable id.

        ``args`` may be omitted for a tag-less task (no dependencies, no outputs).
        """
        if args is None:
            args = TaskArgs()
        return self._o.submit_sub(int(callable_id), args)

    def submit_sub_group(self, callable_id: int, args_list: list):
        """Submit a group of SUB tasks (N TaskArgs → N workers, 1 DAG node)."""
        return self._o.submit_sub_group(int(callable_id), args_list)

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
                    orch.submit_next_level(cid, ..., worker=chip_idx)
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
    def scope(self) -> Iterator["Orchestrator"]:
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

    def open_channel(
        self,
        worker_id: int,
        cpu_to_l2_lanes: int = 1,
        l2_to_cpu_lanes: int = 1,
        lane_depth: int = 64,
        max_message_bytes: int = 256,
    ) -> int:
        """Open a bounded L3/L2 message channel on next-level worker *worker_id*."""
        return int(
            self._o.open_channel(
                int(worker_id),
                int(cpu_to_l2_lanes),
                int(l2_to_cpu_lanes),
                int(lane_depth),
                int(max_message_bytes),
            )
        )

    def close_channel(self, worker_id: int, channel: int) -> None:
        """Close a channel returned by ``open_channel``."""
        self._o.close_channel(int(worker_id), int(channel))

    def channel_send(
        self,
        worker_id: int,
        channel: int,
        route: int,
        data: bytes,
        correlation_id: int = 0,
    ) -> None:
        """Send one inline message from L3 CPU toward L2."""
        self._o.channel_send(int(worker_id), int(channel), int(route), bytes(data), int(correlation_id))

    def channel_recv(self, worker_id: int, channel: int, capacity: int = 256, timeout_us: int = 0) -> tuple[bytes, int, int]:
        """Receive one inline message from L2 toward L3 CPU."""
        data, route, correlation_id = self._o.channel_recv(
            int(worker_id), int(channel), int(capacity), int(timeout_us)
        )
        return bytes(data), int(route), int(correlation_id)

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
