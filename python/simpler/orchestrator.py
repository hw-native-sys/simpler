# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestrator — DAG builder passed to a Worker submit/run callback.

A thin Python facade over the C++ ``Orchestrator``. The Worker creates one
Orchestrator handle at init, retrieves the C++ object via ``Worker.get_orchestrator()``,
and passes the handle to the user's orch function::

    def my_orch(orch, args, cfg):
        # chip_handle/sub_handle come from Worker.register(...)
        # build the args object yourself; tags drive dependency inference
        a = TaskArgs()
        a.add_tensor(make_tensor_arg(input_tensor),  TensorArgType.INPUT)
        a.add_tensor(make_tensor_arg(output_tensor), TensorArgType.OUTPUT)
        orch.submit_next_level(chip_handle, a, cfg, worker=0)

        sub_args = TaskArgs()
        sub_args.add_tensor(make_tensor_arg(output_tensor), TensorArgType.INPUT)
        orch.submit_sub(sub_handle, sub_args)

    handle = w.submit(my_orch, my_args, my_config)
    handle.wait()

Scope and submission-close lifecycle is managed by ``Worker.submit()``;
completion is managed by its ``RunHandle``. ``Worker.run()`` remains the
blocking ``submit(...).wait()`` compatibility entry point.
"""

from __future__ import annotations

import contextlib
import operator
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from _task_interface import _Orchestrator as _COrchestrator  # pyright: ignore[reportMissingImports]

from .callable_identity import CallableHandle
from .task_interface import (
    CallConfig,
    ChipCallable,
    ChipTensor,
    CommBufferSpec,
    CommDomainHandle,
    DataType,
    GlobalCommDomainHandle,
    GlobalCommDomainView,
    RemoteAddressSpace,
    TaskArgs,
    _empty_remote_sidecar_for,
    _remote_sidecar_for,
    _RemoteTaskArgsSidecar,
    _validate_remote_sidecar_access,
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
        return state.digest, state.kind, state.target_namespace, state.eligible_worker_ids
    if expected_namespace is not None and callable_or_handle.target_namespace != expected_namespace:
        raise TypeError(
            f"{kind} cannot run {callable_or_handle.target_namespace}; expected {expected_namespace} "
            f"for {callable_or_handle.hashid}"
        )
    return callable_or_handle.digest, callable_or_handle.kind, callable_or_handle.target_namespace, ()


def _require_next_level_worker_id(value: Any, *, argument: str) -> int:
    """Return an exact integer worker ID without accepting coercible values."""
    if isinstance(value, bool):
        raise TypeError(f"{argument} must be an integer NEXT_LEVEL worker id")
    try:
        worker_id = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{argument} must be an integer NEXT_LEVEL worker id") from exc
    if worker_id < 0:
        raise ValueError(f"{argument} must be a non-negative NEXT_LEVEL worker id")
    return worker_id


def _split_next_level_args(args: TaskArgs) -> tuple[TaskArgs, _RemoteTaskArgsSidecar | None]:
    if isinstance(args, TaskArgs):
        return args, _remote_sidecar_for(args)
    raise TypeError("NEXT_LEVEL submit expects TaskArgs")


def _reject_remote_sidecar_args(args: object, *, kind: str) -> None:
    if isinstance(args, TaskArgs) and _remote_sidecar_for(args) is not None:
        raise TypeError(f"RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits, not {kind}")


def _remote_data_eligible_worker_ids(
    remote_sidecar: _RemoteTaskArgsSidecar | None,
    callable_worker_ids: tuple[int, ...],
) -> list[int]:
    worker_ids = [int(worker_id) for worker_id in callable_worker_ids]
    if remote_sidecar is None:
        return worker_ids

    allowed = set(worker_ids)
    for tensor_sidecar in getattr(remote_sidecar, "tensors", ()):
        if tensor_sidecar is None or not getattr(tensor_sidecar, "present", False):
            continue
        desc = tensor_sidecar.desc
        if RemoteAddressSpace(int(desc.address_space)) == RemoteAddressSpace.HOST_INLINE:
            continue
        handle = getattr(tensor_sidecar, "handle", None)
        consumable_worker_id = int(getattr(handle, "worker_id", desc.owner_worker_id))
        allowed.intersection_update({consumable_worker_id})

    final_worker_ids = [worker_id for worker_id in worker_ids if worker_id in allowed]
    if not final_worker_ids:
        raise ValueError("remote tensor sidecars leave no eligible remote worker")
    return final_worker_ids


# Which graph callbacks the *current thread* is inside. Direct device control
# has to be ordered against the run that issued it, and a process-wide "a run is
# building" flag cannot say that: a public ``Worker.copy_*`` on another thread
# would be charged to whichever run happens to be building and blocked behind
# it.
#
# A run id alone cannot say it either, because it names nothing on its own —
# run 1 exists on every Worker. An L4 callback drives its children's runs on its
# own thread, so the frames nest and a control call has to find the one for the
# Worker it is about to touch. Applying the innermost frame instead would order
# a call on Worker B against Worker A's run, and skip B's own reservation
# entirely.
_CALLBACK_RUN = threading.local()


class _CallbackFrame:
    __slots__ = ("has_submitted_task", "run_id", "worker")

    def __init__(self, worker: Any, run_id: int) -> None:
        self.worker = worker
        self.run_id = int(run_id)
        self.has_submitted_task = False


def _callback_frames() -> list[_CallbackFrame]:
    frames = getattr(_CALLBACK_RUN, "frames", None)
    if frames is None:
        frames = []
        _CALLBACK_RUN.frames = frames
    return frames


def _callback_frame_for(worker: Any) -> _CallbackFrame | None:
    """The innermost open callback on this thread that belongs to *worker*."""
    for frame in reversed(_callback_frames()):
        if frame.worker is worker:
            return frame
    return None


@contextlib.contextmanager
def _callback_run(run_id: int, worker: Any = None):
    """Mark this thread as executing *worker*'s *run_id* graph callback."""
    frames = _callback_frames()
    frames.append(_CallbackFrame(worker, run_id))
    try:
        yield
    finally:
        frames.pop()


def _admit_task_submission(worker: Any = None) -> None:
    """Gate one native task-submission attempt and record possible work in flight.

    Direct device control after this point cannot be ordered against it: the
    task reaches its child through the ready queue and the control through the
    mailbox, and waiting for the run to hold the FIFO head says nothing about
    which of the two arrives first. Two of them on different chips can each hold
    one mailbox and wait for the other.

    The sticky refusal is re-read here for the same reason control re-reads it:
    an open callback that caught a failed rollback would otherwise keep
    submitting on top of device state this worker can no longer reclaim.
    """
    if worker is not None:
        worker._require_no_ordered_cleanup_failure("submit")
    frame = _callback_frame_for(worker)
    if frame is not None:
        frame.has_submitted_task = True


@contextlib.contextmanager
def direct_control(worker: Any, native_orch: Any, api: str):
    """Order one command that reaches a child outside any TaskSlot.

    `malloc`, `copy_*`, domain and region creation, and every `remote_*`
    buffer call travel the mailbox rather than the ready queue, so the
    whole-run FIFO does not sequence them. Two cases, and the reservation is
    held for the *whole* call in both — a check that only samples state leaves
    the command itself outside the decision it just made.

    A call issued inside a graph callback belongs to that run and waits for it
    to hold the FIFO head. A call that belongs to no run is ordered only by
    being alone: it takes the same serializer submission uses, so no run can be
    admitted between the check and the command.
    """
    frame = _callback_frame_for(worker)
    if frame is not None:
        if frame.has_submitted_task:
            raise RuntimeError(
                f"{api}: direct device control cannot follow a task submission in the same run — the task "
                "travels the ready queue and this travels the mailbox, so their order is not defined and two "
                "such pairs can deadlock across chips. Issue all control before the run's first submit_*()"
            )
        # A run that is already known to have left device state behind is not a
        # valid owner for more of it, even though it is still open: its callback
        # may have caught the rollback failure and carried on.
        if worker is not None:
            worker._require_no_ordered_cleanup_failure(api)
        if native_orch is not None:
            native_orch.await_run_admission(frame.run_id)
        yield
        return
    if worker is None:
        yield
        return
    with worker._control_reservation(api):
        yield


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

    def submit_next_level(self, callable_handle: Any, args: TaskArgs, config: CallConfig | None = None, *, worker: int):
        """Submit a NEXT_LEVEL task by registered callable handle.

        ``callable_handle`` must be returned by ``Worker.register``. Tags inside ``args`` drive deps.
        ``worker`` is the exact stable NEXT_LEVEL worker id that runs the
        task. For L3 chip dispatch, these are the existing chip worker ids.
        """
        cfg = config if config is not None else CallConfig()
        cpp_worker_id = _require_next_level_worker_id(worker, argument="worker")
        expected_namespace = (
            None
            if isinstance(callable_handle, CallableHandle)
            and callable_handle.target_namespace == "REMOTE_TASK_DISPATCHER"
            else self._expected_next_level_namespace()
        )
        digest, kind, target_namespace, eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_next_level",
            worker=self._worker,
            expected_namespace=expected_namespace,
        )
        if target_namespace != "REMOTE_TASK_DISPATCHER" and self._worker is not None:
            self._worker._require_local_next_level_target(cpp_worker_id, api="submit_next_level")
        c_args, explicit_remote_sidecar = _split_next_level_args(args)
        if target_namespace == "REMOTE_TASK_DISPATCHER":
            remote_sidecar = (
                explicit_remote_sidecar if explicit_remote_sidecar is not None else _empty_remote_sidecar_for(c_args)
            )
        else:
            if explicit_remote_sidecar is not None:
                raise TypeError("RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits")
            remote_sidecar = None
        _validate_remote_sidecar_access(c_args, remote_sidecar)
        # Validate the post-fork host buffers of this submit (issue #1027). Only
        # the LOCAL_CHIP path dereferences raw host pointers in the forked child;
        # zero-copy buffers need no per-run mirror, just an in-range fit check.
        if target_namespace == "LOCAL_CHIP" and self._worker is not None:
            self._worker._stage_host_buffers_for_chip_submit(c_args)
        final_worker_ids = _remote_data_eligible_worker_ids(remote_sidecar, eligible_worker_ids)
        worker = self._worker
        # Provenance validation precedes run ownership publication. Once a remote
        # ref is published, only the run fence may release it because the native
        # submit can commit before an exception reaches Python.
        child_ptrs = worker._child_ptrs_in_args(c_args) if worker is not None else []
        prov_guard: Any = contextlib.nullcontext()
        if child_ptrs and worker is not None:
            prov_guard = worker._child_prov_lock
        with prov_guard:
            if child_ptrs and worker is not None:
                worker._child_prov_check_dispatch(child_ptrs, cpp_worker_id, api="submit_next_level")
            if worker is not None:
                worker._adopt_remote_sidecar_refs((remote_sidecar,))
            _admit_task_submission(self._worker)
            self._o.submit_next_level(
                digest, kind, target_namespace, c_args, cfg, cpp_worker_id, final_worker_ids, remote_sidecar
            )

    def submit_next_level_group(  # noqa: PLR0912 -- linear per-member sidecar + eligibility + kind4-provenance passes, one branch each
        self,
        callable_handle: Any,
        args_list: list,
        config: CallConfig | None = None,
        *,
        workers: list,
    ):
        """Submit a group of NEXT_LEVEL tasks (N TaskArgs → N worker selections, 1 DAG node).

        ``workers`` contains the exact stable NEXT_LEVEL worker id for each
        member. For L3 chip dispatch, these are the existing chip worker ids.
        """
        cfg = config if config is not None else CallConfig()
        worker_ids = [_require_next_level_worker_id(value, argument="workers entries") for value in workers]
        if len(worker_ids) != len(args_list):
            raise ValueError("workers length must match args_list length")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("workers must not contain duplicate NEXT_LEVEL worker ids")
        expected_namespace = (
            None
            if isinstance(callable_handle, CallableHandle)
            and callable_handle.target_namespace == "REMOTE_TASK_DISPATCHER"
            else self._expected_next_level_namespace()
        )
        digest, kind, target_namespace, eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_next_level_group",
            worker=self._worker,
            expected_namespace=expected_namespace,
        )
        if target_namespace != "REMOTE_TASK_DISPATCHER" and self._worker is not None:
            for worker_id in worker_ids:
                self._worker._require_local_next_level_target(worker_id, api="submit_next_level_group")
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
        if remote_sidecars is not None:
            for c_args, remote_sidecar in zip(c_args_list, remote_sidecars):
                _validate_remote_sidecar_access(c_args, remote_sidecar)
        # Validate post-fork host buffers for chip dispatch (issue #1027), same as
        # the single submit path.
        if target_namespace == "LOCAL_CHIP" and self._worker is not None:
            for c_args in c_args_list:
                self._worker._stage_host_buffers_for_chip_submit(c_args)
        worker_id_sets = (
            [
                _remote_data_eligible_worker_ids(remote_sidecar, eligible_worker_ids)
                for remote_sidecar in remote_sidecars
            ]
            if remote_sidecars is not None
            else [list(eligible_worker_ids) for _ in args_list]
            if eligible_worker_ids
            else []
        )
        # Per-member kind4 dispatch guard: each member's child_memory pointers
        # must be live on that member's exact submitted target.
        # Run this fallible analysis before publishing remote-ref ownership.
        worker = self._worker
        member_checks: list[tuple[list[tuple[int, int]], int]] = []
        if worker is not None:
            for g, c_args in enumerate(c_args_list):
                child_ptrs = worker._child_ptrs_in_args(c_args)
                if not child_ptrs:
                    continue
                member_checks.append((child_ptrs, worker_ids[g]))
        prov_guard: Any = (
            worker._child_prov_lock if (worker is not None and member_checks) else contextlib.nullcontext()
        )
        with prov_guard:
            for child_ptrs, target_worker_id in member_checks:
                assert worker is not None  # member_checks is only populated when worker is present
                worker._child_prov_check_dispatch(child_ptrs, target_worker_id, api="submit_next_level_group")
            if worker is not None and remote_sidecars is not None:
                worker._adopt_remote_sidecar_refs(remote_sidecars)
            _admit_task_submission(self._worker)
            self._o.submit_next_level_group(
                digest, kind, target_namespace, c_args_list, cfg, worker_ids, worker_id_sets, remote_sidecars
            )

    def submit_sub(self, callable_handle: Any, args: TaskArgs | None = None):
        """Submit a SUB task by registered callable handle.

        ``args`` may be omitted for a tag-less task (no dependencies, no outputs).
        """
        if args is None:
            args = TaskArgs()
        digest, kind, target_namespace, _eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_sub",
            worker=self._worker,
            expected_namespace="LOCAL_PYTHON",
        )
        _reject_remote_sidecar_args(args, kind="orch.submit_sub")
        _admit_task_submission(self._worker)
        self._o.submit_sub(digest, kind, target_namespace, args)

    def submit_sub_group(self, callable_handle: Any, args_list: list):
        """Submit a group of SUB tasks (N TaskArgs → N workers, 1 DAG node)."""
        digest, kind, target_namespace, _eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_sub_group",
            worker=self._worker,
            expected_namespace="LOCAL_PYTHON",
        )
        for args in args_list:
            _reject_remote_sidecar_args(args, kind="orch.submit_sub_group")
        _admit_task_submission(self._worker)
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
        # Collective domain setup drives CTRL_COMM_INIT / CTRL_ALLOC_DOMAIN on
        # every member chip, so it is a device effect and must not overtake the
        # active run's mailbox traffic.
        with self._control_admission("allocate_domain"):
            return self._worker._allocate_domain(
                name=str(name),
                workers=tuple(int(w) for w in workers),
                window_size=int(window_size),
                buffers=list(buffers),
            )

    def release_domain(self, handle: CommDomainHandle) -> None:
        """Collective release.  Equivalent to ``handle.release()``."""
        handle.release()

    def allocate_global_domain(
        self,
        *,
        name: str,
        members: Sequence[tuple[int, int]],
        window_size: int,
        buffers: Sequence[CommBufferSpec] = (),
        retain_after_run: bool = False,
    ) -> GlobalCommDomainHandle:
        """Create a CommDomain across local and/or remote L3 nodes without MPI.

        Each member is ``(l3_worker_id, local_l2_worker_id)``. The L3 worker
        may have been registered by ``Worker.add_worker`` or
        ``Worker.add_remote_worker``. L4 collects every L2 export descriptor,
        sends the complete rank-ordered table back to every L3, and commits
        only after all L2 imports succeed.
        ``retain_after_run=True`` keeps the domain live after the current DAG
        drains so a later run can inspect communication results; explicit
        release or ``Worker.close()`` still tears it down.
        """
        if self._worker is None:
            raise RuntimeError("allocate_global_domain requires an Orchestrator bound to a Worker")
        return self._worker._allocate_global_domain(
            name=str(name),
            members=tuple((int(node), int(local)) for node, local in members),
            window_size=int(window_size),
            buffers=list(buffers),
            retain_after_run=bool(retain_after_run),
        )

    def release_global_domain(self, handle: GlobalCommDomainHandle) -> None:
        """Request fence-ordered release of an L4-owned Global CommDomain."""
        handle.release()

    def get_global_domain(self, domain_id: int) -> GlobalCommDomainView:
        """Return the committed L3-local view for a domain created by L4."""
        if self._worker is None:
            raise RuntimeError("get_global_domain requires an Orchestrator bound to a Worker")
        return self._worker._get_global_domain(int(domain_id))

    @staticmethod
    def _global_copy_range(handle: GlobalCommDomainHandle, *, buffer: str | None, offset: int, nbytes: int) -> int:
        absolute = int(offset)
        if absolute < 0 or nbytes <= 0:
            raise ValueError("Global CommDomain copy offset must be non-negative and size must be positive")
        limit = handle.mapping_size
        if buffer is not None:
            buffer_offset, buffer_nbytes = handle.buffer_range(str(buffer))
            if absolute > buffer_nbytes or nbytes > buffer_nbytes - absolute:
                raise ValueError(f"Global CommDomain copy exceeds buffer {buffer!r}")
            absolute += buffer_offset
        elif absolute > limit or nbytes > limit - absolute:
            raise ValueError("Global CommDomain copy exceeds the mapped window")
        return absolute

    def copy_to_global_domain(
        self,
        handle: GlobalCommDomainHandle,
        domain_rank: int,
        data: bytes,
        *,
        buffer: str | None = None,
        offset: int = 0,
    ) -> None:
        """Copy bytes into one rank's mapped window or named buffer."""
        payload = bytes(data)
        absolute = self._global_copy_range(handle, buffer=buffer, offset=int(offset), nbytes=len(payload))
        if self._worker is None:
            raise RuntimeError("copy_to_global_domain requires an Orchestrator bound to a Worker")
        self._worker._copy_to_global_domain(handle, int(domain_rank), payload, absolute)

    def copy_from_global_domain(
        self,
        handle: GlobalCommDomainHandle,
        domain_rank: int,
        nbytes: int,
        *,
        buffer: str | None = None,
        offset: int = 0,
    ) -> bytes:
        """Copy bytes from one rank's mapped window or named buffer."""
        absolute = self._global_copy_range(handle, buffer=buffer, offset=int(offset), nbytes=int(nbytes))
        if self._worker is None:
            raise RuntimeError("copy_from_global_domain requires an Orchestrator bound to a Worker")
        return self._worker._copy_from_global_domain(handle, int(domain_rank), int(nbytes), absolute)

    def create_worker_chip_region(self, *, worker_id: int, payload_bytes: int, counter_bytes: int):
        """Create an L3-L2 communication region on one NEXT_LEVEL chip worker."""
        if self._worker is None:
            raise RuntimeError("create_worker_chip_region requires an Orchestrator bound to a Worker")
        with self._control_admission("create_worker_chip_region"):
            return self._worker._create_worker_chip_region(int(worker_id), int(payload_bytes), int(counter_bytes))

    def create_worker_chip_queue(self, *, worker_id: int, depth: int, input_arena_bytes: int, output_arena_bytes: int):
        """Create an L3-L2 message queue backed by one L3-L2 communication region."""
        if self._worker is None:
            raise RuntimeError("create_worker_chip_queue requires an Orchestrator bound to a Worker")
        from .worker_chip_message_queue import create_worker_chip_queue  # noqa: PLC0415

        # Reserved across the whole build, not just the region creation it
        # nests: the descriptor writes that follow are device effects too. The
        # reservation is re-entrant, so the inner create_worker_chip_region joins this
        # one rather than deadlocking on it.
        with self._control_admission("create_worker_chip_queue"):
            return create_worker_chip_queue(
                self,
                worker_id=int(worker_id),
                depth=int(depth),
                input_arena_bytes=int(input_arena_bytes),
                output_arena_bytes=int(output_arena_bytes),
            )

    # ------------------------------------------------------------------
    # Nested scope (Strict-1 per-scope rings)
    # ------------------------------------------------------------------
    #
    # Tasks and allocations inside a nested ``with orch.scope():`` bind to a
    # deeper heap ring (``min(depth, MAX_RING_DEPTH-1)``) so their
    # memory reclaims independently of the outer scope. ``scope_end`` is
    # non-blocking — it releases scope refs and returns; call
    # the ``RunHandle`` returned by ``Worker.submit`` for a synchronous wait.
    #
    # Usage::
    #
    #     def my_orch(orch, args):
    #         with orch.scope():
    #             orch.submit_next_level(a, ..., worker=0)
    #             orch.submit_next_level(b, ..., worker=0)
    #         orch.submit_next_level(c, ..., worker=0)  # outer-scope ring

    def scope_begin(self) -> None:
        """Open a nested scope explicitly.

        Prefer the ``scope()`` context manager, which pairs the end for you. Every
        ``scope_begin()`` must be matched by a ``scope_end()``.
        """
        self._o.scope_begin()

    def scope_end(self) -> None:
        """Close the scope opened by the matching ``scope_begin()``."""
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

    def _control_admission(self, api: str):
        """Hold this call's ordering against runs for its whole duration.

        Entered before any Worker lock: the wait can be long, and a callback
        holding ``_child_prov_lock`` across it would block the very paths that
        let the active run finish.
        """
        return direct_control(self._worker, self._o, f"Orchestrator.{api}")

    def malloc(self, worker_id: int, size: int) -> int:
        """Allocate memory on next-level worker *worker_id*. Returns a pointer.

        This is the single L3 choke for kind4 device memory: ``Worker.malloc``
        also funnels through here, as does a user's direct ``orch.malloc``. The
        returned pointer's ``(worker_id, ptr)`` provenance is recorded so a later
        free / copy / kind4 dispatch to the wrong worker is rejected.
        """
        with self._control_admission("malloc"):
            wid, sz = int(worker_id), int(size)
            if self._worker is None:
                return int(self._o.malloc(wid, sz))
            with self._worker._child_prov_lock:
                ptr = int(self._o.malloc(wid, sz))
                self._worker._child_prov_record_malloc(wid, ptr, sz)
                return ptr

    def committed_device_memory(self, worker_id: int) -> int:
        """Total device HBM (bytes) committed by next-level worker *worker_id*'s ``MemoryAllocator``.

        A query, but one that travels the same chip mailbox as every other
        command, so it takes the same ordering: read behind a run that is still
        allocating and the number is a snapshot of neither side.
        """
        with self._control_admission("committed_device_memory"):
            return int(self._o.committed_device_memory(int(worker_id)))

    def free(self, worker_id: int, ptr: int) -> None:
        """Free memory on next-level worker *worker_id*."""
        with self._control_admission("free"):
            wid, p = int(worker_id), int(ptr)
            if self._worker is None:
                self._o.free(wid, p)
                return
            self._free_locked(wid, p)

    def _free_locked(self, wid: int, p: int) -> None:
        assert self._worker is not None
        with self._worker._child_prov_lock:
            # Safety-first commit barrier: revoke provenance BEFORE the native
            # free. If the native free succeeds and an async unwind (e.g. a
            # KeyboardInterrupt delivered after the binding returns) fires before
            # a post-free clear could run, a freed address would stay live and a
            # later copy/dispatch would re-authorize it — a UAF. Revoking first
            # turns a native-free failure into a terminal leak (recoverable) but
            # never re-authorizes a maybe-freed address.
            self._worker._child_prov_require_malloc_base(wid, p, api="free")
            self._worker._child_prov_clear_malloc(wid, p)
            self._o.free(wid, p)

    def copy_to(self, worker_id: int, dst: int, src: int, size: int) -> None:
        """Copy *size* bytes from host *src* to worker *dst*."""
        with self._control_admission("copy_to"):
            wid, d = int(worker_id), int(dst)
            if self._worker is None:
                self._o.copy_to(wid, d, int(src), int(size))
                return
            with self._worker._child_prov_lock:
                self._worker._child_prov_require_live_range(wid, d, int(size), api="copy_to")
                self._o.copy_to(wid, d, int(src), int(size))

    def copy_from(self, worker_id: int, dst: int, src: int, size: int) -> None:
        """Copy *size* bytes from worker *src* to host *dst*."""
        with self._control_admission("copy_from"):
            wid, s = int(worker_id), int(src)
            if self._worker is None:
                self._o.copy_from(wid, int(dst), s, int(size))
                return
            with self._worker._child_prov_lock:
                self._worker._child_prov_require_live_range(wid, s, int(size), api="copy_from")
                self._o.copy_from(wid, int(dst), s, int(size))

    def alloc(self, shape: Sequence[int], dtype: DataType) -> ChipTensor:
        """Allocate a runtime-managed intermediate buffer.

        Returns a ``ChipTensor`` whose backing memory comes from a
        per-allocation MAP_SHARED mmap (visible to forked child workers).
        Lifetime is bound to a synthetic task slot that the Orchestrator
        treats as the buffer's producer; the buffer is freed when all
        downstream consumers have completed and the run's scope ends.

        Use this for chip-A → chip-B intermediate buffers instead of
        pre-allocating with ``torch.share_memory_()`` — the runtime owns
        the lifecycle.
        """
        tensor = self._o.alloc(list(shape), dtype)
        if self._worker is not None:
            self._worker._register_worker_chip_orch_comm_host_buffer(tensor)
        return tensor

    # ------------------------------------------------------------------
    # Internal (called by Worker.submit)
    # ------------------------------------------------------------------

    def _scope_begin(self) -> None:
        self._o._scope_begin()

    def _scope_end(self) -> None:
        self._o._scope_end()

    def _begin_run(self) -> int:
        return int(self._o._begin_run())

    def _close_run_submission(self, run_id: int) -> None:
        self._o._close_run_submission(run_id)

    def _fail_run_submission(self, run_id: int, message: str = "") -> None:
        self._o._fail_run_submission(run_id, message)

    def _wait_run(self, run_id: int) -> None:
        self._o._wait_run(run_id)

    def _wait_run_accepted(self, run_id: int) -> None:
        self._o._wait_run_accepted(run_id)

    def _run_accepted(self, run_id: int) -> bool:
        return bool(self._o._run_accepted(run_id))

    def _wait_run_for(self, run_id: int, timeout_seconds: float) -> bool:
        return bool(self._o._wait_run_for(run_id, timeout_seconds))

    def _run_done(self, run_id: int) -> bool:
        return bool(self._o._run_done(run_id))

    def _release_run(self, run_id: int) -> None:
        self._o._release_run(run_id)
