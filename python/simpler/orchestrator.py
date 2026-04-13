# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestrator — DAG builder exposed to the user's orch function during Worker.run().

An Orchestrator instance is Worker's private member. Users receive it as the
first argument of their orch function::

    def my_orch(orch, args):
        r = orch.submit_next_level(chip_callable, chip_args_ptr, config, outputs=[64])
        orch.submit_sub(cid, inputs=[r.outputs[0].ptr])

    w.run(Task(orch=my_orch, args=my_args))

Scope/drain lifecycle is managed by ``Worker.run()``; users never call those
directly.
"""

from typing import Any, Optional

from .task_interface import (
    DistInputSpec,
    DistOutputSpec,
    DistWorker,
    WorkerPayload,
    WorkerType,
)


def _resolve_callable_ptr(callable_: Any) -> int:
    """Accept either a ChipCallable (has buffer_ptr()) or a raw int pointer."""
    if hasattr(callable_, "buffer_ptr"):
        return callable_.buffer_ptr()
    return int(callable_)


class Orchestrator:
    """DAG builder. Valid only inside the orch function passed to Worker.run()."""

    def __init__(self, dist_worker: DistWorker) -> None:
        self._dw = dist_worker

    # ------------------------------------------------------------------
    # User-facing submit API
    # ------------------------------------------------------------------

    def submit_next_level(
        self,
        callable_: Any,
        args: int = 0,
        config: Optional[Any] = None,
        *,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ):
        """Submit a next-level (chip) task.

        Args:
            callable_: ChipCallable or raw callable pointer (int).
            args: Pointer to ChipStorageTaskArgs (int).
            config: ChipCallConfig-like (reads block_dim, aicpu_thread_num,
                enable_profiling). If None, defaults apply.
            inputs: List of input pointers for dependency inference.
            outputs: List of output byte sizes to allocate.
        """
        p = self._build_next_level_payload(callable_, args, config)
        in_specs = [DistInputSpec(x) for x in (inputs or [])]
        out_specs = [DistOutputSpec(s) for s in (outputs or [])]
        return self._dw.submit(WorkerType.NEXT_LEVEL, p, in_specs, out_specs)

    def submit_next_level_group(
        self,
        callable_: Any,
        args_list: list[int],
        config: Optional[Any] = None,
        *,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ):
        """Submit a group of next-level tasks (N args → N workers, 1 DAG node)."""
        p = self._build_next_level_payload(callable_, 0, config)
        in_specs = [DistInputSpec(x) for x in (inputs or [])]
        out_specs = [DistOutputSpec(s) for s in (outputs or [])]
        return self._dw.submit_group(WorkerType.NEXT_LEVEL, p, args_list, in_specs, out_specs)

    def submit_sub(
        self,
        callable_id: int,
        *,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ):
        """Submit a SUB task by registered callable id."""
        p = WorkerPayload()
        p.worker_type = WorkerType.SUB
        p.callable_id = callable_id
        in_specs = [DistInputSpec(x) for x in (inputs or [])]
        out_specs = [DistOutputSpec(s) for s in (outputs or [])]
        return self._dw.submit(WorkerType.SUB, p, in_specs, out_specs)

    def submit_sub_group(
        self,
        callable_id: int,
        args_list: list[int],
        *,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
    ):
        """Submit a group of SUB tasks (N args → N workers, 1 DAG node)."""
        p = WorkerPayload()
        p.worker_type = WorkerType.SUB
        p.callable_id = callable_id
        in_specs = [DistInputSpec(x) for x in (inputs or [])]
        out_specs = [DistOutputSpec(s) for s in (outputs or [])]
        return self._dw.submit_group(WorkerType.SUB, p, args_list, in_specs, out_specs)

    # ------------------------------------------------------------------
    # Internal (called by Worker.run)
    # ------------------------------------------------------------------

    def _scope_begin(self) -> None:
        self._dw.scope_begin()

    def _scope_end(self) -> None:
        self._dw.scope_end()

    def _drain(self) -> None:
        self._dw.drain()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_next_level_payload(callable_: Any, args: int, config: Optional[Any]) -> WorkerPayload:
        p = WorkerPayload()
        p.worker_type = WorkerType.NEXT_LEVEL
        p.callable = _resolve_callable_ptr(callable_)
        p.args = int(args)
        if config is not None:
            p.block_dim = int(config.block_dim)
            p.aicpu_thread_num = int(config.aicpu_thread_num)
            p.enable_profiling = bool(getattr(config, "enable_profiling", False))
        return p
