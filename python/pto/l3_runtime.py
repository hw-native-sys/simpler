"""L3 (single-host, multi-chip) runtime implementation.

Manages per-chip worker processes, builds/executes a task DAG from a
Python orchestration function, and routes tasks to ChipWorkers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from .types import Arg, CompiledPackage, ParamType
from .l3_context import L3OrchestratorContext
from .l3_worker import ChipWorker
from .dag import TaskNode

logger = logging.getLogger(__name__)


class L3Runtime:
    """Multi-chip runtime for a single host.

    Lifecycle:
        1. ``__init__`` — configure platform and device list
        2. ``register()`` — register named computations (Python orch + L2 packages)
        3. ``run()`` — execute: call orch function → build DAG → dispatch to workers
        4. ``close()`` — shut down workers
    """

    def __init__(self, platform: str, devices: list[int]):
        self._platform = platform
        self._devices = list(devices)
        self._kernel_registry: dict[str, CompiledPackage] = {}
        self._orch_registry: dict[str, Callable] = {}
        self._workers: dict[int, ChipWorker] = {}
        self._started = False

    def register(self, name: str, *,
                 orch: Callable = None,
                 kernels: dict[str, CompiledPackage] = None,
                 **kwargs) -> None:
        """Register a named multi-chip computation.

        Args:
            name: Computation name
            orch: Python orchestration function ``f(ctx, args)``
            kernels: Dict mapping kernel names to pre-compiled L2 packages
        """
        if orch is not None:
            self._orch_registry[name] = orch
        if kernels:
            self._kernel_registry.update(kernels)

    def run(self, name: str, args: Any = None) -> Any:
        """Execute a registered multi-chip computation.

        Steps:
            1. Ensure workers are started
            2. Create L3OrchestratorContext
            3. Call the Python orch function — this populates the DAG
            4. Dispatch ready tasks to chip workers
            5. Wait for all tasks to complete
        """
        if name not in self._orch_registry:
            raise KeyError(f"No registered orchestration '{name}'. "
                           f"Available: {list(self._orch_registry.keys())}")

        self._ensure_workers()

        ctx = L3OrchestratorContext(
            device_ids=self._devices,
            kernel_registry=self._kernel_registry,
        )

        orch_func = self._orch_registry[name]
        result = orch_func(ctx, args)

        self._execute_dag(ctx)

        return result

    def close(self) -> None:
        """Shut down all worker processes."""
        for worker in self._workers.values():
            worker.stop()
        for worker in self._workers.values():
            worker.join(timeout=10.0)
        self._workers.clear()
        self._started = False
        self._kernel_registry.clear()
        self._orch_registry.clear()

    def _ensure_workers(self) -> None:
        """Start worker processes if not already running."""
        if self._started:
            return

        for device_id in self._devices:
            worker = ChipWorker(device_id=device_id)
            worker.start()
            self._workers[device_id] = worker

        self._started = True
        logger.info(f"Started {len(self._workers)} chip workers: {self._devices}")

    def _execute_dag(self, ctx: L3OrchestratorContext) -> None:
        """Dispatch all tasks in the DAG to workers and wait for completion."""
        dag = ctx.dag

        if dag.task_count == 0:
            return

        in_flight: dict[int, int] = {}  # task_id → chip

        ready = dag.get_ready_tasks()
        for node in ready:
            self._dispatch_task(node, ctx, in_flight)

        while not dag.all_complete():
            for device_id, worker in self._workers.items():
                result = worker.poll_result(timeout=0.01)
                if result is None:
                    continue

                task_id, success, error = result
                if not success:
                    raise RuntimeError(
                        f"Task {task_id} failed on chip {in_flight.get(task_id, '?')}: {error}")

                in_flight.pop(task_id, None)

                newly_ready = dag.complete(task_id)
                for node in newly_ready:
                    self._dispatch_task(node, ctx, in_flight)

    def _dispatch_task(self, node: TaskNode, ctx: L3OrchestratorContext,
                       in_flight: dict[int, int]) -> None:
        """Send a task to the appropriate chip worker."""
        pkg = ctx.get_kernel_package(node.kernel)

        func_args, arg_types, arg_sizes = _args_to_raw(node.args)

        task_spec = {
            "host_binary": pkg.host_binary,
            "orch_binary": pkg.orch_binary,
            "orch_func": pkg.orch_func,
            "func_args": func_args,
            "arg_types": arg_types,
            "arg_sizes": arg_sizes,
            "kernel_binaries": pkg.kernel_binaries,
            "aicpu_thread_num": pkg.aicpu_thread_num,
            "block_dim": pkg.block_dim,
            "aicpu_binary": pkg.aicpu_binary,
            "aicore_binary": pkg.aicore_binary,
            "orch_thread_num": pkg.orch_thread_num,
        }

        if node.is_group:
            for chip in node.group_chips:
                worker = self._workers.get(chip)
                if worker is None:
                    raise RuntimeError(f"No worker for chip {chip}")
                worker.send_task(node.task_id, task_spec)
        else:
            worker = self._workers.get(node.chip)
            if worker is None:
                raise RuntimeError(f"No worker for chip {node.chip}")
            worker.send_task(node.task_id, task_spec)

        ctx.dag.mark_dispatched(node.task_id)
        in_flight[node.task_id] = node.chip
        logger.debug(f"Dispatched task {node.task_id} (kernel={node.kernel}) to chip {node.chip}")


def _args_to_raw(args: list[Arg]) -> tuple[list, list[int], list[int]]:
    """Convert Arg list to parallel arrays for the worker task spec."""
    import numpy as np

    func_args = []
    arg_types = []
    arg_sizes = []

    for arg in args:
        arg_types.append(int(arg.type))
        if arg.type == ParamType.SCALAR:
            func_args.append(int(arg.data))
            arg_sizes.append(0)
        else:
            arr = arg.data
            if isinstance(arr, np.ndarray):
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
                func_args.append(arr.ctypes.data)
                arg_sizes.append(arr.nbytes)
            else:
                func_args.append(0)
                arg_sizes.append(arg.size)

    return func_args, arg_types, arg_sizes
