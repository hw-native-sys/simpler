"""L3OrchestratorContext — the ``ctx`` object passed to Python orchestration functions.

Provides the API that L3 orchestration code calls: peers(), submit(),
submit_group(), alloc(), scope_begin/end.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .types import Arg, CompiledPackage, ParamType, TensorHandle
from .dag import TaskDAG, TaskNode


class L3OrchestratorContext:
    """Runtime context for L3 Python orchestration functions.

    The orchestration function receives this as ``ctx`` and uses it to
    discover chips and submit tasks.
    """

    def __init__(self, device_ids: list[int],
                 kernel_registry: dict[str, CompiledPackage]):
        self._device_ids = list(device_ids)
        self._kernel_registry = kernel_registry
        self._dag = TaskDAG()
        self._next_handle_id = 1
        self._handles: dict[int, TensorHandle] = {}

    @property
    def dag(self) -> TaskDAG:
        return self._dag

    def peers(self) -> list[int]:
        """Return available chip IDs."""
        return list(self._device_ids)

    def alloc(self, shape: tuple, dtype: str = "float32") -> TensorHandle:
        """Allocate a host-memory tensor and return a handle."""
        arr = np.zeros(shape, dtype=dtype)
        handle_id = self._next_handle_id
        self._next_handle_id += 1
        handle = TensorHandle(
            id=handle_id,
            shape=shape,
            dtype=dtype,
            size=arr.nbytes,
            _data=arr,
        )
        self._handles[handle_id] = handle
        return handle

    def submit(self, chip: int, kernel: str, args: list[Arg]) -> TaskNode:
        """Submit a single-chip task to the DAG.

        Args:
            chip: Target chip ID
            kernel: Name of a registered L2 package
            args: List of Arg descriptors
        """
        if chip not in self._device_ids:
            raise ValueError(f"Chip {chip} not in available devices {self._device_ids}")
        if kernel not in self._kernel_registry:
            raise KeyError(f"Kernel '{kernel}' not registered. "
                           f"Available: {list(self._kernel_registry.keys())}")
        return self._dag.add_task(chip=chip, kernel=kernel, args=args)

    def submit_group(self, chips: list[int], kernel: str,
                     args: list[Arg]) -> TaskNode:
        """Submit a group task (multiple chips as one logical DAG node).

        All chips execute the same kernel in parallel. Used for collective
        operations (allreduce, etc.) where chips communicate via P2P.
        """
        for c in chips:
            if c not in self._device_ids:
                raise ValueError(f"Chip {c} not in available devices")
        if kernel not in self._kernel_registry:
            raise KeyError(f"Kernel '{kernel}' not registered")

        return self._dag.add_task(
            chip=chips[0],  # representative chip
            kernel=kernel,
            args=args,
            is_group=True,
            group_chips=chips,
        )

    def scope_begin(self) -> None:
        pass  # placeholder for future scope tracking

    def scope_end(self) -> None:
        pass

    def get_kernel_package(self, name: str) -> CompiledPackage:
        return self._kernel_registry[name]
