# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SceneTestCase framework — unified scene test infrastructure.

``@scene_test`` decorator + ``SceneTestCase`` base class.
pytest: ``pytest --platform a2a3sim``
standalone: ``python test_xxx.py -p a2a3sim``

A scene test class declares three things:
  CALLABLE: what to compile/register
    L2: orchestration (C++ source) + incores (C++ kernels)
    L3: orchestration (Python DAG fn) + callables (ChipCallable + SubCallable)
  CASES: how to run (per-case platform, config, params)
  generate_args / compute_golden: data + golden comparison
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from .log_config import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging
from .pto_isa import ensure_pto_isa_root

logger = logging.getLogger(__name__)

_compile_cache: dict[tuple[str, str, str], object] = {}


def clear_compile_cache() -> None:
    """Drop every cached ``ChipCallable`` and force a GC pass.

    The cache keeps nanobind-owned ``ChipCallable`` instances alive for the
    whole pytest session. Module-level dicts are cleared by Python in an
    order that can outlive the nanobind module destructor, which then
    prints ``leaked N instances of type _task_interface.ChipCallable`` to
    stderr at interpreter shutdown. Call this from ``pytest_sessionfinish``
    (and other session-end paths) so the instances die while the nanobind
    module is still wired up.
    """
    _compile_cache.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


class Tensor(NamedTuple):
    """Tensor argument spec."""

    name: str
    value: Any  # torch.Tensor


class Scalar(NamedTuple):
    """Scalar argument spec (ctypes scalar)."""

    name: str
    value: Any  # ctypes.c_float, ctypes.c_int64, etc.


# ---------------------------------------------------------------------------
# TaskArgsBuilder — ordered container with named access
# ---------------------------------------------------------------------------


class TaskArgsBuilder:
    """Test-side task arguments container.

    Maintains insertion order (tensors before scalars) and provides
    attribute access by name for use in compute_golden.

    Usage::

        args = TaskArgsBuilder(
            Tensor("a", torch.full((N,), 2.0)),
            Tensor("b", torch.full((N,), 3.0)),
            Tensor("f", torch.zeros(N)),
            Scalar("scale", ctypes.c_float(1.5)),
        )
        args.a  # → tensor
        args.f[:] = args.a + args.b  # in compute_golden
    """

    def __init__(self, *specs):
        self._specs: list = []
        self._data: dict[str, Any] = {}
        self._has_scalar = False
        for spec in specs:
            if isinstance(spec, Tensor):
                self._add_tensor(spec)
            elif isinstance(spec, Scalar):
                self._add_scalar(spec)

    def add_tensor(self, name: str, value: Any) -> None:
        """Add a tensor. Must be called before any add_scalar."""
        self._add_tensor(Tensor(name, value))

    def add_scalar(self, name: str, value: Any) -> None:
        """Add a scalar. After this, add_tensor is not allowed."""
        self._add_scalar(Scalar(name, value))

    def _add_tensor(self, spec: Tensor) -> None:
        if self._has_scalar:
            raise ValueError("Cannot add tensor after scalar (tensor-before-scalar ordering required)")
        self._specs.append(spec)
        self._data[spec.name] = spec.value

    def _add_scalar(self, spec: Scalar) -> None:
        self._has_scalar = True
        self._specs.append(spec)
        self._data[spec.name] = spec.value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"TaskArgsBuilder has no argument '{name}'") from None

    def clone(self) -> TaskArgsBuilder:
        """Deep clone: all tensors are cloned, scalars copied."""
        import torch  # noqa: PLC0415

        new = TaskArgsBuilder.__new__(TaskArgsBuilder)
        new._specs = []
        new._data = {}
        new._has_scalar = False
        for spec in self._specs:
            if isinstance(spec, Tensor):
                cloned = spec.value.clone() if isinstance(spec.value, torch.Tensor) else spec.value
                new_spec = Tensor(spec.name, cloned)
                new._specs.append(new_spec)
                new._data[spec.name] = cloned
            elif isinstance(spec, Scalar):
                import copy  # noqa: PLC0415

                new._has_scalar = True
                cloned_val = copy.copy(spec.value)
                new_spec = Scalar(spec.name, cloned_val)
                new._specs.append(new_spec)
                new._data[spec.name] = cloned_val
        return new

    @property
    def specs(self) -> list:
        """Ordered list of Tensor/Scalar specs."""
        return self._specs

    def tensor_names(self) -> list[str]:
        """Names of all tensor arguments, in order."""
        return [s.name for s in self._specs if isinstance(s, Tensor)]


# ---------------------------------------------------------------------------
# CallableNamespace — dot-access container for L3 callables
# ---------------------------------------------------------------------------


class CallableNamespace:
    """Dot-access container for compiled/registered callables.

    Used by L3 orch functions to access callables by name::

        callables.vector_kernel       # → ChipCallable object
        callables.vector_kernel_sig   # → signature list
        callables.verify              # → callable_id (int)

    Also provides ``keep()`` for lifetime management: L3 orch functions
    that build transient Python objects (e.g. ChipStorageTaskArgs) whose
    raw pointers are submitted to the C++ scheduler must register them
    via ``keep()`` so they outlive the scheduler drain::

        def run_dag(w, callables, task_args, config):
            chip_args, _ = _build_chip_task_args(task_args, callables.vector_kernel_sig)
            callables.keep(chip_args)  # survive until drain finishes
            ...
    """

    def __init__(self, entries: dict):
        self._entries = dict(entries)
        self._keepalive: list = []

    def __getattr__(self, name: str):
        try:
            return self._entries[name]
        except KeyError:
            raise AttributeError(f"CallableNamespace has no entry '{name}'") from None

    def keep(self, *objs):
        """Register objects to keep alive until this namespace is destroyed."""
        if not objs:
            return None
        self._keepalive.extend(objs)
        return objs[0] if len(objs) == 1 else objs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chip_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build `ChipStorageTaskArgs` (POD) from `TaskArgsBuilder`.

    Used by the L2 path (`ChipWorker.run(callable, chip_args, config)`): the
    chip worker expects the runtime.so ABI-shaped POD directly (no tags).

    Returns:
        chip_args: ChipStorageTaskArgs (POD)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipStorageTaskArgs,
        scalar_to_uint64,
    )

    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    chip_args = ChipStorageTaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
            chip_args.add_tensor(make_tensor_arg(spec.value))
            if direction in (ArgDirection.OUT, ArgDirection.INOUT):
                output_names.append(spec.name)
            tensor_idx += 1
        elif isinstance(spec, Scalar):
            chip_args.add_scalar(scalar_to_uint64(spec.value))

    return chip_args, output_names


def _build_l3_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build a tagged `TaskArgs` (vector-backed, with `TensorArgType` tags) from
    `TaskArgsBuilder`.

    Used by the L3 path (`orch.submit_next_level(callable, args, config)`):
    the orchestrator reads the tags to drive dependency inference.

    Returns:
        chip_args: TaskArgs (tagged)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        TaskArgs,
        TensorArgType,
        scalar_to_uint64,
    )

    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    _DIR_TO_TAG = {
        ArgDirection.IN: TensorArgType.INPUT,
        ArgDirection.OUT: TensorArgType.OUTPUT_EXISTING,
        ArgDirection.INOUT: TensorArgType.INOUT,
    }

    chip_args = TaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
            tag = _DIR_TO_TAG.get(direction, TensorArgType.INPUT)
            chip_args.add_tensor(make_tensor_arg(spec.value), tag)
            if direction in (ArgDirection.OUT, ArgDirection.INOUT):
                output_names.append(spec.name)
            tensor_idx += 1
        elif isinstance(spec, Scalar):
            chip_args.add_scalar(scalar_to_uint64(spec.value))

    return chip_args, output_names


def _output_names_from_signature(test_args: TaskArgsBuilder, signature: list) -> list[str]:
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    output_names: list[str] = []
    tensor_idx = 0
    for spec in test_args.specs:
        if not isinstance(spec, Tensor):
            continue
        if tensor_idx >= len(signature):
            raise ValueError(
                f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                f"callable signature (length {len(signature)})."
            )
        if signature[tensor_idx] in (ArgDirection.OUT, ArgDirection.INOUT):
            output_names.append(spec.name)
        tensor_idx += 1
    return output_names


def _callable_signature(callable_spec: dict) -> list:
    cuda = callable_spec.get("cuda")
    if isinstance(cuda, dict) and "signature" in cuda:
        return cuda.get("signature", [])
    return callable_spec.get("orchestration", {}).get("signature", [])


def _tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


class _CudaSceneDeviceBuffers:
    def __init__(self, worker, test_args: TaskArgsBuilder):
        self.worker = worker
        self.test_args = test_args
        self.ptrs: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        self._allocate()

    def _allocate(self) -> None:
        for spec in self.test_args.specs:
            if not isinstance(spec, Tensor):
                continue
            tensor = spec.value
            device = getattr(tensor, "device", None)
            if device is not None and getattr(device, "type", "cpu") != "cpu":
                raise ValueError(f"CUDA scene-test tensor '{spec.name}' must be a CPU tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"CUDA scene-test tensor '{spec.name}' must be contiguous")
            nbytes = _tensor_nbytes(tensor)
            if nbytes <= 0:
                raise ValueError(f"CUDA scene-test tensor '{spec.name}' must be non-empty")
            self.ptrs[spec.name] = int(self.worker.malloc(nbytes))
            self.sizes[spec.name] = nbytes

    def copy_all_to_device(self) -> None:
        for spec in self.test_args.specs:
            if isinstance(spec, Tensor):
                self.worker.copy_to(self.ptrs[spec.name], int(spec.value.data_ptr()), self.sizes[spec.name])

    def copy_outputs_from_device(self, output_names: list[str]) -> None:
        for name in output_names:
            tensor = getattr(self.test_args, name)
            self.worker.copy_from(int(tensor.data_ptr()), self.ptrs[name], self.sizes[name])

    def free(self) -> None:
        for ptr in self.ptrs.values():
            self.worker.free(ptr)
        self.ptrs.clear()
        self.sizes.clear()


def _build_cuda_host_schedule_args(
    test_args: TaskArgsBuilder,
    device_buffers: _CudaSceneDeviceBuffers,
    cuda_spec: dict,
):
    import ctypes  # noqa: PLC0415

    from simpler_setup.cuda_callable_compiler import (  # noqa: PLC0415
        CudaVectorAddArgs,
        CudaVectorAffineArgs,
        CudaVectorAxpyArgs,
        CudaVectorGenericArgs,
        CudaVectorQuaternaryArgs,
        CudaVectorScaleArgs,
        CudaVectorTernaryArgs,
        CudaVectorUnaryArgs,
    )

    arg_builder = cuda_spec.get("arg_builder", "vector_add_f32")
    supported_builders = {
        "vector_add_f32",
        "elementwise_binary_f32",
        "elementwise_unary_f32",
        "elementwise_scale_f32",
        "elementwise_axpy_f32",
        "elementwise_affine_f32",
        "elementwise_triad_f32",
        "elementwise_quad_f32",
        "elementwise_generic_args_f32",
    }
    if arg_builder not in supported_builders:
        raise NotImplementedError(f"Unsupported CUDA scene-test arg_builder: {arg_builder}")

    tensor_names = [spec.name for spec in test_args.specs if isinstance(spec, Tensor)]
    default_arg_count = 2 if arg_builder == "elementwise_unary_f32" else 3
    names = list(cuda_spec.get("args", tensor_names[:default_arg_count]))
    expected_arg_count = {
        "elementwise_unary_f32": 2,
        "elementwise_axpy_f32": 4,
        "elementwise_affine_f32": 5,
        "elementwise_triad_f32": 4,
        "elementwise_quad_f32": 5,
        "elementwise_generic_args_f32": 3,
    }.get(arg_builder, 3)
    if len(names) != expected_arg_count:
        raise ValueError(f"CUDA {arg_builder} scene tests require exactly {expected_arg_count} args")
    tensor_arg_count = {
        "elementwise_scale_f32": 2,
        "elementwise_axpy_f32": 3,
        "elementwise_affine_f32": 3,
        "elementwise_generic_args_f32": 3,
    }.get(arg_builder, expected_arg_count)
    missing = [name for name in names[:tensor_arg_count] if name not in device_buffers.ptrs]
    if missing:
        raise ValueError(f"CUDA {arg_builder} args reference unknown tensors: {', '.join(missing)}")

    n = int(cuda_spec.get("n", getattr(test_args, names[0]).numel()))
    if arg_builder == "elementwise_scale_f32":
        alpha = getattr(test_args, names[2])
        alpha_value = alpha.value if hasattr(alpha, "value") else alpha
        return CudaVectorScaleArgs(
            a=device_buffers.ptrs[names[0]],
            out=device_buffers.ptrs[names[1]],
            alpha=float(alpha_value),
            n=n,
        )
    if arg_builder == "elementwise_unary_f32":
        return CudaVectorUnaryArgs(
            a=device_buffers.ptrs[names[0]],
            out=device_buffers.ptrs[names[1]],
            n=n,
        )
    if arg_builder == "elementwise_axpy_f32":
        alpha = getattr(test_args, names[3])
        alpha_value = alpha.value if hasattr(alpha, "value") else alpha
        return CudaVectorAxpyArgs(
            a=device_buffers.ptrs[names[0]],
            b=device_buffers.ptrs[names[1]],
            out=device_buffers.ptrs[names[2]],
            alpha=float(alpha_value),
            n=n,
        )
    if arg_builder == "elementwise_affine_f32":
        alpha = getattr(test_args, names[3])
        beta = getattr(test_args, names[4])
        alpha_value = alpha.value if hasattr(alpha, "value") else alpha
        beta_value = beta.value if hasattr(beta, "value") else beta
        return CudaVectorAffineArgs(
            a=device_buffers.ptrs[names[0]],
            b=device_buffers.ptrs[names[1]],
            out=device_buffers.ptrs[names[2]],
            alpha=float(alpha_value),
            beta=float(beta_value),
            n=n,
        )
    if arg_builder == "elementwise_triad_f32":
        return CudaVectorTernaryArgs(
            a=device_buffers.ptrs[names[0]],
            b=device_buffers.ptrs[names[1]],
            c=device_buffers.ptrs[names[2]],
            out=device_buffers.ptrs[names[3]],
            n=n,
        )
    if arg_builder == "elementwise_quad_f32":
        return CudaVectorQuaternaryArgs(
            a=device_buffers.ptrs[names[0]],
            b=device_buffers.ptrs[names[1]],
            c=device_buffers.ptrs[names[2]],
            d=device_buffers.ptrs[names[3]],
            out=device_buffers.ptrs[names[4]],
            n=n,
        )
    if arg_builder == "elementwise_generic_args_f32":
        tensor_arg_names = list(cuda_spec.get("tensor_args", []))
        scalar_arg_names = list(cuda_spec.get("scalar_args", []))
        if len(tensor_arg_names) > 4:
            raise ValueError("CUDA elementwise_generic_args_f32 supports at most four tensor_args")
        if len(scalar_arg_names) > 4:
            raise ValueError("CUDA elementwise_generic_args_f32 supports at most four scalar_args")
        missing_tensor_args = [name for name in tensor_arg_names if name not in device_buffers.ptrs]
        if missing_tensor_args:
            raise ValueError(
                "CUDA elementwise_generic_args_f32 tensor_args reference unknown tensors: "
                + ", ".join(missing_tensor_args)
            )
        tensor_args_t = ctypes.c_void_p * 4
        scalar_args_t = ctypes.c_float * 4
        tensor_arg_ptrs = [device_buffers.ptrs[name] for name in tensor_arg_names]
        scalar_args = []
        for name in scalar_arg_names:
            scalar = getattr(test_args, name)
            scalar_value = scalar.value if hasattr(scalar, "value") else scalar
            scalar_args.append(float(scalar_value))
        return CudaVectorGenericArgs(
            a=device_buffers.ptrs[names[0]],
            b=device_buffers.ptrs[names[1]],
            out=device_buffers.ptrs[names[2]],
            tensor_args=tensor_args_t(*(tensor_arg_ptrs + [0] * (4 - len(tensor_arg_ptrs)))),
            scalar_args=scalar_args_t(*(scalar_args + [0.0] * (4 - len(scalar_args)))),
            tensor_arg_count=len(tensor_arg_ptrs),
            scalar_arg_count=len(scalar_args),
            n=n,
        )
    return CudaVectorAddArgs(
        a=device_buffers.ptrs[names[0]],
        b=device_buffers.ptrs[names[1]],
        out=device_buffers.ptrs[names[2]],
        n=n,
    )


class _CudaPersistentDagSceneBuffers:
    _GRAPH_NODE_IO_FIELDS = (
        "input",
        "inputs",
        "output",
        "outputs",
        "output_existing",
        "inout",
        "inouts",
        "scalar",
        "scalars",
    )

    def __init__(self, worker, test_args: TaskArgsBuilder, cuda_spec: dict):
        self.worker = worker
        self.test_args = test_args
        self.cuda_spec = cuda_spec
        self.tensor_buffers = _CudaSceneDeviceBuffers(worker, test_args)
        self.ptrs: list[int] = []
        self.args = None
        self.output_names = list(cuda_spec.get("args", test_args.tensor_names()[:3]))
        self._setup()

    def _malloc(self, nbytes: int) -> int:
        ptr = int(self.worker.malloc(nbytes))
        self.ptrs.append(ptr)
        return ptr

    def _setup(self) -> None:  # noqa: PLR0912, PLR0915
        import ctypes  # noqa: PLC0415

        from simpler_setup.cuda_callable_compiler import (  # noqa: PLC0415
            CudaPersistentDagArgs,
            CudaPersistentDagState,
            CudaPersistentDagTask,
        )

        arg_builder = self.cuda_spec.get("arg_builder")
        tensor_tile_builders = {
            "persistent_dag_tensor_tile_f32",
            "persistent_dag_tensor_core_tile_f32",
        }
        persistent_builders = {
            "persistent_dag_fork_join_f32",
            "persistent_dag_chain_f32",
            "persistent_dag_reuse_f32",
            "persistent_dag_scalar_affine_f32",
            "persistent_dag_scalar_axpy_f32",
            "persistent_dag_scalar_scale_f32",
            "persistent_dag_triad_f32",
            "persistent_dag_quad_f32",
            "persistent_dag_generic_args_f32",
            "persistent_dag_unary_square_f32",
            "persistent_dag_graph_f32",
        } | tensor_tile_builders
        if arg_builder not in persistent_builders:
            raise NotImplementedError(f"Unsupported CUDA persistent scene-test arg_builder: {arg_builder}")
        if arg_builder == "persistent_dag_graph_f32":
            expected_arg_count = None
        elif arg_builder in {"persistent_dag_quad_f32", "persistent_dag_generic_args_f32"}:
            expected_arg_count = 5
        elif arg_builder == "persistent_dag_triad_f32":
            expected_arg_count = 4
        else:
            expected_arg_count = 3
        if expected_arg_count is not None and len(self.output_names) != expected_arg_count:
            raise ValueError(f"CUDA {arg_builder} scene tests require {expected_arg_count} tensor args")
        missing = [name for name in self.output_names if name not in self.tensor_buffers.ptrs]
        if missing:
            raise ValueError(f"CUDA persistent DAG args reference unknown tensors: {', '.join(missing)}")

        if arg_builder == "persistent_dag_graph_f32":
            a_name = self.output_names[0]
            b_name = self.output_names[1] if len(self.output_names) > 1 else self.output_names[0]
            out_name = str(self.cuda_spec.get("output", self.output_names[-1]))
            c_name = None
            d_name = None
            if out_name not in self.tensor_buffers.ptrs:
                raise ValueError(f"CUDA persistent DAG graph output references unknown tensor: {out_name}")
        elif arg_builder in {"persistent_dag_quad_f32", "persistent_dag_generic_args_f32"}:
            a_name, b_name, c_name, d_name, out_name = self.output_names
        elif arg_builder == "persistent_dag_triad_f32":
            a_name, b_name, c_name, out_name = self.output_names
            d_name = None
        else:
            a_name, b_name, out_name = self.output_names
            c_name = None
            d_name = None
        n = int(self.cuda_spec.get("n", getattr(self.test_args, a_name).numel()))
        queue_capacity = int(self.cuda_spec.get("queue_capacity", 2))
        if queue_capacity <= 0:
            raise ValueError("CUDA persistent DAG queue_capacity must be positive")

        output_nbytes = self.tensor_buffers.sizes[out_name]
        self.dev_tmp0 = self._malloc(output_nbytes)
        self.dev_tmp1 = self._malloc(output_nbytes)

        if arg_builder == "persistent_dag_fork_join_f32":
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_chain_f32":
            self.dev_tmp2 = self._malloc(output_nbytes)
            self.dev_tmp3 = self._malloc(output_nbytes)
            dependents_t = ctypes.c_uint32 * 4
            self.host_dependents = dependents_t(2, 2, 3, 4)
            task_t = CudaPersistentDagTask * 5
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.dev_tmp2,
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp3,
                    n=n,
                    dependent_begin=3,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp2,
                    b=self.dev_tmp3,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=4,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 5)(0, 0, 2, 1, 1)
        elif arg_builder == "persistent_dag_reuse_f32":
            self.dev_tmp2 = self._malloc(output_nbytes)
            self.dev_tmp3 = self._malloc(output_nbytes)
            dependents_t = ctypes.c_uint32 * 6
            self.host_dependents = dependents_t(2, 2, 3, 4, 5, 5)
            task_t = CudaPersistentDagTask * 6
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=2,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.dev_tmp2,
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp3,
                    n=n,
                    dependent_begin=4,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp2,
                    b=self.tensor_buffers.ptrs[a_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=5,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp3,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=6,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 6)(0, 0, 2, 1, 1, 2)
        elif arg_builder == "persistent_dag_scalar_axpy_f32":
            scalar0 = float(self.cuda_spec.get("scalar0", 1.5))
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=4,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=scalar0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_scalar_scale_f32":
            scalar0 = float(self.cuda_spec.get("scalar0", 2.0))
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=11,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=0,
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=scalar0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_scalar_affine_f32":
            scalar0 = float(self.cuda_spec.get("scalar0", 1.5))
            scalar1 = float(self.cuda_spec.get("scalar1", 0.5))
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=5,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=scalar0,
                    scalar1=scalar1,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_triad_f32":
            assert c_name is not None
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=6,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    c=self.tensor_buffers.ptrs[c_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_quad_f32":
            assert c_name is not None
            assert d_name is not None
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=8,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    c=self.tensor_buffers.ptrs[c_name],
                    d=self.tensor_buffers.ptrs[d_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_generic_args_f32":
            assert c_name is not None
            assert d_name is not None
            tensor_arg_names = list(self.cuda_spec.get("tensor_args", [c_name, d_name]))
            scalar_arg_values = list(self.cuda_spec.get("scalar_args", [1.5, 0.25]))
            if len(tensor_arg_names) > 4:
                raise ValueError("CUDA persistent_dag_generic_args_f32 supports at most four tensor_args")
            if len(scalar_arg_values) > 4:
                raise ValueError("CUDA persistent_dag_generic_args_f32 supports at most four scalar_args")
            missing_tensor_args = [name for name in tensor_arg_names if name not in self.tensor_buffers.ptrs]
            if missing_tensor_args:
                raise ValueError(
                    "CUDA persistent_dag_generic_args_f32 tensor_args reference unknown tensors: "
                    + ", ".join(missing_tensor_args)
                )
            tensor_args_t = ctypes.c_void_p * 4
            scalar_args_t = ctypes.c_float * 4
            tensor_arg_ptrs = [self.tensor_buffers.ptrs[name] for name in tensor_arg_names]
            scalar_args = [float(value) for value in scalar_arg_values]
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(2, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=9,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    tensor_args=tensor_args_t(*(tensor_arg_ptrs + [0] * (4 - len(tensor_arg_ptrs)))),
                    scalar_args=scalar_args_t(*(scalar_args + [0.0] * (4 - len(scalar_args)))),
                    tensor_arg_count=len(tensor_arg_ptrs),
                    scalar_arg_count=len(scalar_args),
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.dev_tmp1,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 0, 2)
        elif arg_builder == "persistent_dag_unary_square_f32":
            dependents_t = ctypes.c_uint32 * 2
            self.host_dependents = dependents_t(1, 2)
            task_t = CudaPersistentDagTask * 3
            self.host_tasks = task_t(
                CudaPersistentDagTask(
                    func_id=7,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=0,
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp1,
                    b=self.tensor_buffers.ptrs[a_name],
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 3)(0, 1, 1)
        elif arg_builder == "persistent_dag_graph_f32":
            self._setup_graph_descriptor(ctypes, CudaPersistentDagTask, n, output_nbytes)
        else:
            descriptor = self._tensor_tile_descriptor(n)
            if arg_builder == "persistent_dag_tensor_core_tile_f32" and not self._is_tensor_core_tile(descriptor):
                raise ValueError(
                    "CUDA persistent_dag_tensor_core_tile_f32 requires rows/cols multiples of 16 "
                    "and inner divisible by 8"
                )
            tensor_func_id = 10 if arg_builder == "persistent_dag_tensor_core_tile_f32" else 3
            self.dev_tmp2 = self._malloc(output_nbytes)
            dependents_t = ctypes.c_uint32 * 4
            self.host_dependents = dependents_t(1, 2, 3, 3)
            task_t = CudaPersistentDagTask * 4
            self.host_tasks = task_t(
                self._make_tensor_tile_task(
                    CudaPersistentDagTask,
                    descriptor,
                    func_id=tensor_func_id,
                    a=self.tensor_buffers.ptrs[a_name],
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=2,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp0,
                    b=self.tensor_buffers.ptrs[a_name],
                    out=self.dev_tmp1,
                    n=n,
                    dependent_begin=2,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=self.dev_tmp0,
                    b=self.tensor_buffers.ptrs[b_name],
                    out=self.dev_tmp2,
                    n=n,
                    dependent_begin=3,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=self.dev_tmp1,
                    b=self.dev_tmp2,
                    out=self.tensor_buffers.ptrs[out_name],
                    n=n,
                    dependent_begin=4,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            )
            self.host_fanin = (ctypes.c_uint32 * 4)(0, 1, 1, 2)

        flags_t = ctypes.c_uint32 * queue_capacity
        counters_t = ctypes.c_uint32 * 11
        scheduler_processed_by_block_t = ctypes.c_uint32 * 1
        self.host_flags = flags_t(*([0] * queue_capacity))
        self.host_counters = counters_t(*([0] * 11))
        self.host_scheduler_processed_by_block = scheduler_processed_by_block_t(0)

        self.dev_tasks = self._malloc(ctypes.sizeof(self.host_tasks))
        self.dev_dependents = self._malloc(ctypes.sizeof(self.host_dependents))
        self.dev_fanin = self._malloc(ctypes.sizeof(self.host_fanin))
        self.dev_ready_queue = self._malloc(ctypes.sizeof(ctypes.c_uint32 * queue_capacity))
        self.dev_ready_flags = self._malloc(ctypes.sizeof(self.host_flags))
        self.dev_completion_queue = self._malloc(ctypes.sizeof(ctypes.c_uint32 * queue_capacity))
        self.dev_completion_flags = self._malloc(ctypes.sizeof(self.host_flags))
        self.dev_counters = self._malloc(ctypes.sizeof(self.host_counters))
        self.dev_scheduler_processed_by_block = self._malloc(ctypes.sizeof(self.host_scheduler_processed_by_block))
        self.dev_state = self._malloc(ctypes.sizeof(CudaPersistentDagState))

        self.worker.copy_to(self.dev_tasks, ctypes.addressof(self.host_tasks), ctypes.sizeof(self.host_tasks))
        self.worker.copy_to(
            self.dev_dependents,
            ctypes.addressof(self.host_dependents),
            ctypes.sizeof(self.host_dependents),
        )
        state = CudaPersistentDagState(
            tasks=self.dev_tasks,
            task_count=len(self.host_tasks),
            dependents=self.dev_dependents,
            dependent_count=len(self.host_dependents),
            fanin=self.dev_fanin,
            ready_queue=self.dev_ready_queue,
            ready_flags=self.dev_ready_flags,
            completion_queue=self.dev_completion_queue,
            completion_flags=self.dev_completion_flags,
            queue_capacity=queue_capacity,
            queue_head=self.dev_counters,
            queue_tail=self.dev_counters + ctypes.sizeof(ctypes.c_uint32),
            completion_head=self.dev_counters + 2 * ctypes.sizeof(ctypes.c_uint32),
            completion_tail=self.dev_counters + 3 * ctypes.sizeof(ctypes.c_uint32),
            completed_count=self.dev_counters + 4 * ctypes.sizeof(ctypes.c_uint32),
            error_count=self.dev_counters + 5 * ctypes.sizeof(ctypes.c_uint32),
            error_code=self.dev_counters + 6 * ctypes.sizeof(ctypes.c_uint32),
            error_task_id=self.dev_counters + 7 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_blocks=1,
            scheduler_init_count=self.dev_counters + 8 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_loop_count=self.dev_counters + 9 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_processed_count=self.dev_counters + 10 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_processed_by_block=self.dev_scheduler_processed_by_block,
        )
        self.worker.copy_to(self.dev_state, ctypes.addressof(state), ctypes.sizeof(state))
        self.args = CudaPersistentDagArgs(state=self.dev_state)

    def _tensor_tile_descriptor(self, n: int) -> dict[str, int]:
        descriptor = dict(self.cuda_spec.get("tensor_tile", {}))
        rows = int(descriptor.get("rows", 16))
        cols = int(descriptor.get("cols", 16))
        inner = int(descriptor.get("inner", 16))
        if rows <= 0 or cols <= 0 or inner <= 0:
            raise ValueError("CUDA tensor tile rows, cols, and inner must be positive")
        if n % (rows * cols) != 0:
            raise ValueError("CUDA persistent_dag_tensor_tile_f32 output size must be a multiple of rows*cols")

        a_batch_stride = int(descriptor.get("a_batch_stride", rows * inner))
        b_batch_stride = int(descriptor.get("b_batch_stride", inner * cols))
        out_batch_stride = int(descriptor.get("out_batch_stride", rows * cols))
        tile_count = n // (rows * cols)
        a_required = max(n, tile_count * a_batch_stride)
        b_required = max(n, tile_count * b_batch_stride)
        a_name, b_name, _ = self.output_names
        if self.tensor_buffers.sizes[a_name] < a_required * 4 or self.tensor_buffers.sizes[b_name] < b_required * 4:
            raise ValueError("CUDA tensor tile input tensors must cover matmul and elementwise extents")

        return {
            "rows": rows,
            "cols": cols,
            "inner": inner,
            "lda": int(descriptor.get("lda", inner)),
            "ldb": int(descriptor.get("ldb", cols)),
            "ldc": int(descriptor.get("ldc", cols)),
            "a_batch_stride": a_batch_stride,
            "b_batch_stride": b_batch_stride,
            "out_batch_stride": out_batch_stride,
        }

    @staticmethod
    def _is_tensor_core_tile(descriptor: dict[str, int]) -> bool:
        return (
            descriptor["rows"] > 0
            and descriptor["cols"] > 0
            and descriptor["inner"] > 0
            and descriptor["rows"] % 16 == 0
            and descriptor["cols"] % 16 == 0
            and descriptor["inner"] % 8 == 0
        )

    @staticmethod
    def _make_tensor_tile_task(task_type, descriptor: dict[str, int], **kwargs):
        task = task_type(**kwargs)
        for field, value in descriptor.items():
            setattr(task, field, value)
        return task

    def _setup_graph_descriptor(self, ctypes_module, task_type, n: int, output_nbytes: int) -> None:
        graph = self._graph_descriptor(self.cuda_spec)
        task_specs = self._graph_task_specs(graph)
        if not task_specs:
            raise ValueError("CUDA persistent_dag_graph_f32 requires at least one graph task")
        task_defaults = self._graph_task_defaults(graph)
        task_specs = [self._merge_graph_task_defaults(task_defaults, task_spec) for task_spec in task_specs]
        task_overrides = self._graph_task_overrides(graph)
        task_specs = self._merge_graph_task_overrides(task_overrides, task_specs)
        task_specs = [self._resolve_graph_task_callable(graph, task_spec) for task_spec in task_specs]
        task_specs = [self._normalize_graph_task_func_id(task_spec) for task_spec in task_specs]
        task_specs = [self._normalize_graph_task_spec(task_spec) for task_spec in task_specs]

        ptrs = dict(self.tensor_buffers.ptrs)
        temporary_items = list(dict(self.cuda_spec.get("temporaries", {})).items())
        temporary_count = 0

        def add_temporary(name, size_source) -> None:
            nonlocal temporary_count
            key = str(name)
            if key in ptrs:
                return
            if temporary_count == 0:
                ptr = self.dev_tmp0
            elif temporary_count == 1:
                ptr = self.dev_tmp1
            else:
                ptr = self._malloc(self._temporary_nbytes(size_source, output_nbytes))
            ptrs[key] = ptr
            temporary_count += 1

        for name, size_source in temporary_items:
            add_temporary(name, size_source)

        for task_spec in task_specs:
            self._bind_graph_task_output_storage(task_spec, ptrs, add_temporary, output_nbytes)

        graph_dependents = self._graph_dependents_from_task_specs(task_specs, self._graph_edges(graph))
        dependents: list[int] = []
        fanin = [0 for _ in task_specs]
        for task_id, task_spec in enumerate(task_specs):
            task_dependents = graph_dependents[task_id]
            for dependent in task_dependents:
                dependent_id = int(dependent)
                if dependent_id < 0 or dependent_id >= len(task_specs):
                    raise ValueError(
                        "CUDA persistent_dag_graph_f32 dependent task id "
                        f"{dependent_id} for task {task_id} is outside the graph"
                    )
                dependents.append(dependent_id)
                fanin[dependent_id] += 1

        for task_id, task_spec in enumerate(task_specs):
            if "initial_fanin" in task_spec:
                fanin[task_id] = int(task_spec["initial_fanin"])

        dependents_t = ctypes_module.c_uint32 * len(dependents)
        self.host_dependents = dependents_t(*dependents)
        task_t = task_type * len(task_specs)
        task_values = []
        dependent_begin = 0
        for task_id, task_spec in enumerate(task_specs):
            task_dependents = graph_dependents[task_id]
            task_values.append(
                self._make_graph_task(
                    ctypes_module,
                    task_type,
                    task_spec,
                    ptrs,
                    n,
                    dependent_begin,
                    len(task_dependents),
                    fanin[len(task_values)],
                )
            )
            dependent_begin += len(task_dependents)

        self.host_tasks = task_t(*task_values)
        fanin_t = ctypes_module.c_uint32 * len(fanin)
        self.host_fanin = fanin_t(*fanin)

    @staticmethod
    def _graph_descriptor(cuda_spec: dict[str, Any]) -> dict[str, Any]:
        graph = cuda_spec.get("graph")
        graph_path = cuda_spec.get("graph_path", cuda_spec.get("graph_file"))
        if graph_path is None and isinstance(graph, (str, os.PathLike)):
            graph_path = graph
            graph = None
        if graph_path is None:
            if not isinstance(graph, dict):
                raise ValueError("CUDA persistent_dag_graph_f32 requires a graph descriptor")
            return _CudaPersistentDagSceneBuffers._graph_with_task_sidecar_files(cuda_spec, graph)

        graph_file_path = Path(graph_path)
        with graph_file_path.open(encoding="utf-8") as graph_file:
            loaded_graph = json.load(graph_file)
        if not isinstance(loaded_graph, dict):
            raise ValueError("CUDA persistent_dag_graph_f32 graph file must contain a JSON object")
        if graph is None:
            return _CudaPersistentDagSceneBuffers._graph_with_task_sidecar_files(
                cuda_spec,
                loaded_graph,
                graph_file_path.parent,
            )
        if not isinstance(graph, dict):
            raise ValueError("CUDA persistent_dag_graph_f32 graph overlay must be a dictionary")
        merged_graph = {**loaded_graph, **graph}
        return _CudaPersistentDagSceneBuffers._graph_with_task_sidecar_files(
            cuda_spec,
            merged_graph,
            graph_file_path.parent,
        )

    @staticmethod
    def _graph_with_task_sidecar_files(
        cuda_spec: dict[str, Any],
        graph: dict[str, Any],
        graph_dir: Path | None = None,
    ) -> dict[str, Any]:
        merged_graph = dict(graph)
        defaults_path_keys = (
            "task_defaults_path",
            "task_defaults_file",
            "task_template_path",
            "task_template_file",
            "default_task_path",
            "default_task_file",
        )
        default_defaults_path = next((cuda_spec[key] for key in defaults_path_keys if key in cuda_spec), None)
        defaults_path = default_defaults_path
        for key in defaults_path_keys:
            if key in merged_graph:
                defaults_path = merged_graph.pop(key)
                break
        if defaults_path is not None:
            with _CudaPersistentDagSceneBuffers._graph_sidecar_path(defaults_path, graph_dir).open(
                encoding="utf-8"
            ) as defaults_file:
                merged_graph["task_defaults"] = json.load(defaults_file)
        default_metadata_path = cuda_spec.get("task_metadata_path", cuda_spec.get("task_metadata_file"))
        metadata_path = merged_graph.pop(
            "task_metadata_path",
            merged_graph.pop("task_metadata_file", default_metadata_path),
        )
        if metadata_path is not None:
            with _CudaPersistentDagSceneBuffers._graph_sidecar_path(metadata_path, graph_dir).open(
                encoding="utf-8"
            ) as metadata_file:
                merged_graph["task_metadata"] = json.load(metadata_file)
        default_overrides_path = cuda_spec.get("task_overrides_path", cuda_spec.get("task_overrides_file"))
        overrides_path = merged_graph.pop(
            "task_overrides_path",
            merged_graph.pop("task_overrides_file", default_overrides_path),
        )
        if overrides_path is not None:
            with _CudaPersistentDagSceneBuffers._graph_sidecar_path(overrides_path, graph_dir).open(
                encoding="utf-8"
            ) as overrides_file:
                merged_graph["task_overrides"] = json.load(overrides_file)
        default_callables_path = cuda_spec.get("callables_path", cuda_spec.get("callables_file"))
        callables_path = merged_graph.pop(
            "callables_path",
            merged_graph.pop("callables_file", default_callables_path),
        )
        if callables_path is not None:
            with _CudaPersistentDagSceneBuffers._graph_sidecar_path(callables_path, graph_dir).open(
                encoding="utf-8"
            ) as callables_file:
                merged_graph["callables"] = json.load(callables_file)
        return merged_graph

    @staticmethod
    def _graph_sidecar_path(path_value: Any, graph_dir: Path | None) -> Path:
        sidecar_path = Path(path_value)
        if graph_dir is not None and not sidecar_path.is_absolute():
            return graph_dir / sidecar_path
        return sidecar_path

    @staticmethod
    def _graph_task_specs(graph: dict[str, Any]) -> list[dict[str, Any]]:
        if "tasks" in graph and "nodes" in graph:
            raise ValueError("CUDA persistent_dag_graph_f32 graph cannot use both tasks and nodes")
        has_task_list = "tasks" in graph or "nodes" in graph
        has_submit_list = "submits" in graph or "submissions" in graph
        has_submit_group_list = "submit_groups" in graph or "submission_groups" in graph
        if sum((has_task_list, has_submit_list, has_submit_group_list)) > 1:
            raise ValueError(
                "CUDA persistent_dag_graph_f32 graph cannot mix task/node, submit, and submit-group descriptors"
            )
        if "submits" in graph and "submissions" in graph:
            raise ValueError("CUDA persistent_dag_graph_f32 graph cannot use both submits and submissions")
        if "submit_groups" in graph and "submission_groups" in graph:
            raise ValueError("CUDA persistent_dag_graph_f32 graph cannot use both submit_groups and submission_groups")
        if has_submit_group_list:
            return _CudaPersistentDagSceneBuffers._graph_submit_group_task_specs(
                graph.get("submit_groups", graph.get("submission_groups", []))
            )
        tasks = graph.get("tasks", graph.get("nodes", graph.get("submits", graph.get("submissions", []))))
        if isinstance(tasks, dict):
            task_specs = []
            for task_name, task_spec in tasks.items():
                if not isinstance(task_spec, dict):
                    raise ValueError("CUDA persistent_dag_graph_f32 graph task dictionary values must be dictionaries")
                task_specs.append(
                    _CudaPersistentDagSceneBuffers._normalize_graph_task_shape({"name": task_name, **task_spec})
                )
            return task_specs
        return [_CudaPersistentDagSceneBuffers._normalize_graph_task_shape(task_spec) for task_spec in tasks]

    @staticmethod
    def _graph_task_defaults(graph: dict[str, Any]) -> dict[str, Any]:
        defaults = graph.get("task_defaults", graph.get("task_template", graph.get("default_task", {})))
        if defaults is None:
            return {}
        if not isinstance(defaults, dict):
            raise ValueError("CUDA persistent_dag_graph_f32 graph task defaults must be a dictionary")
        return _CudaPersistentDagSceneBuffers._normalize_graph_task_shape(defaults)

    @staticmethod
    def _merge_graph_task_defaults(defaults: dict[str, Any], task_spec: dict[str, Any]) -> dict[str, Any]:
        if not defaults:
            return task_spec
        merged = dict(defaults)
        merged.update(task_spec)
        return _CudaPersistentDagSceneBuffers._normalize_graph_task_shape(merged)

    @staticmethod
    def _graph_task_overrides(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
        overrides = graph.get("task_overrides", graph.get("task_metadata", {}))
        if overrides is None:
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(overrides, dict):
            override_items = overrides.items()
        elif isinstance(overrides, list):
            override_items = []
            for override in overrides:
                if not isinstance(override, dict):
                    raise ValueError("CUDA persistent_dag_graph_f32 graph task override values must be dictionaries")
                override_key = _CudaPersistentDagSceneBuffers._graph_task_override_key(override)
                if override_key is None:
                    raise ValueError(
                        "CUDA persistent_dag_graph_f32 graph task override lists require name, id, or task_id"
                    )
                override_items.append((override_key, override))
        else:
            raise ValueError("CUDA persistent_dag_graph_f32 graph task overrides must be a dictionary or list")
        for task_name, override in override_items:
            if not isinstance(override, dict):
                raise ValueError("CUDA persistent_dag_graph_f32 graph task override values must be dictionaries")
            normalized[str(task_name)] = _CudaPersistentDagSceneBuffers._normalize_graph_task_shape(override)
        return normalized

    @staticmethod
    def _graph_task_override_key(override: dict[str, Any]) -> str | None:
        for key in ("name", "id", "task_id"):
            value = override.get(key)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _merge_graph_task_overrides(
        overrides: dict[str, dict[str, Any]],
        task_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not overrides:
            return task_specs
        unused = set(overrides)
        merged_specs: list[dict[str, Any]] = []
        for task_spec in task_specs:
            keys = [
                str(value)
                for value in (task_spec.get("name"), task_spec.get("id"), task_spec.get("task_id"))
                if value is not None
            ]
            override_key = next((key for key in keys if key in overrides), None)
            if override_key is None:
                merged_specs.append(task_spec)
                continue
            unused.discard(override_key)
            merged = dict(task_spec)
            merged.update(overrides[override_key])
            merged_specs.append(_CudaPersistentDagSceneBuffers._normalize_graph_task_shape(merged))
        if unused:
            unknown = ", ".join(sorted(unused))
            raise ValueError(f"CUDA persistent_dag_graph_f32 graph task overrides reference unknown tasks: {unknown}")
        return merged_specs

    @staticmethod
    def _graph_submit_group_task_specs(submit_groups: Any) -> list[dict[str, Any]]:
        if isinstance(submit_groups, dict):
            group_entries = [{"name": group_name, **group_spec} for group_name, group_spec in submit_groups.items()]
        else:
            group_entries = list(submit_groups)

        task_specs: list[dict[str, Any]] = []
        for group_index, group_spec in enumerate(group_entries):
            if not isinstance(group_spec, dict):
                raise ValueError("CUDA persistent_dag_graph_f32 submit-group entries must be dictionaries")
            task_specs.extend(_CudaPersistentDagSceneBuffers._expand_graph_submit_group(group_spec, group_index))
        return task_specs

    @staticmethod
    def _expand_graph_submit_group(group_spec: dict[str, Any], group_index: int) -> list[dict[str, Any]]:
        args_list = group_spec.get("args_list")
        task_args_list = group_spec.get("task_args_list")
        if args_list is not None and task_args_list is not None:
            raise ValueError("CUDA persistent_dag_graph_f32 submit groups cannot use both args_list and task_args_list")
        if args_list is None:
            args_list = task_args_list
        if args_list is None:
            raise ValueError("CUDA persistent_dag_graph_f32 submit groups require args_list")

        group_defaults = {
            key: value for key, value in group_spec.items() if key not in {"args_list", "task_args_list", "name", "id"}
        }
        group_name = group_spec.get("name", group_spec.get("id"))
        task_specs: list[dict[str, Any]] = []
        for member_index, member_args in enumerate(args_list):
            task_spec = _CudaPersistentDagSceneBuffers._graph_submit_group_member_spec(group_defaults, member_args)
            if group_name is not None and "name" not in task_spec and "id" not in task_spec:
                task_spec["name"] = f"{group_name}[{member_index}]"
            elif group_name is None and "name" not in task_spec and "id" not in task_spec:
                task_spec["name"] = f"group{group_index}[{member_index}]"
            task_specs.append(_CudaPersistentDagSceneBuffers._normalize_graph_task_shape(task_spec))
        return task_specs

    @staticmethod
    def _graph_submit_group_member_spec(group_defaults: dict[str, Any], member_args: Any) -> dict[str, Any]:
        if isinstance(member_args, dict) and _CudaPersistentDagSceneBuffers._looks_like_graph_task(member_args):
            return {**group_defaults, **member_args}
        return {**group_defaults, "args": member_args}

    @staticmethod
    def _looks_like_graph_task(task_spec: dict[str, Any]) -> bool:
        task_fields = {
            "args",
            "attrs",
            "callable",
            "callable_id",
            "cid",
            "data",
            "dependencies",
            "depends_on",
            "dependents",
            "func_id",
            "id",
            "initial_fanin",
            "name",
            "op",
            "task_id",
            "task_args",
        }
        return bool(
            task_fields.intersection(task_spec)
            or any(field in task_spec for field in _CudaPersistentDagSceneBuffers._GRAPH_NODE_IO_FIELDS)
        )

    @staticmethod
    def _graph_edges(graph: dict[str, Any]) -> Any:
        return graph.get("edges", graph.get("links", []))

    @staticmethod
    def _normalize_graph_task_shape(task_spec: dict[str, Any]) -> dict[str, Any]:
        task_spec = _CudaPersistentDagSceneBuffers._expand_graph_task_data(task_spec)
        return _CudaPersistentDagSceneBuffers._normalize_graph_task_identity(task_spec)

    @staticmethod
    def _expand_graph_task_data(task_spec: dict[str, Any]) -> dict[str, Any]:
        data = task_spec.get("data")
        if data is None:
            return task_spec
        if not isinstance(data, dict):
            raise ValueError("CUDA persistent_dag_graph_f32 graph task data must be a dictionary")
        normalized = dict(data)
        normalized.update(task_spec)
        normalized.pop("data", None)
        return normalized

    @staticmethod
    def _normalize_graph_task_identity(task_spec: dict[str, Any]) -> dict[str, Any]:
        name = task_spec.get("name")
        node_id = task_spec.get("id")
        task_id = task_spec.get("task_id")
        identities = [identity for identity in (name, node_id, task_id) if identity is not None]
        if len({str(identity) for identity in identities}) > 1:
            raise ValueError("CUDA persistent_dag_graph_f32 graph task cannot use conflicting name/id/task_id values")
        if name is not None or (node_id is None and task_id is None):
            return task_spec
        return {"name": node_id if node_id is not None else task_id, **task_spec}

    @staticmethod
    def _resolve_graph_task_callable(graph: dict[str, Any], task_spec: dict[str, Any]) -> dict[str, Any]:
        callable_name = task_spec.get("callable")
        op_name = task_spec.get("op")
        if callable_name is not None and op_name is not None and str(callable_name) != str(op_name):
            raise ValueError("CUDA persistent_dag_graph_f32 graph task cannot use conflicting callable and op values")
        if callable_name is None:
            callable_name = op_name
        if callable_name is None:
            return task_spec

        graph_callables = _CudaPersistentDagSceneBuffers._graph_callables_by_name(graph)
        key = str(callable_name)
        if key not in graph_callables:
            raise ValueError(f"CUDA persistent_dag_graph_f32 unknown graph callable: {key}")
        callable_spec = graph_callables[key]
        if not isinstance(callable_spec, dict):
            raise ValueError(f"CUDA persistent_dag_graph_f32 graph callable {key} must be a dictionary")

        resolved = dict(callable_spec)
        resolved.update(task_spec)
        resolved.pop("callable", None)
        resolved.pop("op", None)
        return resolved

    @staticmethod
    def _normalize_graph_task_func_id(task_spec: dict[str, Any]) -> dict[str, Any]:
        func_id = task_spec.get("func_id")
        aliases = [task_spec.get(alias) for alias in ("callable_id", "cid") if alias in task_spec]
        alias_values = {str(alias) for alias in aliases}
        if len(alias_values) > 1 or (func_id is not None and alias_values and str(func_id) not in alias_values):
            raise ValueError("CUDA persistent_dag_graph_f32 graph task has conflicting func_id/callable_id/cid values")
        if func_id is not None or not aliases:
            return task_spec
        normalized = dict(task_spec)
        normalized["func_id"] = aliases[0]
        normalized.pop("callable_id", None)
        normalized.pop("cid", None)
        return normalized

    @staticmethod
    def _graph_callables_by_name(graph: dict[str, Any]) -> dict[str, Any]:
        graph_callables = graph.get("callables", {})
        if isinstance(graph_callables, dict):
            return {
                str(name): {"func_id": callable_spec} if isinstance(callable_spec, int) else callable_spec
                for name, callable_spec in graph_callables.items()
            }
        if not isinstance(graph_callables, list):
            raise ValueError("CUDA persistent_dag_graph_f32 graph callables must be a dictionary or list")

        resolved: dict[str, Any] = {}
        for index, entry in enumerate(graph_callables):
            if isinstance(entry, int):
                resolved[str(index)] = {"func_id": entry}
                continue
            if not isinstance(entry, dict):
                raise ValueError(
                    "CUDA persistent_dag_graph_f32 graph callable list entries must be dictionaries or integers"
                )
            callable_spec = dict(entry)
            name = callable_spec.pop("name", None)
            if name is not None:
                resolved[str(name)] = callable_spec
            resolved[str(index)] = callable_spec
        return resolved

    @staticmethod
    def _bind_graph_task_output_storage(
        task_spec: dict[str, Any],
        ptrs: dict[str, int],
        add_temporary,
        output_nbytes: int,
    ):
        for name in task_spec.get("_inout_names", []):
            if str(name) not in ptrs:
                raise ValueError(
                    f"CUDA persistent_dag_graph_f32 inout task_arg references unknown tensor or temporary: {name}"
                )
        out_name = task_spec.get("out")
        if out_name is None:
            return

        storage_name = task_spec.get("out_storage", out_name)
        requires_existing = task_spec.get("_out_requires_existing") or "out_storage" in task_spec
        if requires_existing and str(storage_name) not in ptrs:
            role = "out_storage" if "out_storage" in task_spec else "output_existing task_arg"
            raise ValueError(
                f"CUDA persistent_dag_graph_f32 {role} references unknown tensor or temporary: {storage_name}"
            )
        if not requires_existing:
            add_temporary(storage_name, output_nbytes)
        if str(out_name) != str(storage_name):
            ptrs[str(out_name)] = ptrs[str(storage_name)]

    @staticmethod
    def _graph_dependents_from_task_specs(
        task_specs: list[dict[str, Any]],
        graph_edges: Any = None,
    ) -> list[list[int]]:
        task_name_to_id = _CudaPersistentDagSceneBuffers._graph_task_name_to_id(task_specs)
        producers: dict[str, list[int]] = {}
        for task_id, task_spec in enumerate(task_specs):
            out_name = task_spec.get("out")
            if out_name is not None:
                producers.setdefault(str(out_name), []).append(task_id)

        dependents = [[] for _ in task_specs]

        for task_id, task_spec in enumerate(task_specs):
            if "dependents" in task_spec and _CudaPersistentDagSceneBuffers._graph_task_dependencies(
                task_spec,
                task_name_to_id,
            ):
                raise ValueError(
                    "CUDA persistent_dag_graph_f32 graph tasks cannot mix dependents with depends_on/dependencies"
                )
            if "dependents" in task_spec:
                task_dependents = task_spec["dependents"]
                if isinstance(task_dependents, (int, str)):
                    task_dependents = [task_dependents]
                dependents[task_id].extend(
                    _CudaPersistentDagSceneBuffers._graph_dependent_task_id(dependent, task_name_to_id)
                    for dependent in task_dependents
                )

        _CudaPersistentDagSceneBuffers._add_graph_edges_to_dependents(
            dependents,
            graph_edges or [],
            task_name_to_id,
            len(task_specs),
        )

        for task_id, task_spec in enumerate(task_specs):
            dependency_ids = _CudaPersistentDagSceneBuffers._graph_task_dependencies(task_spec, task_name_to_id)
            if dependency_ids is None:
                continue
            for dependency_id in dependency_ids:
                if dependency_id < 0 or dependency_id >= len(task_specs):
                    raise ValueError(
                        "CUDA persistent_dag_graph_f32 dependency task id "
                        f"{dependency_id} for task {task_id} is outside the graph"
                    )
                if task_id not in dependents[dependency_id]:
                    dependents[dependency_id].append(task_id)

        for task_id, task_spec in enumerate(task_specs):
            if "dependents" in task_spec or _CudaPersistentDagSceneBuffers._graph_task_dependencies(
                task_spec,
                task_name_to_id,
            ):
                continue
            producer_ids = {
                producer_id
                for name in _CudaPersistentDagSceneBuffers._graph_task_read_names(task_spec)
                if (producer_id := _CudaPersistentDagSceneBuffers._graph_read_producer(producers, name, task_id))
                is not None
            }
            for producer_id in sorted(producer_ids):
                if task_id not in dependents[producer_id]:
                    dependents[producer_id].append(task_id)
        return dependents

    @staticmethod
    def _add_graph_edges_to_dependents(
        dependents: list[list[int]],
        graph_edges: Any,
        task_name_to_id: dict[str, int],
        task_count: int,
    ) -> None:
        for edge in _CudaPersistentDagSceneBuffers._graph_edge_entries(graph_edges):
            source, target = _CudaPersistentDagSceneBuffers._graph_edge_endpoints(edge)
            source_id = _CudaPersistentDagSceneBuffers._graph_dependency_task_id(source, task_name_to_id)
            target_id = _CudaPersistentDagSceneBuffers._graph_dependent_task_id(target, task_name_to_id)
            if source_id < 0 or source_id >= task_count:
                raise ValueError(f"CUDA persistent_dag_graph_f32 edge source task id {source_id} is outside the graph")
            if target_id < 0 or target_id >= task_count:
                raise ValueError(f"CUDA persistent_dag_graph_f32 edge target task id {target_id} is outside the graph")
            if target_id not in dependents[source_id]:
                dependents[source_id].append(target_id)

    @staticmethod
    def _graph_edge_entries(graph_edges: Any) -> list[Any]:
        if graph_edges is None:
            return []
        if isinstance(graph_edges, dict):
            entries = []
            for source, targets in graph_edges.items():
                if isinstance(targets, (list, tuple)):
                    entries.extend((source, target) for target in targets)
                else:
                    entries.append((source, targets))
            return entries
        return list(graph_edges)

    @staticmethod
    def _graph_edge_endpoints(edge: Any) -> tuple[Any, Any]:
        if isinstance(edge, dict):
            source = edge.get("pred", edge.get("from", edge.get("source", edge.get("src"))))
            target = edge.get("succ", edge.get("to", edge.get("target", edge.get("dst"))))
            if source is None or target is None:
                raise ValueError("CUDA persistent_dag_graph_f32 graph edges must provide from/to endpoints")
            return source, target
        if isinstance(edge, str):
            endpoints = edge.split("->")
            if len(endpoints) != 2 or not endpoints[0].strip() or not endpoints[1].strip():
                raise ValueError("CUDA persistent_dag_graph_f32 string graph edges must use '<source> -> <target>'")
            return endpoints[0].strip(), endpoints[1].strip()
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            return edge[0], edge[1]
        raise ValueError("CUDA persistent_dag_graph_f32 graph edges must be endpoint pairs, dictionaries, or strings")

    @staticmethod
    def _graph_task_name_to_id(task_specs: list[dict[str, Any]]) -> dict[str, int]:
        task_name_to_id: dict[str, int] = {}
        for task_id, task_spec in enumerate(task_specs):
            name = task_spec.get("name")
            if name is None:
                continue
            key = str(name)
            if key in task_name_to_id:
                raise ValueError(f"CUDA persistent_dag_graph_f32 duplicate graph task name: {key}")
            task_name_to_id[key] = task_id
        return task_name_to_id

    @staticmethod
    def _graph_task_dependencies(
        task_spec: dict[str, Any],
        task_name_to_id: dict[str, int] | None = None,
    ) -> list[int] | None:
        dependencies = task_spec.get("depends_on")
        dependency_alias = task_spec.get("dependencies")
        if dependencies is not None and dependency_alias is not None:
            raise ValueError("CUDA persistent_dag_graph_f32 graph tasks cannot use both depends_on and dependencies")
        if dependencies is None:
            dependencies = dependency_alias
        if dependencies is None:
            return None
        if isinstance(dependencies, (int, str)):
            dependencies = [dependencies]
        return [
            _CudaPersistentDagSceneBuffers._graph_dependency_task_id(dependency, task_name_to_id or {})
            for dependency in dependencies
        ]

    @staticmethod
    def _graph_dependency_task_id(dependency: Any, task_name_to_id: dict[str, int]) -> int:
        if isinstance(dependency, int):
            return dependency
        key = str(dependency)
        if key in task_name_to_id:
            return task_name_to_id[key]
        try:
            return int(key)
        except ValueError as exc:
            raise ValueError(f"CUDA persistent_dag_graph_f32 unknown dependency task name: {key}") from exc

    @staticmethod
    def _graph_dependent_task_id(dependent: Any, task_name_to_id: dict[str, int]) -> int:
        if isinstance(dependent, int):
            return dependent
        key = str(dependent)
        if key in task_name_to_id:
            return task_name_to_id[key]
        try:
            return int(key)
        except ValueError as exc:
            raise ValueError(f"CUDA persistent_dag_graph_f32 unknown dependent task name: {key}") from exc

    @staticmethod
    def _graph_read_producer(producers: dict[str, list[int]], name: str, task_id: int) -> int | None:
        candidates = producers.get(name, [])
        previous = [producer_id for producer_id in candidates if producer_id < task_id]
        if previous:
            return previous[-1]
        future = [producer_id for producer_id in candidates if producer_id > task_id]
        if future:
            return future[0]
        return None

    @staticmethod
    def _graph_task_read_names(task_spec: dict[str, Any]) -> list[str]:
        names = [task_spec.get(field) for field in ("a", "b", "c", "d")]
        names.extend(task_spec.get("tensor_args", []))
        return [str(name) for name in names if name is not None]

    @staticmethod
    def _normalize_graph_task_spec(task_spec: dict[str, Any]) -> dict[str, Any]:
        task_spec = _CudaPersistentDagSceneBuffers._expand_graph_task_attrs(task_spec)
        task_args = _CudaPersistentDagSceneBuffers._graph_task_args_for_normalization(task_spec)
        if task_args is None:
            return task_spec

        normalized = dict(task_spec)
        normalized.pop("task_args", None)
        for field in _CudaPersistentDagSceneBuffers._GRAPH_NODE_IO_FIELDS:
            normalized.pop(field, None)
        inputs: list[str] = []
        outputs: list[str] = []
        scalar_args: list[Any] = []
        inout_names: list[str] = []
        output_requires_existing = False
        for index, task_arg in enumerate(task_args):
            task_arg = _CudaPersistentDagSceneBuffers._normalize_graph_task_arg_entry(task_arg, index)
            compact_role, compact_name = _CudaPersistentDagSceneBuffers._compact_graph_tensor_task_arg(
                task_arg,
                index,
            )
            if compact_role is not None:
                task_arg = {**task_arg, "role": compact_role, "tensor": compact_name}
            is_scalar, scalar_value = _CudaPersistentDagSceneBuffers._normalize_graph_scalar_task_arg(
                task_arg,
                index,
            )
            if is_scalar:
                scalar_args.append(scalar_value)
                continue
            name = task_arg.get("tensor", task_arg.get("name"))
            if name is None:
                raise ValueError(
                    "CUDA persistent_dag_graph_f32 task_args entries must name a tensor, temporary, or scalar"
                )
            output_requires_existing |= _CudaPersistentDagSceneBuffers._append_graph_tensor_task_arg(
                task_arg,
                index,
                str(name),
                inputs,
                outputs,
                inout_names,
            )

        for field, name in zip(("a", "b", "c", "d"), inputs):
            normalized.setdefault(field, name)
        extra_inputs = inputs[4:]
        if extra_inputs:
            normalized["tensor_args"] = list(normalized.get("tensor_args", [])) + extra_inputs
        if scalar_args:
            normalized["scalar_args"] = list(normalized.get("scalar_args", [])) + scalar_args
        if len(outputs) > 1:
            raise ValueError("CUDA persistent_dag_graph_f32 task_args supports one output per graph task")
        if outputs:
            normalized.setdefault("out", outputs[0])
        if output_requires_existing:
            normalized["_out_requires_existing"] = True
        if inout_names:
            normalized["_inout_names"] = inout_names
        return normalized

    @staticmethod
    def _normalize_graph_task_arg_entry(task_arg: Any, index: int) -> dict[str, Any]:
        if isinstance(task_arg, dict):
            return task_arg
        if isinstance(task_arg, (list, tuple)) and len(task_arg) == 2:
            role, name = task_arg
            if str(role).lower() == "scalar":
                return {"scalar": name}
            return {"tensor": name, "role": role}
        raise ValueError(
            f"CUDA persistent_dag_graph_f32 task_args entry {index} must be a dictionary or role/name pair"
        )

    @staticmethod
    def _expand_graph_task_attrs(task_spec: dict[str, Any]) -> dict[str, Any]:
        attrs = task_spec.get("attrs")
        if attrs is None:
            return task_spec
        if not isinstance(attrs, dict):
            raise ValueError("CUDA persistent_dag_graph_f32 graph task attrs must be a dictionary")
        normalized = dict(attrs)
        normalized.update(task_spec)
        normalized.pop("attrs", None)
        return normalized

    @staticmethod
    def _graph_task_args_for_normalization(task_spec: dict[str, Any]) -> Any:
        task_args = task_spec.get("task_args")
        args_alias = task_spec.get("args")
        if task_args is not None and args_alias is not None:
            raise ValueError("CUDA persistent_dag_graph_f32 graph tasks cannot use both task_args and args")
        has_node_io_fields = any(field in task_spec for field in _CudaPersistentDagSceneBuffers._GRAPH_NODE_IO_FIELDS)
        if (task_args is not None or args_alias is not None) and has_node_io_fields:
            raise ValueError("CUDA persistent_dag_graph_f32 graph tasks cannot mix args with node IO fields")
        if task_args is not None:
            return _CudaPersistentDagSceneBuffers._normalize_graph_task_args_shape(task_args)
        if args_alias is not None:
            return _CudaPersistentDagSceneBuffers._normalize_graph_task_args_shape(args_alias)
        return _CudaPersistentDagSceneBuffers._graph_node_io_task_args(task_spec, has_node_io_fields)

    @staticmethod
    def _normalize_graph_task_args_shape(task_args: Any) -> Any:
        if not isinstance(task_args, dict):
            return task_args
        if any(field in task_args for field in _CudaPersistentDagSceneBuffers._GRAPH_NODE_IO_FIELDS):
            return _CudaPersistentDagSceneBuffers._graph_task_arg_map_entries(task_args)
        if any(field in task_args for field in ("tensor", "name", "role", "tag")):
            return [task_args]
        raise ValueError(
            "CUDA persistent_dag_graph_f32 task_args dictionaries must use role keys or describe one task arg"
        )

    @staticmethod
    def _graph_task_arg_map_entries(task_args: dict[str, Any]) -> list[dict[str, Any]]:
        roles = {
            "input": "input",
            "inputs": "input",
            "output": "output",
            "outputs": "output",
            "output_existing": "output_existing",
            "inout": "inout",
            "inouts": "inout",
            "scalar": "scalar",
            "scalars": "scalar",
        }
        entries: list[dict[str, Any]] = []
        for key, value in task_args.items():
            role = roles.get(str(key))
            if role is None:
                continue
            if isinstance(value, dict):
                values = _CudaPersistentDagSceneBuffers._graph_node_port_values(value)
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value]
            for item in values:
                entries.append({"scalar": item} if role == "scalar" else {role: item})
        return entries

    @staticmethod
    def _graph_node_io_task_args(task_spec: dict[str, Any], has_node_io_fields: bool) -> list[dict[str, Any]] | None:
        if not has_node_io_fields:
            return None

        task_args: list[dict[str, Any]] = []
        for name in _CudaPersistentDagSceneBuffers._graph_node_io_values(task_spec, "input", "inputs"):
            task_args.append({"input": name})
        for name in _CudaPersistentDagSceneBuffers._graph_node_io_values(task_spec, "output", "outputs"):
            task_args.append({"output": name})
        for name in _CudaPersistentDagSceneBuffers._graph_node_io_values(task_spec, "output_existing"):
            task_args.append({"output_existing": name})
        for name in _CudaPersistentDagSceneBuffers._graph_node_io_values(task_spec, "inout", "inouts"):
            task_args.append({"inout": name})
        for name in _CudaPersistentDagSceneBuffers._graph_node_io_values(task_spec, "scalar", "scalars"):
            task_args.append({"scalar": name})
        return task_args

    @staticmethod
    def _graph_node_io_values(task_spec: dict[str, Any], *keys: str) -> list[Any]:
        values: list[Any] = []
        for key in keys:
            if key not in task_spec:
                continue
            value = task_spec[key]
            if isinstance(value, dict):
                values.extend(_CudaPersistentDagSceneBuffers._graph_node_port_values(value))
            elif isinstance(value, (list, tuple)):
                values.extend(value)
            else:
                values.append(value)
        return values

    @staticmethod
    def _graph_node_port_values(value: dict[Any, Any]) -> list[Any]:
        priority = {
            "a": 0,
            "lhs": 0,
            "left": 0,
            "x": 0,
            "input0": 0,
            "b": 1,
            "rhs": 1,
            "right": 1,
            "y": 1,
            "input1": 1,
            "c": 2,
            "input2": 2,
            "d": 3,
            "input3": 3,
            "out": 4,
            "output": 4,
            "value": 4,
        }
        return [
            item[1]
            for item in sorted(
                value.items(),
                key=lambda item: (priority.get(str(item[0]).lower(), 100), str(item[0])),
            )
        ]

    @staticmethod
    def _compact_graph_tensor_task_arg(task_arg: dict[str, Any], index: int) -> tuple[str | None, Any]:
        compact_roles = ("input", "in", "output", "out", "output_existing", "inout")
        matches = [(role, task_arg[role]) for role in compact_roles if role in task_arg]
        if not matches:
            return None, None
        if len(matches) > 1:
            raise ValueError(f"CUDA persistent_dag_graph_f32 task_args entry {index} has multiple compact tensor roles")
        if any(key in task_arg for key in ("tensor", "name", "role", "tag")):
            raise ValueError(
                f"CUDA persistent_dag_graph_f32 task_args entry {index} mixes compact and expanded tensor roles"
            )
        return matches[0]

    @staticmethod
    def _normalize_graph_scalar_task_arg(task_arg: dict[str, Any], index: int) -> tuple[bool, Any]:
        if "scalar" not in task_arg:
            return False, None
        tag = _CudaPersistentDagSceneBuffers._graph_task_arg_role(task_arg, index)
        if tag not in {"input", "in"}:
            raise ValueError(
                f"CUDA persistent_dag_graph_f32 scalar task_args entry {index} has unsupported role: {tag}"
            )
        return True, task_arg["scalar"]

    @staticmethod
    def _graph_task_arg_role(task_arg: dict[str, Any], index: int) -> str:
        role = task_arg.get("role")
        tag = task_arg.get("tag")
        if role is not None and tag is not None and str(role).lower() != str(tag).lower():
            raise ValueError(
                f"CUDA persistent_dag_graph_f32 task_args entry {index} has conflicting role and tag values"
            )
        return str(role if role is not None else tag if tag is not None else "input").lower()

    @staticmethod
    def _append_graph_tensor_task_arg(
        task_arg: dict[str, Any],
        index: int,
        name: str,
        inputs: list[str],
        outputs: list[str],
        inout_names: list[str],
    ) -> bool:
        tag = _CudaPersistentDagSceneBuffers._graph_task_arg_role(task_arg, index)
        if tag in {"input", "in"}:
            inputs.append(name)
            return False
        if tag in {"output", "out"}:
            outputs.append(name)
            return False
        if tag == "output_existing":
            outputs.append(name)
            return True
        if tag == "inout":
            inputs.append(name)
            outputs.append(name)
            inout_names.append(name)
            return True
        raise ValueError(f"CUDA persistent_dag_graph_f32 task_args entry {index} has unsupported role: {tag}")

    def _temporary_nbytes(self, size_source, default_nbytes: int) -> int:
        if isinstance(size_source, int):
            return size_source
        if isinstance(size_source, str):
            if size_source not in self.tensor_buffers.sizes:
                raise ValueError(f"CUDA persistent DAG temporary size references unknown tensor: {size_source}")
            return self.tensor_buffers.sizes[size_source]
        return default_nbytes

    def _graph_scalar_value(self, value) -> float:
        if isinstance(value, str):
            try:
                value = getattr(self.test_args, value)
            except AttributeError as exc:
                raise ValueError(
                    f"CUDA persistent_dag_graph_f32 scalar field references unknown scalar argument: {value}"
                ) from exc
        if hasattr(value, "value"):
            value = value.value
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CUDA persistent_dag_graph_f32 scalar fields must be numeric literals or scalar argument names"
            ) from exc

    def _make_graph_task(
        self,
        ctypes_module,
        task_type,
        task_spec: dict,
        ptrs: dict[str, int],
        n: int,
        dependent_begin: int,
        dependent_count: int,
        initial_fanin: int,
    ):
        tensor_args = list(task_spec.get("tensor_args", []))
        scalar_args = [self._graph_scalar_value(value) for value in task_spec.get("scalar_args", [])]
        if len(tensor_args) > 4:
            raise ValueError("CUDA persistent_dag_graph_f32 supports at most four tensor_args per task")
        if len(scalar_args) > 4:
            raise ValueError("CUDA persistent_dag_graph_f32 supports at most four scalar_args per task")
        if int(task_spec["func_id"]) == 10 and (
            not _CudaPersistentDagSceneBuffers._is_tensor_core_tile(
                {
                    "rows": int(task_spec.get("rows", 0)),
                    "cols": int(task_spec.get("cols", 0)),
                    "inner": int(task_spec.get("inner", 0)),
                }
            )
        ):
            raise ValueError(
                "CUDA persistent_dag_graph_f32 tensor core task requires rows/cols multiples of 16 "
                "and inner divisible by 8"
            )

        tensor_args_t = ctypes_module.c_void_p * 4
        scalar_args_t = ctypes_module.c_float * 4
        tensor_arg_ptrs = [self._graph_ptr(ptrs, name) for name in tensor_args]
        task = task_type(
            func_id=int(task_spec["func_id"]),
            a=self._graph_ptr(ptrs, task_spec.get("a")),
            b=self._graph_ptr(ptrs, task_spec.get("b")),
            c=self._graph_ptr(ptrs, task_spec.get("c")),
            d=self._graph_ptr(ptrs, task_spec.get("d")),
            out=self._graph_ptr(ptrs, task_spec.get("out")),
            n=int(task_spec.get("n", n)),
            dependent_begin=dependent_begin,
            dependent_count=dependent_count,
            initial_fanin=initial_fanin,
            scalar0=self._graph_scalar_value(task_spec.get("scalar0", 0.0)),
            scalar1=self._graph_scalar_value(task_spec.get("scalar1", 0.0)),
            tensor_args=tensor_args_t(*(tensor_arg_ptrs + [0] * (4 - len(tensor_arg_ptrs)))),
            scalar_args=scalar_args_t(*(scalar_args + [0.0] * (4 - len(scalar_args)))),
            tensor_arg_count=len(tensor_arg_ptrs),
            scalar_arg_count=len(scalar_args),
        )
        for field in (
            "rows",
            "cols",
            "inner",
            "lda",
            "ldb",
            "ldc",
            "a_batch_stride",
            "b_batch_stride",
            "out_batch_stride",
        ):
            if field in task_spec:
                setattr(task, field, int(task_spec[field]))
        return task

    @staticmethod
    def _graph_ptr(ptrs: dict[str, int], name) -> int:
        if name is None:
            return 0
        key = str(name)
        if key not in ptrs:
            raise ValueError(f"CUDA persistent_dag_graph_f32 references unknown tensor or temporary: {key}")
        return ptrs[key]

    def reset_for_run(self) -> None:
        import ctypes  # noqa: PLC0415

        self.tensor_buffers.copy_all_to_device()
        self.worker.copy_to(self.dev_fanin, ctypes.addressof(self.host_fanin), ctypes.sizeof(self.host_fanin))
        self.worker.copy_to(
            self.dev_ready_flags,
            ctypes.addressof(self.host_flags),
            ctypes.sizeof(self.host_flags),
        )
        self.worker.copy_to(
            self.dev_completion_flags,
            ctypes.addressof(self.host_flags),
            ctypes.sizeof(self.host_flags),
        )
        self.worker.copy_to(
            self.dev_counters,
            ctypes.addressof(self.host_counters),
            ctypes.sizeof(self.host_counters),
        )
        self.worker.copy_to(
            self.dev_scheduler_processed_by_block,
            ctypes.addressof(self.host_scheduler_processed_by_block),
            ctypes.sizeof(self.host_scheduler_processed_by_block),
        )

    def copy_outputs_from_device(self, output_names: list[str]) -> None:
        self.tensor_buffers.copy_outputs_from_device(output_names)

    def read_counters(self) -> dict[str, int | list[int]]:
        import ctypes  # noqa: PLC0415

        counters_t = ctypes.c_uint32 * 11
        scheduler_processed_by_block_t = ctypes.c_uint32 * 1
        counters = counters_t()
        scheduler_processed_by_block = scheduler_processed_by_block_t()
        self.worker.copy_from(ctypes.addressof(counters), self.dev_counters, ctypes.sizeof(counters))
        self.worker.copy_from(
            ctypes.addressof(scheduler_processed_by_block),
            self.dev_scheduler_processed_by_block,
            ctypes.sizeof(scheduler_processed_by_block),
        )
        return {
            "queue_head": int(counters[0]),
            "queue_tail": int(counters[1]),
            "completion_head": int(counters[2]),
            "completion_tail": int(counters[3]),
            "completed_count": int(counters[4]),
            "error_count": int(counters[5]),
            "error_code": int(counters[6]),
            "error_task_id": int(counters[7]),
            "scheduler_init_count": int(counters[8]),
            "scheduler_loop_count": int(counters[9]),
            "scheduler_processed_count": int(counters[10]),
            "scheduler_processed_by_block": [int(value) for value in scheduler_processed_by_block],
        }

    def free(self) -> None:
        for ptr in self.ptrs:
            self.worker.free(ptr)
        self.ptrs.clear()
        self.tensor_buffers.free()


@contextmanager
def _temporary_env(env_updates):
    """Temporarily set environment variables."""
    if not env_updates:
        yield
        return
    old = {k: os.environ.get(k) for k in env_updates}
    for k, v in env_updates.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _log_round_timings(timings):
    """Print per-round + summary host/device wall (µs) for multi-round runs.

    Replaces the device-log scraping that ``tools/benchmark_rounds.sh`` used to
    do — Worker.run now returns the timing directly, so the per-round table
    can be emitted by the framework without grep+awk over device logs.

    Output goes to ``print`` (not ``logger.info``) because benchmark numbers
    are a user-facing artifact and the project's default log level suppresses
    INFO in standalone test_*.py runs.

    ``timings`` is a list of (host_wall_us, device_wall_us) tuples. The device
    column reports 0 when the runtime was built without PTO2_PROFILING (the
    default build has it on) or when an L3+ DAG is in use (per-task device
    cycles aren't aggregated).
    """
    if not timings:
        return
    n = len(timings)
    host_vals = sorted(t[0] for t in timings)
    dev_vals = sorted(t[1] for t in timings)
    host_avg = sum(host_vals) / n
    # Device avg averages over non-zero rounds only. When --rounds > 1 the
    # framework restricts enable_l2_swimlane (and thus orch_summary capture)
    # to round 0, so the other rounds report 0; including those zeros in the
    # average would silently halve / quarter the reported device wall and
    # mislead the benchmark consumer. dev_count carries the sample size into
    # the summary line for transparency.
    dev_nonzero = [v for v in dev_vals if v > 0.0]
    dev_count = len(dev_nonzero)
    dev_avg = (sum(dev_nonzero) / dev_count) if dev_count else 0.0
    show_device = dev_count > 0

    header = f"  {'Round':<6}  {'Host (us)':>12}"
    if show_device:
        header += f"  {'Device (us)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (h, d) in enumerate(timings):
        line = f"  {i:<6d}  {h:>12.1f}"
        if show_device:
            line += f"  {d:>12.1f}"
        print(line)
    summary = f"  Avg Host: {host_avg:.1f} us"
    if show_device:
        summary += f"  |  Avg Device: {dev_avg:.1f} us [{dev_count}/{n} rounds captured]"
    summary += f"  ({n} rounds)"
    print(summary)

    trim = 10
    if n > 2 * trim:
        tc = n - 2 * trim
        host_trim = sum(host_vals[trim:-trim]) / tc
        msg = f"  Trimmed Avg Host: {host_trim:.1f} us"
        if show_device and dev_count > 2 * trim:
            dev_nonzero_sorted = sorted(dev_nonzero)
            dev_trim_count = dev_count - 2 * trim
            dev_trim = sum(dev_nonzero_sorted[trim:-trim]) / dev_trim_count
            msg += f"  |  Trimmed Avg Device: {dev_trim:.1f} us"
        msg += f"  (dropped {trim} low + {trim} high, {tc} rounds used)"
        print(msg)


def _resolve_callable_paths(cls, cls_dir):
    """Resolve relative source paths in CALLABLE against cls_dir."""
    callable_spec = cls.CALLABLE
    if "callables" in callable_spec:
        # L3: resolve inside each ChipCallable entry
        resolved = []
        for entry in callable_spec["callables"]:
            if "orchestration" in entry:
                entry = dict(entry)
                _resolve_chip_entry_paths(entry, cls_dir)
            resolved.append(entry)
        callable_spec["callables"] = resolved
    else:
        # L2: resolve orchestration + incores directly
        _resolve_chip_entry_paths(callable_spec, cls_dir)


def _resolve_chip_entry_paths(entry, cls_dir):
    """Resolve relative source paths in a chip entry (orchestration + incores)."""
    if "cuda" in entry:
        cuda = entry["cuda"]
        if isinstance(cuda, dict):
            entry["cuda"] = dict(cuda)
            for key in ("source", "graph_path", "graph_file", "task_sources_path", "task_sources_file"):
                if key in entry["cuda"]:
                    entry["cuda"][key] = _resolve_path_value(entry["cuda"][key], cls_dir)
            if isinstance(entry["cuda"].get("graph"), (str, os.PathLike)):
                entry["cuda"]["graph"] = _resolve_path_value(entry["cuda"]["graph"], cls_dir)
            if "task_sources" in entry["cuda"]:
                resolved_sources = []
                for task_source in entry["cuda"]["task_sources"]:
                    task_source = dict(task_source)
                    source_path = task_source.get("source_path", task_source.get("source"))
                    if source_path is not None:
                        task_source["source_path"] = _resolve_path_value(source_path, cls_dir)
                    resolved_sources.append(task_source)
                entry["cuda"]["task_sources"] = resolved_sources
    if "orchestration" in entry:
        orch = entry["orchestration"]
        if isinstance(orch, dict) and "source" in orch and not os.path.isabs(orch["source"]):
            entry["orchestration"] = dict(orch)
            entry["orchestration"]["source"] = str(cls_dir / orch["source"])
    if "incores" in entry:
        resolved = []
        for k in entry["incores"]:
            k = dict(k)
            if "source" in k and not os.path.isabs(k["source"]):
                k["source"] = str(cls_dir / k["source"])
            resolved.append(k)
        entry["incores"] = resolved


def _resolve_path_value(path_value, cls_dir):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(cls_dir / path)


def _extract_name_map(callable_spec: dict) -> dict:
    """Extract name mapping from a CALLABLE spec.

    Each level exports only its own ``callable_id_to_name`` — the mapping
    from next-level-down IDs to human-readable names.  No cross-level
    nesting: the perf data declares its level, the mapping declares its
    level, and the tool matches them.

    * **L2** — ``callable_id`` = incore ``func_id``::

        {"level": 2, "orchestrator_name": "PagedAttn",
         "callable_id_to_name": {"0": "QK", "1": "SF"}}

    * **L3** — ``callable_id`` = index in ``callables`` list::

        {"level": 3, "orchestrator_name": "run_dag",
         "callable_id_to_name": {"0": "vec_kernel", "1": "verify"}}
    """
    if "callables" not in callable_spec:
        # L2: orchestration + incores
        callable_id_to_name: dict[str, str] = {}
        cuda = callable_spec.get("cuda")
        if isinstance(cuda, dict):
            task_name = cuda.get("task_name") or cuda.get("name")
            return {"level": 2, "orchestrator_name": task_name}
        orch = callable_spec.get("orchestration", {})
        orchestrator_name = orch.get("name") if isinstance(orch, dict) else None
        for k in callable_spec.get("incores", []):
            if "name" in k and "func_id" in k:
                callable_id_to_name[str(k["func_id"])] = k["name"]
        result: dict = {"level": 2, "orchestrator_name": orchestrator_name}
        if callable_id_to_name:
            result["callable_id_to_name"] = callable_id_to_name
        return result

    # L3: Python orch function + callables list
    orch = callable_spec.get("orchestration")
    orchestrator_name = getattr(orch, "__name__", None) if callable(orch) else None

    callable_id_to_name = {}
    for idx, entry in enumerate(callable_spec["callables"]):
        callable_id_to_name[str(idx)] = entry.get("name", f"callable_{idx}")

    result = {"level": 3, "orchestrator_name": orchestrator_name}
    if callable_id_to_name:
        result["callable_id_to_name"] = callable_id_to_name
    return result


def _dump_name_map(mapping: dict, output_path: Path) -> Path | None:
    """Write name mapping to JSON if it contains any names. Returns path or None."""
    import json as _json  # noqa: PLC0415

    if not mapping.get("callable_id_to_name") and not mapping.get("orchestrator_name"):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        _json.dump(mapping, f, indent=2)
    return output_path


def _parse_case_selector(value: str) -> tuple[str | None, str | None]:
    """Parse one ``--case`` value into ``(class_name, case_name)``.

    ``Foo`` -> ``(None, "Foo")`` (any class)
    ``ClassA::Foo`` -> ``("ClassA", "Foo")``
    ``ClassA::`` -> ``("ClassA", None)`` (all cases in ClassA)
    ``::Foo`` -> ``(None, "Foo")``
    """
    if "::" in value:
        cls_part, case_part = value.split("::", 1)
        return (cls_part or None, case_part or None)
    return (None, value)


def _match_selectors(cls_name: str, case_name: str, selectors: list[tuple]) -> bool:
    """True if ``(cls_name, case_name)`` matches any selector (empty list means no selector filter)."""
    if not selectors:
        return True
    for sel_cls, sel_case in selectors:
        if (sel_cls is None or sel_cls == cls_name) and (sel_case is None or sel_case == case_name):
            return True
    return False


def _select_cases(test_classes, platform: str, selectors: list[tuple], manual_mode: str):
    """Resolve (class, case) pairs to run. Validates selectors strictly.

    Filters: platform match -> selector match -> manual_mode (exclude/include/only).
    Raises ``ValueError`` on unknown selector class/case or empty selection.
    """
    class_index = {c.__name__: c for c in test_classes}
    for sel_cls, _ in selectors:
        if sel_cls is not None and sel_cls not in class_index:
            available = ", ".join(sorted(class_index)) or "(none)"
            raise ValueError(f"--case: unknown class '{sel_cls}'. Available: {available}")
    for sel_cls, sel_case in selectors:
        if sel_case is None:
            continue
        scoped = [class_index[sel_cls]] if sel_cls else test_classes
        if not any(case["name"] == sel_case for c in scoped for case in c.CASES):
            scope = sel_cls or "any class"
            raise ValueError(f"--case: case '{sel_case}' not found in {scope}")

    selected: list[tuple] = []
    for cls in test_classes:
        for case in cls.CASES:
            if platform not in case["platforms"]:
                continue
            if not _match_selectors(cls.__name__, case["name"], selectors):
                continue
            is_manual = bool(case.get("manual"))
            if manual_mode == "exclude" and is_manual:
                continue
            if manual_mode == "only" and not is_manual:
                continue
            selected.append((cls, case))

    if not selected:
        if selectors:
            sel_str = ", ".join(f"{c or '*'}::{n or '*'}" for c, n in selectors)
            hint = " (matches are manual; pass --manual include or only)" if manual_mode == "exclude" else ""
            raise ValueError(f"--case: no cases matched [{sel_str}] for platform={platform}{hint}")
        raise ValueError(f"No cases matched platform={platform} (manual={manual_mode})")
    return selected


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _outputs_dir() -> Path:
    """Root directory under which per-case output prefixes are created."""
    return _project_root() / "outputs"


def _build_output_prefix(case_label: str) -> Path:
    """Per-case directory for diagnostic artifacts.

    Each case gets its own ``outputs/<case_label>_<timestamp>/`` directory; the
    runtime writes ``l2_perf_records.json``, ``tensor_dump/``, and ``pmu.csv``
    under that root with fixed filenames. Two cases of the same name run in
    the same second is not a contemplated scenario (parallel xdist runs differ
    by class+method).

    The directory is created here: the dep_gen host replay (and any other
    writer) ``fopen``s ``<prefix>/<file>`` directly without an mkdir of its
    own, so the prefix must exist before the runtime call.
    """
    from datetime import datetime  # noqa: PLC0415

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = _sanitize_for_filename(case_label)
    prefix = _outputs_dir() / f"{safe_label}_{timestamp}"
    prefix.mkdir(parents=True, exist_ok=True)
    return prefix


def _run_swimlane_converter(
    input_path: Path | None = None,
    func_names_path: Path | None = None,
) -> None:
    """Invoke the bundled swimlane converter as a subprocess.

    When ``input_path`` is given, the converter derives its output filename from
    the input's timestamp (see ``swimlane_converter._resolve_output_path``).
    Without it, the converter auto-selects the latest ``l2_perf_records_*.json``.
    """
    import logging  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    cmd = [sys.executable, "-m", "simpler_setup.tools.swimlane_converter"]
    if input_path is not None:
        cmd.append(str(input_path))
    if func_names_path is not None:
        cmd += ["--func-names", str(func_names_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        logger.info("Swimlane JSON generation completed")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate swimlane JSON: {e}")
        if e.stdout:
            logger.debug(f"stdout: {e.stdout}")
        if e.stderr:
            logger.debug(f"stderr: {e.stderr}")


def _sanitize_for_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _convert_case_swimlane(
    case_label: str,
    output_prefix: Path,
    callable_spec: dict | None = None,
) -> None:
    """Post-case: invoke the swimlane converter on the perf file the runtime
    just wrote into ``<output_prefix>/l2_perf_records.json``. No diff/rename
    dance — the path is known a priori from CallConfig.output_prefix.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    perf_file = output_prefix / "l2_perf_records.json"
    if not perf_file.exists():
        logger.warning(f"[{case_label}] {perf_file} not produced; skipping conversion")
        return

    # Dump callable name mapping if the CALLABLE spec provides names
    func_names_path = None
    if callable_spec:
        mapping = _extract_name_map(callable_spec)
        safe_label = _sanitize_for_filename(case_label)
        func_names_path = _dump_name_map(mapping, output_prefix / f"name_map_{safe_label}.json")

    _run_swimlane_converter(input_path=perf_file, func_names_path=func_names_path)


def run_class_cases(  # noqa: PLR0913 -- shared layer-5 entry; kwargs mirror CLI surface
    worker,
    cls_inst,
    cases,
    *,
    callable_obj,
    sub_ids,
    rounds,
    skip_golden,
    enable_l2_swimlane,
    enable_dump_tensor,
    enable_pmu,
    enable_dep_gen,
):
    """Execute a pre-filtered list of cases for one class (layers 5-6).

    Caller is responsible for platform/selector/manual filtering. Profiling
    snapshots wrap each case. Validation failures propagate; caller decides
    fail-fast vs collect semantics.
    """
    cls_name = type(cls_inst).__name__
    callable_spec = getattr(type(cls_inst), "CALLABLE", None)
    diagnostics_on = enable_l2_swimlane or enable_dump_tensor or enable_pmu or enable_dep_gen
    for case in cases:
        case_label = f"{cls_name}_{case['name']}"
        # Per-case directory the runtime writes into. Required (non-empty) when
        # any diagnostic flag is on; CallConfig::validate() throws otherwise.
        prefix = _build_output_prefix(case_label) if diagnostics_on else Path("")
        try:
            cls_inst._run_and_validate(
                worker,
                callable_obj,
                case,
                sub_ids=sub_ids,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=str(prefix) if diagnostics_on else "",
            )
        finally:
            if enable_l2_swimlane:
                _convert_case_swimlane(case_label, prefix, callable_spec=callable_spec)


def _compare_outputs(test_args, golden_args, output_names, rtol, atol):
    """Compare output tensors against golden values."""
    import torch  # noqa: PLC0415

    for name in output_names:
        actual = getattr(test_args, name)
        expected = getattr(golden_args, name)
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            diff = (actual - expected).abs().max().item()
            raise AssertionError(f"Golden mismatch on '{name}': max_diff={diff}, rtol={rtol}, atol={atol}")


def _compile_chip_callable_from_spec(spec, platform, runtime, cache_key):
    """Compile a chip entry spec (orchestration + incores) -> ChipCallable. Session-cached."""
    if cache_key in _compile_cache:
        return _compile_cache[cache_key]

    if platform == "cuda" or "cuda" in spec:
        compiled = _compile_cuda_callable_from_spec(spec, runtime)
        _compile_cache[cache_key] = compiled
        return compiled

    from simpler.task_interface import ChipCallable, CoreCallable  # noqa: PLC0415

    from .elf_parser import extract_text_section  # noqa: PLC0415
    from .kernel_compiler import KernelCompiler  # noqa: PLC0415
    from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    orch = spec["orchestration"]
    incores = spec["incores"]

    pto_isa_root = ensure_pto_isa_root()
    kc = KernelCompiler(platform=platform)
    is_sim = platform.endswith("sim")

    orch_binary = kc.compile_orchestration(runtime, orch["source"])
    inc_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_binaries = []
    for k in incores:
        incore = kc.compile_incore(
            k["source"], core_type=k["core_type"], pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs
        )
        if not is_sim:
            incore = extract_text_section(incore)
        kernel_binaries.append((k["func_id"], CoreCallable.build(signature=k.get("signature", []), binary=incore)))

    chip_callable = ChipCallable.build(
        signature=orch.get("signature", []),
        func_name=orch["function_name"],
        binary=orch_binary,
        children=kernel_binaries,
        config_name=orch.get("config_name", ""),
    )
    _compile_cache[cache_key] = chip_callable
    return chip_callable


def _compile_cuda_callable_from_spec(spec, runtime):
    cuda = spec.get("cuda")
    if not isinstance(cuda, dict):
        raise ValueError("CUDA SceneTestCase CALLABLE must include a 'cuda' spec")
    if cuda.get("runtime", runtime) != runtime:
        raise ValueError(f"CUDA CALLABLE runtime {cuda.get('runtime')!r} does not match scene runtime {runtime!r}")

    from .cuda_callable_compiler import (  # noqa: PLC0415
        prepare_cuda_host_schedule_callable,
        prepare_cuda_persistent_device_callable,
    )
    from .kernel_compiler import KernelCompiler  # noqa: PLC0415

    kc = KernelCompiler(platform="cuda")
    if runtime == "persistent_device":
        artifact = kc.compile_cuda_persistent_device(
            _cuda_persistent_task_sources(cuda),
            arch=cuda.get("arch", os.environ.get("PTO_CUDA_ARCH", "compute_80")),
            cache_root=cuda.get("cache_root"),
            nvcc=cuda.get("nvcc", "nvcc"),
        )
        return prepare_cuda_persistent_device_callable(
            artifact,
            grid_dim=int(cuda["grid_dim"]),
            block_dim=int(cuda.get("block_dim", 256)),
            shared_mem_bytes=int(cuda.get("shared_mem_bytes", 0)),
            stream_id=int(cuda.get("stream_id", 0)),
            op=int(cuda.get("op", 1003)),
        )
    if runtime != "host_schedule":
        raise NotImplementedError(f"CUDA SceneTestCase compilation is not implemented for runtime={runtime!r}")

    artifact = kc.compile_cuda_host_schedule(
        cuda["source"],
        task_name=cuda["task_name"],
        arch=cuda.get("arch", os.environ.get("PTO_CUDA_ARCH", "compute_80")),
        context_type=cuda.get("context_type", "PtoTaskContext"),
        context_definition=cuda.get("context_definition", ""),
        host_parameters=tuple(cuda.get("host_parameters", ())),
        host_context_initializer=cuda.get("host_context_initializer", ""),
        cache_root=cuda.get("cache_root"),
        nvcc=cuda.get("nvcc", "nvcc"),
    )
    return prepare_cuda_host_schedule_callable(
        artifact,
        grid_dim=int(cuda["grid_dim"]),
        block_dim=int(cuda.get("block_dim", 256)),
        shared_mem_bytes=int(cuda.get("shared_mem_bytes", 0)),
        stream_id=int(cuda.get("stream_id", 0)),
        op=int(cuda.get("op", 1)),
    )


def _cuda_persistent_task_sources(cuda: dict[str, Any]) -> list[dict[str, Any]]:
    if "task_sources" in cuda:
        return [_normalize_cuda_task_source_entry(task_source) for task_source in cuda["task_sources"]]

    task_sources_path = cuda.get("task_sources_path", cuda.get("task_sources_file"))
    if task_sources_path is None:
        raise ValueError("CUDA persistent_device CALLABLE requires task_sources or task_sources_path")

    task_sources_file_path = Path(task_sources_path)
    with task_sources_file_path.open(encoding="utf-8") as task_sources_file:
        loaded = json.load(task_sources_file)

    if isinstance(loaded, dict):
        loaded = loaded.get("task_sources", loaded.get("sources"))
    if not isinstance(loaded, list):
        raise ValueError("CUDA persistent_device task source sidecar must contain a list")

    return [
        _resolve_cuda_task_source_sidecar_entry(task_source, task_sources_file_path.parent) for task_source in loaded
    ]


def _resolve_cuda_task_source_sidecar_entry(task_source: Any, base_dir: Path) -> dict[str, Any]:
    if not isinstance(task_source, dict):
        raise ValueError("CUDA persistent_device task source sidecar entries must be dictionaries")

    return _normalize_cuda_task_source_entry(task_source, base_dir=base_dir)


def _normalize_cuda_task_source_entry(task_source: Any, base_dir: Path | None = None) -> dict[str, Any]:
    if not isinstance(task_source, dict):
        raise ValueError("CUDA persistent_device task source entries must be dictionaries")

    resolved = dict(task_source)
    source_path = resolved.get("source_path", resolved.get("source"))
    if source_path is not None:
        resolved["source_path"] = _resolve_path_value(source_path, base_dir) if base_dir is not None else source_path
    return resolved


# ---------------------------------------------------------------------------
# @scene_test decorator
# ---------------------------------------------------------------------------


def scene_test(level: int, runtime: str):
    """Decorator marking a SceneTestCase with level and runtime.

    Platforms are declared per-case in CASES, not here.
    """

    def decorator(cls):
        cls._st_level = level
        cls._st_runtime = runtime
        cls_dir = Path(inspect.getfile(cls)).parent
        if hasattr(cls, "CALLABLE"):
            _resolve_callable_paths(cls, cls_dir)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# SceneTestCase base class
# ---------------------------------------------------------------------------


class SceneTestCase:
    """Base class for scene tests at any hierarchy level.

    Subclasses declare CALLABLE, CASES, generate_args(), compute_golden().
    """

    CALLABLE: dict = {}
    CASES: list[dict] = []
    RTOL: float = 1e-5
    ATOL: float = 1e-5
    RUNTIME_ENV: dict = {}

    def generate_args(self, params) -> TaskArgsBuilder:
        """Return TaskArgsBuilder with ordered Tensor/Scalar specs."""
        raise NotImplementedError

    def compute_golden(self, args: TaskArgsBuilder, params) -> None:
        """Compute expected outputs in-place on a cloned TaskArgsBuilder."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Callable compilation
    # ------------------------------------------------------------------

    @classmethod
    def compile_chip_callable(cls, platform):
        """Compile CALLABLE -> ChipCallable (L2). Session-cached."""
        cache_key = (cls.__qualname__, platform, cls._st_runtime)
        return _compile_chip_callable_from_spec(cls.CALLABLE, platform, cls._st_runtime, cache_key)

    @classmethod
    def _compile_l3_callables(cls, platform):
        """Compile all ChipCallable entries in CALLABLE['callables'] (L3)."""
        compiled = {}
        for entry in cls.CALLABLE["callables"]:
            if "orchestration" in entry:
                name = entry["name"]
                cache_key = (cls.__qualname__, name, platform, cls._st_runtime)
                chip = _compile_chip_callable_from_spec(entry, platform, cls._st_runtime, cache_key)
                compiled[name] = chip
                compiled[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        return compiled

    # ------------------------------------------------------------------
    # Worker creation
    # ------------------------------------------------------------------

    @classmethod
    def _create_worker(cls, platform, device_id=0, build=False):
        """Create the L2 Worker for the standalone path.

        Mirrors the ``st_worker`` pytest fixture, which yields a ``Worker``
        (not a raw ``ChipWorker``) — ``_run_and_validate_l2`` is shared by both
        paths and calls ``worker.register(...)`` / ``worker.run(cid, ...)``,
        which only the ``Worker`` wrapper exposes.
        """
        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(level=2, device_id=device_id, platform=platform, runtime=cls._st_runtime, build=build)
        w.init()
        return w

    # ------------------------------------------------------------------
    # Default build methods
    # ------------------------------------------------------------------

    def build_callable(self, platform):
        """Build callable for the current level.

        L2: returns ChipCallable.
        L3: returns dict of {name: ChipCallable, name_sig: signature}.
        """
        if self._st_level == 2:
            return self.compile_chip_callable(platform)
        elif self._st_level == 3:
            return self._compile_l3_callables(platform)
        raise ValueError(f"Unsupported level: {self._st_level}")

    def _build_config(
        self,
        config_dict,
        enable_l2_swimlane=0,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        *,
        output_prefix="",
    ):
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        config = CallConfig()
        config.block_dim = config_dict.get("block_dim", 1)
        config.aicpu_thread_num = config_dict.get("aicpu_thread_num", 3)
        config.enable_l2_swimlane = enable_l2_swimlane
        config.enable_dump_tensor = enable_dump_tensor
        config.enable_pmu = enable_pmu  # 0=disabled, >0=enabled with event type
        config.enable_dep_gen = enable_dep_gen
        # `output_prefix` is required by CallConfig::validate() whenever any
        # diagnostic flag is enabled. Caller threads it down from the per-case
        # directory built by _build_output_prefix().
        if output_prefix:
            config.output_prefix = str(output_prefix)
        return config

    def _resolve_env(self):
        env = self.RUNTIME_ENV
        if not env:
            return {}
        cls_dir = Path(inspect.getfile(type(self))).parent
        out = {}
        for k, v in env.items():
            s = str(v)
            if (k.endswith("_DIR") or k.endswith("_PATH")) and not Path(s).is_absolute():
                s = str((cls_dir / s).resolve())
            out[k] = s
        return out

    # ------------------------------------------------------------------
    # Run + validate
    # ------------------------------------------------------------------

    def _run_and_validate(  # noqa: PLR0913 -- threads CLI diagnostic flags + case context
        self,
        worker,
        callable_obj,
        case,
        sub_ids=None,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        output_prefix="",
    ):
        if self._st_level == 2:
            self._run_and_validate_l2(
                worker,
                callable_obj,
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=output_prefix,
            )
        elif self._st_level == 3:
            self._run_and_validate_l3(
                worker,
                callable_obj,
                sub_ids or {},
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=output_prefix,
            )

    def _run_and_validate_l2(
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        output_prefix="",
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = _callable_signature(self.CALLABLE)

        # The L2 entry point is `Worker.run(cid, args, cfg)`.  Reuse the
        # cid registered by the st_worker fixture / standalone path.  For
        # first-time callers (worker reused across rounds), `_st_l2_cid`
        # caches the cid so subsequent runs skip re-registration.
        cid = getattr(type(self), "_st_l2_cid", None)
        if cid is None:
            cid = worker.register(callable_obj)
            type(self)._st_l2_cid = cid

        if getattr(callable_obj, "runtime", None) == "host_schedule" and "cuda" in self.CALLABLE:
            self._run_and_validate_l2_cuda_host_schedule(
                worker,
                cid,
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=output_prefix,
            )
            return
        if getattr(callable_obj, "runtime", None) == "persistent_device" and "cuda" in self.CALLABLE:
            self._run_and_validate_l2_cuda_persistent_device(
                worker,
                cid,
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=output_prefix,
            )
            return

        # Build args
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)

        # Compute golden (unless skip_golden)
        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        # Save initial output tensor values for reset between rounds
        initial_outputs = {}
        if rounds > 1:
            for name in output_names:
                initial_outputs[name] = getattr(test_args, name).clone()

        # Execute rounds
        timings = []  # populated only when rounds > 1
        for round_idx in range(rounds):
            if round_idx > 0:
                for name, initial in initial_outputs.items():
                    getattr(test_args, name).copy_(initial)

            # enable_l2_swimlane / enable_dep_gen are already forced False by
            # the upstream gate in test_run / run_module when rounds > 1, so an
            # extra `and round_idx == 0` here is dead code; pass them through
            # verbatim. (If the upstream gate is ever relaxed, restore the
            # per-round masking here.)
            config = self._build_config(
                config_dict,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=output_prefix,
            )

            with _temporary_env(self._resolve_env()):
                timing = worker.run(cid, chip_args, config=config)
            if rounds > 1 and timing is not None:
                timings.append((timing.host_wall_us, timing.device_wall_us))

            if not skip_golden:
                _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        if timings:
            _log_round_timings(timings)

    def _run_and_validate_l2_cuda_host_schedule(
        self,
        worker,
        cid,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        output_prefix="",
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        cuda_spec = self.CALLABLE["cuda"]

        test_args = self.generate_args(params)
        output_names = _output_names_from_signature(test_args, _callable_signature(self.CALLABLE))

        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        initial_outputs = {}
        if rounds > 1:
            for name in output_names:
                initial_outputs[name] = getattr(test_args, name).clone()

        device_buffers = _CudaSceneDeviceBuffers(worker, test_args)
        timings = []
        try:
            for round_idx in range(rounds):
                if round_idx > 0:
                    for name, initial in initial_outputs.items():
                        getattr(test_args, name).copy_(initial)

                device_buffers.copy_all_to_device()
                raw_args = _build_cuda_host_schedule_args(test_args, device_buffers, cuda_spec)
                config = self._build_config(
                    config_dict,
                    enable_l2_swimlane=enable_l2_swimlane,
                    enable_dump_tensor=enable_dump_tensor,
                    enable_pmu=enable_pmu,
                    enable_dep_gen=enable_dep_gen,
                    output_prefix=output_prefix,
                )

                with _temporary_env(self._resolve_env()):
                    timing = worker.run(cid, raw_args, config=config)
                device_buffers.copy_outputs_from_device(output_names)
                if rounds > 1 and timing is not None:
                    timings.append((timing.host_wall_us, timing.device_wall_us))

                if not skip_golden:
                    _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)
        finally:
            device_buffers.free()

        if timings:
            _log_round_timings(timings)

    def _run_and_validate_l2_cuda_persistent_device(
        self,
        worker,
        cid,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        output_prefix="",
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        cuda_spec = self.CALLABLE["cuda"]

        test_args = self.generate_args(params)
        output_names = _output_names_from_signature(test_args, _callable_signature(self.CALLABLE))

        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        initial_outputs = {}
        if rounds > 1:
            for name in output_names:
                initial_outputs[name] = getattr(test_args, name).clone()

        device_buffers = _CudaPersistentDagSceneBuffers(worker, test_args, cuda_spec)
        timings = []
        try:
            for round_idx in range(rounds):
                if round_idx > 0:
                    for name, initial in initial_outputs.items():
                        getattr(test_args, name).copy_(initial)

                device_buffers.reset_for_run()
                config = self._build_config(
                    config_dict,
                    enable_l2_swimlane=enable_l2_swimlane,
                    enable_dump_tensor=enable_dump_tensor,
                    enable_pmu=enable_pmu,
                    enable_dep_gen=enable_dep_gen,
                    output_prefix=output_prefix,
                )

                with _temporary_env(self._resolve_env()):
                    timing = worker.run(cid, device_buffers.args, config=config)
                counters = device_buffers.read_counters()
                if counters["error_count"] != 0:
                    raise RuntimeError(
                        "CUDA persistent DAG scheduler error "
                        f"code={counters['error_code']} task_id={counters['error_task_id']} "
                        f"count={counters['error_count']}"
                    )
                if counters["completed_count"] != len(device_buffers.host_tasks):
                    raise RuntimeError(
                        "CUDA persistent DAG completed "
                        f"{counters['completed_count']} of {len(device_buffers.host_tasks)} tasks"
                    )
                device_buffers.copy_outputs_from_device(output_names)
                if rounds > 1 and timing is not None:
                    timings.append((timing.host_wall_us, timing.device_wall_us))

                if not skip_golden:
                    _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)
        finally:
            device_buffers.free()

        if timings:
            _log_round_timings(timings)

    def _run_and_validate_l3(  # noqa: PLR0913 -- threads CLI diagnostic flags + L3 ns context
        self,
        worker,
        compiled_callables,
        sub_ids,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        output_prefix="",
    ):
        # Defensive belt-and-braces: the pytest dispatcher and run_module both
        # block --enable-l2-swimlane for L3 at the CLI boundary. Catch any code
        # path that reaches here with the flag on anyway (direct API use,
        # future refactors) so we fail loud rather than produce garbage perf
        # files. Lift once the runtime embeds device_id in the perf filename.
        if enable_l2_swimlane:
            raise NotImplementedError(
                "L3 profiling is not supported yet (multi-chip-process perf "
                "filename collision). Gate at the CLI level in "
                "conftest.pytest_collection_modifyitems / scene_test.run_module."
            )

        params = case.get("params", {})
        config_dict = case.get("config", {})

        # Build args
        test_args = self.generate_args(params)

        # Compute golden (unless skip_golden)
        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        # Save initial tensor values for reset between rounds
        all_tensor_names = test_args.tensor_names()
        initial_tensors = {}
        if rounds > 1:
            for name in all_tensor_names:
                initial_tensors[name] = getattr(test_args, name).clone()

        # Build CallableNamespace: compiled ChipCallables + sub callable IDs
        ns = CallableNamespace({**compiled_callables, **sub_ids})

        # Get orch function (plain function from CALLABLE)
        orch_fn = self.CALLABLE["orchestration"]

        # Execute rounds
        timings = []
        for round_idx in range(rounds):
            if round_idx > 0:
                for name, initial in initial_tensors.items():
                    getattr(test_args, name).copy_(initial)

            # See _run_and_validate_l2: the per-round masking is dead code
            # under the existing upstream gate. Keep parity by passing through.
            config = self._build_config(
                config_dict,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                output_prefix=output_prefix,
            )

            # Orch fn signature: (orch, args, cfg) — inner fn forwards to
            # the user's scene orch which takes (orch, callables, task_args, config).
            def task_orch(orch, _args, _cfg, _ns=ns, _test_args=test_args, _config=config):
                orch_fn(orch, _ns, _test_args, _config)

            with _temporary_env(self._resolve_env()):
                timing = worker.run(task_orch)
            if rounds > 1 and timing is not None:
                # L3+ DAGs surface host wall only; device_wall_us is 0 because
                # per-task device cycles aren't aggregated up to Worker.run.
                timings.append((timing.host_wall_us, timing.device_wall_us))

            if not skip_golden:
                _compare_outputs(test_args, golden_args, all_tensor_names, self.RTOL, self.ATOL)

        if timings:
            _log_round_timings(timings)

    # ------------------------------------------------------------------
    # pytest auto test method
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_enable_dep_gen(request, *, warn: bool = False) -> bool:
        """``--enable-dep-gen`` CLI value after applying the ``--rounds > 1``
        disable. Single source of truth so the framework's ``test_run`` loop
        and any subclass override (e.g. ``TestDepGenCapture``'s post-validate
        hook) can't drift on the gating rule. Pass ``warn=True`` from the
        framework's first call — it owns the user-facing "disabled because
        rounds > 1" message; subclass overrides leave ``warn`` off since
        ``super().test_run()`` already warned."""
        if not request.config.getoption("--enable-dep-gen", default=False):
            return False
        if request.config.getoption("--rounds", default=1) > 1:
            if warn:
                logger.warning("dep_gen disabled: --rounds > 1")
            return False
        return True

    def test_run(self, st_platform, st_worker, request):
        """Auto test method — runs matching cases for the current platform."""
        raw_selectors = request.config.getoption("--case", default=None) or []
        selectors = [_parse_case_selector(v) for v in raw_selectors]
        manual_mode = request.config.getoption("--manual", default="exclude")
        rounds = request.config.getoption("--rounds", default=1)
        skip_golden = request.config.getoption("--skip-golden", default=False)
        enable_l2_swimlane = request.config.getoption("--enable-l2-swimlane", default=0)
        enable_dump_tensor = request.config.getoption("--dump-tensor", default=False)
        enable_pmu = request.config.getoption("--enable-pmu", default=0)
        enable_dep_gen = self._effective_enable_dep_gen(request, warn=True)
        if rounds > 1:
            if enable_l2_swimlane:
                logger.warning("Profiling disabled: --rounds > 1")
                enable_l2_swimlane = 0
            if enable_dump_tensor:
                logger.warning("Dump tensor disabled: --rounds > 1")
                enable_dump_tensor = False
            if enable_pmu:
                logger.warning("PMU disabled: --rounds > 1")
                enable_pmu = 0

        cls_name = type(self).__name__
        callable_obj = self.build_callable(st_platform)
        sub_ids = getattr(type(self), "_st_sub_ids", {})
        # For L3, use pre-registered chip cids instead of raw ChipCallable
        # objects.
        chip_cids = getattr(type(self), "_st_chip_cids", {})
        if self._st_level == 3 and chip_cids:
            callable_obj = {**chip_cids}

        matched = []
        for case in self.CASES:
            if st_platform not in case["platforms"]:
                continue
            if not _match_selectors(cls_name, case["name"], selectors):
                continue
            is_manual = bool(case.get("manual"))
            if manual_mode == "exclude" and is_manual:
                continue
            if manual_mode == "only" and not is_manual:
                continue
            matched.append(case)

        if not matched:
            import pytest  # noqa: PLC0415

            pytest.skip(f"No cases matched {cls_name} (platform={st_platform}, manual={manual_mode})")

        run_class_cases(
            st_worker,
            self,
            matched,
            callable_obj=callable_obj,
            sub_ids=sub_ids,
            rounds=rounds,
            skip_golden=skip_golden,
            enable_l2_swimlane=enable_l2_swimlane,
            enable_dump_tensor=enable_dump_tensor,
            enable_pmu=enable_pmu,
            enable_dep_gen=enable_dep_gen,
        )

    # ------------------------------------------------------------------
    # Standalone entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_module(module_name):  # noqa: PLR0912, PLR0915 -- CLI parsing + dispatch; branches map to user-facing flags
        """Standalone entry: ``if __name__ == "__main__": SceneTestCase.run_module(__name__)``.

        Supports -d as either a single id or a range ("0-7"). When more than
        one device is provided (or any L3 case needs more than its single
        device), the outer invocation becomes a test dispatcher that spawns
        per-case subprocesses via ``parallel_scheduler``; each child re-enters
        this function in single-group mode via ``--runtime`` + ``--level``.
        """
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--platform", required=True)
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="0",
            help="Device id or range ('0', '4-7', '0,2,5')",
        )
        parser.add_argument(
            "--case",
            action="append",
            default=None,
            help="Case selector; repeatable. Forms: 'Foo' (any class), 'ClassA::Foo', 'ClassA::'",
        )
        parser.add_argument(
            "--manual",
            choices=["exclude", "include", "only"],
            default="exclude",
            help="Manual case handling: exclude (default), include, only",
        )
        parser.add_argument("--rounds", type=int, default=1, help="Run each case N times (default: 1)")
        parser.add_argument("--skip-golden", action="store_true", help="Skip golden comparison (benchmark mode)")
        parser.add_argument(
            "--enable-l2-swimlane",
            nargs="?",
            const=4,
            default=0,
            type=int,
            metavar="PERF_LEVEL",
            help="Enable L2 swimlane. Bare flag=level 4 (full). "
            "1=AICore timing, 2=+dispatch/fanout, 3=+sched phases, 4=+orch phases",
        )
        parser.add_argument("--dump-tensor", action="store_true", help="Dump per-task tensor I/O at runtime")
        parser.add_argument(
            "--enable-dep-gen",
            action="store_true",
            help="Enable dep_gen capture (SubmitTrace ring, first round only)",
        )
        parser.add_argument(
            "--enable-pmu",
            nargs="?",
            const=2,
            default=0,
            type=int,
            metavar="EVENT_TYPE",
            help="Enable PMU collection. Bare flag = PIPE_UTILIZATION(2). "
            "Pass event type to override (e.g. --enable-pmu 4)",
        )
        parser.add_argument("--build", action="store_true", help="Compile runtime from source")
        parser.add_argument(
            "--runtime",
            default=None,
            help="Only run classes with this _st_runtime (child-mode marker when combined with --level)",
        )
        parser.add_argument(
            "--level",
            type=int,
            choices=[2, 3],
            default=None,
            help="Only run classes with this _st_level (child-mode marker when combined with --runtime)",
        )
        parser.add_argument(
            "-x", "--exitfirst", action="store_true", help="Stop on first failing case (matches pytest -x)"
        )
        parser.add_argument(
            "--max-parallel",
            default="auto",
            help=(
                "Max in-flight subprocesses (make-style); decouples -d pool size from "
                "parallelism. 'auto' = min(nproc, len(-d)) on sim, len(-d) on hardware. "
                "Use e.g. '--max-parallel 2' to throttle sim on a CPU-constrained CI "
                "runner without shrinking -d. No short form — pytest reserves lowercase "
                "shorts; standalone mirrors that restriction for consistency."
            ),
        )
        parser.add_argument(
            "-c",
            "--pto-isa-commit",
            default=None,
            help="Checkout PTO-ISA at this git commit before running.",
        )
        parser.add_argument(
            "--clone-protocol",
            choices=["ssh", "https"],
            default="ssh",
            help="Git protocol for auto-cloning PTO-ISA (used with --pto-isa-commit). Default: ssh.",
        )
        parser.add_argument(
            "--log-level",
            choices=LOG_LEVEL_CHOICES,
            default=DEFAULT_LOG_LEVEL,
            help=f"Simpler logger level (debug/V0..V9/info/warn/error/null; default {DEFAULT_LOG_LEVEL})",
        )
        args = parser.parse_args()
        configure_logging(args.log_level)

        os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root(
            commit=args.pto_isa_commit,
            clone_protocol=args.clone_protocol,
            update_if_exists=True,
            verbose=True,
        )

        if args.rounds > 1 and args.enable_l2_swimlane:
            logger.warning("Profiling disabled: --rounds > 1")
            args.enable_l2_swimlane = 0
        if args.rounds > 1 and args.enable_dep_gen:
            logger.warning("dep_gen disabled: --rounds > 1")
            args.enable_dep_gen = False

        from .parallel_scheduler import default_max_parallel, device_range_to_list  # noqa: PLC0415

        device_ids = device_range_to_list(args.device)
        if not device_ids:
            print("ERROR: --device must be a non-empty id or range", file=sys.stderr)
            sys.exit(2)
        args.device_ids = device_ids
        # Keep ``args.device`` as an int for paths that expect a single id
        # (profiling snapshots, device-id binding inside one worker). In child
        # mode this is the single allocated id; in parent mode we use the first
        # slot but the dispatcher doesn't actually run tests here.
        args.device = device_ids[0]

        # Resolve -j (max parallel) — 'auto' is CPU-aware on sim, device-count on hardware.
        if args.max_parallel in (None, "", "auto"):
            args.max_parallel = default_max_parallel(args.platform, device_ids)
        else:
            try:
                args.max_parallel = int(args.max_parallel)
            except (TypeError, ValueError):
                print(f"ERROR: -j must be 'auto' or an integer, got {args.max_parallel!r}", file=sys.stderr)
                sys.exit(2)
            if args.max_parallel < 1:
                print(f"ERROR: -j must be >= 1, got {args.max_parallel}", file=sys.stderr)
                sys.exit(2)
        # Profiling + parallelism is safe: each test case sets its own
        # `output_prefix` on CallConfig (see run_class_cases) so diagnostic
        # artifacts land in distinct directories with no shared filenames.

        module = sys.modules[module_name]
        test_classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, SceneTestCase) and v is not SceneTestCase and hasattr(v, "CASES")
        ]

        # Apply --runtime/--level filters (child mode sets both; parent may also
        # use them when the user wants a narrow run).
        if args.runtime is not None:
            test_classes = [c for c in test_classes if getattr(c, "_st_runtime", None) == args.runtime]
        if args.level is not None:
            test_classes = [c for c in test_classes if getattr(c, "_st_level", None) == args.level]
        if not test_classes:
            print(
                f"No matching classes (runtime={args.runtime}, level={args.level})",
                file=sys.stderr,
            )
            sys.exit(0)

        selectors = [_parse_case_selector(v) for v in (args.case or [])]
        try:
            selected = _select_cases(test_classes, args.platform, selectors, args.manual)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

        selected_by_cls: dict[type, list[dict]] = {}
        for cls, case in selected:
            selected_by_cls.setdefault(cls, []).append(case)

        # L3 profiling not supported yet (multi-chip-process filename collision).
        # Mirror the pytest-side guard so standalone users get the same early-fail.
        if args.enable_l2_swimlane:
            l3_classes = sorted(cls.__name__ for cls in selected_by_cls if cls._st_level == 3)
            if l3_classes:
                print(
                    f"ERROR: --enable-l2-swimlane is not supported for L3 tests yet — "
                    f"multi-chip-process filename collision unresolved. "
                    f"L3 classes selected: {', '.join(l3_classes)}. "
                    f"Either drop --enable-l2-swimlane or scope to L2 with --level 2.",
                    file=sys.stderr,
                )
                sys.exit(2)

        # Child mode: both --runtime and --level set. Run inline without
        # spawning further subprocesses; this is the path dispatcher
        # children take after we re-enter run_module.
        child_mode = args.runtime is not None and args.level is not None

        if not child_mode:
            has_multi_dev_case = any(
                int(case.get("config", {}).get("device_count", 1)) > 1
                for cases in selected_by_cls.values()
                for case in cases
            )
            has_multiple_groups = len({(cls._st_runtime, cls._st_level) for cls in selected_by_cls}) > 1
            needs_orchestration = len(device_ids) > 1 or has_multi_dev_case or has_multiple_groups
            if needs_orchestration:
                ok = _dispatch_test_phases_standalone(module_name, selected_by_cls, args)
                sys.exit(0 if ok else 1)

        # ----- Inline execution (single group or child mode) -----
        by_rt_level: dict[tuple[str, int], list[type]] = {}
        for cls in selected_by_cls:
            by_rt_level.setdefault((cls._st_runtime, cls._st_level), []).append(cls)

        ok = True
        for (runtime, level), group in by_rt_level.items():
            print(f"\n=== Runtime: {runtime}  Level: {level} ===")
            worker, per_class_sub_ids, per_class_chip_cids = _create_standalone_worker(
                group, level, args, selected_by_cls
            )
            try:
                for cls in group:
                    inst = cls()
                    callable_obj = inst.build_callable(args.platform)
                    sub_ids = per_class_sub_ids.get(cls, {})
                    chip_cids = per_class_chip_cids.get(cls, {})
                    # For L3: merge chip cids into callable_obj (replacing
                    # ChipCallable objects with their registered cid).
                    if level == 3 and chip_cids:
                        callable_obj = {**chip_cids}
                    for case in selected_by_cls[cls]:
                        label = f"{cls.__name__}::{case['name']}"
                        print(f"  {label} ... ", end="", flush=True)
                        try:
                            run_class_cases(
                                worker,
                                inst,
                                [case],
                                callable_obj=callable_obj,
                                sub_ids=sub_ids,
                                rounds=args.rounds,
                                skip_golden=args.skip_golden,
                                enable_l2_swimlane=args.enable_l2_swimlane,
                                enable_dump_tensor=args.dump_tensor,
                                enable_pmu=args.enable_pmu,
                                enable_dep_gen=args.enable_dep_gen,
                            )
                            print("PASSED")
                        except Exception as e:  # noqa: BLE001
                            print(f"FAILED: {e}")
                            ok = False
                            if args.exitfirst:
                                raise SystemExit(1) from None
            finally:
                worker.close()

        sys.exit(0 if ok else 1)


def _dispatch_test_phases_standalone(module_name, selected_by_cls, args):  # noqa: PLR0912 -- L3 + L2 phases + chunking + fail-fast
    """Parent-mode test dispatcher for run_module.

    L3 phase: one subprocess per class, scheduled by device count.
    L2 phase: per-runtime fanout — up to max_parallel concurrent subprocesses,
    each owning one device and running its round-robin chunk of classes.

    Returns True on full success, False if any child failed.
    """
    from .parallel_scheduler import Job, format_device_range, run_jobs  # noqa: PLC0415

    module = sys.modules[module_name]
    # Path to the user's test script — sys.argv[0] is the script they invoked.
    script = os.path.abspath(getattr(module, "__file__", sys.argv[0]))

    common = ["-p", args.platform, "--manual", args.manual, "--log-level", args.log_level]
    if args.rounds != 1:
        common += ["--rounds", str(args.rounds)]
    if args.skip_golden:
        common.append("--skip-golden")
    if args.enable_l2_swimlane:
        common += ["--enable-l2-swimlane", str(args.enable_l2_swimlane)]
    if args.dump_tensor:
        common.append("--dump-tensor")
    if args.enable_dep_gen:
        common.append("--enable-dep-gen")
    if args.build:
        common.append("--build")

    # ----- L3 phase: one subprocess per class (not per case).
    # The child's _create_standalone_worker allocates max(cls.CASES.device_count)
    # for the whole class, so the scheduler must grant the class-level max,
    # otherwise a class with a 4-device case can't run any of its 1-device
    # cases when we dispatch them individually with --device <1>. Cases inside
    # a class still run serially in the child, reusing the L3 Worker.
    l3_jobs = []
    for cls, cases in selected_by_cls.items():
        if cls._st_level != 3:
            continue
        if not cases:
            continue
        class_dev_count = max(int(c.get("config", {}).get("device_count", 1)) for c in cases)
        label = f"L3 {cls.__name__} (rt={cls._st_runtime}, dev={class_dev_count})"

        def _build(ids, _cls=cls.__name__, _rt=cls._st_runtime):
            return [
                sys.executable,
                script,
                *common,
                "-d",
                format_device_range(ids),
                "--case",
                f"{_cls}::",
                "--runtime",
                _rt,
                "--level",
                "3",
            ]

        # Per-case output_prefix is chosen inside the child by run_class_cases,
        # so no env var is needed to scope concurrent jobs.
        l3_jobs.append(Job(label=label, device_count=class_dev_count, build_cmd=_build))

    l3_failed = False
    if l3_jobs:
        print(
            f"\n{'=' * 60}\n  L3 phase: {len(l3_jobs)} case(s), pool={args.device_ids}, "
            f"max_parallel={args.max_parallel}\n{'=' * 60}\n"
        )

        def _on_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"  {res.label}: {tag} on devices {res.device_ids}", flush=True)

        try:
            results = run_jobs(
                l3_jobs,
                args.device_ids,
                max_parallel=args.max_parallel,
                fail_fast=args.exitfirst,
                on_job_done=_on_done,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return False
        l3_failed = any(r.returncode != 0 for r in results)
        if l3_failed and args.exitfirst:
            return False

    # ----- L2 phase: runtimes serial (CANN isolation); within a runtime, fan
    # out classes across device_ids as one subprocess per device. Each child
    # owns one ChipWorker and runs its chunk of classes back-to-back (layer-4
    # reuse). Single-device case reduces to one subprocess per runtime.
    l2_by_runtime: dict[str, list[type]] = {}
    for cls in selected_by_cls:
        if cls._st_level == 2:
            l2_by_runtime.setdefault(cls._st_runtime, []).append(cls)

    l2_failed = False
    for rt in sorted(l2_by_runtime):
        classes = l2_by_runtime[rt]
        # Chunk count = min(-j, number of classes). We intentionally do NOT
        # include len(device_ids) here: each chunk uses 1 device and at most
        # max_parallel chunks run concurrently, so a pool bigger than -j just
        # leaves unused ids. Fewer, larger chunks also amortize ChipWorker
        # init (layer-4 reuse) over more cases.
        n = min(args.max_parallel, len(classes))
        if n == 0:
            continue
        # Round-robin distribute classes to N children.
        chunks: list[list[type]] = [[] for _ in range(n)]
        for i, cls in enumerate(classes):
            chunks[i % n].append(cls)

        header = f"  L2 Runtime: {rt}" + (f"  [fanout n={n}]" if n > 1 else "")
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}\n")

        l2_jobs = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            dev = args.device_ids[i]
            case_filters: list[str] = []
            for cls in chunk:
                # User-supplied selectors still filter; we scope to this chunk's
                # classes using "ClassName::<case>" or "ClassName::" (whole class).
                for sel in args.case or []:
                    if "::" in sel:
                        sel_cls, sel_case = sel.split("::", 1)
                        if not sel_cls or sel_cls == cls.__name__:
                            case_filters.append(f"{cls.__name__}::{sel_case}" if sel_case else f"{cls.__name__}::")
                    else:
                        # Bare selector "Foo" = case name in any class — forward as-is,
                        # but still scope to this chunk's classes.
                        case_filters.append(f"{cls.__name__}::{sel}")
                if not args.case:
                    case_filters.append(f"{cls.__name__}::")
            label = f"L2 {rt} dev={dev} ({len(chunk)} class(es))"

            def _build(ids, _rt=rt, _dev=dev, _filters=tuple(case_filters)):
                cmd = [
                    sys.executable,
                    script,
                    *common,
                    "-d",
                    str(_dev),
                    "--runtime",
                    _rt,
                    "--level",
                    "2",
                ]
                for f in _filters:
                    cmd += ["--case", f]
                return cmd

            # device_count=1 for L2 fanout children (each child uses one slot).
            # Per-case output_prefix is chosen inside the child by run_class_cases,
            # so no env var is needed to scope concurrent jobs.
            l2_jobs.append(Job(label=label, device_count=1, build_cmd=_build))

        # Use the same scheduler: pool=device_ids, fail_fast=exitfirst. This
        # gives us automatic parallelism + SIGTERM on fail-fast.
        def _on_l2_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"  {res.label}: {tag}", flush=True)

        try:
            results = run_jobs(
                l2_jobs,
                args.device_ids,
                max_parallel=args.max_parallel,
                fail_fast=args.exitfirst,
                on_job_done=_on_l2_done,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            l2_failed = True
            if args.exitfirst:
                break
            continue

        if any(r.returncode != 0 for r in results):
            l2_failed = True
            if args.exitfirst:
                break

    return not (l3_failed or l2_failed)


def _create_standalone_worker(group, level, args, selected_by_cls):
    """Create a Worker for a (runtime, level) group in run_module.

    ``level`` is passed explicitly by the caller; do not read it from
    ``group[0]._st_level`` because groups are now keyed on (runtime, level)
    and mixed-level files are allowed.

    ``selected_by_cls`` is the dict of the cases that will actually run (after
    ``--case`` / ``--manual`` / platform filtering). L3 ``max_devices`` /
    ``max_sub_workers`` must be computed from these, not from ``cls.CASES``:
    otherwise a manual case with a larger ``device_count`` inflates the
    allocation even when it isn't scheduled.

    Returns ``(worker, per_class_sub_ids, per_class_chip_cids)`` for both
    L2 and L3 so the caller can unpack uniformly. L2 has neither sub
    callables nor pre-registered chip callables, so both dicts are empty.
    """
    first_cls = group[0]
    build = getattr(args, "build", False)
    if level == 2:
        return first_cls._create_worker(args.platform, args.device, build=build), {}, {}

    from simpler.worker import Worker  # noqa: PLC0415

    max_devices = max(
        (c.get("config", {}).get("device_count", 1) for cls in group for c in selected_by_cls.get(cls, [])),
        default=1,
    )
    max_subs = max(
        (c.get("config", {}).get("num_sub_workers", 0) for cls in group for c in selected_by_cls.get(cls, [])),
        default=0,
    )
    # Prefer the allocated list (dispatcher child mode), fall back to
    # contiguous range starting at args.device (legacy inline path).
    allocated = getattr(args, "device_ids", None)
    if allocated and len(allocated) >= max_devices:
        device_ids = allocated[:max_devices]
    else:
        device_ids = list(range(args.device, args.device + max_devices))
    worker = Worker(
        level=3,
        device_ids=device_ids,
        num_sub_workers=max_subs,
        platform=args.platform,
        runtime=first_cls._st_runtime,
        build=build,
    )
    # Register sub callables per-class to avoid name collisions
    per_class_sub_ids: dict[type, dict] = {}
    # Also register ChipCallables here (before init) so the chip children
    # pre-warm them via _CTRL_PREPARE.
    per_class_chip_cids: dict[type, dict] = {}
    for cls in group:
        cls_sub_ids = {}
        cls_chip_cids = {}
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                cid = worker.register(entry["callable"])
                cls_sub_ids[entry["name"]] = cid
            elif "orchestration" in entry:
                name = entry["name"]
                cache_key = (cls.__qualname__, name, args.platform, cls._st_runtime)
                chip = _compile_chip_callable_from_spec(entry, args.platform, cls._st_runtime, cache_key)
                cid = worker.register(chip)
                cls_chip_cids[name] = cid
                cls_chip_cids[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        per_class_sub_ids[cls] = cls_sub_ids
        per_class_chip_cids[cls] = cls_chip_cids
    worker.init()
    return worker, per_class_sub_ids, per_class_chip_cids
