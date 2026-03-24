"""Core types for the unified PTO runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Union

import numpy as np


class ParamType(IntEnum):
    SCALAR = 0
    INPUT = 1
    OUTPUT = 2
    INOUT = 3


class Arg:
    """Unified argument descriptor for runtime submit/run calls.

    Replaces simpler's split arrays (func_args, arg_types, arg_sizes).
    """

    __slots__ = ("type", "data", "size")

    def __init__(self, type: ParamType, data: Any, size: int = 0):
        self.type = type
        self.data = data
        self.size = size

    # ---- Convenience constructors ----

    @classmethod
    def input(cls, data: np.ndarray) -> Arg:
        return cls(ParamType.INPUT, data, data.nbytes)

    @classmethod
    def output(cls, data: np.ndarray) -> Arg:
        return cls(ParamType.OUTPUT, data, data.nbytes)

    @classmethod
    def inout(cls, data: np.ndarray) -> Arg:
        return cls(ParamType.INOUT, data, data.nbytes)

    @classmethod
    def scalar(cls, value: Union[int, float]) -> Arg:
        return cls(ParamType.SCALAR, value, 0)

    def __repr__(self) -> str:
        if self.type == ParamType.SCALAR:
            return f"Arg.scalar({self.data})"
        type_name = self.type.name.lower()
        shape = getattr(self.data, "shape", "?")
        return f"Arg.{type_name}(shape={shape}, size={self.size})"


@dataclass
class TensorHandle:
    """Opaque handle for a host-memory tensor managed by the runtime."""

    id: int
    shape: tuple
    dtype: str
    size: int  # bytes
    _data: np.ndarray = field(repr=False, default=None)

    @property
    def data(self) -> np.ndarray:
        return self._data

    def numpy(self) -> np.ndarray:
        if self._data is None:
            raise ValueError("TensorHandle has no backing data")
        return self._data


@dataclass
class KernelSource:
    """Describes a single kernel source file to compile."""

    source: str
    core_type: str = "aiv"
    func_id: int = 0


@dataclass
class CompiledPackage:
    """A fully compiled L2 execution package.

    Contains all binaries needed to run a single L2 invocation:
    runtime binaries + orchestration .so + kernel binaries + config.
    """

    platform: str
    runtime_name: str
    host_binary: bytes
    aicpu_binary: bytes
    aicore_binary: bytes
    orch_binary: bytes
    orch_func: str
    kernel_binaries: list = field(default_factory=list)  # [(func_id, bytes)]
    block_dim: int = 1
    aicpu_thread_num: int = 1
    orch_thread_num: int = 1
