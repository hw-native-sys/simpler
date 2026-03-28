"""L2 (single-chip) runtime implementation.

Wraps the existing bindings.py ctypes interface behind the unified
init/register/run/close API.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Arg, CompiledPackage, ParamType

logger = logging.getLogger(__name__)

# Ensure simpler's python/ is importable
_PYTHON_DIR = str(Path(__file__).resolve().parent.parent)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


def _args_to_c_arrays(args: list[Arg]):
    """Convert a list of Arg to the parallel arrays that bindings.py expects.

    Returns:
        (func_args, arg_types, arg_sizes, host_tensors)
        host_tensors is a dict mapping index to numpy array for copy-back tracking.
    """
    from bindings import ARG_SCALAR, ARG_INPUT_PTR, ARG_OUTPUT_PTR, ARG_INOUT_PTR

    type_map = {
        ParamType.SCALAR: ARG_SCALAR,
        ParamType.INPUT: ARG_INPUT_PTR,
        ParamType.OUTPUT: ARG_OUTPUT_PTR,
        ParamType.INOUT: ARG_INOUT_PTR,
    }

    func_args = []
    arg_types = []
    arg_sizes = []

    for arg in args:
        arg_types.append(type_map[arg.type])
        if arg.type == ParamType.SCALAR:
            func_args.append(int(arg.data))
            arg_sizes.append(0)
        else:
            # Tensor argument — pass host pointer
            arr = arg.data
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Expected numpy array for tensor arg, got {type(arr)}")
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            ptr = arr.ctypes.data
            func_args.append(ptr)
            arg_sizes.append(arr.nbytes)

    return func_args, arg_types, arg_sizes


class L2Runtime:
    """Single-chip runtime. Wraps bindings.py."""

    def __init__(self, platform: str, device: int = 0):
        self._platform = platform
        self._device = device
        self._registry: dict[str, CompiledPackage] = {}
        self._lib_loaded = False
        self._RuntimeClass = None

    def register(self, name: str, *, pkg: CompiledPackage = None,
                 orch: str = "", kernels: list[dict] = None,
                 runtime: str = "tensormap_and_ringbuffer",
                 orch_func: str = "aicpu_orchestration_entry",
                 block_dim: int = 1, aicpu_thread_num: int = 1,
                 orch_thread_num: int = 1,
                 cache_dir: Optional[str] = None,
                 build_dir: Optional[str] = None,
                 extra_include_dirs: Optional[list[str]] = None,
                 **kwargs) -> None:
        """Register a named computation.

        Either pass a pre-compiled CompiledPackage via ``pkg``, or pass
        source paths via ``orch`` + ``kernels`` to compile on the fly.
        """
        if pkg is not None:
            self._registry[name] = pkg
            return

        # Compile from source
        from .compiler import compile as pto_compile

        pkg = pto_compile(
            platform=self._platform,
            runtime_name=runtime,
            orch_source=orch,
            kernel_sources=kernels or [],
            orch_func=orch_func,
            block_dim=block_dim,
            aicpu_thread_num=aicpu_thread_num,
            orch_thread_num=orch_thread_num,
            cache_dir=cache_dir,
            build_dir=build_dir,
            extra_include_dirs=extra_include_dirs,
        )
        self._registry[name] = pkg

    def run(self, name: str, args: list[Arg]) -> None:
        """Execute a registered computation on the device.

        Performs the full L2 lifecycle: init → launch → finalize.
        """
        if name not in self._registry:
            raise KeyError(f"No registered computation '{name}'. "
                           f"Available: {list(self._registry.keys())}")

        pkg = self._registry[name]
        self._ensure_loaded(pkg)

        func_args, arg_types, arg_sizes = _args_to_c_arrays(args)

        from bindings import launch_runtime

        rt = self._RuntimeClass()
        rt.initialize(
            pkg.orch_binary,
            pkg.orch_func,
            func_args,
            arg_types=arg_types,
            arg_sizes=arg_sizes,
            kernel_binaries=pkg.kernel_binaries,
        )
        launch_runtime(
            rt,
            aicpu_thread_num=pkg.aicpu_thread_num,
            block_dim=pkg.block_dim,
            device_id=self._device,
            aicpu_binary=pkg.aicpu_binary,
            aicore_binary=pkg.aicore_binary,
            orch_thread_num=pkg.orch_thread_num,
        )
        rt.finalize()

    def close(self) -> None:
        """Release resources."""
        self._registry.clear()
        self._RuntimeClass = None
        self._lib_loaded = False

    def _ensure_loaded(self, pkg: CompiledPackage) -> None:
        """Load the host binary and set device if not already done."""
        if self._lib_loaded:
            return

        from bindings import bind_host_binary, set_device

        self._RuntimeClass = bind_host_binary(pkg.host_binary)
        set_device(self._device)
        self._lib_loaded = True
