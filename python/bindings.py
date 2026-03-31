# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLW0602, PLW0603, PLC0415
"""

PTO Runtime ctypes Bindings

Provides a Pythonic interface to the PTO runtime via ctypes.
Users must provide a pre-compiled libpto_runtime.so (built via runtime_compiler.py).

Usage:
    from bindings import bind_host_binary, launch_runtime, set_device

    Runtime = bind_host_binary("/path/to/libpto_runtime.so")
    set_device(0)  # Must be called before initialize

    runtime = Runtime()
    runtime.initialize(chip_callable, orch_args)

    launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
                 device_id=0, aicpu_binary=aicpu_bytes,
                 aicore_binary=aicore_bytes)

    runtime.finalize()
"""

import ctypes
import tempfile
from ctypes import (
    CDLL,
    POINTER,
    c_int,
    c_size_t,
    c_uint8,
    c_void_p,
)
from pathlib import Path
from typing import Optional, Union

# Module-level library reference
_lib = None


# ============================================================================
# ToolchainType enum (defined in toolchain.py, must match compile_strategy.h)
# ============================================================================
from toolchain import ToolchainType  # noqa: E402

# ============================================================================
# Runtime Library Loader
# ============================================================================


class RuntimeLibraryLoader:
    """Loads and manages the PTO runtime C API library."""

    def __init__(self, lib_path: Union[str, Path]):
        """

        Load the PTO runtime library.

        Args:
            lib_path: Path to libpto_runtime.so

        Raises:
            FileNotFoundError: If library file not found
            OSError: If library cannot be loaded
        """

        lib_path = Path(lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib_path = lib_path
        self.lib = CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        self._setup_functions()

    def _setup_functions(self):
        """Set up ctypes function signatures."""

        # get_runtime_size - returns sizeof(Runtime) for user allocation
        self.lib.get_runtime_size.argtypes = []
        self.lib.get_runtime_size.restype = c_size_t

        # init_runtime - placement new + unpack ChipCallable + register kernels + load SO
        self.lib.init_runtime.argtypes = [
            c_void_p,  # runtime
            c_void_p,  # callable (const ChipCallable*)
            c_void_p,  # orch_args (const ChipStorageTaskArgs*)
        ]
        self.lib.init_runtime.restype = c_int

        # launch_runtime - device init + execute runtime
        self.lib.launch_runtime.argtypes = [
            c_void_p,  # runtime
            c_int,  # aicpu_thread_num
            c_int,  # block_dim
            c_int,  # device_id
            POINTER(c_uint8),  # aicpu_binary
            c_size_t,  # aicpu_size
            POINTER(c_uint8),  # aicore_binary
            c_size_t,  # aicore_size
            c_int,  # orch_thread_num
        ]
        self.lib.launch_runtime.restype = c_int

        # finalize_runtime - validate + cleanup
        self.lib.finalize_runtime.argtypes = [c_void_p]
        self.lib.finalize_runtime.restype = c_int

        # Note: register_kernel has been internalized into init_runtime
        # Kernel binaries are now passed directly to init_runtime()

        # set_device - set device and create streams
        self.lib.set_device.argtypes = [c_int]
        self.lib.set_device.restype = c_int

        # device_malloc - allocate device memory
        self.lib.device_malloc.argtypes = [c_size_t]
        self.lib.device_malloc.restype = c_void_p

        # device_free - free device memory
        self.lib.device_free.argtypes = [c_void_p]
        self.lib.device_free.restype = None

        # copy_to_device - copy data from host to device
        self.lib.copy_to_device.argtypes = [c_void_p, c_void_p, c_size_t]
        self.lib.copy_to_device.restype = c_int

        # copy_from_device - copy data from device to host
        self.lib.copy_from_device.argtypes = [c_void_p, c_void_p, c_size_t]
        self.lib.copy_from_device.restype = c_int

        # record_tensor_pair - record tensor pair for copy-back
        self.lib.record_tensor_pair.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t]
        self.lib.record_tensor_pair.restype = None

        # get_incore_compiler - get toolchain for incore kernel compilation
        self.lib.get_incore_compiler.argtypes = []
        self.lib.get_incore_compiler.restype = c_int

        # get_orchestration_compiler - get toolchain for orchestration compilation
        self.lib.get_orchestration_compiler.argtypes = []
        self.lib.get_orchestration_compiler.restype = c_int

        # enable_runtime_profiling - enable profiling for swimlane
        self.lib.enable_runtime_profiling.argtypes = [c_void_p, c_int]
        self.lib.enable_runtime_profiling.restype = c_int


# ============================================================================
# Python Wrapper Classes
# ============================================================================


class Runtime:
    """

    Task dependency runtime.

    Python wrapper around the C Runtime API.
    User allocates memory via ctypes buffer, C++ uses placement new.
    """

    def __init__(self, lib: CDLL):
        """

        Create a new runtime handle.

        Args:
            lib: Loaded ctypes library (RuntimeLibraryLoader.lib)
        """

        self.lib = lib
        # Allocate buffer of size get_runtime_size() for placement new
        size = lib.get_runtime_size()
        self._buffer = ctypes.create_string_buffer(size)
        self._handle = ctypes.cast(self._buffer, c_void_p)

    def initialize(
        self,
        chip_callable,
        orch_args: Optional[list] = None,
    ) -> None:
        """

        Initialize the runtime structure with a ChipCallable.

        Calls init_runtime() in C++ which unpacks the ChipCallable to:
        1. Register kernel binaries (children) to device memory
        2. Load the orchestration SO, resolve the function, and call it

        Args:
            chip_callable: ChipCallable containing orch binary, func_name, + kernel CoreCallables
            orch_args: ChipStorageTaskArgs with orchestration arguments

        Raises:
            RuntimeError: If initialization fails
        """

        orch_args = orch_args or []

        # Convert orch_args to ctypes-compatible form for the C API.
        # ChipStorageTaskArgs (nanobind): pass its C++ object address directly.
        from _task_interface import (  # pyright: ignore[reportMissingImports]
            ChipStorageTaskArgs as _NbChipStorageTaskArgs,  # noqa: PLC0415
        )

        if isinstance(orch_args, _NbChipStorageTaskArgs):
            orch_args_ptr = c_void_p(orch_args.__ptr__())  # pyright: ignore[reportAttributeAccessIssue]
        elif len(orch_args) == 0:
            orch_args_ptr = None
        else:
            raise TypeError(f"orch_args must be a ChipStorageTaskArgs, got {type(orch_args)}")

        # Pass ChipCallable pointer to C API
        self._chip_callable = chip_callable  # prevent GC
        callable_ptr = c_void_p(chip_callable.buffer_ptr())  # pyright: ignore[reportAttributeAccessIssue]

        rc = self.lib.init_runtime(
            self._handle,
            callable_ptr,
            orch_args_ptr,
        )
        if rc != 0:
            raise RuntimeError(f"init_runtime failed: {rc}")

    def finalize(self) -> None:
        """

        Finalize and cleanup the runtime.

        Calls finalize_runtime() in C++ which validates computation results,
        frees device tensors, and calls the Runtime destructor.

        Raises:
            RuntimeError: If finalization fails
        """

        rc = self.lib.finalize_runtime(self._handle)
        if rc != 0:
            raise RuntimeError(f"finalize_runtime failed: {rc}")

    def enable_profiling(self, enabled: bool = True) -> None:
        """
        Enable or disable performance profiling for swimlane visualization.

        Must be called before initialize() to enable profiling.
        When enabled, the runtime records task execution timestamps and
        generates swim_time.json after finalize().

        Args:
            enabled: True to enable profiling, False to disable

        Raises:
            RuntimeError: If enable operation fails
        """
        rc = self.lib.enable_runtime_profiling(self._handle, 1 if enabled else 0)
        if rc != 0:
            raise RuntimeError(f"enable_runtime_profiling failed: {rc}")

    def __del__(self):
        """Clean up runtime resources."""

        # Runtime destructor is called by finalize(), buffer freed by Python GC
        pass


# ============================================================================
# Module-level Functions
# ============================================================================

"""
Note: register_kernel() has been internalized into runtime.initialize().
Kernel binaries are now passed directly to initialize() via the kernel_binaries parameter.
"""


def set_device(device_id: int) -> None:
    """

    Set device and create streams for memory operations.

    Must be called before runtime.initialize() to enable device tensor allocation.
    Only performs minimal initialization:
    - rtSetDevice(device_id)
    - Create AICPU and AICore streams

    Binary loading happens later in launch_runtime().

    Args:
        device_id: Device ID (0-15)

    Raises:
        RuntimeError: If not loaded or device setup fails
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    rc = _lib.set_device(device_id)
    if rc != 0:
        raise RuntimeError(f"set_device failed: {rc}")


def device_malloc(size: int) -> Optional[int]:
    """
    Allocate device memory.

    Args:
        size: Size in bytes to allocate

    Returns:
        Device pointer as integer, or None on failure

    Raises:
        RuntimeError: If not loaded
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    ptr = _lib.device_malloc(size)
    return ptr if ptr else None


def device_free(dev_ptr: int) -> None:
    """
    Free device memory.

    Args:
        dev_ptr: Device pointer to free

    Raises:
        RuntimeError: If not loaded
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    _lib.device_free(ctypes.c_void_p(dev_ptr))


def copy_to_device(dev_ptr: int, host_ptr: int, size: int) -> None:
    """
    Copy data from host to device.

    Args:
        dev_ptr: Device destination pointer
        host_ptr: Host source pointer
        size: Size in bytes to copy

    Raises:
        RuntimeError: If not loaded or copy fails
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    rc = _lib.copy_to_device(ctypes.c_void_p(dev_ptr), ctypes.c_void_p(host_ptr), size)
    if rc != 0:
        raise RuntimeError(f"copy_to_device failed: {rc}")


def copy_from_device(host_ptr: int, dev_ptr: int, size: int) -> None:
    """
    Copy data from device to host.

    Args:
        host_ptr: Host destination pointer
        dev_ptr: Device source pointer
        size: Size in bytes to copy

    Raises:
        RuntimeError: If not loaded or copy fails
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    rc = _lib.copy_from_device(ctypes.c_void_p(host_ptr), ctypes.c_void_p(dev_ptr), size)
    if rc != 0:
        raise RuntimeError(f"copy_from_device failed: {rc}")


def launch_runtime(
    runtime: "Runtime",
    aicpu_thread_num: int,
    block_dim: int,
    device_id: int,
    aicpu_binary: bytes,
    aicore_binary: bytes,
    orch_thread_num: int = 1,
) -> None:
    """

    Execute a runtime on the device.

    Initializes DeviceRunner singleton (if first call), copies runtime to device,
    launches kernels, synchronizes, and copies runtime back from device.

    Args:
        runtime: Runtime to execute (must have been initialized via runtime.initialize())
        aicpu_thread_num: Number of AICPU scheduler threads
        block_dim: Number of blocks (1 block = 1 AIC + 2 AIV)
        device_id: Device ID (0-15)
        aicpu_binary: Binary data of AICPU shared object
        aicore_binary: Binary data of AICore kernel
        orch_thread_num: Number of orchestrator threads (default 1)

    Raises:
        RuntimeError: If not initialized or execution fails
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    # Convert bytes to ctypes arrays
    aicpu_array = (c_uint8 * len(aicpu_binary)).from_buffer_copy(aicpu_binary)
    aicore_array = (c_uint8 * len(aicore_binary)).from_buffer_copy(aicore_binary)

    rc = _lib.launch_runtime(
        runtime._handle,
        aicpu_thread_num,
        block_dim,
        device_id,
        aicpu_array,
        len(aicpu_binary),
        aicore_array,
        len(aicore_binary),
        orch_thread_num,
    )
    if rc != 0:
        raise RuntimeError(f"launch_runtime failed: {rc}")


# ============================================================================
# Compile Strategy Functions
# ============================================================================


def get_incore_compiler() -> ToolchainType:
    """
    Get the toolchain for incore kernel compilation.

    Queries the loaded C++ library to determine which compiler to use,
    based on the current platform and runtime combination.

    Returns:
        ToolchainType indicating the compiler to use

    Raises:
        RuntimeError: If library not loaded
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    return ToolchainType(_lib.get_incore_compiler())


def get_orchestration_compiler() -> ToolchainType:
    """
    Get the toolchain for orchestration function compilation.

    Queries the loaded C++ library to determine which compiler to use,
    based on the current platform and runtime combination.

    Returns:
        ToolchainType indicating the compiler to use

    Raises:
        RuntimeError: If library not loaded
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    return ToolchainType(_lib.get_orchestration_compiler())


# ============================================================================
# Public API
# ============================================================================


def bind_host_binary(lib_path: Union[str, Path, bytes]) -> type:
    """

    Load the PTO runtime library and return Runtime class.

    Args:
        lib_path: Path to libpto_runtime.so (str/Path), or compiled binary data (bytes)

    Returns:
        Runtime class initialized with the library

    Example:
        from bindings import bind_host_binary, launch_runtime, set_device

        Runtime = bind_host_binary("/path/to/libpto_runtime.so")
        set_device(0)

        runtime = Runtime()
        runtime.initialize(chip_callable, orch_args)

        launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
                     device_id=0, aicpu_binary=aicpu_bytes,
                     aicore_binary=aicore_bytes)

        runtime.finalize()
    """

    global _lib

    # If bytes are provided, write to temporary file
    if isinstance(lib_path, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".so") as f:
            f.write(lib_path)
            lib_path = f.name

    loader = RuntimeLibraryLoader(lib_path)
    _lib = loader.lib

    # Create wrapper class with the loaded library
    class _Runtime(Runtime):
        def __init__(self):
            super().__init__(_lib)  # pyright: ignore[reportArgumentType]

    return _Runtime
