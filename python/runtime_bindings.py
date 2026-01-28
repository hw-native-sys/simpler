"""

PTO Runtime ctypes Bindings

Provides a Pythonic interface to the PTO runtime via ctypes.
Users must provide a pre-compiled libpto_runtime.so (built via binary_compiler.py).

Usage:
    from runtime_bindings import load_runtime, register_kernel, launch_runtime

    Runtime = load_runtime("/path/to/libpto_runtime.so")

    runtime = Runtime()
    runtime.initialize()

    register_kernel(0, kernel_add)
    register_kernel(1, kernel_add_scalar)
    register_kernel(2, kernel_mul)

    launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
                 device_id=0, aicpu_binary=aicpu_bytes,
                 aicore_binary=aicore_bytes)

    runtime.finalize()
"""


from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    c_int,
    c_void_p,
    c_uint8,
    c_uint64,
    c_size_t,
)
from pathlib import Path
from typing import Union, List, Callable, Optional
import ctypes
import tempfile


# Module-level library reference
_lib = None

# Orchestration function type: (runtime, args, arg_count) -> int
OrchestrationFunc = CFUNCTYPE(c_int, c_void_p, POINTER(c_uint64), c_int)


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

        # GetRuntimeSize - returns sizeof(Runtime) for user allocation
        self.lib.GetRuntimeSize.argtypes = []
        self.lib.GetRuntimeSize.restype = c_size_t

        # InitRuntime - placement new + build runtime with orchestration
        self.lib.InitRuntime.argtypes = [
            c_void_p,               # runtime
            OrchestrationFunc,      # orch_func
            POINTER(c_uint64),      # func_args
            c_int,                  # func_args_count
        ]
        self.lib.InitRuntime.restype = c_int

        # launch_runtime - device init + execute runtime
        self.lib.launch_runtime.argtypes = [
            c_void_p,           # runtime
            c_int,              # aicpu_thread_num
            c_int,              # block_dim
            c_int,              # device_id
            POINTER(c_uint8),   # aicpu_binary
            c_size_t,           # aicpu_size
            POINTER(c_uint8),   # aicore_binary
            c_size_t,           # aicore_size
        ]
        self.lib.launch_runtime.restype = c_int

        # FinalizeRuntime - validate + cleanup
        self.lib.FinalizeRuntime.argtypes = [c_void_p]
        self.lib.FinalizeRuntime.restype = c_int

        # RegisterKernel - register kernel binary for func_id
        self.lib.RegisterKernel.argtypes = [c_int, POINTER(c_uint8), c_size_t]
        self.lib.RegisterKernel.restype = c_int

        # set_device - set device and create streams
        self.lib.set_device.argtypes = [c_int]
        self.lib.set_device.restype = c_int


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
        # Allocate buffer of size GetRuntimeSize() for placement new
        size = lib.GetRuntimeSize()
        self._buffer = ctypes.create_string_buffer(size)
        self._handle = ctypes.cast(self._buffer, c_void_p)

    def initialize(
        self,
        orch_func: Callable,
        func_args: Optional[List[int]] = None
    ) -> None:
        """

        Initialize the runtime structure with dynamic orchestration.

        Calls InitRuntime() in C++ which calls the orchestration function.
        The orchestration function is responsible for:
        1. Allocating device memory
        2. Copying data to device
        3. Building the task graph
        4. Recording tensor pairs for copy-back

        Args:
            orch_func: Orchestration function to build task graph.
                       Signature: int func(runtime, args, arg_count)
            func_args: Arguments for orchestration (host pointers, sizes, etc.)

        Raises:
            RuntimeError: If initialization fails
        """

        func_args = func_args or []
        func_args_count = len(func_args)

        # Convert func_args to ctypes array
        if func_args_count > 0:
            func_args_array = (c_uint64 * func_args_count)(*func_args)
        else:
            func_args_array = None

        # Store orch_func reference to prevent garbage collection
        self._orch_func = OrchestrationFunc(orch_func)

        rc = self.lib.InitRuntime(
            self._handle,
            self._orch_func,
            func_args_array,
            func_args_count
        )
        if rc != 0:
            raise RuntimeError(f"InitRuntime failed: {rc}")

    def finalize(self) -> None:
        """

        Finalize and cleanup the runtime.

        Calls FinalizeRuntime() in C++ which validates computation results,
        frees device tensors, and calls the Runtime destructor.

        Raises:
            RuntimeError: If finalization fails
        """

        rc = self.lib.FinalizeRuntime(self._handle)
        if rc != 0:
            raise RuntimeError(f"FinalizeRuntime failed: {rc}")

    def __del__(self):
        """Clean up runtime resources."""

        # Runtime destructor is called by finalize(), buffer freed by Python GC
        pass


# ============================================================================
# Module-level Functions
# ============================================================================

def register_kernel(func_id: int, binary_data: bytes) -> None:
    """

    Register a kernel binary for a func_id.

    Receives pre-extracted .text section binary data,
    allocates device GM memory, copies the binary to device,
    and stores the GM address for later use by launch_runtime().

    Args:
        func_id: Function identifier (0, 1, 2, ...)
        binary_data: Kernel .text section binary data

    Raises:
        RuntimeError: If not initialized or registration fails
        ValueError: If binary_data is empty
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call load_runtime() first.")

    if not binary_data:
        raise ValueError("binary_data cannot be empty")

    # Convert bytes to ctypes array
    bin_array = (c_uint8 * len(binary_data)).from_buffer_copy(binary_data)
    rc = _lib.RegisterKernel(func_id, bin_array, len(binary_data))
    if rc != 0:
        raise RuntimeError(f"RegisterKernel failed: {rc}")


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
        raise RuntimeError("Runtime not loaded. Call load_runtime() first.")

    rc = _lib.set_device(device_id)
    if rc != 0:
        raise RuntimeError(f"set_device failed: {rc}")


def launch_runtime(
    runtime: "Runtime",
    aicpu_thread_num: int,
    block_dim: int,
    device_id: int,
    aicpu_binary: bytes,
    aicore_binary: bytes,
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

    Raises:
        RuntimeError: If not initialized or execution fails
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call load_runtime() first.")

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
    )
    if rc != 0:
        raise RuntimeError(f"launch_runtime failed: {rc}")


# ============================================================================
# Public API
# ============================================================================

def load_runtime(lib_path: Union[str, Path, bytes]) -> type:
    """

    Load the PTO runtime library and return Runtime class.

    Args:
        lib_path: Path to libpto_runtime.so (str/Path), or compiled binary data (bytes)

    Returns:
        Runtime class initialized with the library

    Example:
        from runtime_bindings import load_runtime, register_kernel, launch_runtime

        Runtime = load_runtime("/path/to/libpto_runtime.so")

        runtime = Runtime()
        runtime.initialize()

        register_kernel(0, kernel_add)
        register_kernel(1, kernel_add_scalar)
        register_kernel(2, kernel_mul)

        launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
                     device_id=0, aicpu_binary=aicpu_bytes,
                     aicore_binary=aicore_bytes)

        runtime.finalize()
    """

    global _lib

    # If bytes are provided, write to temporary file
    if isinstance(lib_path, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.so') as f:
            f.write(lib_path)
            lib_path = f.name

    loader = RuntimeLibraryLoader(lib_path)
    _lib = loader.lib

    # Create wrapper class with the loaded library
    class _Runtime(Runtime):
        def __init__(self):
            super().__init__(_lib)

    return _Runtime
