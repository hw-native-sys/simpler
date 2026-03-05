"""
HCCL Python bindings for multi-card communication setup.

Prefer C++ helper lib (libhccl_helper.so) — same link as pto-comm-isa (ascendcl, hcomm, runtime).
Build: cd examples/scripts/hccl_helper && mkdir build && cd build && cmake .. && make
Then run with CANN env: source .../set_env.sh

Usage:
    from hccl_bindings import hccl_get_root_info, hccl_init_comm, hccl_barrier, HCCL_ROOT_INFO_BYTES
"""

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    c_void_p,
    c_uint32,
    c_int,
    c_uint64,
    c_char_p,
    Structure,
    create_string_buffer,
)
from pathlib import Path
from typing import Optional, Tuple

# Set after loading libhccl_helper
HCCL_ROOT_INFO_BYTES = 1024

_lib_helper = None


def _find_helper_so() -> Optional[Path]:
    """Locate libhccl_helper.so next to this script or in hccl_helper/build."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "hccl_helper" / "build" / "libhccl_helper.so",
        script_dir / "build" / "libhccl_helper.so",
        script_dir / "libhccl_helper.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_helper():
    """Load libhccl_helper.so (C++ helper, same link as pto-comm-isa)."""
    global _lib_helper, HCCL_ROOT_INFO_BYTES
    if _lib_helper is not None:
        return

    path = _find_helper_so()
    if path is None:
        raise RuntimeError(
            "libhccl_helper.so not found. Build it with CANN env set:\n"
            "  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash\n"
            "  cd examples/scripts/hccl_helper && mkdir build && cd build\n"
            "  cmake .. && make\n"
            "Then run your script with the same CANN env (source setenv.bash)."
        )
    try:
        _lib_helper = ctypes.CDLL(str(path))
    except OSError as e:
        raise RuntimeError(
            f"Failed to load {path}: {e}\n"
            "Ensure CANN env is set (source .../setenv.bash) so dependencies (ascendcl, hcomm, runtime) can be found."
        ) from e

    # C API
    _lib_helper.hccl_helper_root_info_bytes.restype = c_uint32
    HCCL_ROOT_INFO_BYTES = _lib_helper.hccl_helper_root_info_bytes()

    _lib_helper.hccl_helper_get_root_info.argtypes = [c_int, c_void_p, c_uint32]
    _lib_helper.hccl_helper_get_root_info.restype = c_int

    _lib_helper.hccl_helper_init_comm.argtypes = [
        c_int, c_int, c_int, c_int,  # rank_id, n_ranks, n_devices, first_device_id
        c_void_p, c_uint32,           # root_info, root_info_len
        POINTER(c_void_p), POINTER(c_void_p), POINTER(c_uint64), POINTER(c_void_p),
    ]
    _lib_helper.hccl_helper_init_comm.restype = c_int

    _lib_helper.hccl_helper_barrier.argtypes = [c_void_p, c_void_p]
    _lib_helper.hccl_helper_barrier.restype = c_int


def hccl_get_root_info(device_id: int) -> bytes:
    """
    Rank 0: get HcclRootInfo (C++ helper sets device and calls HcclGetRootInfo).

    Returns:
        bytes of length HCCL_ROOT_INFO_BYTES
    """
    _load_helper()
    buf = create_string_buffer(HCCL_ROOT_INFO_BYTES)
    ret = _lib_helper.hccl_helper_get_root_info(device_id, buf, HCCL_ROOT_INFO_BYTES)
    if ret != 0:
        raise RuntimeError(f"hccl_helper_get_root_info failed: {ret}")
    return buf.raw[:HCCL_ROOT_INFO_BYTES]


def hccl_init_comm(
    rank_id: int,
    n_ranks: int,
    n_devices: int,
    first_device_id: int,
    root_info: bytes,
) -> Tuple[int, int, int, int]:
    """
    All ranks: init HCCL comm (same link as pto-comm-isa).

    Returns:
        (comm, device_ctx_ptr, win_base, stream) as integers (void* as int).
    """
    _load_helper()
    if len(root_info) < HCCL_ROOT_INFO_BYTES:
        raise ValueError(f"root_info must be at least {HCCL_ROOT_INFO_BYTES} bytes")

    comm = c_void_p()
    ctx_ptr = c_void_p()
    win_base = c_uint64()
    stream = c_void_p()

    buf = create_string_buffer(len(root_info))
    # copy bytes into mutable buffer
    ctypes.memmove(buf, root_info, len(root_info))

    ret = _lib_helper.hccl_helper_init_comm(
        rank_id,
        n_ranks,
        n_devices,
        first_device_id,
        buf,
        len(root_info),
        ctypes.byref(comm),
        ctypes.byref(ctx_ptr),
        ctypes.byref(win_base),
        ctypes.byref(stream),
    )
    if ret != 0:
        raise RuntimeError(f"hccl_helper_init_comm failed: {ret}")

    return (
        comm.value or 0,
        ctx_ptr.value or 0,
        win_base.value,
        stream.value or 0,
    )


def hccl_barrier(comm_handle: int, stream_handle: int) -> None:
    """HcclBarrier + stream sync (C++ helper)."""
    _load_helper()
    ret = _lib_helper.hccl_helper_barrier(
        ctypes.c_void_p(comm_handle),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(f"hccl_helper_barrier failed: {ret}")
