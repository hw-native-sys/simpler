"""
HCCL Python ctypes bindings for multi-card communication setup.

Provides HcclGetRootInfo, HcclCommInitRootInfo, HcclAllocComResourceByTiling, etc.
Requires CANN with libhccl.so and libacl.so.

Usage:
    from hccl_bindings import hccl_get_root_info, hccl_init_comm, HCCL_ROOT_INFO_BYTES
"""

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    c_void_p,
    c_uint32,
    c_int,
    c_char_p,
    Structure,
    create_string_buffer,
)
from pathlib import Path
from typing import Optional, Tuple

# HCCL_ROOT_INFO_BYTES from hccl_types.h (typically 1024)
HCCL_ROOT_INFO_BYTES = 1024

# HCCL result codes
HCCL_SUCCESS = 0

_libacl = None
_libhccl = None


def _load_libs():
    """Load libacl.so and libhccl.so."""
    global _libacl, _libhccl
    if _libhccl is not None:
        return

    # Try common CANN paths
    candidates_acl = [
        os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] + "/libacl.so" if os.environ.get("LD_LIBRARY_PATH") else None,
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libacl.so",
        "libacl.so",
    ]
    candidates_hccl = [
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libhccl.so",
        "libhccl.so",
    ]

    for p in candidates_acl:
        if p and os.path.exists(p) if os.path.isabs(p) else True:
            try:
                _libacl = ctypes.CDLL(p if os.path.isabs(p) else "libacl.so")
                break
            except OSError:
                pass
    if _libacl is None:
        try:
            _libacl = ctypes.CDLL("libacl.so")
        except OSError:
            raise RuntimeError(
                "Cannot load libacl.so. Ensure CANN is installed and LD_LIBRARY_PATH includes Ascend lib."
            )

    for p in candidates_hccl:
        if p and os.path.exists(p) if os.path.isabs(p) else True:
            try:
                _libhccl = ctypes.CDLL(p if os.path.isabs(p) else "libhccl.so")
                break
            except OSError:
                pass
    if _libhccl is None:
        try:
            _libhccl = ctypes.CDLL("libhccl.so")
        except OSError:
            raise RuntimeError(
                "Cannot load libhccl.so. Ensure CANN is installed and LD_LIBRARY_PATH includes Ascend lib."
            )


def hccl_get_root_info(device_id: int) -> bytes:
    """
    Rank 0 calls this to get HcclRootInfo. Must call set_device(device_id) first.

    Returns:
        bytes of length HCCL_ROOT_INFO_BYTES
    """
    _load_libs()
    # aclrtSetDevice first
    aclrtSetDevice = _libacl.aclrtSetDevice
    aclrtSetDevice.argtypes = [c_uint32]
    aclrtSetDevice.restype = c_int
    ret = aclrtSetDevice(device_id)
    if ret != 0:
        raise RuntimeError(f"aclrtSetDevice({device_id}) failed: {ret}")

    # HcclGetRootInfo
    HcclGetRootInfo = _libhccl.HcclGetRootInfo
    HcclGetRootInfo.argtypes = [c_void_p]
    HcclGetRootInfo.restype = c_int  # HcclResult

    buf = create_string_buffer(HCCL_ROOT_INFO_BYTES)
    ret = HcclGetRootInfo(ctypes.cast(buf, c_void_p))
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcclGetRootInfo failed: {ret}")
    return buf.raw[:HCCL_ROOT_INFO_BYTES]


def hccl_init_comm(
    rank_id: int,
    n_ranks: int,
    n_devices: int,
    first_device_id: int,
    root_info: bytes,
) -> Tuple[int, int, int, int]:
    """
    Initialize HCCL comm and alloc resources.

    Args:
        rank_id: This rank's ID
        n_ranks: Total number of ranks
        n_devices: Number of devices
        first_device_id: First device ID
        root_info: bytes from hccl_get_root_info (rank 0)

    Returns:
        (comm, device_ctx_ptr, win_base, stream) - all as int (void* as integer)
    """
    _load_libs()

    device_id = rank_id % n_devices + first_device_id

    # aclrtSetDevice
    aclrtSetDevice = _libacl.aclrtSetDevice
    aclrtSetDevice.argtypes = [c_uint32]
    aclrtSetDevice.restype = c_int
    ret = aclrtSetDevice(device_id)
    if ret != 0:
        raise RuntimeError(f"aclrtSetDevice({device_id}) failed: {ret}")

    # aclrtCreateStream
    aclrtCreateStream = _libacl.aclrtCreateStream
    aclrtCreateStream.argtypes = [POINTER(c_void_p)]
    aclrtCreateStream.restype = c_int
    stream = c_void_p()
    ret = aclrtCreateStream(ctypes.byref(stream))
    if ret != 0:
        raise RuntimeError(f"aclrtCreateStream failed: {ret}")

    # HcclCommInitRootInfo
    HcclCommInitRootInfo = _libhccl.HcclCommInitRootInfo
    HcclCommInitRootInfo.argtypes = [c_uint32, c_void_p, c_uint32, POINTER(c_void_p)]
    HcclCommInitRootInfo.restype = c_int

    comm = c_void_p()
    buf = create_string_buffer(len(root_info))
    buf.raw[: len(root_info)] = root_info
    ret = HcclCommInitRootInfo(
        n_ranks,
        ctypes.cast(buf, c_void_p),
        rank_id,
        ctypes.byref(comm),
    )
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcclCommInitRootInfo failed: {ret}")

    # HcclGetCommName
    HcclGetCommName = _libhccl.HcclGetCommName
    HcclGetCommName.argtypes = [c_void_p, c_char_p]
    HcclGetCommName.restype = c_int
    group = create_string_buffer(128)
    ret = HcclGetCommName(comm, group)
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcclGetCommName failed: {ret}")

    # HcomGetL0TopoTypeEx
    HcomGetL0TopoTypeEx = _libhccl.HcomGetL0TopoTypeEx
    HcomGetL0TopoTypeEx.argtypes = [c_char_p, POINTER(c_uint32), c_uint32]
    HcomGetL0TopoTypeEx.restype = c_int
    topo = c_uint32(0)
    ret = HcomGetL0TopoTypeEx(group.value, ctypes.byref(topo), 0)
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcomGetL0TopoTypeEx failed: {ret}")

    # HcomGetCommHandleByGroup
    HcomGetCommHandleByGroup = _libhccl.HcomGetCommHandleByGroup
    HcomGetCommHandleByGroup.argtypes = [c_char_p, POINTER(c_void_p)]
    HcomGetCommHandleByGroup.restype = c_int
    comm_handle = c_void_p()
    ret = HcomGetCommHandleByGroup(group.value, ctypes.byref(comm_handle))
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcomGetCommHandleByGroup failed: {ret}")

    # Mc2CommConfigV2 tiling structure
    class Mc2InitTilingInner(Structure):
        _fields_ = [
            ("version", c_uint32),
            ("mc2HcommCnt", c_uint32),
            ("offset", c_uint32 * 8),
            ("debugMode", ctypes.c_uint8),
            ("preparePosition", ctypes.c_uint8),
            ("queueNum", ctypes.c_uint16),
            ("commBlockNum", ctypes.c_uint16),
            ("devType", ctypes.c_uint8),
            ("reserved", ctypes.c_uint8 * 17),
        ]

    class Mc2cCTilingInner(Structure):
        _fields_ = [
            ("skipLocalRankCopy", ctypes.c_uint8),
            ("skipBufferWindowCopy", ctypes.c_uint8),
            ("stepSize", ctypes.c_uint8),
            ("version", ctypes.c_uint8),
            ("reserved", ctypes.c_uint8 * 9),
            ("commEngine", ctypes.c_uint8),
            ("srcDataType", ctypes.c_uint8),
            ("dstDataType", ctypes.c_uint8),
            ("groupName", ctypes.c_char * 128),
            ("algConfig", ctypes.c_char * 128),
            ("opType", c_uint32),
            ("reduceType", c_uint32),
        ]

    class Mc2CommConfigV2(Structure):
        _fields_ = [
            ("init", Mc2InitTilingInner),
            ("inner", Mc2cCTilingInner),
        ]

    tiling = Mc2CommConfigV2()
    ctypes.memset(ctypes.byref(tiling), 0, ctypes.sizeof(tiling))
    tiling.init.version = 100
    tiling.init.mc2HcommCnt = 1
    tiling.init.commBlockNum = 48
    tiling.init.devType = 4
    tiling.init.offset[0] = ctypes.sizeof(Mc2InitTilingInner)
    tiling.inner.opType = 18
    tiling.inner.commEngine = 3
    tiling.inner.version = 1
    tiling.inner.groupName = group.value
    tiling.inner.algConfig = b"BatchWrite=level0:fullmesh"

    # HcclAllocComResourceByTiling
    HcclAllocComResourceByTiling = _libhccl.HcclAllocComResourceByTiling
    HcclAllocComResourceByTiling.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
    HcclAllocComResourceByTiling.restype = c_int

    ctx_ptr = c_void_p()
    ret = HcclAllocComResourceByTiling(
        comm_handle,
        stream,
        ctypes.byref(tiling),
        ctypes.byref(ctx_ptr),
    )
    if ret != HCCL_SUCCESS or ctx_ptr.value is None:
        raise RuntimeError(f"HcclAllocComResourceByTiling failed: {ret}")

    # For MESH topology: ctx_ptr is HcclDeviceContext. Read hostCtx to get windowsIn[rank_id]
    # HcclDeviceContext layout: workSpace(8), workSpaceSize(8), rankId(4), rankNum(4), winSize(8), windowsIn[64](8*64)
    HcclDeviceContext_size = 8 + 8 + 4 + 4 + 8 + 64 * 8 + 64 * 8  # windowsOut too
    host_ctx_buf = (ctypes.c_uint8 * HcclDeviceContext_size)()
    aclrtMemcpy = _libacl.aclrtMemcpy
    aclrtMemcpy.argtypes = [c_void_p, ctypes.c_size_t, c_void_p, ctypes.c_size_t, c_int]
    aclrtMemcpy.restype = c_int
    ACL_MEMCPY_DEVICE_TO_HOST = 2
    ret = aclrtMemcpy(
        ctypes.cast(host_ctx_buf, c_void_p),
        len(host_ctx_buf),
        ctx_ptr,
        len(host_ctx_buf),
        ACL_MEMCPY_DEVICE_TO_HOST,
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy D2H failed: {ret}")

    # Parse: windowsIn offset = 8+8+4+4+8 = 32, each entry 8 bytes
    import struct
    win_offset = 32
    win_base = struct.unpack_from("<Q", host_ctx_buf, win_offset + rank_id * 8)[0]

    return (
        comm.value or 0,
        ctypes.cast(ctx_ptr, c_void_p).value or 0,
        win_base,
        stream.value or 0,
    )


def hccl_barrier(comm_handle: int, stream_handle: int) -> None:
    """HcclBarrier for sync across ranks."""
    _load_libs()
    HcclBarrier = _libhccl.HcclBarrier
    HcclBarrier.argtypes = [c_void_p, c_void_p]
    HcclBarrier.restype = c_int
    ret = HcclBarrier(ctypes.c_void_p(comm_handle), ctypes.c_void_p(stream_handle))
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcclBarrier failed: {ret}")
    aclrtSynchronizeStream = _libacl.aclrtSynchronizeStream
    aclrtSynchronizeStream.argtypes = [c_void_p]
    aclrtSynchronizeStream.restype = c_int
    ret = aclrtSynchronizeStream(ctypes.c_void_p(stream_handle))
    if ret != 0:
        raise RuntimeError(f"aclrtSynchronizeStream failed: {ret}")
