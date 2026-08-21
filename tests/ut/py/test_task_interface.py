# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Tests for the _task_interface nanobind extension and task_interface wrapper."""

import ctypes
import gc
import itertools
import struct
import weakref
from multiprocessing.shared_memory import SharedMemory

import pytest
import simpler.task_interface as task_interface_module
from _task_interface import (  # pyright: ignore[reportMissingImports]
    MAX_TENSOR_DIMS,
    ArgDirection,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    TaskState,
    TensorArgType,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
)
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Tensor,
    create_host_shared_buffer,
    mint_owner_instance_id,
    remote_sidecar_tensor,
    wrap_device_malloc,
    wrap_fork_inherited,
)
from simpler.task_interface import (
    RemoteAddressSpace,
    RemoteBufferExport,
    RemoteBufferHandle,
    RemoteTensorRef,
    _remote_sidecar_for,
)

_REF_BID = itertools.count(1)


def _dev_ref(addr, shapes, dtype, tag=None):
    """A DEVICE_MALLOC ``Tensor`` at ``addr`` for wire-``TaskArgs`` container tests.

    Each call mints a fresh identity, so refs added to one TaskArgs stay distinct. The backend body
    carries ``addr`` (u64 LE), the container-test counterpart of the old ``ChipTensor.make(addr, ...)``.
    """
    nbytes = get_element_size(dtype)
    for s in shapes:
        nbytes *= int(s)
    return wrap_device_malloc(addr, nbytes, mint_owner_instance_id(), next(_REF_BID), "L2").tensor(
        tuple(shapes), int(dtype.value)
    )


def _ref_addr(ref: Tensor) -> int:
    """The device pointer carried in a DEVICE_MALLOC ref's backend body."""
    return int.from_bytes(bytes(ref.buffer.body)[:8], "little")


def _remote_arg_tensor(shapes, nbytes, owner_worker_id=2, buffer_id=9, generation=1, byte_offset=0):
    """The per-argument record a remote L3 TASK carries: the submitter's REMOTE_SIDECAR placeholder."""
    return remote_sidecar_tensor(
        shapes=tuple(shapes),
        dtype=int(DataType.UINT8.value),
        nbytes=int(nbytes),
        owner_worker_id=owner_worker_id,
        buffer_id=buffer_id,
        generation=generation,
        address_space=AddressSpace.HOST,
        byte_offset=int(byte_offset),
    )


def _session_buffer_minter():
    """A stand-in for the remote runner's session buffer minter: one nonce, a fresh buffer_id each."""
    owner_instance_id = mint_owner_instance_id()
    buffer_ids = itertools.count(1)

    def mint(nbytes: int):
        return create_host_shared_buffer(int(nbytes), owner_instance_id, buffer_id=next(buffer_ids))

    return mint


# ============================================================================
# DataType enum
# ============================================================================


class TestDataType:
    def test_enum_values_exist(self):
        assert DataType.FLOAT32 is not None
        assert DataType.FLOAT16 is not None
        assert DataType.INT32 is not None
        assert DataType.INT16 is not None
        assert DataType.INT8 is not None
        assert DataType.UINT8 is not None
        assert DataType.BFLOAT16 is not None
        assert DataType.INT64 is not None
        assert DataType.UINT64 is not None
        assert DataType.UINT16 is not None
        assert DataType.UINT32 is not None
        # A5-only MX block-scale dtypes (no FP8E5M2 — MX uses E4M3FN).
        assert DataType.FP8E4M3FN is not None  # A5 only
        assert DataType.FP8E8M0 is not None  # A5 only
        assert DataType.FP4E2M1 is not None  # A5 only
        assert not hasattr(DataType, "FP8E5M2")

    def test_enum_int_values(self):
        assert DataType.FLOAT32.value == 0
        assert DataType.FLOAT16.value == 1
        assert DataType.INT32.value == 2
        assert DataType.INT16.value == 3
        assert DataType.INT8.value == 4
        assert DataType.UINT8.value == 5
        assert DataType.BFLOAT16.value == 6
        assert DataType.INT64.value == 7
        assert DataType.UINT64.value == 8
        assert DataType.UINT16.value == 9
        assert DataType.UINT32.value == 10
        assert DataType.FP8E4M3FN.value == 12  # A5 only (11 is BOOL)
        assert DataType.FP8E8M0.value == 13  # A5 only
        assert DataType.FP4E2M1.value == 14  # A5 only


class TestTaskState:
    def test_failed_state_is_bound(self):
        assert TaskState.FAILED is not None
        assert TaskState.FAILED.value == 5
        assert TaskState.CONSUMED.value == 6


class TestGetElementSize:
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (DataType.FLOAT32, 4),
            (DataType.FLOAT16, 2),
            (DataType.INT32, 4),
            (DataType.INT16, 2),
            (DataType.INT8, 1),
            (DataType.UINT8, 1),
            (DataType.BFLOAT16, 2),
            (DataType.INT64, 8),
            (DataType.UINT64, 8),
            (DataType.UINT16, 2),
            (DataType.UINT32, 4),
            (DataType.FP8E4M3FN, 1),  # A5 only
            (DataType.FP8E8M0, 1),  # A5 only
            (DataType.FP4E2M1, 1),  # A5 only
        ],
    )
    def test_element_sizes(self, dtype, expected):
        assert get_element_size(dtype) == expected


class TestGetDtypeName:
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (DataType.FLOAT32, "FLOAT32"),
            (DataType.FLOAT16, "FLOAT16"),
            (DataType.INT32, "INT32"),
            (DataType.INT16, "INT16"),
            (DataType.INT8, "INT8"),
            (DataType.UINT8, "UINT8"),
            (DataType.BFLOAT16, "BFLOAT16"),
            (DataType.INT64, "INT64"),
            (DataType.UINT64, "UINT64"),
            (DataType.UINT16, "UINT16"),
            (DataType.UINT32, "UINT32"),
            (DataType.FP8E4M3FN, "FP8E4M3FN"),  # A5 only
            (DataType.FP8E8M0, "FP8E8M0"),  # A5 only
            (DataType.FP4E2M1, "FP4E2M1"),  # A5 only
        ],
    )
    def test_dtype_names(self, dtype, expected):
        assert get_dtype_name(dtype) == expected


# ============================================================================
# torch_interop (canonical torch-aware helpers) + scalar_to_uint64
# ============================================================================


class TestTorchInterop:
    def test_torch_dtype_to_datatype(self):
        import torch  # pyright: ignore[reportMissingImports]

        from simpler_setup.torch_interop import torch_dtype_to_datatype

        assert torch_dtype_to_datatype(torch.float32) == DataType.FLOAT32
        assert torch_dtype_to_datatype(torch.int8) == DataType.INT8
        assert torch_dtype_to_datatype(torch.bfloat16) == DataType.BFLOAT16

    def test_torch_dtype_uint32(self):
        import torch  # pyright: ignore[reportMissingImports]

        from simpler_setup.torch_interop import torch_dtype_to_datatype

        assert torch_dtype_to_datatype(torch.uint16) == DataType.UINT16
        assert torch_dtype_to_datatype(torch.uint32) == DataType.UINT32

    def test_torch_dtype_unsupported(self):
        import torch  # pyright: ignore[reportMissingImports]

        from simpler_setup.torch_interop import torch_dtype_to_datatype

        with pytest.raises(KeyError):
            torch_dtype_to_datatype(torch.complex64)

    def test_make_chip_tensor_arg(self):
        import torch  # pyright: ignore[reportMissingImports]

        from simpler_setup.torch_interop import make_chip_tensor_arg

        t = torch.zeros(4, 8, dtype=torch.float32)
        arg = make_chip_tensor_arg(t)
        assert isinstance(arg, ChipTensor)
        assert arg.data == t.data_ptr()
        assert arg.shapes == (4, 8)
        assert arg.dtype == DataType.FLOAT32
        assert arg.nbytes() == 4 * 8 * 4

    def test_torch_dtype_fp8_fp4_and_make_chip_tensor_arg(self):
        import torch  # pyright: ignore[reportMissingImports]

        from simpler_setup.torch_interop import make_chip_tensor_arg, torch_dtype_to_datatype

        # A5-only MX dtypes (no float8_e5m2). float8_e4m3fn needs torch >= 2.1;
        # float8_e8m0fnu >= 2.7; float4_e2m1fn_x2 (MXFP4 packed) needs a recent
        # torch too. Collect available attrs; skip the whole test if none exist
        # so a silent empty pass cannot hide missing coverage.
        cases = [
            ("float8_e4m3fn", DataType.FP8E4M3FN),  # A5 only
            ("float8_e8m0fnu", DataType.FP8E8M0),  # A5 only
            ("float4_e2m1fn_x2", DataType.FP4E2M1),  # A5 only
        ]
        available = [
            (attr, expected, getattr(torch, attr)) for attr, expected in cases if getattr(torch, attr, None) is not None
        ]
        if not available:
            pytest.skip(
                "torch lacks float8_e4m3fn / float8_e8m0fnu / float4_e2m1fn_x2 "
                "(need a recent torch, ideally >= 2.7 for E8M0/FP4)"
            )
        for attr, expected, torch_dt in available:
            assert torch_dtype_to_datatype(torch_dt) == expected
            # Regression: make_chip_tensor_arg used to raise
            # "Unsupported tensor dtype for Tensor: torch.<attr>".
            t = torch.zeros(2, 4, dtype=torch_dt)
            arg = make_chip_tensor_arg(t)
            assert arg.dtype == expected
            # Each of these dtypes is 1 byte per torch element (FP4 packs 2
            # E2M1 values per byte), so a [2, 4] tensor is 8 bytes.
            assert arg.nbytes() == 2 * 4


class TestScalarToUint64:
    def test_scalar_to_uint64_int(self):
        from simpler.task_interface import scalar_to_uint64

        assert scalar_to_uint64(999) == 999

    def test_scalar_to_uint64_ctypes(self):
        from simpler.task_interface import scalar_to_uint64

        assert scalar_to_uint64(ctypes.c_int64(42)) == 42

    def test_scalar_to_uint64_float_ctypes(self):
        from simpler.task_interface import scalar_to_uint64

        bits = scalar_to_uint64(ctypes.c_float(1.5))
        expected_bits = struct.unpack("I", struct.pack("f", 1.5))[0]
        assert bits == expected_bits


# ============================================================================
# ChipTensor
# ============================================================================


class TestTensor:
    def test_default_constructor(self):
        arg = ChipTensor()
        assert arg is not None

    def test_max_dims_constant(self):
        assert MAX_TENSOR_DIMS == 5

    def test_make(self):
        arg = ChipTensor.make(0xDEAD, (4, 8), DataType.FLOAT32)
        assert arg.data == 0xDEAD
        assert arg.shapes == (4, 8)
        assert arg.ndims == 2
        assert arg.dtype == DataType.FLOAT32

    def test_nbytes(self):
        arg = ChipTensor.make(0, (10, 20), DataType.FLOAT32)
        assert arg.nbytes() == 10 * 20 * 4

    def test_nbytes_int8(self):
        arg = ChipTensor.make(0, (256,), DataType.INT8)
        assert arg.nbytes() == 256

    def test_shapes_setter(self):
        arg = ChipTensor()
        arg.shapes = (3, 5, 7)
        assert arg.ndims == 3
        assert arg.shapes == (3, 5, 7)

    def test_max_dims(self):
        arg = ChipTensor.make(0, (1, 2, 3, 4, 5), DataType.INT32)
        assert arg.ndims == 5

    def test_exceed_max_dims(self):
        with pytest.raises((ValueError, RuntimeError)):
            ChipTensor.make(0, (1, 2, 3, 4, 5, 6), DataType.INT32)

    def test_dtype_readwrite(self):
        arg = ChipTensor.make(0, (1,), DataType.FLOAT32)
        arg.dtype = DataType.INT64
        assert arg.dtype == DataType.INT64

    def test_data_readwrite(self):
        arg = ChipTensor.make(0x1000, (1,), DataType.FLOAT32)
        assert arg.data == 0x1000
        arg.data = 0x2000
        assert arg.data == 0x2000

    def test_repr(self):
        arg = ChipTensor.make(0x1000, (4, 8), DataType.FLOAT16)
        r = repr(arg)
        assert "Tensor" in r
        assert "4" in r
        assert "8" in r
        assert "FLOAT16" in r


# ============================================================================
# ChipStorageTaskArgs
# ============================================================================


class TestChipStorageTaskArgs:
    def test_empty(self):
        args = ChipStorageTaskArgs()
        assert len(args) == 0
        assert args.tensor_count() == 0
        assert args.scalar_count() == 0

    def test_add_tensor(self):
        args = ChipStorageTaskArgs()
        t = ChipTensor.make(0xBEEF, (4, 8), DataType.FLOAT32)
        args.add_tensor(t)
        assert args.tensor_count() == 1
        assert args.scalar_count() == 0
        assert len(args) == 1

    def test_add_scalar(self):
        args = ChipStorageTaskArgs()
        args.add_scalar(42)
        assert args.scalar_count() == 1
        assert args.tensor_count() == 0
        assert len(args) == 1

    def test_mixed(self):
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(0x1, (2,), DataType.INT32))
        args.add_tensor(ChipTensor.make(0x2, (3,), DataType.FLOAT16))
        args.add_scalar(99)
        args.add_scalar(100)
        assert args.tensor_count() == 2
        assert args.scalar_count() == 2
        assert len(args) == 4

    def test_tensor_before_scalar_enforced(self):
        args = ChipStorageTaskArgs()
        args.add_scalar(42)
        with pytest.raises(RuntimeError):
            args.add_tensor(ChipTensor.make(0x1, (2,), DataType.INT32))

    def test_tensor_access(self):
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(0xA, (4,), DataType.FLOAT32))
        args.add_tensor(ChipTensor.make(0xB, (8,), DataType.INT32))
        assert args.tensor(0).data == 0xA
        assert args.tensor(1).data == 0xB
        assert args.tensor(0).shapes == (4,)
        assert args.tensor(1).shapes == (8,)

    def test_scalar_access(self):
        args = ChipStorageTaskArgs()
        args.add_scalar(111)
        args.add_scalar(222)
        assert args.scalar(0) == 111
        assert args.scalar(1) == 222

    def test_tensor_out_of_range(self):
        args = ChipStorageTaskArgs()
        with pytest.raises((IndexError, RuntimeError)):
            args.tensor(0)

    def test_scalar_out_of_range(self):
        args = ChipStorageTaskArgs()
        with pytest.raises((IndexError, RuntimeError)):
            args.scalar(0)

    def test_clear(self):
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(0, (1,), DataType.INT8))
        args.add_scalar(42)
        args.clear()
        assert len(args) == 0
        assert args.tensor_count() == 0
        assert args.scalar_count() == 0


# ============================================================================
# TensorArgType
# ============================================================================


class TestTensorArgType:
    def test_enum_values(self):
        assert TensorArgType.INPUT.value == 0
        assert TensorArgType.OUTPUT.value == 1
        assert TensorArgType.INOUT.value == 2

    def test_enum_identity(self):
        assert TensorArgType.INPUT is not None
        assert TensorArgType.OUTPUT is not None
        assert TensorArgType.INOUT is not None


# ============================================================================
# TaskArgs (unified vector-backed builder with per-tensor TensorArgType tags)
# ============================================================================


class TestTaskArgs:
    def test_empty(self):
        args = TaskArgs()
        assert len(args) == 0
        assert args.tensor_count() == 0
        assert args.scalar_count() == 0

    def test_add_tensor_default_tag(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0xBEEF, (4, 8), DataType.FLOAT32))
        assert args.tensor_count() == 1
        assert args.tag(0) == TensorArgType.INPUT

    def test_add_tensor_with_tag(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0xBEEF, (4, 8), DataType.FLOAT32), TensorArgType.OUTPUT)
        assert args.tag(0) == TensorArgType.OUTPUT

    def test_multiple_refs_with_tags(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0x1, (2,), DataType.INT32), TensorArgType.INPUT)
        args.add_tensor(_dev_ref(0x2, (3,), DataType.FLOAT16), TensorArgType.OUTPUT)
        args.add_tensor(_dev_ref(0x3, (4,), DataType.INT8), TensorArgType.INOUT)
        args.add_tensor(_dev_ref(0x4, (5,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        args.add_tensor(_dev_ref(0x5, (6,), DataType.INT32), TensorArgType.NO_DEP)
        assert args.tensor_count() == 5
        assert args.tag(0) == TensorArgType.INPUT
        assert args.tag(1) == TensorArgType.OUTPUT
        assert args.tag(2) == TensorArgType.INOUT
        assert args.tag(3) == TensorArgType.OUTPUT_EXISTING
        assert args.tag(4) == TensorArgType.NO_DEP

    def test_set_tag(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0x1, (2,), DataType.INT32))
        assert args.tag(0) == TensorArgType.INPUT
        args.set_tag(0, TensorArgType.INOUT)
        assert args.tag(0) == TensorArgType.INOUT

    def test_add_scalar(self):
        args = TaskArgs()
        args.add_scalar(42)
        assert args.scalar_count() == 1
        assert args.tensor_count() == 0
        assert len(args) == 1

    def test_mixed_with_tags(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0x1, (2,), DataType.INT32), TensorArgType.INPUT)
        args.add_tensor(_dev_ref(0x2, (3,), DataType.FLOAT16), TensorArgType.OUTPUT)
        args.add_scalar(99)
        args.add_scalar(100)
        assert args.tensor_count() == 2
        assert args.scalar_count() == 2
        assert len(args) == 4
        assert _ref_addr(args.tensor(0)) == 0x1
        assert _ref_addr(args.tensor(1)) == 0x2
        assert args.scalar(0) == 99
        assert args.scalar(1) == 100

    def test_tensor_before_scalar_enforced(self):
        args = TaskArgs()
        args.add_scalar(42)
        with pytest.raises(RuntimeError):
            args.add_tensor(_dev_ref(0x1, (2,), DataType.INT32))

    def test_ref_access(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0xA, (4,), DataType.FLOAT32))
        args.add_tensor(_dev_ref(0xB, (8,), DataType.INT32))
        assert _ref_addr(args.tensor(0)) == 0xA
        assert _ref_addr(args.tensor(1)) == 0xB
        assert tuple(args.tensor(0).shapes) == (4,)
        assert tuple(args.tensor(1).shapes) == (8,)

    def test_scalar_access(self):
        args = TaskArgs()
        args.add_scalar(111)
        args.add_scalar(222)
        assert args.scalar(0) == 111
        assert args.scalar(1) == 222

    def test_tensor_out_of_range(self):
        args = TaskArgs()
        with pytest.raises((IndexError, RuntimeError)):
            args.tensor(0)

    def test_scalar_out_of_range(self):
        args = TaskArgs()
        with pytest.raises((IndexError, RuntimeError)):
            args.scalar(0)

    def test_tag_out_of_range(self):
        args = TaskArgs()
        with pytest.raises((IndexError, RuntimeError)):
            args.tag(0)

    def test_set_tag_out_of_range(self):
        args = TaskArgs()
        with pytest.raises((IndexError, RuntimeError)):
            args.set_tag(0, TensorArgType.INPUT)

    def test_clear(self):
        args = TaskArgs()
        args.add_tensor(_dev_ref(0x100, (1,), DataType.INT8), TensorArgType.OUTPUT)
        args.add_scalar(42)
        args.clear()
        assert len(args) == 0
        assert args.tensor_count() == 0
        assert args.scalar_count() == 0

    def test_no_capacity_limit_tensors(self):
        """TaskArgs is vector-backed — no per-class capacity limit on refs."""
        args = TaskArgs()
        for i in range(20):
            args.add_tensor(_dev_ref(i + 1, (1,), DataType.INT8))
        assert args.tensor_count() == 20

    def test_no_capacity_limit_scalars(self):
        args = TaskArgs()
        for i in range(200):
            args.add_scalar(i)
        assert args.scalar_count() == 200


class TestRemoteTaskArgsSidecar:
    def test_remote_task_args_is_not_public_api(self):
        assert not hasattr(task_interface_module, "RemoteTaskArgs")
        assert "RemoteTaskArgs" not in task_interface_module.__all__

    def test_remote_buffer_ref_uses_whole_backing_and_sidecar_view(self):
        handle = RemoteBufferHandle._from_remote_allocation(
            worker_id=3,
            buffer_id=11,
            generation=2,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=64,
            remote_addr=0xCAFE,
            rkey_or_token=0xBEEF,
        )
        ref = RemoteTensorRef(handle=handle, offset=8, shape=(4,), dtype=DataType.UINT8)

        args = TaskArgs()
        args.add_tensor(ref, TensorArgType.OUTPUT)
        args.add_scalar(9)

        assert args.tensor_count() == 1
        # An arg destined for a remote worker carries a REMOTE_SIDECAR placeholder ref (no local backing).
        assert args.tensor(0).buffer.backend_kind == BackendKind.REMOTE_SIDECAR
        assert args.tensor(0).buffer.nbytes == 64
        assert args.tensor(0).byte_offset == 8
        assert args.tag(0) == TensorArgType.OUTPUT
        assert args.scalar(0) == 9

        sidecar = _remote_sidecar_for(args)
        assert sidecar is not None
        assert len(sidecar.tensors) == 1
        assert sidecar.tensors[0] is not None
        assert sidecar.tensors[0].present
        desc = sidecar.tensors[0].desc
        assert desc.address_space == RemoteAddressSpace.REMOTE_DEVICE
        assert desc.owner_worker_id == 3
        assert desc.buffer_id == 11
        assert desc.generation == 2
        assert desc.offset == 8
        assert desc.nbytes == 4
        assert desc.remote_addr == 0xCAFE
        assert desc.rkey_or_token == 0xBEEF

    def test_remote_sidecar_storage_is_bound_to_task_args_lifetime(self):
        gc.collect()
        before_count = len(task_interface_module._REMOTE_TASK_ARGS_STORAGE)  # noqa: SLF001

        def make_remote_args_ref():
            handle = RemoteBufferHandle._from_remote_allocation(
                worker_id=3,
                buffer_id=11,
                generation=2,
                address_space=RemoteAddressSpace.REMOTE_DEVICE,
                nbytes=64,
            )
            args = TaskArgs()
            args.add_tensor(RemoteTensorRef(handle=handle, shape=(4,), dtype=DataType.UINT8))
            assert args in task_interface_module._REMOTE_TASK_ARGS_STORAGE  # noqa: SLF001
            assert _remote_sidecar_for(args) is not None
            return weakref.ref(args)

        ref = make_remote_args_ref()
        for _ in range(3):
            gc.collect()

        assert ref() is None
        assert len(task_interface_module._REMOTE_TASK_ARGS_STORAGE) == before_count  # noqa: SLF001

    def test_remote_buffer_handle_is_opaque_to_public_constructor(self):
        with pytest.raises(TypeError, match="Worker.remote_malloc"):
            RemoteBufferHandle(
                worker_id=3,
                buffer_id=11,
                generation=2,
                address_space=RemoteAddressSpace.REMOTE_DEVICE,
                nbytes=64,
                remote_addr=0xCAFE,
                rkey_or_token=0xBEEF,
            )
        with pytest.raises(TypeError, match="Worker.remote_malloc"):
            RemoteBufferHandle(
                worker_id=0,
                buffer_id=0,
                generation=0,
                address_space=RemoteAddressSpace.HOST_INLINE,
                nbytes=4,
            )

        handle = RemoteBufferHandle._from_remote_allocation(
            worker_id=3,
            buffer_id=11,
            generation=2,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=64,
            remote_addr=0xCAFE,
            rkey_or_token=0xBEEF,
        )
        assert handle.worker_id == 3
        assert handle.nbytes == 64
        assert not handle.released
        assert not hasattr(handle, "rkey_or_token")

    def test_host_inline_ref_uses_inline_payload_arena(self):
        args = TaskArgs()
        args.add_tensor(
            RemoteTensorRef.host_inline(b"abcd", shape=(4,), dtype=DataType.UINT8),
            TensorArgType.INPUT,
        )

        sidecar = _remote_sidecar_for(args)
        assert sidecar is not None
        assert sidecar.inline_payload == b"abcd"
        assert sidecar.tensors[0] is not None
        desc = sidecar.tensors[0].desc
        assert desc.address_space == RemoteAddressSpace.HOST_INLINE
        assert desc.inline_payload_offset == 0
        assert desc.inline_payload_len == 4
        assert desc.buffer_id == 0
        assert desc.generation == 0

    def test_host_inline_ref_rejects_payload_shape_mismatch(self):
        with pytest.raises(ValueError, match="HOST_INLINE payload length"):
            RemoteTensorRef.host_inline(b"abcd", shape=(2,), dtype=DataType.UINT8)

    def test_remote_ref_rejects_out_of_bounds_range(self):
        handle = RemoteBufferHandle._from_remote_allocation(
            worker_id=1,
            buffer_id=2,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_DEVICE,
            nbytes=4,
        )
        with pytest.raises(ValueError, match="exceeds"):
            RemoteTensorRef(handle=handle, offset=2, shape=(4,), dtype=DataType.UINT8)

    def test_remote_buffer_export_is_opaque_to_public_constructor_and_repr(self):
        with pytest.raises(TypeError, match="Worker.remote_export"):
            RemoteBufferExport(
                owner_worker_id=0,
                buffer_id=1,
                generation=1,
                address_space=RemoteAddressSpace.REMOTE_WINDOW,
                offset=0,
                nbytes=4,
                export_id=1,
                remote_addr=0xCAFE,
                rkey_or_token=0xBEEF,
                ub_ldst_va=0,
                access_flags=3,
                transport_profile="sim",
                transport_descriptor=b"psm_secret",
            )

        exported = RemoteBufferExport._from_remote_export(
            owner_worker_id=0,
            buffer_id=1,
            generation=1,
            address_space=RemoteAddressSpace.REMOTE_WINDOW,
            offset=0,
            nbytes=4,
            export_id=1,
            remote_addr=0xCAFE,
            rkey_or_token=0xBEEF,
            ub_ldst_va=0,
            access_flags=3,
            transport_profile="sim",
            transport_descriptor=b"psm_secret",
        )
        assert exported.owner_worker_id == 0
        assert exported.nbytes == 4
        assert not hasattr(exported, "remote_addr")
        assert not hasattr(exported, "rkey_or_token")
        assert not hasattr(exported, "ub_ldst_va")
        assert not hasattr(exported, "transport_descriptor")
        assert "psm_secret" not in repr(exported)
        assert "0xCAFE" not in repr(exported)


class TestRemoteL3SessionTaskArgsMaterialization:
    def test_task_payload_decode_preserves_scope_stats_config(self):
        from simpler.remote_l3_protocol import decode_task_payload

        prefix = b"/tmp/remote-scope"
        config = struct.pack("<iiiiii", 5, 0, 0, 0, 0, 1) + struct.pack("<I", len(prefix)) + prefix
        args = struct.pack("<III", 0, 0, 0)
        wire = (b"\xab" * 32) + config + args

        payload = decode_task_payload(wire)

        assert payload.config.aicpu_thread_num == 5
        assert payload.config.enable_scope_stats is True
        assert payload.config.output_prefix == prefix.decode()

    def test_host_inline_descriptor_materializes_local_tensor_data(self):
        from simpler.remote_l3_protocol import (
            RemoteAddressSpace as WireRemoteAddressSpace,
        )
        from simpler.remote_l3_protocol import (
            RemoteTaskArgsWire,
            RemoteTensorDesc,
            RemoteTensorSidecar,
        )
        from simpler.remote_l3_session import _materialize_task_args

        tensor = _remote_arg_tensor((4,), nbytes=4)
        desc = RemoteTensorDesc(
            address_space=WireRemoteAddressSpace.HOST_INLINE,
            owner_worker_id=0,
            buffer_id=0,
            offset=0,
            nbytes=4,
            remote_addr=0,
            rkey_or_token=0,
            generation=0,
            inline_payload_offset=0,
            inline_payload_len=4,
            flags=0,
        )
        wire = RemoteTaskArgsWire((tensor,), (RemoteTensorSidecar(True, desc),), (), b"abcd")

        args, inline_backings = _materialize_task_args(
            wire, {}, worker_id=3, mint_inline_buffer=_session_buffer_minter()
        )

        assert len(inline_backings) == 1
        backing = inline_backings[0]
        arg = args.tensor(0)
        assert arg.buffer.identity == backing.identity
        assert arg.byte_offset == 0
        assert ctypes.string_at(backing.base, 4) == b"abcd"
        backing.close()

    def test_host_inline_backing_is_released_when_the_caller_closes_it(self):
        from simpler.remote_l3_protocol import (
            RemoteAddressSpace as WireRemoteAddressSpace,
        )
        from simpler.remote_l3_protocol import (
            RemoteTaskArgsWire,
            RemoteTensorDesc,
            RemoteTensorSidecar,
        )
        from simpler.remote_l3_session import _materialize_task_args

        tensor = _remote_arg_tensor((4,), nbytes=4)
        desc = RemoteTensorDesc(
            address_space=WireRemoteAddressSpace.HOST_INLINE,
            owner_worker_id=0,
            buffer_id=0,
            offset=0,
            nbytes=4,
            remote_addr=0,
            rkey_or_token=0,
            generation=0,
            inline_payload_offset=0,
            inline_payload_len=4,
            flags=0,
        )
        wire = RemoteTaskArgsWire((tensor,), (RemoteTensorSidecar(True, desc),), (), b"abcd")
        minter = _session_buffer_minter()

        _first, first_backings = _materialize_task_args(wire, {}, worker_id=3, mint_inline_buffer=minter)
        _second, second_backings = _materialize_task_args(wire, {}, worker_id=3, mint_inline_buffer=minter)

        # Two tasks' payloads are two backings, so a consumer's map-once import cache is never asked
        # to bind one identity to both.
        assert first_backings[0].identity != second_backings[0].identity
        for backing in (*first_backings, *second_backings):
            assert backing.shm is not None
            name = backing.shm.name
            backing.close()
            assert backing.shm is None
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)

    def test_remote_buffer_descriptor_materializes_session_registry_address(self):
        from simpler.remote_l3_protocol import (
            RemoteAddressSpace as WireRemoteAddressSpace,
        )
        from simpler.remote_l3_protocol import (
            RemoteTaskArgsWire,
            RemoteTensorDesc,
            RemoteTensorSidecar,
        )
        from simpler.remote_l3_session import _materialize_task_args, _RemoteBufferEntry

        backing = create_host_shared_buffer(8, mint_owner_instance_id(), buffer_id=9)
        try:
            ctypes.memmove(backing.base, b"01234567", 8)
            entry = _RemoteBufferEntry(backing, 8, 1, WireRemoteAddressSpace.REMOTE_DEVICE)
            tensor = _remote_arg_tensor((4,), nbytes=8, byte_offset=2)
            desc = RemoteTensorDesc(
                address_space=WireRemoteAddressSpace.REMOTE_DEVICE,
                owner_worker_id=2,
                buffer_id=9,
                offset=2,
                nbytes=4,
                remote_addr=entry.addr,
                rkey_or_token=0,
                generation=1,
                inline_payload_offset=0,
                inline_payload_len=0,
                flags=0,
            )
            wire = RemoteTaskArgsWire((tensor,), (RemoteTensorSidecar(True, desc),), (), b"")

            args, inline_backings = _materialize_task_args(
                wire, {(9, 1): entry}, worker_id=2, mint_inline_buffer=_session_buffer_minter()
            )

            assert inline_backings == []
            arg = args.tensor(0)
            # An interior range is a byte_offset on the view over the WHOLE backing, never a second
            # identity: two views of one backing must key alike for dependency inference.
            assert arg.buffer.identity == backing.identity
            assert arg.buffer.nbytes == 8
            assert arg.byte_offset == 2
            assert ctypes.string_at(backing.base + arg.byte_offset, 4) == b"2345"
        finally:
            backing.close()

    def test_two_interior_views_of_one_backing_share_its_identity(self):
        from simpler.remote_l3_protocol import (
            RemoteAddressSpace as WireRemoteAddressSpace,
        )
        from simpler.remote_l3_protocol import (
            RemoteTaskArgsWire,
            RemoteTensorDesc,
            RemoteTensorSidecar,
        )
        from simpler.remote_l3_session import _materialize_task_args, _RemoteBufferEntry

        backing = create_host_shared_buffer(8, mint_owner_instance_id(), buffer_id=9)
        try:
            entry = _RemoteBufferEntry(backing, 8, 1, WireRemoteAddressSpace.REMOTE_DEVICE)

            def sub_range(offset):
                return RemoteTensorDesc(
                    address_space=WireRemoteAddressSpace.REMOTE_DEVICE,
                    owner_worker_id=2,
                    buffer_id=9,
                    offset=offset,
                    nbytes=4,
                    remote_addr=entry.addr + offset,
                    rkey_or_token=0,
                    generation=1,
                    inline_payload_offset=0,
                    inline_payload_len=0,
                    flags=0,
                )

            tensor = _remote_arg_tensor((4,), nbytes=8)
            tensor_second = _remote_arg_tensor((4,), nbytes=8, byte_offset=4)
            wire = RemoteTaskArgsWire(
                (tensor, tensor_second),
                (RemoteTensorSidecar(True, sub_range(0)), RemoteTensorSidecar(True, sub_range(4))),
                (),
                b"",
            )

            args, _inline_backings = _materialize_task_args(
                wire, {(9, 1): entry}, worker_id=2, mint_inline_buffer=_session_buffer_minter()
            )

            assert args.tensor(0).buffer.identity == args.tensor(1).buffer.identity
            assert (args.tensor(0).byte_offset, args.tensor(1).byte_offset) == (0, 4)
        finally:
            backing.close()

    def test_materialize_rejects_a_wire_offset_that_disagrees_with_the_sidecar(self):
        from simpler.remote_l3_protocol import (
            RemoteAddressSpace as WireRemoteAddressSpace,
        )
        from simpler.remote_l3_protocol import (
            RemoteTaskArgsWire,
            RemoteTensorDesc,
            RemoteTensorSidecar,
        )
        from simpler.remote_l3_session import _materialize_task_args, _RemoteBufferEntry

        backing = create_host_shared_buffer(8, mint_owner_instance_id(), buffer_id=9)
        try:
            entry = _RemoteBufferEntry(backing, 8, 1, WireRemoteAddressSpace.REMOTE_DEVICE)
            tensor = _remote_arg_tensor((4,), nbytes=8, byte_offset=0)
            desc = RemoteTensorDesc(
                address_space=WireRemoteAddressSpace.REMOTE_DEVICE,
                owner_worker_id=2,
                buffer_id=9,
                offset=2,
                nbytes=4,
                remote_addr=entry.addr,
                rkey_or_token=0,
                generation=1,
                inline_payload_offset=0,
                inline_payload_len=0,
                flags=0,
            )
            wire = RemoteTaskArgsWire((tensor,), (RemoteTensorSidecar(True, desc),), (), b"")

            with pytest.raises(ValueError, match="byte_offset disagrees"):
                _materialize_task_args(wire, {(9, 1): entry}, worker_id=2, mint_inline_buffer=_session_buffer_minter())
        finally:
            backing.close()

    def test_materialized_args_forward_to_a_next_level_submit(self):
        from simpler.orchestrator import _split_next_level_args
        from simpler.remote_l3_protocol import (
            RemoteAddressSpace as WireRemoteAddressSpace,
        )
        from simpler.remote_l3_protocol import (
            RemoteTaskArgsWire,
            RemoteTensorDesc,
            RemoteTensorSidecar,
        )
        from simpler.remote_l3_session import _materialize_task_args, _RemoteBufferEntry

        backing = create_host_shared_buffer(8, mint_owner_instance_id(), buffer_id=9)
        try:
            entry = _RemoteBufferEntry(backing, 8, 1, WireRemoteAddressSpace.REMOTE_DEVICE)
            desc = RemoteTensorDesc(
                address_space=WireRemoteAddressSpace.REMOTE_DEVICE,
                owner_worker_id=2,
                buffer_id=9,
                offset=0,
                nbytes=4,
                remote_addr=entry.addr,
                rkey_or_token=0,
                generation=1,
                inline_payload_offset=0,
                inline_payload_len=0,
                flags=0,
            )
            wire = RemoteTaskArgsWire(
                (_remote_arg_tensor((4,), nbytes=8),), (RemoteTensorSidecar(True, desc),), (), b""
            )

            args, _inline_backings = _materialize_task_args(
                wire, {(9, 1): entry}, worker_id=2, mint_inline_buffer=_session_buffer_minter()
            )

            # What a remote orchestration function does with its args: name them in a child's
            # TaskArgs and submit that to the next level. The chip POD accepts neither step.
            child_args = TaskArgs()
            child_args.add_tensor(args.tensor(0), TensorArgType.INPUT)
            forwarded, remote_sidecar = _split_next_level_args(child_args)
            assert forwarded is child_args
            assert remote_sidecar is None
        finally:
            backing.close()


# ============================================================================
# ArgDirection
# ============================================================================


class TestArgDirection:
    def test_enum_values(self):
        assert ArgDirection.SCALAR.value == 0
        assert ArgDirection.IN.value == 1
        assert ArgDirection.OUT.value == 2
        assert ArgDirection.INOUT.value == 3

    @pytest.mark.parametrize(
        "direction,expected",
        [
            (ArgDirection.SCALAR, "SCALAR"),
            (ArgDirection.IN, "IN"),
            (ArgDirection.OUT, "OUT"),
            (ArgDirection.INOUT, "INOUT"),
        ],
    )
    def test_arg_direction_name(self, direction, expected):
        assert arg_direction_name(direction) == expected


# ============================================================================
# CoreCallable
# ============================================================================


class TestCoreCallable:
    def test_build_and_access(self):
        sig = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
        binary = b"\x01\x02\x03\x04"
        cc = CoreCallable.build(signature=sig, binary=binary)
        assert cc.sig_count == 3
        assert cc.sig(0) == ArgDirection.IN
        assert cc.sig(1) == ArgDirection.OUT
        assert cc.sig(2) == ArgDirection.SCALAR
        assert cc.binary_size == 4

    def test_empty_signature(self):
        cc = CoreCallable.build(signature=[], binary=b"\xab")
        assert cc.sig_count == 0
        assert cc.binary_size == 1

    def test_large_binary(self):
        binary = bytes(range(256)) * 40  # 10240 bytes
        cc = CoreCallable.build(signature=[ArgDirection.IN], binary=binary)
        assert cc.binary_size == 10240

    def test_empty_binary(self):
        cc = CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT], binary=b"")
        assert cc.sig_count == 2
        assert cc.binary_size == 0

    def test_buffer_ptr_and_size(self):
        cc = CoreCallable.build(signature=[ArgDirection.IN], binary=b"\x00" * 100)
        assert cc.buffer_ptr() != 0
        assert cc.buffer_size() > 100

    def test_sig_out_of_range(self):
        cc = CoreCallable.build(signature=[ArgDirection.IN], binary=b"\x00")
        with pytest.raises((IndexError, RuntimeError)):
            cc.sig(1)
        with pytest.raises((IndexError, RuntimeError)):
            cc.sig(-1)

    def test_repr(self):
        cc = CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT], binary=b"\x00" * 8)
        r = repr(cc)
        assert "CoreCallable" in r
        assert "sig_count=2" in r
        assert "binary_size=8" in r


# ============================================================================
# ChipCallable
# ============================================================================


class TestChipCallable:
    def _make_child(self, sig, binary):
        return CoreCallable.build(signature=sig, binary=binary)

    def test_signature_capacity(self):
        chip = ChipCallable.build(
            signature=[ArgDirection.IN] * 256,
            func_name="capacity",
            binary=b"",
            children=[],
        )
        assert chip.sig_count == 256

    def test_signature_overflow_reports_counts(self):
        with pytest.raises(ValueError, match=r"requested tensor count 257.*supported tensor count 256"):
            ChipCallable.build(
                signature=[ArgDirection.IN] * 257,
                func_name="overflow",
                binary=b"",
                children=[],
            )

    def test_build_with_children(self):
        child0 = self._make_child([ArgDirection.IN], b"\x01\x02\x03\x04")
        child1 = self._make_child([ArgDirection.OUT, ArgDirection.SCALAR], b"\x05\x06")
        chip = ChipCallable.build(
            signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
            func_name="test_func",
            binary=b"\xaa" * 16,
            children=[(10, child0), (20, child1)],
        )
        assert chip.sig_count == 3
        assert chip.sig(0) == ArgDirection.IN
        assert chip.sig(1) == ArgDirection.OUT
        assert chip.sig(2) == ArgDirection.INOUT
        assert chip.binary_size == 16
        assert chip.child_count == 2
        assert chip.child_func_id(0) == 10
        assert chip.child_func_id(1) == 20

    def test_child_alignment(self):
        """All child offsets must be multiples of 64."""
        child0 = self._make_child([ArgDirection.IN], b"\x01" * 7)
        child1 = self._make_child([ArgDirection.OUT], b"\x02" * 100)
        child2 = self._make_child([ArgDirection.SCALAR], b"\x03" * 1)
        chip = ChipCallable.build(
            signature=[ArgDirection.IN],
            func_name="test_func",
            binary=b"\xbb" * 10,
            children=[(1, child0), (2, child1), (3, child2)],
        )
        for i in range(chip.child_count):
            assert chip.child_offset(i) % 64 == 0, f"child_offset({i}) = {chip.child_offset(i)} not aligned to 64"

    def test_no_children(self):
        chip = ChipCallable.build(
            signature=[ArgDirection.IN, ArgDirection.OUT],
            func_name="test_func",
            binary=b"\xcc" * 32,
            children=[],
        )
        assert chip.sig_count == 2
        assert chip.binary_size == 32
        assert chip.child_count == 0

    def test_nested_child_access(self):
        child = self._make_child([ArgDirection.IN, ArgDirection.OUT], b"\xdd" * 8)
        chip = ChipCallable.build(
            signature=[ArgDirection.SCALAR],
            func_name="test_func",
            binary=b"\xee" * 4,
            children=[(42, child)],
        )
        retrieved = chip.child(0)
        assert retrieved.sig_count == 2
        assert retrieved.sig(0) == ArgDirection.IN
        assert retrieved.sig(1) == ArgDirection.OUT
        assert retrieved.binary_size == 8

    def test_child_out_of_range(self):
        chip = ChipCallable.build(
            signature=[ArgDirection.IN],
            func_name="test_func",
            binary=b"\x00",
            children=[],
        )
        with pytest.raises((IndexError, RuntimeError)):
            chip.child(0)
        with pytest.raises((IndexError, RuntimeError)):
            chip.child_func_id(0)

    def test_buffer_ptr_and_size(self):
        child = self._make_child([ArgDirection.IN], b"\x00" * 50)
        chip = ChipCallable.build(
            signature=[ArgDirection.IN],
            func_name="test_func",
            binary=b"\x00" * 100,
            children=[(1, child)],
        )
        assert chip.buffer_ptr() != 0
        assert chip.buffer_size() > 100

    def test_repr(self):
        child = self._make_child([ArgDirection.IN], b"\x00")
        chip = ChipCallable.build(
            signature=[ArgDirection.IN, ArgDirection.OUT],
            func_name="test_func",
            binary=b"\x00" * 8,
            children=[(1, child)],
        )
        r = repr(chip)
        assert "ChipCallable" in r
        assert "sig_count=2" in r
        assert "child_count=1" in r


# ============================================================================
# ChipTensor.child_memory
# ============================================================================


class TestChildMemory:
    def test_default_is_false(self):
        t = ChipTensor()
        assert t.child_memory is False

    def test_make_default_is_false(self):
        t = ChipTensor.make(0x1000, (4,), DataType.FLOAT32)
        assert t.child_memory is False

    def test_make_child_memory_true(self):
        t = ChipTensor.make(0xDEAD, (8,), DataType.FLOAT16, child_memory=True)
        assert t.child_memory is True
        assert t.data == 0xDEAD
        assert t.shapes == (8,)
        assert t.dtype == DataType.FLOAT16

    def test_set_child_memory(self):
        t = ChipTensor.make(0x1000, (4,), DataType.FLOAT32)
        assert t.child_memory is False
        t.child_memory = True
        assert t.child_memory is True

    def test_repr_shows_child_memory_when_set(self):
        t = ChipTensor.make(0x1000, (4,), DataType.FLOAT32, child_memory=True)
        r = repr(t)
        assert "child_memory=True" in r

    def test_repr_hides_child_memory_when_default(self):
        t = ChipTensor.make(0x1000, (4,), DataType.FLOAT32)
        r = repr(t)
        assert "child_memory" not in r

    def test_chip_storage_preserves_child_memory(self):
        args = ChipStorageTaskArgs()
        t = ChipTensor.make(0x2000, (16,), DataType.INT32, child_memory=True)
        args.add_tensor(t)
        out = args.tensor(0)
        assert out.child_memory is True
        assert out.data == 0x2000


class TestAddRefAccessSubset:
    # §3 access: add_tensor rejects an arg whose TensorArgType needs access the handle does not grant.
    def test_add_tensor_rejects_write_arg_on_read_only_backing(self):
        oid = mint_owner_instance_id()
        h = wrap_fork_inherited(0x1000, 64, owner_instance_id=oid, buffer_id=1, access=AccessMode.READ)
        ref = h.tensor((16,), DataType.FLOAT32)  # READ-only backing
        args = TaskArgs()
        with pytest.raises(ValueError, match="access"):
            args.add_tensor(ref, TensorArgType.OUTPUT_EXISTING)
        with pytest.raises(ValueError, match="access"):
            args.add_tensor(ref, TensorArgType.INOUT)
        args.add_tensor(ref, TensorArgType.INPUT)  # READ⊆READ ok

    def test_add_tensor_accepts_matching_access(self):
        oid = mint_owner_instance_id()
        h = create_host_shared_buffer(64, oid, buffer_id=1)  # READWRITE
        ref = h.tensor((16,), DataType.FLOAT32)
        try:
            args = TaskArgs()
            args.add_tensor(ref, TensorArgType.INPUT)  # READ⊆RW
            args.add_tensor(ref, TensorArgType.OUTPUT_EXISTING)  # WRITE⊆RW
            args.add_tensor(ref, TensorArgType.INOUT)  # RW⊆RW
            assert args.tensor_count() == 3
        finally:
            h.close()
