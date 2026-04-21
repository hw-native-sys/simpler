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
import struct

import pytest
from _task_interface import (  # pyright: ignore[reportMissingImports]
    CONTINUOUS_TENSOR_MAX_DIMS,
    ArgDirection,
    ChipCallable,
    ChipStorageTaskArgs,
    ContinuousTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
)

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
        ],
    )
    def test_dtype_names(self, dtype, expected):
        assert get_dtype_name(dtype) == expected


# ============================================================================
# task_interface.py wrapper (torch integration)
# ============================================================================


class TestTaskInterfaceWrapper:
    def test_torch_dtype_to_datatype(self):
        import torch  # pyright: ignore[reportMissingImports]
        from simpler.task_interface import torch_dtype_to_datatype

        assert torch_dtype_to_datatype(torch.float32) == DataType.FLOAT32
        assert torch_dtype_to_datatype(torch.int8) == DataType.INT8
        assert torch_dtype_to_datatype(torch.bfloat16) == DataType.BFLOAT16

    def test_torch_dtype_unsupported(self):
        import torch  # pyright: ignore[reportMissingImports]
        from simpler.task_interface import torch_dtype_to_datatype

        with pytest.raises(KeyError):
            torch_dtype_to_datatype(torch.complex64)

    def test_make_tensor_arg(self):
        import torch  # pyright: ignore[reportMissingImports]
        from simpler.task_interface import make_tensor_arg

        t = torch.zeros(4, 8, dtype=torch.float32)
        arg = make_tensor_arg(t)
        assert isinstance(arg, ContinuousTensor)
        assert arg.data == t.data_ptr()
        assert arg.shapes == (4, 8)
        assert arg.dtype == DataType.FLOAT32
        assert arg.nbytes() == 4 * 8 * 4

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
# ContinuousTensor
# ============================================================================


class TestContinuousTensor:
    def test_default_constructor(self):
        arg = ContinuousTensor()
        assert arg is not None

    def test_max_dims_constant(self):
        assert CONTINUOUS_TENSOR_MAX_DIMS == 5

    def test_make(self):
        arg = ContinuousTensor.make(0xDEAD, (4, 8), DataType.FLOAT32)
        assert arg.data == 0xDEAD
        assert arg.shapes == (4, 8)
        assert arg.ndims == 2
        assert arg.dtype == DataType.FLOAT32

    def test_nbytes(self):
        arg = ContinuousTensor.make(0, (10, 20), DataType.FLOAT32)
        assert arg.nbytes() == 10 * 20 * 4

    def test_nbytes_int8(self):
        arg = ContinuousTensor.make(0, (256,), DataType.INT8)
        assert arg.nbytes() == 256

    def test_shapes_setter(self):
        arg = ContinuousTensor()
        arg.shapes = (3, 5, 7)
        assert arg.ndims == 3
        assert arg.shapes == (3, 5, 7)

    def test_max_dims(self):
        arg = ContinuousTensor.make(0, (1, 2, 3, 4, 5), DataType.INT32)
        assert arg.ndims == 5

    def test_exceed_max_dims(self):
        with pytest.raises((ValueError, RuntimeError)):
            ContinuousTensor.make(0, (1, 2, 3, 4, 5, 6), DataType.INT32)

    def test_dtype_readwrite(self):
        arg = ContinuousTensor.make(0, (1,), DataType.FLOAT32)
        arg.dtype = DataType.INT64
        assert arg.dtype == DataType.INT64

    def test_data_readwrite(self):
        arg = ContinuousTensor.make(0x1000, (1,), DataType.FLOAT32)
        assert arg.data == 0x1000
        arg.data = 0x2000
        assert arg.data == 0x2000

    def test_repr(self):
        arg = ContinuousTensor.make(0x1000, (4, 8), DataType.FLOAT16)
        r = repr(arg)
        assert "ContinuousTensor" in r
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
        t = ContinuousTensor.make(0xBEEF, (4, 8), DataType.FLOAT32)
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
        args.add_tensor(ContinuousTensor.make(0x1, (2,), DataType.INT32))
        args.add_tensor(ContinuousTensor.make(0x2, (3,), DataType.FLOAT16))
        args.add_scalar(99)
        args.add_scalar(100)
        assert args.tensor_count() == 2
        assert args.scalar_count() == 2
        assert len(args) == 4

    def test_tensor_before_scalar_enforced(self):
        args = ChipStorageTaskArgs()
        args.add_scalar(42)
        with pytest.raises(RuntimeError):
            args.add_tensor(ContinuousTensor.make(0x1, (2,), DataType.INT32))

    def test_tensor_access(self):
        args = ChipStorageTaskArgs()
        args.add_tensor(ContinuousTensor.make(0xA, (4,), DataType.FLOAT32))
        args.add_tensor(ContinuousTensor.make(0xB, (8,), DataType.INT32))
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
        args.add_tensor(ContinuousTensor.make(0, (1,), DataType.INT8))
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
        t = ContinuousTensor.make(0xBEEF, (4, 8), DataType.FLOAT32)
        args.add_tensor(t)
        assert args.tensor_count() == 1
        assert args.tag(0) == TensorArgType.INPUT

    def test_add_tensor_with_tag(self):
        args = TaskArgs()
        t = ContinuousTensor.make(0xBEEF, (4, 8), DataType.FLOAT32)
        args.add_tensor(t, TensorArgType.OUTPUT)
        assert args.tag(0) == TensorArgType.OUTPUT

    def test_multiple_tensors_with_tags(self):
        args = TaskArgs()
        args.add_tensor(ContinuousTensor.make(0x1, (2,), DataType.INT32), TensorArgType.INPUT)
        args.add_tensor(ContinuousTensor.make(0x2, (3,), DataType.FLOAT16), TensorArgType.OUTPUT)
        args.add_tensor(ContinuousTensor.make(0x3, (4,), DataType.INT8), TensorArgType.INOUT)
        args.add_tensor(ContinuousTensor.make(0x4, (5,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        args.add_tensor(ContinuousTensor.make(0x5, (6,), DataType.INT32), TensorArgType.NO_DEP)
        assert args.tensor_count() == 5
        assert args.tag(0) == TensorArgType.INPUT
        assert args.tag(1) == TensorArgType.OUTPUT
        assert args.tag(2) == TensorArgType.INOUT
        assert args.tag(3) == TensorArgType.OUTPUT_EXISTING
        assert args.tag(4) == TensorArgType.NO_DEP

    def test_set_tag(self):
        args = TaskArgs()
        args.add_tensor(ContinuousTensor.make(0x1, (2,), DataType.INT32))
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
        args.add_tensor(ContinuousTensor.make(0x1, (2,), DataType.INT32), TensorArgType.INPUT)
        args.add_tensor(ContinuousTensor.make(0x2, (3,), DataType.FLOAT16), TensorArgType.OUTPUT)
        args.add_scalar(99)
        args.add_scalar(100)
        assert args.tensor_count() == 2
        assert args.scalar_count() == 2
        assert len(args) == 4
        assert args.tensor(0).data == 0x1
        assert args.tensor(1).data == 0x2
        assert args.scalar(0) == 99
        assert args.scalar(1) == 100

    def test_tensor_before_scalar_enforced(self):
        args = TaskArgs()
        args.add_scalar(42)
        with pytest.raises(RuntimeError):
            args.add_tensor(ContinuousTensor.make(0x1, (2,), DataType.INT32))

    def test_tensor_access(self):
        args = TaskArgs()
        args.add_tensor(ContinuousTensor.make(0xA, (4,), DataType.FLOAT32))
        args.add_tensor(ContinuousTensor.make(0xB, (8,), DataType.INT32))
        assert args.tensor(0).data == 0xA
        assert args.tensor(1).data == 0xB
        assert args.tensor(0).shapes == (4,)
        assert args.tensor(1).shapes == (8,)

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
        args.add_tensor(ContinuousTensor.make(0, (1,), DataType.INT8), TensorArgType.OUTPUT)
        args.add_scalar(42)
        args.clear()
        assert len(args) == 0
        assert args.tensor_count() == 0
        assert args.scalar_count() == 0

    def test_no_capacity_limit_tensors(self):
        """TaskArgs is vector-backed — no per-class capacity limit on tensors."""
        args = TaskArgs()
        for i in range(20):
            args.add_tensor(ContinuousTensor.make(i, (1,), DataType.INT8))
        assert args.tensor_count() == 20

    def test_no_capacity_limit_scalars(self):
        args = TaskArgs()
        for i in range(200):
            args.add_scalar(i)
        assert args.scalar_count() == 200


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
# ContinuousTensor.child_memory
# ============================================================================


class TestChildMemory:
    def test_default_is_false(self):
        t = ContinuousTensor()
        assert t.child_memory is False

    def test_make_default_is_false(self):
        t = ContinuousTensor.make(0x1000, (4,), DataType.FLOAT32)
        assert t.child_memory is False

    def test_make_child_memory_true(self):
        t = ContinuousTensor.make(0xDEAD, (8,), DataType.FLOAT16, child_memory=True)
        assert t.child_memory is True
        assert t.data == 0xDEAD
        assert t.shapes == (8,)
        assert t.dtype == DataType.FLOAT16

    def test_set_child_memory(self):
        t = ContinuousTensor.make(0x1000, (4,), DataType.FLOAT32)
        assert t.child_memory is False
        t.child_memory = True
        assert t.child_memory is True

    def test_repr_shows_child_memory_when_set(self):
        t = ContinuousTensor.make(0x1000, (4,), DataType.FLOAT32, child_memory=True)
        r = repr(t)
        assert "child_memory=True" in r

    def test_repr_hides_child_memory_when_default(self):
        t = ContinuousTensor.make(0x1000, (4,), DataType.FLOAT32)
        r = repr(t)
        assert "child_memory" not in r

    def test_chip_storage_preserves_child_memory(self):
        args = ChipStorageTaskArgs()
        t = ContinuousTensor.make(0x2000, (16,), DataType.INT32, child_memory=True)
        args.add_tensor(t)
        out = args.tensor(0)
        assert out.child_memory is True
        assert out.data == 0x2000
