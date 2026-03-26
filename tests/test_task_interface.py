"""Tests for the _task_interface nanobind extension and task_interface wrapper."""

import ctypes
import struct
import sys
from pathlib import Path

import pytest

# Ensure python/ is on the import path so _task_interface and task_interface resolve
_python_dir = str(Path(__file__).resolve().parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from _task_interface import (
    DataType,
    TaskArgKind,
    TASK_ARG_MAX_DIMS,
    get_element_size,
    get_dtype_name,
    TaskArg,
    TaskArgArray,
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


class TestGetElementSize:
    @pytest.mark.parametrize("dtype,expected", [
        (DataType.FLOAT32, 4),
        (DataType.FLOAT16, 2),
        (DataType.INT32, 4),
        (DataType.INT16, 2),
        (DataType.INT8, 1),
        (DataType.UINT8, 1),
        (DataType.BFLOAT16, 2),
        (DataType.INT64, 8),
        (DataType.UINT64, 8),
    ])
    def test_element_sizes(self, dtype, expected):
        assert get_element_size(dtype) == expected


class TestGetDtypeName:
    @pytest.mark.parametrize("dtype,expected", [
        (DataType.FLOAT32, "FLOAT32"),
        (DataType.FLOAT16, "FLOAT16"),
        (DataType.INT32, "INT32"),
        (DataType.INT16, "INT16"),
        (DataType.INT8, "INT8"),
        (DataType.UINT8, "UINT8"),
        (DataType.BFLOAT16, "BFLOAT16"),
        (DataType.INT64, "INT64"),
        (DataType.UINT64, "UINT64"),
    ])
    def test_dtype_names(self, dtype, expected):
        assert get_dtype_name(dtype) == expected


# ============================================================================
# TaskArg
# ============================================================================

class TestTaskArg:
    def test_default_constructor(self):
        arg = TaskArg()
        assert arg is not None

    def test_make_tensor(self):
        arg = TaskArg.make_tensor(0xDEAD, (4, 8), DataType.FLOAT32)
        assert arg.kind == TaskArgKind.TENSOR
        assert arg.tensor_data == 0xDEAD
        assert arg.tensor_shapes == (4, 8)
        assert arg.tensor_ndims == 2
        assert arg.tensor_dtype == DataType.FLOAT32

    def test_make_scalar(self):
        arg = TaskArg.make_scalar(42)
        assert arg.kind == TaskArgKind.SCALAR
        assert arg.scalar == 42

    def test_nbytes(self):
        arg = TaskArg.make_tensor(0, (10, 20), DataType.FLOAT32)
        assert arg.nbytes() == 10 * 20 * 4

    def test_nbytes_int8(self):
        arg = TaskArg.make_tensor(0, (256,), DataType.INT8)
        assert arg.nbytes() == 256

    def test_tensor_shapes_setter(self):
        arg = TaskArg()
        arg.kind = TaskArgKind.TENSOR
        arg.tensor_shapes = (3, 5, 7)
        assert arg.tensor_ndims == 3
        assert arg.tensor_shapes == (3, 5, 7)

    def test_set_tensor_shape(self):
        arg = TaskArg.make_tensor(0, (10, 20, 30), DataType.INT32)
        arg.set_tensor_shape(1, 99)
        assert arg.tensor_shapes == (10, 99, 30)

    def test_max_dims(self):
        assert TASK_ARG_MAX_DIMS == 5
        # 5 dims should work
        arg = TaskArg.make_tensor(0, (1, 2, 3, 4, 5), DataType.INT32)
        assert arg.tensor_ndims == 5

    def test_exceed_max_dims(self):
        with pytest.raises((ValueError, RuntimeError)):
            TaskArg.make_tensor(0, (1, 2, 3, 4, 5, 6), DataType.INT32)

    def test_repr_tensor(self):
        arg = TaskArg.make_tensor(0x1000, (4, 8), DataType.FLOAT16)
        r = repr(arg)
        assert "TENSOR" in r
        assert "4" in r
        assert "8" in r
        assert "FLOAT16" in r

    def test_repr_scalar(self):
        arg = TaskArg.make_scalar(123)
        r = repr(arg)
        assert "SCALAR" in r
        assert "123" in r

    def test_kind_readwrite(self):
        arg = TaskArg()
        arg.kind = TaskArgKind.SCALAR
        assert arg.kind == TaskArgKind.SCALAR
        arg.kind = TaskArgKind.TENSOR
        assert arg.kind == TaskArgKind.TENSOR

    def test_tensor_dtype_readwrite(self):
        arg = TaskArg.make_tensor(0, (1,), DataType.FLOAT32)
        arg.tensor_dtype = DataType.INT64
        assert arg.tensor_dtype == DataType.INT64


# ============================================================================
# TaskArgArray
# ============================================================================

class TestTaskArgArray:
    def test_empty(self):
        arr = TaskArgArray()
        assert len(arr) == 0

    def test_append_and_len(self):
        arr = TaskArgArray()
        arr.append(TaskArg.make_scalar(1))
        arr.append(TaskArg.make_scalar(2))
        assert len(arr) == 2

    def test_getitem(self):
        arr = TaskArgArray()
        arr.append(TaskArg.make_scalar(42))
        assert arr[0].scalar == 42

    def test_getitem_reference_internal(self):
        arr = TaskArgArray()
        arr.append(TaskArg.make_tensor(0, (10,), DataType.FLOAT32))
        ref = arr[0]
        ref.tensor_data = 0xBEEF
        assert arr[0].tensor_data == 0xBEEF

    def test_getitem_out_of_range(self):
        arr = TaskArgArray()
        with pytest.raises((IndexError, RuntimeError)):
            arr[0]

    def test_clear(self):
        arr = TaskArgArray()
        arr.append(TaskArg.make_scalar(1))
        arr.clear()
        assert len(arr) == 0

    def test_ctypes_ptr_nonzero(self):
        arr = TaskArgArray()
        arr.append(TaskArg.make_scalar(99))
        ptr = arr.ctypes_ptr()
        assert isinstance(ptr, int)
        assert ptr != 0

    def test_ctypes_ptr_reads_correct_memory(self):
        """Verify that ctypes can read the memory at ctypes_ptr()."""
        arr = TaskArgArray()
        arr.append(TaskArg.make_scalar(0x1234_5678_9ABC_DEF0))

        ptr = arr.ctypes_ptr()
        # TaskArg is 48 bytes. First 4 bytes = kind (SCALAR = 1)
        kind_bytes = (ctypes.c_uint32).from_address(ptr)
        assert kind_bytes.value == 1  # TaskArgKind::SCALAR

        # Scalar value is at offset 8 (4 kind + 4 pad)
        scalar_bytes = (ctypes.c_uint64).from_address(ptr + 8)
        assert scalar_bytes.value == 0x1234_5678_9ABC_DEF0


# ============================================================================
# task_interface.py wrapper (torch integration)
# ============================================================================

class TestTaskInterfaceWrapper:
    def test_torch_dtype_to_datatype(self):
        from task_interface import torch_dtype_to_datatype
        import torch
        assert torch_dtype_to_datatype(torch.float32) == DataType.FLOAT32
        assert torch_dtype_to_datatype(torch.int8) == DataType.INT8
        assert torch_dtype_to_datatype(torch.bfloat16) == DataType.BFLOAT16

    def test_torch_dtype_unsupported(self):
        from task_interface import torch_dtype_to_datatype
        import torch
        with pytest.raises(KeyError):
            torch_dtype_to_datatype(torch.complex64)

    def test_make_tensor_arg(self):
        from task_interface import make_tensor_arg
        import torch
        t = torch.zeros(4, 8, dtype=torch.float32)
        arg = make_tensor_arg(t)
        assert arg.kind == TaskArgKind.TENSOR
        assert arg.tensor_data == t.data_ptr()
        assert arg.tensor_shapes == (4, 8)
        assert arg.tensor_dtype == DataType.FLOAT32
        assert arg.nbytes() == 4 * 8 * 4

    def test_make_scalar_arg_int(self):
        from task_interface import make_scalar_arg
        arg = make_scalar_arg(999)
        assert arg.kind == TaskArgKind.SCALAR
        assert arg.scalar == 999

    def test_make_scalar_arg_ctypes(self):
        from task_interface import make_scalar_arg
        arg = make_scalar_arg(ctypes.c_int64(42))
        assert arg.kind == TaskArgKind.SCALAR
        assert arg.scalar == 42

    def test_make_scalar_arg_float_ctypes(self):
        from task_interface import make_scalar_arg
        arg = make_scalar_arg(ctypes.c_float(1.5))
        assert arg.kind == TaskArgKind.SCALAR
        # Verify bit-cast: 1.5f = 0x3FC00000
        expected_bits = struct.unpack("I", struct.pack("f", 1.5))[0]
        assert arg.scalar == expected_bits
