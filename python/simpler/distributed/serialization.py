"""Wire serialization helpers shared by distributed dispatch components."""

from __future__ import annotations

import ctypes
import mmap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from simpler.task_interface import CallConfig, ContinuousTensor, DataType, TaskArgs, TensorArgType, get_element_size

from .proto import dispatch_pb2
from .tensor_pool import TensorPool
from .transport_backend import RxeDataPlaneClient, TransportBackendError, TransportUnavailable

_OUTPUT_TAGS = {
    TensorArgType.OUTPUT,
    TensorArgType.INOUT,
    TensorArgType.OUTPUT_EXISTING,
}

_REMOTE_OUTPUT_TAGS = {
    TensorArgType.OUTPUT,
    TensorArgType.OUTPUT_EXISTING,
}


@dataclass(frozen=True)
class RemoteTensorWriteback:
    tensor_index: int
    handle: dispatch_pb2.TensorHandle


def encode_config(config: CallConfig) -> bytes:
    cfg = dispatch_pb2.CallConfigWire(
        block_dim=int(config.block_dim),
        aicpu_thread_num=int(config.aicpu_thread_num),
        enable_l2_swimlane=bool(config.enable_l2_swimlane),
        enable_dump_tensor=bool(config.enable_dump_tensor),
        enable_pmu=int(config.enable_pmu),
        output_prefix=str(config.output_prefix),
    )
    return cfg.SerializeToString()


def decode_config(blob: bytes) -> CallConfig:
    cfg = CallConfig()
    if not blob:
        return cfg
    wire = dispatch_pb2.CallConfigWire()
    wire.ParseFromString(blob)
    cfg.block_dim = int(wire.block_dim)
    cfg.aicpu_thread_num = int(wire.aicpu_thread_num)
    cfg.enable_l2_swimlane = bool(wire.enable_l2_swimlane)
    cfg.enable_dump_tensor = bool(wire.enable_dump_tensor)
    cfg.enable_pmu = int(wire.enable_pmu)
    cfg.output_prefix = wire.output_prefix
    return cfg


def encode_task_args(args: Optional[TaskArgs]) -> tuple[list[dispatch_pb2.ContinuousTensorRef], list[int]]:
    if args is None:
        return [], []
    tensors = []
    for i in range(args.tensor_count()):
        tensor = args.tensor(i)
        tag = args.tag(i)
        tensors.append(
            dispatch_pb2.ContinuousTensorRef(
                data=int(tensor.data),
                shape=[int(x) for x in tensor.shapes[: int(tensor.ndims)]],
                dtype=int(tensor.dtype.value),
                tag=int(tag.value),
            )
        )
    scalars = [int(args.scalar(i)) for i in range(args.scalar_count())]
    return tensors, scalars


def decode_task_args(
    tensor_refs: Iterable[dispatch_pb2.ContinuousTensorRef],
    scalar_args: Iterable[int],
) -> TaskArgs:
    args = TaskArgs()
    for ref in tensor_refs:
        shape = tuple(int(x) for x in ref.shape)
        dtype = DataType(int(ref.dtype))
        tag = TensorArgType(int(ref.tag))
        args.add_tensor(ContinuousTensor.make(int(ref.data), shape, dtype), tag)
    for scalar in scalar_args:
        args.add_scalar(int(scalar))
    return args


def encode_tensor_ref(
    data: bytes,
    *,
    shape: Iterable[int],
    dtype: DataType,
    tag: TensorArgType,
    pool: TensorPool,
    ttl_ms: Optional[int] = None,
    force_handle: bool = False,
) -> dispatch_pb2.TensorRef:
    return pool.put_bytes(
        data,
        shape=shape,
        dtype=int(dtype.value),
        tag=int(tag.value),
        ttl_ms=ttl_ms,
        force_handle=force_handle,
    )


def decode_task_args_with_tensor_refs(
    tensor_refs: Iterable[dispatch_pb2.TensorRef],
    scalar_args: Iterable[int],
    pool: TensorPool,
) -> tuple[TaskArgs, list[object]]:
    args, keepalive, _ = decode_task_args_with_tensor_refs_and_writebacks(tensor_refs, scalar_args, pool)
    return args, keepalive


def decode_task_args_with_tensor_refs_and_writebacks(
    tensor_refs: Iterable[dispatch_pb2.TensorRef],
    scalar_args: Iterable[int],
    pool: TensorPool,
) -> tuple[TaskArgs, list[object], list[RemoteTensorWriteback]]:
    args = TaskArgs()
    keepalive: list[object] = []
    writebacks: list[RemoteTensorWriteback] = []
    for tensor_index, ref in enumerate(tensor_refs):
        shape = tuple(int(x) for x in ref.shape)
        dtype = DataType(int(ref.dtype))
        tag = TensorArgType(int(ref.tag))
        nbytes = _shape_nbytes(shape, dtype)
        remote_output = (
            tag in _REMOTE_OUTPUT_TAGS
            and ref.HasField("handle")
            and ref.handle.transport == "rxe"
            and ref.handle.node_id != pool.node_id
        )
        data = b"" if remote_output else pool.materialize_ref(ref)
        size = max(1, nbytes if remote_output else len(data))
        buf = mmap.mmap(-1, size)
        if data:
            buf.write(data)
        else:
            buf.write(b"\x00")
        keepalive.append(buf)
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        args.add_tensor(ContinuousTensor.make(ptr, shape, dtype), tag)
        if remote_output:
            writebacks.append(RemoteTensorWriteback(tensor_index=tensor_index, handle=ref.handle))
    for scalar in scalar_args:
        args.add_scalar(int(scalar))
    return args, keepalive, writebacks


def encode_output_tensor_refs(
    args: TaskArgs,
    pool: TensorPool,
    writebacks: Optional[Iterable[RemoteTensorWriteback]] = None,
) -> list[dispatch_pb2.TensorRef]:
    refs = []
    writeback_by_index = {item.tensor_index: item for item in (writebacks or [])}
    rxe_client = None
    for i in range(args.tensor_count()):
        tag = args.tag(i)
        if tag not in _OUTPUT_TAGS:
            continue
        tensor = args.tensor(i)
        data = ctypes.string_at(int(tensor.data), _tensor_nbytes(tensor))
        writeback = writeback_by_index.get(i)
        if writeback is not None:
            try:
                if writeback.handle.transport != "rxe":
                    raise RuntimeError(f"unsupported remote output transport {writeback.handle.transport!r}")
                rxe_client = rxe_client or RxeDataPlaneClient.from_env()
                local = ctypes.create_string_buffer(data, len(data))
                rxe_client.write_handle(writeback.handle, ctypes.addressof(local), len(data))
                rxe_client.fence()
                refs.append(
                    dispatch_pb2.TensorRef(
                        handle=writeback.handle,
                        shape=[int(x) for x in tensor.shapes[: int(tensor.ndims)]],
                        dtype=int(tensor.dtype.value),
                        tag=int(tag.value),
                    )
                )
                continue
            except (RuntimeError, TransportBackendError, TransportUnavailable):
                pass
        refs.append(
            pool.put_bytes(
                data,
                shape=[int(x) for x in tensor.shapes[: int(tensor.ndims)]],
                dtype=int(tensor.dtype.value),
                tag=int(tag.value),
            )
        )
    return refs


def _tensor_nbytes(tensor) -> int:  # noqa: ANN001
    nbytes = tensor.nbytes
    return int(nbytes() if callable(nbytes) else nbytes)


def _shape_nbytes(shape: Iterable[int], dtype: DataType) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * _dtype_nbytes(dtype)


def _dtype_nbytes(dtype: DataType) -> int:
    return int(get_element_size(dtype))
