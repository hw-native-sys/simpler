"""Wire serialization helpers shared by distributed dispatch components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from simpler.task_interface import CallConfig, ContinuousTensor, DataType, TaskArgs, TensorArgType

from .proto import dispatch_pb2


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
