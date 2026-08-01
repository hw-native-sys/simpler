# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-local L3 callbacks used by the A3 MPI compute + TLOAD smoke."""

from __future__ import annotations

from .task_interface import CallConfig, DataType, TaskArgs, Tensor, TensorArgType

_SMOKE_COUNT = 256


def _digest_from_scalars(args: TaskArgs, start: int) -> bytes:
    return b"".join(int(args.scalar(start + i)).to_bytes(8, "little") for i in range(4))


def remote_compute_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit one local vector-add task from an MPI L3 rank to its selected L2."""
    from .remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("remote compute task expects domain_id, local_worker_id, and four digest scalars")
    domain_id = int(args.scalar(0))
    local_worker_id = int(args.scalar(1))
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    context = orch.get_global_domain(domain_id)[local_worker_id]

    chip_args = TaskArgs()
    for buffer_name in ("lhs", "rhs"):
        chip_args.add_tensor(
            Tensor.make(
                data=context.buffer_ptrs[buffer_name],
                shapes=(_SMOKE_COUNT,),
                dtype=DataType.FLOAT32,
                child_memory=True,
            ),
            TensorArgType.INPUT,
        )
    chip_args.add_tensor(
        Tensor.make(
            data=context.buffer_ptrs["input"],
            shapes=(_SMOKE_COUNT,),
            dtype=DataType.FLOAT32,
            child_memory=True,
        ),
        TensorArgType.OUTPUT_EXISTING,
    )
    orch.submit_next_level(chip_handle, chip_args, cfg, worker=local_worker_id)


def remote_rank_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the peer-TLOAD kernel from an MPI L3 rank to its selected L2."""
    from .remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("global TLOAD remote task expects domain_id, local_worker_id, and four digest scalars")
    domain_id = int(args.scalar(0))
    local_worker_id = int(args.scalar(1))
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    context = orch.get_global_domain(domain_id)[local_worker_id]

    chip_args = TaskArgs()
    chip_args.add_tensor(
        Tensor.make(
            data=context.buffer_ptrs["input"],
            shapes=(_SMOKE_COUNT,),
            dtype=DataType.FLOAT32,
            child_memory=True,
        ),
        TensorArgType.INPUT,
    )
    chip_args.add_tensor(
        Tensor.make(
            data=context.buffer_ptrs["result"],
            shapes=(_SMOKE_COUNT,),
            dtype=DataType.FLOAT32,
            child_memory=True,
        ),
        TensorArgType.OUTPUT_EXISTING,
    )
    chip_args.add_scalar(context.domain_size)
    chip_args.add_scalar(context.device_ctx)
    orch.submit_next_level(chip_handle, chip_args, cfg, worker=local_worker_id)


def remote_compute_group_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Drive every L2 owned by one MPI L3 rank with distinct domain buffers."""
    from .remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("MPI compute group task expects domain_id, local worker count, and four digest scalars")
    domain_id = int(args.scalar(0))
    local_worker_count = int(args.scalar(1))
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    domain = orch.get_global_domain(domain_id)
    group_args = []
    workers = []
    for local_worker_id in range(local_worker_count):
        context = domain[local_worker_id]
        chip_args = TaskArgs()
        for buffer_name in ("lhs", "rhs"):
            chip_args.add_tensor(
                Tensor.make(
                    data=context.buffer_ptrs[buffer_name],
                    shapes=(_SMOKE_COUNT,),
                    dtype=DataType.FLOAT32,
                    child_memory=True,
                ),
                TensorArgType.INPUT,
            )
        chip_args.add_tensor(
            Tensor.make(
                data=context.buffer_ptrs["input"],
                shapes=(_SMOKE_COUNT,),
                dtype=DataType.FLOAT32,
                child_memory=True,
            ),
            TensorArgType.OUTPUT_EXISTING,
        )
        group_args.append(chip_args)
        workers.append(local_worker_id)
    orch.submit_next_level_group(chip_handle, group_args, cfg, workers=workers)


def remote_rank_group_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Drive peer TLOAD on every L2 owned by one MPI L3 rank."""
    from .remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("MPI TLOAD group task expects domain_id, local worker count, and four digest scalars")
    domain_id = int(args.scalar(0))
    local_worker_count = int(args.scalar(1))
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    domain = orch.get_global_domain(domain_id)
    group_args = []
    workers = []
    for local_worker_id in range(local_worker_count):
        context = domain[local_worker_id]
        chip_args = TaskArgs()
        chip_args.add_tensor(
            Tensor.make(
                data=context.buffer_ptrs["input"],
                shapes=(_SMOKE_COUNT,),
                dtype=DataType.FLOAT32,
                child_memory=True,
            ),
            TensorArgType.INPUT,
        )
        chip_args.add_tensor(
            Tensor.make(
                data=context.buffer_ptrs["result"],
                shapes=(_SMOKE_COUNT,),
                dtype=DataType.FLOAT32,
                child_memory=True,
            ),
            TensorArgType.OUTPUT_EXISTING,
        )
        chip_args.add_scalar(context.domain_size)
        chip_args.add_scalar(context.device_ctx)
        group_args.append(chip_args)
        workers.append(local_worker_id)
    orch.submit_next_level_group(chip_handle, group_args, cfg, workers=workers)
