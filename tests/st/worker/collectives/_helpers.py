# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared helpers for collective scene tests.

Provides comm-window scratch-parameter computation and orch-function
building so each collective test (allreduce, allgather, reduce_scatter,
broadcast, all_to_all) can reuse the same domain-allocation pattern.
"""

from __future__ import annotations

import ctypes

import torch
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, Tensor, TensorArgType

from simpler_setup import Tensor as STensor
from simpler_setup.scene_test import TaskArgsBuilder
from simpler_setup.torch_interop import make_tensor_arg

# ---------------------------------------------------------------------------
# Allreduce constants (must match kernel COUNT)
# ---------------------------------------------------------------------------
ALLREDUCE_COUNT = 256
ALLREDUCE_DTYPE_NBYTES = 4  # float32
ALLREDUCE_MAX_RANKS = 16

# Other collectives: 64 floats per rank, float32
COUNT_PER_RANK = 64
DTYPE_NBYTES = 4
MAX_RANKS = 16
SIGNAL_TAIL_NBYTES = MAX_RANKS * DTYPE_NBYTES  # one int32 slot per potential rank


# ---------------------------------------------------------------------------
# Allreduce scratch-params (per mode, per nranks)
# ---------------------------------------------------------------------------

_ALLREDUCE_MODE_NAMES = ["onephase", "twophase", "ring", "bidirectional_ring", "ibing"]


def _allreduce_scratch_params(mode: str, nranks: int) -> tuple[int, int, int]:
    """Compute (float_elems, scratch_nbytes, window_size) for an allreduce mode."""
    if mode == "onephase":
        float_elems = ALLREDUCE_COUNT
        scratch_nbytes = float_elems * ALLREDUCE_DTYPE_NBYTES + ALLREDUCE_MAX_RANKS * ALLREDUCE_DTYPE_NBYTES
    elif mode == "twophase":
        chunk_elems = ALLREDUCE_COUNT // nranks
        float_elems = nranks * chunk_elems
        scratch_nbytes = float_elems * ALLREDUCE_DTYPE_NBYTES + 2 * ALLREDUCE_MAX_RANKS * ALLREDUCE_DTYPE_NBYTES
    elif mode == "ring":
        chunk_elems = ALLREDUCE_COUNT // nranks
        float_elems = (nranks + 1) * chunk_elems
        scratch_nbytes = (
            float_elems * ALLREDUCE_DTYPE_NBYTES + 2 * (nranks - 1) * ALLREDUCE_MAX_RANKS * ALLREDUCE_DTYPE_NBYTES
        )
    elif mode == "bidirectional_ring":
        subchunk_elems = ALLREDUCE_COUNT // (2 * nranks)
        float_elems = 2 * nranks * subchunk_elems  # = ALLREDUCE_COUNT
        scratch_nbytes = (
            float_elems * ALLREDUCE_DTYPE_NBYTES + (2 * (nranks - 1) + 1) * ALLREDUCE_MAX_RANKS * ALLREDUCE_DTYPE_NBYTES
        )
    elif mode == "ibing":
        chunk_elems = ALLREDUCE_COUNT // nranks
        float_elems = (nranks + 2) * chunk_elems
        scratch_nbytes = (
            float_elems * ALLREDUCE_DTYPE_NBYTES + (2 * (nranks - 1) + 1) * ALLREDUCE_MAX_RANKS * ALLREDUCE_DTYPE_NBYTES
        )
    else:
        raise ValueError(f"Unsupported allreduce mode: {mode!r}")
    window_size = max(scratch_nbytes, 4 * 1024)
    return float_elems, scratch_nbytes, window_size


# ---------------------------------------------------------------------------
# Allreduce orch function (shared by all modes)
# ---------------------------------------------------------------------------


def allreduce_orch_fn(orch, callables, task_args, config):
    """L3 orch: allocate domain, submit per-rank allreduce tasks.

    Reads nranks and mode_id from task_args scalars. Selects the
    ChipCallable by mode name (e.g. ``allreduce_onephase``).
    """
    nranks = int(task_args.nranks.value)
    if not (2 <= nranks <= ALLREDUCE_MAX_RANKS):
        raise ValueError(f"allreduce nranks must be between 2 and {ALLREDUCE_MAX_RANKS}, got {nranks}")
    mode_id = int(task_args.mode_id.value)
    if not (0 <= mode_id < len(_ALLREDUCE_MODE_NAMES)):
        raise ValueError(f"invalid allreduce mode_id: {mode_id}")
    mode = _ALLREDUCE_MODE_NAMES[mode_id]

    # ibing is only supported for P=2
    if mode == "ibing" and nranks != 2:
        raise ValueError(f"ibing mode is only supported for nranks=2, got nranks={nranks}")
    # Chunked modes require ALLREDUCE_COUNT divisible by nranks (or 2*nranks).
    if mode in ("twophase", "ring", "ibing") and ALLREDUCE_COUNT % nranks != 0:
        raise ValueError(f"ALLREDUCE_COUNT={ALLREDUCE_COUNT} must be divisible by nranks={nranks} for {mode} mode")
    if mode == "bidirectional_ring" and ALLREDUCE_COUNT % (2 * nranks) != 0:
        raise ValueError(
            f"ALLREDUCE_COUNT={ALLREDUCE_COUNT} must be divisible by 2*nranks={2 * nranks} for {mode} mode"
        )

    chip = getattr(callables, f"allreduce_{mode}")
    float_elems, scratch_nbytes, window_size = _allreduce_scratch_params(mode, nranks)

    with orch.allocate_domain(
        name="default",
        workers=list(range(nranks)),
        window_size=window_size,
        buffers=[CommBufferSpec(name="scratch", dtype="float32", count=float_elems, nbytes=scratch_nbytes)],
    ) as handle:
        for i in range(nranks):
            domain = handle[i]
            chip_args = TaskArgs()
            chip_args.add_tensor(make_tensor_arg(getattr(task_args, f"in_{i}")), TensorArgType.INPUT)
            chip_args.add_tensor(make_tensor_arg(getattr(task_args, f"out_{i}")), TensorArgType.OUTPUT_EXISTING)
            chip_args.add_tensor(
                Tensor.make(
                    data=domain.buffer_ptrs["scratch"],
                    shapes=(float_elems,),
                    dtype=DataType.FLOAT32,
                    child_memory=True,
                ),
                TensorArgType.INOUT,
            )
            chip_args.add_scalar(domain.domain_size)
            chip_args.add_scalar(domain.device_ctx)
            orch.submit_next_level(chip, chip_args, config, worker=i)


# ---------------------------------------------------------------------------
# Allreduce golden
# ---------------------------------------------------------------------------


def allreduce_expected_output(nranks: int) -> list[float]:
    """output[i] = nranks*i + 100*nranks*(nranks-1)//2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(ALLREDUCE_COUNT)]


# ---------------------------------------------------------------------------
# Generic collective orch helpers
# ---------------------------------------------------------------------------


def generic_collective_orch_fn(
    orch,
    callables,
    task_args,
    config,
    *,
    chip_name: str,
    float_elems: int,
    scratch_nbytes: int,
    window_size: int,
    extra_scalars: list | None = None,
):
    """Generic L3 orch for single-mode collectives (allgather, reduce_scatter, broadcast, all_to_all).

    Reads nranks from ``task_args.nranks`` (Scalar). Allocates a comm domain
    and submits the ChipCallable named ``chip_name`` for each rank.

    Each rank's input/output tensors are named ``in_<i>`` / ``out_<i>``.
    Optional ``extra_scalars`` are appended after ``domain_size`` and
    ``device_ctx`` scalars (e.g. broadcast needs ``root``).
    """
    nranks = int(task_args.nranks.value)
    if not (2 <= nranks <= MAX_RANKS):
        raise ValueError(f"collective nranks must be between 2 and {MAX_RANKS}, got {nranks}")
    chip = getattr(callables, chip_name)
    extras = extra_scalars or []

    with orch.allocate_domain(
        name="default",
        workers=list(range(nranks)),
        window_size=window_size,
        buffers=[CommBufferSpec(name="scratch", dtype="float32", count=float_elems, nbytes=scratch_nbytes)],
    ) as handle:
        for i in range(nranks):
            domain = handle[i]
            chip_args = TaskArgs()
            chip_args.add_tensor(make_tensor_arg(getattr(task_args, f"in_{i}")), TensorArgType.INPUT)
            chip_args.add_tensor(make_tensor_arg(getattr(task_args, f"out_{i}")), TensorArgType.OUTPUT_EXISTING)
            chip_args.add_tensor(
                Tensor.make(
                    data=domain.buffer_ptrs["scratch"],
                    shapes=(float_elems,),
                    dtype=DataType.FLOAT32,
                    child_memory=True,
                ),
                TensorArgType.INOUT,
            )
            chip_args.add_scalar(domain.domain_size)
            for s in extras:
                chip_args.add_scalar(s)
            chip_args.add_scalar(domain.device_ctx)
            orch.submit_next_level(chip, chip_args, config, worker=i)


# ---------------------------------------------------------------------------
# Common golden functions
# ---------------------------------------------------------------------------


def allgather_expected_output(nranks: int) -> list[float]:
    """Rank-ordered concatenation of all inputs: output[r*C+i] = r*100 + i."""
    return [float(r * 100 + i) for r in range(nranks) for i in range(COUNT_PER_RANK)]


def reduce_scatter_expected_output(nranks: int, dest: int) -> list[float]:
    """output[j] = sum_r (dest*C+j + r*100) = nranks*(dest*C+j) + 100*nranks*(nranks-1)/2."""
    return [
        float(nranks * (dest * COUNT_PER_RANK + j) + 100 * nranks * (nranks - 1) // 2) for j in range(COUNT_PER_RANK)
    ]


def broadcast_expected_output(root: int) -> list[float]:
    """Every rank receives root's payload: output[i] = root*100 + i."""
    return [float(root * 100 + i) for i in range(COUNT_PER_RANK)]


def all_to_all_expected_output(nranks: int, rank: int) -> list[float]:
    """output[src*C + j] = src*1000 + rank*100 + j."""
    return [float(src * 1000 + rank * 100 + j) for src in range(nranks) for j in range(COUNT_PER_RANK)]


# ---------------------------------------------------------------------------
# generate_args helpers
# ---------------------------------------------------------------------------


def make_allreduce_args(nranks: int, mode_id: int) -> TaskArgsBuilder:
    """Build per-rank input/output tensors + nranks/mode_id scalars.

    input[rank][i] = i + rank*100. Output initially zeros.
    """
    from simpler_setup import Scalar as SScalar  # noqa: PLC0415

    builder_specs = []
    for rank in range(nranks):
        inp = torch.tensor(
            [i + rank * 100 for i in range(ALLREDUCE_COUNT)],
            dtype=torch.float32,
        ).share_memory_()
        out = torch.zeros(ALLREDUCE_COUNT, dtype=torch.float32).share_memory_()
        builder_specs.append(STensor(f"in_{rank}", inp))
        builder_specs.append(STensor(f"out_{rank}", out))
    builder_specs.append(SScalar("nranks", ctypes.c_int64(nranks)))
    builder_specs.append(SScalar("mode_id", ctypes.c_int64(mode_id)))
    return TaskArgsBuilder(*builder_specs)
