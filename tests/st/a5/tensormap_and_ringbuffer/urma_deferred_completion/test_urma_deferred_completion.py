#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-rank URMA transfers whose successors consume deferred outputs."""

from contextlib import nullcontext

import pytest
import torch
from simpler import UrmaConfig
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

NRANKS = 2
DTYPE_NBYTES = 4
CONSUMER_TILE_ELEMS = 64 * 128
SMALL_ELEMS = 128 * 128
CHUNKED_ELEMS = 256 * 1024 * 1024 // DTYPE_NBYTES + 8192
URMA_DATA_OFFSET_NBYTES = 256
DISJOINT_PUT_MODE = 2
BENCHMARK_PUT_MODE = 3
TWO_JETTY_PUT_MODE = 4
BATCH_PAYLOAD_ELEMS = 1024
BATCH_GUARD_ELEMS = 512
BATCH_REQUEST_COUNT = 5
BENCHMARK_REQUEST_COUNT = 33
BENCHMARK_WINDOW_SLOTS = 64
BENCHMARK_PAYLOAD_BYTES = (4096, 32768)
BENCHMARK_BATCH_SIZES = (0, 1, 4, 8, 16, 32)


def _batch_payload_ranges(elem_count, mode):
    if mode == BENCHMARK_PUT_MODE:
        count = elem_count // BENCHMARK_WINDOW_SLOTS
        for i in range(BENCHMARK_REQUEST_COUNT):
            offset = BATCH_GUARD_ELEMS + i * (count + BATCH_GUARD_ELEMS)
            yield slice(offset, offset + count)
        return
    tail_elems = (BATCH_REQUEST_COUNT - 1) * BATCH_PAYLOAD_ELEMS + (BATCH_REQUEST_COUNT + 1) * BATCH_GUARD_ELEMS
    offset = BATCH_GUARD_ELEMS
    count = elem_count - tail_elems if elem_count == CHUNKED_ELEMS else BATCH_PAYLOAD_ELEMS
    for _ in range(BATCH_REQUEST_COUNT):
        yield slice(offset, offset + count)
        offset += count + BATCH_GUARD_ELEMS
        count = BATCH_PAYLOAD_ELEMS


def urma_deferred_completion_orch_fn(orch, callables, task_args, config):
    elem_count = task_args.in_0.numel()
    input_nbytes = elem_count * DTYPE_NBYTES
    benchmark = task_args.mode == BENCHMARK_PUT_MODE
    disjoint_put = task_args.mode in (DISJOINT_PUT_MODE, BENCHMARK_PUT_MODE, TWO_JETTY_PUT_MODE)
    config.enable_urma_batch = disjoint_put and bool(task_args.repeats_or_batch_size)
    allocate = orch.get_or_allocate_domain if benchmark else orch.allocate_domain
    handle = allocate(
        workspace_engines={"urma": UrmaConfig(jetties_per_aiv=2)}
        if task_args.mode == TWO_JETTY_PUT_MODE
        else ("urma",),
        name="urma_deferred_completion",
        members=list(range(NRANKS)),
        retain_after_run=benchmark,
        window_size=max(URMA_DATA_OFFSET_NBYTES + 2 * input_nbytes, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(name="urma_reserved", dtype="int32", count=64, nbytes=URMA_DATA_OFFSET_NBYTES),
            CommBufferSpec(name="input_window", dtype="float32", count=elem_count, nbytes=input_nbytes),
            CommBufferSpec(name="receive_window", dtype="float32", count=elem_count, nbytes=input_nbytes),
        ],
    )
    # Reuse the measured queues across rounds and release them at Worker.close().
    with nullcontext() if benchmark else handle:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            host_source = _rehosted_buffer_for(task_args, source)
            orch.copy_to(handle[rank].buffers["input_window"], host_source)
            if disjoint_put:
                # The peer's distinct seed remains visible in the guard regions.
                orch.copy_to(handle[rank].buffers["receive_window"], host_source)
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((elem_count,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                domain.buffers["receive_window"].tensor((elem_count,), DataType.FLOAT32), TensorArgType.INOUT
            )
            for name in ("out", "result"):
                args.add_tensor(
                    _rehosted_ref_for(task_args, getattr(task_args, f"{name}_{rank}")), TensorArgType.OUTPUT_EXISTING
                )
            args.add_scalar(handle.device_context(rank, workspace_engines=("urma",)))
            args.add_scalar(elem_count)
            args.add_scalar(task_args.mode)
            args.add_scalar(task_args.repeats_or_batch_size)
            orch.submit_next_level(callables.urma_deferred_completion, args, config, worker=rank)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class _UrmaRegression(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_deferred_completion_orch_fn,
        "callables": [
            {
                "name": "urma_deferred_completion",
                "orchestration": {
                    "source": "kernels/orchestration/urma_deferred_completion_orch.cpp",
                    "function_name": "urma_deferred_completion_orchestration",
                    "config_name": "urma_deferred_completion_orchestration_config",
                    "signature": [D.IN, D.INOUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_tget_async.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
                    },
                ],
            }
        ],
    }
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        from simpler_setup import Scalar  # noqa: PLC0415

        specs = []
        elem_count = params["elems"]
        for rank in range(NRANKS):
            inp = torch.arange(elem_count, dtype=torch.int64).remainder_(251).to(torch.float32)
            inp.mul_(0.1).add_(rank * 100)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(elem_count, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(elem_count, dtype=torch.float32)),
                ]
            )
        specs.extend([Scalar("mode", params["mode"]), Scalar("repeats_or_batch_size", params["repeats_or_batch_size"])])
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            out = getattr(args, f"out_{rank}")
            if params["mode"] in (DISJOINT_PUT_MODE, BENCHMARK_PUT_MODE, TWO_JETTY_PUT_MODE):
                out.copy_(getattr(args, f"in_{1 - rank}"))
                source = getattr(args, f"in_{rank}")
                for region in _batch_payload_ranges(out.numel(), params["mode"]):
                    out[region].copy_(source[region])
            else:
                source_rank = rank if params["mode"] else 1 - rank
                out.copy_(getattr(args, f"in_{source_rank}"))
            torch.add(out, 1.0, out=getattr(args, f"result_{rank}"))


def _case(name, mode=0, repeats=1, elems=SMALL_ELEMS, *, batch_size=None, manual=False):
    if elems <= 0 or elems % CONSUMER_TILE_ELEMS:
        raise ValueError("URMA regression consumer requires a positive multiple of 8192 elements")
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "manual": True,
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {
                "mode": mode,
                "repeats_or_batch_size": repeats if batch_size is None else batch_size,
                "elems": elems,
            },
            "manual": manual,
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaGet(_UrmaRegression):
    CASES = _case("peer_fetch")


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaPut(_UrmaRegression):
    CASES = _case("peer_put", mode=1)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaRepeatedFetch(_UrmaRegression):
    CASES = _case("repeated_fetch", repeats=16)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaRepeatedPut(_UrmaRegression):
    CASES = _case("repeated_put", mode=1, repeats=16)


@pytest.mark.skip(reason="A5 chunked URMA fetch hit a device scheduler timeout in CI run 41609948645")
@pytest.mark.coverage_skip(
    reason="A5 chunked URMA fetch hit a device scheduler timeout in CI run 41609948645",
    platforms=["a2a3", "a5"],
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaChunkedFetch(_UrmaRegression):
    CASES = _case("chunked_fetch", elems=CHUNKED_ELEMS)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaChunkedPut(_UrmaRegression):
    CASES = _case("chunked_put", mode=1, elems=CHUNKED_ELEMS)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaDisjointPut(_UrmaRegression):
    CASES = (
        _case("immediate", mode=DISJOINT_PUT_MODE, batch_size=0)
        + _case("batch_one", mode=DISJOINT_PUT_MODE, batch_size=1)
        + _case("batch_two_with_tail", mode=DISJOINT_PUT_MODE, batch_size=2)
        + _case("two_jetties_with_tail", mode=TWO_JETTY_PUT_MODE, batch_size=2)
        + _case("batch_three_with_tail", mode=DISJOINT_PUT_MODE, batch_size=3)
        + _case("explicit_end", mode=DISJOINT_PUT_MODE, batch_size=2**32 - 1)
        + _case("chunked_batch", mode=DISJOINT_PUT_MODE, batch_size=2, elems=CHUNKED_ELEMS, manual=True)
    )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaPutBenchmark(_UrmaRegression):
    CASES = [
        case
        for nbytes in BENCHMARK_PAYLOAD_BYTES
        for batch_size in BENCHMARK_BATCH_SIZES
        for case in _case(
            f"put_{nbytes // 1024}k_{f'batch{batch_size}' if batch_size else 'immediate'}",
            mode=BENCHMARK_PUT_MODE,
            batch_size=batch_size,
            elems=BENCHMARK_WINDOW_SLOTS * (nbytes // DTYPE_NBYTES),
            manual=True,
        )
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
