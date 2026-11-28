#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Peer URMA transfers whose successors consume deferred outputs."""

import pytest
import torch
from simpler.comm_domain import UrmaConfig
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
GATHER_SCATTER_MODE = 2
NATIVE_WAIT_MODE = 4
MULTI_PEER_MODE = 6
DESCRIPTOR_MODES = (GATHER_SCATTER_MODE, NATIVE_WAIT_MODE, MULTI_PEER_MODE)
# Two cache lines for descriptors and one for the rendezvous per lane.
DESCRIPTOR_PREFIX_BYTES = 2 * 3 * 64


def urma_deferred_completion_orch_fn(orch, callables, task_args, config):
    elem_count = task_args.in_0.numel()
    input_nbytes = elem_count * DTYPE_NBYTES
    ranks = 3 if task_args.mode == MULTI_PEER_MODE else NRANKS
    descriptor_transfer = task_args.mode in DESCRIPTOR_MODES
    prefix_bytes = DESCRIPTOR_PREFIX_BYTES if descriptor_transfer else URMA_DATA_OFFSET_NBYTES
    buffers = [
        CommBufferSpec(name="urma_reserved", dtype="int32", count=prefix_bytes // 4, nbytes=prefix_bytes),
        CommBufferSpec(name="input_window", dtype="float32", count=elem_count, nbytes=input_nbytes),
        CommBufferSpec(name="receive_window", dtype="float32", count=elem_count, nbytes=input_nbytes),
    ]
    if descriptor_transfer:
        buffers.append(CommBufferSpec(name="readback", dtype="float32", count=elem_count, nbytes=input_nbytes))
    with orch.allocate_domain(
        workspace_engines={"urma": UrmaConfig(jetties_per_aiv=1)},
        name="urma_deferred_completion",
        workers=list(range(ranks)),
        window_size=max(sum(buf.nbytes for buf in buffers), 4 * 1024 * 1024),
        buffers=buffers,
    ) as handle:
        for rank in range(ranks):
            if task_args.mode in (NATIVE_WAIT_MODE, MULTI_PEER_MODE):
                orch.copy_to_domain(handle, rank, bytes(prefix_bytes), buffer="urma_reserved")
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
        for rank in range(ranks):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((elem_count,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                domain.buffers["receive_window"].tensor((elem_count,), DataType.FLOAT32), TensorArgType.INOUT
            )
            if descriptor_transfer:
                output = domain.buffers["readback"].tensor((elem_count,), DataType.FLOAT32)
            else:
                output = _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}"))
            args.add_tensor(output, TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(handle.device_context(rank, workspace_engines=("urma",)))
            args.add_scalar(elem_count)
            args.add_scalar(task_args.mode)
            args.add_scalar(task_args.repeats)
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
        ranks = 3 if params["mode"] == MULTI_PEER_MODE else NRANKS
        for rank in range(ranks):
            inp = torch.arange(elem_count, dtype=torch.int64).remainder_(251).to(torch.float32)
            inp.mul_(0.1).add_(rank * 100)
            specs.append(TensorArg(f"in_{rank}", inp))
            if params["mode"] not in DESCRIPTOR_MODES:
                specs.append(TensorArg(f"out_{rank}", torch.zeros(elem_count, dtype=torch.float32)))
            specs.append(TensorArg(f"result_{rank}", torch.zeros(elem_count, dtype=torch.float32)))
        specs.extend([Scalar("mode", params["mode"]), Scalar("repeats", params["repeats"])])
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        ranks = 3 if params["mode"] == MULTI_PEER_MODE else NRANKS
        for rank in range(ranks):
            source_rank = rank if params["mode"] else 1 - rank
            expected = getattr(args, f"in_{source_rank}")
            if params["mode"] == MULTI_PEER_MODE:
                expected = expected.clone().view(2, -1)
                half_lane = expected.shape[1] // 2
                other = getattr(args, f"in_{(rank + 2) % ranks}").view_as(expected)
                expected[:, half_lane:] = other[:, half_lane:]
                expected = expected.flatten()
            if params["mode"] not in DESCRIPTOR_MODES:
                getattr(args, f"out_{rank}").copy_(expected)
            getattr(args, f"result_{rank}").copy_(expected + 1.0)


def _case(name, mode=0, repeats=1, elems=SMALL_ELEMS, *, manual=True):
    if elems <= 0 or elems % CONSUMER_TILE_ELEMS:
        raise ValueError("URMA regression consumer requires a positive multiple of 8192 elements")
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "manual": manual,
            "config": {"device_count": 3 if mode == MULTI_PEER_MODE else NRANKS, "num_sub_workers": 0},
            "params": {"mode": mode, "repeats": repeats, "elems": elems},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaGet(_UrmaRegression):
    CASES = _case("peer_fetch")


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaPut(_UrmaRegression):
    CASES = _case("peer_put", mode=1)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaGatherScatter(_UrmaRegression):
    CASES = _case("gather_put_scatter_get", mode=GATHER_SCATTER_MODE, repeats=2, manual=False)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaNativeWait(_UrmaRegression):
    CASES = _case("descriptor_native_wait", mode=NATIVE_WAIT_MODE, repeats=2)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiPeerSession(_UrmaRegression):
    CASES = _case("descriptor_multi_peer", mode=MULTI_PEER_MODE, repeats=2)


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


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
