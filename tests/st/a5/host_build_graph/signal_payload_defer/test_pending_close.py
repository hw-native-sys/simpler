# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Close must drain accepted cross-rank publications before releasing domains."""

import copy
from pathlib import Path

import pytest
import torch
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec, scene_level

from .test_signal_payload_defer import ELEMENTS, FIELDS
from .test_signal_payload_defer import TestSignalPayloadTwoRanks as PayloadScene

HERE = Path(__file__).resolve().parent


def compile_close_case(platform):
    spec = copy.deepcopy(PayloadScene.CALLABLE["callables"][0])
    for kernel in spec["incores"]:
        if kernel["func_id"] == 6:
            kernel["source"] = str(HERE / "kernels/aiv/delayed_publisher.cpp")
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (str(HERE), "pending_close", platform, _pto_isa_compile_cache_token())
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.parametrize(
    "ranks",
    [pytest.param(2, marks=pytest.mark.device_count(2)), pytest.param(4, marks=pytest.mark.device_count(4))],
)
@scene_level(3)
def test_close_waits_for_all_peer_payloads(st_platform, st_device_ids, ranks, tmp_path):
    assert len(st_device_ids) >= ranks
    params = dict(ranks=ranks, epoch=1, domain_tag=1, dirty=True, consumer_type=2, retained=False)
    scene = PayloadScene()
    actual = scene.generate_args(params)
    expected = scene.generate_args(params)
    scene.compute_golden(expected, params)
    # These mappings predate Worker.init's fork and remain caller-owned after close.
    for name in actual.tensor_names():
        getattr(actual, name).share_memory_()
    worker = Worker(
        level=3, platform=st_platform, runtime="host_build_graph", device_ids=st_device_ids[:ranks], num_sub_workers=0
    )
    callable_handle = worker.register(compile_close_case(st_platform))
    closed = False
    try:
        worker.init()
        seeds = {}
        for name in ("signal_seed", *(f"seed_{field}" for field in FIELDS)):
            value = getattr(actual, name)
            seeds[name] = worker.create_buffer(value.nbytes)
            torch.frombuffer(seeds[name].shm.buf, dtype=value.dtype, count=value.numel()).copy_(value.flatten())
        domains = []

        def orchestrate(orch, _args, config):
            specs = [
                CommBufferSpec(name=name, dtype="int32", count=ranks * 32, nbytes=ranks * 128)
                for name in ("arrived", "ready")
            ] + [
                CommBufferSpec(
                    name=name,
                    dtype="int32" if name == "routes" else "float32",
                    count=ranks * ELEMENTS,
                    nbytes=ranks * ELEMENTS * 4,
                )
                for name in FIELDS
            ]
            domain = orch.allocate_domain(
                name="pending_peer_payloads",
                members=list(range(ranks)),
                window_size=sum(spec.nbytes for spec in specs),
                buffers=specs,
                retain_after_run=True,
            )
            domains.append(domain)
            for rank in range(ranks):
                for name in ("arrived", "ready"):
                    orch.copy_to(domain[rank].buffers[name], seeds["signal_seed"])
                for name in FIELDS:
                    orch.copy_to(domain[rank].buffers[name], seeds[f"seed_{name}"])
            for rank in range(ranks):
                args = TaskArgs()
                for name in ("arrived", "ready"):
                    args.add_tensor(domain[rank].buffers[name].tensor((ranks, 32), DataType.INT32), TensorArgType.INOUT)
                for name in FIELDS:
                    dtype = DataType.INT32 if name == "routes" else DataType.FLOAT32
                    args.add_tensor(domain[rank].buffers[name].tensor((ranks, ELEMENTS), dtype), TensorArgType.INOUT)
                for name in (*FIELDS, "info"):
                    value = getattr(actual, f"info_{rank}" if name == "info" else f"out_{name}_{rank}")
                    dtype = DataType.INT32 if name in ("routes", "info") else DataType.FLOAT32
                    args.add_tensor(worker.make_tensor_arg(value, tuple(value.shape), dtype), TensorArgType.INOUT)
                for value in (domain.device_context(rank), rank, ranks, 1, 1, 2):
                    args.add_scalar(value)
                orch.submit_next_level(callable_handle, args, config, worker=rank)
            domain.release()

        handle = worker.submit(orchestrate, config=CallConfig())
        assert not handle.done, "the accepted run must still be outstanding at close"
        worker.close()
        closed = True
        assert handle.done, "close returned before the accepted run finished"
        handle.result()
        assert len(domains) == 1 and domains[0].released and domains[0].freed
        scene.compare_outputs(actual, expected, actual.tensor_names(), params)
        (tmp_path / "pending_peer_close.txt").write_text(f"{ranks} ranks: all payloads consumed and domain freed\n")
    finally:
        if not closed:
            worker.close()
