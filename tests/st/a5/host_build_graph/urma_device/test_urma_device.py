#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from struct import pack

from simpler.comm_domain import UrmaConfig
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for
from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_deferred_completion import (
    _case,
    _UrmaRegression,
)


def _transfer(orch, callables, inputs, config):
    count = inputs.in_0.numel()
    regions = [
        CommBufferSpec(name="src", dtype="float32", count=count, nbytes=count * 4),
        CommBufferSpec(name="dst", dtype="float32", count=count, nbytes=count * 4),
    ]
    with orch.allocate_domain(
        name="hbg_urma",
        workers=[0, 1],
        window_size=max(4 * 1024 * 1024, 8 * count),
        buffers=regions,
        workspace_engines={"urma": UrmaConfig(jetties_per_aiv=1)},
    ) as domain:
        guard = pack("=32f", *([-77.0] * 32))
        for rank in (0, 1):
            orch.copy_to(domain[rank].buffers["src"], _rehosted_buffer_for(inputs, getattr(inputs, f"in_{rank}")))
            for offset in (0, count * 4 - len(guard)):
                orch.copy_to_domain(domain, rank, guard, buffer="dst", offset=offset)

        def submit(rank, mode, predecessor=None):
            args = TaskArgs()
            member = domain[rank]
            args.add_tensor(member.buffers["src"].tensor((count,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(member.buffers["dst"].tensor((count,), DataType.FLOAT32), TensorArgType.INOUT)
            for name in ("out", "result"):
                output = _rehosted_ref_for(inputs, getattr(inputs, f"{name}_{rank}"))
                args.add_tensor(output, TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(domain.device_context(rank, workspace_engines=("urma",)))
            args.add_scalar(count)
            args.add_scalar(mode)
            args.add_scalar(1)
            if predecessor is not None:
                args.add_dep(predecessor)
            return orch.submit_next_level(callables.urma_deferred_completion, args, config, worker=rank)

        for rank in (0, 1):
            if inputs.mode == 1:
                # Complete the issuing Run before a second Run reads the peer's destination.
                submit(rank, 2, submit(rank, 1))
            else:
                submit(rank, 0)


@scene_test(level=3, runtime="host_build_graph")
class _HbgUrmaDevice(SceneTestCase):
    CALLABLE = deepcopy(_UrmaRegression.CALLABLE)
    CALLABLE["orchestration"] = _transfer
    _call = CALLABLE["callables"][0]
    _root = Path(__file__).parent
    _call["orchestration"].update(
        source=str(_root / "kernels/orchestration/transfer.cpp"),
        function_name="aicpu_orchestration_entry",
        config_name="aicpu_orchestration_config",
    )
    _call["incores"][0]["source"] = str(_root / "kernels/aiv/transfer.cpp")
    _call["incores"][1]["source"] = str(
        _root.parents[1] / "tensormap_and_ringbuffer/urma_deferred_completion/kernels/aiv/kernel_consumer.cpp"
    )
    RTOL = _UrmaRegression.RTOL
    ATOL = _UrmaRegression.ATOL
    generate_args = _UrmaRegression.generate_args

    def compute_golden(self, args, params):
        _UrmaRegression.compute_golden(self, args, params)
        for rank in (0, 1):
            for name, guard in (("out", -77.0), ("result", -76.0)):
                tensor = getattr(args, f"{name}_{rank}")
                tensor[:32].fill_(guard)
                tensor[-32:].fill_(guard)


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaGet(_HbgUrmaDevice):
    CASES = _case("peer_fetch")


@scene_test(level=3, runtime="host_build_graph")
class TestHbgUrmaPut(_HbgUrmaDevice):
    CASES = _case("peer_put", mode=1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
