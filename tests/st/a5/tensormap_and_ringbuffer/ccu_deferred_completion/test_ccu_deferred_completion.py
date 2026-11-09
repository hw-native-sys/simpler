# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real embedded CCU modules and deferred AIV successors on two A5 ranks."""

import os
import shlex
import struct
import subprocess
from contextlib import ExitStack
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CcuFunction, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[4]
_RUNTIME = "tensormap_and_ringbuffer"
_ELEMS = 2048
_RANKS = 2


def _compile(platform, directory, *, rejection_probe=False):
    module = directory / "ccu_positional.so"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(_ROOT / "src/common"),
            str(_HERE / "kernels/host/ccu_positional.cpp"),
            "-o",
            str(module),
        ],
        check=True,
    )
    binary = module.read_bytes()
    module.unlink()
    spec = {
        "orchestration": {
            "source": str(_HERE / "kernels/orchestration/ccu_deferred_orch.cpp"),
            "function_name": "ccu_deferred_orchestration",
            "config_name": "ccu_deferred_orchestration_config",
            "signature": [D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": index,
                "source": str(_HERE / f"kernels/aiv/{name}.cpp"),
                "core_type": "aiv",
                "signature": signature,
            }
            for index, name, signature in [
                (0, "initialize", [D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR]),
                (1, "submit_ccu", [D.IN, D.INOUT, D.IN, D.INOUT, D.SCALAR, D.SCALAR]),
                (2, "verify", [D.IN, D.IN, D.IN, D.INOUT, D.SCALAR, D.SCALAR]),
            ]
        ],
        "ccu_functions": [CcuFunction.build(binary, f"broadcast_root{root}") for root in range(_RANKS)],
    }
    if rejection_probe:
        spec["orchestration"].update(
            source=str(_HERE / "kernels/orchestration/reject_ccu_orch.cpp"),
            function_name="ccu_rejection_orchestration",
            config_name="ccu_rejection_config",
            signature=[D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.IN, D.SCALAR, D.SCALAR],
        )
        spec["incores"][1].update(
            source=str(_HERE / "kernels/aiv/reject_ccu.cpp"),
            signature=[D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.IN, D.SCALAR, D.SCALAR],
        )
    scenario = "rejections" if rejection_probe else "broadcast"
    key = l3_compile_cache_key(__name__, "CcuDeferredCompletion", scenario, platform, _RUNTIME)
    return compile_chip_callable_spec(spec, platform, _RUNTIME, key)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(_RANKS)
@pytest.mark.parametrize(
    ("rounds", "rejection_probe"),
    [(4, False), (1000, False), (4, True)],
    ids=["smoke", "thousand_rounds", "invalid_and_cross_domain"],
)
def test_ccu_deferred_successors(st_platform, st_device_ids, tmp_path, rounds, rejection_probe):
    chip = _compile(st_platform, tmp_path, rejection_probe=rejection_probe)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:_RANKS]],
        num_sub_workers=0,
    )
    registered = worker.register(chip)
    domains = []
    try:
        worker.init()
        results = [worker.create_buffer(rounds * 4 * 4) for _ in range(_RANKS)]
        for result in results:
            result.shm.buf[:] = bytes([0xFF]) * (rounds * 4 * 4)

        def orchestration(orch, _args, cfg):
            with ExitStack() as contexts:
                foreign = None
                if rejection_probe:
                    foreign = contexts.enter_context(
                        orch.allocate_domain(
                            name="foreign",
                            members=[0, 1],
                            window_size=4 * 1024 * 1024,
                            buffers=[CommBufferSpec("foreign", "float32", _ELEMS, _ELEMS * 4)],
                        )
                    )
                    domains.append(foreign)
                domain = contexts.enter_context(
                    orch.allocate_domain(
                        name="ccu_deferred",
                        members=[0, 1],
                        workspace_engines=("ccu",),
                        ccu_callables=(registered,),
                        window_size=4 * 1024 * 1024,
                        buffers=[
                            CommBufferSpec(name, "float32", _ELEMS, _ELEMS * 4)
                            for name in ("source", "destination", "scratch")
                        ],
                    )
                )
                domains.append(domain)
                for rank in range(_RANKS):
                    args = TaskArgs()
                    for name in ("source", "destination", "scratch"):
                        args.add_tensor(
                            domain.local_context(rank).buffers[name].tensor((_ELEMS,), DataType.FLOAT32),
                            TensorArgType.INOUT,
                        )
                    args.add_tensor(results[rank].tensor((rounds * 4,), DataType.INT32), TensorArgType.INOUT)
                    if foreign is not None:
                        args.add_tensor(
                            foreign.local_context(rank).buffers["foreign"].tensor((_ELEMS,), DataType.FLOAT32),
                            TensorArgType.INPUT,
                        )
                    args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
                    args.add_scalar(rounds)
                    orch.submit_next_level(registered, args, cfg, worker=rank)

        worker.run(orchestration)
        for rank, result in enumerate(results):
            observed = struct.unpack_from(f"<{rounds * 4}i", result.shm.buf)
            for round_id in range(rounds):
                expected = (0, (round_id // 2) % 2, 1024 if round_id % 2 else 64, 16 + round_id % 8 * 16)
                actual = observed[round_id * 4 : (round_id + 1) * 4]
                assert actual == expected, f"rank={rank} round={round_id}: status/root/count/offset={actual}"
        assert len(domains) == (2 if rejection_probe else 1) and all(domain.freed for domain in domains)
        assert worker.live_domains == {}
    finally:
        worker.close()
