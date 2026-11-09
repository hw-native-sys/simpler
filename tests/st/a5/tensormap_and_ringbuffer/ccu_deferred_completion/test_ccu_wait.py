# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One AIV waits for, consumes, and reuses real CCU requests on each rank."""

import os
import shlex
import struct
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CcuFunction, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[4]
_RUNTIME = "tensormap_and_ringbuffer"
_RANKS = 2
_ELEMS = 2048
_RESULT_WORDS = 8


def _compile_wait(platform, directory):
    module = directory / "wait_positional.so"
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
    signature = [D.INOUT, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR]
    spec = {
        "orchestration": {
            "source": str(_HERE / "kernels/orchestration/wait_orch.cpp"),
            "function_name": "ccu_wait_orchestration",
            "config_name": "ccu_wait_config",
            "signature": signature,
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(_HERE / "kernels/aiv/wait_rounds.cpp"),
                "core_type": "aiv",
                "signature": signature,
            }
        ],
        "ccu_functions": [CcuFunction.build(binary, f"broadcast_root{root}") for root in range(_RANKS)],
    }
    key = l3_compile_cache_key(__name__, "CcuWait", "immediate_consumption", platform, _RUNTIME)
    return compile_chip_callable_spec(spec, platform, _RUNTIME, key)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(_RANKS)
@pytest.mark.parametrize("rounds", [4, 1000], ids=["smoke", "thousand_rounds"])
def test_ccu_wait_consumes_before_next_request(st_platform, st_device_ids, tmp_path, rounds):
    chip = _compile_wait(st_platform, tmp_path)
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
        words = rounds * _RESULT_WORDS
        results = [worker.create_buffer(words * 4) for _ in range(_RANKS)]
        for result in results:
            result.shm.buf[:] = bytes([0xFF]) * (words * 4)

        def orchestration(orch, _args, cfg):
            with orch.allocate_domain(
                name="ccu_wait",
                members=[0, 1],
                ccu_callables=(registered,),
                workspace_engines=("ccu",),
                window_size=4 * 1024 * 1024,
                buffers=[
                    CommBufferSpec(name, "float32", _ELEMS, _ELEMS * 4) for name in ("source", "destination", "scratch")
                ],
            ) as domain:
                domains.append(domain)
                for rank in range(_RANKS):
                    args = TaskArgs()
                    for name in ("source", "destination", "scratch"):
                        args.add_tensor(
                            domain.local_context(rank).buffers[name].tensor((_ELEMS,), DataType.FLOAT32),
                            TensorArgType.INOUT,
                        )
                    args.add_tensor(results[rank].tensor((words,), DataType.INT32), TensorArgType.INOUT)
                    args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
                    args.add_scalar(rounds)
                    orch.submit_next_level(registered, args, cfg, worker=rank)

        worker.run(orchestration)
        for rank, result in enumerate(results):
            observed = struct.unpack_from(f"<{words}i", result.shm.buf)
            for round_id in range(rounds):
                expected = (0, 0, (round_id // 2) % 2, 1024 if round_id % 2 else 64, 16 + round_id % 8 * 16, -1, 1, 0)
                actual = observed[round_id * _RESULT_WORDS : (round_id + 1) * _RESULT_WORDS]
                assert actual == expected, (
                    f"rank={rank} round={round_id}: "
                    f"errors/prime_errors/root/count/offset/first_mismatch/completed/deferred={actual}"
                )
        assert len(domains) == 1 and domains[0].freed
        assert worker.live_domains == {}
    finally:
        worker.close()
