# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A rank-local builder rejection must roll back before another domain can run."""

import json
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
from tests.st.a5.tensormap_and_ringbuffer.ccu_deferred_completion.test_ccu_deferred_completion import _compile

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[4]
_RUNTIME = "tensormap_and_ringbuffer"
_RANKS = 2
_ELEMS = 2048


def _compile_rejecting_builder(platform, directory):
    module = directory / "prep_failure_rank1.so"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(_ROOT / "src/common"),
            f"-DPREP_FAILURE_MARKER_ROOT={json.dumps(str(directory))}",
            str(_HERE / "kernels/host/prep_failure_rank1.cpp"),
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
        "ccu_functions": [CcuFunction.build(binary, "reject_rank1")],
    }
    key = l3_compile_cache_key(__name__, "CcuPrepareFailure", "rank1_reject", platform, _RUNTIME)
    return compile_chip_callable_spec(spec, platform, _RUNTIME, key)


def _allocate(orch, registered):
    return orch.allocate_domain(
        name="prepare_recovery",
        members=[0, 1],
        workspace_engines=("ccu",),
        ccu_callables=(registered,),
        window_size=4 * 1024 * 1024,
        buffers=[CommBufferSpec(name, "float32", _ELEMS, _ELEMS * 4) for name in ("source", "destination", "scratch")],
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(_RANKS)
def test_single_rank_prepare_failure_rolls_back_before_recovery(st_platform, st_device_ids, tmp_path, monkeypatch):
    rejecting = _compile_rejecting_builder(st_platform, tmp_path)
    normal = _compile(st_platform, tmp_path)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:_RANKS]],
        num_sub_workers=0,
    )
    rejected = worker.register(rejecting)
    preparations = []
    submissions = []
    prepare = worker._prepare_ccu_domain

    def observe_prepare(domain, definitions):
        # Observation only: the real child control broadcast and SDK builder execute unchanged.
        preparations.append(domain)
        return prepare(domain, definitions)

    monkeypatch.setattr(worker, "_prepare_ccu_domain", observe_prepare)
    try:
        worker.init()

        def reject(orch, _args, cfg):
            with pytest.raises(RuntimeError, match="Worker control broadcast"):
                with _allocate(orch, rejected):
                    submissions.append("unexpected")
                    pytest.fail("rank-local builder rejection admitted an L2 submission")
            assert submissions == []
            assert len(preparations) == 1
            assert preparations[0].released
            assert not preparations[0].freed
            assert worker.live_domains == {}

        worker.run(reject)
        assert (tmp_path / "rank0-accepted").is_file(), "rank 0 did not execute the real builder"
        assert (tmp_path / "rank1-rejected").is_file(), "failure preceded the intentional rank 1 rejection"
        failed = preparations[0]
        assert failed.freed, "failed preparation did not confirm physical window reclamation"
        assert worker._domain_free_results[failed.allocation_id] is None
        assert worker.live_domains == {}
        assert not any(key[0] == rejected.digest for key in worker._ccu_prepared_domains)
        # Native unregister also refuses if a partial CCU instance remains in the provider.
        worker.unregister(rejected)
        registered = worker.register(normal)
        rounds = 4
        results = [worker.create_buffer(rounds * 16) for _ in range(_RANKS)]
        for result in results:
            result.shm.buf[:] = bytes([0xFF]) * (rounds * 16)

        def recover(orch, _args, cfg):
            with _allocate(orch, registered) as domain:
                assert domain.allocation_id != failed.allocation_id
                for rank in range(_RANKS):
                    args = TaskArgs()
                    for name in ("source", "destination", "scratch"):
                        args.add_tensor(
                            domain.local_context(rank).buffers[name].tensor((_ELEMS,), DataType.FLOAT32),
                            TensorArgType.INOUT,
                        )
                    args.add_tensor(results[rank].tensor((rounds * 4,), DataType.INT32), TensorArgType.INOUT)
                    args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
                    args.add_scalar(rounds)
                    submissions.append(rank)
                    orch.submit_next_level(registered, args, cfg, worker=rank)

        worker.run(recover)
        assert submissions == [0, 1]
        for rank, result in enumerate(results):
            values = struct.unpack_from(f"<{rounds * 4}i", result.shm.buf)
            for round_id in range(rounds):
                expected = (0, (round_id // 2) % 2, 1024 if round_id % 2 else 64, 16 + round_id % 8 * 16)
                assert values[round_id * 4 : (round_id + 1) * 4] == expected, (rank, round_id, values)
        assert len(preparations) == 2 and preparations[1].freed
        assert worker.live_domains == {}
    finally:
        worker.close()
