# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded, real CCU requests that outlive their submitting AIV kernel."""

import os
import re
import shlex
import struct
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, CcuFunction, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[4]
_RUNTIME = "tensormap_and_ringbuffer"
_RANKS = 2
_ELEMS = 64
_MAX_AIV = 160
_RESULT_WORDS = (_MAX_AIV + 1) * 16
_CONTROL_WORDS = (_MAX_AIV + 4) * 16


def _compile_faults(platform, directory):
    module = directory / "fault_module.so"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(_ROOT / "src/common"),
            str(_HERE / "kernels/host/fault_module.cpp"),
            "-o",
            str(module),
        ],
        check=True,
    )
    binary = module.read_bytes()
    module.unlink()
    spec = {
        "orchestration": {
            "source": str(_HERE / "kernels/orchestration/fault_orch.cpp"),
            "function_name": "fault_orchestration",
            "config_name": "fault_orchestration_config",
            "signature": [D.IN, D.INOUT, D.IN, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": index,
                "source": str(_HERE / f"kernels/aiv/{name}.cpp"),
                "core_type": "aiv",
                "signature": signature,
            }
            for index, name, signature in [
                (0, "fault_submit", [D.IN, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR]),
                (1, "fault_witness", [D.INOUT, D.SCALAR, D.SCALAR]),
                (2, "fault_consumer", [D.IN, D.INOUT, D.SCALAR]),
            ]
        ],
        "ccu_functions": [CcuFunction.build(binary, "fault_broadcast")],
    }
    key = l3_compile_cache_key(__name__, "CcuFaults", "pending_at_fin", platform, _RUNTIME)
    return compile_chip_callable_spec(spec, platform, _RUNTIME, key)


def _run_fault_scenario(worker, registered, results, domains, *, mismatch):
    def orchestration(orch, _args, cfg):
        with orch.allocate_domain(
            name="ccu_pending_at_fin",
            members=[0, 1],
            workspace_engines=("ccu",),
            ccu_callables=(registered,),
            window_size=4 * 1024 * 1024,
            buffers=[
                CommBufferSpec("source", "float32", _ELEMS, _ELEMS * 4),
                CommBufferSpec("destination", "float32", _ELEMS, _ELEMS * 4),
                CommBufferSpec("control", "int32", _CONTROL_WORDS, _CONTROL_WORDS * 4),
            ],
        ) as domain:
            domains.append(domain)
            for rank in range(_RANKS):
                orch.copy_to_domain(
                    domain,
                    rank,
                    struct.pack(f"<{_ELEMS}f", *(float(100 + rank * 1000 + i) for i in range(_ELEMS))),
                    buffer="source",
                )
                orch.copy_to_domain(domain, rank, bytes(_ELEMS * 4), buffer="destination")
                orch.copy_to_domain(domain, rank, bytes(_CONTROL_WORDS * 4), buffer="control")
            for rank in range(_RANKS):
                args = TaskArgs()
                for name, dtype, size, direction in [
                    ("source", DataType.FLOAT32, _ELEMS, TensorArgType.INPUT),
                    ("destination", DataType.FLOAT32, _ELEMS, TensorArgType.INOUT),
                    ("control", DataType.INT32, _CONTROL_WORDS, TensorArgType.INPUT),
                ]:
                    args.add_tensor(domain.local_context(rank).buffers[name].tensor((size,), dtype), direction)
                args.add_tensor(results[rank].tensor((_RESULT_WORDS,), DataType.INT32), TensorArgType.INOUT)
                args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
                args.add_scalar(int(mismatch))
                args.add_scalar(rank)
                orch.submit_next_level(registered, args, cfg, worker=rank)

    config = CallConfig()
    config.aicpu_thread_num = 4
    worker.run(orchestration, config=config)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(_RANKS)
def test_ccu_is_pending_after_aiv_fin(st_platform, st_device_ids, tmp_path):
    chip = _compile_faults(st_platform, tmp_path)
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
        results = [worker.create_buffer(_RESULT_WORDS * 4) for _ in range(_RANKS)]
        for result in results:
            result.shm.buf[:] = bytes([0xFF]) * (_RESULT_WORDS * 4)

        _run_fault_scenario(worker, registered, results, domains, mismatch=False)
        rank0_width = struct.unpack_from("<I", results[0].shm.buf, 19 * 4)[0]
        for rank, result in enumerate(results):
            words = struct.unpack_from(f"<{_RESULT_WORDS}I", result.shm.buf)
            assert words[0] == 0, f"rank={rank}: successor saw incorrect CCU output ({words[0]})"
            width = words[19]
            assert 0 < width <= _MAX_AIV, f"rank={rank}: invalid witness cohort width {width}"
            for block in range(width):
                pending, submitted, consumer_ran, block_count = words[(block + 1) * 16 : (block + 1) * 16 + 4]
                assert submitted == 1 and consumer_ran == 0 and block_count == width, (
                    rank,
                    block,
                    pending,
                    submitted,
                    consumer_ran,
                    block_count,
                )
                if rank == 0:
                    assert pending == 1, f"rank 0 lane {block}: CCU was not pending after the full AIV cohort started"
                else:
                    peer_ready, peer_width = words[(block + 1) * 16 + 4 : (block + 1) * 16 + 6]
                    assert peer_ready == 1 and peer_width == rank0_width, (
                        f"rank 1 lane {block}: missing rank 0 successor/pending handshake "
                        f"(ready={peer_ready}, width={peer_width}, expected={rank0_width})"
                    )
        assert len(domains) == 1 and domains[0].freed
        assert worker.live_domains == {}
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(_RANKS)
def test_ccu_failed_request_reports_error_and_reclaims_domain(st_platform, st_device_ids, tmp_path, capfd):
    chip = _compile_faults(st_platform, tmp_path)
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
        results = [worker.create_buffer(_RESULT_WORDS * 4) for _ in range(_RANKS)]
        for result in results:
            result.shm.buf[:] = bytes([0xFF]) * (_RESULT_WORDS * 4)
        # Both real requests are accepted. The builder rejects unequal peer
        # scalars after metadata exchange and publishes a confirmed FAILED result.
        with pytest.raises(RuntimeError, match=r"507018|-105|CCU_FAILED"):
            _run_fault_scenario(worker, registered, results, domains, mismatch=True)
        captured = capfd.readouterr()
        evidence = captured.out + captured.err
        (tmp_path / "failed-runtime.log").write_text(evidence, encoding="utf-8")
        print(evidence, flush=True)
        assert "sched_error_code=105" in evidence, evidence
        assert "sched_error_code=106" not in evidence, evidence
        assert "CCU_UNCONFIRMED" not in evidence, evidence
        observations = re.findall(
            r"\[CCU failure\] failed_token=0x([0-9a-f]+) successor_token=0x([0-9a-f]+) "
            r"successors=(\d+) state=(\d+) dispatched=(\d+) flags=0x[0-9a-f]+ scan_complete=(\d+) unstarted=(\d+)",
            evidence,
        )
        assert len(observations) == _RANKS, evidence
        for producer, successor, count, state, dispatched, complete, unstarted in observations:
            assert producer != successor and int(producer, 16) != (1 << 64) - 1, evidence
            assert int(count) >= 1 and state == "0" and dispatched == "0", evidence
            assert complete == "1" and unstarted == "1", evidence
        assert len(domains) == 1 and domains[0].freed, (
            "confirmed CCU failure left its domain physically retained; "
            f"free_results={getattr(worker, '_domain_free_results', {})!r}; "
            f"cleanup_error={getattr(worker, '_ordered_cleanup_error', None)!r}"
        )
        assert worker.live_domains == {}, "confirmed CCU failure left a live domain owner"
        # Failure skips ordinary D2H: an unchanged host sentinel is not evidence
        # about whether the dependency successor started.
    finally:
        worker.close()
