#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public-API workloads and independent numerical/causal oracles for resident completion."""

from __future__ import annotations

import json
import sys
import time
from functools import cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec

from .helpers import assert_resident_capture, close_after_failure, l3_case, read_host_tensor, stage_host_tensor

HERE = Path(__file__).resolve().parent


@cache
def compile_urma(platform: str):
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/urma.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": index,
                "source": str(HERE / "kernels/aiv" / source),
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT],
            }
            for index, source in enumerate(
                ("urma_submit.cpp", "urma_consumer.cpp", "urma_probe.cpp", "urma_cq_contract.cpp")
            )
        ],
    }
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (str(HERE), platform, "urma", _pto_isa_compile_cache_token())
    )


def run_urma(
    platform: str,
    devices: list[int],
    directory: Path,
    elements: int,
    put: bool = False,
    synchronous: bool = False,
    contract: bool = False,
    samples: int = 2,
    dump_probe: bool = False,
):
    assert len(devices) >= 2, "real peer URMA requires two assigned devices"
    compiled = compile_urma(platform)
    # CommDomain buffers are 64-byte aligned; leave at least 32 guard elements.
    allocated = ((elements + 32 + 15) // 16) * 16
    sources = [
        ((torch.arange(allocated, dtype=torch.float32) % 251) + (rank + 1) * 1000).share_memory_() for rank in range(2)
    ]
    initial = [torch.full((allocated,), -999.0, dtype=torch.float32).share_memory_() for _ in range(2)]
    results = [torch.full((allocated,), -999.0, dtype=torch.float32).share_memory_() for _ in range(2)]
    metadata = [torch.zeros(8192, dtype=torch.int64).share_memory_() for _ in range(2)]
    worker = Worker(
        level=3,
        platform=platform,
        runtime="host_build_graph",
        device_ids=list(map(int, devices[:2])),
        num_sub_workers=0,
    )
    callable_handle = worker.register(compiled)
    worker.init()
    source_staging = [worker.create_buffer(allocated * 4) for _ in range(2)]
    destination_staging = [worker.create_buffer(allocated * 4) for _ in range(2)]
    capture = directory / "urma"
    capture.mkdir(parents=True)
    config = CallConfig()
    capture_level = 2 if elements >= 16 * 1024 * 1024 else 4
    config.enable_chip_swimlane = capture_level
    config.output_prefix = str(capture)
    saved_domain = []

    def orchestrate(orch, _args, _config):
        buffers = [
            CommBufferSpec(name=name, dtype="float32", count=allocated, nbytes=allocated * 4)
            for name in ("source", "destination")
        ]
        domain = orch.get_or_allocate_domain(
            name="async_urma_contract",
            members=[0, 1],
            window_size=allocated * 8,
            buffers=buffers,
            retain_after_run=True,
            workspace_engines=("urma",),
        )
        if saved_domain:
            assert domain is saved_domain[0]
        else:
            saved_domain.append(domain)
        for rank in range(2):
            stage_host_tensor(source_staging[rank], sources[rank])
            stage_host_tensor(destination_staging[rank], initial[rank])
            orch.copy_to(domain[rank].buffers["source"], source_staging[rank])
            orch.copy_to(domain[rank].buffers["destination"], destination_staging[rank])
        for rank in range(2):
            args = TaskArgs()
            args.add_tensor(domain[rank].buffers["source"].tensor((allocated,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                domain[rank].buffers["destination"].tensor((allocated,), DataType.FLOAT32), TensorArgType.INOUT
            )
            args.add_tensor(worker.make_tensor_arg(results[rank], (allocated,), DataType.FLOAT32), TensorArgType.INOUT)
            args.add_tensor(worker.make_tensor_arg(metadata[rank], (8192,), DataType.INT64), TensorArgType.INOUT)
            args.add_scalar(domain[rank].device_ctx)
            args.add_scalar(elements)
            args.add_scalar(int(put))
            args.add_scalar(2 if contract else int(synchronous))
            rank_config = CallConfig()
            rank_config.enable_chip_swimlane = capture_level
            rank_config.enable_dump_args = 1 if dump_probe else 0
            rank_config.output_prefix = str(capture / f"run_{cycle}_rank_{rank}")
            Path(rank_config.output_prefix).mkdir()
            orch.submit_next_level(callable_handle, args, rank_config, worker=rank)

    measurements = []
    try:
        for cycle in range(samples):
            for meta in metadata:
                meta.zero_()
            started = time.perf_counter_ns()
            worker.submit(orchestrate, config=config).wait(timeout=60)
            wall_ns = time.perf_counter_ns() - started
            for rank in range(2):
                meta = metadata[rank]
                if contract:
                    assert int(meta[0]) == 5 and int(meta[32]) == 0, meta[:64]
                    continue
                assert int(meta[16]) == 0, "resident deferred context unavailable; no URMA support was exercised"
                assert int(meta[32]) == 0, "URMA submission, public record export, or probe failed"
                assert int(meta[48]) == 0, "URMA consumer entered before completion"
                golden = torch.full_like(results[rank], -999.0)
                golden[:elements] = sources[rank if put else 1 - rank][:elements] + 7
                torch.testing.assert_close(results[rank], golden, rtol=0, atol=0)
                worker.copy_from(
                    destination_staging[rank], worker.live_domains["async_urma_contract"][rank].buffers["destination"]
                )
                transferred = read_host_tensor(destination_staging[rank], initial[rank])
                golden_transfer = initial[rank].clone()
                golden_transfer[:elements] = sources[1 - rank][:elements]
                torch.testing.assert_close(transferred, golden_transfer, rtol=0, atol=0)
                rank_path = capture / f"run_{cycle}_rank_{rank}"
                if capture_level == 4:
                    assert_resident_capture(rank_path)
                if elements != 0:
                    files = list(rank_path.rglob("chip_swimlane_records.json"))
                    assert len(files) == 1, files
                    raw = json.loads(files[0].read_text())
                    identities = raw["aicore_execution_identities"]
                    assert identities["producer"] == "aicore"
                    # The orchestration submits transfer, independent probe, then consumer.
                    # Join public task IDs through the exported execution identity records.
                    producer_traces = {r["reg_task_id"] for r in identities["records"] if r["task_id"] == 0}
                    consumer_traces = {r["reg_task_id"] for r in identities["records"] if r["task_id"] == 2}
                    producer_rows = [row for row in raw["aicore_tasks"] if row[2] in producer_traces]
                    consumer_rows = [row for row in raw["aicore_tasks"] if row[2] in consumer_traces]
                    assert len(producer_rows) == 1
                    assert len(consumer_rows) == 1
                    measurements.append(
                        {
                            "cycle": cycle,
                            "rank": rank,
                            "wall_ns": wall_ns,
                            "producer_ticks": producer_rows[0][4] - producer_rows[0][3],
                            "completion_gap_ticks": consumer_rows[0][3] - producer_rows[0][4],
                            "to_consumer_ticks": consumer_rows[0][3] - producer_rows[0][3],
                            "through_consumer_ticks": consumer_rows[0][4] - producer_rows[0][3],
                            "ticks_per_ms": int(meta[176]),
                            "elements": elements,
                            "put": put,
                            "synchronous": synchronous,
                        }
                    )
                    if elements >= 16 * 1024 * 1024 and not synchronous:
                        assert int(meta[64]) > 0 and int(meta[80]) > producer_rows[0][4], (
                            "no real URMA PENDING observation after kernel return"
                        )
        (directory / "urma_measurements.json").write_text(json.dumps(measurements, indent=2))
        print(json.dumps(measurements))
    finally:
        snapshot_indices = {
            "returned": 0,
            "context_errors": 16,
            "submit_probe_errors": 32,
            "consumer_errors": 48,
            "pending_samples": 64,
            "last_pending_tick": 80,
            "cq_context": 128,
            "target_cqe": 144,
            "sq_context": 192,
            "target_bb": 208,
            "queue_row": 224,
            "cq_word": 256,
            "cq_tail": 257,
            "sq_head": 258,
            "sq_tail": 259,
            "cq_depth": 260,
            "destination_first_bits": 261,
            "destination_last_bits": 262,
        }
        (directory / "urma_metadata.json").write_text(
            json.dumps([{key: int(meta[index]) for key, index in snapshot_indices.items()} for meta in metadata])
        )
        if sys.exc_info()[0] is None:
            worker.close()
            assert saved_domain[0].freed
        else:
            close_after_failure(worker)


pytestmark = [pytest.mark.platforms(["a5"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(2)]


@pytest.mark.parametrize("put", [False, True], ids=["get", "put"])
@pytest.mark.parametrize("elements", [0, 8, 16 * 1024 * 1024], ids=["empty", "small", "pending"])
@l3_case
def test_urma_device_completion(st_platform, st_device_ids, tmp_path, put, elements):
    run_urma(st_platform, st_device_ids, tmp_path, elements, put)


@l3_case
def test_urma_cq_hardware_contract(st_platform, st_device_ids, tmp_path):
    run_urma(st_platform, st_device_ids, tmp_path, 8, contract=True)


@pytest.mark.manual
@pytest.mark.parametrize("put", [False, True], ids=["get", "put"])
@pytest.mark.parametrize("synchronous", [False, True], ids=["deferred", "synchronous"])
@l3_case
def test_urma_latency_comparison(st_platform, st_device_ids, tmp_path, put, synchronous):
    run_urma(st_platform, st_device_ids, tmp_path, 16 * 1024 * 1024, put, synchronous, samples=11)
