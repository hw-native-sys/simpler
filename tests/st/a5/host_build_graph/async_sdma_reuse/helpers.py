# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact SDMA data and public receipt checks across a retained workspace."""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import torch
from simpler.task_interface import (
    CORE_ASYNC_CONDITION_SDMA_POST_DONE,
    CallConfig,
    CommBufferSpec,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.task_interface import ArgDirection as D
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec
from simpler_setup.tools.swimlane_converter import read_perf_data

HERE = Path(__file__).resolve().parent
MAX_POSTS = 66
QUEUE_NUM = 48
META_WORDS = 256
RECEIPT_BASE = 64
RECEIPT_END = RECEIPT_BASE + 2 * QUEUE_NUM
LARGE_ELEMENTS = 1024 * 1024
GUARD_ELEMENTS = 16
SMALL_STRIDE = 48
ALLOCATED = LARGE_ELEMENTS + 2 * GUARD_ELEMENTS + (MAX_POSTS - 1) * SMALL_STRIDE
POISON = -999.0
GUARD = 0x13579BDF
SUBMIT_VISITS, RECEIPT_COUNT, ERROR, CONSUME_VISITS, EARLY, TASK_TOKEN = range(0, 48, 8)
PRIOR_INDEX, PRIOR_RECORD, PRIOR_VALUE, PRIOR_EXPECTED, PENDING_SAMPLE = 48, 49, 50, 51, 56


def l3_case(function):
    function._st_level = 3
    return function


def span(index):
    if index == 0:
        return GUARD_ELEMENTS, LARGE_ELEMENTS
    return LARGE_ELEMENTS + 2 * GUARD_ELEMENTS + (index - 1) * SMALL_STRIDE + GUARD_ELEMENTS, 8


def source_values(rank, epoch, posts):
    result = torch.full((ALLOCATED,), POISON, dtype=torch.float32)
    for index in range(posts):
        start, elements = span(index)
        result[start : start + elements] = (
            torch.arange(elements, dtype=torch.float32) % 251 + epoch * 10000 + (rank + 1) * 1000 + index * 5
        )
    return result


def initial_metadata():
    result = torch.full((MAX_POSTS, META_WORDS), GUARD, dtype=torch.int64)
    result[:, :RECEIPT_END] = 0
    return result


def stage(buffer, tensor):
    torch.frombuffer(buffer.shm.buf, dtype=tensor.dtype, count=tensor.numel()).copy_(tensor.flatten())


def read(buffer, template):
    return (
        torch.frombuffer(buffer.shm.buf, dtype=template.dtype, count=template.numel()).clone().reshape(template.shape)
    )


@cache
def compile_series(platform):
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/series.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT] + [D.SCALAR] * 3,
        },
        "incores": [
            {
                "func_id": index,
                "source": str(HERE / "kernels/aiv" / source),
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT],
                "async_capability": {
                    "mode": "DEFERRED",
                    "max_conditions": 64,
                    "condition_types": CORE_ASYNC_CONDITION_SDMA_POST_DONE,
                }
                if index == 0
                else {"mode": "SYNCHRONOUS"},
            }
            for index, source in enumerate(("submit.cpp", "consume.cpp"))
        ],
    }
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (str(HERE), platform, "sdma_reuse", _pto_isa_compile_cache_token())
    )


def receipt_rows(metadata, posts):
    """Validate exact per-task bookkeeping without deriving expectations from scheduler counters."""
    expected = initial_metadata()
    rows: list[dict[str, Any]] = []
    for index in range(posts):
        meta = metadata[index]
        count = int(meta[RECEIPT_COUNT])
        assert 1 <= count <= QUEUE_NUM, (index, count)
        records = [(int(meta[RECEIPT_BASE + 2 * i]), int(meta[RECEIPT_BASE + 2 * i + 1])) for i in range(count)]
        assert all(address > 0 and address % 8 == 0 and value > 0 for address, value in records)
        assert len({address for address, _ in records}) == count
        assert len({value for _, value in records}) == 1
        expected[index, SUBMIT_VISITS] = expected[index, CONSUME_VISITS] = 1
        expected[index, RECEIPT_COUNT] = count
        expected[index, TASK_TOKEN] = 2 * index
        expected[index, RECEIPT_BASE : RECEIPT_BASE + 2 * count] = meta[RECEIPT_BASE : RECEIPT_BASE + 2 * count]
        pending = int(meta[PENDING_SAMPLE])
        assert pending in (0, 1)
        sample = None
        if pending:
            prior, record = int(meta[PRIOR_INDEX]) - 1, int(meta[PRIOR_RECORD]) - 1
            assert 0 <= prior < posts and prior != index
            assert 0 <= record < int(metadata[prior, RECEIPT_COUNT])
            threshold = int(metadata[prior, RECEIPT_BASE + 2 * record + 1])
            value = int(meta[PRIOR_VALUE])
            assert 0 <= value < int(meta[PRIOR_EXPECTED]) == threshold < records[0][1]
            sample = {"prior_index": prior, "record_index": record, "value": value, "expected": threshold}
            expected[index, PRIOR_INDEX : PRIOR_EXPECTED + 1] = meta[PRIOR_INDEX : PRIOR_EXPECTED + 1]
            expected[index, PENDING_SAMPLE] = 1
        rows.append(
            {
                "index": index,
                "task_ref": 2 * index,
                "post_id": records[0][1],
                "records": records,
                "pending_sample": sample,
            }
        )
    torch.testing.assert_close(metadata, expected, rtol=0, atol=0)
    return rows


def receipt_continuity(rows: list[dict[str, Any]], prior: dict[str, Any] | None):
    ordered = sorted(rows, key=lambda row: row["post_id"])
    ids = [row["post_id"] for row in ordered]
    assert ids and ids == list(range(ids[0], ids[0] + len(rows))), "public post IDs repeat or skip"
    for row in ordered:
        addresses = [address for address, _ in row["records"]]
        if prior is not None:
            assert row["post_id"] == prior["post_id"] + 1
            assert addresses[: len(prior["addresses"])] == prior["addresses"], (
                "public record prefix changed across the retained session"
            )
        prior = {"post_id": row["post_id"], "addresses": addresses}
    return ordered, prior


def audit_capture(directory, posts, rows):
    files = list(directory.rglob("chip_swimlane_records.json"))
    assert len(files) == 1, files
    raw = json.loads(files[0].read_text())
    decoded = read_perf_data(files[0])
    protocol = raw["aicore_protocol"]
    assert protocol["schema"] == "hbg-native-protocol-v10"
    assert decoded["protocol_audit"]["protocol_verified"]
    assert protocol["scheduler_error"] == protocol["error_claimed"] == 0
    assert len(protocol["tasks"]) == len(protocol["executions"]) == len(decoded["tasks"]) == 2 * posts
    groups = {group["task_ref"]: group for group in protocol["async"]["groups"]}
    assert len(groups) == len(protocol["async"]["finishes"]) == posts
    executions = {execution["task_ref"]: execution for execution in protocol["executions"]}
    tasks = {task["task_ref"]: task for task in protocol["tasks"]}
    for row in rows:
        task = row["task_ref"]
        assert tasks[task]["dependencies"] == [] and tasks[task + 1]["dependencies"] == [task]
        assert tasks[task]["kernel_ids"] == [65535, 0, 65535]
        assert tasks[task + 1]["kernel_ids"] == [65535, 1, 65535]
        group = groups[task]
        assert group["state"] == "READY" and group["registered_count"] == len(row["records"])
        assert all(condition["type"] == 1 for condition in group["conditions"])
        row["worker_index"] = executions[task]["worker_index"]
        row["terminal_worker_index"] = group["terminal"]["worker_index"]
    assert all(finish["state"] == "DRAINED" for finish in protocol["async"]["finishes"])
    return {"capture": str(files[0]), "run_epoch": protocol["run_epoch"], "audit": decoded["protocol_audit"]}


class SdmaReuseRunner:
    def __init__(self, platform, devices, directory):
        assert len(devices) >= 2, "real SDMA reuse requires two assigned devices"
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.domain_identity: tuple | None = None
        self.previous: list[dict[str, Any] | None] = [None, None]
        self.reports: list[dict[str, Any]] = []
        self.metadata = [initial_metadata().share_memory_() for _ in range(2)]
        self.results = [torch.full((ALLOCATED,), POISON, dtype=torch.float32).share_memory_() for _ in range(2)]
        self.worker = Worker(
            level=3,
            platform=platform,
            runtime="host_build_graph",
            device_ids=list(map(int, devices[:2])),
            num_sub_workers=0,
        )
        try:
            self.callable = self.worker.register(compile_series(platform))
            self.worker.init()
            self.staging = [
                {name: self.worker.create_buffer(ALLOCATED * 4) for name in ("source", "destination")} for _ in range(2)
            ]
        except BaseException:
            self._cleanup()
            raise

    def _args(self, domain, rank, posts):
        args = TaskArgs()
        for name, direction in (("source", TensorArgType.INPUT), ("destination", TensorArgType.INOUT)):
            args.add_tensor(domain[rank].buffers[name].tensor((ALLOCATED,), DataType.FLOAT32), direction)
        args.add_tensor(
            self.worker.make_tensor_arg(self.results[rank], (ALLOCATED,), DataType.FLOAT32), TensorArgType.INOUT
        )
        args.add_tensor(
            self.worker.make_tensor_arg(self.metadata[rank], (MAX_POSTS * META_WORDS,), DataType.INT64),
            TensorArgType.INOUT,
        )
        for value in (domain.device_context(rank), 1 - rank, posts):
            args.add_scalar(int(value))
        return args

    def run(self, posts):
        assert 1 <= posts <= MAX_POSTS
        self.iteration += 1
        capture = self.directory / f"run_{self.iteration}"
        capture.mkdir()
        source = [source_values(rank, self.iteration, posts) for rank in range(2)]
        poison = torch.full((ALLOCATED,), POISON, dtype=torch.float32)
        for rank in range(2):
            self.metadata[rank].copy_(initial_metadata())
            self.results[rank].copy_(poison)
        config = CallConfig()
        config.enable_chip_swimlane = 4
        config.output_prefix = str(capture)

        def orchestrate(orch, _args, _config):
            buffers = [
                CommBufferSpec(name=name, dtype="float32", count=ALLOCATED, nbytes=ALLOCATED * 4)
                for name in ("source", "destination")
            ]
            domain = orch.get_or_allocate_domain(
                name="resident_sdma_reuse",
                members=[0, 1],
                window_size=ALLOCATED * 8,
                buffers=buffers,
                retain_after_run=True,
                workspace_engines=("sdma",),
                sdma_sdk_queue_num=QUEUE_NUM,
            )
            identity = (domain.domain_id, domain.generation, tuple(domain.device_context(rank) for rank in range(2)))
            if self.domain_identity is None:
                self.domain_identity = identity
            else:
                assert identity == self.domain_identity, "retained SDMA domain identity changed"
            for rank in range(2):
                stage(self.staging[rank]["source"], source[rank])
                stage(self.staging[rank]["destination"], poison)
                orch.copy_to(domain[rank].buffers["source"], self.staging[rank]["source"])
                orch.copy_to(domain[rank].buffers["destination"], self.staging[rank]["destination"])
            for rank in range(2):
                rank_config = CallConfig()
                rank_config.enable_chip_swimlane = 4
                rank_config.output_prefix = str(capture / f"rank_{rank}")
                Path(rank_config.output_prefix).mkdir()
                orch.submit_next_level(self.callable, self._args(domain, rank, posts), rank_config, worker=rank)

        try:
            self.worker.submit(orchestrate, config=config).wait(timeout=60)
        except BaseException as error:
            (capture / "sdma-reuse-failure.json").write_text(
                json.dumps({"error": str(error), "host_buffers_prove_copyback": False}, indent=2) + "\n"
            )
            raise
        report: dict[str, Any] = {"posts_per_rank": posts, "domain_identity": self.domain_identity, "ranks": []}
        domain = self.worker.live_domains["resident_sdma_reuse"]
        for rank in range(2):
            golden = poison.clone()
            transfer_golden = poison.clone()
            for index in range(posts):
                start, elements = span(index)
                golden[start : start + elements] = source[1 - rank][start : start + elements] + 7
                transfer_golden[start : start + elements] = source[1 - rank][start : start + elements]
            torch.testing.assert_close(self.results[rank], golden, rtol=0, atol=0)
            for name, expected in (("source", source[rank]), ("destination", transfer_golden)):
                self.worker.copy_from(self.staging[rank][name], domain[rank].buffers[name])
                torch.testing.assert_close(read(self.staging[rank][name], expected), expected, rtol=0, atol=0)
            rows = receipt_rows(self.metadata[rank], posts)
            audit = audit_capture(capture / f"rank_{rank}", posts, rows)
            ordered, self.previous[rank] = receipt_continuity(rows, self.previous[rank])
            ids = [row["post_id"] for row in ordered]
            if posts == MAX_POSTS:
                assert ids[-1] - ids[0] >= 65
            handoffs = sum(a["worker_index"] != b["worker_index"] for a, b in zip(ordered, ordered[1:]))
            report["ranks"].append(
                {
                    "rank": rank,
                    "audit": audit,
                    "receipts": rows,
                    "post_id_min": ids[0],
                    "post_id_max": ids[-1],
                    "observed_cross_core_submitter_handoffs": handoffs,
                    "observed_prior_pending_samples": sum(row["pending_sample"] is not None for row in rows),
                }
            )
        report["limits"] = {
            "more_than_64_simultaneously_inflight_verified": False,
            "actual_SDK_backpressure_verified": False,
            "note": (
                "Public post-id continuity proves wrap/reuse only. Pending and core handoff are reported only "
                "when observed. Kernel duration is not a measurement of SDK wait."
            ),
        }
        (capture / "sdma-reuse-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
        self.reports.append(report)
        return report

    def _cleanup(self):
        try:
            self.worker.close()
        except Exception as error:
            warnings.warn(f"SDMA reuse cleanup also failed: {error}", RuntimeWarning, stacklevel=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self.worker.close()
        else:
            self._cleanup()
