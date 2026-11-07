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
import warnings
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import cache
from pathlib import Path

import torch
from simpler.task_interface import (
    CORE_ASYNC_CONDITION_COUNTER,
    CORE_ASYNC_CONDITION_SDMA_POST_DONE,
    CallConfig,
    CommBufferSpec,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.task_interface import ArgDirection as D
from simpler.worker import Worker

from simpler_setup.scene_test import _bound_host_logs, _pto_isa_compile_cache_token, compile_chip_callable_spec

HERE = Path(__file__).resolve().parent
CONDITION_LIMIT = 64
SIGNAL_STRIDE = 32
CONDITIONS_PER_EXECUTION = CONDITION_LIMIT + 1
PAYLOAD_WORDS = 32
OBSERVATION_WORDS = 32
CONTROL_STRIDE = 16
CONTROL_CELLS = 16
POISON = -1000000007


def l3_case(function):
    """Expose the existing collector's execution-level metadata for a plain pytest function."""
    function._st_level = 3
    return function


class Scenario(IntEnum):
    BASIC = 0
    TWO_CONDITIONS = 1
    FANIN = 2
    PROGRESS = 3
    VISIBILITY = 4
    GRAPH = 5
    NEVER = 6
    FULL_SYNC_START = 7


@dataclass(frozen=True)
class Case:
    scenario: Scenario = Scenario.BASIC
    conditions: int = 1
    waiters: int = 1
    blocks: int = 1
    mask: int = 2
    expected: int = 1
    epoch: int = 1
    delay_ms: int = 0
    tail_ms: int = 0
    sync_start: bool = False
    instances: int = 1
    skip: bool = False
    capacity_delta: int = 0
    initial_signal: int = 0


def payload(epoch: int, execution: int) -> torch.Tensor:
    return torch.arange(PAYLOAD_WORDS, dtype=torch.int32) + epoch * 100000 + execution * PAYLOAD_WORDS + 1


def stage_host_tensor(buffer, tensor):
    """L3 control-plane copies require a named Buffer, not a raw torch address."""
    torch.frombuffer(buffer.shm.buf, dtype=tensor.dtype, count=tensor.numel()).copy_(tensor.flatten())


def read_host_tensor(buffer, template):
    # Clone so assertions never retain a view after the owning Worker closes.
    return (
        torch.frombuffer(buffer.shm.buf, dtype=template.dtype, count=template.numel()).clone().reshape(template.shape)
    )


def error_log_cursor():
    """Bound diagnostics belong to the invocation, including when stdout omits native logs."""
    return {path: path.stat().st_size for path in _bound_host_logs()}


def failure_evidence(error, captured, cursor):
    evidence = [str(error), captured.out, captured.err]
    for path in _bound_host_logs():
        with path.open("rb") as stream:
            stream.seek(cursor.get(path, 0))
            evidence.append(stream.read().decode(errors="replace"))
    return "\n".join(evidence)


def close_after_failure(worker):
    try:
        worker.close()
    except Exception as cleanup_error:
        warnings.warn(f"async test cleanup also failed: {cleanup_error}", RuntimeWarning, stacklevel=2)


def _active_executions(case: Case, capacity_waiters: int, actual_waiters: int):
    for instance in range(case.instances):
        base = instance * capacity_waiters * case.blocks * 3
        for waiter in range(actual_waiters):
            for block in range(case.blocks):
                for lane in range(3):
                    if case.mask & (1 << lane):
                        yield instance, waiter, base + waiter * case.blocks * 3 + block * 3 + lane


def assert_observations(case: Case, capacity_waiters: int, records, output, control) -> None:
    """Expected executions come from the input graph, never from runtime completion counters."""
    assert not torch.count_nonzero(records[:, 2]), (
        "resident deferred context is unavailable: this is missing async support, not an accepted synchronous fallback"
    )
    assert not torch.count_nonzero(records[:, 3]), "a consumer entered before its conditions or dependencies"
    assert not torch.count_nonzero(records[:, 4]), "test producer could not observe registration progress"
    assert not torch.count_nonzero(control[:, 4, 0]), (
        "rescue publisher was required: progress or cohort contract failed"
    )
    if case.scenario == Scenario.PROGRESS:
        # Capacity is the public orchestration API's topology input, not an async queue limit.
        capacities = control[:, 5, 0].tolist()
        assert len(set(capacities)) == 1 and 0 < capacities[0] <= capacity_waiters
        actual_waiters = int(capacities[0]) + case.capacity_delta
        assert 0 < actual_waiters <= capacity_waiters
    else:
        actual_waiters = case.waiters

    slots = case.instances * capacity_waiters * case.blocks * 3
    logical_capacity = case.instances * capacity_waiters
    golden_records = torch.zeros_like(records)
    golden_output = torch.full_like(output, POISON)
    for instance, waiter, execution in _active_executions(case, capacity_waiters, actual_waiters):
        if not case.skip:
            golden_records[execution, :2] = 1
        golden_output[execution] = payload(case.epoch, execution) + 7
        consumer = slots + instance * capacity_waiters + waiter
        golden_records[consumer, :2] = 1
    for instance in range(case.instances):
        publisher = slots + logical_capacity + instance
        if not case.skip and case.scenario != Scenario.NEVER:
            golden_records[publisher, :2] = 1
        if case.scenario in (Scenario.FANIN, Scenario.GRAPH):
            golden_records[publisher + case.instances, :2] = 1
    torch.testing.assert_close(records, golden_records, rtol=0, atol=0)
    torch.testing.assert_close(output, golden_output, rtol=0, atol=0)
    if not case.skip:
        expected_executions = actual_waiters * case.blocks * case.mask.bit_count()
        assert torch.all(control[:, 0, 0] == expected_executions), "registration attempt missing or duplicated"
    if case.scenario == Scenario.VISIBILITY:
        assert torch.all(control[:, 3, 0] == control[:, 5, 0]), "not every consumer-capable lane warmed its cache"


def assert_resident_capture(path: Path) -> None:
    captures = sorted(path.rglob("chip_swimlane_records.json"))
    assert captures, f"missing invocation evidence under {path}"
    for capture in captures:
        raw = json.loads(capture.read_text())
        protocol = raw.get("aicore_protocol", {})
        assert protocol.get("producer") == "aicore", f"not an AICore resident invocation: {capture}"
        assert raw.get("aicore_tasks"), f"no kernel executions in {capture}"


def _counter_callable_spec():
    return {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/counter.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 5 + [D.SCALAR] * 15,
        },
        "incores": [
            {
                "func_id": index,
                "source": str(HERE / "kernels" / source),
                "core_type": core,
                "signature": [D.INOUT] * (2 if index == 10 else 5),
                # Zero-condition invocations retain this source-level bound;
                # the 65-condition case must still exceed the declared 64 slots.
                "async_capability": {
                    "mode": "DEFERRED",
                    "max_conditions": 1 if index == 10 else CONDITION_LIMIT,
                    "condition_types": CORE_ASYNC_CONDITION_COUNTER,
                }
                if index < 3 or index in (9, 10, 11, 12)
                else {"mode": "SYNCHRONOUS"},
            }
            for index, source, core in (
                (0, "aic/waiter.cpp", "aic"),
                (1, "aiv/waiter_0.cpp", "aiv"),
                (2, "aiv/waiter_1.cpp", "aiv"),
                (3, "aiv/action.cpp", "aiv"),
                (4, "aic/action.cpp", "aic"),
                (5, "aiv/action.cpp", "aiv"),
                (6, "aiv/action.cpp", "aiv"),
                (7, "aic/action.cpp", "aic"),
                (8, "aic/action.cpp", "aic"),
                (9, "aiv/graph_waiter.cpp", "aiv"),
                (10, "aiv/registration_gate.cpp", "aiv"),
                (11, "aic/registered_waiter.cpp", "aic"),
                (12, "aiv/registered_waiter_1.cpp", "aiv"),
            )
        ],
    }


@cache
def compile_counter(platform: str):
    return compile_chip_callable_spec(
        _counter_callable_spec(),
        platform,
        "host_build_graph",
        (str(HERE), platform, "counter", _pto_isa_compile_cache_token()),
    )


class CounterRunner:
    """Own an L3 worker so close tests cannot terminate another case's pooled worker."""

    _compile_callable = staticmethod(compile_counter)

    def __init__(self, platform: str, devices: list[int], directory: Path, case: Case):
        self.case = case
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.compiled = self._compile_callable(platform)
        # This is a fixture storage bound. Actual progress widths use rt_available_aiv_count().
        self.capacity_waiters = 256 if case.scenario == Scenario.PROGRESS else case.waiters
        self.slots = case.instances * self.capacity_waiters * case.blocks * 3
        rows = self.slots + case.instances * self.capacity_waiters + 2 * case.instances
        self.records = torch.zeros((rows, OBSERVATION_WORDS), dtype=torch.int64).share_memory_()
        self.output = torch.full((self.slots, PAYLOAD_WORDS), POISON, dtype=torch.int32).share_memory_()
        self.control = torch.zeros((case.instances, CONTROL_CELLS, CONTROL_STRIDE), dtype=torch.int64).share_memory_()
        self.initial_signals = torch.full(
            (self.slots * CONDITIONS_PER_EXECUTION, SIGNAL_STRIDE), POISON, dtype=torch.int32
        ).share_memory_()
        self.initial_payload = torch.full((self.slots, PAYLOAD_WORDS), POISON, dtype=torch.int32).share_memory_()
        self.worker = Worker(
            level=3, platform=platform, runtime="host_build_graph", device_ids=[int(devices[0])], num_sub_workers=0
        )
        self.iteration = 0
        self.closed = False
        self.callable_handle = self.worker.register(self.compiled)
        try:
            self.worker.init()
            self.signal_staging = self.worker.create_buffer(self.initial_signals.nbytes)
            self.payload_staging = self.worker.create_buffer(self.initial_payload.nbytes)
        except BaseException:
            close_after_failure(self.worker)
            raise

    def _args(self, domain, case: Case):
        args = TaskArgs()
        args.add_tensor(
            domain.buffers["signals"].tensor(tuple(self.initial_signals.shape), DataType.INT32), TensorArgType.INOUT
        )
        args.add_tensor(
            domain.buffers["payload"].tensor(tuple(self.initial_payload.shape), DataType.INT32), TensorArgType.INOUT
        )
        for tensor, dtype in (
            (self.records, DataType.INT64),
            (self.output, DataType.INT32),
            (self.control, DataType.INT64),
        ):
            args.add_tensor(self.worker.make_tensor_arg(tensor, tuple(tensor.shape), dtype), TensorArgType.INOUT)
        for value in (
            case.scenario,
            case.conditions,
            case.waiters,
            case.blocks,
            case.mask,
            case.expected,
            case.epoch,
            case.delay_ms,
            case.tail_ms,
            case.sync_start,
            case.instances,
            case.skip,
            case.capacity_delta,
            self.slots,
            self.iteration,
        ):
            args.add_scalar(int(value))
        return args

    def submit(self, case: Case | None = None, *, reset_window: bool = True):
        case = case or self.case
        self.records.zero_()
        self.control.zero_()
        self.output.fill_(POISON)
        self.initial_signals.fill_(POISON)
        self.initial_signals[:, 0] = case.initial_signal
        # Registration uses an invocation sequence, independent of repeated
        # business epochs. Only an explicit window reset initializes its slots.
        self.initial_signals[CONDITION_LIMIT::CONDITIONS_PER_EXECUTION, 0] = 0
        self.initial_payload.fill_(POISON)
        if case.conditions == 0 or case.initial_signal >= case.expected or case.skip:
            for _, _, execution in _active_executions(case, self.capacity_waiters, case.waiters):
                self.initial_payload[execution] = payload(case.epoch, execution)
        self.iteration += 1
        self.capture = self.directory / f"run_{self.iteration}"
        self.capture.mkdir()
        config = CallConfig()
        config.enable_chip_swimlane = 4
        config.output_prefix = str(self.capture)

        def orchestrate(orch, _args, _config):
            buffers = [
                CommBufferSpec(
                    name="signals",
                    dtype="int32",
                    count=self.initial_signals.numel(),
                    nbytes=self.initial_signals.nbytes,
                ),
                CommBufferSpec(
                    name="payload",
                    dtype="int32",
                    count=self.initial_payload.numel(),
                    nbytes=self.initial_payload.nbytes,
                ),
            ]
            domain = orch.get_or_allocate_domain(
                name="async_completion_contract",
                members=[0],
                window_size=sum(item.nbytes for item in buffers),
                buffers=buffers,
                retain_after_run=True,
            )
            if reset_window:
                stage_host_tensor(self.signal_staging, self.initial_signals)
                stage_host_tensor(self.payload_staging, self.initial_payload)
                orch.copy_to(domain[0].buffers["signals"], self.signal_staging)
                orch.copy_to(domain[0].buffers["payload"], self.payload_staging)
            orch.submit_next_level(self.callable_handle, self._args(domain[0], case), config, worker=0)

        return self.worker.submit(orchestrate, config=config)

    def check(self, case: Case | None = None, *, check_guards: bool = True):
        case = case or self.case
        assert_observations(case, self.capacity_waiters, self.records, self.output, self.control)
        assert_resident_capture(self.capture)
        if check_guards:
            domain = self.worker.live_domains["async_completion_contract"]
            self.worker.copy_from(self.signal_staging, domain[0].buffers["signals"])
            copied = read_host_tensor(self.signal_staging, self.initial_signals)
            assert torch.all(copied[:, 1:] == POISON), "condition observation/publication overwrote adjacent bytes"

    def run(self, case: Case | None = None, *, reset_window: bool = True):
        handle = self.submit(case, reset_window=reset_window)
        handle.wait(timeout=30)
        self.check(case)
        return handle

    def require_support(self):
        probe = replace(self.case, scenario=Scenario.BASIC, conditions=1, delay_ms=0, tail_ms=0, skip=False)
        self.run(probe)

    def close(self):
        if not self.closed:
            self.worker.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self.close()
        else:
            close_after_failure(self.worker)


@cache
def compile_sdma(platform: str, *, declared: bool = True):
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/sdma.cpp"),
            "function_name": "aicpu_orchestration_entry",
            # Upload the result poison so unwritten guards have defined device contents.
            "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": index,
                "source": str(HERE / "kernels/aiv" / source),
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.OUT, D.INOUT] if index != 2 else [D.INOUT],
                "async_capability": {
                    "mode": "DEFERRED",
                    "max_conditions": CONDITION_LIMIT,
                    "condition_types": CORE_ASYNC_CONDITION_SDMA_POST_DONE,
                }
                if index == 0
                else {"mode": "SYNCHRONOUS"},
            }
            for index, source in enumerate(("sdma_submit.cpp", "sdma_consumer.cpp", "sdma_probe.cpp"))
        ],
    }
    if not declared:
        spec["orchestration"].update(
            function_name="unannotated_orchestration_entry", config_name="aicpu_orchestration_config"
        )
        spec["incores"][0]["source"] = str(HERE / "kernels/aiv/sdma_submit_unannotated.cpp")
        for kernel in spec["incores"]:
            kernel.pop("async_capability")
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (str(HERE), platform, "sdma", declared, _pto_isa_compile_cache_token())
    )


def run_sdma(platform: str, devices: list[int], directory: Path, elements: int, *, declared: bool = True):
    assert len(devices) >= 2, "real peer SDMA requires two assigned devices"
    compiled = compile_sdma(platform, declared=declared)
    # CommDomain buffers are 64-byte aligned; leave at least 32 guard elements.
    allocated = ((elements + 32 + 15) // 16) * 16
    sources = [
        ((torch.arange(allocated, dtype=torch.float32) % 251) + (rank + 1) * 1000).share_memory_() for rank in range(2)
    ]
    initial = [torch.full((allocated,), -999.0, dtype=torch.float32).share_memory_() for _ in range(2)]
    results = [torch.full((allocated,), -999.0, dtype=torch.float32).share_memory_() for _ in range(2)]
    metadata = [torch.zeros(256, dtype=torch.int64).share_memory_() for _ in range(2)]
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
    capture = directory / "sdma"
    capture.mkdir(parents=True)
    config = CallConfig()
    config.enable_chip_swimlane = 4
    config.output_prefix = str(capture)

    def orchestrate(orch, _args, _config):
        buffers = [
            CommBufferSpec(name=name, dtype="float32", count=allocated, nbytes=allocated * 4)
            for name in ("source", "destination")
        ]
        domain = orch.allocate_domain(
            name="async_sdma_contract",
            members=[0, 1],
            window_size=allocated * 8,
            buffers=buffers,
            retain_after_run=True,
            workspace_engines=("sdma",),
            sdma_sdk_queue_num=1,
        )
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
            args.add_tensor(worker.make_tensor_arg(metadata[rank], (256,), DataType.INT64), TensorArgType.INOUT)
            args.add_scalar(domain[rank].device_ctx)
            args.add_scalar(elements)
            args.add_scalar(1 - rank)
            rank_config = CallConfig()
            rank_config.enable_chip_swimlane = 4
            rank_config.output_prefix = str(capture / f"rank_{rank}")
            Path(rank_config.output_prefix).mkdir()
            orch.submit_next_level(callable_handle, args, rank_config, worker=rank)

    try:
        worker.submit(orchestrate, config=config).wait(timeout=30)
        for rank in range(2):
            meta = metadata[rank]
            assert int(meta[16]) == 0, "resident deferred context unavailable; no SDMA support was exercised"
            assert int(meta[32]) == 0, "SDMA submission, public record export, or probe failed"
            assert int(meta[48]) == 0, "SDMA consumer entered before completion"
            golden = torch.full_like(results[rank], -999.0)
            golden[:elements] = sources[1 - rank][:elements] + 7
            torch.testing.assert_close(results[rank], golden, rtol=0, atol=0)
            worker.copy_from(
                destination_staging[rank], worker.live_domains["async_sdma_contract"][rank].buffers["destination"]
            )
            transferred = read_host_tensor(destination_staging[rank], initial[rank])
            golden_transfer = initial[rank].clone()
            golden_transfer[:elements] = sources[1 - rank][:elements]
            torch.testing.assert_close(transferred, golden_transfer, rtol=0, atol=0)
            rank_path = capture / f"rank_{rank}"
            assert_resident_capture(rank_path)
            if elements == 1024 * 1024:
                files = list(rank_path.rglob("chip_swimlane_records.json"))
                assert len(files) == 1, files
                raw = json.loads(files[0].read_text())
                producer_traces = {
                    task["trace_base"] for task in raw["aicore_protocol"]["tasks"] if 0 in task["kernel_ids"]
                }
                producer_rows = [row for row in raw["aicore_tasks"] if row[2] in producer_traces]
                assert len(producer_rows) == 1
                assert int(meta[64]) > 0 and int(meta[80]) > producer_rows[0][4], (
                    "no real SDMA PENDING observation after kernel return; numerical correctness alone is insufficient"
                )
    finally:
        if sys.exc_info()[0] is None:
            worker.close()
        else:
            close_after_failure(worker)
