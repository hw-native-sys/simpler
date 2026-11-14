#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from functools import cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec
from simpler_setup.tools.hbg.native_protocol import has_native_graph, protocol_version
from simpler_setup.tools.swimlane_converter import read_perf_data

from . import helpers
from .helpers import POISON, Case, CounterRunner, Scenario, l3_case, read_host_tensor

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]

HERE = Path(__file__).resolve().parent


@cache
def _compile_omitted_registration(platform):
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/counter.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 5 + [D.SCALAR] * 14,
        },
        "incores": [
            {
                "func_id": index,
                "source": str(HERE / "kernels" / source),
                "core_type": core,
                "signature": [D.INOUT] * 5,
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
            )
        ],
    }
    waiter = next(kernel for kernel in spec["incores"] if kernel["func_id"] == 1)
    waiter["source"] = str(HERE / "kernels/aiv/waiter_omit_registration.cpp")
    return compile_chip_callable_spec(
        spec,
        platform,
        "host_build_graph",
        (str(HERE), platform, "omitted_registration_negative_control", _pto_isa_compile_cache_token()),
    )


def _require_completed_empty_capture(runner):
    captures = list(runner.capture.rglob("chip_swimlane_records.json"))
    assert len(captures) == 1, captures
    raw = json.loads(captures[0].read_text())
    protocol = raw["aicore_protocol"]
    assert protocol_version(protocol) and not has_native_graph(protocol)
    decoded = read_perf_data(captures[0])
    assert decoded["protocol_audit"]["protocol_verified"]
    assert protocol["scheduler_error"] == protocol["error_claimed"] == 0
    assert all(worker["error_flags"] == 0 for worker in protocol["workers"])
    tasks = protocol["tasks"]
    assert len(tasks) == len(decoded["tasks"]) == len(protocol["executions"]) == 2
    waiter, consumer = tasks
    assert waiter["task_ref"] == 0 and waiter["kernel_ids"] == [65535, 1, 65535]
    assert consumer["task_ref"] == 1 and consumer["kernel_ids"] == [65535, 5, 65535]
    assert waiter["dependencies"] == [] and consumer["dependencies"] == [0]
    assert {task["task_id"] for task in decoded["tasks"]} == {0, 1}
    # Zero actual registrations use the synchronous completion path. No CHECK
    # or FINISH is required; successful ordinary completion alone cannot prove
    # that a business-condition registration was present.
    (runner.capture / "omitted-registration-audit.json").write_text(
        json.dumps(decoded["protocol_audit"], indent=2) + "\n"
    )


def _require_only_early_consumer_failure(runner):
    expected_records = torch.zeros_like(runner.records)
    expected_records[1, :2] = 1
    expected_records[runner.slots, :2] = 1
    expected_records[runner.slots, 3] = 1
    torch.testing.assert_close(runner.records, expected_records, rtol=0, atol=0)
    expected_control = torch.zeros_like(runner.control)
    expected_control[0, 0, 0] = 1
    torch.testing.assert_close(runner.control, expected_control, rtol=0, atol=0)
    assert torch.all(runner.output == POISON)
    domain = runner.worker.live_domains["async_completion_contract"]
    for name, staging, initial in (
        ("signals", runner.signal_staging, runner.initial_signals),
        ("payload", runner.payload_staging, runner.initial_payload),
    ):
        runner.worker.copy_from(staging, domain[0].buffers[name])
        torch.testing.assert_close(read_host_tensor(staging, initial), initial, rtol=0, atol=0)
    with pytest.raises(AssertionError) as failure:
        runner.check()
    assert str(failure.value) == "a consumer entered before its conditions or dependencies"
    (runner.capture / "business-oracle-rejection.txt").write_text(str(failure.value) + "\n")


@l3_case
def test_business_oracle_rejects_omitted_registration(st_platform, st_device_ids, tmp_path, monkeypatch):
    with CounterRunner(st_platform, st_device_ids, tmp_path / "normal_count1", Case()) as normal:
        normal.run()
    case = Case(scenario=Scenario.NEVER)
    monkeypatch.setattr(helpers, "compile_counter", _compile_omitted_registration)
    with CounterRunner(st_platform, st_device_ids, tmp_path / "omitted_registration", case) as broken:
        # NEVER has no publisher: 0 < 1 remains true independently of scheduling.
        # Any runtime failure or timeout here fails the experiment, not its oracle.
        broken.submit().wait(timeout=30)
        _require_completed_empty_capture(broken)
        _require_only_early_consumer_failure(broken)
