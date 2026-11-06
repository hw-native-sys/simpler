#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Omitted capability and resource declarations keep runtime registration usable."""

from functools import cache

import pytest

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec

from .helpers import (
    HERE,
    Case,
    CounterRunner,
    Scenario,
    _counter_callable_spec,
    close_after_failure,
    error_log_cursor,
    failure_evidence,
    l3_case,
)

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]


@cache
def compile_unknown(platform):
    spec = _counter_callable_spec()
    spec["orchestration"].update(
        function_name="unannotated_orchestration_entry", config_name="aicpu_orchestration_config"
    )
    for kernel in spec["incores"]:
        kernel.pop("async_capability")
    return compile_chip_callable_spec(
        spec,
        platform,
        "host_build_graph",
        (str(HERE), platform, "unknown-default", _pto_isa_compile_cache_token()),
    )


class UnknownRunner(CounterRunner):
    _compile_callable = staticmethod(compile_unknown)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case(conditions=0), id="zero-count"),
        pytest.param(Case(conditions=1, delay_ms=20), id="pending-counter"),
        pytest.param(Case(conditions=64, delay_ms=20), id="capacity-64"),
        pytest.param(Case(blocks=3, mask=7, conditions=2, delay_ms=20), id="mixed-spmd"),
        pytest.param(Case(blocks=3, mask=7, sync_start=1, delay_ms=20), id="sync-start"),
        pytest.param(Case(scenario=Scenario.PROGRESS, capacity_delta=1, delay_ms=20), id="more-waiters-than-cores"),
        pytest.param(Case(scenario=Scenario.GRAPH, instances=2, delay_ms=20), id="two-graph-instances"),
        pytest.param(Case(conditions=0, skip=True), id="skipped"),
    ],
)
@l3_case
def test_unknown_kernel_uses_default_registration(st_platform, st_device_ids, tmp_path, case):
    with UnknownRunner(st_platform, st_device_ids, tmp_path, case) as runner:
        runner.run()


@l3_case
def test_unknown_kernel_capacity_overflow_is_reported(st_platform, st_device_ids, tmp_path, capfd):
    runner = UnknownRunner(st_platform, st_device_ids, tmp_path, Case(conditions=65))
    try:
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as failure:
            runner.submit().wait(timeout=30)
        evidence = failure_evidence(failure.value, capfd.readouterr(), cursor)
        (tmp_path / "overflow.log").write_text(evidence)
        # A retained external/unknown run reports its validated signed terminal
        # code before refusing resource retirement; it need not reach copy-back.
        assert "ASYNC_WAIT_OVERFLOW" in evidence or "terminal_code=-102 " in evidence, evidence
    finally:
        # An unknown kernel may have submitted external work before the error.
        # Preserve the failure path's resource-retention verdict during cleanup.
        close_after_failure(runner.worker)
