# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Finite managed-workspace refusal with retained graphs, not device OOM."""

import re

import pytest

from tests.st.a2a3.kernel_capture._capture_capacity import build_probe, probe_output, run_probe, write_input_identities

pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3", "a5"]),
    pytest.mark.runtime_product("kernel"),
    pytest.mark.execution_mode("capture_replay"),
]


@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
        pytest.param("tensormap_and_ringbuffer", marks=pytest.mark.runtime("tensormap_and_ringbuffer")),
    ],
)
def test_finite_workspace_refusal_preserves_graphs(tmp_path, request, runtime):
    command, identity = build_probe(
        tmp_path / "build", request.config.getoption("--platform"), runtime, "capture_capacity"
    )
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    output = probe_output(f"capacity-{runtime}-device-{device}-")
    write_input_identities(output, identity)
    stdout = run_probe(command, device, output)
    assert re.search(
        rf"CAPTURE_WORKSPACE_CAPACITY_PASS runtime={runtime} budget=268435456 "
        r"models=3 grants=2 offset_views=2 refused=-1007 recovered=1",
        stdout,
    )
    for stage in (
        "launch-observer-reached",
        "refusal-before-kernel-submission",
        "refusal-budget-balanced",
        "abort-only-refused-plan",
        "refusal-keeps-live-callable",
        "readmit-small-capture",
        "workspace-reclaimed",
    ):
        assert f"stage={stage} rc=0" in stdout


@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
        pytest.param("tensormap_and_ringbuffer", marks=pytest.mark.runtime("tensormap_and_ringbuffer")),
    ],
)
def test_cumulative_workspace_release_readmits_same_request(tmp_path, request, runtime):
    command, identity = build_probe(
        tmp_path / "build", request.config.getoption("--platform"), runtime, "capture_capacity_pressure"
    )
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    output = probe_output(f"cumulative-{runtime}-device-{device}-")
    write_input_identities(output, identity)
    stdout = run_probe(command, device, output)
    match = re.search(
        rf"CAPTURE_CUMULATIVE_CAPACITY_PASS runtime={runtime} budget=268435456 request=33554432 "
        r"accepted=([3-8]) maximum_models=9 grants=2 refused=-1007 same_pressure_recovered=1",
        stdout,
    )
    assert match, stdout
    assert stdout.count("stage=retained-budget-increased rc=0") == int(match.group(1))
    for stage in (
        "at-least-three-live-models",
        "refusal-before-kernel-submission",
        "abort-only-refused-plan",
        "refusal-budget-balanced",
        "bounded-cumulative-refusal",
        "same-pressure-readmitted",
        "workspace-reclaimed",
    ):
        assert f"stage={stage} rc=0" in stdout
