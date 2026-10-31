# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host DRAM and HBM access use the same matrix as the MPI hardware runner."""

from __future__ import annotations

import json

import pytest
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from tests.st.worker.comm_domain import test_domain_tensor_remote as remote_domain
from tests.st.worker.comm_domain.host_domain_matrix import MODULE, _build_kernel, node_orch, run_matrix

_RUNTIME = "tensormap_and_ringbuffer"
tcp_daemon = remote_domain.tcp_daemon


def _assert_matrix(output, nodes, checks, handoffs):
    summary = json.loads(output.splitlines()[-1])
    assert output.count("CASE_PASS ") == checks
    assert summary["result"] == "MATRIX_PASS" and summary["checks"] == checks
    assert summary["nodes"] == nodes and summary["domain_pools"] == 3 * nodes
    assert summary["handoff_steps"] == handoffs and summary["released"]


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.parametrize(
    "nodes,checks,handoffs",
    [
        pytest.param(1, 18, 7, marks=pytest.mark.device_count(2), id="1"),
        pytest.param(2, 72, 18, marks=pytest.mark.device_count(4), id="2"),
    ],
)
def test_host_domain_full_access_matrix(st_platform, st_device_ids, capsys, nodes, checks, handoffs):
    worker = Worker(level=4, num_sub_workers=0)
    chips = []
    try:
        callable_obj = _build_kernel(st_platform)
        for node in range(nodes):
            child = Worker(
                level=3,
                platform=st_platform,
                runtime=_RUNTIME,
                device_ids=[int(device) for device in st_device_ids[2 * node : 2 * node + 2]],
                num_sub_workers=0,
                comm_profile="sim",
                global_device_ranks=[2 * node, 2 * node + 1],
            )
            chips.append(child.register(callable_obj))
            assert worker.add_worker(child) == node
        kernel = worker.register(callable_obj)

        def forward(orch, args, cfg):
            node_orch(orch, args, cfg, chip_handle=chips[int(args.scalar(1))])

        route = worker.register(forward)
        worker.init()
        run_matrix(worker, nodes, kernel, route)
        _assert_matrix(capsys.readouterr().out, nodes, checks, handoffs)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(4)
def test_host_domain_full_access_matrix_through_tcp(tcp_daemon, st_platform, st_device_ids, capsys):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30)
    try:
        for node in range(2):
            assert (
                worker.add_remote_worker(
                    RemoteWorkerSpec(
                        endpoint=tcp_daemon,
                        platform=st_platform,
                        runtime=_RUNTIME,
                        device_ids=tuple(int(device) for device in st_device_ids[2 * node : 2 * node + 2]),
                        comm_profile="sim",
                        global_device_ranks=(2 * node, 2 * node + 1),
                    )
                )
                == node
            )
        kernel = worker.register(_build_kernel(st_platform))
        route = worker.register(RemoteCallable(MODULE + ":node_orch"), workers=[0, 1])
        worker.init()
        run_matrix(worker, 2, kernel, route)
        _assert_matrix(capsys.readouterr().out, 2, 72, 18)
    finally:
        worker.close()
