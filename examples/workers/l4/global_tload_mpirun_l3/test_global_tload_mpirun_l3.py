# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Network1 ST for examples/workers/l4/global_tload_mpirun_l3.

Runs the real two-machine mpirun path on the network1 pair: one ``mpirun`` from
the parent machine launches an L3 rank per machine, the parent dispatches a
TLOAD task per NPU device on each machine, and the full-group
``a3-fabric-v1`` Global CommDomain built over the MPI descriptor exchange
carries the cross-machine peer reads.

On top of the standard network1 runner environment this test needs ``mpirun`` +
``mpi4py`` on both machines and ``NETWORK1_MPI_PYTHON``: one absolute path, valid
on BOTH machines, of a per-machine launcher that sources CANN, enters that
machine's copy of this run's tree, and execs its ``.venv`` python (a single
``mpirun`` command line launches every rank, so the interpreter path and the
rank-side ``examples...`` import must resolve on each machine).
"""

import importlib.util
import os
import shutil

import pytest

from simpler_setup import SceneTestLevel, scene_level

from .main import run

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]


def _device_spec(device_ids) -> str:
    return ",".join(str(device_id) for device_id in device_ids)


def _require_mpirun_network1_env() -> tuple[str, str, str]:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not on PATH")
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("mpi4py is not installed (required inside the mpirun rank processes)")
    local_ip = os.environ.get("NETWORK1_LOCAL_IP", "")
    if not local_ip:
        pytest.skip("NETWORK1_LOCAL_IP is required: mpirun places rank 0 on this machine's address")
    mpi_python = os.environ.get("NETWORK1_MPI_PYTHON", "")
    if not mpi_python:
        pytest.skip("NETWORK1_MPI_PYTHON is required: per-machine rank launcher at one shared absolute path")
    return mpirun, local_ip, mpi_python


@scene_level(SceneTestLevel.NETWORK1)
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.network1_remote_device_count(2)
def test_global_tload_mpirun_l3(
    st_platform, st_device_ids, st_network1_peer, st_network1_remote_device_ids, st_network1_logs
):
    mpirun, local_ip, mpi_python = _require_mpirun_network1_env()
    remote_host, _daemon_port = st_network1_peer.endpoint.rsplit(":", 1)
    rc = run(
        local_host=local_ip,
        remote_host=remote_host,
        python_executable=mpi_python,
        local_devices=_device_spec(st_device_ids),
        remote_devices=_device_spec(st_network1_remote_device_ids),
        platform=st_platform,
        session_timeout=st_network1_peer.session_timeout_s,
        mpirun_path=mpirun,
    )
    assert rc == 0
