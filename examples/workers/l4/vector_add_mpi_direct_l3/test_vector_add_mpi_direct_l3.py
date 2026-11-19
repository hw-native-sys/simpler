# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Network1 ST for the complete L4-to-L3 direct MPI path."""

import importlib.util
import os
import shutil

import pytest

from simpler_setup import SceneTestLevel, scene_level

from .main import run


def _device_spec(device_ids) -> str:
    return ",".join(str(device_id) for device_id in device_ids)


def _require_mpi_direct_network1_env() -> tuple[str, str, str]:
    mpi_launcher = shutil.which("mpirun")
    if mpi_launcher is None:
        mpi_launcher = shutil.which("mpiexec")
    if mpi_launcher is None:
        pytest.skip("mpirun or mpiexec is not on PATH")
    assert mpi_launcher is not None
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("mpi4py is not installed")
    local_ip = os.environ.get("NETWORK1_LOCAL_IP", "")
    if not local_ip:
        pytest.skip("NETWORK1_LOCAL_IP is required for the L4/rank-0 startup gate")
    mpi_python = os.environ.get("NETWORK1_MPI_PYTHON", "")
    if not mpi_python:
        pytest.skip("NETWORK1_MPI_PYTHON is required on both MPI hosts")
    return mpi_launcher, local_ip, mpi_python


@scene_level(SceneTestLevel.NETWORK1)
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.network1_remote_device_count(2)
def test_vector_add_mpi_direct_l3(
    st_platform, st_device_ids, st_network1_peer, st_network1_remote_device_ids, st_network1_logs
):
    mpirun, local_ip, mpi_python = _require_mpi_direct_network1_env()
    remote_host, _daemon_port = st_network1_peer.endpoint.rsplit(":", 1)
    rc = run(
        local_host=local_ip,
        remote_host=remote_host,
        python_executable=mpi_python,
        local_devices=_device_spec(st_device_ids),
        remote_devices=_device_spec(st_network1_remote_device_ids),
        platform=st_platform,
        startup_timeout=st_network1_peer.session_timeout_s,
        session_timeout=st_network1_peer.session_timeout_s,
        mpirun_path=mpirun,
    )
    assert rc == 0
