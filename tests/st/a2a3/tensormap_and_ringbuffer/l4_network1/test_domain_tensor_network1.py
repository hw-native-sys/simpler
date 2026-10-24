# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One domain-Tensor contract on Fabric-local, TCP and MPI Network1 topologies."""

import pytest
from simpler.worker import MpiL3GroupSpec, RemoteWorkerSpec, Worker

from examples.workers.l3.multi_chip_dispatch.main import build_chip_callable
from examples.workers.l4.global_tload_mpirun_l3.test_global_tload_mpirun_l3 import _require_mpirun_network1_env
from simpler_setup import SceneTestLevel, scene_level
from tests.st.worker.comm_domain.test_domain_tensor_remote import exercise_domain_tensors
from tests.st.worker.comm_domain.test_unified_domain import exercise_local_domain

pytestmark = [
    pytest.mark.platforms(["a2a3"]),
    pytest.mark.runtime("tensormap_and_ringbuffer"),
    pytest.mark.device_count(2),
    pytest.mark.network1_remote_device_count(2),
]


@scene_level(SceneTestLevel.NETWORK1)
def test_domain_tensor_local(st_platform, st_device_ids, st_network1_logs):
    exercise_local_domain(st_platform, st_device_ids)


@scene_level(SceneTestLevel.NETWORK1)
def test_domain_tensor_tcp(
    st_platform, st_device_ids, st_network1_peer, st_network1_remote_device_ids, st_network1_logs
):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=st_network1_peer.session_timeout_s)
    callable_obj = build_chip_callable(st_platform)
    child = Worker(
        level=3,
        platform=st_platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=st_device_ids,
        num_sub_workers=0,
        comm_profile="a3-fabric-v1",
        global_device_ranks=(0, 1),
    )
    chip = child.register(callable_obj)
    local = worker.add_worker(child)
    remote = worker.add_remote_worker(
        RemoteWorkerSpec(
            endpoint=st_network1_peer.endpoint,
            platform=st_platform,
            device_ids=tuple(st_network1_remote_device_ids),
            comm_profile="a3-fabric-v1",
            global_device_ranks=(2, 3),
            session_listen_host=st_network1_peer.session_listen_host,
            allow_wildcard_session_bind=True,
        )
    )
    members = [(node, chip_index) for node in (local, remote) for chip_index in range(2)]
    exercise_domain_tensors(worker, members, callable_obj, local_chip=(local, chip))


@scene_level(SceneTestLevel.NETWORK1)
def test_domain_tensor_mpi(
    st_platform, st_device_ids, st_network1_peer, st_network1_remote_device_ids, st_network1_logs
):
    mpirun, local_ip, python = _require_mpirun_network1_env()
    remote_host, _port = st_network1_peer.endpoint.rsplit(":", 1)
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=st_network1_peer.session_timeout_s)
    nodes = worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=(local_ip, remote_host),
            platform=st_platform,
            device_ids_by_rank=(tuple(st_device_ids), tuple(st_network1_remote_device_ids)),
            comm_profile="a3-fabric-v1",
            global_device_ranks_by_rank=((0, 1), (2, 3)),
            mpirun_path=mpirun,
            mpirun_args=("-wdir", "/tmp"),
            python_executable=python,
        )
    )
    members = [(node, chip_index) for node in nodes for chip_index in range(2)]
    exercise_domain_tensors(worker, members, build_chip_callable(st_platform))
