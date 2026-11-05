# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The same domain Tensor and context flow on Fabric-local, TCP and MPI."""

import pytest
from simpler.worker import MpiL3GroupSpec, RemoteWorkerSpec, Worker

from examples.workers.l3.multi_chip_dispatch.main import build_chip_callable
from examples.workers.l4.global_tload_mixed_l3.main import _build_tload_callable
from examples.workers.l4.global_tload_mpirun_l3.test_global_tload_mpirun_l3 import _require_mpirun_network1_env
from simpler_setup import SceneTestLevel, scene_level
from tests.st.worker.comm_domain.test_domain_tensor_remote import exercise_domain_tensors

pytestmark = [
    pytest.mark.platforms(["a2a3"]),
    pytest.mark.runtime("tensormap_and_ringbuffer"),
    pytest.mark.device_count(2),
    pytest.mark.network1_remote_device_count(2),
]


@scene_level(SceneTestLevel.NETWORK1)
@pytest.mark.parametrize("topology", ["local", "tcp", "mpi"])
def test_domain_tensor_network1(
    topology, st_platform, st_device_ids, st_network1_peer, st_network1_remote_device_ids, st_network1_logs
):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=st_network1_peer.session_timeout_s)
    local = None
    if topology == "mpi":
        mpirun, local_ip, python = _require_mpirun_network1_env()
        remote_host, _port = st_network1_peer.endpoint.rsplit(":", 1)
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
    else:
        child = Worker(
            level=3,
            platform=st_platform,
            runtime="tensormap_and_ringbuffer",
            device_ids=st_device_ids,
            num_sub_workers=0,
            comm_profile="a3-fabric-v1",
            global_device_ranks=(0, 1),
        )
        chips = tuple(
            child.register(obj)
            for obj in (
                build_chip_callable(st_platform),
                _build_tload_callable(st_platform, "tensormap_and_ringbuffer"),
            )
        )
        node = worker.add_worker(child)
        local, nodes = (node, chips), [node]
        if topology == "tcp":
            nodes.append(
                worker.add_remote_worker(
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
            )
    members = [(node, chip) for node in nodes for chip in range(2)]
    exercise_domain_tensors(worker, members, st_platform, local=local)
