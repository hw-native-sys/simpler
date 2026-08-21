# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
import simpler.mpi_direct_runtime as runtime_mod
import simpler.mpi_direct_supervisor as supervisor_mod
from simpler import remote_l3_session
from simpler.mpi_direct_protocol import MPI_DIRECT_STARTUP_TOKEN_ENV, MpiDirectTag
from simpler.mpi_direct_runtime import _pre_mpi_gate
from simpler.mpi_direct_supervisor import (
    _EXPORTED_ENV_VARS,
    _build_command,
    _family_launcher_args,
    _gate_recv,
    _gate_send,
    _host_slots,
    _mpi_vendor_family,
    _startup_gate,
)
from simpler.mpi_direct_topology import MpiDirectTopology, load_runtime_manifest, load_runtime_manifest_data


def _topology_dict(*, second_host: str = "host-a") -> dict:
    return {
        "controller_rank": 0,
        "controller_host": "host-a",
        "startup_timeout_s": 10,
        "session_timeout_s": 5,
        "heartbeat_interval_s": 1,
        "launcher_args": [],
        "executor_ranks": [
            {
                "rank": 1,
                "worker_id": 0,
                "host": "host-a",
                "platform": "a2a3sim",
                "device_ids": [0],
                "comm_profile": "sim",
            },
            {
                "rank": 2,
                "worker_id": 1,
                "host": second_host,
                "platform": "a2a3sim",
                "device_ids": [1],
                "comm_profile": "sim",
            },
        ],
    }


def test_topology_requires_dense_rank_and_worker_mapping():
    data = _topology_dict()
    data["executor_ranks"][1]["worker_id"] = 4
    with pytest.raises(ValueError, match="worker_ids must be dense"):
        MpiDirectTopology.from_dict(data)

    data = _topology_dict()
    data["executor_ranks"] = list(reversed(data["executor_ranks"]))
    with pytest.raises(ValueError, match="ranks must be dense and ordered"):
        MpiDirectTopology.from_dict(data)


@pytest.mark.parametrize("controller_host", ["localhost", "127.0.0.1", "::1"])
def test_topology_rejects_loopback_controller_on_multiple_hosts(controller_host):
    data = _topology_dict(second_host="host-b")
    data["controller_host"] = controller_host
    with pytest.raises(ValueError, match="controller_host=.*loopback.*topology spans multiple hosts"):
        MpiDirectTopology.from_dict(data)


def test_topology_rejects_default_loopback_controller_on_multiple_hosts():
    data = _topology_dict(second_host="host-b")
    data.pop("controller_host")
    with pytest.raises(ValueError, match="controller_host=.*loopback.*topology spans multiple hosts"):
        MpiDirectTopology.from_dict(data)


def test_topology_allows_loopback_controller_on_one_host():
    data = _topology_dict(second_host="localhost")
    data["controller_host"] = "localhost"
    data["executor_ranks"][0]["host"] = "localhost"
    MpiDirectTopology.from_dict(data)


def test_topology_rejects_noncontiguous_host_order():
    data = _topology_dict(second_host="host-a")
    data["executor_ranks"][0]["host"] = "host-b"
    with pytest.raises(ValueError, match="hosts must be contiguous"):
        MpiDirectTopology.from_dict(data)


def test_runtime_manifest_round_trip_binds_one_session(tmp_path: Path):
    topology = MpiDirectTopology.from_dict(_topology_dict())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(topology.runtime_manifest(91)), encoding="utf-8")
    loaded, session_id = load_runtime_manifest(str(path))
    assert loaded == topology
    assert session_id == 91
    assert loaded.executor_for_rank(2).worker_id == 1
    assert loaded.executor_for_rank(1).global_device_ranks == (0,)
    assert loaded.executor_for_rank(2).global_device_ranks == (1,)


def test_runtime_manifest_can_be_consumed_without_a_shared_file():
    topology = MpiDirectTopology.from_dict(_topology_dict())
    loaded, session_id = load_runtime_manifest_data(topology.runtime_manifest(92))
    assert loaded == topology
    assert session_id == 92


def test_topology_rejects_duplicate_explicit_global_device_ranks():
    data = _topology_dict()
    data["executor_ranks"][0]["global_device_ranks"] = [7]
    data["executor_ranks"][1]["global_device_ranks"] = [7]
    with pytest.raises(ValueError, match="global_device_ranks must be unique"):
        MpiDirectTopology.from_dict(data)


def test_supervisor_openmpi_command_uses_one_static_world_without_shell():
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))
    command = _build_command(
        topology,
        mpirun_path="/opt/mpi/bin/mpirun",
        topology_path="/work/simpler/topology.json",
        session_id=91,
        controller="case.py",
        launcher_family="openmpi",
    )
    assert command[:4] == ["/opt/mpi/bin/mpirun", "--host", "host-a:2,host-b:1", "--map-by"]
    assert command[-11:-9] == ["-np", "3"]
    assert command[-6:] == [
        "--topology",
        "/work/simpler/topology.json",
        "--session-id",
        "91",
        "--controller",
        "case.py",
    ]


def test_supervisor_mpich_command_uses_local_hostfile_without_openmpi_flags(monkeypatch):
    for name in _EXPORTED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/work/simpler")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/ascend")
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))
    assert _host_slots(topology) == (("host-a", 2), ("host-b", 1))
    assert _family_launcher_args(topology, "mpich", "/tmp/hosts") == [
        "-f",
        "/tmp/hosts",
        "-genv",
        "PYTHONPATH",
        "/work/simpler",
        "-genv",
        "ASCEND_HOME_PATH",
        "/opt/ascend",
    ]
    command = _build_command(
        topology,
        mpirun_path="/opt/mpi/bin/mpiexec",
        topology_path="/work/simpler/topology.json",
        session_id=91,
        controller="case.py",
        launcher_family="mpich",
        hostfile_path="/tmp/hosts",
    )
    assert command[:3] == ["/opt/mpi/bin/mpiexec", "-f", "/tmp/hosts"]
    assert command[command.index("-np") : command.index("-np") + 2] == ["-np", "3"]
    assert "-x" not in command
    assert "--map-by" not in command


def test_run_supervisor_keeps_mpich_hostfile_until_job_finishes(tmp_path, monkeypatch):
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(_topology_dict(second_host="host-b")), encoding="utf-8")
    observed: dict[str, str] = {}

    class FakeListener:
        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            pass

        def listen(self, _backlog):
            pass

        def getsockname(self):
            return ("host-a", 4567)

        def close(self):
            pass

    class FakeProcess:
        returncode = None
        pid = 1

        def poll(self):
            return None

        def wait(self, *_args, **_kwargs):
            assert Path(observed["hostfile"]).is_file()
            return 0

    def popen(command, **_kwargs):
        hostfile = command[command.index("-f") + 1]
        observed["hostfile"] = hostfile
        assert Path(hostfile).is_file()
        return FakeProcess()

    def startup_gate(_topology, _token, _listener, _proc):
        assert Path(observed["hostfile"]).is_file()

    monkeypatch.setattr(supervisor_mod, "_detect_mpi4py_family", lambda _python: ("mpich", "MPICH"))
    monkeypatch.setattr(supervisor_mod.socket, "socket", lambda *_args, **_kwargs: FakeListener())
    monkeypatch.setattr(supervisor_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor_mod, "_startup_gate", startup_gate)

    assert supervisor_mod.run_supervisor(str(topology_path), "controller.py", launcher_family="mpich") == 0
    assert not Path(observed["hostfile"]).exists()


def test_supervisor_inline_manifest_command_adds_pre_mpi_gate():
    topology = MpiDirectTopology.from_dict(_topology_dict())
    command = _build_command(
        topology,
        mpirun_path="mpiexec",
        topology_path=None,
        session_id=91,
        controller="case.py",
        launcher_family="mpich",
        hostfile_path="/tmp/hosts",
        manifest_json="encoded-manifest",
        python_executable="/opt/simpler-python",
        startup_host="host-a",
        startup_port=4567,
    )
    assert "/opt/simpler-python" in command
    assert "--manifest-json" in command
    assert "--topology" not in command
    assert command[-4:] == ["--startup-host", "host-a", "--startup-port", "4567"]
    assert "token" not in command


def test_startup_token_is_exported_without_appearing_in_launcher_args(monkeypatch):
    monkeypatch.setenv(MPI_DIRECT_STARTUP_TOKEN_ENV, "secret-token")
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))

    mpich_args = _family_launcher_args(topology, "mpich", "/tmp/hosts")
    assert ["-genvlist", MPI_DIRECT_STARTUP_TOKEN_ENV] == mpich_args[-2:]
    assert "secret-token" not in mpich_args

    openmpi_args = _family_launcher_args(topology, "openmpi", None)
    assert ["-x", MPI_DIRECT_STARTUP_TOKEN_ENV] == openmpi_args[-2:]
    assert "secret-token" not in openmpi_args


def test_pre_mpi_gate_releases_all_ready_ranks():
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(topology.world_size)
    port = int(listener.getsockname()[1])
    errors = []

    def enter(rank):
        try:
            _pre_mpi_gate("127.0.0.1", port, "token", rank, 3.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=enter, args=(rank,)) for rank in range(topology.world_size)]
    for thread in threads:
        thread.start()

    class RunningProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    _startup_gate(topology, "token", listener, RunningProcess())  # type: ignore[arg-type]
    for thread in threads:
        thread.join(3.0)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)


def test_pre_mpi_gate_sleeps_between_connection_retries(monkeypatch):
    now = [0.0]
    sleeps = []

    def monotonic():
        return now[0]

    def sleep(delay):
        sleeps.append(delay)
        now[0] = 1.0

    def refused_connection(*_args, **_kwargs):
        raise ConnectionRefusedError("gate is not listening")

    monkeypatch.setattr(runtime_mod.time, "monotonic", monotonic)
    monkeypatch.setattr(runtime_mod.time, "sleep", sleep)
    monkeypatch.setattr(runtime_mod.socket, "create_connection", refused_connection)

    with pytest.raises(TimeoutError, match="startup gate connection timed out"):
        _pre_mpi_gate("controller", 4567, "token", 0, 0.1)

    assert sleeps == [runtime_mod._MPI_GATE_RETRY_INTERVAL_S]


def test_startup_gate_ignores_invalid_peer_and_releases_ready_ranks():
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(topology.world_size)
    port = int(listener.getsockname()[1])
    responses = []

    bad_peer = socket.create_connection(("127.0.0.1", port), timeout=1.0)
    _gate_send(bad_peer, {"token": "wrong", "rank": 0, "state": "ready"})
    bad_peer.close()

    def enter(rank):
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as peer:
            _gate_send(peer, {"token": "token", "rank": rank, "state": "ready"})
            responses.append(_gate_recv(peer))

    threads = [threading.Thread(target=enter, args=(rank,)) for rank in range(topology.world_size)]
    for thread in threads:
        thread.start()

    class RunningProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    _startup_gate(topology, "token", listener, RunningProcess())  # type: ignore[arg-type]
    for thread in threads:
        thread.join(3.0)

    assert responses == [{"token": "token", "state": "go_mpi"}] * topology.world_size
    assert all(not thread.is_alive() for thread in threads)


def test_executor_cleanup_closes_worker_when_channel_close_fails(monkeypatch):
    topology = MpiDirectTopology.from_dict(_topology_dict())
    channel_closed = False

    class FailingChannel:
        def close(self):
            nonlocal channel_closed
            channel_closed = True
            raise RuntimeError("heartbeat failed")

    class FakeWorker:
        closed = False

        def close(self):
            self.closed = True

    channel = FailingChannel()
    worker = FakeWorker()
    monkeypatch.setattr(runtime_mod, "_ExecutorFrameSocket", lambda *_args: channel)
    monkeypatch.setattr(remote_l3_session, "_run_command_loop", lambda *_args: None)

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        runtime_mod._run_executor(None, None, topology, 1, topology.executors[0], worker)

    assert channel_closed
    assert worker.closed


def test_executor_receive_sleeps_between_empty_mpi_probes(monkeypatch):
    topology = MpiDirectTopology.from_dict(_topology_dict())

    class FakeStatus:
        pass

    class FakeMPI:
        ANY_TAG = -1
        BYTE = object()

        @staticmethod
        def Status():
            return FakeStatus()

    class EmptyWorld:
        def improbe(self, **_kwargs):
            return None

    channel = runtime_mod._ExecutorFrameSocket(FakeMPI, EmptyWorld(), topology.executors[0], 1, 1.0)
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        channel._health_error = RuntimeError("stop probe")

    monkeypatch.setattr(runtime_mod.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="executor heartbeat failed"):
        channel._receive_message()

    assert sleeps == [runtime_mod._MPI_POLL_INTERVAL_S]


def test_controller_progress_sleeps_when_idle(monkeypatch):
    class FakeStatus:
        pass

    class FakeMPI:
        ANY_SOURCE = -1
        ANY_TAG = -1
        BYTE = object()

        @staticmethod
        def Status():
            return FakeStatus()

    class EmptyWorld:
        def improbe(self, **_kwargs):
            return None

    class EmptyHub:
        pending_frame_bytes = 0

        def poll_outbound(self, _timeout):
            return None

    progress = runtime_mod._ControllerProgress(FakeMPI, EmptyWorld(), EmptyHub())
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        progress._stop.set()

    monkeypatch.setattr(runtime_mod.time, "sleep", sleep)
    progress._run()

    assert sleeps == [runtime_mod._MPI_POLL_INTERVAL_S]


def test_mpi_vendor_family_matches_supported_launchers():
    assert _mpi_vendor_family("Open MPI") == "openmpi"
    assert _mpi_vendor_family("MPICH") == "mpich"
    assert _mpi_vendor_family("MVAPICH2") == "mpich"
    with pytest.raises(RuntimeError, match="unsupported mpi4py MPI vendor"):
        _mpi_vendor_family("unknown")


def test_direct_mpi_tags_are_small_fixed_lanes():
    assert tuple(int(tag) for tag in MpiDirectTag) == (1, 2, 3, 4)
