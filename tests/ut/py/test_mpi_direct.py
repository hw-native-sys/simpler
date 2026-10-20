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
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import simpler.mpi_direct_runtime as runtime_mod
import simpler.mpi_direct_supervisor as supervisor_mod
from simpler import mpi_direct_protocol as protocol_mod
from simpler import remote_l3_session
from simpler.mpi_direct_protocol import MPI_DIRECT_STARTUP_TOKEN_ENV, MpiDirectTag
from simpler.mpi_direct_runtime import _pre_mpi_gate
from simpler.mpi_direct_supervisor import (
    _EXPORTED_ENV_VARS,
    _build_command,
    _family_launcher_args,
    _gate_send,
    _host_slots,
    _mpi_vendor_family,
    _startup_gate,
)
from simpler.mpi_direct_topology import MpiDirectTopology, load_runtime_manifest_data
from simpler.worker import Worker


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


_RANK_ENV_NAMES = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID")


@pytest.mark.parametrize("rank_name", _RANK_ENV_NAMES[:-1])
@pytest.mark.parametrize("slurm_rank", ["0", "not-a-rank"])
def test_launcher_rank_ignores_inherited_slurm_when_mpi_rank_exists(monkeypatch, rank_name, slurm_rank):
    for name in _RANK_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rank_name, "2")
    monkeypatch.setenv("SLURM_PROCID", slurm_rank)
    assert runtime_mod._launcher_rank() == 2


def test_launcher_rank_fallback_and_native_disagreement(monkeypatch):
    for name in _RANK_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="MPI launcher did not provide") as missing:
        runtime_mod._launcher_rank()
    assert all(name in str(missing.value) for name in _RANK_ENV_NAMES)
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert runtime_mod._launcher_rank() == 3
    monkeypatch.setenv("PMI_RANK", "2")
    monkeypatch.setenv("PMIX_RANK", "1")
    with pytest.raises(RuntimeError, match="rank variables disagree"):
        runtime_mod._launcher_rank()


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


def test_runtime_manifest_round_trip_binds_one_session():
    topology = MpiDirectTopology.from_dict(_topology_dict())
    manifest = json.loads(json.dumps(topology.runtime_manifest(91)))
    loaded, session_id = load_runtime_manifest_data(manifest)
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


def test_topology_automatic_ranks_skip_explicit_ranks_on_later_executors():
    data = _topology_dict()
    data["executor_ranks"][0]["device_ids"] = [0, 1, 2]
    data["executor_ranks"][1].update(device_ids=[3, 4], global_device_ranks=[0, 2])
    topology = MpiDirectTopology.from_dict(data)
    assert topology.executors[0].global_device_ranks == (1, 3, 4)
    assert topology.executors[1].global_device_ranks == (0, 2)


@pytest.mark.parametrize("count", [runtime_mod.FRAME_HEADER_BYTES - 1, runtime_mod.MAX_FRAME_BYTES + 1])
def test_mpi_receive_rejects_invalid_size_before_receiving(count):
    status = SimpleNamespace(Get_count=lambda _datatype: count)
    mpi = SimpleNamespace(Status=lambda: status, ANY_TAG=-1, BYTE=object())
    message = Mock()
    world = SimpleNamespace(improbe=lambda **_kwargs: message)
    with pytest.raises(RuntimeError, match="frame length.*outside the SLR3 bounds"):
        runtime_mod._receive_frame(mpi, world, 0)
    message.Recv.assert_not_called()


def test_supervisor_openmpi_command_uses_one_static_world_without_shell():
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))
    command = _build_command(
        topology,
        mpirun_path="/opt/mpi/bin/mpirun",
        manifest_json="encoded-manifest",
        startup_port=4567,
        controller="case.py",
        launcher_family="openmpi",
    )
    assert command[:4] == ["/opt/mpi/bin/mpirun", "--host", "host-a:2,host-b:1", "--map-by"]
    assert command[command.index("-np") : command.index("-np") + 2] == ["-np", "3"]
    assert command[-8:] == [
        "--manifest-json",
        "encoded-manifest",
        "--controller",
        "case.py",
        "--startup-host",
        "host-a",
        "--startup-port",
        "4567",
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
        "-genvlist",
        MPI_DIRECT_STARTUP_TOKEN_ENV,
    ]
    command = _build_command(
        topology,
        mpirun_path="/opt/mpi/bin/mpiexec",
        manifest_json="encoded-manifest",
        startup_port=4567,
        controller="case.py",
        launcher_family="mpich",
        hostfile_path="/tmp/hosts",
    )
    assert command[:3] == ["/opt/mpi/bin/mpiexec", "-f", "/tmp/hosts"]
    assert command[command.index("-np") : command.index("-np") + 2] == ["-np", "3"]
    assert "-x" not in command
    assert "--map-by" not in command


@pytest.fixture
def supervisor_setup(tmp_path, monkeypatch):
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(_topology_dict(second_host="host-b")), encoding="utf-8")
    listener = Mock(getsockname=Mock(return_value=("host-a", 4567)))
    process = Mock(wait=Mock(return_value=0), poll=Mock(return_value=None))
    monkeypatch.setattr(supervisor_mod, "_detect_mpi4py_family", lambda _python: ("mpich", "MPICH"))
    monkeypatch.setattr(supervisor_mod.socket, "socket", lambda *_args: listener)
    monkeypatch.setattr(supervisor_mod.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(supervisor_mod, "_startup_gate", Mock())
    return str(topology_path), process


def test_run_supervisor_keeps_mpich_hostfile_until_job_finishes(supervisor_setup, monkeypatch):
    topology_path, process = supervisor_setup
    monkeypatch.setenv(MPI_DIRECT_STARTUP_TOKEN_ENV, "parent-secret")
    observed = []

    def check_hostfile(*_args):
        command = supervisor_mod.subprocess.Popen.call_args.args[0]
        hostfile = Path(command[command.index("-f") + 1])
        assert hostfile.is_file()
        assert hostfile.read_text() == "host-a:2\nhost-b:1\n"
        observed.append(hostfile)
        return 0

    supervisor_mod._startup_gate.side_effect = check_hostfile
    process.wait.side_effect = check_hostfile
    assert supervisor_mod.run_supervisor(topology_path, "controller.py", launcher_family="mpich") == 0
    launched = supervisor_mod.subprocess.Popen.call_args
    token = launched.kwargs["env"][MPI_DIRECT_STARTUP_TOKEN_ENV]
    assert token and token != "parent-secret"
    assert token not in launched.args[0]
    assert os.environ[MPI_DIRECT_STARTUP_TOKEN_ENV] == "parent-secret"
    assert len(observed) == 2
    assert all(not hostfile.exists() for hostfile in observed)


def test_supervisor_inline_manifest_command_adds_pre_mpi_gate():
    topology = MpiDirectTopology.from_dict(_topology_dict())
    command = _build_command(
        topology,
        mpirun_path="mpiexec",
        controller="case.py",
        launcher_family="mpich",
        hostfile_path="/tmp/hosts",
        manifest_json="encoded-manifest",
        python_executable="/opt/simpler-python",
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


@pytest.mark.parametrize("invalid_peer", [False, True])
def test_pre_mpi_gate_releases_all_ready_ranks(invalid_peer):
    topology = MpiDirectTopology.from_dict(_topology_dict(second_host="host-b"))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(topology.world_size)
    port = int(listener.getsockname()[1])
    errors = []
    if invalid_peer:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as peer:
            _gate_send(peer, {"token": "wrong", "rank": 0, "state": "ready"})

    def enter(rank):
        try:
            _pre_mpi_gate("127.0.0.1", port, "token", rank, 3.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=enter, args=(rank,)) for rank in range(topology.world_size)]
    for thread in threads:
        thread.start()
    _startup_gate(topology, "token", listener, SimpleNamespace(poll=lambda: None))  # type: ignore[arg-type]
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


def test_startup_gate_rejects_silent_peer_without_consuming_startup_budget(monkeypatch):
    topology = MpiDirectTopology.from_dict({**_topology_dict(), "startup_timeout_s": 20})
    now = [0.0]
    responses = []

    class Peer:
        timeout = None
        closed = False

        def __init__(self, rank):
            self.rank = rank

        def settimeout(self, timeout):
            self.timeout = timeout

        def close(self):
            self.closed = True

    silent = Peer(None)
    ready = [Peer(rank) for rank in range(topology.world_size)]
    pending = iter([silent, *ready])
    listener = SimpleNamespace(
        settimeout=lambda _timeout: None, close=lambda: None, accept=lambda: (next(pending), None)
    )

    def receive(peer, deadline):
        peer.settimeout(deadline - now[0])
        if peer is silent:
            now[0] += peer.timeout
            raise socket.timeout("peer sent no data")
        return {"token": "token", "rank": peer.rank, "state": "ready"}

    monkeypatch.setattr(supervisor_mod, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(supervisor_mod, "_gate_recv", receive)
    monkeypatch.setattr(supervisor_mod, "_gate_send", lambda _peer, message: responses.append(message))
    _startup_gate(topology, "token", listener, SimpleNamespace(poll=lambda: None))  # type: ignore[arg-type]
    assert silent.closed
    assert silent.timeout is not None
    assert silent.timeout <= 5
    assert all(peer.closed for peer in ready)
    assert responses == [{"token": "token", "state": "go_mpi"}] * topology.world_size


@pytest.mark.parametrize("termination_signal", [signal.SIGTERM, signal.SIGHUP])
def test_supervisor_restores_signal_handlers_after_job_cleanup(supervisor_setup, monkeypatch, termination_signal):
    topology_path, process = supervisor_setup
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    cleanup = []

    def wait():
        handler = signal.getsignal(termination_signal)
        assert callable(handler), "supervisor must handle termination before launching MPI"
        handler(termination_signal, None)

    def terminate(proc):
        assert proc is process
        assert signal.getsignal(termination_signal) != previous[termination_signal]
        cleanup.append(proc)

    process.wait = wait
    monkeypatch.setattr(supervisor_mod, "_terminate_job", terminate)
    with pytest.raises(SystemExit) as stopped:
        supervisor_mod.run_supervisor(topology_path, "controller.py", launcher_family="mpich")
    assert stopped.value.code == 128 + termination_signal
    assert cleanup == [process]
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.parametrize("budget,byte_delay", [(0.25, 0.12), (20.0, 1.2)])
def test_startup_gate_fragmented_read_has_one_absolute_budget(monkeypatch, budget, byte_delay):
    topology = MpiDirectTopology.from_dict({**_topology_dict(), "startup_timeout_s": budget})
    now = [0.0]

    class Peer:
        def __init__(self, message, delay):
            payload = json.dumps(message).encode()
            self.data = bytearray(len(payload).to_bytes(4, "big") + payload)
            self.delay = delay
            self.timeout = budget

        def settimeout(self, timeout):
            self.timeout = timeout

        def recv(self, size):
            if self.delay > self.timeout:
                now[0] += self.timeout
                raise socket.timeout("fragment missed deadline")
            now[0] += self.delay
            data = bytes(self.data[:1])
            del self.data[:1]
            return data

        def close(self):
            pass

        def sendall(self, _data):
            pass

    peers = iter(
        [Peer({}, byte_delay)]
        + [Peer({"token": "token", "rank": rank, "state": "ready"}, 0) for rank in range(topology.world_size)]
    )
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(protocol_mod, "time", clock, raising=False)
    monkeypatch.setattr(supervisor_mod, "time", clock)
    listener = Mock(accept=lambda: (next(peers), None))
    process = Mock(poll=Mock(return_value=None))
    if budget < 5:
        with pytest.raises(TimeoutError):
            _startup_gate(topology, "token", listener, process)
    else:
        _startup_gate(topology, "token", listener, process)
    assert now[0] <= min(budget, 5.0)


@pytest.mark.parametrize(
    "message,error",
    [
        ({"rank": -1}, "invalid.*rank"),
        ({"rank": 99}, "invalid.*rank"),
        ({"rank": "bad"}, "invalid.*rank"),
        ({"rank": None}, "invalid.*rank"),
        ({"rank": 0, "state": "duplicate"}, "duplicate.*rank"),
        ({"state": "bogus"}, "invalid startup state"),
        ({"state": "failed", "error": "startup failed"}, "failed before MPI"),
    ],
)
def test_startup_gate_rejects_authenticated_protocol_errors(monkeypatch, message, error):
    topology = MpiDirectTopology.from_dict(_topology_dict())
    invalid = {"token": "token", "rank": 0, "state": "ready", **message}
    messages = (
        [{"token": "token", "rank": 0, "state": "ready"}, invalid] if message.get("state") == "duplicate" else [invalid]
    )
    if message.get("state") == "duplicate":
        invalid["state"] = "ready"
    peers = [Mock(message=item) for item in messages]
    pending = iter(peers)
    listener = Mock(accept=lambda: (next(pending), None))
    process = Mock(poll=Mock(side_effect=[None] * len(peers) + [0]))
    monkeypatch.setattr(supervisor_mod, "_gate_recv", lambda peer, *_args: peer.message)
    with pytest.raises(RuntimeError, match=error):
        _startup_gate(topology, "token", listener, process)
    assert all(peer.close.called for peer in peers)


@pytest.mark.parametrize("children", ["sub", "chip", "worker"])
def test_controller_attach_rejects_local_children(children):
    config = {"num_sub_workers": 1} if children == "sub" else {"device_ids": [0]} if children == "chip" else {}
    worker = Worker(level=4, **config)
    if children == "worker":
        worker.add_worker(Worker(level=3, num_sub_workers=0))
    context = runtime_mod.MpiDirectControllerContext.__new__(runtime_mod.MpiDirectControllerContext)
    context.topology = MpiDirectTopology.from_dict(_topology_dict())
    context.session_id, context.cluster_id = 1, "test"
    context._hub, context._workers, context._closed = Mock(), [], False
    with pytest.raises(ValueError, match="local child workers"):
        context.attach(worker)
    assert not context._workers


def test_mpi_controller_rejects_local_worker_added_after_attach():
    worker = Worker(level=4)
    worker._configure_mpi_direct(MpiDirectTopology.from_dict(_topology_dict()), 1, Mock())
    with pytest.raises(ValueError, match="local child workers"):
        worker.add_worker(Worker(level=3))


def test_controller_rechecks_local_children_before_starting_hierarchy(monkeypatch):
    worker = Worker(level=4)
    worker._configure_mpi_direct(MpiDirectTopology.from_dict(_topology_dict()), 1, Mock())
    worker._config["num_sub_workers"] = 1
    fork = Mock(side_effect=AssertionError("unsafe fork reached"))
    monkeypatch.setattr(os, "fork", fork)
    worker._startup_deadline = time.monotonic() + 5
    with pytest.raises(ValueError, match="local child workers"):
        worker._start_hierarchical()
    fork.assert_not_called()


def test_progress_failure_aborts_from_progress_owner_thread(monkeypatch):
    owner = threading.get_ident()
    world = Mock()
    world.Abort.side_effect = (
        lambda _code: pytest.fail("abort must run in the progress owner") if threading.get_ident() != owner else None
    )
    hub = Mock(poll_outbound=Mock(return_value=None), check_health=Mock(side_effect=RuntimeError("heartbeat lost")))
    monkeypatch.setattr(runtime_mod, "_receive_frame", lambda *_args: None)
    monkeypatch.setattr(runtime_mod.os, "_exit", Mock(side_effect=SystemExit(1)))
    progress = runtime_mod._ControllerProgress(SimpleNamespace(ANY_SOURCE=-1), world, hub)
    with pytest.raises(SystemExit):
        progress._run()
    hub.fail.assert_called_once()
    world.Abort.assert_called_once_with(1)


def test_progress_drain_timeout_exits_without_concurrent_mpi_abort(monkeypatch):
    world = Mock()
    progress = runtime_mod._ControllerProgress(None, world, Mock())
    progress._thread = Mock(is_alive=Mock(return_value=True))
    monkeypatch.setattr(runtime_mod.os, "_exit", Mock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        progress.stop(0.1)
    world.Abort.assert_not_called()


def test_executor_cleanup_closes_worker_when_channel_close_fails(monkeypatch):
    topology = MpiDirectTopology.from_dict(_topology_dict())
    channel = Mock(close=Mock(side_effect=RuntimeError("heartbeat failed")))
    worker = Mock()
    monkeypatch.setattr(runtime_mod, "_ExecutorFrameSocket", lambda *_args: channel)
    monkeypatch.setattr(remote_l3_session, "_run_command_loop", lambda *_args: None)

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        runtime_mod._run_executor(None, None, topology, 1, topology.executors[0], worker)

    channel.close.assert_called_once()
    worker.close.assert_called_once()


def test_executor_receive_does_not_sleep_on_dispatch_path(monkeypatch):
    topology = MpiDirectTopology.from_dict(_topology_dict())

    mpi = SimpleNamespace(ANY_TAG=-1, BYTE=object(), Status=SimpleNamespace)

    class EmptyWorld:
        def improbe(self, **_kwargs):
            channel._health_error = RuntimeError("stop probe")

    channel = runtime_mod._ExecutorFrameSocket(mpi, EmptyWorld(), topology.executors[0], 1, 1.0)
    sleep = Mock()
    monkeypatch.setattr(runtime_mod.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="executor heartbeat failed"):
        channel._receive_message()

    sleep.assert_not_called()


def test_controller_progress_does_not_sleep_on_dispatch_path(monkeypatch):
    mpi = SimpleNamespace(ANY_SOURCE=-1, ANY_TAG=-1, BYTE=object(), Status=SimpleNamespace)

    class EmptyWorld:
        probes = 0

        def improbe(self, **_kwargs):
            self.probes += 1
            if self.probes == 2:
                progress._stop.set()

    hub = SimpleNamespace(
        pending_frame_bytes=0, check_health=lambda _timeout: None, poll_outbound=lambda _timeout: None
    )
    progress = runtime_mod._ControllerProgress(mpi, EmptyWorld(), hub)
    sleep = Mock()
    monkeypatch.setattr(runtime_mod.time, "sleep", sleep)
    progress._run()

    sleep.assert_not_called()


def test_mpi_vendor_family_matches_supported_launchers():
    assert _mpi_vendor_family("Open MPI") == "openmpi"
    assert _mpi_vendor_family("MPICH") == "mpich"
    assert _mpi_vendor_family("MVAPICH2") == "mpich"
    with pytest.raises(RuntimeError, match="unsupported mpi4py MPI vendor"):
        _mpi_vendor_family("unknown")


def test_direct_mpi_tags_are_small_fixed_lanes():
    assert tuple(int(tag) for tag in MpiDirectTag) == (1, 2, 3, 4)


def test_direct_mpi_wire_constants_match_native(monkeypatch):
    native = pytest.importorskip("_task_interface")
    from simpler.mpi_direct_protocol import _assert_native_wire_constants  # noqa: PLC0415

    _assert_native_wire_constants()
    monkeypatch.setattr(native, "_REMOTE_FRAME_HEADER_BYTES", 1)
    with pytest.raises(RuntimeError, match="frame limits mismatch"):
        _assert_native_wire_constants()


def test_controller_preserves_original_error_when_teardown_also_fails(monkeypatch):
    class Context:
        def close(self):
            raise RuntimeError("secondary close failure")

    def controller(_context):
        raise ValueError("original controller failure")

    monkeypatch.setattr(runtime_mod, "MpiDirectControllerContext", lambda *_args: Context())
    monkeypatch.setattr(runtime_mod, "_load_controller", lambda _target: controller)
    with pytest.raises(ValueError, match="original controller failure"):
        runtime_mod._run_controller(None, None, MpiDirectTopology.from_dict(_topology_dict()), 1, "controller")


def test_topology_rejects_heartbeat_longer_than_session_timeout():
    data = _topology_dict()
    data["heartbeat_interval_s"] = data["session_timeout_s"]
    with pytest.raises(ValueError, match="heartbeat_interval_s must be shorter"):
        MpiDirectTopology.from_dict(data)


def test_fork_fd_snapshot_does_not_close_a_reused_descriptor(tmp_path):
    from simpler.worker import _close_fork_child_fds  # noqa: PLC0415

    first = os.open(tmp_path / "launcher", os.O_CREAT | os.O_RDWR, 0o600)
    replacement = os.open(tmp_path / "worker", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        snapshot = dict(runtime_mod._open_fds())
        os.dup2(replacement, first)
        _close_fork_child_fds(((first, snapshot[first]),))
        assert os.fstat(first).st_ino == os.fstat(replacement).st_ino
        _close_fork_child_fds(((replacement, snapshot[replacement]),))
        with pytest.raises(OSError):
            os.fstat(replacement)
    finally:
        for fd in (first, replacement):
            try:
                os.close(fd)
            except OSError:
                pass


def mpi_direct_cpu_orch(_orch, args, _config):
    assert args.scalar(0) == 42


def mpi_direct_cpu_controller(context):
    import ctypes  # noqa: PLC0415

    from simpler.task_interface import TaskArgs  # noqa: PLC0415
    from simpler.worker import RemoteCallable  # noqa: PLC0415

    worker = context.create_worker(num_sub_workers=0)
    handle = worker.register(
        RemoteCallable("test_mpi_direct:mpi_direct_cpu_orch"),
        workers=[spec.worker_id for spec in context.topology.executors],
    )
    worker.init()
    for spec in context.topology.executors:
        buffer = worker.remote_malloc(worker=spec.worker_id, nbytes=8)
        try:
            source = ctypes.c_uint64(42)
            result = ctypes.c_uint64()
            worker.remote_copy_to(buffer, source, 8)
            worker.remote_copy_from(buffer, result, 8)
            assert result.value == 42
        finally:
            worker.remote_free(buffer)

    def parent(orch, _args, config):
        for spec in context.topology.executors:
            args = TaskArgs()
            args.add_scalar(42)
            orch.submit_next_level(handle, args, config, worker=spec.worker_id)

    worker.run(parent)


def _interrupt_idle_mpi_job(proc, mode, controller_path, supervisor_ready):
    deadline = time.monotonic() + 30
    while not supervisor_ready.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert supervisor_ready.exists(), "controller did not finish its MPI roundtrip before termination"
    if mode == "idle-executor-stop":
        processes = subprocess.check_output(["ps", "-ww", "-eo", "pid=,ppid=,args="], text=True)
        ranks = []
        for line in processes.splitlines():
            pid, ppid, command = line.strip().split(None, 2)
            if str(controller_path) in command and command.split(None, 3)[1:3] == ["-m", "simpler.mpi_direct_runtime"]:
                ranks.append((int(pid), int(ppid)))
        all_pids = {pid for pid, _ppid in ranks}
        executors = [pid for pid, ppid in ranks if ppid not in all_pids and pid != int(supervisor_ready.read_text())]
        assert len(executors) == 2, processes
        os.kill(executors[0], signal.SIGSTOP)
    else:
        signum = signal.SIGTERM if mode == "supervisor-term" else signal.SIGHUP
        proc.send_signal(signum)


@pytest.mark.parametrize(
    "mode",
    [
        "roundtrip",
        "rank0-crash",
        "unused-executors",
        "slurm-allocation",
        "supervisor-term",
        "supervisor-hup",
        "idle-executor-stop",
    ],
)
def test_real_mpi_cpu_world_and_launcher_cleanup(tmp_path, mode):
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("mpirun or mpiexec is not on PATH")
    pytest.importorskip("mpi4py")
    topology = _topology_dict()
    topology["controller_host"] = "127.0.0.1"
    for spec in topology["executor_ranks"]:
        spec.update(host="127.0.0.1", device_ids=[], num_sub_workers=1)
    if mode == "idle-executor-stop":
        topology.update(session_timeout_s=1.5, heartbeat_interval_s=0.1)
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(topology), encoding="utf-8")
    controller_path = tmp_path / "controller.py"
    crash_marker = tmp_path / "crash-reached"
    supervisor_ready = tmp_path / "supervisor-ready"
    controller_path.write_text(
        "import os, signal\n"
        "from test_mpi_direct import mpi_direct_cpu_controller\n"
        "def main(context):\n"
        + ("    pass\n" if mode == "unused-executors" else "    mpi_direct_cpu_controller(context)\n")
        + (
            f"    open({str(crash_marker)!r}, 'w').close()\n    os.kill(os.getpid(), signal.SIGKILL)\n"
            if mode == "rank0-crash"
            else ""
        )
        + (
            f"    open({str(supervisor_ready)!r}, 'w').write(str(os.getpid()))\n"
            "    import threading\n    threading.Event().wait()\n"
            if mode.startswith("supervisor-") or mode == "idle-executor-stop"
            else ""
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parent), str(Path.cwd()), env.get("PYTHONPATH", "")))
    if mode == "slurm-allocation":
        env["SLURM_PROCID"] = "0"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "simpler.mpi_direct_supervisor",
            "--topology",
            str(topology_path),
            "--controller",
            str(controller_path),
            "--mpirun",
            launcher,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        if mode.startswith("supervisor-") or mode == "idle-executor-stop":
            _interrupt_idle_mpi_job(proc, mode, controller_path, supervisor_ready)
        output, _ = proc.communicate(timeout=8 if mode == "idle-executor-stop" else 45)
        if mode == "idle-executor-stop":
            assert "timed out waiting for HEALTH frame" in output, output
        assert (proc.returncode == 0) if mode in ("roundtrip", "slurm-allocation") else (proc.returncode != 0), output
        if mode.startswith("supervisor-"):
            assert proc.returncode == 128 + (signal.SIGTERM if mode == "supervisor-term" else signal.SIGHUP), output
        if mode == "rank0-crash":
            assert crash_marker.exists(), output
        if mode == "unused-executors":
            assert "lifecycle shutdown" in output, output
        processes = subprocess.check_output(["ps", "-ww", "-eo", "pid,args"], text=True)
        assert str(controller_path) not in processes, processes
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        processes = subprocess.check_output(["ps", "-ww", "-eo", "pid=,args="], text=True)
        for line in processes.splitlines():
            if str(controller_path) in line:
                try:
                    os.kill(int(line.split(None, 1)[0]), signal.SIGKILL)
                except ProcessLookupError:
                    pass
