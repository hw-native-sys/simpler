# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for MPI artifact transport and ordered node preparation."""

from __future__ import annotations

import builtins
import dataclasses
import gzip
import hashlib
import importlib
import io
import json
import multiprocessing
import sys
import tarfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import simpler.mpi_group_mailbox as mailbox_mod
import simpler.worker as worker_mod
from simpler import mpi_l3_session
from simpler.callable_identity import build_python_import_descriptor, compute_callable_hashid, hashid_to_digest
from simpler.remote_l3_protocol import CallableKind, FrameHeader, FrameType, decode_frame, encode_frame
from simpler.startup_artifacts import StartupArtifactSpec
from simpler.worker import MpiL3GroupSpec, Worker


def _preparation_api():
    assert importlib.util.find_spec("simpler.startup_preparation") is not None, "node startup preparation is missing"
    return importlib.import_module("simpler.startup_preparation")


def _spec(**kwargs):
    return MpiL3GroupSpec(
        hosts=("localhost", "localhost"),
        platform="a2a3sim",
        device_ids_by_rank=((0,), (1,)),
        **kwargs,
    )


def _archive(tmp_path, prefix="mpi_startup_contract"):
    stream = io.BytesIO()
    source = b"BOUND = None\ndef dispatch():\n    return BOUND\n"
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        entry = tarfile.TarInfo("entry.py")
        entry.size = len(source)
        archive.addfile(entry, io.BytesIO(source))
    payload = gzip.compress(stream.getvalue(), mtime=0)
    path = tmp_path / "parent-only.tar.gz"
    path.write_bytes(payload)
    return StartupArtifactSpec(path, hashlib.sha256(payload).hexdigest(), prefix), payload


def _assert_fork_inherits_binding(module_name, handle):
    if sys.platform == "win32":
        assert importlib.import_module(module_name).dispatch() is handle
        return
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=lambda: writer.send(importlib.import_module(module_name).dispatch() is handle))
    child.start()
    writer.close()
    try:
        assert reader.poll(2.0), "forked execution did not observe the module binding"
        assert reader.recv() is True
        child.join(timeout=2.0)
        assert child.exitcode == 0
    finally:
        reader.close()
        if child.is_alive():
            child.terminate()
        child.join(timeout=2.0)


class _Broadcast:
    def __init__(self, rank, envelope=None):
        self.rank = rank
        self.envelope = envelope

    def Get_rank(self):
        return self.rank

    def bcast(self, value, root=0):
        assert root == 0
        if self.rank == 0:
            self.envelope = value
        else:
            assert value is None
        return self.envelope


class _SessionComm:
    def __init__(self, failure_gate=None, on_gate=None):
        self.failure_gate = failure_gate
        self.on_gate = on_gate
        self.statuses = []

    def Get_rank(self):
        return 0

    def allgather(self, value):
        self.statuses.append(value)
        if self.on_gate is not None:
            self.on_gate(len(self.statuses))
        peer = (1, True, "")
        if len(self.statuses) == self.failure_gate:
            peer = (1, False, "peer injected failure")
        return [value, peer]

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]


class _Mailbox:
    address = 0x12340000

    def __init__(self, events):
        self.events = events
        self.ready = False
        self.closed = False
        self.errors = []
        self.group_state = mailbox_mod.MailboxGroupState.READY
        self.request_state = mailbox_mod.MailboxRequestState.SHUTDOWN_READY

    def manifest(self):
        return {"name": "test-mailbox", "mailbox_bytes": mailbox_mod.MAILBOX_SIZE, "world_size": 2}

    def terminal_reason(self):
        return "; ".join(self.errors)

    def mark_terminal(self, message):
        self.errors.append(str(message))
        self.group_state = mailbox_mod.MailboxGroupState.TERMINAL

    def publish_ready(self):
        self.ready = True
        self.events.append("ready")

    def accept_request(self, *, last_sequence_id):
        return types.SimpleNamespace(
            opcode=mailbox_mod.MailboxOpcode.SHUTDOWN,
            sequence_id=1,
            target=mailbox_mod.MailboxTarget.GROUP,
            payloads=(encode_frame(FrameHeader(FrameType.SHUTDOWN, 11, 3, 1), b""),),
        )

    def complete_shutdown(self, **kwargs):
        pass

    def publish_closed(self):
        self.events.append("drain")

    def close(self, **kwargs):
        self.closed = True
        self.events.append("mailbox_close")


@pytest.fixture
def session_harness(monkeypatch, tmp_path):
    events = []
    mailbox = _Mailbox(events)
    state = types.SimpleNamespace(
        events=events,
        mailbox=mailbox,
        worker=None,
        init_error=None,
        close_error=None,
        init_callback=None,
        processor_registry=None,
    )

    class InnerWorker:
        def __init__(self, **kwargs):
            self.initialized = False
            state.worker = self

        def init(self, *, _startup_deadline):
            assert threading.current_thread() is threading.main_thread()
            self.startup_deadline = _startup_deadline
            events.append("init")
            if state.init_callback is not None:
                state.init_callback()
            if state.init_error is not None:
                raise state.init_error
            self.initialized = True

        def close(self):
            events.append("worker_close")
            if state.close_error is not None:
                raise state.close_error

    def processor(connection, manifest, inner_worker, inner_handles, registry, *args, **kwargs):
        state.processor_registry = registry
        state.task_completed = kwargs.get("task_completed", args[1] if len(args) > 1 else None)
        connection.sendall(encode_frame(FrameHeader(FrameType.HELLO, 11, 3, 0), b""))
        frame = decode_frame(connection.recv(65536))
        assert frame.header.frame_type is FrameType.SHUTDOWN
        events.append("processor_stop")

    monkeypatch.setattr(mpi_l3_session, "Worker", InnerWorker)
    monkeypatch.setattr(mpi_l3_session, "open_rank_mailbox", lambda *args, **kwargs: mailbox)
    monkeypatch.setattr(mpi_l3_session, "_run_command_loop", processor)
    monkeypatch.setattr(mpi_l3_session, "MpiGlobalDomainExchange", lambda *args, **kwargs: None)
    monkeypatch.setenv("SIMPLER_STARTUP_ARTIFACT_CACHE", str(tmp_path / "node-cache"))
    state.manifest = {
        "session_id": 11,
        "worker_id": 3,
        "platform": "a2a3sim",
        "device_ids": [0],
        "startup_remaining_s": 5.0,
        "session_timeout_s": 5.0,
    }
    state.group = {"worker_ids": [3, 7], "mailbox": mailbox.manifest(), "startup_artifacts": []}

    def run(comm=None):
        state.comm = comm if comm is not None else _SessionComm()
        return mpi_l3_session._run_group_session(
            dispatch_comm=state.comm,
            domain_comm=None,
            group_manifest=state.group,
            manifest=state.manifest,
        )

    state.run = run
    return state


@pytest.fixture
def hook_module(monkeypatch):
    module = types.ModuleType("mpi_startup_test_hooks")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_group_spec_defaults_normalize_hook_per_rank():
    spec = _spec()
    assert spec.startup_artifacts == ()
    assert spec.startup_hooks_by_rank == (None, None)


def test_group_spec_snapshots_artifact_and_hook_sequences(tmp_path):
    api = _preparation_api()
    artifact, _payload = _archive(tmp_path)
    hook = api.NodeStartupHook("user_module:prepare", {"rank_name": "left"})
    artifacts, hooks = [artifact], [hook, None]
    spec = _spec(startup_artifacts=artifacts, startup_hooks_by_rank=hooks)
    artifacts.clear()
    hooks.clear()
    assert spec.startup_artifacts == (artifact,)
    assert spec.startup_hooks_by_rank == (hook, None)


@pytest.mark.parametrize(
    "field,value",
    [("startup_artifacts", [object()]), ("startup_hooks_by_rank", [None]), ("startup_hooks_by_rank", [object(), None])],
)
def test_group_spec_rejects_invalid_startup_configuration(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        _spec(**{field: value})


def test_hook_is_frozen_and_snapshots_nested_json():
    api = _preparation_api()
    parameters = {"nested": {"devices": [1, 2]}, "optional": None}
    hook = api.NodeStartupHook("user_module:Factory.prepare", parameters)
    parameters["nested"]["devices"].append(3)
    expected = {
        "import_path": "user_module:Factory.prepare",
        "parameters": {"nested": {"devices": [1, 2]}, "optional": None},
    }
    assert json.loads(json.dumps(hook.to_manifest())) == expected
    exported = hook.to_manifest()
    exported["parameters"]["nested"]["devices"].clear()
    assert hook.to_manifest() == expected
    with pytest.raises(dataclasses.FrozenInstanceError):
        hook.import_path = "other:prepare"


@pytest.mark.parametrize(
    "target", ["", "module", "module:", ":prepare", "a/b:prepare", "a:1bad", "a:<locals>.f", "a::f"]
)
def test_hook_rejects_non_import_paths(target):
    with pytest.raises(ValueError, match="import"):
        _preparation_api().NodeStartupHook(target)


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        [],
        {1: "bad"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"nested": {"bad": object()}},
        {"tuple": (1, 2)},
    ],
)
def test_hook_rejects_non_json_object_or_nonfinite_parameters(parameters):
    with pytest.raises((TypeError, ValueError), match="JSON|json|parameter|finite"):
        _preparation_api().NodeStartupHook("user_module:prepare", parameters)


def test_parent_manifest_contains_descriptors_and_rank_specific_hooks(monkeypatch, tmp_path):
    api = _preparation_api()
    artifact, _payload = _archive(tmp_path)
    hook = api.NodeStartupHook("user_module:prepare", {"local_device": 0})
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_mpirun_worker_group(_spec(startup_artifacts=(artifact,), startup_hooks_by_rank=(hook, None)))
    mailbox = _Mailbox([])
    worker._worker = MagicMock()
    process = types.SimpleNamespace(pid=42, poll=lambda: None)
    monkeypatch.setattr(mailbox_mod.MpiGroupMailbox, "create", lambda **kwargs: mailbox)
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        worker_mod.threading, "Thread", lambda *args, **kwargs: types.SimpleNamespace(start=lambda: None)
    )
    try:
        worker._activate_mpirun_worker_groups(time.monotonic() + 5.0)
        group = worker._mpi_l3_groups[0]
        manifest = json.loads(Path(group.manifest_path).read_text(encoding="utf-8"))
        assert manifest["startup_artifacts"] == [artifact.to_manifest()]
        assert manifest["rank_manifests"][0]["startup_hook"] == hook.to_manifest()
        assert manifest["rank_manifests"][1]["startup_hook"] is None
        assert "payload" not in manifest["startup_artifacts"][0]
        assert "cache_root" not in json.dumps(manifest)
    finally:
        group = worker._mpi_l3_groups[0]
        group.process = None
        group.monitor_thread = None
        worker._worker = None
        worker.close()


def test_rank0_broadcasts_verified_bytes_and_remote_never_opens_parent_paths(monkeypatch, tmp_path):
    artifact, payload = _archive(tmp_path)
    manifest_path = tmp_path / "parent-manifest.json"
    manifest_path.write_text(json.dumps({"startup_artifacts": [artifact.to_manifest()]}), encoding="utf-8")
    root = _Broadcast(0)
    local = mpi_l3_session._load_group_manifest_from_rank0(root, str(manifest_path))
    assert local["startup_artifacts"] == [
        {"protocol_version": 1, "sha256": artifact.sha256, "module_prefix": artifact.module_prefix, "payload": payload}
    ]
    assert "archive_path" not in local["startup_artifacts"][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("remote rank attempted parent filesystem access")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    remote = mpi_l3_session._load_group_manifest_from_rank0(_Broadcast(1, root.envelope), str(manifest_path))
    assert remote == local


def test_rank0_archive_read_error_is_broadcast_to_every_rank(tmp_path):
    artifact, _payload = _archive(tmp_path)
    Path(artifact.archive_path).unlink()
    manifest_path = tmp_path / "parent-manifest.json"
    manifest_path.write_text(json.dumps({"startup_artifacts": [artifact.to_manifest()]}), encoding="utf-8")
    root = _Broadcast(0)
    with pytest.raises(RuntimeError, match="rank0.*") as root_error:
        mpi_l3_session._load_group_manifest_from_rank0(root, str(manifest_path))
    with pytest.raises(RuntimeError) as peer_error:
        mpi_l3_session._load_group_manifest_from_rank0(_Broadcast(1, root.envelope), str(manifest_path))
    assert str(peer_error.value) == str(root_error.value)
    assert artifact.sha256 in str(root_error.value)


def test_manifest_payload_read_consumes_the_same_absolute_startup_budget(monkeypatch, tmp_path):
    now = [100.0]
    monkeypatch.setattr(mpi_l3_session.time, "monotonic", lambda: now[0])
    artifact, payload = _archive(tmp_path)
    manifest_path = tmp_path / "group.json"
    manifest_path.write_text(
        json.dumps(
            {
                "startup_artifacts": [artifact.to_manifest()],
                "rank_manifests": [{"startup_remaining_s": 20.0, "session_timeout_s": 5.0}],
            }
        ),
        encoding="utf-8",
    )
    original = mpi_l3_session.read_startup_artifacts

    def read(descriptions):
        now[0] += 3.0
        return original(descriptions)

    monkeypatch.setattr(mpi_l3_session, "read_startup_artifacts", read)
    group = mpi_l3_session._load_group_manifest_from_rank0(_Broadcast(0), str(manifest_path))
    assert group["startup_artifacts"][0]["payload"] == payload
    assert group["rank_manifests"][0]["startup_deadline_monotonic"] == 120.0


def test_archive_is_active_before_real_dispatcher_import(session_harness, tmp_path, monkeypatch):
    state = session_harness
    artifact, payload = _archive(tmp_path, "mpi_startup_real_import")
    state.group["startup_artifacts"] = [
        {"protocol_version": 1, "sha256": artifact.sha256, "module_prefix": artifact.module_prefix, "payload": payload}
    ]
    digest = compute_callable_hashid(build_python_import_descriptor(artifact.module_prefix + ".entry", "dispatch"))
    state.manifest["remote_task_dispatcher"] = [
        {"kind": "PYTHON_IMPORT", "hashid": digest, "target": f"{artifact.module_prefix}.entry:dispatch"}
    ]
    monkeypatch.delitem(sys.modules, artifact.module_prefix, raising=False)
    monkeypatch.delitem(sys.modules, artifact.module_prefix + ".entry", raising=False)
    assert not (tmp_path / "node-cache").exists()
    assert state.run() == 0, state.mailbox.errors
    assert state.processor_registry[hashid_to_digest(digest)]() is None
    assert state.mailbox.ready
    assert len(state.comm.statuses) == 4


def test_hook_receives_local_chip_handles_before_main_thread_init_and_closes_after_drain(
    session_harness, hook_module, monkeypatch, tmp_path
):
    state = session_harness
    artifact, payload = _archive(tmp_path, "mpi_startup_bound_import")
    state.group["startup_artifacts"] = [
        {"protocol_version": 1, "sha256": artifact.sha256, "module_prefix": artifact.module_prefix, "payload": payload}
    ]
    digest = b"c" * 32
    handle = object()
    python_handle = object()
    monkeypatch.setattr(
        mpi_l3_session,
        "_install_manifest_inner_registry",
        lambda *args: {
            (CallableKind.CHIP_CALLABLE, digest): handle,
            (CallableKind.PYTHON_IMPORT, b"p" * 32): python_handle,
        },
    )

    class HookState:
        def task_completed(self):
            state.events.append("task_completed")

        def close(self):
            assert "worker_close" in state.events and "processor_stop" in state.events
            state.events.append("hook_close")

    def prepare(context):
        assert threading.current_thread() is threading.main_thread()
        assert context.rank == 0
        assert context.inner_worker is state.worker and not state.worker.initialized
        assert context.chip_handles == {digest.hex(): handle}
        assert context.parameters == {"label": "node-zero"}
        assert (context.artifact_roots[artifact.sha256] / "entry.py").is_file()
        importlib.import_module(artifact.module_prefix + ".entry").BOUND = handle
        state.events.append("hook")
        return HookState()

    hook_module.prepare = prepare
    state.manifest["startup_hook"] = {
        "import_path": hook_module.__name__ + ":prepare",
        "parameters": {"label": "node-zero"},
    }

    def initialize_children():
        _assert_fork_inherits_binding(artifact.module_prefix + ".entry", handle)
        state.events.append("bound")

    state.init_callback = initialize_children
    assert state.run() == 0
    assert callable(state.task_completed)
    state.task_completed()
    assert state.events[-1] == "task_completed"
    assert state.events.index("hook") < state.events.index("init") < state.events.index("ready")
    assert "bound" in state.events
    assert state.events.index("worker_close") < state.events.index("hook_close")


@pytest.mark.parametrize("gate,stage", [(1, "artifact"), (2, "registry"), (3, "hook"), (4, "init")])
def test_peer_failure_at_each_stage_blocks_ready_and_later_stages(session_harness, hook_module, gate, stage):
    state = session_harness
    hook_module.prepare = lambda context: state.events.append("hook")
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    assert state.run(_SessionComm(failure_gate=gate)) == 1
    assert not state.mailbox.ready
    assert len(state.comm.statuses) == gate
    error = state.mailbox.terminal_reason().lower()
    assert "rank 1" in error and stage in error and "peer injected failure" in error
    if gate < 3:
        assert "hook" not in state.events
    if gate < 4:
        assert "init" not in state.events
    assert "worker_close" in state.events and state.mailbox.closed


@pytest.mark.parametrize("stage", ["artifact", "registry", "hook", "init"])
def test_local_startup_error_names_rank_and_stage_and_stops_before_ready(
    session_harness, hook_module, monkeypatch, tmp_path, stage
):
    state = session_harness

    def fail(*args):
        raise ValueError("injected local failure")

    if stage == "artifact":
        artifact, payload = _archive(tmp_path)
        state.group["startup_artifacts"] = [
            {"protocol_version": 1, "sha256": "0" * 64, "module_prefix": artifact.module_prefix, "payload": payload}
        ]
    elif stage == "registry":
        monkeypatch.setattr(mpi_l3_session, "_install_manifest_inner_registry", fail)
    elif stage == "hook":
        hook_module.prepare = fail
        state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    else:
        state.init_error = ValueError("injected local failure")
    assert state.run() == 1
    error = state.mailbox.terminal_reason().lower()
    assert "rank 0" in error and stage in error
    if stage == "artifact":
        assert "0" * 64 in error
    else:
        assert "injected local failure" in error
    assert not state.mailbox.ready and state.mailbox.closed
    assert state.comm.statuses[-1][1] is False


@pytest.mark.parametrize("stage", ["registry", "hook"])
def test_import_stage_error_includes_supplied_artifact_digest(session_harness, tmp_path, stage):
    state = session_harness
    artifact, payload = _archive(tmp_path, "mpi_startup_error_" + stage)
    state.group["startup_artifacts"] = [
        {"protocol_version": 1, "sha256": artifact.sha256, "module_prefix": artifact.module_prefix, "payload": payload}
    ]
    target = artifact.module_prefix + ".entry:missing_function"
    if stage == "registry":
        digest = compute_callable_hashid(
            build_python_import_descriptor(artifact.module_prefix + ".entry", "missing_function")
        )
        state.manifest["remote_task_dispatcher"] = [{"kind": "PYTHON_IMPORT", "hashid": digest, "target": target}]
    else:
        state.manifest["startup_hook"] = {"import_path": target, "parameters": {}}
    assert state.run() == 1
    error = state.mailbox.terminal_reason()
    assert stage in error and "rank 0" in error
    assert artifact.sha256 in error


@pytest.mark.skipif(sys.platform == "win32", reason="MPI rank drain probe uses POSIX fork")
def test_wedged_processor_exits_rank_without_closing_live_hook_state(
    session_harness, hook_module, monkeypatch, tmp_path
):
    state = session_harness
    hook_closed = tmp_path / "hook-closed"
    worker_closed = tmp_path / "worker-closed"

    class HookState:
        def close(self):
            hook_closed.write_text("closed", encoding="utf-8")

    def prepare(context):
        context.inner_worker.close = lambda: worker_closed.write_text("closed", encoding="utf-8")
        return HookState()

    def stalled_processor(connection, *args):
        connection.sendall(encode_frame(FrameHeader(FrameType.HELLO, 11, 3, 0), b""))
        threading.Event().wait(10.0)

    hook_module.prepare = prepare
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    monkeypatch.setattr(mpi_l3_session, "_run_command_loop", stalled_processor)

    def run_child():
        original_join = threading.Thread.join

        def bounded_join(thread, timeout=None):
            if thread.name.startswith("simpler-mpi-command-rank-"):
                timeout = min(timeout or 0.05, 0.05)
            return original_join(thread, timeout)

        threading.Thread.join = bounded_join
        state.run()

    child = multiprocessing.get_context("fork").Process(target=run_child)
    child.start()
    try:
        child.join(timeout=2.0)
        assert child.exitcode == 124, "rank must fail-stop while its command processor still uses hook state"
        assert not worker_closed.exists() and not hook_closed.exists()
    finally:
        if child.is_alive():
            child.terminate()
        child.join(timeout=2.0)


@pytest.mark.skipif(sys.platform == "win32", reason="MPI rank teardown probe uses POSIX fork")
def test_failed_worker_close_keeps_live_hook_state_until_rank_exit(session_harness, hook_module, tmp_path):
    state = session_harness
    hook_closed = tmp_path / "unsafe-hook-close"

    class HookState:
        def close(self):
            hook_closed.write_text("closed", encoding="utf-8")

    hook_module.prepare = lambda context: HookState()
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    state.close_error = RuntimeError("accepted run fences still live; teardown deferred")
    # Real Worker.close explicitly retains its tree on a failed drain. Model
    # that existing inventory contract at the actual MPI session cleanup seam.
    state.init_callback = lambda: setattr(state.worker, "_has_live_resources", lambda: True)
    child = multiprocessing.get_context("fork").Process(target=state.run)
    child.start()
    try:
        child.join(timeout=2.0)
        assert child.exitcode == 124, "Unconfirmed worker teardown must retain hook state and fail-stop the rank"
        assert not hook_closed.exists(), "Live invocation storage must survive a failed inner-worker close"
    finally:
        if child.is_alive():
            child.terminate()
        child.join(timeout=2.0)


def test_failed_init_keeps_hook_alive_until_worker_close_and_preserves_startup_cause(session_harness, hook_module):
    state = session_harness

    class HookState:
        def close(self):
            assert "worker_close" in state.events
            state.events.append("hook_close")
            raise RuntimeError("secondary hook cleanup error")

    hook_module.prepare = lambda context: HookState()
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    state.init_error = RuntimeError("primary init failure")
    state.close_error = RuntimeError("secondary worker cleanup error")
    # This startup mock has no live tree or accepted work. Supplying the real
    # inventory contract distinguishes it from an unconfirmed failed teardown.
    state.init_callback = lambda: setattr(state.worker, "_has_live_resources", lambda: False)
    assert state.run() == 1
    assert "primary init failure" in state.mailbox.terminal_reason()
    assert "hook_close" in state.events and state.mailbox.closed
    assert not state.mailbox.ready


@pytest.mark.parametrize("result", [42, object()])
def test_hook_rejects_noncloseable_return_value(session_harness, hook_module, result):
    state = session_harness
    hook_module.prepare = lambda context: result
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    assert state.run() == 1
    assert "hook" in state.mailbox.terminal_reason().lower()
    assert "init" not in state.events


def test_startup_uses_one_absolute_deadline_before_artifact_and_hook_stages(session_harness, monkeypatch, hook_module):
    state = session_harness
    now = [100.0]
    monkeypatch.setattr(mpi_l3_session.time, "monotonic", lambda: now[0])
    state.manifest["startup_remaining_s"] = 20.0
    hook_module.prepare = lambda context: now.__setitem__(0, 111.0)
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}
    assert state.run() == 0
    assert state.worker.startup_deadline == 120.0


@pytest.mark.skipif(sys.platform == "win32", reason="MPI job timeout probe uses POSIX fork")
@pytest.mark.parametrize("stall", ["hook", "peer_collective"])
def test_stalled_startup_exits_rank_within_budget(session_harness, hook_module, stall):
    state = session_harness
    state.manifest["startup_remaining_s"] = 0.1
    hook_module.prepare = lambda context: threading.Event().wait(10.0)
    state.manifest["startup_hook"] = {"import_path": hook_module.__name__ + ":prepare", "parameters": {}}

    class StallComm(_SessionComm):
        def allgather(self, value):
            if stall == "peer_collective":
                threading.Event().wait(10.0)
            return super().allgather(value)

    child = multiprocessing.get_context("fork").Process(target=lambda: state.run(StallComm()))
    child.start()
    try:
        child.join(timeout=2.0)
        assert child.exitcode == 124, "startup deadline failed to terminate the stalled rank"
    finally:
        if child.is_alive():
            child.terminate()
        child.join(timeout=2.0)
