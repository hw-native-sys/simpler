# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""P0.3 — simulation Remote L3 startup budget / activation correctness.

Device-free tests that pin the single-root-deadline contract:

- adding Remote L3 workers must not multiply the startup budget: every remote
  draws from one root deadline via a decreasing remaining slice, propagated as
  the manifest ``startup_remaining_s`` (distinct from the runtime
  ``session_timeout_s``);
- sessions are opened and activated only in ``_activate_remote_sessions`` (after
  the last local fork), never pre-fork in ``_init_hierarchical``;
- non-positive / non-finite startup and session timeouts fail before any
  resource is created, on both the parent and the remote (untrusted-wire) side.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import simpler.remote_l3_session as session_mod
import simpler.remote_l3_worker as daemon_mod
import simpler.worker as worker_mod
from simpler.worker import RemoteWorkerSpec, Worker


def _spec(port: int = 19073) -> RemoteWorkerSpec:
    return RemoteWorkerSpec(endpoint=f"127.0.0.1:{port}", platform="a2a3sim")


def _l4_with_remotes(n: int, **config) -> Worker:
    w = Worker(level=4, num_sub_workers=0, **config)
    for i in range(n):
        w.add_remote_worker(_spec(19073 + i))
    return w


class TestManifestBudgetField:
    def test_manifest_carries_startup_remaining_distinct_from_session_timeout(self):
        w = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30.0)
        try:
            manifest = w._build_remote_manifest(spec=_spec(), worker_id=0, session_id=1, startup_remaining_s=12.5)
            assert manifest["startup_remaining_s"] == 12.5
            assert manifest["session_timeout_s"] == 30.0
            # The startup budget must not be fixed to the runtime command timeout.
            assert manifest["startup_remaining_s"] != manifest["session_timeout_s"]
        finally:
            w.close()


class TestParentTimeoutValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_remote_session_timeout_rejected(self, bad):
        w = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=bad)
        try:
            with pytest.raises(ValueError, match="remote_session_timeout_s"):
                w._remote_session_timeout_s()
        finally:
            w.close()

    def test_positive_finite_remote_session_timeout_accepted(self):
        w = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=17.0)
        try:
            assert w._remote_session_timeout_s() == 17.0
        finally:
            w.close()


class TestRemoteSideValidation:
    """Both remote entry points validate the wire budget before creating resources."""

    @pytest.mark.parametrize("mod", [session_mod, daemon_mod])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_startup_remaining_rejected(self, mod, bad):
        with pytest.raises(ValueError, match="startup_remaining_s"):
            mod._startup_remaining_s({"startup_remaining_s": bad, "session_timeout_s": 30.0})

    @pytest.mark.parametrize("mod", [session_mod, daemon_mod])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_session_timeout_rejected(self, mod, bad):
        with pytest.raises(ValueError, match="session_timeout_s"):
            mod._session_timeout_s({"session_timeout_s": bad})

    @pytest.mark.parametrize("mod", [session_mod, daemon_mod])
    def test_startup_remaining_falls_back_to_session_timeout(self, mod):
        # A pre-P0.3 parent omits startup_remaining_s; the remote must still bound
        # itself by the (positive-finite) session timeout rather than crash.
        assert mod._startup_remaining_s({"session_timeout_s": 25.0}) == 25.0

    @pytest.mark.parametrize("mod", [session_mod, daemon_mod])
    def test_startup_remaining_used_when_present(self, mod):
        assert mod._startup_remaining_s({"startup_remaining_s": 8.0, "session_timeout_s": 30.0}) == 8.0


class TestParentPreflightBeforeResources:
    """An invalid remote_session_timeout_s fails before the parent builds any
    startup resource — no mailbox shm, no pre-fork _Worker, ever constructed."""

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_timeout_creates_no_startup_resources(self, monkeypatch, bad):
        shm_ctor = MagicMock(side_effect=AssertionError("SharedMemory constructed before timeout preflight"))
        worker_ctor = MagicMock(side_effect=AssertionError("_Worker constructed before timeout preflight"))
        monkeypatch.setattr(worker_mod, "SharedMemory", shm_ctor)
        monkeypatch.setattr(worker_mod, "_Worker", worker_ctor)

        w = Worker(level=4, num_sub_workers=1, remote_session_timeout_s=bad)
        w.add_remote_worker(_spec())
        try:
            with pytest.raises(ValueError, match="remote_session_timeout_s"):
                w._init_hierarchical()
            assert shm_ctor.call_count == 0
            assert worker_ctor.call_count == 0
        finally:
            w.close()

    def test_no_remotes_does_not_validate_remote_timeout(self, monkeypatch):
        # The preflight is gated on actual remote presence: a worker with no
        # remote workers never validates remote_session_timeout_s (only a
        # level >= 4 parent can carry remotes), so an unused/invalid value does
        # not fail an otherwise-valid tree.
        monkeypatch.setattr(worker_mod, "_Worker", MagicMock())
        preflight = MagicMock(side_effect=AssertionError("remote timeout validated without any remote worker"))
        monkeypatch.setattr(Worker, "_remote_session_timeout_s", preflight)

        w = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=-1.0)
        try:
            w._init_hierarchical()
            assert preflight.call_count == 0
        finally:
            w._worker = None
            w.close()

    def test_activate_remote_sessions_no_remotes_is_noop(self, monkeypatch):
        # With no remotes, activation returns before validating the timeout or
        # touching the C++ Worker — the empty tree is a clean no-op.
        preflight = MagicMock(side_effect=AssertionError("remote timeout validated with no remotes to activate"))
        monkeypatch.setattr(Worker, "_remote_session_timeout_s", preflight)

        w = Worker(level=4, num_sub_workers=0)
        mock_worker = MagicMock()
        w._worker = mock_worker
        try:
            w._activate_remote_sessions(time.monotonic() + 5.0)
            assert preflight.call_count == 0
            assert mock_worker.add_remote_l3_socket.call_count == 0
            assert w._remote_sessions == []
        finally:
            w._worker = None
            w.close()


class TestDaemonPreflightBeforeSpawn:
    """The daemon validates both numeric timeouts before any spawn resource —
    ready pipe, manifest tempfile, runner Popen — is created."""

    @staticmethod
    def _manifest(**overrides) -> dict:
        manifest = {
            "session_id": 1,
            "worker_id": 0,
            "parent_worker_level": 4,
            "remote_worker_level": 3,
            "platform": "a2a3sim",
            "transport": "sim",
            "session_timeout_s": 30.0,
            "startup_remaining_s": 10.0,
        }
        manifest.update(overrides)
        return manifest

    @pytest.mark.parametrize(
        ("manifest_kwargs", "match"),
        [
            ({"session_timeout_s": 0.0}, "session_timeout_s"),
            ({"session_timeout_s": -1.0}, "session_timeout_s"),
            ({"session_timeout_s": float("inf")}, "session_timeout_s"),
            ({"session_timeout_s": float("nan")}, "session_timeout_s"),
            ({"startup_remaining_s": 0.0}, "startup_remaining_s"),
            ({"startup_remaining_s": -1.0}, "startup_remaining_s"),
            ({"startup_remaining_s": float("inf")}, "startup_remaining_s"),
            ({"startup_remaining_s": float("nan")}, "startup_remaining_s"),
        ],
    )
    def test_invalid_timeout_spawns_nothing(self, monkeypatch, manifest_kwargs, match):
        pipe_mock = MagicMock(side_effect=AssertionError("os.pipe before timeout preflight"))
        tempfile_mock = MagicMock(side_effect=AssertionError("NamedTemporaryFile before timeout preflight"))
        popen_mock = MagicMock(side_effect=AssertionError("Popen before timeout preflight"))
        monkeypatch.setattr(daemon_mod.os, "pipe", pipe_mock)
        monkeypatch.setattr(daemon_mod.tempfile, "NamedTemporaryFile", tempfile_mock)
        monkeypatch.setattr(daemon_mod.subprocess, "Popen", popen_mock)

        with pytest.raises(ValueError, match=match):
            daemon_mod._start_session(self._manifest(**manifest_kwargs))
        assert pipe_mock.call_count == 0
        assert tempfile_mock.call_count == 0
        assert popen_mock.call_count == 0

    def test_valid_session_timeout_but_invalid_startup_remaining_spawns_nothing(self, monkeypatch):
        # A valid session_timeout_s must not shield an invalid startup_remaining_s
        # from the pre-spawn gate.
        popen_mock = MagicMock(side_effect=AssertionError("Popen before timeout preflight"))
        monkeypatch.setattr(daemon_mod.subprocess, "Popen", popen_mock)
        with pytest.raises(ValueError, match="startup_remaining_s"):
            daemon_mod._start_session(self._manifest(session_timeout_s=30.0, startup_remaining_s=-1.0))
        assert popen_mock.call_count == 0


class TestActivationAfterFork:
    def test_init_hierarchical_does_not_open_sessions(self, monkeypatch):
        """Opening a session (which starts the remote subtree) is deferred out of
        the pre-fork _init_hierarchical into post-fork _activate_remote_sessions."""

        def _must_not_open(self, **_kwargs):
            raise AssertionError("remote session opened before the last local fork")

        monkeypatch.setattr(Worker, "_open_remote_session", _must_not_open)

        w = _l4_with_remotes(2)
        try:
            w._init_hierarchical()
            assert w._remote_sessions == []
        finally:
            if w._worker is not None:
                w._worker.close()
                w._worker = None
            w.close()


class TestSingleRootBudget:
    def _drive_activation(self, monkeypatch, *, n_remotes, root_budget_s, per_open_delay_s=0.02):
        granted: list[float] = []
        socket_timeouts: list[float] = []

        def fake_open(self, *, spec, worker_id, session_id, timeout_s, startup_remaining_s):
            granted.append(startup_remaining_s)
            socket_timeouts.append(timeout_s)
            time.sleep(per_open_delay_s)
            return _FakeSession(worker_id, session_id)

        monkeypatch.setattr(Worker, "_open_remote_session", fake_open)

        w = _l4_with_remotes(n_remotes)
        w._worker = MagicMock()
        deadline = time.monotonic() + root_budget_s
        try:
            w._activate_remote_sessions(deadline)
        finally:
            w._worker = None
            w.close()
        return granted, socket_timeouts, w

    def test_budget_not_multiplied_by_remote_count(self, monkeypatch):
        granted, socket_timeouts, _ = self._drive_activation(monkeypatch, n_remotes=3, root_budget_s=5.0)
        assert len(granted) == 3
        # Every remote's granted startup budget fits inside the single root budget.
        for g in granted:
            assert 0 < g <= 5.0
        # And the budget shrinks per remote (shared deadline), never a fresh full
        # timeout each — the multiplication bug.
        assert granted[0] > granted[1] > granted[2]
        # Socket handshake timeout tracks the same remaining budget.
        for s, g in zip(socket_timeouts, granted, strict=True):
            assert s == g

    def test_attach_called_once_per_remote(self, monkeypatch):
        granted: list[float] = []

        def fake_open(self, *, spec, worker_id, session_id, timeout_s, startup_remaining_s):
            granted.append(startup_remaining_s)
            return _FakeSession(worker_id, session_id)

        monkeypatch.setattr(Worker, "_open_remote_session", fake_open)
        w = _l4_with_remotes(2)
        mock_worker = MagicMock()
        w._worker = mock_worker
        try:
            w._activate_remote_sessions(time.monotonic() + 5.0)
            assert mock_worker.add_remote_l3_socket.call_count == 2
            assert len(w._remote_sessions) == 2
        finally:
            w._worker = None
            w.close()

    def test_expired_deadline_fails_fast_without_opening(self, monkeypatch):
        opened = []

        def fake_open(self, *, spec, worker_id, session_id, timeout_s, startup_remaining_s):
            opened.append(worker_id)
            return _FakeSession(worker_id, session_id)

        monkeypatch.setattr(Worker, "_open_remote_session", fake_open)
        w = _l4_with_remotes(1)
        w._worker = MagicMock()
        try:
            with pytest.raises(RuntimeError, match="startup deadline exceeded"):
                w._activate_remote_sessions(time.monotonic() - 1.0)
            assert opened == []
            assert w._remote_sessions == []
        finally:
            w._worker = None
            w.close()


class _FakeSession:
    """Minimal stand-in for _RemoteSession that add_remote_l3_socket can read."""

    def __init__(self, worker_id: int, session_id: int):
        self.worker_id = worker_id
        self.session_id = session_id
        self.command_host = "127.0.0.1"
        self.command_port = 1
        self.health_host = "127.0.0.1"
        self.health_port = 2
        self.pid = 0
