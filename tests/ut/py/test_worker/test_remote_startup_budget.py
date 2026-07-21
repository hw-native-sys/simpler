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
