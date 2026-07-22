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

import socket
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


class _FakeClock:
    """Deterministic monotonic clock; advance it by explicit ticks so every
    derived ``deadline - now()`` value is exact (no real-sleep magnitude races)."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestManifestBudgetField:
    def test_manifest_carries_startup_remaining_distinct_from_session_timeout(self):
        w = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30.0)
        try:
            manifest = w._build_remote_manifest(spec=_spec(), worker_id=0, session_id=1, startup_remaining_s=12.5)
            assert manifest["startup_remaining_s"] == 12.5
            assert manifest["session_timeout_s"] == 30.0
            # The startup budget must not be fixed to the runtime command timeout.
            assert manifest["startup_remaining_s"] != manifest["session_timeout_s"]
            # The cross-host wire carries a duration, never an absolute deadline
            # (monotonic clocks are not comparable across machines).
            assert isinstance(manifest["startup_remaining_s"], float)
            assert "startup_deadline" not in manifest
            assert "deadline" not in manifest
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


class TestNumericEndpointContract:
    """Remote L3 endpoints are numeric-only (or localhost); a hostname is rejected
    at add_remote_worker time, before any startup resource exists."""

    def test_add_remote_worker_rejects_hostname_at_registration(self):
        w = Worker(level=4, num_sub_workers=0)
        try:
            with pytest.raises(ValueError, match="numeric IP"):
                w.add_remote_worker(RemoteWorkerSpec(endpoint="node17:19073", platform="a2a3sim"))
            assert w._remote_worker_specs == []  # rejected before it is registered
        finally:
            w.close()

    def test_add_remote_worker_accepts_numeric_and_localhost(self):
        w = Worker(level=4, num_sub_workers=0)
        try:
            w.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19073", platform="a2a3sim"))
            w.add_remote_worker(RemoteWorkerSpec(endpoint="localhost:19074", platform="a2a3sim"))
            assert len(w._remote_worker_specs) == 2
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
    def _drive_activation(self, monkeypatch, *, n_remotes, root_budget_s, per_open_tick_s=0.5):
        clock = _FakeClock()
        monkeypatch.setattr(worker_mod.time, "monotonic", clock.monotonic)

        deadlines: list[float] = []
        granted: list[float] = []  # the startup_remaining_s the real open would derive

        def fake_open(self, *, spec, worker_id, session_id, deadline):
            deadlines.append(deadline)
            granted.append(deadline - worker_mod.time.monotonic())
            clock.advance(per_open_tick_s)  # models per-remote open cost, deterministically
            return _FakeSession(worker_id, session_id)

        monkeypatch.setattr(Worker, "_open_remote_session", fake_open)

        w = _l4_with_remotes(n_remotes)
        mock_worker = MagicMock()
        w._worker = mock_worker
        deadline = clock.monotonic() + root_budget_s
        try:
            w._activate_remote_sessions(deadline)
        finally:
            w._worker = None
            w.close()
        return deadlines, granted, mock_worker

    def test_budget_not_multiplied_by_remote_count(self, monkeypatch):
        deadlines, granted, _ = self._drive_activation(monkeypatch, n_remotes=3, root_budget_s=5.0)
        assert len(granted) == 3
        # THE anti-multiplication invariant: one shared absolute root deadline is
        # threaded to every remote — the bug would surface as distinct, freshly
        # computed deadlines per remote.
        assert len(set(deadlines)) == 1
        # Each remote's derived startup slice fits inside the single root budget…
        for g in granted:
            assert 0 < g <= 5.0
        # …and shrinks per remote (shared deadline + advancing clock).
        assert granted[0] > granted[1] > granted[2]

    def test_attach_and_runtime_timeouts_are_split(self, monkeypatch):
        _deadlines, _granted, mock_worker = self._drive_activation(monkeypatch, n_remotes=3, root_budget_s=5.0)
        calls = mock_worker.add_remote_l3_socket.call_args_list
        assert len(calls) == 3
        # attach_timeout / runtime_timeout are the last two positional args.
        attach = [c.args[-2] for c in calls]
        runtime = [c.args[-1] for c in calls]
        # runtime_timeout is the full runtime command budget: constant, ==
        # session_timeout, NOT min(session_timeout, remaining), never shrinking.
        assert runtime == [30.0, 30.0, 30.0]
        # attach_timeout tracks the remaining root budget: shrinks, bounded, and
        # is a distinct quantity from runtime_timeout (the split the old min() lost).
        assert attach[0] > attach[1] > attach[2]
        for a in attach:
            assert 0 < a <= 5.0
            assert a != 30.0

    def test_final_recheck_fires_when_last_attach_overruns_deadline(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(worker_mod.time, "monotonic", clock.monotonic)

        def fake_open(self, *, spec, worker_id, session_id, deadline):
            clock.advance(0.4)  # open leaves budget > 0, so both per-iteration rechecks pass
            return _FakeSession(worker_id, session_id)

        monkeypatch.setattr(Worker, "_open_remote_session", fake_open)
        w = _l4_with_remotes(1)
        mock_worker = MagicMock()
        # The last attach itself consumes the remaining slice, so now >= deadline
        # only AFTER add_remote_l3_socket returns — only a post-loop recheck catches it.
        mock_worker.add_remote_l3_socket.side_effect = lambda *a, **k: clock.advance(1.0)
        w._worker = mock_worker
        deadline = clock.monotonic() + 1.0  # 0.4 (open) + 1.0 (attach) > 1.0
        try:
            with pytest.raises(RuntimeError, match="startup deadline exceeded after attach"):
                w._activate_remote_sessions(deadline)
            assert mock_worker.add_remote_l3_socket.call_count == 1  # attach did run
            assert len(w._remote_sessions) == 1  # left recorded so init()'s rollback closes it
        finally:
            w._worker = None
            w.close()

    def test_attach_called_once_per_remote(self, monkeypatch):
        def fake_open(self, *, spec, worker_id, session_id, deadline):
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

        def fake_open(self, *, spec, worker_id, session_id, deadline):
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


class TestReadyCommitGate:
    """The final root-deadline gate lives in init()'s READY-commit critical
    section, so a thread descheduled after attach cannot publish READY late."""

    def test_gate_fires_inside_commit_before_publishing_ready(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(worker_mod.time, "monotonic", clock.monotonic)

        def fake_start(self):
            # Time crosses the deadline during post-startup work. No remote
            # session on THIS worker — the gate must still fire (a local child
            # could have remote descendants this deadline also bounds).
            clock.advance(self._startup_timeout_s + 1.0)

        monkeypatch.setattr(Worker, "_init_hierarchical", lambda self: None)
        monkeypatch.setattr(Worker, "_start_hierarchical", fake_start)
        monkeypatch.setattr(Worker, "_cleanup_partial_init", lambda self: None)

        w = Worker(level=4, num_sub_workers=0)
        try:
            with pytest.raises(RuntimeError, match="startup deadline exceeded before READY"):
                w.init()
            assert w._lifecycle is not worker_mod._Lifecycle.READY
        finally:
            w._worker = None
            w.close()


class TestBoundedConnect:
    """Resolution and every per-address connect are bounded by the single root
    deadline (unlike socket.create_connection, which restarts the clock per
    address and never bounds getaddrinfo)."""

    def test_resolve_fails_fast_past_deadline_without_calling_getaddrinfo(self, monkeypatch):
        gai = MagicMock(side_effect=AssertionError("getaddrinfo on an already-past deadline"))
        monkeypatch.setattr(worker_mod.socket, "getaddrinfo", gai)
        with pytest.raises(TimeoutError, match="startup deadline exceeded"):
            Worker._resolve_within_deadline("localhost", 9, time.monotonic() - 1.0)
        assert gai.call_count == 0

    def test_resolve_rejects_hostname_instead_of_risking_unbounded_dns(self):
        # A hostname would need an unbounded, uncancellable getaddrinfo; reject it
        # outright rather than let a hung resolver pin init() in INITIALIZING.
        with pytest.raises(ValueError, match="must be a numeric IP"):
            Worker._resolve_within_deadline("example.invalid", 9, time.monotonic() + 5.0)

    def test_resolve_accepts_numeric_and_localhost(self):
        for host in ("127.0.0.1", "localhost"):
            infos = Worker._resolve_within_deadline(host, 9, time.monotonic() + 5.0)
            assert infos
            assert infos[0][4][0] == "127.0.0.1"

    def test_resolve_never_issues_a_dns_query(self, monkeypatch):
        # AI_NUMERICHOST must be set so getaddrinfo cannot block on NSS/DNS.
        calls = []
        real = worker_mod.socket.getaddrinfo

        def spy(host, port, **kwargs):
            calls.append(kwargs.get("flags", 0))
            return real(host, port, **kwargs)

        monkeypatch.setattr(worker_mod.socket, "getaddrinfo", spy)
        Worker._resolve_within_deadline("127.0.0.1", 9, time.monotonic() + 5.0)
        assert calls and (calls[0] & socket.AI_NUMERICHOST)

    def test_connect_bounds_each_address_by_shrinking_remaining(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(worker_mod.time, "monotonic", clock.monotonic)
        addr_a = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 9))
        addr_b = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 9))
        monkeypatch.setattr(Worker, "_resolve_within_deadline", lambda h, p, d: [addr_a, addr_b])

        timeouts: list[float] = []
        made: list = []

        class _FakeSock:
            def settimeout(self, t):
                timeouts.append(t)

            def connect(self, sockaddr):
                clock.advance(3.0)  # each connect attempt burns budget
                if sockaddr == addr_a[4]:
                    raise ConnectionRefusedError("first address black-holed")

            def close(self):
                pass

        def fake_socket(*_a):
            s = _FakeSock()
            made.append(s)
            return s

        monkeypatch.setattr(worker_mod.socket, "socket", fake_socket)
        sock = Worker._connect_within_deadline("host", 9, clock.monotonic() + 10.0)

        # The second address gets the SHRUNKEN remaining (10 - 3), not a fresh full
        # 10s — the per-address multiplication create_connection would allow.
        assert timeouts == [10.0, 7.0]
        assert sock is made[1]

    def test_connect_fails_fast_when_deadline_passes_before_next_address(self, monkeypatch):
        addr = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 9))
        monkeypatch.setattr(Worker, "_resolve_within_deadline", lambda h, p, d: [addr])
        made = MagicMock(side_effect=AssertionError("socket created past deadline"))
        monkeypatch.setattr(worker_mod.socket, "socket", made)
        with pytest.raises(TimeoutError, match="startup deadline exceeded"):
            Worker._connect_within_deadline("h", 9, time.monotonic() - 1.0)
        assert made.call_count == 0


class TestOpenRemoteSession:
    """The single-deadline contract inside _open_remote_session itself."""

    def test_fails_fast_on_past_deadline_without_connecting(self, monkeypatch):
        # Build the worker first: add_remote_worker validates the (numeric) host
        # via getaddrinfo, which we then poison to assert the connect path is skipped.
        w = _l4_with_remotes(1)
        gai = MagicMock(side_effect=AssertionError("getaddrinfo on an already-past deadline"))
        mksock = MagicMock(side_effect=AssertionError("socket() on an already-past deadline"))
        monkeypatch.setattr(worker_mod.socket, "getaddrinfo", gai)
        monkeypatch.setattr(worker_mod.socket, "socket", mksock)
        try:
            with pytest.raises(TimeoutError, match="startup deadline exceeded"):
                w._open_remote_session(spec=_spec(), worker_id=0, session_id=1, deadline=time.monotonic() - 1.0)
            assert gai.call_count == 0
            assert mksock.call_count == 0
        finally:
            w.close()

    def test_recomputes_budget_before_send_and_refuses_nonpositive_slice(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(worker_mod.time, "monotonic", clock.monotonic)
        sent: list = []

        class _FakeSock:
            def settimeout(self, _t):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def fake_connect(self, host, port, deadline):
            clock.advance(10.0)  # resolve + connect consume the entire budget
            return _FakeSock()

        monkeypatch.setattr(Worker, "_connect_within_deadline", fake_connect)
        monkeypatch.setattr(Worker, "_send_remote_daemon_json", lambda self, sock, payload: sent.append(payload))
        w = _l4_with_remotes(1)
        try:
            # connect succeeds within budget, but by send time the budget is gone,
            # so it raises before sending a <= 0 startup_remaining_s the remote bounces.
            with pytest.raises(TimeoutError, match="startup deadline exceeded"):
                w._open_remote_session(spec=_spec(), worker_id=0, session_id=1, deadline=clock.monotonic() + 1.0)
            assert sent == []
        finally:
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
