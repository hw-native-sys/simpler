# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler.worker import Worker


@pytest.mark.parametrize("startup_abort", [False, True])
def test_unconfirmed_host_owner_does_not_prevent_child_shutdown(monkeypatch, startup_abort):
    worker = Worker(level=3, num_sub_workers=0)
    events = []

    def retained():
        events.append("owner")
        raise RuntimeError("Host owner retained: unconfirmed views")

    worker._cleanup_journal.add("host-domain", "local", retained)
    monkeypatch.setattr(worker, "_broadcast_child_shutdown", lambda _shms: events.append("shutdown"))
    monkeypatch.setattr(worker, "_reclaim_child_groups", lambda _deadline: events.append("reap"))
    monkeypatch.setattr(worker, "_abort_hierarchical", lambda **_kwargs: events.append("abort"))
    with pytest.raises(RuntimeError, match="unconfirmed views"):
        worker._teardown_worker_tree(startup_abort=startup_abort)
    assert events[0] == "owner" and events[-1] == "owner"
    assert ("abort" if startup_abort else "reap") in events[1:-1]
    assert not worker._cleanup_journal.empty


def test_unrelated_pretransport_failure_still_fences_teardown(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    monkeypatch.setattr(worker, "_flush_pending_remote_frees", Mock(side_effect=RuntimeError("remote free owed")))
    shutdown = Mock()
    monkeypatch.setattr(worker, "_broadcast_child_shutdown", shutdown)
    with pytest.raises(RuntimeError, match="remote free owed"):
        worker._teardown_worker_tree(startup_abort=False)
    shutdown.assert_not_called()


def test_direct_host_release_failure_allows_finalize_but_retains_owner():
    worker = Worker(level=2, platform="a2a3sim", device_id=0)
    chip = Mock()
    chip.workspace_report.return_value = ("disabled", {})
    chip.comm_release_domain_windows.side_effect = RuntimeError("release failed")
    worker._chip_worker = chip
    resource = SimpleNamespace(closed=False, buffer=None, views_released=False, close=Mock())
    worker._direct_host_domains["identity"] = (resource, 9, 11, ctypes.c_uint64(1))
    worker._cleanup_journal.add("host-domain", "direct-11", lambda: worker._release_direct_host_resource("identity"))
    with pytest.raises(RuntimeError, match="native teardown did not confirm"):
        worker._teardown_worker_tree(startup_abort=False)
    chip.finalize.assert_called_once()
    chip.comm_release_domain_windows.assert_called_once()
    assert worker._chip_worker is None
    assert not resource.views_released
    resource.close.assert_not_called()
    assert "identity" in worker._direct_host_domains
    with pytest.raises(RuntimeError, match="native teardown did not confirm"):
        worker._teardown_worker_tree(startup_abort=False)
    chip.comm_release_domain_windows.assert_called_once()


def test_child_journal_survivor_is_not_treated_as_view_release(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    owner_release = Mock(side_effect=RuntimeError("Host views unknown"))
    worker._cleanup_journal.add("host-domain", "local", owner_release)

    def reap(_deadline):
        worker._cleanup_journal.add("child", "survivor", Mock(side_effect=RuntimeError("child still alive")))

    monkeypatch.setattr(worker, "_reclaim_child_groups", reap)
    with pytest.raises(RuntimeError, match="child still alive"):
        worker._teardown_worker_tree(startup_abort=False)
    assert not worker._chip_pids
    assert len(worker._cleanup_journal) == 2
