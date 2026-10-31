# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host registration belongs to the selected platform backend."""

from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock, call

import pytest
from simpler import host_domain_backend as backend
from simpler import worker as worker_module
from simpler.worker import Worker


def test_a3_registration_uses_runtime_identity_and_checks_authorization(monkeypatch):
    native = MagicMock()
    monkeypatch.setattr(backend, "_HostDomainMemory", lambda _path: native)
    chip = MagicMock()
    chip.host_backing_process_identity.return_value = 12345
    owner = backend.host_backing_type("a2a3")(host_runtime_path="/test/host_runtime.so")
    native.require_mapping_support.assert_called_once_with()
    owner.authorize([backend.host_process_identity("a2a3", chip)])
    native.authorize.assert_called_once_with([12345])


def test_optional_host_identity_does_not_enable_unsupported_backing(monkeypatch):
    def unexpected_fabric_call():
        raise AssertionError("non-Fabric communicator requested a Fabric identity")

    chip = MagicMock()
    chip.host_backing_process_identity.side_effect = unexpected_fabric_call
    assert backend.host_process_identity("a2a3sim", chip) == 0
    backend.host_backing_type("a2a3sim")().authorize([0, 0])
    assert backend.host_process_identity("a5", chip) == 0
    with pytest.raises(ValueError, match="not implemented"):
        backend.host_backing_type("a5")


@pytest.mark.parametrize("fail_once", [False, True])
def test_parent_host_bootstrap_reads_pid_replies_before_authorizing_members(monkeypatch, fail_once):
    worker = Worker(level=3, platform="a2a3sim", device_ids=[12, 13], num_sub_workers=0)
    pids, requests, resource = [12345, 23456], [], MagicMock()
    path = "/shared/\u8def\u5f84/rootinfo"

    def control(chip_idx, name):
        nonlocal fail_once
        request = SharedMemory(name=name)
        try:
            buf = request.buf
            assert buf is not None
            rank, count, want_pid, reply_pid = worker_module._COMM_INIT_HEADER.unpack_from(buf)
            assert (rank, count, reply_pid) == (chip_idx, 2, 0)
            offset = worker_module._COMM_INIT_HEADER.size
            assert bytes(buf[offset : offset + len(path.encode()) + 1]) == path.encode() + b"\0"
            requests.append((rank, want_pid))
            if fail_once and want_pid and rank == 1:
                fail_once = False
                raise RuntimeError("PID reply failed")
            worker_module._COMM_INIT_HEADER.pack_into(buf, 0, rank, count, want_pid, pids[rank] if want_pid else 0)
        finally:
            request.close()

    monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: path)
    worker._worker = MagicMock(control_comm_init=control)
    try:
        worker._ensure_comm_base()
        assert worker._comm_base_ready and not worker._host_identity_ready
        assert worker._host_process_ids == {0: 0, 1: 0}
        if fail_once:
            with pytest.raises(RuntimeError, match="PID reply failed"):
                worker._ensure_comm_base(require_host_identity=True)
            assert not worker._host_identity_ready and worker._host_process_ids == {0: 0, 1: 0}
        worker._ensure_comm_base(require_host_identity=True)
        assert worker._host_identity_ready and worker._host_process_ids == dict(enumerate(pids))
        worker._allocate_host_domain_backing(resource, (1, 0))
        assert resource.mock_calls == [call.allocate(), call.backend.authorize([23456, 12345])]
        before = list(requests)
        worker._ensure_comm_base(require_host_identity=True)
        assert requests == before and sorted(requests[:2]) == [(0, 0), (1, 0)]
    finally:
        worker._worker = None
        worker.close()


def test_host_mapping_runtime_support_is_checked_before_allocation(monkeypatch):
    native = MagicMock()
    native.require_mapping_support.side_effect = RuntimeError("missing Host detach entry")
    monkeypatch.setattr(backend, "_HostDomainMemory", lambda _path: native)
    with pytest.raises(RuntimeError, match="missing Host detach"):
        backend.A3HostBacking(host_runtime_path="/test/host_runtime.so")
    native.allocate.assert_not_called()
