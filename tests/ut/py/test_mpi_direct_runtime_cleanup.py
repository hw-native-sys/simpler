# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler import mpi_direct_runtime as runtime_mod
from simpler.mpi_direct_cleanup import MpiDirectCleanupReport, _RankCleanupReporter
from simpler.mpi_direct_topology import MpiDirectTopology


@pytest.mark.parametrize("buffer_fails", [False, True])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_fatal_command_error_is_independent_of_release_result(monkeypatch, buffer_fails, worker_fails):
    topology = MpiDirectTopology.from_dict(
        {
            "controller_host": "127.0.0.1",
            "executor_ranks": [{"rank": 1, "worker_id": 0, "host": "127.0.0.1", "platform": "a2a3sim"}],
        }
    )
    command_error = RuntimeError("controller SLR3 frame identity mismatch")

    def loop(*_args, cleanup_errors=None):
        if buffer_fails and cleanup_errors is not None:
            cleanup_errors.append(RuntimeError("buffer release failed"))
        raise command_error

    monkeypatch.setitem(sys.modules, "simpler.remote_l3_session", SimpleNamespace(_run_command_loop=loop))
    channel = Mock(shutdown_observed=False)
    monkeypatch.setattr(runtime_mod, "_ExecutorFrameSocket", lambda *_args: channel)
    worker = Mock(close=Mock(side_effect=RuntimeError("worker release failed") if worker_fails else None))
    records = []
    reporter = _RankCleanupReporter("127.0.0.1", 1, "token", MpiDirectCleanupReport("job", 1, 1, 0, "L3"))
    monkeypatch.setattr(reporter, "_send", records.append)

    with pytest.raises(RuntimeError) as raised:
        runtime_mod._run_executor(None, None, topology, 1, topology.executors[0], worker, reporter)
    assert raised.value is command_error
    worker.close.assert_called_once()
    channel.close.assert_called_once()
    assert len(records) == 1
    report = records[0]
    assert report.command == "not_observed"
    assert report.cleanup == ("failed" if buffer_fails or worker_fails else "succeeded")
    assert all("identity mismatch" not in failure for failure in report.failures)
    assert any("buffer release failed" in failure for failure in report.failures) == buffer_fails
    assert any("worker release failed" in failure for failure in report.failures) == worker_fails
