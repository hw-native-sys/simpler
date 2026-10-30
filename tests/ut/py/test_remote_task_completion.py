# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A node hook retires invocation state only after the real command loop drains."""

import struct
from types import SimpleNamespace

import pytest
from simpler import remote_l3_session
from simpler.remote_l3_protocol import Frame, FrameHeader, FrameType, decode_frame
from simpler.task_interface import CallConfig, TaskArgs


@pytest.mark.parametrize("failure", [None, "native", "callback"])
def test_task_completed_runs_after_drain_before_completion(monkeypatch, failure):
    events = []
    digest = bytes.fromhex("ab" * 32)
    frames = iter(
        [
            Frame(FrameHeader(FrameType.TASK, 1, 3, 1), digest),
            Frame(FrameHeader(FrameType.TASK, 1, 3, 2), digest),
            Frame(FrameHeader(FrameType.SHUTDOWN, 1, 3, 3), b""),
        ]
    )
    replies = []

    class Connection:
        def sendall(self, data):
            reply = decode_frame(data)
            replies.append(reply)
            if reply.header.frame_type == FrameType.COMPLETION:
                events.append("reply")

    class InnerWorker:
        def run(self, fn, args, config):
            events.append("entry")
            fn(None, args, config)
            if failure == "native":
                raise RuntimeError("native DAG failed")
            events.append("drained")

    def completed():
        events.append("retire")
        if failure == "callback":
            raise RuntimeError("retirement failed")

    monkeypatch.setattr(remote_l3_session, "read_frame", lambda conn: next(frames))
    monkeypatch.setattr(
        remote_l3_session,
        "decode_task_payload",
        lambda data: SimpleNamespace(
            callable_digest=digest,
            args=None,
            config=CallConfig(),
        ),
    )
    monkeypatch.setattr(remote_l3_session, "_materialize_task_args", lambda *a, **k: (TaskArgs(), []))
    monkeypatch.setattr(remote_l3_session, "_level_capture_prefix", lambda prefix, worker: prefix)
    remote_l3_session._run_command_loop(
        Connection(),
        {"session_id": 1, "worker_id": 3},
        InnerWorker(),
        manifest_dispatch_registry={digest: lambda *args: None},
        task_completed=completed,
    )
    task_replies = [reply for reply in replies if reply.header.frame_type == FrameType.COMPLETION]
    assert len(task_replies) == 2
    expected = ["entry", "reply"] if failure == "native" else ["entry", "drained", "retire", "reply"]
    assert events == expected * 2
    assert [struct.unpack_from("<Qi", reply.payload)[1] for reply in task_replies] == [
        int(failure is not None),
        int(failure is not None),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
