# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual socket notifications and virtual deadline boundaries, without a device."""

import json
import socket
import sys

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.dfx_joined_launch.notify import NotificationInbox

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux native-owner SOCK_SEQPACKET transport")


def test_notifications_for_other_generations_are_kept_for_their_owner():
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with receiver, sender:
        inbox = NotificationInbox(receiver)
        for generation in (9, 10):
            sender.send(json.dumps({"type": "entered", "generation": generation}).encode())
        assert inbox.wait("entered", 1, generation=10)["generation"] == 10
        assert inbox.wait("entered", 1, generation=9)["generation"] == 9


def test_missing_notification_expires_with_virtual_time_only():
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with receiver, sender:
        now = [100.0]

        def expire(readers, _writers, _errors, budget):
            assert readers == [receiver] and budget == 2.0
            now[0] += budget
            return [], [], []

        inbox = NotificationInbox(receiver, clock=lambda: now[0], poll=expire)
        with pytest.raises(TimeoutError, match="notification"):
            inbox.wait("entered", 2, generation=10)
        assert now[0] == 102.0


def test_closed_notification_channel_is_an_error():
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with receiver:
        sender.close()
        with pytest.raises(RuntimeError, match="closed"):
            NotificationInbox(receiver).wait("entered", 1)
