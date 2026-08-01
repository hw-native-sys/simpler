# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes

import pytest
from simpler.mpi_group_mailbox import (
    MailboxGroupState,
    MailboxOpcode,
    MailboxRequestState,
    MailboxTarget,
    MpiGroupError,
    MpiGroupMailbox,
    MpiRankError,
    _as_byte_view,
    open_rank_mailbox,
)


def test_shared_memory_view_is_normalized_to_unsigned_bytes():
    raw = (ctypes.c_ubyte * 4)()
    original_view = memoryview(raw)
    byte_view = _as_byte_view(original_view)

    byte_view[:] = b"test"

    assert byte_view is not original_view
    assert byte_view.format == "B"
    assert byte_view.ndim == 1
    assert bytes(raw) == b"test"


def test_named_mailbox_reopens_by_manifest_name_and_only_rank0_opens():
    owner = MpiGroupMailbox.create(world_size=2)
    reopened = None
    try:
        manifest = owner.manifest()
        assert open_rank_mailbox(manifest, rank=1) is None

        reopened = open_rank_mailbox(manifest, rank=0)
        assert reopened is not None
        assert reopened.name == owner.name
        assert reopened.world_size == 2
        assert reopened.group_state is MailboxGroupState.INITIALIZING

        reopened.publish_ready()
        assert owner.group_state is MailboxGroupState.READY
    finally:
        if reopened is not None:
            reopened.close()
        owner.close(unlink=True)


@pytest.mark.parametrize(
    ("target", "target_rank", "payloads"),
    [
        (MailboxTarget.GROUP, -1, (b"broadcast",)),
        (MailboxTarget.RANK, 1, (b"rank-1",)),
        (MailboxTarget.PER_RANK, -1, (b"rank-0", b"rank-1")),
    ],
)
def test_request_envelope_round_trips_all_target_types(target, target_rank, payloads):
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=1,
            opcode=MailboxOpcode.TASK,
            target=target,
            target_rank=target_rank,
            payloads=payloads,
        )

        request = mailbox.accept_request(last_sequence_id=0)
        assert request.sequence_id == 1
        assert request.opcode is MailboxOpcode.TASK
        assert request.target is target
        assert request.target_rank == target_rank
        assert request.payloads == payloads
        assert mailbox.request_state is MailboxRequestState.TASK_ACCEPTED

        mailbox.complete_request(sequence_id=1, payloads=(b"ok",))
        result = mailbox.read_result(sequence_id=1)
        assert result.payloads == (b"ok",)
        assert mailbox.request_state is MailboxRequestState.IDLE
    finally:
        mailbox.close(unlink=True)


def test_accept_copies_payload_before_publishing_task_accepted():
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=1,
            opcode=MailboxOpcode.TASK,
            target=MailboxTarget.RANK,
            target_rank=0,
            payloads=(b"immutable-copy",),
        )
        request = mailbox.accept_request(last_sequence_id=0)
        assert mailbox.request_state is MailboxRequestState.TASK_ACCEPTED

        mailbox.overwrite_request_payload_for_test(b"changed-after-accept")
        assert request.payloads == (b"immutable-copy",)
    finally:
        mailbox.close(unlink=True)


def test_duplicate_sequence_marks_group_terminal():
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=7,
            opcode=MailboxOpcode.PING,
            target=MailboxTarget.GROUP,
            target_rank=-1,
            payloads=(b"",),
        )
        with pytest.raises(MpiGroupError, match="sequence_id 7 is not newer than 7"):
            mailbox.accept_request(last_sequence_id=7)
        assert mailbox.group_state is MailboxGroupState.TERMINAL
    finally:
        mailbox.close(unlink=True)


def test_rank_failure_is_reported_with_rank_and_terminal_is_reusable_only_when_requested():
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=1,
            opcode=MailboxOpcode.TASK,
            target=MailboxTarget.GROUP,
            target_rank=-1,
            payloads=(b"task",),
        )
        mailbox.accept_request(last_sequence_id=0)
        mailbox.fail_request(
            sequence_id=1,
            errors=(MpiRankError(rank=1, error_type="ValueError", message="rank one failed"),),
            terminal=False,
        )
        with pytest.raises(MpiGroupError, match=r"rank 1.*ValueError.*rank one failed"):
            mailbox.read_result(sequence_id=1)
        assert mailbox.group_state is MailboxGroupState.READY

        mailbox.write_request(
            sequence_id=2,
            opcode=MailboxOpcode.PING,
            target=MailboxTarget.GROUP,
            target_rank=-1,
            payloads=(b"",),
        )
    finally:
        mailbox.close(unlink=True)


def test_terminal_group_rejects_new_requests():
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.mark_terminal("collective timed out")
        with pytest.raises(MpiGroupError, match="collective timed out"):
            mailbox.write_request(
                sequence_id=1,
                opcode=MailboxOpcode.PING,
                target=MailboxTarget.GROUP,
                target_rank=-1,
                payloads=(b"",),
            )
    finally:
        mailbox.close(unlink=True)


def test_rank0_failure_is_reported_and_marks_terminal():
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=1,
            opcode=MailboxOpcode.CONTROL,
            target=MailboxTarget.GROUP,
            target_rank=-1,
            payloads=(b"control",),
        )
        mailbox.accept_request(last_sequence_id=0)
        mailbox.fail_request(
            sequence_id=1,
            errors=(MpiRankError(rank=0, error_type="RuntimeError", message="rank zero exited"),),
            terminal=True,
        )
        with pytest.raises(MpiGroupError, match=r"rank 0.*rank zero exited"):
            mailbox.read_result(sequence_id=1)
        assert mailbox.group_state is MailboxGroupState.TERMINAL
    finally:
        mailbox.close(unlink=True)


def test_shutdown_has_distinct_ready_and_done_states():
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=1,
            opcode=MailboxOpcode.SHUTDOWN,
            target=MailboxTarget.GROUP,
            target_rank=-1,
            payloads=(b"shutdown-frame",),
        )
        assert mailbox.request_state is MailboxRequestState.SHUTDOWN_READY
        request = mailbox.accept_request(last_sequence_id=0)
        assert request.opcode is MailboxOpcode.SHUTDOWN
        assert mailbox.request_state is MailboxRequestState.TASK_ACCEPTED
        mailbox.complete_shutdown(sequence_id=1)
        assert mailbox.request_state is MailboxRequestState.SHUTDOWN_DONE
    finally:
        mailbox.close(unlink=True)
