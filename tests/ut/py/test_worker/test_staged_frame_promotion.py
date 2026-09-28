#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The chip child's publication of a staged run promoted to native preparation.

A run staged behind a predecessor that has not launched yet stages *validated-only* and the lane
prepares it later, when that predecessor reaches the device. The child publishes that promotion so
the parent can authorize the early launch the run has earned.

The word it may write is the point. The state word is the parent's handoff — the parent takes a
staged frame by compare-exchanging ``FRAME_STAGED`` to ``ACTIVATE`` — and the child runs in
another process, so reading that word and storing it back cannot exclude the exchange. These
cases drive the real publisher with the parent's command already written, which is the
interleaving a check-then-store loses.
"""

import ctypes
import struct

import pytest
from simpler.worker import (
    _ACTIVATE,
    _FRAME_STAGED,
    _NATIVE_PREPARED,
    _OFF_FRAME_DISPATCH_ID,
    _OFF_FRAME_GENERATION,
    _OFF_FRAME_GROUP_INDEX,
    _OFF_FRAME_GROUP_SIZE,
    _OFF_FRAME_PROTOCOL,
    _OFF_FRAME_RUN_ID,
    _OFF_FRAME_SLOT_ID,
    _OFF_FRAME_TASK_SLOT,
    _OFF_PREPARATION_DISPOSITION,
    _OFF_STATE,
    _TASK_PROTOCOL_VERSION,
    _VALIDATED_ONLY,
    _buffer_field_addr,
    _mailbox_load_i32,
    _mailbox_store_i32,
    _publish_native_promotion,
    _read_task_frame_identity,
)
from simpler.worker import MAILBOX_FRAME_SIZE as _FRAME_SIZE

_IDENTITY = (_TASK_PROTOCOL_VERSION, 11, 2, 3, 40, 0, 0, 1)
_OTHER_IDENTITY = (_TASK_PROTOCOL_VERSION, 12, 2, 4, 41, 0, 0, 1)

_IDENTITY_OFFSETS = (
    _OFF_FRAME_PROTOCOL,
    _OFF_FRAME_RUN_ID,
    _OFF_FRAME_SLOT_ID,
    _OFF_FRAME_GENERATION,
    _OFF_FRAME_DISPATCH_ID,
    _OFF_FRAME_TASK_SLOT,
    _OFF_FRAME_GROUP_INDEX,
    _OFF_FRAME_GROUP_SIZE,
)


class _Frame:
    """One task frame in ordinary memory, at the offsets both sides of the mailbox use."""

    def __init__(self, identity, state, disposition):
        self._storage = (ctypes.c_char * _FRAME_SIZE)()
        self.buf = memoryview(self._storage).cast("B")
        self.addr = _buffer_field_addr(self.buf, 0)
        self.write_identity(identity)
        self.state = state
        self.disposition = disposition

    def write_identity(self, identity):
        for offset, value in zip(_IDENTITY_OFFSETS, identity):
            struct.pack_into("=Q", self.buf, offset, value)

    @property
    def state(self):
        return _mailbox_load_i32(self.addr + _OFF_STATE)

    @state.setter
    def state(self, value):
        _mailbox_store_i32(self.addr + _OFF_STATE, value)

    @property
    def disposition(self):
        return _mailbox_load_i32(self.addr + _OFF_PREPARATION_DISPOSITION)

    @disposition.setter
    def disposition(self, value):
        _mailbox_store_i32(self.addr + _OFF_PREPARATION_DISPOSITION, value)


@pytest.fixture
def staged_frame():
    return _Frame(_IDENTITY, _FRAME_STAGED, _VALIDATED_ONLY)


def test_the_frame_models_the_wire_layout(staged_frame):
    # A positive control for every case below: if the identity were written somewhere the reader
    # does not look, the identity guard would appear to hold for the wrong reason.
    assert _read_task_frame_identity(staged_frame.buf) == _IDENTITY
    assert staged_frame.state == _FRAME_STAGED
    assert staged_frame.disposition == _VALIDATED_ONLY


def test_a_promotion_publishes_the_disposition_and_leaves_the_frame_staged(staged_frame):
    assert _publish_native_promotion(staged_frame.addr, staged_frame.buf, _IDENTITY, _NATIVE_PREPARED) is True
    assert staged_frame.disposition == _NATIVE_PREPARED
    # Written by nobody here: the endpoint reads the disposition on every staged poll, so the
    # promotion needs no state-word store to be seen.
    assert staged_frame.state == _FRAME_STAGED


def test_a_promotion_racing_the_parents_activation_does_not_take_the_frame_back(staged_frame):
    """The interleaving a check-then-store loses.

    The parent's exchange lands between the child deciding the frame is staged and the child
    writing. The parent records the activation as published and will not publish it again, so a
    state-word store here would strand the run: the child would never see ACTIVATE.
    """
    staged_frame.state = _ACTIVATE

    assert _publish_native_promotion(staged_frame.addr, staged_frame.buf, _IDENTITY, _NATIVE_PREPARED) is True

    assert staged_frame.state == _ACTIVATE
    assert staged_frame.disposition == _NATIVE_PREPARED


def test_a_promotion_for_a_frame_that_moved_to_another_run_is_refused(staged_frame):
    staged_frame.write_identity(_OTHER_IDENTITY)

    assert _publish_native_promotion(staged_frame.addr, staged_frame.buf, _IDENTITY, _NATIVE_PREPARED) is False

    assert staged_frame.disposition == _VALIDATED_ONLY


@pytest.mark.parametrize("disposition", [_VALIDATED_ONLY, 0])
def test_only_a_native_preparation_is_published(staged_frame, disposition):
    # The reverse direction is a stale read, and nothing else is a disposition at all.
    assert _publish_native_promotion(staged_frame.addr, staged_frame.buf, _IDENTITY, disposition) is False
    assert staged_frame.disposition == _VALIDATED_ONLY
