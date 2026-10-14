#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""How a child hands a request frame back, and what it may touch while doing so.

Transport and execution lifetime are separate pools: a request frame carries one submission's
bytes, and the run-state entry its header names carries the run. The frame goes back as soon as
the payload is copied out, which is what lets the parent refill it while the run it carried is
still executing — and is also what makes the hand-back's *store order* load-bearing.

Every case here drives the real child loop against a real shared buffer and asserts on the order
of the stores it made, not on the final state. A final state cannot distinguish these:

  acknowledged last    The parent treats the consumption ACK alone as permission to refill the
                       frame. A store this side makes after it lands in a payload that is already
                       published, so an ``_IDLE`` arriving late erases a ``_TASK_READY`` and the
                       task it carried is never served. Reading "frame idle, ACK written" at the
                       end is true of both orders; only the sequence separates them.
  attributed correctly A header this side cannot attribute to a live occupant names no run it may
                       report against. Writing an ACK or an error through it would charge this
                       frame's problem to whichever run holds that entry now.

The two kinds of unattributable header are deliberately not treated alike, because who can
resolve them differs — see ``_RequestHeaderInvalid`` in ``simpler.worker``.
"""

import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import pytest
import simpler.worker as worker_mod
from simpler.task_interface import (
    MAILBOX_RUN_ENTRY_STRIDE,
    MAILBOX_RUN_ENTRY_TABLE_OFFSET,
    mailbox_size,
)
from simpler.worker import (
    _IDLE,
    _OFF_STATE,
    MAILBOX_FRAME_SIZE,
    _mailbox_addr,
    _mailbox_load_i32,
    _mailbox_load_u64,
    _mailbox_store_i32,
    _mailbox_store_u64,
)

#: One granted set, so the entry table is `M = granted + K` entries wide.
_GRANT = 1
_LOOP_BUDGET_S = 5.0


def _entry_offset(index: int) -> int:
    return MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * MAILBOX_RUN_ENTRY_STRIDE


class _ChildLoopFixture:
    """One mailbox, one child loop in a thread, and a record of every store it made.

    The recorder wraps the two publication primitives the whole protocol goes through, so the
    sequence it collects is the child's complete write history for this mailbox — the thing these
    cases are about. Both wrappers forward to the real store, so the loop under test is the
    production one and the shared state it leaves behind is the production state.
    """

    def __init__(self, monkeypatch, *, task_result=(0, "")):
        self.shm = SharedMemory(create=True, size=mailbox_size(_GRANT))
        buf = self.shm.buf
        assert buf is not None
        self.buf: memoryview = buf
        self.base = _mailbox_addr(self.shm)
        self.state_addr = self.base + _OFF_STATE
        self.frame_addr = self.base + MAILBOX_FRAME_SIZE
        self.entry_count = _GRANT + worker_mod.MAILBOX_REQUEST_FRAMES
        self.stores: list[tuple[str, int, int]] = []
        self.task_calls = 0
        self._task_result = task_result
        self._lock = threading.Lock()

        real_i32 = worker_mod._mailbox_store_i32
        real_u64 = worker_mod._mailbox_store_u64

        def record_i32(addr, value):
            with self._lock:
                self.stores.append(("i32", int(addr), int(value)))
            real_i32(addr, value)

        def record_u64(addr, value):
            with self._lock:
                self.stores.append(("u64", int(addr), int(value)))
            real_u64(addr, value)

        monkeypatch.setattr(worker_mod, "_mailbox_store_i32", record_i32)
        monkeypatch.setattr(worker_mod, "_mailbox_store_u64", record_u64)
        _mailbox_store_i32(self.state_addr, _IDLE)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _handle_task(self, _frame_buf, _entry_buf):
        self.task_calls += 1
        return self._task_result

    def _run(self):
        worker_mod._run_mailbox_loop(
            self.buf,
            self.state_addr,
            handle_task=self._handle_task,
            handle_control=lambda _sub_cmd: (0, ""),
        )

    # -- publication, exactly as the parent endpoint does it -------------------

    def publish_entry(self, index: int, *, entry_generation: int, dispatch_id: int = 1) -> int:
        """Write one run's identity into its entry and make the entry occupied."""
        offset = _entry_offset(index)
        addr = self.base + offset
        entry = self.buf[offset : offset + MAILBOX_RUN_ENTRY_STRIDE]
        try:
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_PROTOCOL, worker_mod._TASK_PROTOCOL_VERSION)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_RUN_ID, 5)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_SLOT_ID, index)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_LEASE_GENERATION, 11)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_DISPATCH_ID, dispatch_id)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_TASK_SLOT, 3)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_GROUP_INDEX, 0)
            struct.pack_into("=Q", entry, worker_mod._OFF_ENTRY_GROUP_SIZE, 1)
        finally:
            entry.release()
        _mailbox_store_i32(addr + worker_mod._OFF_ENTRY_ACCEPTED, 0)
        _mailbox_store_u64(addr + worker_mod._OFF_ENTRY_FRAME_CONSUMED_GENERATION, 0)
        _mailbox_store_i32(addr + worker_mod._OFF_ENTRY_STATE, _IDLE)
        _mailbox_store_u64(addr + worker_mod._OFF_ENTRY_GENERATION, entry_generation)
        return addr

    def publish_frame(self, *, entry_index: int, entry_generation: int, frame_generation: int) -> None:
        """Write the payload header that binds a frame to an entry, state word last."""
        frame = self.buf[MAILBOX_FRAME_SIZE : 2 * MAILBOX_FRAME_SIZE]
        try:
            frame[worker_mod._OFF_TASK_CALLABLE_HASH : worker_mod._OFF_TASK_ARGS_BLOB] = (
                bytes([0x42]) * worker_mod.CALLABLE_HASH_DIGEST_BYTES
            )
            struct.pack_into("=ii", frame, worker_mod._OFF_TASK_ARGS_BLOB, 0, 0)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_ENTRY_INDEX, entry_index)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_ENTRY_GENERATION, entry_generation)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_GENERATION, frame_generation)
        finally:
            frame.release()
        _mailbox_store_i32(self.frame_addr + _OFF_STATE, worker_mod._TASK_READY)

    # -- observation ----------------------------------------------------------

    def start(self):
        self.thread.start()

    def wait_for_ack(self, entry_addr: int, frame_generation: int):
        """Wait for the consumption acknowledgement, which is the end of the hand-back.

        Never for the frame's own `_IDLE`: that is the *first* of the two stores a correct
        hand-back makes, so a case waiting on it and then asserting the ACK exists can be
        descheduled in between and report the right order as a missing store.
        """
        addr = entry_addr + worker_mod._OFF_ENTRY_FRAME_CONSUMED_GENERATION
        deadline = time.monotonic() + _LOOP_BUDGET_S
        while time.monotonic() < deadline:
            if _mailbox_load_u64(addr) == frame_generation:
                return
            time.sleep(0.001)
        raise AssertionError("the child never acknowledged the request frame")

    def wait_for_entry_state(self, entry_addr: int, state: int):
        deadline = time.monotonic() + _LOOP_BUDGET_S
        while time.monotonic() < deadline:
            if _mailbox_load_i32(entry_addr + worker_mod._OFF_ENTRY_STATE) == state:
                return
            time.sleep(0.001)
        raise AssertionError(f"the child never published entry state {state}")

    def wait_for_abandoned_frame(self):
        """Wait for a frame returned without an ACK, which is the whole of that hand-back.

        The only case where `_IDLE` is the last store: a header this side may not acknowledge
        leaves nothing else to publish, so waiting on it is waiting on the end of the sequence
        rather than on its first half.
        """
        deadline = time.monotonic() + _LOOP_BUDGET_S
        while time.monotonic() < deadline:
            if _mailbox_load_i32(self.frame_addr + _OFF_STATE) == _IDLE:
                return
            time.sleep(0.001)
        raise AssertionError("the child never returned the request frame")

    def wait_for_thread_exit(self):
        self.thread.join(_LOOP_BUDGET_S)
        assert not self.thread.is_alive(), f"the child loop did not return within {_LOOP_BUDGET_S}s"

    def store_index(self, addr: int, value: Optional[int] = None) -> Optional[int]:
        """Where a store to `addr` (optionally with `value`) first appears in the sequence."""
        with self._lock:
            for position, (_kind, stored_addr, stored_value) in enumerate(self.stores):
                if stored_addr == addr and (value is None or stored_value == value):
                    return position
        return None

    def stores_within(self, start: int, length: int) -> list[int]:
        """The positions of every store landing inside ``[start, start + length)``."""
        with self._lock:
            return [
                position for position, (_kind, addr, _value) in enumerate(self.stores) if start <= addr < start + length
            ]

    def close(self):
        _mailbox_store_i32(self.state_addr, worker_mod._SHUTDOWN)
        self.thread.join(_LOOP_BUDGET_S)
        if self.thread.is_alive():
            # The loop holds raw addresses into this mapping, so unmapping it under a surviving
            # thread is a use-after-free in that thread rather than a leak. The segment is left
            # mapped and the timeout is surfaced: a child that will not stop is this case's
            # finding, and hiding it behind a crash in the next one helps nobody.
            raise AssertionError(
                f"the child loop did not stop within {_LOOP_BUDGET_S}s, so the mailbox is still in use"
            )
        # The shared buffer's own view is an export of the segment, so it is released before the
        # close that would otherwise raise BufferError and replace this case's finding.
        self.buf.release()
        self.shm.close()
        self.shm.unlink()


class TestTheConsumptionAcknowledgement:
    def test_the_ack_is_the_childs_last_store_to_a_request_frame(self, monkeypatch):
        """``_IDLE`` into the frame, then the ACK — never the other way round.

        The parent's reclamation keys on the ACK alone, so from the instant it lands the frame may
        already hold the next submission. An ``_IDLE`` store sequenced after it therefore does not
        tidy up this frame; it erases whatever the parent published into it, and the task that
        payload carried is never served and its run never terminates.

        Asserted as an ordering over the loop's own stores because the end state cannot tell the
        two orders apart: both finish with the frame idle and the ACK written.
        """
        fixture = _ChildLoopFixture(monkeypatch)
        try:
            entry_addr = fixture.publish_entry(0, entry_generation=3)
            fixture.publish_frame(entry_index=0, entry_generation=3, frame_generation=7)
            fixture.start()
            fixture.wait_for_ack(entry_addr, 7)

            ack_at = fixture.store_index(entry_addr + worker_mod._OFF_ENTRY_FRAME_CONSUMED_GENERATION, 7)
            idle_at = fixture.store_index(fixture.frame_addr + _OFF_STATE, _IDLE)
            assert ack_at is not None, "the child never acknowledged the payload"
            assert idle_at is not None, "the child never released the frame's own state word"
            assert idle_at < ack_at, (
                f"the frame was released at store {idle_at} and acknowledged at {ack_at}: the ACK must be the "
                f"last store this side makes to that frame, or it lands in the parent's next payload"
            )
            later = [
                position
                for position in fixture.stores_within(fixture.frame_addr, MAILBOX_FRAME_SIZE)
                if position > ack_at
            ]
            assert not later, f"the child stored into the frame at {later} after acknowledging it"
        finally:
            fixture.close()

    def test_the_terminal_state_is_published_after_the_frame_is_returned(self, monkeypatch):
        """The run's terminal comes last, and after every frame access.

        From the instant the parent observes a terminal it owns that entry alone and may release
        and refill it — so the ACK, which is written *through* the entry, has to precede it. The
        two stores are to different objects and nothing but this order keeps the second from
        landing on a successor's identity.
        """
        fixture = _ChildLoopFixture(monkeypatch)
        try:
            entry_addr = fixture.publish_entry(0, entry_generation=3)
            fixture.publish_frame(entry_index=0, entry_generation=3, frame_generation=7)
            fixture.start()
            fixture.wait_for_entry_state(entry_addr, worker_mod._TASK_DONE)

            ack_at = fixture.store_index(entry_addr + worker_mod._OFF_ENTRY_FRAME_CONSUMED_GENERATION, 7)
            terminal_at = fixture.store_index(entry_addr + worker_mod._OFF_ENTRY_STATE, worker_mod._TASK_DONE)
            assert ack_at is not None and terminal_at is not None
            assert ack_at < terminal_at, (
                f"the terminal state was published at store {terminal_at} before the frame was acknowledged at "
                f"{ack_at}, so the parent could release the entry the ACK is written through"
            )
        finally:
            fixture.close()


class TestAnUnattributableHeader:
    def test_a_stale_header_leaves_the_live_occupant_untouched(self, monkeypatch):
        """A frame naming a generation the entry has moved past writes nothing into it.

        This is a legal race, not corruption: the parent releases an entry and reclaims the frame
        from its own record in the same step, so a frame still carrying the old generation can be
        seen by the child afterwards. The entry now belongs to another run, and an ACK or an error
        written through it would be charged to that run — which is why the child returns the frame
        and touches the entry not at all. Nothing waits on the missing ACK: the parent's own record
        already released it.
        """
        fixture = _ChildLoopFixture(monkeypatch)
        try:
            # The entry's current occupant, two generations on from what the frame names.
            entry_addr = fixture.publish_entry(0, entry_generation=7, dispatch_id=9)
            before = len(fixture.stores)
            fixture.publish_frame(entry_index=0, entry_generation=3, frame_generation=7)
            fixture.start()
            fixture.wait_for_abandoned_frame()
            # Give the loop room to make a wrong store before concluding it made none.
            time.sleep(0.05)

            touched = [
                position
                for position in fixture.stores_within(entry_addr, MAILBOX_RUN_ENTRY_STRIDE)
                if position >= before
            ]
            assert not touched, f"the child stored into the live occupant's entry at {touched}"
            assert fixture.task_calls == 0, "the child ran a task from a frame it could not attribute"
            assert _mailbox_load_u64(entry_addr + worker_mod._OFF_ENTRY_GENERATION) == 7
            assert _mailbox_load_i32(entry_addr + worker_mod._OFF_ENTRY_STATE) == _IDLE
            assert fixture.thread.is_alive(), "a stale frame is an ordinary race and must not stop the endpoint"
        finally:
            fixture.close()

    @pytest.mark.parametrize(
        ("entry_index", "entry_generation", "why"),
        [
            # No publication can name an entry outside the table this endpoint was built for, and
            # the index is used to compute a raw address — so a store through it would land past
            # the shared region entirely.
            (1 << 20, 3, "an entry index outside the table"),
            # Publication always makes the generation odd; even means released.
            (0, 4, "an even generation no publication writes"),
        ],
    )
    def test_an_invalid_header_stops_the_endpoint_without_touching_an_entry(
        self, monkeypatch, entry_index, entry_generation, why
    ):
        """A header no publication could have written is answered endpoint-wide, not per run.

        The child cannot name the run such a frame belongs to, so it has no entry it may publish
        into and no successor it may charge; and the parent, holding a record that still looks
        live, has no reclamation of its own to make from it. Neither side can resolve it locally.
        The bounded outcome is this endpoint's own stop, which is what gives every run it still
        holds a terminal under one reason — and the reason itself is published, so the failure is
        attributed rather than appearing as a child that merely went away.
        """
        del why
        fixture = _ChildLoopFixture(monkeypatch)
        try:
            entry_addr = fixture.publish_entry(0, entry_generation=3)
            before = len(fixture.stores)
            fixture.publish_frame(entry_index=entry_index, entry_generation=entry_generation, frame_generation=7)
            fixture.start()
            fixture.wait_for_thread_exit()

            shutdown_addr = worker_mod._buffer_field_addr(fixture.buf, worker_mod._OFF_SHUTDOWN)
            assert _mailbox_load_i32(shutdown_addr) == worker_mod._SHUTDOWN_REQUESTED, (
                "the child did not request its own stop, so the parent is left waiting on a record it cannot "
                "reclaim and on a run nothing will terminate"
            )
            raw = bytes(
                fixture.buf[
                    worker_mod.MAILBOX_OFF_ERROR_MSG : worker_mod.MAILBOX_OFF_ERROR_MSG
                    + worker_mod.MAILBOX_ERROR_MSG_SIZE
                ]
            )
            message = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
            assert "request frame" in message, f"the stop carried no reason naming the header: {message!r}"

            touched = [
                position
                for position in fixture.stores_within(entry_addr, MAILBOX_RUN_ENTRY_STRIDE)
                if position >= before
            ]
            assert not touched, f"the child stored into entry 0 at {touched} for a frame it could not attribute"
            assert fixture.task_calls == 0, "the child ran a task from a frame it could not attribute"
            assert _mailbox_load_i32(fixture.frame_addr + _OFF_STATE) == _IDLE, (
                "the frame was left claimed, so the child would re-read the same invalid header"
            )
        finally:
            fixture.close()
