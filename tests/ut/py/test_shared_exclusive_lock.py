#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for `_SharedExclusiveLock`, the serializer behind run admission.

Control commands that belong to no run hold it shared — they must exclude
admission, not each other — while a submit holds it alone. These are pure
threading tests: no worker, no device.
"""

from __future__ import annotations

import threading

from simpler.worker import _SharedExclusiveLock

# Long enough that a genuinely blocked acquire is not mistaken for a slow one,
# short enough that a broken lock fails the suite in seconds rather than hanging.
BLOCKED_S = 0.3
ACQUIRED_S = 5.0


def _run(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_shared_holders_overlap() -> None:
    """Every shared holder is admitted at once — the point of the split."""
    lock = _SharedExclusiveLock()
    holders = 8
    inside = threading.Barrier(holders + 1)
    release = threading.Event()

    def hold() -> None:
        with lock.shared():
            inside.wait(timeout=ACQUIRED_S)
            release.wait(timeout=ACQUIRED_S)

    threads = [_run(hold) for _ in range(holders)]
    # Barrier trips only if all `holders` are inside their `with` simultaneously.
    inside.wait(timeout=ACQUIRED_S)
    release.set()
    for thread in threads:
        thread.join(timeout=ACQUIRED_S)
        assert not thread.is_alive()


def test_exclusive_excludes_shared() -> None:
    lock = _SharedExclusiveLock()
    held = threading.Event()
    entered_shared = threading.Event()
    release = threading.Event()

    def hold_exclusive() -> None:
        with lock.exclusive():
            held.set()
            release.wait(timeout=ACQUIRED_S)

    def take_shared() -> None:
        with lock.shared():
            entered_shared.set()

    writer = _run(hold_exclusive)
    assert held.wait(timeout=ACQUIRED_S)
    reader = _run(take_shared)
    assert not entered_shared.wait(timeout=BLOCKED_S), "shared entered while exclusive was held"

    release.set()
    assert entered_shared.wait(timeout=ACQUIRED_S), "shared never entered after exclusive released"
    for thread in (writer, reader):
        thread.join(timeout=ACQUIRED_S)
        assert not thread.is_alive()


def test_shared_excludes_exclusive() -> None:
    lock = _SharedExclusiveLock()
    held = threading.Event()
    entered_exclusive = threading.Event()
    release = threading.Event()

    def hold_shared() -> None:
        with lock.shared():
            held.set()
            release.wait(timeout=ACQUIRED_S)

    def take_exclusive() -> None:
        with lock.exclusive():
            entered_exclusive.set()

    reader = _run(hold_shared)
    assert held.wait(timeout=ACQUIRED_S)
    writer = _run(take_exclusive)
    assert not entered_exclusive.wait(timeout=BLOCKED_S), "exclusive entered while shared was held"

    release.set()
    assert entered_exclusive.wait(timeout=ACQUIRED_S), "exclusive never entered after shared released"
    for thread in (reader, writer):
        thread.join(timeout=ACQUIRED_S)
        assert not thread.is_alive()


def test_waiting_exclusive_blocks_new_shared() -> None:
    """Writer-preferring: a stream of control commands cannot starve a submit."""
    lock = _SharedExclusiveLock()
    first_held = threading.Event()
    entered_exclusive = threading.Event()
    entered_late_shared = threading.Event()
    release_first = threading.Event()

    def hold_shared() -> None:
        with lock.shared():
            first_held.set()
            release_first.wait(timeout=ACQUIRED_S)

    def take_exclusive() -> None:
        with lock.exclusive():
            entered_exclusive.set()

    def take_late_shared() -> None:
        with lock.shared():
            entered_late_shared.set()

    reader = _run(hold_shared)
    assert first_held.wait(timeout=ACQUIRED_S)
    writer = _run(take_exclusive)
    # Give the writer time to register itself as waiting before the late reader.
    assert not entered_exclusive.wait(timeout=BLOCKED_S)
    late_reader = _run(take_late_shared)
    assert not entered_late_shared.wait(timeout=BLOCKED_S), "shared jumped the queue ahead of a waiting exclusive"

    release_first.set()
    assert entered_exclusive.wait(timeout=ACQUIRED_S)
    assert entered_late_shared.wait(timeout=ACQUIRED_S)
    for thread in (reader, writer, late_reader):
        thread.join(timeout=ACQUIRED_S)
        assert not thread.is_alive()
