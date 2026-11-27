# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind an incoming parent task to the one inner run that it submitted."""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time

# Retry an interrupted fan-in snapshot once before quarantining the worker.
RUN_CANCELLATION_ATTEMPTS = 2

_current_parent: contextvars.ContextVar[tuple[object, ParentCancellation] | None] = contextvars.ContextVar(
    "simpler_parent_run", default=None
)


def bind_parent_run(worker, orch, run_id):
    parent = _current_parent.get()
    if parent is not None and parent[0] is worker:
        parent[1].bind(orch, run_id)


class ParentTaskCancelled(RuntimeError):
    def __init__(self):
        super().__init__("parent dispatch failed before task entry (peer error 108)")


class ParentCancellation:
    def __init__(self):
        self._lock = threading.Lock()
        self._requested = False
        self._binding = None

    def request(self):
        with self._lock:
            self._requested = True
            binding = self._binding
        if binding is not None:
            orch, run_id = binding
            for attempt in range(RUN_CANCELLATION_ATTEMPTS):
                try:
                    orch._request_parent_cancel(run_id)
                    return
                except MemoryError:
                    # The native sweep retains all unreleased references for a retry.
                    with self._lock:
                        if self._binding is not binding:
                            return
                    if attempt == 0:
                        time.sleep(0.01)
            orch._abandon_parent_cancel(run_id)

    def bind(self, orch, run_id):
        with self._lock:
            if self._requested:
                raise ParentTaskCancelled
            self._binding = (orch, run_id)

    @contextlib.contextmanager
    def scope(self, worker):
        token = _current_parent.set((worker, self))
        try:
            yield
        finally:
            _current_parent.reset(token)
            with self._lock:
                self._binding = None


class SessionCancellation:
    """One ordered command in flight. The independent cancel lane may arrive first."""

    def __init__(self):
        self._lock = threading.Lock()
        self._retired = 0
        self._pending = 0
        self._active = None

    def accept_frame(self, frame, session_id, worker_id):
        from .remote_l3_protocol import FrameType  # noqa: PLC0415

        header = frame.header
        if (
            header.frame_type != FrameType.CANCEL_TASK
            or header.session_id != session_id
            or header.worker_id != worker_id
            or header.sequence <= 0
            or frame.payload
        ):
            raise ValueError("invalid task cancellation on health lane")
        self.request(header.sequence)

    def request(self, sequence):
        with self._lock:
            if sequence <= self._retired:
                return
            active = self._active
            if active is None or sequence != active[0]:
                self._pending = max(sequence, self._pending)
                return
        active[1].request()

    @contextlib.contextmanager
    def task(self, sequence, worker):
        cancel = ParentCancellation()
        with self._lock:
            self._active = (sequence, cancel)
            pending = self._pending == sequence
        try:
            if pending:
                raise ParentTaskCancelled
            with cancel.scope(worker):
                yield
        finally:
            with self._lock:
                self._retired = sequence
                self._active = None
                if self._pending <= sequence:
                    self._pending = 0


class MailboxCancellation:
    """One watcher per nested worker. The parent owns the entry again after task returns."""

    def __init__(self, load_requested):
        self._load_requested = load_requested
        self._cv = threading.Condition()
        self._active = None
        self._closed = False
        self._watcher = threading.Thread(target=self._watch, name="simpler-parent-cancel", daemon=True)
        self._watcher.start()

    @contextlib.contextmanager
    def task(self, addr, worker):
        cancel = ParentCancellation()
        with self._cv:
            self._active = (addr, cancel)
            self._cv.notify()
        try:
            with cancel.scope(worker):
                yield
        finally:
            with self._cv:
                self._active = None
                self._cv.notify()

    def _watch(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._active is not None or self._closed)
                if self._closed:
                    return
                assert self._active is not None
                addr, cancel = self._active
                if not self._load_requested(addr):
                    self._cv.wait(0.01)
                    continue
                self._active = None
            cancel.request()

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify()
        self._watcher.join()
