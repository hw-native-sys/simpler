# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Process-wide concurrency budget for scene-test compiler subprocesses."""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass

_MAX_AUTO_COMPILE_WORKERS = 8


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def default_compile_workers() -> int:
    """Reserve two logical CPUs for pytest/Python and cap compiler fanout at eight."""
    cpu_count = _available_cpu_count()
    return min(_MAX_AUTO_COMPILE_WORKERS, max(1, cpu_count - 2))


@dataclass
class _CompilePoolState:
    workers: int
    active: int = 0
    override_active: bool = False


_budget_guard = threading.Condition()
_state = _CompilePoolState(default_compile_workers())


def current_compile_workers() -> int:
    with _budget_guard:
        return _state.workers


@contextlib.contextmanager
def compile_worker_budget(max_workers: int):
    """Temporarily set the compiler budget; only one process-wide override may be active."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    with _budget_guard:
        if _state.override_active:
            raise RuntimeError("only one process-wide compile worker budget override may be active")
        previous = _state.workers
        _state.workers = max_workers
        _budget_guard.notify_all()
        _state.override_active = True
    try:
        yield
    finally:
        with _budget_guard:
            _state.workers = previous
            _budget_guard.notify_all()
            _state.override_active = False


@contextlib.contextmanager
def compile_slot():
    """Hold one process-wide compiler token."""
    with _budget_guard:
        _budget_guard.wait_for(lambda: _state.active < _state.workers)
        _state.active += 1
    try:
        yield
    finally:
        with _budget_guard:
            _state.active -= 1
            _budget_guard.notify_all()
