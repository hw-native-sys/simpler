# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Share an explicit CMake job budget across one runtime build invocation."""

import contextlib
import contextvars
import multiprocessing
import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, TypeVar

_BuildResult = TypeVar("_BuildResult")


class _RuntimeBuildBudget:
    def __init__(self, build_count: int):
        self._max_jobs = min(multiprocessing.cpu_count(), 32)
        configured = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        self._limited = configured not in (None, "")
        if configured:
            try:
                requested = int(configured)
            except ValueError as exc:
                raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer") from exc
            if requested <= 0:
                raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer")
            self._max_jobs = min(self._max_jobs, requested)
        self._available = self._max_jobs
        self._pending = build_count
        self._condition = threading.Condition()

    @contextlib.contextmanager
    def jobs(self) -> Iterator[int]:
        if not self._limited:
            yield self._max_jobs
            return

        with self._condition:
            self._condition.wait_for(lambda: self._available > 0)
            # Leave a share for queued targets; as the queue drains, later
            # targets can use the released jobs instead of a fixed small share.
            jobs = max(1, self._available // max(1, self._pending))
            self._available -= jobs
            self._pending = max(0, self._pending - 1)
        try:
            yield jobs
        finally:
            with self._condition:
                self._available += jobs
                self._condition.notify_all()


_current_budget: contextvars.ContextVar[Optional[_RuntimeBuildBudget]] = contextvars.ContextVar(
    "runtime_build_budget", default=None
)


@contextlib.contextmanager
def runtime_build_budget(build_count: int) -> Iterator[_RuntimeBuildBudget]:
    """Reuse the caller's budget, or establish one for its expected CMake builds."""
    budget = _current_budget.get()
    if budget is not None:
        yield budget
        return
    budget = _RuntimeBuildBudget(build_count)
    token = _current_budget.set(budget)
    try:
        yield budget
    finally:
        _current_budget.reset(token)


def submit_runtime_build(
    executor: ThreadPoolExecutor, function: Callable[..., _BuildResult], *args, **kwargs
) -> Future[_BuildResult]:
    """Carry the shared budget into a runtime or target executor thread."""
    return executor.submit(contextvars.copy_context().run, function, *args, **kwargs)
