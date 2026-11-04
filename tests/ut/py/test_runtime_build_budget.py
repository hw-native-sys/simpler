# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime CMake job sharing, including errors and nested executor contexts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import pytest

from simpler_setup.runtime_build_budget import runtime_build_budget, submit_runtime_build


def test_queued_targets_split_all_jobs_and_release_them_after_failure(monkeypatch):
    monkeypatch.setattr("simpler_setup.runtime_build_budget.multiprocessing.cpu_count", lambda: 64)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "8")
    with runtime_build_budget(3) as budget:
        with pytest.raises(RuntimeError, match="build failed"), ExitStack() as builds:
            jobs = [builds.enter_context(budget.jobs()) for _ in range(3)]
            assert sum(jobs) == 8
            assert min(jobs) >= 1
            raise RuntimeError("build failed")
        # With no targets queued, a final build can use the entire budget.
        with budget.jobs() as jobs:
            assert jobs == 8


def test_nested_executors_reuse_the_budget_and_restore_the_callers_context(monkeypatch):
    monkeypatch.setattr("simpler_setup.runtime_build_budget.multiprocessing.cpu_count", lambda: 64)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")

    def nested_build():
        with runtime_build_budget(1) as inherited:
            return inherited

    with runtime_build_budget(3) as parent, ThreadPoolExecutor(max_workers=1) as executor:
        assert submit_runtime_build(executor, nested_build).result(timeout=5) is parent
    with runtime_build_budget(1) as independent:
        assert independent is not parent
        with independent.jobs() as jobs:
            assert jobs == 2


def test_shared_budget_keeps_the_cpu_ceiling(monkeypatch):
    from simpler_setup import runtime_build_budget as budget_module  # noqa: PLC0415

    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "128")
    monkeypatch.setattr(budget_module.multiprocessing, "cpu_count", lambda: 4)
    with runtime_build_budget(1) as budget, budget.jobs() as jobs:
        assert jobs == 4
