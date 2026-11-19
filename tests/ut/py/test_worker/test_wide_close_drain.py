# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``close()`` with an unwaited prefix wider than three runs: what it drains, and what survives it.

The scene suite's wide-close case establishes that a close with more than three unwaited runs
registered completes on a real route. It cannot establish what this does, because the fixture owns
the ``close()``: here the case owns it, so the state *after* it is readable — every handle's own
outcome, read repeatedly, and the registries the Worker is supposed to have emptied.

Sub-workers rather than a chip endpoint, because the subject is the orchestrator's run resources
and its teardown drain, neither of which is backend code. A run leaves the accepted set only in
``RunHandle.wait()``, on the thread that calls it, so a caller that waits nothing leaves exactly as
many runs outstanding as it submitted — the prefix is built by construction rather than by timing.
"""

import pytest
from simpler.worker import Worker

# Above the three every per-set owner used to be laid out for, so a close that drained a narrow
# prefix and left the rest behind fails rather than passing.
_WIDE_RUNS = 5


class SentinelError(RuntimeError):
    """Raised by a sub callable so one run of the prefix has a failure of its own."""


def _wide_worker(failing_index=None):
    """A Worker granted `_WIDE_RUNS` run-resource sets, with one callable per behaviour."""
    worker = Worker(level=3, num_sub_workers=1, pipeline_depth=_WIDE_RUNS)
    ok_handle = worker.register(lambda args: None)
    fail_handle = worker.register(_raise_sentinel) if failing_index is not None else None
    worker.init()
    return worker, ok_handle, fail_handle


def _raise_sentinel(args):
    del args
    raise SentinelError("this run's own failure")


def _submit_prefix(worker, ok_handle, fail_handle, failing_index):
    """`_WIDE_RUNS` runs, none of them waited, returned in submission order."""
    handles = []
    for index in range(_WIDE_RUNS):
        target = fail_handle if index == failing_index else ok_handle

        def graph(orch, _args, _cfg, target=target):
            orch.submit_sub(target)

        handles.append(worker.submit(graph))
    return handles


def _outstanding(worker):
    with worker._hierarchical_start_cv:  # noqa: SLF001 -- the lifecycle lock close publishes under
        return [handle for handle in worker._accepted_run_handles if not handle._cleanup_published]  # noqa: SLF001


def test_close_drains_a_wide_unwaited_prefix_and_leaves_no_run_registered():
    """Five runs, no wait, one close — and nothing of theirs is left on the Worker."""
    worker, ok_handle, _ = _wide_worker()
    handles = _submit_prefix(worker, ok_handle, None, failing_index=None)

    outstanding = _outstanding(worker)
    assert len(outstanding) > 3, (
        f"only {len(outstanding)} run(s) were outstanding at close, so this case's close would drain no more "
        f"than the three every per-set owner used to be laid out for"
    )
    assert len(outstanding) == _WIDE_RUNS

    worker.close()

    assert not worker._accepted_run_handles, (  # noqa: SLF001
        "close returned with runs still registered as accepted, so it did not drain the prefix it was left"
    )
    assert not worker._abandoned_run_handles, (  # noqa: SLF001
        "close abandoned a run of the prefix rather than draining it"
    )
    # Resource lifetime: after a close that drained everything, the Worker reports no
    # teardown-owned resource left at all — the native tree, the children and their mappings
    # included. A set still held by an undrained run keeps the native tree alive, so this is the
    # Worker's own answer to whether the prefix released what it borrowed.
    assert not worker._has_live_resources()  # noqa: SLF001
    # Every handle is terminal with its own outcome, and reads the same on every later look —
    # including these, which are all after the Worker is shut.
    for index, handle in enumerate(handles):
        assert handle.done, f"run {index} was not terminal after close"
        assert handle.result() is None, f"run {index} reported a failure it never had"
        assert handle.result() is None, f"run {index}'s second read did not return its first outcome"
        handle.wait()


def test_a_failure_in_a_wide_closed_prefix_stays_the_run_s_own():
    """One run of the prefix fails; close still drains the rest, and each outcome stays its own.

    The failing run is submitted last, so every other run of the prefix was admitted before any
    failure existed. That is deliberate: the subject here is what a *close* does with a prefix one
    of whose runs failed, not whether a failure blocks a later admission — which is row 31's
    question and has its own cases.
    """
    failing_index = _WIDE_RUNS - 1
    worker, ok_handle, fail_handle = _wide_worker(failing_index=failing_index)
    handles = _submit_prefix(worker, ok_handle, fail_handle, failing_index=failing_index)
    assert len(_outstanding(worker)) == _WIDE_RUNS

    # The failure is the run's, so it is also what the close reports: a drain that swallowed it
    # would leave the caller with no account of a run that did not succeed.
    with pytest.raises(RuntimeError):
        worker.close()

    assert not worker._accepted_run_handles, (  # noqa: SLF001
        "a failure inside the prefix left the other runs registered after close"
    )
    for index, handle in enumerate(handles):
        assert handle.done, f"run {index} was not terminal after close"
        if index == failing_index:
            with pytest.raises(RuntimeError) as first:
                handle.result()
            with pytest.raises(RuntimeError) as second:
                handle.result()
            assert first.value is second.value, "a failed run returned a different error object on a later read"
            continue
        assert handle.result() is None, f"run {index} took another run's failure as its own outcome"
