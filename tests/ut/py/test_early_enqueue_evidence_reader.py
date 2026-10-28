# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What the early-enqueue suites' joined-launch evidence reader accepts as proof of an ordering.

Both `tensormap_and_ringbuffer` early-enqueue suites decide their whole-operator ordering claim
from records their chip children write to per-pid host-log files. Two properties of that reader are
not observable from a passing hardware run, because a hardware run that happens to be fast enough
satisfies them by accident:

delayed   a child writes the joined-launch record when a successor's native submission completes
          and a run's boundary markers when that run retires. They are separate records at
          separate times, so a wait satisfied by the first can be read before the second exists —
          and the drain loop is the only thing that advances the cursor, so a row unread by then is
          absent for good.
scoped    a run key is ``(dispatch id, pipeline slot)`` and each chip child allocates both for
          itself, so two endpoints of one Worker both name ``(1, 0)``. A cursor opened for the
          second endpoint still sees whatever the first child writes late.

The suites' own functions are loaded and exercised here, per arch, rather than reimplemented: a
copy of the rule would pass while the rule itself regressed. Nothing here touches a device — the
cursor is planted, so the reader is driven over records that no run produced.
"""

import ast
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

#: The suites carrying this reader. Both are loaded, because they hold separate copies of it and a
#: repair applied to one would otherwise go missing from the other.
_SUITES = {
    "a2a3_tmr": "st/a2a3/tensormap_and_ringbuffer/early_enqueue/test_early_enqueue.py",
    "a5_tmr": "st/a5/tensormap_and_ringbuffer/early_enqueue/test_early_enqueue.py",
}

_PREDECESSOR = (1, 0)
_SUCCESSOR = (2, 1)
_PAIR = (_PREDECESSOR, _SUCCESSOR)


def _load_suite(name, relative_path, monkeypatch):
    """One early-enqueue scene module, loaded by path.

    By path rather than by import, so this does not depend on the scene tree being an importable
    package, and under a private module name so the ``SceneTestCase`` classes it defines are not
    reachable for collection from this lane.
    """
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(f"_evidence_reader_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=sorted(_SUITES), ids=sorted(_SUITES))
def suite(request, monkeypatch):
    return _load_suite(request.param, _SUITES[request.param], monkeypatch)


@pytest.fixture
def hbg_suite(monkeypatch):
    return _load_suite("a5_hbg", "st/a5/host_build_graph/early_enqueue/test_early_enqueue.py", monkeypatch)


def _planted_trace(suite, pids, identities, boundaries):
    """A suite's own ``_RunTrace`` with its parsed state planted and no files behind it.

    Built through ``__new__`` so the real ``boundaries_of_run`` lookup is what the checks below
    exercise. Each child is given a path that does not exist, which is a state the cursor already
    tolerates, so reading it yields nothing rather than raising.
    """
    trace = object.__new__(suite._RunTrace)
    trace._pids = list(pids)
    trace._paths = {pid: Path(f"/nonexistent/host.{pid}.log") for pid in pids}
    trace._offsets = dict.fromkeys(pids, 0)
    trace._origins = dict.fromkeys(pids, 0)
    trace.run_of_invocation = dict(identities)
    trace.boundary_of_invocation = dict(boundaries)
    return trace


def _planted_case_runs(suite, keys):
    """A suite's own ``_CaseRuns`` holding exactly these keys, with no mailbox behind it."""
    case_runs = object.__new__(suite._CaseRuns)
    case_runs.floor = 0
    case_runs._keys = set(keys)
    return case_runs


def _planted_boundary(device, aicore_start, whole_operator_end):
    """One boundary row in the shape ``_RunTrace._note_boundaries`` produces.

    A position the emitter could not take carries a non-zero rc and no time, which is why an
    absent reading is ``None`` here rather than zero.
    """
    return {
        "device": device,
        "aicore_start": aicore_start,
        "aicore_rc": 0 if aicore_start is not None else 1,
        "whole_operator_end": whole_operator_end,
        "whole_operator_rc": 0 if whole_operator_end is not None else 1,
        "hz": 1_000_000,
    }


def _planted_join(pid, predecessor, successor):
    """One joined-launch record in the shape ``_join_record`` produces."""
    return {
        "pid": pid,
        "inv": 0,
        "predecessor": predecessor,
        "successor": successor,
        "observed": True,
        "unfired": True,
        "query_rc": 0,
    }


def test_a_late_successor_boundary_holds_the_evidence_wait_open(suite):
    """The join record is not evidence that the readings the ordering claim needs have arrived."""
    pid = 4242
    records = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR)]
    case_runs = _planted_case_runs(suite, [_PREDECESSOR, _SUCCESSOR])
    identities = {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR}

    half = _planted_trace(suite, [pid], identities, {(pid, 1): _planted_boundary(6, 100, 200)})
    assert suite._established_pairs(records, case_runs, pid) == {_PAIR}, (
        "the record does establish the pair, so what follows is about the readings it does not carry"
    )
    assert not suite._boundaries_settled(half, records, case_runs, pid), (
        "the wait was satisfied while the successor's boundary row was still missing, which is the "
        "premature read this predicate exists to prevent"
    )
    assert suite._whole_operator_order(half, records, case_runs, pid) == (set(), set(), set(), {_PAIR}), (
        "a pair with one half of its readings must be unmeasured, never ordered"
    )

    whole = _planted_trace(
        suite,
        [pid],
        identities,
        {(pid, 1): _planted_boundary(6, 100, 200), (pid, 2): _planted_boundary(6, 300, 400)},
    )
    assert suite._boundaries_settled(whole, records, case_runs, pid), (
        "with both rows present the wait must stop, or a passing case would always burn its budget"
    )
    assert suite._whole_operator_order(whole, records, case_runs, pid) == ({_PAIR}, set(), set(), set())


def test_a_successor_released_too_early_is_still_overlapping(suite):
    """Waiting for the readings does not soften what they then have to say.

    The predicate holds out for a comparable pair, so the one case that must never be reported as
    an absence is a pair whose readings are present and wrong.
    """
    pid = 4242
    records = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR)]
    case_runs = _planted_case_runs(suite, [_PREDECESSOR, _SUCCESSOR])
    trace = _planted_trace(
        suite,
        [pid],
        {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR},
        {(pid, 1): _planted_boundary(6, 100, 400), (pid, 2): _planted_boundary(6, 300, 500)},
    )
    assert suite._boundaries_settled(trace, records, case_runs, pid)
    ordered, overlapping, within_tick, unmeasured = suite._whole_operator_order(trace, records, case_runs, pid)
    assert (ordered, overlapping, within_tick, unmeasured) == (set(), {_PAIR}, set(), set()), (
        "a successor released before its predecessor's whole operator ended must stay an overlap"
    )


def test_another_endpoints_record_is_not_this_endpoints_evidence(suite):
    """Two endpoints of one Worker name their runs identically, so the pid has to decide."""
    mine, theirs = 4242, 5353
    case_runs = _planted_case_runs(suite, [_PREDECESSOR, _SUCCESSOR])
    records = [_planted_join(theirs, _PREDECESSOR, _SUCCESSOR)]
    trace = _planted_trace(
        suite,
        [mine, theirs],
        {(theirs, 1): _PREDECESSOR, (theirs, 2): _SUCCESSOR},
        {(theirs, 1): _planted_boundary(4, 100, 200), (theirs, 2): _planted_boundary(4, 300, 400)},
    )

    assert not suite._established_pairs(records, case_runs, mine), (
        "the other endpoint's record established a pair for this endpoint, so a claim about this "
        "device could be carried by another device's runs"
    )
    assert not suite._boundaries_settled(trace, records, case_runs, mine)
    assert suite._established_pairs(records, case_runs, theirs) == {_PAIR}, (
        "the record is real evidence for the endpoint that wrote it; only its attribution is scoped"
    )
    # Unscoped is what the defect looked like: the pair is established, and its readings are found
    # on whichever child happens to hold rows under the same key.
    assert suite._whole_operator_order(trace, records, case_runs)[0] == {_PAIR}

    detail = suite._boundary_detail(trace, {_PAIR})
    assert set(detail) == {(mine, _PAIR), (theirs, _PAIR)}, (
        "a per-pair key would report only the child read last, hiding which endpoint the readings came from"
    )
    assert detail[mine, _PAIR] == {"predecessor": None, "successor": None}
    assert suite._boundary_detail(trace, {_PAIR}, mine) == {(mine, _PAIR): {"predecessor": None, "successor": None}}


def _three_run_wait_predicate(suite, trace, case_runs):
    """Use the scene's actual wait condition, not a separately reimplemented oracle."""
    method = suite.TestA5ThreeRunCapacity.test_three_runs_are_launched_and_accepted_at_once
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    waits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_await_records"
    ]
    assert len(waits) == 1 and isinstance(waits[0].args[3], ast.Lambda)
    expression = ast.Expression(waits[0].args[3])
    return eval(
        compile(expression, inspect.getfile(method), "eval"), {**vars(suite), "trace": trace, "case_runs": case_runs}
    )


@pytest.mark.parametrize("last_start", [500, 400, 350], ids=["ordered", "within_tick", "overlapping"])
def test_hbg_three_run_wait_drains_a_late_boundary_without_hiding_overlap(hbg_suite, monkeypatch, last_start):
    pid, third = 4242, (3, 2)
    runs = _planted_case_runs(hbg_suite, [_PREDECESSOR, _SUCCESSOR, third])
    trace = _planted_trace(
        hbg_suite,
        [pid],
        {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR, (pid, 3): third},
        {(pid, 1): _planted_boundary(0, 100, 200), (pid, 2): _planted_boundary(0, 300, 400)},
    )
    joins = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR), _planted_join(pid, _SUCCESSOR, third)]
    reads = []

    def take():
        reads.append(None)
        if len(reads) == 1:
            return joins
        trace.boundary_of_invocation[pid, 3] = _planted_boundary(0, last_start, 600)
        return []

    monkeypatch.setattr(trace, "take", take)
    records = []
    assert hbg_suite._await_records(trace, records, 1.0, _three_run_wait_predicate(hbg_suite, trace, runs))
    assert len(reads) == 2, "the join rows arrived before the last boundary; the real wait must keep draining"
    ordered, overlapping, within_tick, unmeasured = hbg_suite._whole_operator_order(trace, records, runs)
    assert not unmeasured
    assert _PAIR in ordered
    last_pair = (_SUCCESSOR, third)
    assert (last_pair in overlapping) == (last_start < 400)
    assert (last_pair in within_tick) == (last_start == 400)
    assert (last_pair in ordered) == (last_start > 400)


def test_hbg_three_run_wait_does_not_accept_a_permanently_missing_boundary(hbg_suite, monkeypatch):
    pid, third = 4242, (3, 2)
    runs = _planted_case_runs(hbg_suite, [_PREDECESSOR, _SUCCESSOR, third])
    trace = _planted_trace(
        hbg_suite,
        [pid],
        {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR, (pid, 3): third},
        {(pid, 1): _planted_boundary(0, 100, 200), (pid, 2): _planted_boundary(0, 300, 400)},
    )
    joins = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR), _planted_join(pid, _SUCCESSOR, third)]
    monkeypatch.setattr(trace, "take", lambda: joins)
    records = []
    assert not hbg_suite._await_records(trace, records, 0, _three_run_wait_predicate(hbg_suite, trace, runs))
    assert hbg_suite._whole_operator_order(trace, records, runs)[3] == {(_SUCCESSOR, third)}
