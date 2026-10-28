# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A correct final output or a nonmonotonic task order does not prove execution before readiness."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def ready_native():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_ready_order_execution.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_ready_order_hbg_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _sample(native, params, mode="valid", *, b_before_p=False):
    test = native.TestHbgReadyOrderExecution()
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    actual = golden.clone()
    rows = actual.records.view(-1, 16)
    groups = [
        [base + block * 3 + slot for block in range(n) for slot in range(3) if mask & (1 << slot)]
        for n, mask, base, *_ in actual.plan.tolist()
    ]
    p, a, b = groups[0], groups[-2], groups[-1]
    q = groups[1] if len(groups) == 4 else []
    b_tail_count = int(actual.plan[-1, 1]).bit_count()

    def enter(group):
        return [(row, 13) for row in group]

    def finish(group):
        return [(row, 14) for row in group]

    if mode == "host_order":
        events = enter(p) + finish(p) + enter(q) + finish(q) + enter(a) + finish(a) + enter(b) + finish(b)
    elif mode == "no_overlap":
        events = enter(p) + enter(q) + finish(q) + enter(b) + finish(b) + finish(p) + enter(a) + finish(a)
    elif q:
        assert not b_before_p
        if mode == "bridge_before_p":
            start = enter(q) + finish(q) + enter(p)
        else:
            start = enter(p) + enter(q) + finish(q)
        if mode == "late_b":
            start += finish(p) + enter(b) + finish(b[:-b_tail_count]) + enter(a[:1])
        elif mode == "early_a":
            start += enter(b[:b_tail_count]) + enter(a[:1]) + finish(p)
            start += enter(b[b_tail_count:]) + finish(b[:-b_tail_count])
        else:
            assert mode in ("valid", "bridge_before_p")
            start += enter(b[:b_tail_count]) + finish(p) + enter(b[b_tail_count:])
            start += finish(b[:-b_tail_count]) + enter(a[:1])
        events = start + finish(b[-b_tail_count:]) + enter(a[1:]) + finish(a)
    else:
        start = enter(b) + enter(p) if b_before_p else enter(p) + enter(b)
        if mode == "late_b":
            start = enter(p) + finish(p) + enter(b) + finish(b[:-b_tail_count]) + enter(a[:1])
        elif mode == "early_a":
            start += finish(b[:-b_tail_count]) + enter(a[:1]) + finish(p)
        else:
            assert mode == "valid"
            start += finish(b[:-b_tail_count]) + finish(p) + enter(a[:1])
        events = start + finish(b[-b_tail_count:]) + enter(a[1:]) + finish(a)
    for ordinal, (row, column) in enumerate(events, 1):
        rows[row, column] = ordinal
    test._contract_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    test._contract_publish_count = 1
    return test, actual, golden


@pytest.mark.parametrize("case_index", range(392))
def test_every_case_has_actual_fanin_and_bounded_peer_waits(ready_native, case_index):
    case = ready_native.TestHbgReadyOrderExecution.CASES[case_index]
    params = case["params"]
    test, actual, golden = _sample(ready_native, params)
    plan = golden.plan.tolist()
    p, a, b = plan[0], plan[-2], plan[-1]
    assert p[0] == 1 and p[1] == (2 if b[1] == 1 else 1)
    if b[1].bit_count() > 1:
        assert len(plan) == 4 and plan[1][:2] == [1, 2]
        assert golden.sync_flags.tolist() == [0, 0, *params["sync_flags"]]
        assert golden.wait_peers.tolist() == [3, 0, 3, 2]
        assert [row[5:7] for row in plan] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert b[8] == b[20] == 1
    else:
        assert len(plan) == 3
        assert golden.sync_flags.tolist() == [0, *params["sync_flags"]]
        assert golden.wait_peers.tolist() == [2, 2, 1]
        assert [row[5:7] for row in plan] == [[0, 0], [1, 1], [0, 0]]
    assert a[8] == a[20] == 0
    assert torch.equal(golden.plan, golden.host_plan)
    end = 0
    for index, row in enumerate(plan):
        n, mask, base, tag, delay, *_ = row
        assert base == end and tag == index + 1 and row[7] == n - 1 and delay == 0
        if index >= len(plan) - 2:
            capacity = params["clusters"] * (2 if mask in (2, 4) else 1)
            assert n == capacity + (not golden.sync_flags[index].item())
        end += n * 3 + 1
    assert golden.records.numel() == end * 16
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mode", ["host_order", "late_b"])
def test_b_before_a_without_a_still_blocked_is_not_evidence(ready_native, mode):
    params = ready_native.TestHbgReadyOrderExecution.CASES[3]["params"]
    test, actual, golden = _sample(ready_native, params, mode)
    with pytest.raises(AssertionError, match="B did not enter while A was blocked on P"):
        test.compare_outputs(actual, golden, ["records"], params)


def test_ready_order_does_not_allow_a_global_b_completion_barrier(ready_native):
    params = ready_native.TestHbgReadyOrderExecution.CASES[3]["params"]
    test, actual, golden = _sample(ready_native, params, "no_overlap")
    with pytest.raises(AssertionError, match="A did not enter before independent B finished"):
        test.compare_outputs(actual, golden, ["records"], params)


def test_a_still_has_to_obey_its_real_dependency(ready_native):
    params = ready_native.TestHbgReadyOrderExecution.CASES[3]["params"]
    test, actual, golden = _sample(ready_native, params, "early_a")
    with pytest.raises(AssertionError, match="consumer entered before all producer blocks returned"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("b_before_p", [False, True])
def test_p_can_start_on_its_nonmember_core_before_or_after_b(ready_native, b_before_p):
    params = ready_native.TestHbgReadyOrderExecution.CASES[3]["params"]
    test, actual, golden = _sample(ready_native, params, b_before_p=b_before_p)
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize(
    "corruption", ["peer_timeout", "missing_branch", "missing_p", "early_output", "legacy", "stale"]
)
def test_complete_event_order_cannot_hide_other_contract_failures(ready_native, corruption):
    params = ready_native.TestHbgReadyOrderExecution.CASES[3]["params"]
    test, actual, golden = _sample(ready_native, params)
    rows = actual.records.view(-1, 16)
    if corruption == "peer_timeout":
        rows[3, 5] = 1
    elif corruption == "missing_branch":
        rows[3, 6] = 0
    elif corruption == "missing_p":
        rows[int(actual.plan[0, 1] == 2), 0] = 0
    elif corruption == "early_output":
        rows[int(actual.plan[1, 2]), 12] = 1
    elif corruption == "legacy":
        test._contract_worker.scheduler_runtime_mode = 4
    else:
        test._contract_worker.scheduler_mode_publish_count = 1
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records"], params)


def test_matrix_preserves_all_modes_and_reuses_existing_kernel_functions(ready_native):
    cls = ready_native.TestHbgReadyOrderExecution
    cases = cls.CASES
    assert len(cases) == len({case["name"] for case in cases}) == 392
    assert all("skip_golden" not in c and "manual" not in c and "marks" not in c for c in cases)
    for platform in ("a5sim", "a5"):
        shapes = {
            (p["mask"], p["other_mask"], p["sync_flags"])
            for c in cases
            if c["platforms"] == [platform]
            for p in [c["params"]]
        }
        assert shapes == {(a, b, (sa, sb)) for a in range(1, 8) for b in range(1, 8) for sa in (0, 1) for sb in (0, 1)}
    assert all(
        Path(entry["source"]).name == f"sync_ordered_slot_{slot}.cpp"
        for slot, entry in enumerate(cls.CALLABLE["incores"])
    )


@pytest.mark.parametrize("mask", (3, 5, 6, 7))
def test_mix_b_has_a_readiness_bridge_that_cannot_hold_its_predecessors_core(ready_native, mask):
    params = {"scenario": "ready_before", "mask": 6, "other_mask": mask, "clusters": 8, "sync_flags": (1, 1)}
    args = ready_native.TestHbgReadyOrderExecution().generate_args(params)
    assert args.plan.shape == (4, 32), "Mix B needs P, an entry bridge Q, A and B"
    p, q, a, b = args.plan.tolist()
    assert p[:2] == [1, 1] and q[:2] == [1, 2]
    assert [row[5:7] for row in (p, q, a, b)] == [[0, 0], [0, 0], [1, 1], [1, 1]]
    assert a[8] == a[20] == 0
    assert b[8] == b[20] == 1
    assert args.wait_peers.tolist() == [3, 0, 3, 2]
    assert args.sync_flags.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("mask", (3, 5, 6, 7))
@pytest.mark.parametrize("mode", ("host_order", "late_b", "no_overlap", "early_a", "bridge_before_p"))
def test_mix_b_oracle_rejects_missing_readiness_and_overlap_evidence(ready_native, mask, mode):
    params = {"scenario": "ready_before", "mask": 7, "other_mask": mask, "clusters": 8, "sync_flags": (1, 1)}
    test, actual, golden = _sample(ready_native, params, mode)
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records"], params)
