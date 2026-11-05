# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Counterexamples for the heterogeneous sync probe oracle; these do not replace native protocol validation."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def sync_ordered():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_sync_ordered_execution.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_sync_ordered_hbg_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _sample(native, params, *, overlap=None, reverse=False):
    test = native.TestHbgSyncOrderedExecution()
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    actual = golden.clone()
    rows = actual.records.view(-1, 16)
    groups = [
        [base + block * 3 + slot for block in range(n) for slot in range(3) if mask & (1 << slot)]
        for n, mask, base, *_ in actual.plan.tolist()
    ]
    if reverse:
        groups.reverse()
    if overlap is None:
        overlap = params["scenario"] == "overlap"
    if overlap:
        events = [(row, 13) for group in groups for row in group] + [(row, 14) for group in groups for row in group]
    else:
        events = [(row, col) for group in groups for col in (13, 14) for row in group]
    for event, (row, col) in enumerate(events, 1):
        rows[row, col] = event
    test._contract_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    test._contract_publish_count = 1
    return test, actual, golden


@pytest.mark.parametrize("case_index", range(224))
def test_each_case_preserves_shape_capacity_barrier_and_complete_oracle(sync_ordered, case_index):
    case = sync_ordered.TestHbgSyncOrderedExecution.CASES[case_index]
    params = case["params"]
    test, actual, golden = _sample(sync_ordered, params)
    assert len(case["platforms"]) == 1
    clusters = 8 if case["platforms"] == ["a5sim"] else 32
    assert len(golden.plan) == len(golden.sync_flags) == 2
    assert torch.equal(golden.plan, golden.host_plan)
    end = 0
    for task, (n, mask, base, tag, delay, deps, checks, slow, *rest) in enumerate(golden.plan.tolist()):
        capacity = clusters * (2 if mask in (2, 4) else 1)
        sync = golden.sync_flags[task].item()
        assert sync in (0, 1)
        assert n == (capacity if sync else capacity + 1)
        assert base == end and tag == task + 1 and delay > 0 and slow == n - 1
        assert deps == checks == (task if params["scenario"] == "edge" else 0)
        assert rest[:deps] == rest[12 : 12 + checks] == ([0] if deps else [])
        assert golden.records.view(-1, 16)[base + 3 * n, 2].item() == (n * mask.bit_count() if sync else 0)
        assert golden.records.view(-1, 16)[base + 3 * n, 6].item() == (
            mask.bit_count() if params["scenario"] == "overlap" else 0
        )
        end += n * 3 + 1
    assert golden.records.numel() == end * 16
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", [(1, 7), (2, 6), (6, 1), (7, 2), (1, 2), (3, 5), (2, 4)])
def test_serial_independent_execution_is_not_accepted_as_overlap(sync_ordered, mask, other):
    params = {"scenario": "overlap", "mask": mask, "other_mask": other, "clusters": 8, "sync_flags": (1, 1)}
    test, actual, golden = _sample(sync_ordered, params, overlap=False)
    with pytest.raises(AssertionError, match="independent tasks did not overlap"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("reverse", [False, True])
def test_ready_order_is_not_host_submission_order(sync_ordered, reverse):
    params = {"scenario": "overlap", "mask": 7, "other_mask": 2, "clusters": 8, "sync_flags": (1, 1)}
    test, actual, golden = _sample(sync_ordered, params, reverse=reverse)
    test.compare_outputs(actual, golden, ["records"], params)


def test_real_dependency_must_wait_for_all_predecessor_returns(sync_ordered):
    params = {"scenario": "edge", "mask": 7, "other_mask": 2, "clusters": 8, "sync_flags": (1, 1)}
    test, actual, golden = _sample(sync_ordered, params, overlap=True)
    with pytest.raises(AssertionError, match="consumer entered before all producer blocks returned"):
        test.compare_outputs(actual, golden, ["records"], params)


def test_event_order_must_prove_barrier_not_only_eventual_arrival_count(sync_ordered):
    params = {"scenario": "overlap", "mask": 1, "other_mask": 7, "clusters": 8, "sync_flags": (1, 1)}
    test, actual, golden = _sample(sync_ordered, params)
    rows = actual.records.view(-1, 16)
    # All arrival counters and event permutations remain correct, but block 0
    # returns before the last block enters the same rendezvous.
    rows[0, 14], rows[21, 13] = rows[21, 13].clone(), rows[0, 14].clone()
    with pytest.raises(AssertionError, match="sync return precedes all task entries"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_arrival",
        "extra_arrival",
        "timeout",
        "peer_timeout",
        "missing_peer",
        "no_peer_wait",
        "bad_identity",
        "inactive",
    ],
)
def test_bad_barrier_or_identity_cannot_pass(sync_ordered, corruption):
    params = {"scenario": "overlap", "mask": 6, "other_mask": 1, "clusters": 8, "sync_flags": (1, 1)}
    test, actual, golden = _sample(sync_ordered, params)
    rows = actual.records.view(-1, 16)
    guard = int(actual.plan[0, 2] + 3 * actual.plan[0, 0])
    if corruption == "missing_arrival":
        rows[guard, 2] -= 1
    elif corruption == "extra_arrival":
        rows[guard, 2] += 1
    elif corruption == "timeout":
        rows[guard, 3] = 1
    elif corruption == "peer_timeout":
        rows[guard, 5] = 1
    elif corruption == "missing_peer":
        rows[guard, 4] = 0
    elif corruption == "no_peer_wait":
        rows[guard, 6] = 0
    elif corruption == "bad_identity":
        rows[1, 9] = 1
    else:
        rows[0, 13:15] = torch.tensor([1, 2])
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records"], params)


def test_matrix_has_every_mask_pair_and_both_sync_ordinary_transitions(sync_ordered):
    cases = sync_ordered.TestHbgSyncOrderedExecution.CASES
    assert len(cases) == len({case["name"] for case in cases}) == 224
    assert all("skip_golden" not in c and "manual" not in c and "marks" not in c for c in cases)
    for platform in ("a5sim", "a5"):
        subset = [c["params"] for c in cases if c["platforms"] == [platform]]
        for scenario in ("overlap", "edge"):
            pairs = {
                (p["mask"], p["other_mask"]) for p in subset if p["scenario"] == scenario and p["sync_flags"] == (1, 1)
            }
            assert pairs == {(a, b) for a in range(1, 8) for b in range(1, 8)}
        for mask in range(1, 8):
            flags = {p["sync_flags"] for p in subset if p["mask"] == p["other_mask"] == mask}
            assert flags == {(1, 1), (1, 0), (0, 1)}
