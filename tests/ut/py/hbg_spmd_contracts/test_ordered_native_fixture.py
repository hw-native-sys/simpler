# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU checks of ordered native records, failure gates and per-invocation provenance."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def ordered_native():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_ordered_execution.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_ordered_hbg_native_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _sample(native, params, *, overlap=None, reverse=False):
    test = native.TestHbgOrderedExecution()
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    actual = golden.clone()
    records = actual.records.view(-1, 16)
    groups = [
        [base + block * 3 + (0 if mask == 1 else 1) for block in range(n)] for n, mask, base, *_ in actual.plan.tolist()
    ]
    if reverse:
        groups.reverse()
    if overlap is None:
        overlap = params["scenario"] == "overlap"
    if overlap:
        sequence = [(row, 13) for group in groups for row in group] + [(row, 14) for group in groups for row in group]
    else:
        sequence = [(row, column) for group in groups for row in group for column in (13, 14)]
    for event, (row, column) in enumerate(sequence, 1):
        records[row, column] = event
    test._contract_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    test._contract_publish_count = 1
    return test, actual, golden


@pytest.mark.parametrize("case_index", range(32))
def test_each_native_case_has_valid_shape_identity_events_and_dependency_projection(ordered_native, case_index):
    params = ordered_native.TestHbgOrderedExecution.CASES[case_index]["params"]
    test, actual, golden = _sample(ordered_native, params)
    plan = golden.plan.tolist()
    end = 0
    for task, row in enumerate(plan):
        n, mask, base, tag, delay, count, checks, slow = row[:8]
        assert mask in (1, 2) and 0 < n <= 512
        assert base == end and tag == task + 1 and delay > 0 and 0 <= slow < n
        assert 0 <= count <= 12 and count == checks
        assert row[8 : 8 + count] == row[20 : 20 + checks]
        assert all(0 <= predecessor < task for predecessor in row[8 : 8 + count])
        end += n * 3 + 1
    assert end * 16 == golden.records.numel()
    assert torch.equal(golden.plan, golden.host_plan)
    assert golden.records.view(-1, 16)[plan[0][2] + plan[0][0] * 3, 7].item() == 2 * sum(row[0] for row in plan)
    test.compare_outputs(actual, golden, ["records"], params)
    assert test._contract_publish_count == 2


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_visit",
        "missing_visit",
        "missing_record",
        "wrong_block",
        "wrong_n",
        "wrong_slot",
        "wrong_task",
        "missing_context",
        "out_of_range_context",
        "negative_entry",
        "missing_entry",
        "reversed_interval",
        "empty_interval",
        "duplicate_event",
        "gapped_event",
        "incomplete_fanin",
        "wrong_event_count",
        "inactive_execution",
        "unexpected_reserved_word",
    ],
)
def test_corrupt_record_cannot_pass_complete_comparison(ordered_native, corruption):
    params = {"scenario": "shape", "mask": 1, "blocks": 2}
    test, actual, golden = _sample(ordered_native, params)
    rows = actual.records.view(-1, 16)
    if corruption == "missing_record":
        rows[0] = 0
    elif corruption == "reversed_interval":
        rows[0, 13], rows[0, 14] = rows[0, 14].clone(), rows[0, 13].clone()
    elif corruption == "inactive_execution":
        rows[1, 13:15] = torch.tensor([1, 2])
    else:
        mutations = {
            "duplicate_visit": (0, 0, 2),
            "missing_visit": (0, 0, 0),
            "wrong_block": (0, 8, 2),
            "wrong_n": (0, 9, 1),
            "wrong_slot": (0, 10, 2),
            "wrong_task": (0, 11, 2),
            "missing_context": (6, 0, 1),
            "out_of_range_context": (6, 1, 1),
            "negative_entry": (0, 13, -1),
            "missing_entry": (0, 13, 0),
            "empty_interval": (0, 14, rows[0, 13].item()),
            "duplicate_event": (0, 14, rows[3, 14].item()),
            "gapped_event": (0, 14, 9),
            "incomplete_fanin": (0, 12, 1),
            "wrong_event_count": (6, 7, 0),
            "unexpected_reserved_word": (0, 15, 1),
        }
        row, column, value = mutations[corruption]
        rows[row, column] = value
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", [(1, 1), (2, 2), (1, 2), (2, 1)])
def test_real_dependency_rejects_consumer_entry_before_last_producer_return(ordered_native, mask, other):
    params = {"scenario": "edge", "mask": mask, "other_mask": other, "blocks": 3}
    test, actual, golden = _sample(ordered_native, params, overlap=True)
    assert torch.count_nonzero(actual.records.view(-1, 16)[:, 12]).item() == 0
    with pytest.raises(AssertionError, match="consumer entered before all producer blocks returned"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", [(1, 1), (2, 2), (1, 2), (2, 1)])
def test_nonoverlapping_independent_tasks_fail_overlap_acceptance(ordered_native, mask, other):
    params = {"scenario": "overlap", "mask": mask, "other_mask": other, "blocks": 3}
    test, actual, golden = _sample(ordered_native, params, overlap=False)
    with pytest.raises(AssertionError, match="independent tasks did not overlap"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", [(1, 1), (2, 2), (1, 2), (2, 1)])
def test_overlap_allows_either_independent_task_to_enter_first(ordered_native, mask, other):
    params = {"scenario": "overlap", "mask": mask, "other_mask": other, "blocks": 3}
    test, actual, golden = _sample(ordered_native, params, overlap=True, reverse=True)
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mode", [0, 3, 4, 5, -1])
def test_correct_records_do_not_allow_nonresident_mode(ordered_native, mode):
    params = {"scenario": "shape", "mask": 1, "blocks": 2}
    test, actual, golden = _sample(ordered_native, params)
    test._contract_worker.scheduler_runtime_mode = mode
    with pytest.raises(AssertionError, match="resident required"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("count", [0, 1])
def test_correct_records_do_not_allow_stale_resident_publication(ordered_native, mode, count):
    params = {"scenario": "shape", "mask": 2, "blocks": 2}
    test, actual, golden = _sample(ordered_native, params)
    test._contract_worker.scheduler_runtime_mode = mode
    test._contract_worker.scheduler_mode_publish_count = count
    with pytest.raises(AssertionError, match="no fresh scheduler publication"):
        test.compare_outputs(actual, golden, ["records"], params)


def _install_cpu_run(monkeypatch, native, worker, callable_obj, observed, broken_step=None, failure="stale"):
    def run(self, actual_worker, actual_callable, case, *args, **kwargs):
        assert actual_worker is worker and actual_callable is callable_obj
        params = case["params"]
        _, actual, golden = _sample(native, params)
        for _ in range(kwargs.get("rounds", 1)):
            step = len(observed)
            observed.append(params["blocks"])
            worker.scheduler_runtime_mode = 3 if step == broken_step and failure == "legacy" else 2
            if step != broken_step or failure != "stale":
                worker.scheduler_mode_publish_count += 1
            self.compare_outputs(actual, golden, ["records"], params)

    # The real ordered and inherited fresh-mode hooks remain in the call chain.
    monkeypatch.setattr(native.SceneTestCase, "_run_and_validate", run)


@pytest.mark.parametrize("mask", [1, 2])
def test_retained_reuse_checks_every_round_and_keeps_worker_and_registration(ordered_native, monkeypatch, mask):
    case = next(
        c for c in ordered_native.TestHbgOrderedExecution.CASES if c["name"] == f"retained_large_small_mask{mask}"
    )
    worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=11)
    callable_obj = object()
    observed = []
    _install_cpu_run(monkeypatch, ordered_native, worker, callable_obj, observed)
    test = ordered_native.TestHbgOrderedExecution()
    test._run_and_validate(worker, callable_obj, case, rounds=2)
    assert observed == [512, 512, 1, 1, 65, 65, 1, 1, 512, 512]
    assert worker.scheduler_mode_publish_count == test._contract_publish_count == 21


@pytest.mark.parametrize("broken_step", range(10))
@pytest.mark.parametrize("failure", ["stale", "legacy"])
def test_no_reuse_round_can_reuse_old_or_legacy_provenance(ordered_native, monkeypatch, broken_step, failure):
    case = next(c for c in ordered_native.TestHbgOrderedExecution.CASES if c["name"] == "retained_large_small_mask1")
    worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=7)
    callable_obj = object()
    observed = []
    _install_cpu_run(monkeypatch, ordered_native, worker, callable_obj, observed, broken_step, failure)
    test = ordered_native.TestHbgOrderedExecution()
    expected = "no fresh scheduler publication" if failure == "stale" else "resident required"
    with pytest.raises(AssertionError, match=expected):
        test._run_and_validate(worker, callable_obj, case, rounds=2)
    assert len(observed) == broken_step + 1


def test_ordered_acceptance_cannot_disable_golden_or_provenance(ordered_native):
    worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=3)
    test = ordered_native.TestHbgOrderedExecution()
    with pytest.raises(AssertionError, match="cannot skip golden or invocation provenance"):
        test._run_and_validate(worker, object(), test.CASES[0], skip_golden=True)


def test_matrix_is_unique_unguarded_and_reuses_existing_orchestration(ordered_native):
    cases = ordered_native.TestHbgOrderedExecution.CASES
    assert len(cases) == len({case["name"] for case in cases}) == 32
    assert all("skip_golden" not in case and "manual" not in case and "marks" not in case for case in cases)
    source = Path(ordered_native.TestHbgOrderedExecution.CALLABLE["orchestration"]["source"])
    assert source == Path(ordered_native.__file__).parent / "kernels/cla_orchestration.cpp"
