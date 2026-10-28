# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only adversarial checks of native GM Mix records and fresh-run guards."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MASKS = (3, 5, 6, 7)
_PAIRS = ((3, 7), (7, 3), (6, 1), (1, 6), (5, 2), (2, 5), (6, 6), (7, 7))
_ACTIVE_SLOTS = [(mask, slot) for mask in _MASKS for slot in range(3) if mask & (1 << slot)]


@pytest.fixture(scope="module")
def gm_native():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_gm_mix_execution.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_gm_mix_native_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _active_groups(plan):
    return [
        [base + block * 3 + slot for block in range(n) for slot in range(3) if mask & (1 << slot)]
        for n, mask, base, *_ in plan.tolist()
    ]


def _stamp_events(actual, sequence):
    records = actual.records.view(-1, 16)
    records[:, 13:15] = 0
    for event, (row, column) in enumerate(sequence, 1):
        records[row, column] = event


def _sample(native, params, *, overlap=None, reverse=False):
    test = native.TestHbgGmMixExecution()
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    actual = golden.clone()
    groups = _active_groups(actual.plan)
    if reverse:
        groups.reverse()
    if overlap is None:
        overlap = params["scenario"] == "overlap"
    if overlap:
        sequence = [(row, 13) for group in groups for row in group] + [(row, 14) for group in groups for row in group]
    else:
        sequence = [(row, column) for group in groups for row in group for column in (13, 14)]
    _stamp_events(actual, sequence)
    test._contract_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    test._contract_publish_count = 1
    return test, actual, golden


@pytest.mark.parametrize("case_index", range(68))
def test_every_gm_case_checks_all_active_logical_slots_and_original_shape(gm_native, case_index):
    params = gm_native.TestHbgGmMixExecution.CASES[case_index]["params"]
    test, actual, golden = _sample(gm_native, params)
    rows = golden.records.view(-1, 16)
    expected = 0
    end = 0
    for task, plan in enumerate(golden.plan.tolist()):
        n, mask, base, tag, delay, count, checks, slow = plan[:8]
        assert base == end and tag == task + 1 and 0 < n <= 512
        assert mask in (1, 2, 3, 5, 6, 7) and delay > 0 and 0 <= slow < n
        assert 0 <= count <= 12 and count == checks
        assert plan[8 : 8 + count] == plan[20 : 20 + checks]
        assert all(0 <= predecessor < task for predecessor in plan[8 : 8 + count])
        for block in range(n):
            for slot in range(3):
                row = rows[base + block * 3 + slot]
                if mask & (1 << slot):
                    assert row[0].item() == 1
                    assert row[8:12].tolist() == [block + 1, n, slot + 1, tag]
                    expected += 1
                else:
                    assert torch.count_nonzero(row).item() == 0
        end += n * 3 + 1
    assert end * 16 == golden.records.numel()
    assert rows[:, 0].sum().item() == expected
    first = golden.plan[0]
    assert rows[first[2].item() + first[0].item() * 3, 7].item() == 2 * expected
    assert torch.equal(golden.plan, golden.host_plan)
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,slot", _ACTIVE_SLOTS)
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_identity",
        "duplicate_visit",
        "missing_events",
        "wrong_slot",
        "wrong_n",
        "wrong_block",
        "duplicate_ordinal",
    ],
)
def test_no_active_mix_slot_can_be_omitted_or_replaced_by_another(gm_native, mask, slot, corruption):
    params = {"scenario": "shape", "mask": mask, "blocks": 2}
    test, actual, golden = _sample(gm_native, params)
    rows = actual.records.view(-1, 16)
    if corruption == "missing_identity":
        rows[slot, :13] = 0
    elif corruption == "duplicate_visit":
        rows[slot, 0] = 2
    elif corruption == "missing_events":
        rows[slot, 13:15] = 0
    elif corruption == "wrong_slot":
        rows[slot, 10] = (slot + 1) % 3 + 1
    elif corruption == "wrong_n":
        rows[slot, 9] = 1
    elif corruption == "wrong_block":
        rows[slot, 8] = 2
    else:
        rows[slot, 14] = rows[slot + 3, 14]
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,slot", [(3, 2), (5, 1), (6, 0)])
@pytest.mark.parametrize("field", [0, 8, 13])
def test_inactive_resource_lanes_cannot_fabricate_kernel_records(gm_native, mask, slot, field):
    params = {"scenario": "shape", "mask": mask, "blocks": 2}
    test, actual, golden = _sample(gm_native, params)
    actual.records.view(-1, 16)[slot, field] = 1
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("blocks", [1, 2, 65, 512])
def test_mask_six_has_exactly_two_vector_identities_per_block_and_no_cube_event(gm_native, blocks):
    params = {"scenario": "shape", "mask": 6, "blocks": blocks}
    test, actual, golden = _sample(gm_native, params)
    rows = actual.records.view(-1, 16)
    for n, _, base, *_ in actual.plan.tolist():
        assert torch.count_nonzero(rows[[base + block * 3 for block in range(n)]]).item() == 0
    assert rows[:, 0].sum().item() == 4 * blocks
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", _PAIRS)
def test_one_late_mix_subtask_blocks_a_true_consumer_even_when_all_outputs_match(gm_native, mask, other):
    params = {"scenario": "edge", "mask": mask, "other_mask": other, "blocks": 2}
    test, actual, golden = _sample(gm_native, params)
    producer, consumer = _active_groups(actual.plan)
    late = producer[-1]
    sequence = [(late, 13)]
    sequence += [(row, column) for row in producer[:-1] for column in (13, 14)]
    sequence += [(row, column) for row in consumer for column in (13, 14)]
    sequence += [(late, 14)]
    _stamp_events(actual, sequence)
    assert torch.equal(actual.records.view(-1, 16)[:, :13], golden.records.view(-1, 16)[:, :13])
    with pytest.raises(AssertionError, match="consumer entered before all producer blocks returned"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", _PAIRS)
def test_nonoverlap_cannot_pass_independent_gm_task_overlap_gate(gm_native, mask, other):
    params = {"scenario": "overlap", "mask": mask, "other_mask": other, "blocks": 2}
    test, actual, golden = _sample(gm_native, params, overlap=False)
    with pytest.raises(AssertionError, match="independent tasks did not overlap"):
        test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mask,other", _PAIRS)
def test_either_independent_task_may_enter_first_without_changing_the_overlap_contract(gm_native, mask, other):
    params = {"scenario": "overlap", "mask": mask, "other_mask": other, "blocks": 2}
    test, actual, golden = _sample(gm_native, params, overlap=True, reverse=True)
    test.compare_outputs(actual, golden, ["records"], params)


@pytest.mark.parametrize("mode,count", [(0, 2), (3, 2), (4, 2), (2, 1), (1, 1)])
def test_gm_records_never_substitute_for_fresh_resident_provenance(gm_native, mode, count):
    params = {"scenario": "shape", "mask": 6, "blocks": 2}
    test, actual, golden = _sample(gm_native, params)
    test._contract_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=count)
    with pytest.raises(AssertionError):
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

    monkeypatch.setattr(native.SceneTestCase, "_run_and_validate", run)


@pytest.mark.parametrize("mask", _MASKS)
def test_each_gm_reuse_shape_checks_every_round_with_the_same_worker_and_callable(gm_native, monkeypatch, mask):
    case = next(
        case for case in gm_native.TestHbgGmMixExecution.CASES if case["name"] == f"mix_retained_large_small_mask{mask}"
    )
    worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=7)
    callable_obj = object()
    observed = []
    _install_cpu_run(monkeypatch, gm_native, worker, callable_obj, observed)
    test = gm_native.TestHbgGmMixExecution()
    test._run_and_validate(worker, callable_obj, case, rounds=3)
    assert observed == [n for n in (512, 1, 65, 1, 512) for _ in range(3)]
    assert worker.scheduler_mode_publish_count == test._contract_publish_count == 22


@pytest.mark.parametrize("broken_step", range(15))
@pytest.mark.parametrize("failure", ["stale", "legacy"])
def test_no_gm_reuse_round_may_use_stale_or_legacy_provenance(gm_native, monkeypatch, broken_step, failure):
    case = next(
        case for case in gm_native.TestHbgGmMixExecution.CASES if case["name"] == "mix_retained_large_small_mask6"
    )
    worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=7)
    callable_obj = object()
    observed = []
    _install_cpu_run(monkeypatch, gm_native, worker, callable_obj, observed, broken_step, failure)
    expected = "no fresh scheduler publication" if failure == "stale" else "resident required"
    with pytest.raises(AssertionError, match=expected):
        gm_native.TestHbgGmMixExecution()._run_and_validate(worker, callable_obj, case, rounds=3)
    assert len(observed) == broken_step + 1


def test_gm_matrix_counts_all_eighty_four_invocations_without_silent_weakening(gm_native):
    cases = gm_native.TestHbgGmMixExecution.CASES
    assert len(cases) == len({case["name"] for case in cases}) == 68
    assert sum(len(case["params"].get("invocation_blocks", (0,))) for case in cases) == 84
    for mask in (3, 5, 6, 7):
        shapes = {case["params"].get("blocks") for case in cases if case["params"]["mask"] == mask}
        assert {1, 2, 3, 15, 16, 17, 63, 64, 65} <= shapes
    assert all("skip_golden" not in case and "manual" not in case and "marks" not in case for case in cases)
    kernels = gm_native.TestHbgGmMixExecution.CALLABLE["incores"]
    assert [(kernel["func_id"], kernel["core_type"]) for kernel in kernels] == [(0, "aic"), (1, "aiv"), (2, "aiv")]
