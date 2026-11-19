# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU checks of the native CLA acceptance plan and failure sentinels."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def native_cla():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_cla_hbg.py"
    spec = importlib.util.spec_from_file_location("_cla_hbg_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_native_projection_keeps_fifty_eight_nodes_and_eighty_eight_edges(native_cla):
    plan, _, nodes = native_cla.make_plan({})
    assert len(nodes) == 58 and sum(n["kind"] == "dummy" for n in nodes) == 2
    assert not any(n["kind"] == "alloc" for n in nodes)
    assert plan.shape == (58, 32) and plan[:, 5].sum().item() == 88
    assert sum(row[0] * bin(row[1]).count("1") for row in plan.tolist()) == 736


@pytest.mark.parametrize("case_index", range(7))
def test_each_native_case_has_bounded_forward_refs_disjoint_records_and_guards(native_cla, case_index):
    case = native_cla.TestClaHbgContract.CASES[case_index]
    plan, total, nodes = native_cla.make_plan(case["params"])
    previous_end = 0
    for i, row in enumerate(plan.tolist()):
        n, mask, base, tag, delay, count, check_count, slow = row[:8]
        assert base == previous_end and tag == i + 1 and n > 0 and delay > 0 and 0 <= slow < n
        assert 0 <= count <= 12 and 0 <= check_count <= 12
        assert all(0 <= predecessor < i for predecessor in row[8 : 8 + count] + row[20 : 20 + check_count])
        assert all(plan[p, 1] != 0 for p in row[20 : 20 + check_count])
        previous_end += n * 3 + 1
    assert previous_end == total
    test = native_cla.TestClaHbgContract()
    args = test.generate_args(case["params"])
    test.compute_golden(args, case["params"])
    rows = args.records.view(-1, 16)
    assert rows[:, 0].sum().item() == sum(n["executions"] for n in nodes)
    assert torch.count_nonzero(rows[:, 12:]).item() == 0
    assert torch.equal(args.plan, args.host_plan)


def test_dummy_does_not_hide_its_true_kernel_predecessors_from_the_sentinel(native_cla):
    plan, _, nodes = native_cla.make_plan({})
    refs = {node["label"]: i for i, node in enumerate(nodes)}
    for label in ("r3t12", "r3t14"):
        row = plan[refs[label]].tolist()
        assert refs["r1t34"] in row[8 : 8 + row[5]]
        checks = set(row[20 : 20 + row[6]])
        assert {refs["r1t33"], refs["r2t15"], refs["r2t20"]} <= checks
        assert refs["r1t34"] not in checks


def test_aiv1_variant_changes_only_the_mix_slot_not_block_count_or_edges(native_cla):
    original, size, _ = native_cla.make_plan({})
    variant, other_size, _ = native_cla.make_plan({"aiv1": True})
    assert size == other_size
    assert torch.equal(original[:, 2:], variant[:, 2:]) and torch.equal(original[:, 0], variant[:, 0])
    assert set(variant[original[:, 1] == 3, 1].tolist()) == {5}


def test_delaying_first_block_does_not_redefine_task_completion_as_last_index(native_cla):
    original, _, _ = native_cla.make_plan({})
    variant, _, _ = native_cla.make_plan({"tail_first": True})
    assert torch.equal(original[:, :7], variant[:, :7]) and torch.equal(original[:, 8:], variant[:, 8:])
    assert torch.count_nonzero(variant[:, 7]).item() == 0


@pytest.mark.parametrize("mode", [0, 3, 4, 5])
def test_native_gate_rejects_legacy_even_after_all_outputs_pass(native_cla, mode):
    test = native_cla.TestClaHbgContract()
    expected = test.generate_args({})
    test.compute_golden(expected, {})
    test._cla_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=2)
    test._cla_count = 1
    with pytest.raises(AssertionError, match="resident required"):
        test.compare_outputs(expected.clone(), expected, ["records"], {})


def test_early_consumer_sentinel_cannot_be_repaired_by_later_correct_outputs(native_cla):
    test = native_cla.TestClaHbgContract()
    expected = test.generate_args({})
    test.compute_golden(expected, {})
    actual = expected.clone()
    actual.records[12] = 1
    test._cla_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    test._cla_count = 1
    with pytest.raises(AssertionError, match="predecessors are incomplete"):
        test.compare_outputs(actual, expected, ["records"], {})


def test_stale_resident_latch_is_not_a_pass(native_cla):
    test = native_cla.TestClaHbgContract()
    args = test.generate_args({})
    test.compute_golden(args, {})
    test._cla_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=1)
    test._cla_count = 1
    with pytest.raises(AssertionError, match="no fresh scheduler publication"):
        test.compare_outputs(args, args, ["records"], {})


@pytest.mark.parametrize("field,message", [(0, "missing pointer"), (1, "does not match the plan")])
def test_context_guard_is_reported_before_downstream_missing_output(native_cla, field, message):
    test = native_cla.TestClaHbgContract()
    expected = test.generate_args({})
    test.compute_golden(expected, {})
    actual = expected.clone()
    row = expected.plan[0].tolist()
    actual.records[(row[2] + row[0] * 3) * 16 + field] = 1
    actual.records[12] = 1
    with pytest.raises(AssertionError, match=message):
        test.compare_outputs(actual, expected, ["records"], {})
