# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU checks of the real async-boundary fixture; mocked records are not device evidence."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from simpler_setup import SceneTestCase
from simpler_setup.scene_test import _discover_module_test_classes

_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_async_boundary.py"


@pytest.fixture(scope="module")
def async_native():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(_FIXTURE_PATH.parent))
        spec = importlib.util.spec_from_file_location("_hbg_async_boundary_native_fixture", _FIXTURE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module


def _invoke_fixture(monkeypatch, native, *, mode=2, increments=(1, 1, 1), rounds=1, corruption=None, skip=None):
    test = native.TestHbgAsyncBoundary()
    case = {
        "name": "cpu_guard",
        "params": {"mask": 3, "blocks": 2, "tasks": 2, "tail_delay": 0},
    }
    kwargs = {"rounds": rounds}
    if skip == "argument":
        kwargs["skip_golden"] = True
    elif skip == "case":
        case["skip_golden"] = True
    elif skip == "class":
        monkeypatch.setattr(native.TestHbgAsyncBoundary, "SKIP_GOLDEN", True)
    worker = SimpleNamespace(scheduler_mode_publish_count=10, scheduler_runtime_mode=mode)
    callable_obj = object()
    seen = []

    def fake_device_rounds(self, supplied_worker, supplied_callable, supplied_case, *, rounds=1, **kwargs):
        assert supplied_worker is worker
        assert supplied_callable is callable_obj
        assert supplied_case is case
        assert rounds == max(3, kwargs_requested_rounds)
        params = case["params"]
        golden = self.generate_args(params)
        self.compute_golden(golden, params)
        for invocation in range(rounds):
            actual = golden.clone()
            if corruption is not None:
                corruption(actual)
            worker.scheduler_mode_publish_count += increments[invocation]
            self.compare_outputs(actual, golden, ["records", "summary"], params)
            seen.append((supplied_worker, supplied_callable))

    kwargs_requested_rounds = rounds
    monkeypatch.setattr(SceneTestCase, "_run_and_validate_l2", fake_device_rounds)
    test._run_and_validate(worker, callable_obj, case, **kwargs)
    return test, worker, callable_obj, seen


@pytest.mark.parametrize("mask", (1, 2, 4, 3, 5, 6, 7))
@pytest.mark.parametrize("blocks", (1, 2))
def test_async_golden_records_every_active_slot_and_preserves_satisfied_counters(async_native, mask, blocks):
    test = async_native.TestHbgAsyncBoundary()
    params = {"mask": mask, "blocks": blocks, "tasks": 2, "tail_delay": 0}
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    rows = golden.records.view(2, blocks * 3 + 1, 16)
    expected = 2 * blocks * mask.bit_count()
    assert golden.summary[0].item() == expected
    assert torch.count_nonzero(golden.summary[1:]).item() == 0
    assert torch.count_nonzero(rows[:, :, 1:8]).item() == 0
    assert torch.count_nonzero(rows[:, -1, :]).item() == 0
    for task in range(2):
        for block in range(blocks):
            for slot in range(3):
                row = rows[task, block * 3 + slot]
                if mask & (1 << slot):
                    assert row[0].item() == 1
                    assert row[8:12].tolist() == [block + 1, blocks, slot + 1, task + 1]
                    assert row[12:16].tolist() == [0, 0, 0, async_native._ASYNC_COMPLETE]
                else:
                    assert torch.count_nonzero(row).item() == 0


@pytest.mark.parametrize("mode", (1, 2))
@pytest.mark.parametrize("rounds", (1, 2, 3, 5))
def test_async_fixture_reuses_same_worker_and_callable_for_at_least_three_validated_rounds(
    monkeypatch, async_native, mode, rounds
):
    count = max(3, rounds)
    test, worker, callable_obj, seen = _invoke_fixture(
        monkeypatch, async_native, mode=mode, rounds=rounds, increments=(1,) * count
    )
    assert len(seen) == count
    assert all(actual_worker is worker and actual_callable is callable_obj for actual_worker, actual_callable in seen)
    assert test._contract_publish_count == worker.scheduler_mode_publish_count == 10 + count


@pytest.mark.parametrize("mode", (0, 3, 4, 5, -1))
def test_async_fixture_rejects_nonresident_with_correct_records(monkeypatch, async_native, mode):
    with pytest.raises(AssertionError, match="resident required"):
        _invoke_fixture(monkeypatch, async_native, mode=mode)


@pytest.mark.parametrize("increments", ((0, 1, 1), (-1, 1, 1), (1, 0, 1), (1, 1, 0)))
def test_async_fixture_rejects_stale_publication_on_every_round(monkeypatch, async_native, increments):
    with pytest.raises(AssertionError, match="no fresh scheduler publication"):
        _invoke_fixture(monkeypatch, async_native, increments=increments)


@pytest.mark.parametrize("skip", ("argument", "case", "class"))
def test_async_fixture_cannot_skip_records_or_provenance(monkeypatch, async_native, skip):
    with pytest.raises(AssertionError, match="async boundary acceptance cannot skip"):
        _invoke_fixture(monkeypatch, async_native, skip=skip)


@pytest.mark.parametrize(
    "field,value",
    [
        (12, 1),
        (12, 2),
        (12, 4),
        (12, 8),
        (12, 16),
        (13, 1),
        (14, 1),
        (15, 0),
        (15, -1),
        (0, 2),
        (8, 0),
        (9, 1),
        (10, 2),
        (11, 2),
    ],
    ids=[
        "valid_token",
        "nonnull_count",
        "nonnull_error",
        "nonnull_entries",
        "nonzero_capacity",
        "first_registration_accepted",
        "second_registration_accepted",
        "missing_completion_sentinel",
        "wrong_completion_sentinel",
        "duplicate_execution",
        "wrong_block",
        "wrong_block_count",
        "wrong_slot",
        "wrong_task",
    ],
)
def test_async_fixture_rejects_corrupted_execution_records(monkeypatch, async_native, field, value):
    def corrupt(actual):
        actual.records[field] = value

    with pytest.raises(AssertionError):
        _invoke_fixture(monkeypatch, async_native, corruption=corrupt)


@pytest.mark.parametrize("corruption", ("missing", "inactive", "guard", "fanin"))
def test_async_fixture_keeps_original_identity_guard_and_fanin_checks(monkeypatch, async_native, corruption):
    def corrupt(actual):
        rows = actual.records.view(2, 7, 16)
        if corruption == "missing":
            rows[0, 0] = 0
        elif corruption == "inactive":
            rows[0, 2, 15] = async_native._ASYNC_COMPLETE
        elif corruption == "guard":
            rows[0, -1, 0] = 1
        else:
            actual.summary[1] = 1

    with pytest.raises(AssertionError):
        _invoke_fixture(monkeypatch, async_native, corruption=corrupt)


def test_async_fixture_collects_exactly_fourteen_cases_and_not_parent_matrix(async_native):
    cls = async_native.TestHbgAsyncBoundary
    assert _discover_module_test_classes(async_native) == [cls]
    assert len(cls.CASES) == len({case["name"] for case in cls.CASES}) == 14
    assert {(case["params"]["mask"], case["params"]["blocks"]) for case in cls.CASES} == {
        (mask, blocks) for mask in range(1, 8) for blocks in (1, 2)
    }
    assert all(case["platforms"] == ["a5", "a5sim"] for case in cls.CASES)
    assert all(not ({"skip_golden", "manual", "marks"} & case.keys()) for case in cls.CASES)


def test_async_fixture_reuses_original_orchestration_fanin_and_compile_helper(async_native):
    cls = async_native.TestHbgAsyncBoundary
    base = async_native._spmd.TestHbgSpmdContract
    assert cls.CALLABLE is not base.CALLABLE
    assert cls.CALLABLE["orchestration"] == base.CALLABLE["orchestration"]
    assert cls.CALLABLE["incores"][3] == base.CALLABLE["incores"][3]
    assert cls.compile_chip_callable.__func__ is base.compile_chip_callable.__func__
    for slot in range(3):
        kernel = cls.CALLABLE["incores"][slot]
        assert kernel["func_id"] == slot
        assert kernel["core_type"] == ("aic" if slot == 0 else "aiv")
        assert Path(kernel["source"]) == _FIXTURE_PATH.parent / f"kernels/async_boundary_{slot}.cpp"
        assert Path(base.CALLABLE["incores"][slot]["source"]) == (
            _FIXTURE_PATH.parent / f"kernels/record_slot_{slot}.cpp"
        )
