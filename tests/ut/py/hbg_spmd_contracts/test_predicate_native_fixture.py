# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only checks of predicate acceptance fixtures; mocked modes are not device evidence."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpler_setup import SceneTestCase

_FIXTURES = Path(__file__).resolve().parents[3] / "st/a5/host_build_graph"


def _load_fixture(name, directory, filename):
    spec = importlib.util.spec_from_file_location(name, _FIXTURES / directory / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def predicate_native(pytestconfig):
    pytestconfig.addinivalue_line("markers", "scheduler_mode(mode): existing HBG runtime selection contract")
    return _load_fixture("_hbg_predicate_native_fixture", "predicated_dispatch", "test_predicated_dispatch.py")


@pytest.fixture(scope="module")
def predicate_refusal():
    return _load_fixture("_hbg_predicate_refusal_fixture", "mix_predicated_dispatch", "test_mix_predicated_dispatch.py")


@pytest.fixture(params=(1, 2), ids=("predicate_false", "predicate_true"))
def predicate_case(request):
    return request.param


def _invoke_fixture(monkeypatch, native, case, mode, increments, *, corrupt=False, skip_golden=False):
    test = native.TestPredicatedDispatchA5()
    params = {"case": case}
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    actual = golden.clone()
    if corrupt:
        actual.y[0] += 1
    worker = SimpleNamespace(scheduler_mode_publish_count=10, scheduler_runtime_mode=mode)

    def fake_device_rounds(self, supplied_worker, *args, **kwargs):
        assert supplied_worker is worker
        for increment in increments:
            worker.scheduler_mode_publish_count += increment
            self.compare_outputs(actual, golden, ["x", "y", "gate"], params)

    monkeypatch.setattr(SceneTestCase, "_run_and_validate", fake_device_rounds)
    test._run_and_validate(worker, skip_golden=skip_golden)
    return test, worker


@pytest.mark.parametrize("mode,increments", [(1, (1,)), (2, (1, 1, 1))])
def test_predicate_fixture_accepts_fresh_resident_rounds(
    monkeypatch, predicate_native, predicate_case, mode, increments
):
    test, worker = _invoke_fixture(monkeypatch, predicate_native, predicate_case, mode, increments)
    assert test._selection_publish_count == worker.scheduler_mode_publish_count == 10 + sum(increments)


@pytest.mark.parametrize("mode", [0, 3, 4, 5, -1])
def test_predicate_fixture_rejects_other_modes_with_correct_golden(monkeypatch, predicate_native, predicate_case, mode):
    with pytest.raises(AssertionError, match="requires resident scheduling"):
        _invoke_fixture(monkeypatch, predicate_native, predicate_case, mode, (1,))


@pytest.mark.parametrize("increments", [(0,), (-1,), (1, 0)], ids=("unchanged", "decreased", "stale_second_round"))
def test_predicate_fixture_rejects_stale_selection_each_round(
    monkeypatch, predicate_native, predicate_case, increments
):
    with pytest.raises(AssertionError, match="no fresh scheduler selection"):
        _invoke_fixture(monkeypatch, predicate_native, predicate_case, 2, increments)


def test_predicate_fixture_still_rejects_wrong_values(monkeypatch, predicate_native, predicate_case):
    with pytest.raises(AssertionError):
        _invoke_fixture(monkeypatch, predicate_native, predicate_case, 2, (1,), corrupt=True)


def test_predicate_fixture_cannot_skip_value_and_provenance_checks(monkeypatch, predicate_native, predicate_case):
    with pytest.raises(AssertionError, match="requires outputs and fresh resident provenance"):
        _invoke_fixture(monkeypatch, predicate_native, predicate_case, 2, (1,), skip_golden=True)


def test_predicate_refusal_matrix_covers_mix_and_pure_multiblock(predicate_refusal):
    expected = {(mask, blocks) for mask in (3, 5, 6, 7) for blocks in (1, 2)} | {(mask, 2) for mask in (1, 2, 4)}
    assert len(predicate_refusal.REFUSED_SHAPES) == len(expected) == 11
    assert set(predicate_refusal.REFUSED_SHAPES) == expected
    marks = {
        mark.args[0]: mark.args[1]
        for mark in predicate_refusal.test_predicated_cooperative_task_is_refused.pytestmark
        if mark.name == "parametrize"
    }
    assert set(marks["mask,blocks"]) == expected
    assert tuple(marks["gate"]) == (0, 1)
