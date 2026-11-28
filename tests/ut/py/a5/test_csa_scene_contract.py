# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""The actual CSA scene must retain its workload, numeric oracle and fresh provenance gate."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from simpler_setup import TaskArgsBuilder, TensorArg


@pytest.fixture(scope="module")
def csa():
    path = Path(__file__).resolve().parents[3] / "support/a5/host_build_graph/csa/business.py"
    spec = importlib.util.spec_from_file_location("_csa_business_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def oracle_pass(monkeypatch, csa):
    calls = []
    monkeypatch.setattr(csa.original.TestDeepSeekV4ProCSAHBG, "compare_outputs", lambda *args: calls.append(args))
    return calls


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
def test_wrong_mode_is_rejected_after_original_seven_output_validation(csa, monkeypatch, mode):
    calls = oracle_pass(monkeypatch, csa)
    test = csa.TestCSABusiness()
    test._csa_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=2)
    test._csa_publish_count = 1
    with pytest.raises(AssertionError, match="HBG calendar is required"):
        test.compare_outputs(None, None, [], {})
    assert len(calls) == 1


@pytest.mark.parametrize("fresh", [True, False])
def test_each_invocation_requires_its_own_calendar_publication(csa, monkeypatch, fresh):
    oracle_pass(monkeypatch, csa)
    test = csa.TestCSABusiness()
    test._csa_worker = SimpleNamespace(scheduler_runtime_mode=5, scheduler_mode_publish_count=2 if fresh else 1)
    test._csa_publish_count = 1
    if fresh:
        test.compare_outputs(None, None, [], {})
        assert test._csa_publish_count == 2
        with pytest.raises(AssertionError, match="fresh scheduler"):
            test.compare_outputs(None, None, [], {})
    else:
        with pytest.raises(AssertionError, match="fresh scheduler"):
            test.compare_outputs(None, None, [], {})


def test_scheduler_gate_cannot_hide_numerical_failure(csa, monkeypatch):
    def reject(*args):
        raise AssertionError("numerical sentinel")

    monkeypatch.setattr(csa.original.TestDeepSeekV4ProCSAHBG, "compare_outputs", reject)
    test = csa.TestCSABusiness()
    with pytest.raises(AssertionError, match="numerical sentinel"):
        test.compare_outputs(None, None, [], {})


def test_business_contract_cannot_skip_its_goldens(csa):
    with pytest.raises(AssertionError, match="cannot skip"):
        csa.TestCSABusiness()._run_and_validate(None, skip_golden=True)


def test_real_kernels_and_full_abi_are_preserved(csa):
    scene = csa.TestCSABusiness
    assert scene.CALLABLE == csa.original.TestDeepSeekV4ProCSAHBG.CALLABLE
    assert scene._st_runtime == "host_build_graph"
    assert len(scene.CALLABLE["incores"]) == 54
    assert len(scene.CALLABLE["orchestration"]["signature"]) == 52
    assert scene.SKIP_GOLDEN is False
    assert len(scene.CASES) == len({case["name"] for case in scene.CASES}) == 12
    assert all("skip_golden" not in case and "marks" not in case for case in scene.CASES)


def test_actual_l2_round_loop_restores_all_seven_buffers_before_each_invocation(csa, monkeypatch):
    framework = sys.modules["simpler_setup.scene_test"]
    names = list(csa.original.golden.comparators())
    test = csa.TestCSABusiness()
    rounds = []
    comparisons = []

    def initial_args(params):
        return TaskArgsBuilder(*(TensorArg(name, torch.full((4,), float(i))) for i, name in enumerate(names)))

    def compute(args, params):
        for i, name in enumerate(names):
            getattr(args, name).fill_(100 + i)

    def run(handle, args, config):
        for i, name in enumerate(names):
            assert torch.equal(getattr(args, name), torch.full((4,), float(i))), (len(rounds), name)
        compute(args, {})
        rounds.append(handle)
        worker.scheduler_mode_publish_count += 1

    def compare(self, actual, expected, output_names, params):
        assert output_names == names
        for name in names:
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        comparisons.append(len(rounds))

    worker = SimpleNamespace(register=lambda obj: 42, run=run, scheduler_mode_publish_count=0, scheduler_runtime_mode=5)
    monkeypatch.setattr(test, "generate_args", initial_args)
    monkeypatch.setattr(test, "compute_golden", compute)
    monkeypatch.setattr(test, "_build_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(csa.original.TestDeepSeekV4ProCSAHBG, "compare_outputs", compare)
    monkeypatch.setattr(framework, "_child_memory_args", lambda *args: nullcontext(SimpleNamespace(tensors={})))
    monkeypatch.setattr(framework, "_build_l2_ref_args", lambda args, *rest, **kwargs: (args, names))
    test._run_and_validate(worker, object(), {"name": "reset"}, rounds=3)
    assert rounds == [42, 42, 42]
    assert comparisons == [1, 2, 3]
