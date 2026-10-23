# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only checks of the native fixture; these do not execute any device kernel."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_spmd_contract.py"


@pytest.fixture(scope="module")
def native():
    spec = importlib.util.spec_from_file_location("_hbg_spmd_native_fixture", _FIXTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", [0, 3, 4, 5, -1])
def test_native_gate_rejects_non_resident_even_when_values_are_correct(native, mode):
    worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=2)
    with pytest.raises(AssertionError, match="resident required"):
        native.require_fresh_resident(worker, 1)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("count", [0, 1])
def test_native_gate_rejects_stale_successful_latch(native, mode, count):
    worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=count)
    with pytest.raises(AssertionError, match="no fresh scheduler publication"):
        native.require_fresh_resident(worker, 1)


@pytest.mark.parametrize("mode", [1, 2])
def test_native_gate_accepts_fresh_resident_selection(native, mode):
    worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=2)
    assert native.require_fresh_resident(worker, 1) == 2


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("blocks", [1, 33, 512])
def test_native_golden_includes_nonzero_block_zero_and_untouched_guards(native, mask, blocks):
    test = native.TestHbgSpmdContract()
    params = {"mask": mask, "blocks": blocks, "tasks": 2, "tail_delay": 0}
    args = test.generate_args(params)
    test.compute_golden(args, params)
    records = args.records.view(2, blocks * 3 + 1, 16)
    count = 2 * blocks * bin(mask).count("1")
    assert records[:, :-1, 0].sum().item() == count == args.summary[0].item()
    assert torch.count_nonzero(records[:, -1, :]).item() == 0
    assert torch.count_nonzero(records[:, :, 1:8]).item() == 0
    for slot in range(3):
        if mask & (1 << slot):
            assert records[0, slot, 8].item() == 1
            assert records[1, (blocks - 1) * 3 + slot, 8:12].tolist() == [blocks, blocks, slot + 1, 2]


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "wrong_n", "guard", "premature_fanin"])
def test_native_comparator_rejects_each_distinct_failure(native, corruption):
    test = native.TestHbgSpmdContract()
    params = {"mask": 7, "blocks": 2, "tasks": 2, "tail_delay": 0}
    golden = test.generate_args(params)
    test.compute_golden(golden, params)
    actual = golden.clone()
    if corruption == "duplicate":
        actual.records[0] = 2
    elif corruption == "missing":
        actual.records[0:16] = 0
    elif corruption == "wrong_n":
        actual.records[9] = 1
    elif corruption == "guard":
        actual.records[2 * 3 * 16] = 1
    else:
        actual.summary[1] = 1
    test._contract_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    test._contract_publish_count = 1
    with pytest.raises(AssertionError):
        test.compare_outputs(actual, golden, ["records", "summary"], params)


def test_native_matrix_has_unique_names_and_no_silent_skip_or_xfail(native):
    cases = native.TestHbgSpmdContract.CASES
    assert len(cases) == len({case["name"] for case in cases}) == 90
    assert all("skip_golden" not in case and "manual" not in case and "marks" not in case for case in cases)
