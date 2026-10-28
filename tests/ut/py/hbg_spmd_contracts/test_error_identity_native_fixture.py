# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent bad-record checks for the native first-error publication probe."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def native():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_error_identity.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_error_identity_native_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize("case", range(50))
def test_each_native_case_has_complete_golden_and_private_guards(native, case):
    assert len(native.TestHbgErrorIdentity.CASES) == 50
    params = native.TestHbgErrorIdentity.CASES[case]["params"]
    probe = native.TestHbgErrorIdentity()
    args = probe.generate_args(params)
    probe.compute_golden(args, params)
    native.validate_probe(args.controls, args.checks, params)


@pytest.mark.parametrize("shared", (0, 1))
@pytest.mark.parametrize("word", range(96))
def test_every_identity_and_guard_word_is_checked(native, shared, word):
    params = dict(lane=1, blocks=65, shared=shared, epoch=-1, scenario=0)
    controls, checks = native.expected_probe(params)
    controls[word] ^= 1
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("block", range(65))
def test_missing_or_partial_block_observation_is_rejected(native, block):
    params = dict(lane=0, blocks=65, shared=1, epoch=0, scenario=0)
    controls, checks = native.expected_probe(params, [64])
    checks.view(65, 16)[block, 0] = 0
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("winner", (0, 1, 31, 32, 63, 64))
def test_race_allows_any_valid_winner_but_not_mixed_winner_fields(native, winner):
    params = dict(lane=1, blocks=65, shared=1, epoch=61, scenario=0)
    controls, checks = native.expected_probe(params, [winner])
    native.validate_probe(controls, checks, params)
    controls[45] += 1
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("scenario", range(5))
def test_partial_identity_validity_bits_are_not_optional(native, scenario):
    params = dict(lane=0, blocks=1, shared=0, epoch=-1, scenario=scenario)
    controls, checks = native.expected_probe(params)
    controls[47] &= (1 << 32) - 1
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("mode,count", ((3, 2), (4, 2), (0, 2), (2, 1)))
def test_real_comparator_rejects_fallback_or_stale_invocation(native, mode, count):
    params = dict(lane=0, blocks=1, shared=0, epoch=0, scenario=0)
    controls, checks = native.expected_probe(params)
    probe = native.TestHbgErrorIdentity()
    probe._error_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=count)
    probe._error_publish_count = 1
    with pytest.raises(AssertionError):
        probe.compare_outputs(SimpleNamespace(controls=controls, checks=checks), None, [], params)


@pytest.mark.parametrize("kwargs", ({"skip_golden": True}, {"case": {"skip_golden": True}}))
def test_native_probe_cannot_skip_its_oracle(native, kwargs):
    with pytest.raises(AssertionError):
        native.TestHbgErrorIdentity()._run_and_validate(None, **kwargs)


@pytest.mark.parametrize("winner", (0, 1))
@pytest.mark.parametrize("field", ("arrival", "missing_lane", "wrong_lane"))
def test_mixed_race_requires_two_actual_participants_and_the_winners_lane(native, winner, field):
    params = dict(lane=2, blocks=2, shared=1, epoch=-1, scenario=5)
    controls, checks = native.expected_probe(params, [winner])
    native.validate_probe(controls, checks, params)
    if field == "arrival":
        controls[64] = 1
    elif field == "missing_lane":
        checks.view(2, 16)[1, 0] = 0
    else:
        controls[47] ^= 1
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("scenario", (6, 7))
@pytest.mark.parametrize("fault", ("wrong_writer", "missing_status", "missing_notice"))
def test_directed_publication_requires_status_after_the_designated_writer_returns(native, scenario, fault):
    params = dict(lane=2, blocks=2, shared=1, epoch=61, scenario=scenario)
    controls, checks = native.expected_probe(params)
    native.validate_probe(controls, checks, params)
    if fault == "wrong_writer":
        controls, checks = native.expected_probe(params, [1 - (scenario - 6)])
    elif fault == "missing_status":
        controls[48] = 0
    else:
        controls[65] = 0
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)
