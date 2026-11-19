# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bad-record sensitivity for the native rotating-publisher litmus."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def native():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_admission_signal_publication.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_admission_signal_native_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize("case", range(37))
def test_each_rotating_writer_case_has_exact_full_width_golden(native, case):
    assert len(native.TestHbgAdmissionSignalPublication.CASES) == 37
    params = native.TestHbgAdmissionSignalPublication.CASES[case]["params"]
    probe = native.TestHbgAdmissionSignalPublication()
    args = probe.generate_args(params)
    probe.compute_golden(args, params)
    native.validate_probe(args.controls, args.checks, params)


@pytest.mark.parametrize("platform,clusters", (("a5", 32), ("a5sim", 8)))
def test_shared_sync_start_fits_platform_and_exercises_all_positions(native, platform, clusters):
    cases = [case for case in native.TestHbgAdmissionSignalPublication.CASES if platform in case["platforms"]]
    assert len(cases) == 29
    shared = [case["params"] for case in cases if case["params"]["shared"]]
    assert {params["blocks"] for params in shared} == {1, 2, clusters}
    assert {params["delayed_rank"] for params in shared if params["blocks"] == clusters} == {
        -1,
        0,
        1,
        2,
        clusters * 3 - 1,
    }


@pytest.mark.parametrize("word", range(176))
def test_every_control_and_guard_word_is_checked(native, word):
    params = dict(blocks=1, rounds=1, shared=0, epoch=-1)
    controls, checks = native.expected_probe(params)
    assert controls.numel() == 176
    controls[word] ^= 1
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("word", range(64))
def test_every_lane_record_and_guard_is_checked(native, word):
    params = dict(blocks=1, rounds=1, shared=0, epoch=-1)
    controls, checks = native.expected_probe(params)
    checks[word] ^= 1
    with pytest.raises(AssertionError):
        native.validate_probe(controls, checks, params)


@pytest.mark.parametrize("mode,count", ((3, 2), (4, 2), (0, 2), (2, 1)))
def test_real_comparator_requires_fresh_resident_provenance(native, mode, count):
    params = dict(blocks=1, rounds=1, shared=1, epoch=0)
    controls, checks = native.expected_probe(params)
    probe = native.TestHbgAdmissionSignalPublication()
    probe._contract_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=count)
    probe._contract_publish_count = 1
    with pytest.raises(AssertionError):
        probe.compare_outputs(SimpleNamespace(controls=controls, checks=checks), None, [], params)


@pytest.mark.parametrize("kwargs", ({"skip_golden": True}, {"case": {"skip_golden": True}}))
def test_probe_cannot_skip_its_visibility_oracle(native, kwargs):
    with pytest.raises(AssertionError):
        native.TestHbgAdmissionSignalPublication()._run_and_validate(None, **kwargs)
