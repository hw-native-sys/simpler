# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native model groups must be exported once without an artificial sync variant."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
PRODUCTION = Path(__file__).resolve().parents[1] / "production"
SPEC = importlib.util.spec_from_file_location("native_export_cases", PRODUCTION / "export_cases.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("flavor", ["pypto_lib", "pypto_lib_latest"])
@pytest.mark.parametrize("policy", ["half", "all-cap64", "both"])
def test_native_group_has_one_unmodified_variant(monkeypatch, flavor, policy, tmp_path):
    calls = []
    monkeypatch.setattr(MODULE, "_export_variant", lambda *args: calls.append(args))
    variants = MODULE.export(tmp_path, tmp_path, flavor, tmp_path, tmp_path, policy)
    assert variants == [flavor]
    assert len(calls) == 1
    assert calls[0][2:4] == (flavor, flavor)


def test_dsl_still_exports_both_sync_policies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(MODULE, "_export_variant", lambda *args: calls.append(args))
    variants = MODULE.export(tmp_path, tmp_path, "pypto_lib_dsl", tmp_path, tmp_path)
    assert variants == ["pypto_lib_dsl", "pypto_lib_dsl_all_sync64"]
    assert [call[4] for call in calls] == ["half", "all-cap64"]
