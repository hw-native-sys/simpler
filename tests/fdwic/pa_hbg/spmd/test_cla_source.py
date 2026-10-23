# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit provenance gate against the user-supplied capture; no device is used."""

import importlib
from pathlib import Path


def test_cla_original_capture_matches_frozen_workload(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    source = root / "examples/a5/cla-swimlane.json"
    assert source.is_file(), "original examples/a5/cla-swimlane.json required; source validation cannot be skipped"
    monkeypatch.syspath_prepend(str(root / "tests/ut/py"))
    module = importlib.import_module("hbg_spmd_contracts.cla_trace")
    assert module.load_capture(source) == module.load_fixture()
