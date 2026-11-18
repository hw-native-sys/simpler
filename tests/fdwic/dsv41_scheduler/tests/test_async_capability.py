# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Benchmark declarations share the public completion-capability boundary."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
SPEC = importlib.util.spec_from_file_location(
    "benchmark_build_sources", Path(__file__).resolve().parents[1] / "production/build_sources.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_none_and_unknown_normalize_to_full_deferred_capacity():
    assert MODULE.capability_words({}) == (1, 2, 64, 3)
    assert MODULE.capability_words({"async_capability": None}) == (1, 2, 64, 3)
    assert MODULE.capability_words(
        dict(async_capability=dict(version=1, mode="UNKNOWN", max_conditions=0, condition_types=0))
    ) == (1, 2, 64, 3)


def test_malformed_declarations_are_rejected():
    for value in (
        {},
        dict(version=2, mode="UNKNOWN", max_conditions=0, condition_types=0),
        dict(version=1, mode="UNKNOWN", max_conditions=1, condition_types=0),
        dict(version=1, mode="SYNCHRONOUS", max_conditions=1, condition_types=0),
    ):
        with pytest.raises(ValueError):
            MODULE.capability_words({"async_capability": value})


def test_two_explicit_modes_preserve_the_declared_capacity():
    assert MODULE.capability_words(
        dict(async_capability=dict(version=1, mode="SYNCHRONOUS", max_conditions=0, condition_types=0))
    ) == (1, 1, 0, 0)
    assert MODULE.capability_words(
        dict(async_capability=dict(version=1, mode="DEFERRED", max_conditions=2, condition_types=1))
    ) == (1, 2, 2, 1)
    with pytest.raises(ValueError):
        MODULE.capability_words(
            dict(async_capability=dict(version=1, mode="DEFERRED", max_conditions=0, condition_types=1))
        )
