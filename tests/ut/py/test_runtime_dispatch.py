# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools._runtime_dispatch."""

from __future__ import annotations

import pytest

from simpler_setup.tools import _runtime_dispatch as rd
from simpler_setup.tools import hbg, tmr


def test_get_resolves_the_named_runtime_to_its_package():
    assert rd.get(rd.HBG_RUNTIME) is hbg
    assert rd.get(rd.TMR_RUNTIME) is tmr


@pytest.mark.parametrize(
    "runtime_name",
    [
        None,  # a capture from before the name existed
        "",
        "   ",
        "host_build_grpah",  # a typo, one letter from the real thing
        "future_runtime",
    ],
)
def test_a_runtime_this_tool_cannot_decode_is_refused(runtime_name):
    """Guessing a layout yields labels that read as valid and are wrong.

    An hbg sub-task decoded as tmr becomes a plausible `r3t5` with a billion-scale
    ring, so every entry point refuses the name rather than picking a default.
    """
    with pytest.raises(ValueError, match="runtime"):
        rd.resolve_runtime(runtime_name)
    with pytest.raises(ValueError, match="runtime"):
        rd.get(runtime_name)


def test_resolve_runtime_names_its_source_in_the_message():
    with pytest.raises(ValueError, match=r"deps\.json: runtime is missing"):
        rd.resolve_runtime(None, source="deps.json: runtime")


def test_normalize_task_id_int_accepts_ints_and_numeric_strings():
    assert rd.normalize_task_id_int(12) == 12
    assert rd.normalize_task_id_int("12") == 12
    assert rd.normalize_task_id_int(None) is None
    assert rd.normalize_task_id_int("not-a-number") is None


def test_normalize_task_id_int_folds_negative_values_to_unsigned_64_bit():
    assert rd.normalize_task_id_int(-1) == (1 << 64) - 1
    assert rd.normalize_task_id_int(-2) == (1 << 64) - 2


def test_raw_halves_splits_at_bit_32_without_decoding():
    raw = (0xDEAD << 32) | 0xBEEF
    assert rd.raw_halves(raw) == (0xDEAD, 0xBEEF)
    # Two ids differing only above bit 32 split into two distinct high halves --
    # this must never collapse them, unlike a layout-aware decode would.
    assert rd.raw_halves(raw) != rd.raw_halves(raw | (1 << 52))
