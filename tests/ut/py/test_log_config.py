# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for simpler_setup.log_config + simpler._log integer-threshold logic."""

import logging

import pytest
from simpler._log import _split_threshold

from simpler_setup.log_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_CHOICES,
    configure_logging,
    parse_level,
)


def test_default_level_is_v5():
    assert DEFAULT_LOG_LEVEL == "v5"
    assert parse_level("v5") == 20  # = logging.INFO
    assert parse_level("info") == 20


def test_choices_cover_severity_and_v_tiers():
    for name in ("debug", "info", "warn", "error", "null"):
        assert name in LOG_LEVEL_CHOICES
    for v in range(10):
        assert f"v{v}" in LOG_LEVEL_CHOICES


def test_null_mutes_severity():
    configure_logging("null")
    sim = logging.getLogger("simpler")
    assert sim.getEffectiveLevel() >= 60


def test_v0_lets_everything_through():
    configure_logging("v0")
    sim = logging.getLogger("simpler")
    # V0 = 15, so DEBUG-tagged records are still suppressed by Python (DEBUG=10),
    # but every V tier and INFO/WARN/ERROR pass.
    assert sim.getEffectiveLevel() == 15


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (10, (0, 0)),  # DEBUG → severity DEBUG
        (15, (1, 0)),  # V0    → severity INFO, info_v=0
        (20, (1, 5)),  # V5    → severity INFO, info_v=5 (default)
        (24, (1, 9)),  # V9    → severity INFO, info_v=9
        (30, (2, 0)),  # WARN  → severity WARN
        (40, (3, 0)),  # ERROR → severity ERROR
        (60, (4, 0)),  # NUL   → severity NUL
    ],
)
def test_split_threshold(threshold, expected):
    assert _split_threshold(threshold) == expected
