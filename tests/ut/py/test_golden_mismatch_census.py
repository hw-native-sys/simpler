# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What a golden mismatch tells the reader.

The census exists because `max_diff` alone is ambiguous between two situations
that call for opposite responses: a tolerance that is too tight for the format,
and a handful of elements that are outright wrong. Each test below is one of
those shapes.
"""

import pytest

torch = pytest.importorskip("torch")

from simpler_setup.scene_test import _describe_mismatch  # noqa: E402


def test_every_element_slightly_off_reports_as_such():
    expected = torch.full((64,), 1.0)
    actual = expected + 0.05
    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.01)

    assert "elems=64" in report
    assert "differ=64" in report
    assert "over_tol=64" in report


def test_three_bad_elements_among_many_correct_ones_are_countable():
    # The a2a3sim fp16 shape: almost everything matches, a few elements are
    # exactly half their golden. max_diff alone cannot tell this from the case
    # above, and only one of the two is a tolerance question.
    expected = torch.full((4096,), 0.125)
    actual = expected.clone()
    actual[7] = 0.0625
    actual[1000] = 0.0625
    actual[4095] = 0.0625
    report = _describe_mismatch(actual, expected, rtol=0.005, atol=0.02)

    assert "elems=4096" in report
    assert "differ=3" in report
    assert "over_tol=3" in report
    # The worst offenders carry their index and both values, so the 2x
    # relationship is readable without re-running under a patch.
    assert "0.0625 vs 0.125" in report


def test_within_tolerance_but_not_equal_separates_differ_from_over_tol():
    expected = torch.full((100,), 1.0)
    actual = expected.clone()
    actual[:40] += 0.001  # differs, but inside atol
    actual[40] += 0.5  # the only real failure
    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.01)

    assert "differ=41" in report
    assert "over_tol=1" in report


def test_max_diff_stays_first_and_unprefixed():
    # Triage notes and logs quote `max_diff=`; keep it leading so they still read.
    expected = torch.zeros(4)
    actual = torch.tensor([0.0, 0.0, 2.0, 0.0])
    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.0)

    assert report.startswith("max_diff=2.0, rtol=0.0, atol=0.0")


def test_non_finite_elements_are_counted_not_hidden():
    # A single inf makes max_diff inf and every other number meaningless; say how
    # many there are rather than leaving the reader with one useless figure.
    expected = torch.zeros(8)
    actual = expected.clone()
    actual[3] = float("inf")
    actual[5] = float("nan")
    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.0)

    assert "non_finite=2" in report
    # max_diff comes from the finite elements, so it is not swallowed by the inf.
    assert "max_diff=0.0," in report


def test_multidimensional_indices_are_reported_as_tuples():
    expected = torch.zeros(2, 3, 4)
    actual = expected.clone()
    actual[1, 2, 3] = 1.0
    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.0)

    assert "(1, 2, 3)=1.0 vs 0.0" in report


def test_imaginary_only_mismatch_is_not_reported_as_zero():
    # torch.allclose decides whether this census is printed at all, and it
    # accepts complex. Widening with .float() instead would discard the
    # imaginary part behind a warning and report max_diff=0.0 on a comparison
    # that had just failed.
    expected = torch.tensor([1 + 0j, 2 + 0j], dtype=torch.complex64)
    actual = torch.tensor([1 + 1j, 2 + 0j], dtype=torch.complex64)
    assert not torch.allclose(actual, expected, rtol=0.0, atol=0.0)

    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.0)

    assert "max_diff=1.0" in report
    assert "max_diff=0.0" not in report
    assert "over_tol=1" in report
    # The offender keeps both components rather than being coerced to a float.
    assert "1j" in report


def test_integer_offenders_are_reported_as_integers():
    expected = torch.zeros(4, dtype=torch.int32)
    actual = expected.clone()
    actual[2] = 7
    report = _describe_mismatch(actual, expected, rtol=0.0, atol=0.0)

    assert "(2,)=7 vs 0" in report
