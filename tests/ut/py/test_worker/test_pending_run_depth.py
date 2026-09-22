# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`Worker(pending_run_depth=...)`: the public surface of the logical admission queue.

`pending_run_depth` bounds how many non-terminal logical runs a level >= 3 Worker admits. It is a
separate budget from the native pipeline-slot depth negotiated with the chip children: zero derives
it from that depth, so an unconfigured Worker admits exactly what it did before the two separated.

Validation belongs to the constructor, before any fork, shared-memory mapping or C++ configuration
happens, so these cases construct a Worker and never start it.
"""

import pytest
from simpler.worker import _PENDING_RUN_DEPTH_MAX, Worker, _validated_pending_run_depth


def test_absent_key_derives_from_the_negotiated_depth():
    # Zero is the "derive" sentinel the C++ side reads; an unconfigured Worker
    # must carry it rather than a guessed positive cap.
    assert Worker(level=3)._pending_run_depth == 0
    assert _validated_pending_run_depth({}, 3) == 0


@pytest.mark.parametrize("depth", [0, 1, 2, 4, 64])
def test_non_negative_ints_are_accepted(depth):
    assert Worker(level=3, pending_run_depth=depth)._pending_run_depth == depth


def test_below_the_native_depth_is_a_legal_configuration():
    # It serializes rather than failing: the successor role needs a second FIFO
    # entry, and a cap of one denies it.
    assert Worker(level=3, pending_run_depth=1)._pending_run_depth == 1


@pytest.mark.parametrize("bad", [-1, -4])
def test_negative_depth_is_rejected(bad):
    with pytest.raises(ValueError, match="pending_run_depth must be >= 0"):
        Worker(level=3, pending_run_depth=bad)


def test_the_uint32_boundary_value_is_accepted():
    assert Worker(level=3, pending_run_depth=_PENDING_RUN_DEPTH_MAX)._pending_run_depth == _PENDING_RUN_DEPTH_MAX


@pytest.mark.parametrize("bad", [2**32, 2**32 + 1, 2**64])
def test_values_above_uint32_are_rejected_before_any_side_effect(bad):
    # The C++ budget is a uint32_t. Without this the constructor accepts the
    # value and nanobind conversion fails at init, after children are forked
    # and shared memory is mapped.
    with pytest.raises(ValueError, match="pending_run_depth must be <="):
        Worker(level=3, pending_run_depth=bad)


@pytest.mark.parametrize("bad", [1.0, "2", None, [2]])
def test_non_int_depth_is_rejected(bad):
    with pytest.raises(ValueError, match="pending_run_depth must be an int"):
        Worker(level=3, pending_run_depth=bad)


@pytest.mark.parametrize("bad", [True, False])
def test_bool_is_rejected_rather_than_read_as_one_or_zero(bad):
    # bool is an int subclass, so True would silently mean a cap of one.
    with pytest.raises(ValueError, match="pending_run_depth must be an int"):
        Worker(level=3, pending_run_depth=bad)


@pytest.mark.parametrize("level", [1, 2])
def test_unsupported_below_level_three(level):
    # A level < 3 Worker owns no admission FIFO — its runs belong to the chip
    # child — so the key is refused rather than silently ignored.
    with pytest.raises(ValueError, match="requires a level >= 3 Worker"):
        Worker(level=level, pending_run_depth=4)


def test_level_two_without_the_key_is_unaffected():
    assert Worker(level=2)._pending_run_depth == 0


def test_validation_precedes_worker_side_effects():
    # The constructor is the rejection point, so a bad value never reaches a
    # fork, a shared-memory mapping or the C++ orchestrator.
    with pytest.raises(ValueError):
        Worker(level=3, device_ids=[0], pending_run_depth=-1)
