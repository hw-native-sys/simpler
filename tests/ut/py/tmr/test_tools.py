# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools.tmr (display, task_id_fields, scope_key, make_formatter)."""

from __future__ import annotations

import pytest
from _task_ids import tmr_task as _id

from simpler_setup.tools import tmr


def test_ring_refuses_a_value_no_mint_can_produce():
    """A ring wider than its field is a corrupt record, not a large ring.

    C++ narrows with ``static_cast<uint8_t>``, so the invalid sentinel would read back
    as ring 255 and render as a plausible task. Reporting the raw id instead is what
    keeps a corrupt record from being drawn as a real one.
    """
    with pytest.raises(ValueError, match="ring"):
        tmr.display((1 << 64) - 1)
    with pytest.raises(ValueError, match="ring"):
        tmr.task_id_fields(1 << 40)


def test_mint_and_decode_read_the_same_field():
    """The mirror composes an id from the constants it decodes one with.

    ``test_task_id_layout`` compares those constants against task_id.h; this is what
    keeps a mint from placing the field somewhere its own decode does not look.
    """
    minted = tmr.TaskId.make(3, 1234)
    assert minted.ring() == 3
    assert minted.local_id() == 1234
    assert minted.raw == _id(3, 1234)


def test_mint_truncates_an_over_wide_field_the_way_cpp_does():
    """``make`` takes a uint8_t ring and a uint32_t local id, so a wider value is cut."""
    assert tmr.TaskId.make(0x1FF, 0).ring() == tmr.TaskId.RING_MASK
    assert tmr.TaskId.make(0, 1 << 32).local_id() == 0


def test_display_keeps_the_ring_form():
    """Every tmr label carries the ring that scopes its local task id."""
    assert tmr.display(_id(0, 0)) == "r0t0"
    assert tmr.display(_id(0, 100)) == "r0t100"
    assert tmr.display(_id(2, 100)) == "r2t100"


def test_display_falls_back_to_str_for_a_value_that_is_not_an_integer():
    assert tmr.display("not-a-number") == "not-a-number"


def test_task_id_fields_carries_ring_id():
    assert tmr.task_id_fields(_id(2, 100)) == {"ring_id": 2}


def test_scope_key_groups_by_ring():
    """One ring per scope depth, so a differing ring is a differing scope."""
    assert tmr.scope_key(_id(1, 7)) == tmr.scope_key(_id(1, 9))
    assert tmr.scope_key(_id(1, 0)) != tmr.scope_key(_id(2, 0))


def test_make_formatter_never_compacts_even_when_every_ring_is_zero():
    """Ring 0 is a ring, not an absence of one -- unlike hbg's GLOBAL space."""
    fmt = tmr.make_formatter([_id(0, 1), _id(0, 2), _id(0, 3)])
    assert [fmt(_id(0, n)) for n in (1, 2, 3)] == ["r0t1", "r0t2", "r0t3"]


def test_display_disagrees_with_hbgs_reading_of_the_same_word():
    """A tmr id on ring 1 has bit 32 set; hbg would read that as a GLOBAL task whose
    parent field happens to be 1 (see hbg's own test of the same raw value).
    Neither decoder can detect the other's value, which is why the runtime name
    the document carries is what picks the layout, not anything in the id itself.
    """
    assert tmr.display(_id(1, 9)) == "r1t9"


def test_this_runtime_contributes_no_lane_of_its_own():
    """Empty is tmr's real answer, not an unimplemented stub.

    The lane hbg contributes draws modular-task expansion, which tmr has no
    counterpart for. Everything tmr shows is drawn by the common converter from
    records both runtimes emit, so it adds nothing here even for a capture full
    of its own data.
    """
    tasks = [{"task_id": _id(0, 1), "start_time_us": 1.0, "end_time_us": 2.0}]
    phases = [[{"phase": "resolve", "task_id": _id(0, 1), "start_time_us": 1.0, "end_time_us": 2.0}]]

    assert tmr.contribute_events(tasks, phases, ctx=None) == []
