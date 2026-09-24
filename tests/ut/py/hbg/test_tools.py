# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools.hbg (display, task_id_fields, scope_key, make_formatter, graph lane)."""

from __future__ import annotations

import pytest
from _task_ids import hbg_global as _global
from _task_ids import hbg_param as _param
from _task_ids import hbg_sub_task as _sub_task

from simpler_setup.tools import hbg
from simpler_setup.tools.hbg import graph_lane


def test_display_names_the_space_it_decodes():
    """Each id space gets its own label shape.

    A SUB_TASK label has to carry the parent too: the low field is only an index
    within one body, so two modular tasks replaying one Definition hold the same low
    field for their respective first sub-task.
    """
    assert hbg.display(_global(12)) == "t12"
    assert hbg.display(_sub_task(3, 0)) == "g3t0"
    assert hbg.display(_sub_task(7, 5)) == "g7t5"
    assert hbg.display(_param(2)) == "p2"
    assert hbg.display(_sub_task(3, 0)) != hbg.display(_sub_task(4, 0))


def test_display_falls_back_to_str_for_a_value_that_is_not_an_integer():
    assert hbg.display("not-a-number") == "not-a-number"


def test_display_raises_on_a_corrupt_id_space_rather_than_a_plausible_label():
    """The invalid sentinel (and any id sharing its top two bits) must not render as GLOBAL."""
    with pytest.raises(ValueError, match="id space"):
        hbg.display((1 << 64) - 1)


def test_display_disagrees_with_tmrs_reading_of_the_same_word():
    """A word with bit 32 set is a GLOBAL task whose parent field happens to be 1 --
    tmr would read the same word as ring 1 (see tmr's own test of the same raw
    value). Neither decoder can detect the other's value, which is why the runtime
    name the document carries is what picks the layout, not anything in the id itself.
    """
    assert hbg.display((1 << 32) | 9) == "t9"


def test_task_id_fields_carries_id_space_and_parent_only_for_a_sub_task():
    assert hbg.task_id_fields(_global(12)) == {"id_space": 0}
    assert hbg.task_id_fields(_sub_task(3, 0)) == {"id_space": 1, "parent_task_id": 3}
    assert hbg.task_id_fields(_param(2)) == {"id_space": 2}


def test_mint_and_decode_read_the_same_fields():
    """The mirror composes an id from the constants it decodes one with.

    ``test_task_id_layout`` compares those constants against task_id.h; this is what
    keeps a mint from placing a field somewhere its own decode does not look.
    """
    minted = hbg.TaskId.make_sub_task(7, 1234)
    assert minted.space() is hbg.Space.SUB_TASK
    assert minted.parent_id() == 7
    assert minted.local_id() == 1234
    assert minted.raw == _sub_task(7, 1234)


def test_mint_truncates_an_over_wide_field_the_way_cpp_does():
    """``make_sub_task`` masks the parent to PARENT_BITS and narrows the local id to 32 bits."""
    assert hbg.TaskId.make_sub_task((1 << hbg.TaskId.PARENT_BITS) | 5, 0).parent_id() == 5
    assert hbg.TaskId.make_global(1 << 32).local_id() == 0


def test_scope_key_groups_by_id_space_and_parent():
    """Tasks of the run itself share one scope; each modular task's body is its own."""
    assert hbg.scope_key(_global(1)) == hbg.scope_key(_global(2))
    assert hbg.scope_key(_sub_task(3, 0)) == hbg.scope_key(_sub_task(3, 9))
    assert hbg.scope_key(_sub_task(3, 0)) != hbg.scope_key(_sub_task(4, 0))
    assert hbg.scope_key(_global(1)) != hbg.scope_key(_sub_task(0, 1))


def test_make_formatter_compacts_only_when_every_id_is_global():
    fmt = hbg.make_formatter([_global(1), _global(2), _global(3)])
    assert [fmt(n) for n in (1, 2, 3)] == ["1", "2", "3"]


def test_make_formatter_keeps_the_prefix_once_any_id_is_not_global():
    fmt = hbg.make_formatter([_sub_task(3, 0), _param(5), 12])
    assert fmt(_sub_task(3, 0)) == "g3t0"
    assert fmt(_param(5)) == "p5"
    assert fmt(12) == "t12"


def test_make_formatter_falls_back_to_str_for_a_value_that_is_not_an_integer():
    fmt = hbg.make_formatter([1, 2, 3])
    assert fmt("not-a-number") == "not-a-number"


def test_sub_task_decode_answers_no_for_every_id_that_is_not_one():
    """Filtering a mixed stream must answer "not mine" rather than raise.

    The lane collector runs this over every task row and scheduler record in the
    capture, most of which are not sub-tasks. A GLOBAL or PARAM id, and anything
    that is not a number at all, has to fall through quietly.
    """
    assert graph_lane._decode_sub_task_id(_sub_task(3, 5)) == (3, 5)
    assert graph_lane._decode_sub_task_id(_global(3)) is None
    assert graph_lane._decode_sub_task_id(_param(3)) is None
    assert graph_lane._decode_sub_task_id("not-a-number") is None


def test_sub_task_decode_reads_the_space_not_the_bits_below_it():
    """A value in the low half never reaches the space field, whatever it holds.

    The decode has to read the field it names and no other: every value below bit 32
    is a local id, so none of them can make an id look like a sub-task.
    """
    for high in range(256):
        assert graph_lane._decode_sub_task_id((high << 32) | 4) is None


def test_sub_task_decode_reports_a_corrupt_space_rather_than_filtering_it_out():
    """Another space answers "not a sub-task"; a space no mint produces is a corrupt record.

    The filter needs the first answer and gets it, because only the field's undecodable
    value raises -- and that value means the same thing here as anywhere else this
    runtime's id is read.
    """
    with pytest.raises(ValueError, match="id space"):
        graph_lane._decode_sub_task_id((1 << 64) - 1)
