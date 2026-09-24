#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for _capture_builder: the properties it exists to enforce.

The builder's whole claim is that a test cannot express a document no run
could write. Each test here is one clause of that claim, plus the one that
makes the others worth anything: what it builds decodes.
"""

import pytest
from _capture_builder import Capture, legal_phase_names

from simpler_setup.tools import swimlane_converter as sc
from simpler_setup.tools._runtime_dispatch import HBG_RUNTIME, TMR_RUNTIME, get


def test_what_the_builder_makes_is_a_document_the_decoder_accepts():
    """The property every other test here depends on."""
    document = (
        Capture(runtime=TMR_RUNTIME, level=3)
        .task(task_id=0x101, start=1_000, end=1_100)
        .task(task_id=0x102, start=1_200, end=1_300)
        .sched_phase(phase="complete", start=900, end=1_400)
        .build()
    )

    decoded = sc._decode_perf_data(document)

    assert decoded["runtime"] == TMR_RUNTIME
    assert [task["task_id"] for task in decoded["tasks"]] == [0x101, 0x102]
    assert len(decoded["scheduler_records"][0]) == 1


def test_a_capture_must_name_its_runtime():
    """No default: a capture that names no runtime cannot be decoded."""
    with pytest.raises(TypeError):
        Capture(level=2)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValueError, match="Capture"):
        Capture(runtime="some_other_runtime")


def test_a_phase_no_runtime_records_is_refused():
    """The fixture failure this exists for: a phase name nothing emits."""
    capture = Capture(runtime=TMR_RUNTIME, level=3)
    with pytest.raises(ValueError, match="does not record the phase 'ready_claim'"):
        capture.sched_phase(phase="ready_claim", start=1, end=2)


def test_a_phase_the_other_runtime_records_is_refused():
    """Naming a runtime and carrying the other's phase is the §5.8 failure."""
    with pytest.raises(ValueError, match="does not record the phase 'graph_prepare'"):
        Capture(runtime=TMR_RUNTIME, level=3).sched_phase(phase="graph_prepare", start=1, end=2)
    with pytest.raises(ValueError, match="does not record the phase 'release'"):
        Capture(runtime=HBG_RUNTIME, level=3).sched_phase(phase="release", start=1, end=2)


def test_each_runtimes_own_phases_are_accepted():
    """The converse: what a runtime does record passes, including its markers."""
    Capture(runtime=TMR_RUNTIME, level=3).sched_phase(phase="dummy_task", start=1, end=2)
    Capture(runtime=TMR_RUNTIME, level=3).sched_phase(phase="predicated_skip", start=1, end=2)
    Capture(runtime=HBG_RUNTIME, level=3).sched_phase(phase="resolve_standalone", start=1, end=2)
    Capture(runtime=HBG_RUNTIME, level=3).sched_phase(phase="state_probe", start=1, end=2)
    # Shared phases pass under either name.
    for runtime in (HBG_RUNTIME, TMR_RUNTIME):
        Capture(runtime=runtime, level=3).sched_phase(phase="complete", start=1, end=2)


def test_legal_phase_names_come_from_the_runtime_packages():
    """The vocabulary is read, not restated -- so a new phase needs no edit here."""
    assert "release" in legal_phase_names(TMR_RUNTIME)
    assert "release" not in legal_phase_names(HBG_RUNTIME)
    assert "state_probe" in legal_phase_names(HBG_RUNTIME)
    assert "state_probe" not in legal_phase_names(TMR_RUNTIME)
    # A name both runtimes emit is in both vocabularies. They are still two phases
    # that share a label, which is why neither side reads the other's table.
    assert {"complete", "dispatch"} <= legal_phase_names(HBG_RUNTIME)
    assert {"complete", "dispatch"} <= legal_phase_names(TMR_RUNTIME)


def test_idle_is_a_wire_phase_for_hbg_alone():
    """``idle`` is recorded by one runtime and synthesized for the other.

    hbg's AICore scheduler publishes a SchedulerIdleRecord per spin, so a capture
    of it really can carry the phase. tmr has no AICore scheduler and nothing
    publishes one, so the idle a report shows for it is reconstructed from the gaps
    between work records -- a number the tool derived, not a phase a run wrote. A
    fixture that writes one would be a document no run could produce.
    """
    assert "idle" in legal_phase_names(HBG_RUNTIME)
    assert "idle" not in legal_phase_names(TMR_RUNTIME)


def test_legal_phase_names_are_the_wire_vocabulary_not_a_render_table():
    """Read from ``PHASES``, so a phase stays legal regardless of how it is drawn.

    The colour and report-label tables answer presentation questions and are keyed
    by report label, so a wire-level discriminator is absent from them. Assembling
    the vocabulary out of those tables is what once let ``idle`` pass for tmr.
    """
    for runtime_name in (HBG_RUNTIME, TMR_RUNTIME):
        runtime = get(runtime_name)
        assert legal_phase_names(runtime_name) == runtime.PHASES
        # Every alias is a name the wire can carry, whatever a report calls it.
        assert set(runtime.PHASE_ALIASES) <= runtime.PHASES
        # Every drawn phase is one the wire can carry, minus nothing: a colour for a
        # phase no producer emits would be a bar that never appears.
        assert set(runtime.PHASE_COLORS) <= runtime.PHASES


def test_row_widths_and_epochs_are_generated():
    """A caller describes a task; the wire format is not theirs to get wrong."""
    document = Capture(runtime=TMR_RUNTIME, level=3).task(task_id=7, start=100, end=110).build()

    (aicore_row,) = document["aicore_tasks"]
    (scheduler_row,) = document["scheduler_tasks"]["records"]
    assert "scheduler_records" not in document  # no phases added, so no streams section
    assert len(aicore_row) == 7
    assert len(scheduler_row) == 5
    assert aicore_row[-1] == 0
    assert scheduler_row[-1] == 0


def test_phase_records_carry_an_epoch():
    document = Capture(runtime=TMR_RUNTIME, level=3).sched_phase(phase="complete", start=1, end=2).build()

    (record,) = document["scheduler_records"]["streams"][0]["records"]
    assert record["run_epoch"] == 0


def test_a_tasks_two_rows_share_a_join_key():
    """The decoder joins on (run_epoch, core_id, reg_task_id); task() fixes it."""
    document = (
        Capture(runtime=TMR_RUNTIME, level=3).task(task_id=0x101, core_id=1, start=100, end=110, run_epoch=4).build()
    )

    (aicore_row,) = document["aicore_tasks"]
    (scheduler_row,) = document["scheduler_tasks"]["records"]
    aicore_key = (aicore_row[6], aicore_row[0], aicore_row[2])
    scheduler_key = (scheduler_row[4], scheduler_row[0], scheduler_row[1])
    assert aicore_key == scheduler_key == (4, 1, 1)


def test_repeated_tasks_get_distinct_join_keys():
    """Two tasks on one core must not collide -- the decoder rejects duplicates."""
    document = (
        Capture(runtime=TMR_RUNTIME, level=3)
        .task(task_id=0x101, start=100, end=110)
        .task(task_id=0x102, start=200, end=210)
        .build()
    )

    sc._decode_perf_data(document)  # would raise on a duplicate join key
    reg_ids = [row[2] for row in document["aicore_tasks"]]
    assert len(set(reg_ids)) == 2


def test_only_hbg_can_carry_an_aicore_scheduler_stream():
    """tmr has no AICore scheduler, so a stream from one is not a document it writes."""
    Capture(runtime=HBG_RUNTIME, level=3).aicore_sched_phase(phase="state_probe", start=1, end=2)
    with pytest.raises(ValueError, match="has no AICore scheduler"):
        Capture(runtime=TMR_RUNTIME, level=3).aicore_sched_phase(phase="complete", start=1, end=2)


def test_an_aicore_stream_names_a_vector_core():
    """The two producers differ in more than a label: one runs on an AICore."""
    document = (
        Capture(runtime=HBG_RUNTIME, level=3)
        .aicore_sched_phase(phase="dispatch", start=1, end=2, scheduler_id=3)
        .build()
    )

    (stream,) = document["scheduler_records"]["streams"]
    assert stream["producer"] == "aicore"
    assert stream["core_type"] == "aiv"
    assert stream["physical_core_id"] == 3


def test_a_malformed_document_is_still_expressible_by_mutating_the_result():
    """Testing a rejection stays possible, and stays visible at the call site.

    Widening the row rather than narrowing it: a width the reader has never
    accepted keeps this test independent of which widths it accepts today.
    """
    document = Capture(runtime=TMR_RUNTIME, level=3).task(task_id=7, start=100, end=110).build()
    document["aicore_tasks"][0].append(0)

    with pytest.raises(ValueError, match="aicore_tasks\\[0\\] must contain .*columns"):
        sc._decode_perf_data(document)
