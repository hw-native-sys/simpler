# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A scene child's captured output is rendered bounded, and kept whole.

The resource phase runs each L3 case as its own pytest process and prints that
child's entire report into a collapsible group. A child whose native host log
is not bound to a file writes every record to its stderr, so the report carries
the whole stream under ``Captured stderr`` — one case contributed a quarter of
a gigabyte, which the runner then has to upload before the job can finish.

What matters here is the line that is *not* dropped. Three populations are
retained unconditionally, and each for its own reason: anything that is not a
native record, because tracebacks and assertion diffs are what the group exists
to show; any record above ``TIMING``, because a fault report is what a reader
came for and the spool file is in a runner directory no artifact collects; and
everything at all when the spool failed, because the console is then the only
copy of the stream there is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest_scene_output", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(index, level="TIMING"):
    """One native host-log record, in the format `format_record` emits."""
    return f"[mono_ns={1000 + index}][T0xfffe55397120][{level}] emit_host_span: [STRACE] v=1 name=node.activate"


def test_a_short_stream_is_passed_through_unchanged():
    cf = _load_root_conftest()
    body = "\n".join([_record(i) for i in range(10)] + ["E   assert 1 == 2"]) + "\n"
    assert cf._bounded_child_output(body, "/tmp/whole.log") == body


def test_only_the_oldest_records_are_dropped():
    cf = _load_root_conftest()
    records = [_record(i) for i in range(cf._HOST_LOG_CONSOLE_RECORDS + 25)]

    rendered = cf._bounded_child_output("\n".join(records), "/tmp/whole.log")

    kept = [line for line in rendered.splitlines() if line.startswith("[mono_ns=")]
    assert kept == records[25:], "the surviving records must be the newest, in order"
    assert (
        "[25 earlier host-log TIMING record(s) omitted from this view; the whole stream is at /tmp/whole.log]"
        in rendered
    )


def test_nothing_is_dropped_when_the_stream_was_not_preserved():
    # A spool that failed is the case where the console is the only copy there
    # is. Dropping then destroys the only evidence, so the budget does not
    # apply at all -- an oversized group is a cost, a lost stream is not
    # recoverable.
    cf = _load_root_conftest()
    body = "\n".join(_record(i) for i in range(cf._HOST_LOG_CONSOLE_RECORDS * 3))
    assert cf._bounded_child_output(body, None) == body


def test_a_fault_record_survives_a_burst_that_follows_it():
    # The sequence that made the level matter: one ERROR, then enough TIMING
    # spans to exceed the budget. Dropping oldest-first by record alone would
    # take the ERROR, and the spool file lives in the runner's temporary
    # directory that no artifact on this workflow collects -- so a reviewer
    # could not retrieve what the console dropped.
    cf = _load_root_conftest()
    fault = _record(0, "ERROR")
    warning = _record(1, "WARN")
    flood = [_record(i) for i in range(2, cf._HOST_LOG_CONSOLE_RECORDS + 202)]
    rendered = cf._bounded_child_output("\n".join([fault, warning, *flood]), "/tmp/whole.log")

    lines = rendered.splitlines()
    assert lines[0] == fault, "a fault report may not be dropped as flood"
    assert warning in lines
    kept_timing = [line for line in lines if "[TIMING]" in line]
    assert len(kept_timing) == cf._HOST_LOG_CONSOLE_RECORDS
    assert kept_timing == flood[200:], "only the oldest TIMING records go"


def test_every_non_record_line_survives_in_place():
    cf = _load_root_conftest()
    # The report a reader actually needs, with the flood interleaved around it.
    report = [
        "=================================== FAILURES ===================================",
        "E       AssertionError: the widest instant held 2 launched run(s)",
        "tests/st/a2a3/host_build_graph/early_enqueue/test_early_enqueue.py:1760: AssertionError",
        "----------------------------- Captured stderr call -----------------------------",
        "=========================== short test summary info ============================",
        "FAILED tests/st/.../test_early_enqueue.py::TestThreeRunCapacity::test_sixteen_runs",
    ]
    flood = [_record(i) for i in range(cf._HOST_LOG_CONSOLE_RECORDS * 2)]
    # Interleaved, so a renderer that merely took a tail would lose the head of
    # the report and one that took a head would lose the summary.
    body = "\n".join([report[0], report[1], *flood[:300], report[2], report[3], *flood[300:], report[4], report[5]])

    rendered = cf._bounded_child_output(body, "/tmp/whole.log")

    kept = [line for line in rendered.splitlines() if not line.startswith("[")]
    assert kept == report, "a line that is not a host-log record may not be dropped or reordered"


def test_a_line_that_merely_mentions_mono_ns_is_not_a_record():
    cf = _load_root_conftest()
    # The filter is the record format, not the substring: a case's own output
    # quoting a record must not be mistaken for one and dropped.
    body = "\n".join(["E   assert '[mono_ns=5]' in captured"] * (cf._HOST_LOG_CONSOLE_RECORDS + 50))
    assert cf._bounded_child_output(body, "/tmp/whole.log") == body


def test_an_empty_body_renders_empty():
    cf = _load_root_conftest()
    assert cf._bounded_child_output("", "/tmp/whole.log") == ""


def test_the_spool_keeps_the_stream_verbatim(tmp_path, monkeypatch):
    cf = _load_root_conftest()
    monkeypatch.setattr(cf.tempfile, "gettempdir", lambda: str(tmp_path))
    body = "\n".join(_record(i) for i in range(5)) + "\n"

    spooled = cf._spool_child_output("tests/st/a2a3/x.py::TestA::test_b", body)

    assert spooled is not None
    assert Path(spooled).read_text() == body, "the file is what makes the console view safe to bound"


def test_two_nodeids_sharing_a_long_prefix_get_their_own_files(tmp_path, monkeypatch):
    # The readable part of the name is truncated, so identity rests on the
    # digest of the whole nodeid rather than on what survives truncation.
    cf = _load_root_conftest()
    monkeypatch.setattr(cf.tempfile, "gettempdir", lambda: str(tmp_path))
    shared = "tests/st/a2a3/host_build_graph/early_enqueue/test_early_enqueue.py::" + "A" * 120

    first = cf._spool_child_output(shared + "::test_one", "first")
    second = cf._spool_child_output(shared + "::test_two", "second")

    assert first != second
    assert Path(first).read_text() == "first"
    assert Path(second).read_text() == "second"


def test_an_unspoolable_output_reports_and_renders_anyway(tmp_path, monkeypatch, capsys):
    cf = _load_root_conftest()

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cf.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cf.Path, "mkdir", refuse)

    assert cf._spool_child_output("case", "body") is None
    assert "[OUTPUT SPOOL ERROR]" in capsys.readouterr().out
