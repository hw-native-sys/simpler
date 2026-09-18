#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The public `simpler.trace` surface.

The properties under test are the ones the API exists to make hold by
construction: a caller's span lands in the reserved namespace and nowhere else,
it is stamped with the clock the C++ spans use, and it costs a gate query when
the gate is closed.
"""

import asyncio
import subprocess
import sys
import textwrap

import pytest
import simpler
from simpler import trace

from simpler_setup.tools.strace_timing import external_producer, parse_spans, span_family


@pytest.fixture
def emitted(monkeypatch):
    """Capture what reaches `_emit_host_span`, with the gate held open."""
    records = []
    monkeypatch.setattr(trace, "_host_spans_active", lambda: True)
    monkeypatch.setattr(trace, "_emit_host_span", lambda *args: records.append(args))
    return records


@pytest.fixture
def gate_closed(monkeypatch):
    """Hold the runtime gate shut and fail the test if anything still emits."""

    def unexpected(*args):
        raise AssertionError(f"emitted with host spans off: {args}")

    monkeypatch.setattr(trace, "_host_spans_active", lambda: False)
    monkeypatch.setattr(trace, "_emit_host_span", unexpected)


def _emit_in_child(tmp_path, body):
    """Emit real records from a child process; return its spans and its stdout.

    A child, because the host logger binds its directory once per process and a
    test that bound this one's would take the whole session's records with it.
    Whatever the body prints goes to stdout, since the logger owns stderr until
    the directory is bound.
    """
    script = textwrap.dedent(
        """
        from _task_interface import _monotonic_now_ns, _set_host_log_directory
        from simpler import TIMING, trace
        from simpler.task_interface import _flush_host_log, _initialize_host_log

        _initialize_host_log(TIMING)
        _set_host_log_directory({directory!r})
        {body}
        _flush_host_log()
        """
    ).format(directory=str(tmp_path), body=textwrap.dedent(body).strip())
    completed = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    lines = []
    for log in sorted(tmp_path.glob("host.*.log")):
        lines.extend(log.read_text(encoding="utf-8").splitlines())
    return list(parse_spans(lines)), completed.stdout


def test_a_span_this_api_emits_reads_back_as_an_external_producers_own(tmp_path):
    """End to end: the record a caller writes classifies as external and attributes to it."""
    spans, _ = _emit_in_child(
        tmp_path,
        """
        tracer = trace.producer("pypto")
        with tracer.span("decode_layer", batch=16, layer=3):
            pass
        """,
    )

    assert [span.name for span in spans] == ["ext.pypto.decode_layer"]
    span = spans[0]
    assert span_family(span.name) == "external"
    assert external_producer(span.name) == "pypto"
    assert span.dur > 0
    assert set(span.attrs.split()) == {"batch=16", "layer=3"}


def test_a_caller_cannot_land_a_span_in_one_of_our_families(emitted):
    """The prefix is unconditional, so one of our words is only ever a leaf.

    Nothing validates the name against our level words, because there is nothing
    a caller could pass that reaches them: the producer segment sits between
    `ext.` and whatever it wrote. A rejecting check would add a failure mode
    without adding a guarantee.
    """
    tracer = trace.producer("pypto")
    for impersonation in ("node.dispatch", "chip.run", "core.task", "network1.submit", "ext.other.span"):
        with tracer.span(impersonation):
            pass

    names = [record[0] for record in emitted]
    assert names == [
        "ext.pypto.node.dispatch",
        "ext.pypto.chip.run",
        "ext.pypto.core.task",
        "ext.pypto.network1.submit",
        "ext.pypto.ext.other.span",
    ]
    for name in names:
        assert span_family(name) == "external"
        assert external_producer(name) == "pypto"


def test_a_closed_gate_emits_nothing_and_says_so(gate_closed):
    """`enabled()` and the emit paths read one gate, so they cannot disagree."""
    assert not trace.enabled()
    assert not trace.producer("pypto").enabled()

    with trace.span("phase", batch=16):
        pass

    @trace.span("decorated")
    def work():
        return 7

    assert work() == 7


def test_the_span_carries_the_clock_the_native_records_are_stamped_with(tmp_path):
    """One clock by construction: the wrapper reads it, the caller never does.

    The child prints the native clock either side of the span, so the assertion
    is that the record's own `ts` lies inside that window — not that two clocks
    on this platform happen to agree.
    """
    spans, stdout = _emit_in_child(
        tmp_path,
        """
        before = _monotonic_now_ns()
        with trace.producer("pypto").span("phase"):
            pass
        after = _monotonic_now_ns()
        print(before, after)
        """,
    )
    before, after = (int(value) for value in stdout.split())

    assert len(spans) == 1
    assert before <= spans[0].ts
    assert spans[0].ts + spans[0].dur <= after


def test_a_caller_that_names_no_producer_writes_under_the_unnamed_one(emitted):
    """The default is a constant, not a name derived from the running program."""
    with trace.span("phase"):
        pass

    (name,) = [record[0] for record in emitted]
    assert name == "ext.app.phase"
    assert external_producer(name) == "app"


def test_a_marker_is_a_span_of_no_work_and_never_reports_zero_duration(emitted):
    """There is no separate instant call, and no record carries `dur=0`.

    `dur=0` is what our own emitting side writes for a phase that was never
    stamped — `c_api_shared.cpp` skips a device phase whose duration reads back 0
    — so a public API that produced it would put two meanings on one value.
    """
    assert not hasattr(trace, "instant")

    with trace.span("checkpoint", token=17):
        pass

    (record,) = emitted
    assert record[5] > 0
    assert record[6] == "token=17"


def test_a_raising_block_still_emits_its_span_and_still_raises(emitted):
    """`with` is the shape a caller reaches for around code that can fail."""
    with pytest.raises(ValueError, match="boom"), trace.span("phase"):
        raise ValueError("boom")

    (name,) = [record[0] for record in emitted]
    assert span_family(name) == "external"
    assert name.endswith(".phase")


def test_a_decorated_function_is_timed_per_call_not_per_decoration(monkeypatch):
    """The gate is read when the call runs, so a later threshold change is honored.

    Decoration happens at import, and a worker seeds the threshold well after
    that. Freezing the decision at decoration time would silently drop every span
    from a module imported before `Worker.init`.
    """
    records = []
    monkeypatch.setattr(trace, "_emit_host_span", lambda *args: records.append(args))
    monkeypatch.setattr(trace, "_host_spans_active", lambda: False)

    @trace.producer("pypto").span("work", role="test")
    def work(value):
        return value * 2

    assert work(3) == 6
    assert records == []

    monkeypatch.setattr(trace, "_host_spans_active", lambda: True)
    assert work(4) == 8
    assert [record[0] for record in records] == ["ext.pypto.work"]
    assert records[0][6] == "role=test"


def test_a_decorated_function_keeps_its_identity(emitted):
    @trace.span("work")
    def work():
        """Docstring."""

    assert work.__name__ == "work"
    assert work.__doc__ == "Docstring."


def test_a_decorated_coroutine_is_timed_over_its_body_not_its_creation(emitted):
    """A sync wrapper would close the span before the body ran at all.

    Calling a coroutine function returns immediately with a coroutine; the work
    happens at `await`. So the span has to cover the await, or it reports a few
    hundred nanoseconds of object construction for a function that took
    milliseconds — wrong data rather than no data.
    """

    @trace.span("work")
    async def work(value):
        await asyncio.sleep(0.01)
        return value * 2

    assert asyncio.iscoroutinefunction(work)
    assert asyncio.run(work(3)) == 6

    (record,) = emitted
    assert record[0].endswith(".work")
    # 10 ms of sleep, so anything near the cost of creating the coroutine fails.
    assert record[5] > 5_000_000


def test_a_raising_coroutine_still_emits_its_span_and_still_raises(emitted):
    @trace.span("work")
    async def work():
        await asyncio.sleep(0)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(work())

    assert [record[0].split(".")[-1] for record in emitted] == ["work"]


def test_a_decorated_coroutine_is_gated_per_call_like_the_sync_one(monkeypatch):
    records = []
    monkeypatch.setattr(trace, "_emit_host_span", lambda *args: records.append(args))
    monkeypatch.setattr(trace, "_host_spans_active", lambda: False)

    @trace.span("work")
    async def work():
        return 7

    assert asyncio.run(work()) == 7
    assert records == []

    monkeypatch.setattr(trace, "_host_spans_active", lambda: True)
    assert asyncio.run(work()) == 7
    assert len(records) == 1


def test_our_correlation_keys_stay_out_of_the_public_surface(emitted):
    """`inv` / `hid` / `depth` are ours; a caller has no value for them.

    They are the keys the invocation-keyed views group on, and no external span
    reaches those views, so every record from here carries 0 rather than a value
    a caller invented.
    """
    with trace.span("phase"):
        pass

    _, invocation, callable_hash, depth, _, _, _ = emitted[0]
    assert (invocation, callable_hash, depth) == (0, 0, 0)


def test_a_producer_name_that_would_break_attribution_is_refused():
    """A dotted producer shifts the segments attribution reads."""
    for refused in ("pypto.lib", "", "with space", "a=b"):
        with pytest.raises(ValueError):
            trace.producer(refused)

    with pytest.raises(ValueError):
        trace.producer(None)  # pyright: ignore[reportArgumentType] -- a caller unchecked by a type checker


def test_a_span_with_no_name_of_its_own_is_refused(gate_closed):
    """`ext.<producer>.` names a producer and no span, so it attributes to nobody."""
    with pytest.raises(ValueError):
        trace.span("")
    with pytest.raises(ValueError):
        trace.producer("pypto").span("")


def test_trace_is_advertised_next_to_the_other_public_surface():
    assert "trace" in simpler.__all__
    assert "trace" in dir(simpler)
    assert simpler.trace is trace


def test_importing_simpler_does_not_require_the_extension():
    """`import simpler` stays usable where `_task_interface` is missing or stale.

    The build-stamp guard makes a stale extension an ordinary state of a source
    tree, and the logging helpers must survive it. `trace` is one more lazy
    submodule, so it is the attribute access that needs the extension, not the
    package import.
    """
    code = (
        "import sys; sys.modules['_task_interface'] = None; "
        "import simpler; print('trace' in simpler.__all__); "
        "\ntry:\n    simpler.trace\n    print('LOADED')\nexcept ImportError:\n    print('BLOCKED')"
    )
    out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["True", "BLOCKED"], f"{out.stdout!r} {out.stderr!r}"


def test_st_log_drain_survives_temporary_capture_directories(tmp_path):
    """The process log outlives captures whose directories are deleted between runs."""
    script = tmp_path / "test_capture_log_lifetime.py"
    script.write_text(
        textwrap.dedent(
            """
            import tempfile
            import pytest
            from _task_interface import _emit_host_span, _monotonic_now_ns, _set_host_log_directory
            from simpler import TIMING
            from simpler.task_interface import _initialize_host_log
            from simpler_setup.tools.strace_timing import parse_spans
            from tests.st.conftest import drain_host_log

            @pytest.mark.parametrize("first_invocation", [1, 3])
            def test_two_captures(capfd, drain_host_log, first_invocation):
                _initialize_host_log(TIMING)
                for invocation in (first_invocation, first_invocation + 1):
                    with tempfile.TemporaryDirectory() as capture:
                        _set_host_log_directory(capture)
                        _emit_host_span("chip.run", invocation, 1, 0, _monotonic_now_ns(), 1, "")
                        spans = list(parse_spans(drain_host_log(capfd).splitlines()))
                        assert [(span.name, span.inv) for span in spans] == [("chip.run", invocation)]
            """
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(script),
            "--confcutdir",
            str(tmp_path),
            "-o",
            "tmp_path_retention_policy=failed",
            "-q",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
