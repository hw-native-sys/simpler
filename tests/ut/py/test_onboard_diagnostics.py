# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Auxiliary evidence capture may not decide a case.

``capture_case`` is an autouse fixture on the a2a3 onboard scene tests, so
anything it raises is reported by pytest as a teardown error against a case
whose body already finished. It reads directories it does not own -- the
fallback device-log roots belong to another user on a shared host -- and
``Path.glob`` raises on the first advance of its generator rather than at the
call, so a ``try`` around the loop body alone does not contain it.

These drive the helper directly rather than through a device: what is under
test is the containment, which is the same whatever the underlying error was.
"""

import os
from pathlib import Path

import pytest

from tools import onboard_diagnostics
from tools.onboard_diagnostics import capture_case, safe_iterdir, safe_rglob, tail_lines


@pytest.fixture
def refusing_collection(monkeypatch):
    """Evidence collection that fails the way a foreign log root does.

    The occupancy sampler is stubbed with it: the real one shells out to
    `task-submit` and `npu-smi`, which these cases neither need nor should
    wait on.
    """

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", "/root/ascend/log/debug")

    def quiet_sampler(destination, _stop):
        destination.write_text("")

    monkeypatch.setattr(onboard_diagnostics, "sample_occupancy", quiet_sampler)
    monkeypatch.setattr(onboard_diagnostics, "collect_evidence", refuse)


def test_evidence_failure_does_not_reach_the_case(refusing_collection):
    with capture_case():
        pass


def test_evidence_failure_restores_the_log_path(monkeypatch, refusing_collection):
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", "/previously/set")
    with capture_case():
        assert os.environ["ASCEND_PROCESS_LOG_PATH"] != "/previously/set"
    assert os.environ["ASCEND_PROCESS_LOG_PATH"] == "/previously/set"


def test_evidence_failure_removes_a_log_path_it_introduced(monkeypatch, refusing_collection):
    monkeypatch.delenv("ASCEND_PROCESS_LOG_PATH", raising=False)
    with capture_case():
        assert "ASCEND_PROCESS_LOG_PATH" in os.environ
    assert "ASCEND_PROCESS_LOG_PATH" not in os.environ


def test_the_cases_own_failure_is_the_one_that_propagates(refusing_collection):
    # The teardown's own error must not replace what the case was reporting.
    with pytest.raises(AssertionError, match="the case's own failure"):
        with capture_case():
            raise AssertionError("the case's own failure")


def test_an_unreadable_directory_yields_no_sources(monkeypatch, capsys):
    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "glob", refuse)
    monkeypatch.setattr(Path, "rglob", refuse)
    assert safe_iterdir(Path("/root/ascend/log/debug"), "plog/plog-1_*.log") == []
    assert safe_rglob(Path("/root/ascend/log/debug"), "*.log") == []
    # Reported rather than silent: an evidence gap a reader cannot see is worse
    # than one they can.
    assert capsys.readouterr().out.count("[EVIDENCE COLLECTION ERROR]") == 2


def test_a_directory_that_raises_only_on_iteration_is_contained(monkeypatch, capsys):
    # The real trigger, which a glob that raises when *called* does not
    # reproduce: `Path.glob` returns a generator, and an unreadable directory
    # raises on its first advance. That is why wrapping the loop body is not
    # enough and the call site is materialized inside the handler.
    def lazily_refuse(*_args, **_kwargs):
        def generator():
            raise PermissionError(13, "Permission denied", "/root/ascend/log/debug")
            yield  # pragma: no cover -- unreachable, present to make this a generator

        return generator()

    monkeypatch.setattr(Path, "glob", lazily_refuse)
    monkeypatch.setattr(Path, "rglob", lazily_refuse)
    assert safe_iterdir(Path("/root/ascend/log/debug"), "plog/plog-1_*.log") == []
    assert safe_rglob(Path("/root/ascend/log/debug"), "*.log") == []
    assert capsys.readouterr().out.count("[EVIDENCE COLLECTION ERROR]") == 2


def test_a_missing_directory_yields_no_sources(tmp_path):
    assert safe_iterdir(tmp_path / "absent", "*.log") == []
    assert safe_rglob(tmp_path / "absent", "*.log") == []


def test_the_console_view_is_bounded_and_names_the_whole_file(tmp_path):
    source = tmp_path / "host.1.log"
    source.write_text("\n".join(f"line {index}" for index in range(500)))

    rendered = tail_lines(source, 120)

    body = [line for line in rendered.splitlines() if line.startswith("line ")]
    assert body == [f"line {index}" for index in range(380, 500)]
    assert "380 earlier line(s) omitted" in rendered
    assert str(source) in rendered, "a truncated view has to say where the rest is"


def test_a_short_file_is_shown_whole(tmp_path):
    source = tmp_path / "occupancy.log"
    source.write_text("one\ntwo\n")
    assert tail_lines(source, 120) == "one\ntwo"


def test_an_unreadable_file_reports_instead_of_raising(tmp_path):
    rendered = tail_lines(tmp_path / "absent.log", 10)
    assert "[EVIDENCE COLLECTION ERROR]" in rendered
