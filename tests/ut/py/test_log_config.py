# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for simpler's single-axis log-level configuration."""

import logging
import subprocess
import sys
import threading

import pytest
from simpler._log import (
    TIMING,
    _ladder_level,
    _normalize_threshold,
    _UnifiedLogHandler,
    attach_unified_log_handler,
    get_logger,
)

from simpler_setup.log_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_CHOICES,
    configure_logging,
    parse_level,
)


def test_default_level_is_timing():
    assert DEFAULT_LOG_LEVEL == "timing"
    assert TIMING == 25
    assert parse_level("timing") == TIMING
    assert parse_level("info") == 20


def test_choices_are_single_axis():
    assert LOG_LEVEL_CHOICES == ["debug", "info", "timing", "warn", "error", "null"]
    for v in range(10):
        assert f"v{v}" not in LOG_LEVEL_CHOICES


def test_concurrent_root_configuration_emits_one_line():
    code = """
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from simpler_setup.log_config import configure_logging

gate = Barrier(8)
def configure(_):
    gate.wait(timeout=5)
    configure_logging("info")

with ThreadPoolExecutor(8) as pool:
    list(pool.map(configure, range(8)))
assert len(logging.getLogger().handlers) == 1
logging.getLogger("application").warning("once")
"""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=15)
    assert run.stderr.splitlines() == ["[WARNING] once"]


def test_null_mutes_severity():
    configure_logging("null")
    sim = logging.getLogger("simpler")
    assert sim.getEffectiveLevel() >= 60


def test_timing_hides_info_but_keeps_timing():
    configure_logging("timing")
    sim = logging.getLogger("simpler")
    assert not sim.isEnabledFor(logging.INFO)
    assert sim.isEnabledFor(TIMING)


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (10, 10),
        (11, 20),
        (20, 20),
        (21, 25),
        (25, 25),
        (26, 30),
        (30, 30),
        (31, 40),
        (40, 40),
        (41, 60),
        (60, 60),
    ],
)
def test_normalize_threshold(threshold, expected):
    assert _normalize_threshold(threshold) == expected


@pytest.fixture
def detached_unified_handler():
    """A logger with no forwarding handler and a level that admits warnings.

    Both halves exist because this logger is process-global and this file is not
    its only writer.

    Any earlier test that initialized a Worker has already installed the real
    handler, and the installer is idempotent — so without clearing it first, every
    assertion below would silently exercise that handler instead of this test's
    recorder. And `configure_logging("null")` earlier in this file leaves the
    level at `NUL` without restoring it, which silently drops every record these
    tests emit; the suite's execution order is not the order these tests are
    written in, so "a later test sets it back" is not something to rely on.
    """
    logger = get_logger()
    handlers_before = list(logger.handlers)
    level_before = logger.level
    logger.handlers = [handler for handler in handlers_before if not isinstance(handler, _UnifiedLogHandler)]
    logger.setLevel(TIMING)
    yield logger
    logger.setLevel(level_before)
    logger.handlers = handlers_before


@pytest.mark.parametrize(
    ("levelno", "expected"),
    [
        (logging.DEBUG, logging.DEBUG),
        (logging.DEBUG - 5, logging.DEBUG),
        (logging.INFO, logging.INFO),
        (TIMING, TIMING),
        (TIMING - 1, logging.INFO),
        (logging.WARNING, logging.WARNING),
        (35, logging.WARNING),
        (logging.ERROR, logging.ERROR),
        (logging.CRITICAL, logging.ERROR),
    ],
)
def test_ladder_level_rounds_a_record_toward_the_milder_name(levelno, expected):
    """A record at 35 is a warning someone gave a custom number, not an error.

    This is the opposite direction from `_normalize_threshold`, which rounds a
    *threshold* up so that asking for 35 does not silently admit warnings.
    """
    assert _ladder_level(levelno) == expected


def test_forwarding_sends_the_formatted_record_to_the_host_logger(detached_unified_handler):
    emitted = []
    assert attach_unified_log_handler(lambda *args: emitted.append(args), lambda: "/tmp/bound") is True

    detached_unified_handler.warning("wire-%s", "check")

    assert emitted == [(logging.WARNING, "test_forwarding_sends_the_formatted_record_to_the_host_logger", "wire-check")]


def test_forwarding_is_idempotent(detached_unified_handler):
    assert attach_unified_log_handler(lambda *args: None, lambda: "/tmp/bound") is True
    assert attach_unified_log_handler(lambda *args: None, lambda: "/tmp/bound") is False
    installed = [h for h in detached_unified_handler.handlers if isinstance(h, _UnifiedLogHandler)]
    assert len(installed) == 1


def test_forwarding_leaves_propagation_alone_so_pytest_still_captures(detached_unified_handler, caplog):
    """The host log is the complete copy; the console and caplog are views of it.

    Turning propagation off would make the file the only destination and blind
    every `caplog.at_level(..., logger="simpler")` assertion in this suite.
    """
    emitted = []
    attach_unified_log_handler(lambda *args: emitted.append(args), lambda: "/tmp/bound")

    with caplog.at_level(logging.WARNING, logger="simpler"):
        detached_unified_handler.warning("seen-by-both")

    assert "seen-by-both" in caplog.text
    assert [args[2] for args in emitted] == ["seen-by-both"]
    assert detached_unified_handler.propagate is True


def test_a_failing_sink_does_not_raise_into_the_caller(detached_unified_handler, monkeypatch):
    """A logging call must not become the reason a run fails."""

    def explode(*_args):
        raise RuntimeError("sink is down")

    attach_unified_log_handler(explode, lambda: "/tmp/bound")
    # monkeypatch, not a `finally` that sets True: this is a process-wide flag
    # this test does not own, and restoring a guessed value is how a test leaves
    # the next one running under settings nobody chose.
    monkeypatch.setattr(logging, "raiseExceptions", False)

    detached_unified_handler.warning("still-returns")


def test_records_are_not_forwarded_while_the_host_log_writes_to_stderr(detached_unified_handler):
    """No destination of its own means the console already has the record.

    The root logger's handler prints to stderr, and so does the host logger until
    it is given a directory — so forwarding then would put one record on one
    stream twice, in two envelopes, for every run that does not ask for
    diagnostics.
    """
    emitted = []
    directory = [""]
    assert attach_unified_log_handler(lambda *args: emitted.append(args), lambda: directory[0]) is True

    detached_unified_handler.warning("console-only")
    assert emitted == []

    # The directory arrives with a run's config, after the handler is installed.
    directory[0] = "/tmp/bound"
    detached_unified_handler.warning("also-to-the-log")
    assert [args[2] for args in emitted] == ["also-to-the-log"]


def test_concurrent_installation_yields_exactly_one_handler(detached_unified_handler):
    """Two workers initializing at once must not double every record.

    The installer is called from `Worker.init()` and `ChipWorker.init()`, whose
    locks are per instance — so nothing above it serializes two instances racing,
    and an unguarded check-and-add would let both add a handler.
    """
    threads = 8
    ready = threading.Barrier(threads)
    installed: list[bool] = []
    lock = threading.Lock()

    def attach():
        ready.wait()
        result = attach_unified_log_handler(lambda *args: None, lambda: "/tmp/bound")
        with lock:
            installed.append(result)

    workers = [threading.Thread(target=attach) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert sum(installed) == 1, installed
    assert sum(isinstance(h, _UnifiedLogHandler) for h in detached_unified_handler.handlers) == 1


def test_importing_simpler_installs_no_forwarding_handler():
    """The extension must stay off the import path of the logging surface.

    `import simpler` has to keep working when `_task_interface` is missing or
    stale — the build-stamp guard raises on every source-tree move until a
    rebuild — so the handler is installed where the threshold is seeded, not here.
    """
    code = (
        "import simpler, logging; "
        "from simpler._log import _UnifiedLogHandler; "
        "print(any(isinstance(h, _UnifiedLogHandler) for h in logging.getLogger('simpler').handlers))"
    )
    out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", f"{out.stdout!r} {out.stderr!r}"


def test_seeding_the_native_threshold_also_installs_the_forwarding_handler(detached_unified_handler):
    """The wiring, not just the mechanism.

    Every test above installs the handler itself, so without this one the line in
    `_initialize_host_log` that does it in production could be deleted and the
    suite would stay green.
    """
    from simpler.task_interface import _initialize_host_log  # noqa: PLC0415 -- needs the extension

    assert not any(isinstance(h, _UnifiedLogHandler) for h in detached_unified_handler.handlers)

    _initialize_host_log(TIMING)

    assert any(isinstance(h, _UnifiedLogHandler) for h in detached_unified_handler.handlers)
