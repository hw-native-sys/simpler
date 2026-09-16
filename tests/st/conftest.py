# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conftest for scene tests (tests/st/).

sys.path is handled by pyproject.toml [tool.pytest.ini_options] pythonpath.
"""

import os
import re
import time
from pathlib import Path

import pytest


@pytest.fixture
def drain_host_log():
    """Read new host-log output from the process's active sink.

    A `[STRACE]` record reaches stderr or ``host.<pid>.log`` through the process
    writer thread, so a bare ``capfd.readouterr()`` both races the writer and
    misses records after a run binds the logger to its output directory. Track
    the file cursor as well as captured output so repeated drains consume each
    record exactly once regardless of which sink is active.

    The wait is bounded but it is not the verdict. Producers are quiescent by the
    time a test reads — the run has completed — so `pending_record_count`
    reaching zero is a real drain rather than a deadline standing in for
    correctness, and the caller's own assertion stays the thing that decides.
    Exhausting the bound means the writer is genuinely stuck, and the message
    reports the drop counter so a queue loss is not mistaken for a slow drain.
    """
    from _task_interface import _host_log_directory  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    from simpler.task_interface import (  # noqa: PLC0415
        _flush_host_log,
        _host_log_dropped_records,
        _host_log_pending_records,
    )

    file_directory = _host_log_directory()
    fixture_start_ns = time.monotonic_ns()
    file_offset = 0
    if file_directory:
        host_log = Path(file_directory) / f"host.{os.getpid()}.log"
        if host_log.exists():
            file_offset = host_log.stat().st_size

    def _read_file_tail() -> str:
        nonlocal file_directory, file_offset
        active_directory = _host_log_directory()
        if not active_directory:
            file_directory = ""
            file_offset = 0
            return ""

        changed_directory = active_directory != file_directory
        if changed_directory:
            file_directory = active_directory
            file_offset = 0
        host_log = Path(active_directory) / f"host.{os.getpid()}.log"
        try:
            with host_log.open("rb") as stream:
                stream.seek(file_offset)
                chunk = stream.read()
                file_offset = stream.tell()
        except FileNotFoundError:
            return ""
        text = chunk.decode("utf-8", errors="replace")
        if not changed_directory:
            return text

        # The bind may happen after fixture setup but before this first read.
        # Starting at EOF would discard this test's already-flushed records;
        # starting at zero without a cutoff would replay an earlier session.
        # Every unified record carries the same monotonic envelope, so retain
        # precisely the records produced after this fixture began.
        kept = []
        for line in text.splitlines(keepends=True):
            match = re.match(r"\[mono_ns=(\d+)\]", line)
            if match is not None and int(match.group(1)) >= fixture_start_ns:
                kept.append(line)
        return "".join(kept)

    def _drain(capfd, timeout_s: float = 5.0) -> str:
        dropped_before = _host_log_dropped_records()
        chunks: list[str] = []
        deadline = time.monotonic() + timeout_s
        while True:
            flushed = _flush_host_log(100)
            captured = capfd.readouterr()
            chunks.extend((captured.err, captured.out))
            if flushed and _host_log_pending_records() == 0:
                chunks.append(_read_file_tail())
                return "".join(chunks)
            if time.monotonic() >= deadline:
                pending = _host_log_pending_records()
                dropped = _host_log_dropped_records() - dropped_before
                raise AssertionError(
                    f"host-log writer did not drain within {timeout_s:.1f}s: "
                    f"pending={pending}, dropped_delta={dropped}, last_flush={flushed}"
                )
            time.sleep(0.01)

    return _drain
