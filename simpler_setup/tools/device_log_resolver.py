#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolve Ascend device log files with deterministic precedence."""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime
from pathlib import Path


def _euid_home() -> Path:
    """Real home of the effective uid — what the driver's slog daemon uses.

    Device logs are written by the driver under the *euid's* real home (e.g.
    ``/root`` when tests run as root via sudo / ``task-submit``). ``$HOME`` is
    NOT a safe proxy: sudo leaves ``$HOME`` pointing at the invoking user's
    directory, so ``Path.home()`` (which trusts ``$HOME``) resolves to the wrong
    place and the device log is never found. Read the passwd entry instead.

    Falls back to ``Path.home()`` when the euid has no passwd entry (an
    arbitrary container uid, e.g. ``docker --user 1234`` with no matching
    ``/etc/passwd`` row), where ``$HOME`` is the best guess available.
    """
    import pwd  # noqa: PLC0415

    try:
        return Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except KeyError:
        return Path.home()


def get_log_root() -> Path:
    """Return the CANN debug-log root, honoring the log-relocation env vars.

    Precedence matches CANN's own resolution:
      1) ASCEND_PROCESS_LOG_PATH -> <it>/debug             (highest)
      2) ASCEND_WORK_PATH        -> <it>/log/debug
      3) <euid real home>/ascend/log/debug                 (default)

    The default uses the effective uid's real home (see ``_euid_home``), NOT
    ``$HOME`` — the driver writes device logs under the euid's real home, while
    sudo / ``task-submit`` frequently leave ``$HOME`` pointing at the invoking
    user, so trusting ``$HOME`` looks in the wrong directory.

    A relocated root is used only when it exists on disk; otherwise we fall
    through, so a stale/mis-set env var doesn't mask the real log location.
    """
    process_log_path = os.environ.get("ASCEND_PROCESS_LOG_PATH")
    if process_log_path:
        env_root = Path(process_log_path).expanduser() / "debug"
        if env_root.exists():
            return env_root
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        env_root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if env_root.exists():
            return env_root
    return _euid_home() / "ascend" / "log" / "debug"


def infer_device_id_from_log_path(log_path: Path) -> str | None:
    """Infer device id from any path segment like device-0."""
    for part in log_path.parts:
        match = re.fullmatch(r"device-(\d+)", part)
        if match:
            return match.group(1)
    return None


def _latest_log_from_dir(log_dir: Path) -> Path | None:
    if not log_dir.exists() or not log_dir.is_dir():
        return None

    # stat() each match safely: a log can be rotated/deleted between glob and
    # stat (the 20 MB rotation this tool is built to handle), so skip vanished
    # files rather than crash with FileNotFoundError.
    files_with_mtime = []
    for p in log_dir.glob("*.log"):
        try:
            files_with_mtime.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue
    if not files_with_mtime:
        return None
    files_with_mtime.sort(key=lambda x: x[0], reverse=True)
    return files_with_mtime[0][1]


def _extract_l2_perf_records_timestamp(l2_perf_records_path: Path | None) -> datetime | None:
    if l2_perf_records_path is None:
        return None

    filename_match = re.search(r"(\d{8}_\d{6})\.json$", l2_perf_records_path.name)
    if filename_match:
        try:
            return datetime.strptime(filename_match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    if l2_perf_records_path.exists():
        return datetime.fromtimestamp(l2_perf_records_path.stat().st_mtime)
    return None


def _resolve_explicit_device_log(device_log: str) -> tuple[Path | None, str]:
    if glob.has_magic(device_log):
        # Expand "~" before globbing; glob.glob does not expand it.
        expanded_pattern = str(Path(device_log).expanduser())
        matches_with_mtime = []
        for p in (Path(m) for m in glob.glob(expanded_pattern)):
            try:
                if p.is_file():
                    matches_with_mtime.append((p.stat().st_mtime, p))
            except FileNotFoundError:
                continue
        if not matches_with_mtime:
            return None, f"explicit --device-log glob had no accessible matches: {device_log}"
        best = max(matches_with_mtime, key=lambda x: x[0])[1]
        return best, f"explicit --device-log glob: {device_log}"

    path = Path(device_log).expanduser()
    if path.exists() and path.is_file():
        return path, "explicit --device-log file"

    if path.exists() and path.is_dir():
        best = _latest_log_from_dir(path)
        if best is None:
            return None, f"explicit --device-log directory has no .log files: {path}"
        return best, f"explicit --device-log directory: {path}"

    return None, f"explicit --device-log path not found: {path}"


def _resolve_nearest_log(root: Path, l2_perf_records_path: Path | None) -> tuple[Path | None, str]:
    device_dirs = sorted([p for p in root.glob("device-*") if p.is_dir()])
    if not device_dirs:
        return None, f"no device-* directories found under {root}"

    # Cache each mtime once (avoids a second stat below) and skip files that
    # vanish mid-scan from log rotation.
    candidates_with_mtime = []
    for device_dir in device_dirs:
        for log_file in device_dir.glob("*.log"):
            try:
                candidates_with_mtime.append((log_file.stat().st_mtime, log_file))
            except FileNotFoundError:
                continue

    if not candidates_with_mtime:
        return None, f"no accessible .log files found under {root}/device-*"

    l2_perf_records_dt = _extract_l2_perf_records_timestamp(l2_perf_records_path)
    if l2_perf_records_dt is None:
        best = max(candidates_with_mtime, key=lambda x: x[0])[1]
        return best, "auto-scan device-* (newest log)"

    l2_perf_records_ts = l2_perf_records_dt.timestamp()
    best = min(candidates_with_mtime, key=lambda x: abs(x[0] - l2_perf_records_ts))[1]
    return best, f"auto-scan device-* (closest log to l2_perf_records timestamp {l2_perf_records_dt:%Y-%m-%d %H:%M:%S})"


def resolve_device_log_path(
    device_id: str | None = None,
    device_log: str | None = None,
    l2_perf_records_path: Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve device log path with deterministic precedence.

    Priority:
      1) --device-log explicit path/dir/glob
      2) --device-id -> <log_root>/device-<id>/ newest .log
      3) auto-scan all device-* and choose nearest to l2_perf_records timestamp
    """
    if device_log:
        return _resolve_explicit_device_log(device_log)

    root = get_log_root()

    if device_id is not None:
        device_dir = root / f"device-{device_id}"
        best = _latest_log_from_dir(device_dir)
        if best is None:
            return None, f"device-id selection failed: no .log files in {device_dir}"
        return best, f"device-id selection: device-{device_id} under {root}"

    return _resolve_nearest_log(root, l2_perf_records_path)


def resolve_device_log_paths(
    device_id: str | None = None,
    device_log: str | None = None,
) -> tuple[list[Path], str]:
    """Resolve device log(s), returning ALL files for a multi-match glob.

    A 20 MB-rotated run splits its blocks across several files, so an explicit
    ``--device-log`` glob must parse every match (mtime order), not just the
    newest — otherwise the earlier rounds of a long run are silently dropped.
    Explicit-file / directory / device-id selection still resolve to a single
    newest log, returned as a one-element list.
    """
    if device_log and glob.has_magic(device_log):
        expanded_pattern = str(Path(device_log).expanduser())
        files_with_mtime = []
        for p in (Path(m) for m in glob.glob(expanded_pattern)):
            try:
                if p.is_file():
                    files_with_mtime.append((p.stat().st_mtime, p))
            except FileNotFoundError:
                continue
        if not files_with_mtime:
            return [], f"explicit --device-log glob had no accessible matches: {device_log}"
        files_with_mtime.sort(key=lambda x: x[0])
        logs = [p for _, p in files_with_mtime]
        return logs, f"explicit --device-log glob ({len(logs)} files): {device_log}"

    single, strategy = resolve_device_log_path(device_id=device_id, device_log=device_log)
    return ([single] if single is not None else []), strategy
