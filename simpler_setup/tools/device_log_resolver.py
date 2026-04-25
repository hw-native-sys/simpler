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


def get_log_root() -> Path:
    """Return log root: ASCEND_WORK_PATH first, then ~/ascend fallback."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        env_root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if env_root.exists():
            return env_root
    return Path.home() / "ascend" / "log" / "debug"


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

    candidates = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0]


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
        matches = [Path(m) for m in glob.glob(expanded_pattern)]
        matches = [p for p in matches if p.is_file()]
        if not matches:
            return None, f"explicit --device-log glob had no matches: {device_log}"
        best = max(matches, key=lambda p: p.stat().st_mtime)
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

    candidates = []
    for device_dir in device_dirs:
        for log_file in device_dir.glob("*.log"):
            candidates.append(log_file)

    if not candidates:
        return None, f"no .log files found under {root}/device-*"

    l2_perf_records_dt = _extract_l2_perf_records_timestamp(l2_perf_records_path)
    if l2_perf_records_dt is None:
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        return best, "auto-scan device-* (newest log)"

    l2_perf_records_ts = l2_perf_records_dt.timestamp()
    best = min(candidates, key=lambda p: abs(p.stat().st_mtime - l2_perf_records_ts))
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
