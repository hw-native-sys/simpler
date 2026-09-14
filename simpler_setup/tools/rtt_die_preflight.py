#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full A5 AICPU affinity preflight: orch via atomic-flag handshake, sched via COND die scores."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from simpler_setup.environment import PROJECT_ROOT

SCHEDULER_COUNT = 4
PROBE_SCHEMA_VERSION = 3
PLAN_SCHEMA_VERSION = 3
MEASUREMENT_METHOD = "atomic-flag-orch+cond-die-v1"
PLAN_SOURCE_PROBE_FAILED = "probe-failed-contiguous"
PLAN_SOURCE_MANUAL = "manual"
PLAN_SOURCE_AUTO_FIRST_RUN = "auto-first-run"
SUPPORTED_PLAN_SOURCES = frozenset(
    {MEASUREMENT_METHOD, PLAN_SOURCE_PROBE_FAILED, PLAN_SOURCE_MANUAL, PLAN_SOURCE_AUTO_FIRST_RUN}
)
# Phys pick die0,die1,die1,die0 → pack to logical [P0,P3,P1,P2] so S0/S1→die0, S2/S3→die1.
PHYS_PICK_DIES = (0, 1, 1, 0)
PROBE_SAMPLES_PER_CORE = 100
ACTIVE_THREAD_COUNT = SCHEDULER_COUNT + 1
MAX_LAUNCH_THREAD_COUNT = 14
PREFLIGHT_TIMEOUT_SECONDS = 30
# Helper/dispatcher compilation is outside the device-probe budget.
HELPER_BUILD_TIMEOUT_SECONDS = 600
PREFLIGHT_TIMEOUT_EXIT_CODE = 124
TIMEOUT_RECORD_SCHEMA_VERSION = 1
PLAN_ENV = "SIMPLER_AICPU_AFFINITY_PLAN"
RELATIVE_PLAN = Path("build/config/aicpu_affinity_plan.json")


class AffinityProbeTimeout(subprocess.TimeoutExpired):
    """Raised only when the device RTT/COND probe exceeds PREFLIGHT_TIMEOUT_SECONDS."""


_TOOL = Path(__file__).resolve().parent / "aicpu_device_query"
_CACHE = PROJECT_ROOT / "build" / "cache" / "aicpu_device_query"
_DISPATCHER = PROJECT_ROOT / "build" / "lib" / "a5" / "dispatcher" / "libsimpler_aicpu_dispatcher.so"
_CACHE_LOCK = _CACHE / ".build.lock"
_CACHE_STAMP = _CACHE / ".build.stamp"


def _with_device_suffix(path: Path, device_id: int) -> Path:
    """Resolve a base/template path to one device's JSON without doubling an existing suffix."""
    raw = str(path)
    if "{device}" in raw:
        return Path(raw.replace("{device}", str(device_id)))
    suffix = path.suffix if path.suffix else ".json"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    if not stem.endswith(f".{device_id}"):
        stem += f".{device_id}"
    return path.with_name(f"{stem}{suffix}")


def default_plan_path(device_id: int) -> Path:
    """Return the absolute per-device plan path shared by auto-preflight and the runtime reader."""
    configured = os.environ.get(PLAN_ENV, "").strip()
    base = Path(configured) if configured else Path.cwd() / RELATIVE_PLAN
    return _with_device_suffix(base, device_id).resolve()


def cpus_side_path(plan_path: Path) -> Path:
    """Return the thin runtime companion path for one exact per-device JSON path."""
    return plan_path.with_suffix(".cpus")


def timeout_record_path(plan_path: Path) -> Path:
    """Return the per-device automatic-preflight timeout marker path."""
    return plan_path.with_suffix(".timeout")


def timeout_record_looks_usable(path: Path, device_id: int) -> bool:
    """Return whether an automatic preflight timeout marker is valid for this device."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return (
            isinstance(record, dict)
            and record.get("schema_version") == TIMEOUT_RECORD_SCHEMA_VERSION
            and record.get("device_id") == int(device_id)
            and record.get("status") == "timeout"
            and record.get("timeout_seconds") == PREFLIGHT_TIMEOUT_SECONDS
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_timeout_record(plan_path: Path, device_id: int, *, reason: str) -> Path:
    """Atomically persist the marker that suppresses repeated automatic probes."""
    marker = timeout_record_path(plan_path)
    _atomic_write_text(
        marker,
        json.dumps(
            {
                "schema_version": TIMEOUT_RECORD_SCHEMA_VERSION,
                "device_id": int(device_id),
                "status": "timeout",
                "timeout_seconds": PREFLIGHT_TIMEOUT_SECONDS,
                "reason": reason,
                "created_at": int(time.time()),
            },
            indent=2,
        )
        + "\n",
    )
    return marker


def clear_timeout_record(plan_path: Path) -> None:
    """Allow an explicit manual probe to retry after an automatic timeout."""
    timeout_record_path(plan_path).unlink(missing_ok=True)


def _helper_source_files() -> list[Path]:
    """Return source files that affect the standalone query helper artifacts."""
    roots = (
        _TOOL,
        PROJECT_ROOT / "src" / "a5" / "platform" / "include",
        PROJECT_ROOT / "src" / "a5" / "platform" / "onboard" / "host",
        PROJECT_ROOT / "src" / "common" / "log",
        PROJECT_ROOT / "src" / "common" / "platform" / "include",
    )
    return sorted(
        {
            path
            for root in roots
            if root.exists()
            for path in (root.rglob("*") if root.is_dir() else (root,))
            if path.is_file()
        }
    )


def _helper_source_fingerprint(env: Mapping[str, str]) -> str:
    """Hash helper sources and toolchain identity for the shared build cache."""
    digest = hashlib.sha256()
    for key in ("ASCEND_HOME_PATH", "CC", "CXX"):
        digest.update(key.encode("utf-8"))
        digest.update(b"=")
        digest.update(env.get(key, "").encode("utf-8"))
        digest.update(b"\0")
    for path in _helper_source_files():
        try:
            relative = path.relative_to(PROJECT_ROOT)
            digest.update(str(relative).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"cannot fingerprint helper source {path}: {exc}") from exc
    return digest.hexdigest()


@contextmanager
def _helper_build_lock(*, deadline: float | None = None, timeout_seconds: float = HELPER_BUILD_TIMEOUT_SECONDS):
    """Serialize helper builds across processes under the helper-build budget."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK.open("w", encoding="utf-8") as lock_file:
        locked = False
        try:
            while not locked:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if deadline is not None and time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired("aicpu_device_query build lock", timeout_seconds)
                    time.sleep(0.05)
            yield
        finally:
            if locked:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _artifact_looks_usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _cache_stamp_matches(fingerprint: str) -> bool:
    try:
        record = json.loads(_CACHE_STAMP.read_text(encoding="utf-8"))
        return record.get("schema_version") == 1 and record.get("fingerprint") == fingerprint
    except (OSError, UnicodeError, TypeError, AttributeError, ValueError, json.JSONDecodeError):
        return False


def _write_cache_stamp(fingerprint: str) -> None:
    _atomic_write_text(
        _CACHE_STAMP,
        json.dumps({"schema_version": 1, "fingerprint": fingerprint}, sort_keys=True) + "\n",
    )


def _run_with_deadline(
    command: Sequence[str],
    *,
    deadline: float | None = None,
    timeout_seconds: float = PREFLIGHT_TIMEOUT_SECONDS,
    timeout_error: type[subprocess.TimeoutExpired] = subprocess.TimeoutExpired,
    **kwargs: Any,
) -> Any:
    """Run one child process under an absolute deadline (probe or helper-build)."""
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise timeout_error(command, timeout_seconds)
        kwargs["timeout"] = remaining
    try:
        return subprocess.run(command, check=kwargs.pop("check", False), **kwargs)
    except subprocess.TimeoutExpired as exc:
        if timeout_error is subprocess.TimeoutExpired:
            raise
        raise timeout_error(exc.cmd, timeout_seconds) from exc


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_cpu_list(
    values: Sequence[object], field: str, *, min_count: int, max_count: int = MAX_LAUNCH_THREAD_COUNT
) -> list[int]:
    """Validate a unique AICPU id list against the driver mask width and supported launch bound."""
    cpus = [_require_int(value, f"{field}[]") for value in values]
    if not min_count <= len(cpus) <= max_count:
        raise ValueError(f"{field} must contain between {min_count} and {max_count} CPUs")
    if len(set(cpus)) != len(cpus):
        raise ValueError(f"{field} contains duplicate CPUs")
    if any(cpu < 0 or cpu >= 64 for cpu in cpus):
        raise ValueError(f"{field} CPU ids must be in [0, 63]")
    return cpus


def occupy_mask_from_cpus(cpus: Sequence[int]) -> int:
    """Encode validated logical CPU ids as the driver's 64-bit OCCUPY mask."""
    mask = 0
    for cpu in cpus:
        mask |= 1 << int(cpu)
    return mask


def validate_device_node(
    device_node: Mapping[str, Any], *, expected_soc: str, expected_device: int
) -> tuple[list[int], list[int]]:
    """Validate the complete persisted JSON/side contract before either file is published."""
    if not expected_soc.startswith("Ascend950") or device_node.get("soc_name") != expected_soc:
        raise ValueError(f"unsupported or mismatched plan soc_name: {device_node.get('soc_name')!r}")
    if _require_int(device_node.get("device_id"), "device_id") != expected_device:
        raise ValueError("plan device_id does not match --device")
    if _require_int(device_node.get("active_count"), "active_count") != ACTIVE_THREAD_COUNT:
        raise ValueError(f"plan active_count must be {ACTIVE_THREAD_COUNT}")
    source = device_node.get("plan_source")
    if source not in SUPPORTED_PLAN_SOURCES:
        raise ValueError(f"unsupported plan_source: {source!r}")
    raw_pool = device_node.get("user_pool_cpus")
    raw_allowed = device_node.get("allowed_cpus")
    if not isinstance(raw_pool, list) or not isinstance(raw_allowed, list):
        raise ValueError("user_pool_cpus and allowed_cpus must be arrays")
    pool = validate_cpu_list(raw_pool, "user_pool_cpus", min_count=ACTIVE_THREAD_COUNT)
    allowed = validate_cpu_list(
        raw_allowed, "allowed_cpus", min_count=ACTIVE_THREAD_COUNT, max_count=ACTIVE_THREAD_COUNT
    )
    if not set(allowed).issubset(pool):
        raise ValueError("allowed_cpus must be a subset of user_pool_cpus")
    raw_mask = device_node.get("occupy_mask")
    try:
        occupy_mask = int(raw_mask, 0) if isinstance(raw_mask, str) else _require_int(raw_mask, "occupy_mask")
    except ValueError as exc:
        raise ValueError("occupy_mask must be a valid 64-bit integer") from exc
    if occupy_mask <= 0 or occupy_mask >= 1 << 64 or occupy_mask != occupy_mask_from_cpus(pool):
        raise ValueError("occupy_mask must exactly match user_pool_cpus")
    if _require_int(device_node.get("orch_cpu"), "orch_cpu") != allowed[-1]:
        raise ValueError("orch_cpu must be the last allowed CPU")
    schedulers = device_node.get("schedulers")
    if not isinstance(schedulers, list) or len(schedulers) != SCHEDULER_COUNT:
        raise ValueError(f"schedulers must contain exactly {SCHEDULER_COUNT} entries")
    for idx, scheduler in enumerate(schedulers):
        if not isinstance(scheduler, dict):
            raise ValueError("scheduler entries must be objects")
        if _require_int(scheduler.get("logical_idx"), "logical_idx") != idx:
            raise ValueError("scheduler logical_idx values must be contiguous")
        if _require_int(scheduler.get("cpu_id"), "scheduler.cpu_id") != allowed[idx]:
            raise ValueError("scheduler CPU order must match allowed_cpus")
        if _require_int(scheduler.get("assigned_die"), "assigned_die") != (0 if idx < 2 else 1):
            raise ValueError("scheduler assigned_die must match the logical die contract")
    return pool, allowed


def side_file_looks_usable(path: Path, device_id: int) -> bool:
    """Perform the hardware-independent subset of side-file validation used before auto-probing."""
    try:
        entries: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator or key in entries:
                return False
            entries[key] = value
        required = {"schema_version", "device_id", "soc", "source", "occupy_mask", "active_count", "cpus"}
        if set(entries) != required:
            return False
        if int(entries["schema_version"], 10) != PLAN_SCHEMA_VERSION:
            return False
        if int(entries["device_id"], 10) != device_id or int(entries["active_count"], 10) != ACTIVE_THREAD_COUNT:
            return False
        if not entries["soc"].startswith("Ascend950") or entries["source"] not in SUPPORTED_PLAN_SOURCES:
            return False
        cpu_parts = entries["cpus"].split(",")
        if any(not part.strip() for part in cpu_parts):
            return False
        cpus = validate_cpu_list(
            [int(part.strip(), 10) for part in cpu_parts],
            "cpus",
            min_count=ACTIVE_THREAD_COUNT,
            max_count=ACTIVE_THREAD_COUNT,
        )
        occupy_mask = int(entries["occupy_mask"], 0)
        pool_count = bin(occupy_mask).count("1")
        return (
            0 < occupy_mask < 1 << 64
            and ACTIVE_THREAD_COUNT <= pool_count <= MAX_LAUNCH_THREAD_COUNT
            and all((occupy_mask >> cpu) & 1 for cpu in cpus)
        )
    except (OSError, UnicodeError, ValueError):
        return False


def _ascend_env(*, deadline: float | None = None) -> dict[str, str]:
    home = os.environ.get("ASCEND_HOME_PATH", "").strip()
    home_p = (
        Path(home)
        if home
        else next(
            (
                p
                for p in (Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann-9.2.0"))
                if p.is_dir()
            ),
            None,
        )
    )
    if home_p is None or not (home_p / "set_env.sh").is_file():
        raise RuntimeError("ASCEND_HOME_PATH not found")
    out = _run_with_deadline(
        ["bash", "-c", f'source "{home_p / "set_env.sh"}" && env -0'],
        deadline=deadline,
        timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS,
        check=True,
        stdout=subprocess.PIPE,
    )
    env = dict(os.environ)
    for entry in out.stdout.split(b"\0"):
        if b"=" in entry:
            k, _, v = entry.partition(b"=")
            env[k.decode()] = v.decode()
    env["ASCEND_HOME_PATH"] = str(home_p)
    return env


def _cmake_build(
    src: Path, build: Path, env: dict[str, str], *, cross: bool = False, deadline: float | None = None
) -> None:
    cfg = ["cmake", "-S", str(src), "-B", str(build), f"-DSIMPLER_ROOT={PROJECT_ROOT}"]
    if cross:
        ah = Path(env["ASCEND_HOME_PATH"]) / "tools" / "hcc" / "bin"
        cfg += [
            f"-DCMAKE_C_COMPILER={ah / 'aarch64-target-linux-gnu-gcc'}",
            f"-DCMAKE_CXX_COMPILER={ah / 'aarch64-target-linux-gnu-g++'}",
        ]
    _run_with_deadline(
        cfg,
        deadline=deadline,
        timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS,
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    _run_with_deadline(
        ["cmake", "--build", str(build), f"-j{os.cpu_count() or 1}"],
        deadline=deadline,
        timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS,
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )


def _ensure_query_device_hal_artifacts() -> tuple[Path, dict[str, str]]:
    """Compile helper/dispatcher artifacts under the helper-build budget (not the probe budget)."""
    if not (_TOOL / "host").is_dir():
        raise RuntimeError(f"aicpu_device_query sources missing under {_TOOL}")
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to build the aicpu_device_query backend")
    build_deadline = time.monotonic() + HELPER_BUILD_TIMEOUT_SECONDS
    env = _ascend_env(deadline=build_deadline)
    dispatcher_override = bool(env.get("SIMPLER_DISPATCHER_SO"))
    query_override = bool(env.get("SIMPLER_AICPU_QUERY_SO"))
    dispatcher = Path(env["SIMPLER_DISPATCHER_SO"]) if dispatcher_override else _DISPATCHER
    query_so = Path(env["SIMPLER_AICPU_QUERY_SO"]) if query_override else _CACHE / "device" / "libaicpu_query.so"
    host_bin = _CACHE / "host" / "query_device_hal"

    with _helper_build_lock(deadline=build_deadline, timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS):
        if not _artifact_looks_usable(dispatcher):
            if dispatcher_override:
                raise RuntimeError(f"Missing dispatcher SO: {dispatcher}")
            _run_with_deadline(
                [
                    sys.executable,
                    "-m",
                    "simpler_setup.build_runtimes",
                    "--lib-dir",
                    str(PROJECT_ROOT / "build" / "lib"),
                    "--cache-dir",
                    str(PROJECT_ROOT / "build" / "cache"),
                    "--platforms",
                    "a5",
                ],
                deadline=build_deadline,
                timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS,
                check=True,
                cwd=PROJECT_ROOT,
                env={**env, "PYTHONPATH": f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"},
            )
            dispatcher = _DISPATCHER
            if not _artifact_looks_usable(dispatcher):
                raise RuntimeError(f"Missing dispatcher SO: {dispatcher}")

        fingerprint = _helper_source_fingerprint(env)
        stamp_matches = _cache_stamp_matches(fingerprint)
        if not query_override and (not _artifact_looks_usable(query_so) or not stamp_matches):
            _cmake_build(_TOOL / "device", _CACHE / "device", env, cross=True, deadline=build_deadline)
            query_so = _CACHE / "device" / "libaicpu_query.so"
        if not _artifact_looks_usable(query_so):
            raise RuntimeError(f"Missing query SO: {query_so}")
        if not _artifact_looks_usable(host_bin) or not stamp_matches:
            _cmake_build(_TOOL / "host", _CACHE / "host", env, deadline=build_deadline)
        if not _artifact_looks_usable(host_bin):
            raise RuntimeError(f"Missing host launcher: {host_bin}")
        if not stamp_matches:
            _write_cache_stamp(fingerprint)

    env = {**env, "SIMPLER_DISPATCHER_SO": str(dispatcher), "SIMPLER_AICPU_QUERY_SO": str(query_so)}
    return host_bin, env


def _run_query_device_hal(device_id: int, mode: str) -> str:
    """Ensure helpers exist, then run query_device_hal under the 30s probe budget only."""
    if mode not in ("--rtt-json", "--json"):
        raise ValueError(f"unsupported mode: {mode}")
    host_bin, env = _ensure_query_device_hal_artifacts()
    probe_deadline = time.monotonic() + PREFLIGHT_TIMEOUT_SECONDS
    return _run_with_deadline(
        [str(host_bin), str(device_id), mode],
        deadline=probe_deadline,
        timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS,
        timeout_error=AffinityProbeTimeout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=env,
    ).stdout


def pick_orchestrator(pool: Sequence[Mapping[str, Any]]) -> int:
    """Return pool_idx with the smallest avg_handshake_ticks (tie: smaller idx)."""
    if not pool:
        raise ValueError("empty pool")
    best = min(pool, key=lambda e: (int(e["avg_handshake_ticks"]), int(e["pool_idx"])))
    return int(best["pool_idx"])


def pack_schedulers_from_die_scores(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Phys die0,die1,die1,die0 picks → logical [P0,P3,P1,P2] (S0/S1 die0, S2/S3 die1)."""
    if len(candidates) < SCHEDULER_COUNT:
        raise ValueError(f"need at least {SCHEDULER_COUNT} non-orch candidates")
    remaining = [dict(e) for e in candidates]
    picks: list[dict[str, Any]] = []
    for target_die in PHYS_PICK_DIES:
        key = "die0_sum_ticks" if target_die == 0 else "die1_sum_ticks"
        remaining.sort(key=lambda e, k=key: (int(e[k]), int(e["cpu_id"])))
        chosen = dict(remaining.pop(0))
        chosen["assigned_die"] = target_die
        picks.append(chosen)
    logical = [picks[0], picks[3], picks[1], picks[2]]
    for i, entry in enumerate(logical):
        entry["logical_idx"] = i
        entry["assigned_die"] = 0 if i < 2 else 1
    return [int(e["cpu_id"]) for e in logical], logical


def build_allowed_cpus_from_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Turn device probe JSON into an authoritative device plan node."""
    if probe.get("pool_too_small"):
        raise ValueError("RTT preflight requires 4 scheduler CPUs and 1 orchestrator CPU")

    raw_user_pool = probe.get("user_pool_cpus")
    if not isinstance(raw_user_pool, list):
        raise ValueError("user_pool_cpus must be an array")
    user_pool = validate_cpu_list(raw_user_pool, "user_pool_cpus", min_count=ACTIVE_THREAD_COUNT)

    pool_raw = probe.get("pool")
    if not isinstance(pool_raw, list) or len(pool_raw) != len(user_pool):
        raise ValueError("probe pool must contain exactly one entry per user_pool CPU")

    pool_entries: list[dict[str, Any]] = []
    for raw in pool_raw:
        if not isinstance(raw, dict):
            raise ValueError("pool entry must be an object")
        pool_entries.append(
            {
                "pool_idx": _require_int(raw.get("pool_idx"), "pool_idx"),
                "cpu_id": _require_int(raw.get("cpu_id"), "cpu_id"),
                "avg_handshake_ticks": _require_int(raw.get("avg_handshake_ticks"), "avg_handshake_ticks"),
                "die0_sum_ticks": _require_int(raw.get("die0_sum_ticks"), "die0_sum_ticks"),
                "die1_sum_ticks": _require_int(raw.get("die1_sum_ticks"), "die1_sum_ticks"),
                "is_orch": _require_int(raw.get("is_orch"), "is_orch"),
            }
        )

    pool_indices = [int(entry["pool_idx"]) for entry in pool_entries]
    if sorted(pool_indices) != list(range(len(user_pool))):
        raise ValueError("probe pool_idx values must cover [0, pool_count)")
    by_idx = sorted(pool_entries, key=lambda entry: int(entry["pool_idx"]))
    if [int(entry["cpu_id"]) for entry in by_idx] != user_pool:
        raise ValueError("probe pool CPU order does not match user_pool_cpus")
    if any(int(entry["avg_handshake_ticks"]) <= 0 for entry in pool_entries):
        raise ValueError("probe handshake measurements must be positive")
    for entry in pool_entries:
        if int(entry["is_orch"]) not in (0, 1):
            raise ValueError("probe is_orch must be 0 or 1")
        if not int(entry["is_orch"]) and (int(entry["die0_sum_ticks"]) <= 0 or int(entry["die1_sum_ticks"]) <= 0):
            raise ValueError("scheduler die measurements must be positive")

    orch_idx = _require_int(probe.get("orch_pool_idx"), "orch_pool_idx")
    orch_entries = [e for e in pool_entries if int(e["pool_idx"]) == orch_idx]
    marked_orchestrators = [entry for entry in pool_entries if int(entry["is_orch"]) == 1]
    if len(orch_entries) != 1 or marked_orchestrators != orch_entries:
        raise ValueError("probe must mark exactly the reported orchestrator")
    orch = orch_entries[0]
    sched_cpus, schedulers = pack_schedulers_from_die_scores(
        [e for e in pool_entries if int(e["pool_idx"]) != orch_idx]
    )
    allowed = sched_cpus + [int(orch["cpu_id"])]
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed_cpus contains duplicates")

    return {
        "soc_name": probe["soc_name"],
        "device_id": _require_int(probe.get("device_id"), "device_id"),
        "architecture": "a5",
        "plan_source": MEASUREMENT_METHOD,
        "measurement_method": MEASUREMENT_METHOD,
        "user_pool_cpus": user_pool,
        "occupy_mask": f"0x{occupy_mask_from_cpus(user_pool):x}",
        "active_count": ACTIVE_THREAD_COUNT,
        "allowed_cpus": allowed,
        "orch_cpu": int(orch["cpu_id"]),
        "orch_avg_handshake_ticks": int(orch["avg_handshake_ticks"]),
        "schedulers": [
            {
                "logical_idx": int(s["logical_idx"]),
                "cpu_id": int(s["cpu_id"]),
                "assigned_die": int(s["assigned_die"]),
                "die0_sum_ticks": int(s["die0_sum_ticks"]),
                "die1_sum_ticks": int(s["die1_sum_ticks"]),
            }
            for s in schedulers
        ],
        "pool_too_small": False,
    }


def validate_probe_result(raw: object, expected_device: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("probe output must be a JSON object")
    if _require_int(raw.get("schema_version"), "schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported affinity probe schema_version")
    if raw.get("measurement_method") != MEASUREMENT_METHOD:
        raise ValueError("unsupported measurement_method")
    soc_name = raw.get("soc_name")
    if not isinstance(soc_name, str) or not soc_name.startswith("Ascend950"):
        raise ValueError(f"unsupported probe soc_name: {soc_name!r}")
    if _require_int(raw.get("device_id"), "device_id") != expected_device:
        raise ValueError("probe output device_id does not match --device")
    if raw.get("pool_too_small"):
        raise ValueError("RTT preflight pool has fewer than 5 CPUs")
    if _require_int(raw.get("samples_per_core", PROBE_SAMPLES_PER_CORE), "samples_per_core") != PROBE_SAMPLES_PER_CORE:
        raise ValueError("unexpected samples_per_core")
    # Fully validate the measurements before the caller is allowed to persist them.
    build_allowed_cpus_from_probe(raw)
    return raw  # type: ignore[return-value]


def run_affinity_probe(device_id: int) -> dict[str, Any]:
    try:
        stdout = _run_query_device_hal(device_id, "--rtt-json")
    except AffinityProbeTimeout:
        raise
    except subprocess.CalledProcessError as exc:
        if exc.returncode == PREFLIGHT_TIMEOUT_EXIT_CODE:
            raise AffinityProbeTimeout(exc.cmd, PREFLIGHT_TIMEOUT_SECONDS) from exc
        raise RuntimeError(f"affinity probe backend failed with exit code {exc.returncode}") from exc
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("affinity probe backend returned invalid JSON") from exc
    return validate_probe_result(raw, device_id)


def make_device_plan(*, soc_name: str, device_id: int, device_node: Mapping[str, Any]) -> dict[str, Any]:
    """Build a schema-v3 JSON document containing exactly one device."""
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "_comment": (
            "A5 AICPU affinity plan (authoritative allowed_cpus). "
            "Logical S0/S1 own die0; S2/S3 own die1. Generated by simpler_setup.tools.rtt_die_preflight."
        ),
        "socs": {soc_name: {"devices": {str(device_id): dict(device_node)}}},
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _side_file_text(device_node: Mapping[str, Any]) -> str:
    return (
        f"schema_version={PLAN_SCHEMA_VERSION}\n"
        f"device_id={int(device_node['device_id'])}\n"
        f"soc={device_node['soc_name']}\n"
        f"source={device_node['plan_source']}\n"
        f"occupy_mask={device_node['occupy_mask']}\n"
        f"active_count={int(device_node['active_count'])}\n"
        f"cpus={','.join(str(int(cpu)) for cpu in device_node['allowed_cpus'])}\n"
    )


def plan_files_look_usable(plan_path: Path, device_id: int) -> bool:
    """Validate that an existing JSON/side pair is one coherent generated device plan."""
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != PLAN_SCHEMA_VERSION:
            return False
        socs = raw.get("socs")
        if not isinstance(socs, dict) or len(socs) != 1:
            return False
        soc_name, soc_node = next(iter(socs.items()))
        if not isinstance(soc_name, str) or not isinstance(soc_node, dict):
            return False
        devices = soc_node.get("devices")
        if not isinstance(devices, dict) or set(devices) != {str(device_id)}:
            return False
        device_node = devices[str(device_id)]
        if not isinstance(device_node, dict):
            return False
        validate_device_node(device_node, expected_soc=soc_name, expected_device=device_id)
        side = cpus_side_path(plan_path)
        return side_file_looks_usable(side, device_id) and side.read_text(encoding="utf-8") == _side_file_text(
            device_node
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False


def atomic_write_plan_and_side(path: Path, plan: Mapping[str, Any], device_node: Mapping[str, Any]) -> None:
    """Publish JSON first and the runtime-authoritative side file last, rolling back on write failure."""
    side = cpus_side_path(path)
    previous = {target: target.read_text(encoding="utf-8") if target.is_file() else None for target in (path, side)}
    published: list[Path] = []
    try:
        _atomic_write_text(path, json.dumps(plan, indent=2) + "\n")
        published.append(path)
        _atomic_write_text(side, _side_file_text(device_node))
        published.append(side)
    except (OSError, UnicodeError):
        for target in reversed(published):
            old_text = previous[target]
            if old_text is None:
                target.unlink(missing_ok=True)
            else:
                _atomic_write_text(target, old_text)
        raise


def build_contiguous_fallback_node(
    *,
    soc_name: str,
    device_id: int,
    occupy_cpus: Sequence[int],
    plan_source: str = PLAN_SOURCE_PROBE_FAILED,
) -> dict[str, Any]:
    """Build an explicitly requested five-thread contiguous plan. Probe failures never call this."""
    pool = validate_cpu_list(occupy_cpus, "occupy_cpus", min_count=ACTIVE_THREAD_COUNT)
    allowed = list(pool[:ACTIVE_THREAD_COUNT])
    return {
        "soc_name": soc_name,
        "device_id": device_id,
        "architecture": "a5",
        "plan_source": plan_source,
        "user_pool_cpus": pool,
        "occupy_mask": f"0x{occupy_mask_from_cpus(pool):x}",
        "active_count": ACTIVE_THREAD_COUNT,
        "allowed_cpus": allowed,
        "orch_cpu": allowed[-1],
        "schedulers": [
            {
                "logical_idx": idx,
                "cpu_id": cpu,
                "assigned_die": 0 if idx < 2 else 1,
                "die0_sum_ticks": 0,
                "die1_sum_ticks": 0,
            }
            for idx, cpu in enumerate(allowed[:-1])
        ],
        "pool_too_small": False,
    }


def build_device_node_from_allowed(
    *,
    soc_name: str,
    device_id: int,
    allowed_cpus: Sequence[int],
    occupy_cpus: Sequence[int],
    plan_source: str = PLAN_SOURCE_MANUAL,
) -> dict[str, Any]:
    allowed = validate_cpu_list(
        allowed_cpus, "allowed_cpus", min_count=ACTIVE_THREAD_COUNT, max_count=ACTIVE_THREAD_COUNT
    )
    occupy = validate_cpu_list(occupy_cpus, "occupy_cpus", min_count=ACTIVE_THREAD_COUNT)
    if not set(allowed).issubset(occupy):
        raise ValueError("allowed_cpus must be a subset of occupy_cpus")
    if plan_source not in SUPPORTED_PLAN_SOURCES:
        raise ValueError(f"unsupported plan_source: {plan_source}")
    return {
        "soc_name": soc_name,
        "device_id": device_id,
        "architecture": "a5",
        "plan_source": plan_source,
        "user_pool_cpus": occupy,
        "occupy_mask": f"0x{occupy_mask_from_cpus(occupy):x}",
        "active_count": ACTIVE_THREAD_COUNT,
        "allowed_cpus": allowed,
        "orch_cpu": allowed[-1],
        "schedulers": [
            {
                "logical_idx": idx,
                "cpu_id": int(cpu),
                "assigned_die": 0 if idx < 2 else 1,
                "die0_sum_ticks": 0,
                "die1_sum_ticks": 0,
            }
            for idx, cpu in enumerate(allowed[:-1])
        ],
        "pool_too_small": False,
    }


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def persist_device_plan(out_path: Path, *, soc_name: str, device_id: int, device_node: dict[str, Any]) -> None:
    validate_device_node(device_node, expected_soc=soc_name, expected_device=device_id)
    plan = make_device_plan(soc_name=soc_name, device_id=device_id, device_node=device_node)
    atomic_write_plan_and_side(out_path, plan, device_node)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe A5 AICPU affinity and write the authoritative allowed_cpus plan."
    )
    parser.add_argument("--device", type=int, default=0, help="Logical ACL device id")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Exact per-device JSON path (default: build/config/aicpu_affinity_plan.<device>.json)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="Run the full affinity preflight on device")
    mode.add_argument(
        "--allowed-cpus",
        help="Offline authoritative allowed_cpus (exactly S0,S1,S2,S3,O); requires --soc and --occupy-cpus",
    )
    mode.add_argument(
        "--fallback-occupy",
        help="Write contiguous fallback plan from occupy CPU list (comma-separated); requires --soc",
    )
    parser.add_argument("--soc", help="SoC name for offline / fallback modes")
    parser.add_argument("--occupy-cpus", help="Full OCCUPY CPU list required by offline --allowed-cpus")
    parser.add_argument(
        "--plan-source",
        default=None,
        choices=sorted(SUPPORTED_PLAN_SOURCES),
        help="Optional plan_source override (e.g. auto-first-run when ChipWorker.init probes)",
    )
    args = parser.parse_args(argv)

    out_path = args.out.resolve() if args.out is not None else default_plan_path(args.device)
    try:
        if args.probe:
            if args.soc is not None or args.occupy_cpus is not None:
                parser.error("--probe obtains --soc and OCCUPY from hardware")
            if args.plan_source != PLAN_SOURCE_AUTO_FIRST_RUN:
                clear_timeout_record(out_path)
            probe = run_affinity_probe(args.device)
            device_node = build_allowed_cpus_from_probe(probe)
            if args.plan_source is not None:
                device_node["plan_source"] = args.plan_source
            soc_name = device_node["soc_name"]
        elif args.fallback_occupy is not None:
            if args.soc is None or args.occupy_cpus is not None:
                parser.error("--fallback-occupy requires --soc and cannot be combined with --occupy-cpus")
            clear_timeout_record(out_path)
            soc_name = args.soc
            device_node = build_contiguous_fallback_node(
                soc_name=soc_name,
                device_id=args.device,
                occupy_cpus=parse_int_list(args.fallback_occupy),
                plan_source=args.plan_source if args.plan_source is not None else PLAN_SOURCE_PROBE_FAILED,
            )
        else:
            if args.soc is None or args.allowed_cpus is None or args.occupy_cpus is None:
                parser.error("offline mode requires --soc, --allowed-cpus, and --occupy-cpus")
            clear_timeout_record(out_path)
            soc_name = args.soc
            device_node = build_device_node_from_allowed(
                soc_name=soc_name,
                device_id=args.device,
                allowed_cpus=parse_int_list(args.allowed_cpus),
                occupy_cpus=parse_int_list(args.occupy_cpus),
                plan_source=args.plan_source if args.plan_source is not None else PLAN_SOURCE_MANUAL,
            )

        persist_device_plan(out_path, soc_name=soc_name, device_id=args.device, device_node=device_node)
    except AffinityProbeTimeout:
        marker = write_timeout_record(out_path, args.device, reason="preflight-timeout")
        print(
            f"rtt_die_preflight: device probe timed out after {PREFLIGHT_TIMEOUT_SECONDS}s; "
            f"wrote timeout record {marker}; no affinity plan written",
            file=sys.stderr,
        )
        return 1
    except subprocess.TimeoutExpired as exc:
        print(
            f"rtt_die_preflight: helper build timed out after {exc.timeout}s; no affinity plan written",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"rtt_die_preflight: {exc}; no affinity plan written", file=sys.stderr)
        return 1
    side = cpus_side_path(out_path)
    print(
        f"Wrote affinity plan: {out_path} side={side} soc={soc_name} device={args.device} "
        f"allowed_cpus={device_node['allowed_cpus']} source={device_node['plan_source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
