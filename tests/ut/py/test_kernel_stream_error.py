# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The caller observes native execution status before any Simpler host drain."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_A5_LEGACY_GRAPH = 3


_RESIDENT_TIMEOUT_SIGNATURES = (
    r"aicpu_execute: runtime failed with rc=-100(?!\d)",
    r"simpler_aicpu_exec: aicpu_execute failed with rc=-100(?!\d)",
)
_RESIDENT_TIMEOUT_PROFILE = {
    "SIMPLER_SCHEDULER_TIMEOUT_MS": "2000",
    "SIMPLER_OP_EXECUTE_TIMEOUT_US": "3000000",
    "SIMPLER_STREAM_SYNC_TIMEOUT_MS": "6000",
}
_RESIDENT_OBSERVERS = ["prepare", "launch", "peek", "caller_sync"]
_RESIDENT_SHAPES = ((4, 2), (1, 0), (7, 0), (7, 2))


def _resident_timeout_evidence(logs, shape=(4, 2)):
    matches = []
    owned_lines = []
    root = logs.resolve()
    for path in sorted(logs.rglob("*.log")):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        for line in path.read_text(errors="replace").splitlines():
            owned_lines.append({"path": str(path.relative_to(logs)), "line": line})
            if any(re.search(pattern, line) for pattern in _RESIDENT_TIMEOUT_SIGNATURES):
                matches.append({"path": str(path.relative_to(logs)), "line": line})
    assert matches, f"No resident device -100 signature in owned log directory {logs}"
    if shape != (4, 2):
        for pattern in (
            r"A5 HBG AICore Scheduler timeout site=95(?!\d)",
            rf"A5 HBG AICore scheduler: unfinished task=.*\bactive_mask=0x{shape[0]:x}(?![0-9a-fA-F])",
        ):
            evidence = [item for item in owned_lines if re.search(pattern, item["line"])]
            assert evidence, f"No owned execution-timeout evidence for shape {shape}: {pattern}"
            matches.extend(evidence)
    return matches


def _wait_for_resident_timeout_evidence(logs, shape):
    started = time.monotonic()
    deadline = started + 5.0
    while True:
        try:
            evidence = _resident_timeout_evidence(logs, shape)
        except (AssertionError, FileNotFoundError):
            evidence = []
        remaining = deadline - time.monotonic()
        if evidence or remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    print(
        "RESIDENT_DIAGNOSTIC_WAIT "
        + json.dumps(
            {
                "shape": list(shape),
                "owned_evidence_found": bool(evidence),
                "wait_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    if not evidence:
        print(
            "RESIDENT_CHILD_DIAGNOSTIC_CONTEXT " + json.dumps(_resident_failure_context(logs, "", "", [])), flush=True
        )
    return evidence


@pytest.mark.parametrize("arrival", [1.5, None], ids=["delayed-owned", "expired"])
def test_resident_failure_child_waits_for_owned_evidence(tmp_path, monkeypatch, arrival):
    shape = (7, 2)
    owned = tmp_path / "device.log"
    signature = "aicpu_execute: runtime failed with rc=-100\n"
    owned.write_text(signature)
    clock = [0.0]

    def deliver(seconds):
        assert 0 < seconds <= 0.05
        clock[0] += seconds
        if arrival is not None and clock[0] >= arrival:
            owned.write_text(
                signature + "A5 HBG AICore Scheduler timeout site=95\n"
                "A5 HBG AICore scheduler: unfinished task=9 active_mask=0x7 flags=0x0\n"
            )

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", deliver)
    evidence = _wait_for_resident_timeout_evidence(tmp_path, shape)
    assert len(evidence) == (3 if arrival is not None else 0)
    assert clock[0] == pytest.approx(1.5 if arrival is not None else 5.0)


def test_resident_failure_child_wait_rejects_foreign_evidence(tmp_path, monkeypatch, capsys):
    owned = tmp_path / "owned"
    owned.mkdir()
    foreign = tmp_path / "foreign.log"
    foreign.write_text(
        "aicpu_execute: runtime failed with rc=-100\n"
        "A5 HBG AICore Scheduler timeout site=95\n"
        "A5 HBG AICore scheduler: unfinished task=9 active_mask=0x7 flags=0x0\n"
    )
    (owned / "device.log").symlink_to(foreign)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert _wait_for_resident_timeout_evidence(owned, (7, 2)) == []
    assert clock[0] == pytest.approx(5.0)
    assert "foreign.log" not in capsys.readouterr().out


def _resident_timeout_profile(environment):
    profile = {name: environment.get(name) for name in _RESIDENT_TIMEOUT_PROFILE}
    assert profile == _RESIDENT_TIMEOUT_PROFILE, f"Resident qualification requires existing CI budgets: {profile}"
    return profile


def _observe_resident_entry(lib, ctx, run, caller, args, config, descriptor, budget):
    stages = []
    assert lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor)) == 0
    stages.append("prepare")
    print("RESIDENT_STAGE " + json.dumps({"stage": "prepare", "rc": 0}), flush=True)
    mode = lib.get_last_scheduler_runtime_mode(ctx)
    count = lib.get_scheduler_mode_publish_count(ctx)
    assert mode in (1, 2) and count > 0, (mode, count)
    assert lib.simpler_launch_run_on_stream(ctx, run, caller, None) == 0
    stages.append("launch")
    print("RESIDENT_STAGE " + json.dumps({"stage": "launch", "rc": 0}), flush=True)
    peek = lib.aclrtPeekAtLastError(0)
    stages.append("peek")
    print("RESIDENT_STAGE " + json.dumps({"stage": "peek", "rc": peek}), flush=True)
    print("RESIDENT_STAGE " + json.dumps({"stage": "caller_sync_started", "peek": peek}), flush=True)
    started = time.monotonic()
    rc = lib.aclrtSynchronizeStreamWithTimeout(caller, budget)
    stages.append("caller_sync")
    print("RESIDENT_STAGE " + json.dumps({"stage": "caller_sync", "rc": rc}), flush=True)
    return {
        "execution_stages": stages,
        "scheduler_runtime_mode": mode,
        "scheduler_mode_publish_count": count,
        "peek_before": peek,
        "caller_sync_rc": rc,
        "observer_seconds": time.monotonic() - started,
        "stream_sync_timeout_ms": budget,
    }


def _validate_resident_status(observed, failure, fixture, logs, shape=(4, 2)):
    assert shape in _RESIDENT_SHAPES
    assert observed["failure"] is failure
    assert observed["shape"] == list(shape) and observed["blocks"] == 1
    assert observed["execution_stages"] == _RESIDENT_OBSERVERS, observed
    assert observed["scheduler_runtime_mode"] in (1, 2) and observed["scheduler_mode_publish_count"] > 0
    assert observed["peek_before"] == 0
    assert observed["stream_sync_timeout_ms"] == int(fixture["timeout_profile"]["SIMPLER_STREAM_SYNC_TIMEOUT_MS"])
    assert observed["caller_sync_rc"] == (507018 if failure else 0), observed
    if failure:
        assert observed["cleanup"] == "isolated_child_exit_unconfirmed"
        deadline = time.monotonic() + 2
        while True:
            try:
                return _resident_timeout_evidence(logs, shape)
            except AssertionError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.1, remaining))
    assert observed["output_verified"] and observed["cleanup"] == "released"
    return []


def _validate_resident_output(records, summary, shape):
    assert shape in _RESIDENT_SHAPES
    assert len(records) == 4 * 16 and len(summary) == 16
    rows = [list(records[offset : offset + 16]) for offset in range(0, len(records), 16)]
    mask, _stalled_lane = shape
    for lane, row in enumerate(rows):
        if lane < 3 and mask & (1 << lane):
            assert row[:2] == [1, 1] and row[2:8] == [0] * 6, (lane, row)
            assert row[8:12] == [1, 1, lane + 1, 1] and 0 < row[12] <= row[13], (lane, row)
            assert row[14:] == [0, 0], (lane, row)
        else:
            assert row == [0] * 16, (lane, row)
    assert list(summary) == [bin(mask).count("1")] + [0] * 15


def _resident_cpu_output(shape):
    rows = [[0] * 16 for _ in range(4)]
    for lane in range(3):
        if shape[0] & (1 << lane):
            rows[lane] = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, lane + 1, 1, 10, 20, 0, 0]
    return [word for row in rows for word in row], [bin(shape[0]).count("1")] + [0] * 15


@pytest.mark.parametrize("shape", _RESIDENT_SHAPES)
def test_resident_shape_checks_all_active_rows_and_summary(shape):
    records, summary = _resident_cpu_output(shape)
    _validate_resident_output(records, summary, shape)
    for lane in range(3):
        damaged = records.copy()
        damaged[lane * 16] = 2
        with pytest.raises(AssertionError):
            _validate_resident_output(damaged, summary, shape)
    for lane in range(3):
        if shape[0] & (1 << lane):
            for word, value in ((1, 0), (2, 1), (10, 9), (12, 0), (13, 9), (15, 1)):
                damaged = records.copy()
                damaged[lane * 16 + word] = value
                with pytest.raises(AssertionError):
                    _validate_resident_output(damaged, summary, shape)
    records[3 * 16] = 1
    with pytest.raises(AssertionError):
        _validate_resident_output(records, summary, shape)
    records, summary = _resident_cpu_output(shape)
    for word in range(16):
        damaged = summary.copy()
        damaged[word] += 1
        with pytest.raises(AssertionError):
            _validate_resident_output(records, damaged, shape)


@pytest.mark.parametrize("shape", _RESIDENT_SHAPES[1:])
def test_resident_new_shape_requires_exact_owned_execution_timeout(tmp_path, shape):
    signature = "aicpu_execute: runtime failed with rc=-100\n"
    site = "A5 HBG AICore Scheduler timeout site=95\n"
    unfinished = f"A5 HBG AICore scheduler: unfinished task=9 state=1 active_mask=0x{shape[0]:x} flags=0x0\n"
    path = tmp_path / "device.log"
    path.write_text(signature + site + unfinished)
    assert len(_resident_timeout_evidence(tmp_path, shape)) == 3
    for text in (
        signature + unfinished,
        signature + site,
        signature + site.replace("95", "950") + unfinished,
        signature + site.replace("95", "91") + unfinished,
        signature + site + unfinished.replace(f"0x{shape[0]:x} ", f"0x{shape[0]:x}0 "),
        signature + site + unfinished.replace(f"0x{shape[0]:x} ", "0x4 "),
        signature.replace("-100", "-1000") + site + unfinished,
    ):
        path.write_text(text)
        with pytest.raises(AssertionError):
            _resident_timeout_evidence(tmp_path, shape)
    path.write_text(signature)
    foreign = tmp_path.parent / (tmp_path.name + "-foreign.log")
    foreign.write_text(site + unfinished)
    (tmp_path / "foreign.log").symlink_to(foreign)
    with pytest.raises(AssertionError):
        _resident_timeout_evidence(tmp_path, shape)


@pytest.mark.parametrize("shape", _RESIDENT_SHAPES[1:])
def test_resident_new_shape_status_rejects_wrong_shape_and_sdk_code(tmp_path, shape):
    fixture = {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}
    observed = _resident_observation_fixture()
    observed["shape"] = list(shape)
    (tmp_path / "device.log").write_text(
        "aicpu_execute: runtime failed with rc=-100\nA5 HBG AICore Scheduler timeout site=95\n"
        f"A5 HBG AICore scheduler: unfinished task=9 active_mask=0x{shape[0]:x} flags=0x0\n"
    )
    assert len(_validate_resident_status(observed, True, fixture, tmp_path, shape)) == 3
    for sdk in (0, 507015, 507011, -100):
        observed["caller_sync_rc"] = sdk
        with pytest.raises(AssertionError):
            _validate_resident_status(observed, True, fixture, tmp_path, shape)
    observed.update(caller_sync_rc=507018, shape=[4, 2])
    with pytest.raises(AssertionError):
        _validate_resident_status(observed, True, fixture, tmp_path, shape)


@pytest.mark.parametrize("signature", _RESIDENT_TIMEOUT_SIGNATURES)
def test_resident_timeout_accepts_its_owned_device_status(tmp_path, signature):
    (tmp_path / "device.log").write_text(signature.removesuffix(r"(?!\d)") + "\n")
    assert len(_resident_timeout_evidence(tmp_path)) == 1


@pytest.mark.parametrize(
    "message",
    [
        "caller sync 507018",
        "Scheduler fatal code=100",
        "aicpu_execute: runtime failed with rc=-101",
        "aicpu_execute: runtime failed with rc=-1000",
        "simpler_aicpu_exec: aicpu_execute failed with rc=-1000",
    ],
)
def test_resident_timeout_rejects_an_unqualified_error(tmp_path, message):
    (tmp_path / "device.log").write_text(message)
    with pytest.raises(AssertionError, match="No resident device -100"):
        _resident_timeout_evidence(tmp_path)


def test_resident_timeout_rejects_foreign_logs(tmp_path):
    logs = tmp_path / "owned"
    logs.mkdir()
    foreign = tmp_path / "foreign.log"
    foreign.write_text("aicpu_execute: runtime failed with rc=-100\n")
    (logs / "device.log").symlink_to(foreign)
    with pytest.raises(AssertionError, match="No resident device -100"):
        _resident_timeout_evidence(logs)


def _resident_observation_fixture():
    return {
        "failure": True,
        "shape": [4, 2],
        "blocks": 1,
        "execution_stages": list(_RESIDENT_OBSERVERS),
        "scheduler_runtime_mode": 1,
        "scheduler_mode_publish_count": 1,
        "peek_before": 0,
        "stream_sync_timeout_ms": 6000,
        "caller_sync_rc": 507018,
        "cleanup": "isolated_child_exit_unconfirmed",
    }


def test_resident_timeout_requires_both_the_sdk_status_and_owned_device_error(tmp_path):
    observed = _resident_observation_fixture()
    fixture = {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}
    with pytest.raises(AssertionError, match="No resident device -100"):
        _validate_resident_status(observed, True, fixture, tmp_path)
    (tmp_path / "device.log").write_text("aicpu_execute: runtime failed with rc=-100\n")
    assert len(_validate_resident_status(observed, True, fixture, tmp_path)) == 1


def test_resident_timeout_waits_only_for_owned_diagnostic_delivery(tmp_path, monkeypatch):
    logs = tmp_path / "owned"
    logs.mkdir()
    foreign = tmp_path / "foreign.log"
    foreign.write_text("aicpu_execute: runtime failed with rc=-100\n")
    (logs / "foreign.log").symlink_to(foreign)
    clock = [0.0]

    def deliver(seconds):
        clock[0] += seconds
        (logs / "device.log").write_text("aicpu_execute: runtime failed with rc=-100\n")

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", deliver)
    evidence = _validate_resident_status(
        _resident_observation_fixture(), True, {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}, logs
    )
    assert 0 < clock[0] <= 2
    assert [match["path"] for match in evidence] == ["device.log"]


def test_resident_timeout_diagnostic_delivery_expires_without_qualifying_a_foreign_log(tmp_path, monkeypatch):
    logs = tmp_path / "owned"
    logs.mkdir()
    foreign = tmp_path / "foreign.log"
    foreign.write_text("aicpu_execute: runtime failed with rc=-100\n")
    (logs / "foreign.log").symlink_to(foreign)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(AssertionError, match="No resident device -100"):
        _validate_resident_status(
            _resident_observation_fixture(), True, {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}, logs
        )
    assert clock[0] == pytest.approx(2)


@pytest.mark.parametrize("mode", [0, 3, 4, 5])
def test_resident_timeout_refuses_a_legacy_or_unpublished_mode(tmp_path, mode):
    observed = _resident_observation_fixture()
    observed["scheduler_runtime_mode"] = mode
    with pytest.raises(AssertionError):
        _validate_resident_status(observed, True, {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}, tmp_path)


@pytest.mark.parametrize("sdk_code", [0, 507015, 507011, -100])
def test_resident_timeout_requires_the_reported_kernel_sdk_code(tmp_path, sdk_code):
    observed = _resident_observation_fixture()
    observed["caller_sync_rc"] = sdk_code
    (tmp_path / "device.log").write_text("aicpu_execute: runtime failed with rc=-100\n")
    with pytest.raises(AssertionError):
        _validate_resident_status(observed, True, {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}, tmp_path)


@pytest.mark.parametrize("observer", ["native_wait", "native_finalize", "private_result_d2h", "output_d2h"])
def test_resident_timeout_refuses_a_prior_execution_observer(tmp_path, observer):
    observed = _resident_observation_fixture()
    observed["execution_stages"].insert(2, observer)
    with pytest.raises(AssertionError):
        _validate_resident_status(observed, True, {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}, tmp_path)


def test_resident_entry_observes_the_caller_without_a_native_drain(capsys):
    calls = []
    before_sync = []

    def leaf(name, result=0):
        def call(*_args):
            calls.append(name)
            return result

        return call

    def caller_sync(*_args):
        calls.append("caller_sync")
        facts = [
            json.loads(line.removeprefix("RESIDENT_STAGE "))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("RESIDENT_STAGE ")
        ]
        before_sync.extend(facts)
        assert [fact["stage"] for fact in facts] == ["prepare", "launch", "peek", "caller_sync_started"]
        return 507018

    lib = SimpleNamespace(
        **{
            "simpler_prepare_run": leaf("prepare"),
            "get_last_scheduler_runtime_mode": leaf("mode", 2),
            "get_scheduler_mode_publish_count": leaf("count", 1),
            "simpler_launch_run_on_stream": leaf("launch"),
            "aclrtPeekAtLastError": leaf("peek"),
            "aclrtSynchronizeStreamWithTimeout": caller_sync,
        }
    )
    observed = _observe_resident_entry(
        lib, None, None, None, SimpleNamespace(__ptr__=lambda: None), ctypes.c_int(), ctypes.c_int(), 6000
    )
    assert calls == ["prepare", "mode", "count", "launch", "peek", "caller_sync"]
    assert observed["execution_stages"] == _RESIDENT_OBSERVERS and observed["caller_sync_rc"] == 507018
    after_sync = [
        json.loads(line.removeprefix("RESIDENT_STAGE "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("RESIDENT_STAGE ")
    ]
    assert [fact["stage"] for fact in before_sync + after_sync] == [
        "prepare",
        "launch",
        "peek",
        "caller_sync_started",
        "caller_sync",
    ]
    assert after_sync == [{"stage": "caller_sync", "rc": 507018}]


@pytest.mark.parametrize("shape", _RESIDENT_SHAPES)
@pytest.mark.parametrize("failure", [False, True])
def test_resident_child_preserves_its_owners_after_the_first_caller_error(
    tmp_path, monkeypatch, capsys, shape, failure
):
    import test_kernel_mode_c_api as native_api  # noqa: PLC0415

    calls = []
    weak_owners, events, waited = [], [], []
    forbidden = {
        "simpler_wait_run",
        "simpler_poll_run",
        "simpler_finalize_run",
        "finalize_device",
        "aclrtMemcpy",
        "aclrtFree",
        "aclrtDestroyStream",
        "aclrtDestroyContext",
        "aclrtResetDevice",
        "aclFinalize",
        "destroy_device_context",
    }

    class SdkLeaves:
        def __getattr__(self, name):
            def call(*args):
                calls.append(name)
                assert not failure or name not in forbidden, f"Unexpected observer or cleanup: {name}"
                if name == "aclrtMemcpy":
                    records, summary = _resident_cpu_output(shape)
                    values = records if args[1] == 4 * 128 else summary
                    image = (ctypes.c_int64 * len(values))(*values)
                    ctypes.memmove(args[0], image, args[1])
                if name in {"aclrtCreateContext", "aclrtCreateStream", "aclrtMalloc"}:
                    weak_owners.append(weakref.ref(args[0]._obj))
                    args[0]._obj.value = 0x1000 + len(calls) * 0x1000
                return {
                    "create_device_context": 1,
                    "get_last_scheduler_runtime_mode": 2,
                    "get_scheduler_mode_publish_count": 1,
                    "aclrtSynchronizeStreamWithTimeout": 507018 if failure else 0,
                }.get(name, 0)

            setattr(self, name, call)
            return call

    class Carrier:
        pass

    def storage(_lib):
        carrier, run = Carrier(), ctypes.c_void_p(0x8000)
        weak_owners.extend([weakref.ref(carrier), weakref.ref(run)])
        return carrier, run

    class ContainedExit(Exception):
        pass

    def contained_exit(code):
        assert code == 0 and events[-1] == "wait"
        raise ContainedExit

    for name, value in _RESIDENT_TIMEOUT_PROFILE.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "resident-fixture.json").write_text(
        json.dumps({"timeout_profile": _RESIDENT_TIMEOUT_PROFILE, "stall_ms": 2500})
    )
    (tmp_path / "callable.bin").write_bytes(b"not executed by SDK leaves")
    monkeypatch.setattr(native_api, "_load", lambda *_args: SdkLeaves())
    monkeypatch.setattr(native_api, "_binaries", lambda *_args: (b"cpu", b"core", b"dispatcher"))
    monkeypatch.setattr(native_api, "_native_run_storage", storage)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(os, "_exit", contained_exit)
    original_observer = _observe_resident_entry
    original_wait = _wait_for_resident_timeout_evidence
    original_print = print
    logs = tmp_path / "ascend"
    logs.mkdir()
    (logs / "device.log").write_text(
        "aicpu_execute: runtime failed with rc=-100\nA5 HBG AICore Scheduler timeout site=95\n"
        f"A5 HBG AICore scheduler: unfinished task=9 active_mask=0x{shape[0]:x} flags=0x0\n"
    )
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(logs))

    def printed(*values, **kwargs):
        if values and str(values[0]).startswith("RESIDENT_CALLER_STATUS "):
            events.append("status")
        original_print(*values, **kwargs)

    def check_wait(owned, selected_shape):
        assert events[-1] == "status" and selected_shape == shape and owned == logs
        assert len(weak_owners) == 6 and all(owner() is not None for owner in weak_owners)
        assert calls[-3:] == [
            "simpler_launch_run_on_stream",
            "aclrtPeekAtLastError",
            "aclrtSynchronizeStreamWithTimeout",
        ]
        before = list(calls)
        assert original_wait(owned, selected_shape)
        assert calls == before
        assert all(owner() is not None for owner in weak_owners)
        waited.append(True)
        events.append("wait")

    monkeypatch.setattr(sys.modules[__name__], "print", printed, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "_wait_for_resident_timeout_evidence", check_wait)

    def check_scalars(lib, ctx, run, caller, args, *remaining):
        assert [args.scalar(i) for i in range(4)] == [1, *shape, 2500 if failure else 0]
        return original_observer(lib, ctx, run, caller, args, *remaining)

    monkeypatch.setattr(sys.modules[__name__], "_observe_resident_entry", check_scalars)
    if failure:
        with pytest.raises(ContainedExit):
            _probe_resident_timeout(0, True, tmp_path, shape)
        assert calls[-3:] == [
            "simpler_launch_run_on_stream",
            "aclrtPeekAtLastError",
            "aclrtSynchronizeStreamWithTimeout",
        ]
    else:
        _probe_resident_timeout(0, False, tmp_path, shape)
        assert calls.index("aclrtMemcpy") > calls.index("aclrtSynchronizeStreamWithTimeout")
        assert calls.index("simpler_finalize_run") > calls.index("aclrtMemcpy")
    assert waited == ([True] if failure else [])
    assert calls.count("aclrtSynchronizeStreamWithTimeout") == 1
    status = next(
        line.removeprefix("RESIDENT_CALLER_STATUS ")
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("RESIDENT_CALLER_STATUS ")
    )
    observed = json.loads(status)
    assert observed["shape"] == list(shape) and observed["failure"] is failure
    assert observed["cleanup"] == ("isolated_child_exit_unconfirmed" if failure else "released")


@pytest.mark.parametrize("name", list(_RESIDENT_TIMEOUT_PROFILE))
def test_resident_timeout_preserves_the_existing_ci_budgets(name):
    profile = dict(_RESIDENT_TIMEOUT_PROFILE)
    assert _resident_timeout_profile(profile) == profile
    profile[name] = "1"
    with pytest.raises(AssertionError, match="existing CI budgets"):
        _resident_timeout_profile(profile)


def _compile_resident_fixture(output):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    started = time.monotonic()
    profile = _resident_timeout_profile(os.environ)
    compiler = KernelCompiler("a5")
    sources = _ROOT / "tests/fdwic/pa_hbg/spmd/kernels"
    includes = list(compiler.get_orchestration_include_dirs("host_build_graph"))
    includes.append(str(compiler.platform_dir / "onboard/aicore"))
    children = []
    source_hashes = {}
    for func_id in range(4):
        source = sources / (f"runtime_failure_{func_id}.cpp" if func_id < 3 else "verify_fanin.cpp")
        signature = [ArgDirection.INOUT] if func_id < 3 else [ArgDirection.IN, ArgDirection.INOUT]
        binary = compiler.compile_incore(
            source_path=str(source),
            core_type="aic" if func_id == 0 else "aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=includes,
        )
        (output / f"core-{func_id}.elf").write_bytes(binary)
        children.append((func_id, CoreCallable.build(signature=signature, binary=extract_text_section(binary))))
        source_hashes[str(source.relative_to(_ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    source = sources / "runtime_failure_orchestration.cpp"
    binary = compiler.compile_orchestration(
        runtime_name="host_build_graph", source_path=str(source), build_dir=str(output)
    )
    chip = ChipCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.INOUT],
        func_name="aicpu_orchestration_entry",
        binary=binary,
        children=children,
    )
    (output / "callable.bin").write_bytes(ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size())))
    source_hashes[str(source.relative_to(_ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("runtime_failure_probe.h", "execution_record.h", "record.h"):
        source = sources / name
        source_hashes[str(source.relative_to(_ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    fixture = {
        "timeout_profile": profile,
        "shape": [4, 2],
        "blocks": 1,
        "stall_ms": 2500,
        "source_sha256": source_hashes,
        "build_seconds": time.monotonic() - started,
    }
    fixture["artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
        if path.is_file() and path.name != "resident-fixture.json"
    }
    (output / "resident-fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture


def _run_resident_child(argv, output, *, env):
    stdout_path, stderr_path = output / "caller-stdout.log", output / "caller-stderr.log"
    timeout_error = None
    result = None
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        try:
            result = subprocess.run(argv, stdout=stdout_file, stderr=stderr_file, timeout=120, check=False, env=env)
        except subprocess.TimeoutExpired as error:
            timeout_error = error
    if timeout_error is not None:
        for path, partial in ((stdout_path, timeout_error.stdout), (stderr_path, timeout_error.stderr)):
            if partial:
                with path.open("ab") as stream:
                    stream.write(partial if isinstance(partial, bytes) else partial.encode())

    def tail(path):
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 1024 * 1024))
            return stream.read().decode(errors="replace")

    stdout, stderr = tail(stdout_path), tail(stderr_path)
    notice = "Resident child timed out; stop/release unconfirmed\n" if timeout_error is not None else ""
    (output / "caller.log").write_text(notice + stdout + stderr)
    if timeout_error is not None:
        raise AssertionError(notice + stdout + stderr) from timeout_error
    assert result is not None
    return result, stdout, stderr


def test_resident_child_timeout_preserves_partial_bytes_and_stages(tmp_path, monkeypatch):
    partial = b'RESIDENT_STAGE {"stage":"launch","rc":0}\n'

    def timed_out(argv, *, stdout, stderr, timeout, **_kwargs):
        assert timeout == 120
        stdout.write(b"owned stdout prefix\n")
        stderr.write(b"owned stderr prefix\n")
        raise subprocess.TimeoutExpired(argv, timeout, output=partial, stderr=b"partial SDK log\n")

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(AssertionError, match="stop/release unconfirmed"):
        _run_resident_child(["not-executed"], tmp_path, env={})
    assert (tmp_path / "caller-stdout.log").read_bytes() == b"owned stdout prefix\n" + partial
    text = (tmp_path / "caller.log").read_text()
    assert partial.decode() in text and "owned stderr prefix" in text and "partial SDK log" in text


def _resident_failure_context(logs, stdout, stderr, records):
    context = {"observed": records, "stdout_tail": stdout[-16384:], "stderr_tail": stderr[-16384:], "owned_logs": []}
    try:
        root = logs.resolve()
        for path in sorted(logs.rglob("*.log")):
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            if len(context["owned_logs"]) == 64:
                context["inventory_truncated"] = True
                break
            item = {"path": str(path.relative_to(logs))}
            context["owned_logs"].append(item)
            try:
                item["bytes"] = path.stat().st_size
                if len(context["owned_logs"]) <= 8:
                    with path.open("rb") as stream:
                        stream.seek(max(0, item["bytes"] - 8192))
                        item["tail"] = stream.read(8192).decode(errors="replace")
            except OSError as error:
                item["read_error"] = str(error)
    except OSError as error:
        context["inventory_error"] = str(error)
    return context


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("host_build_graph")
@pytest.mark.parametrize("shape", _RESIDENT_SHAPES, ids=["aiv", "aic", "mix-aic", "mix-aiv"])
def test_resident_timeout_reaches_the_first_caller_observer(st_device_ids, tmp_path, shape):
    fixture = _compile_resident_fixture(tmp_path)
    for failure in (False, True):
        output = tmp_path / ("bounded-timeout" if failure else "success")
        output.mkdir()
        logs = output / "ascend"
        logs.mkdir()
        result, stdout, stderr = _run_resident_child(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "resident-timeout",
                str(st_device_ids[0]),
                str(int(failure)),
                str(tmp_path),
                str(shape[0]),
                str(shape[1]),
            ],
            output,
            env={**os.environ, "ASCEND_PROCESS_LOG_PATH": str(logs), "ASCEND_GLOBAL_LOG_LEVEL": "3"},
        )
        records = []
        try:
            assert result.returncode == 0, stdout + stderr
            records = [
                json.loads(line.removeprefix("RESIDENT_CALLER_STATUS "))
                for line in stdout.splitlines()
                if line.startswith("RESIDENT_CALLER_STATUS ")
            ]
            assert len(records) == 1, stdout + stderr
            evidence = _validate_resident_status(records[0], failure, fixture, logs, shape)
        except (AssertionError, ValueError, OSError):
            context = _resident_failure_context(logs, stdout, stderr, records)
            print("RESIDENT_FAILURE_CONTEXT " + json.dumps(context), flush=True)
            raise
        (output / "result.json").write_text(json.dumps({"observed": records[0], "device_error": evidence}, indent=2))


def test_resident_validation_failure_exposes_bounded_owned_context(tmp_path, monkeypatch, capsys):
    fixture = {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}
    foreign = tmp_path / "foreign.log"
    foreign.write_text("foreign diagnostic contents")
    monkeypatch.setattr(sys.modules[__name__], "_compile_resident_fixture", lambda _output: fixture)

    def child(_argv, output, *, env):
        observed = _resident_observation_fixture()
        failure = output.name == "bounded-timeout"
        observed.update(failure=failure, caller_sync_rc=507015 if failure else 0)
        if not failure:
            observed.update(output_verified=True, cleanup="released")
        logs = Path(env["ASCEND_PROCESS_LOG_PATH"])
        (logs / "device.log").write_text("x" * 20000 + "owned device tail")
        (logs / "foreign.log").symlink_to(foreign)
        stdout = "s" * 20000 + "\nRESIDENT_CALLER_STATUS " + json.dumps(observed)
        return SimpleNamespace(returncode=0), stdout, "owned SDK stderr"

    monkeypatch.setattr(sys.modules[__name__], "_run_resident_child", child)
    with pytest.raises(AssertionError) as failure:
        test_resident_timeout_reaches_the_first_caller_observer([0], tmp_path, (4, 2))
    assert "507015" in str(failure.value)
    lines = capsys.readouterr().out.splitlines()
    context = json.loads(next(line.removeprefix("RESIDENT_FAILURE_CONTEXT ") for line in lines))
    assert context["observed"][0]["caller_sync_rc"] == 507015
    assert len(context["stdout_tail"]) == 16384 and context["stderr_tail"] == "owned SDK stderr"
    assert [item["path"] for item in context["owned_logs"]] == ["device.log"]
    assert context["owned_logs"][0]["bytes"] > 8192
    assert len(context["owned_logs"][0]["tail"]) == 8192
    assert context["owned_logs"][0]["tail"].endswith("owned device tail")
    assert "foreign diagnostic contents" not in json.dumps(context)


@pytest.mark.parametrize("shape", _RESIDENT_SHAPES)
def test_resident_matched_pair_selects_one_fixture_and_the_same_shape(tmp_path, monkeypatch, shape):
    builds = []
    children = []
    fixture = {"timeout_profile": _RESIDENT_TIMEOUT_PROFILE}
    monkeypatch.setattr(
        sys.modules[__name__], "_compile_resident_fixture", lambda output: builds.append(output) or fixture
    )

    def child(argv, output, *, env):
        assert argv[-2:] == [str(shape[0]), str(shape[1])]
        assert argv[-3] == str(tmp_path)
        failure = output.name == "bounded-timeout"
        children.append(failure)
        observed = _resident_observation_fixture()
        observed.update(shape=list(shape), failure=failure, caller_sync_rc=507018 if failure else 0)
        if not failure:
            observed.update(output_verified=True, cleanup="released")
        logs = Path(env["ASCEND_PROCESS_LOG_PATH"])
        (logs / "device.log").write_text(
            "aicpu_execute: runtime failed with rc=-100\nA5 HBG AICore Scheduler timeout site=95\n"
            f"A5 HBG AICore scheduler: unfinished task=19 active_mask=0x{shape[0]:x} flags=0x0\n"
        )
        return SimpleNamespace(returncode=0), "RESIDENT_CALLER_STATUS " + json.dumps(observed), ""

    monkeypatch.setattr(sys.modules[__name__], "_run_resident_child", child)
    test_resident_timeout_reaches_the_first_caller_observer([0], tmp_path, shape)
    assert builds == [tmp_path] and children == [False, True]
    for name in ("success", "bounded-timeout"):
        result = json.loads((tmp_path / name / "result.json").read_text())
        assert result["observed"]["shape"] == list(shape)


def _bind_resident_sdk(lib):
    declarations = {
        "aclInit": [ctypes.c_char_p],
        "aclFinalize": [],
        "aclrtSetDevice": [ctypes.c_int],
        "aclrtResetDevice": [ctypes.c_int],
        "aclrtCreateContext": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "aclrtSetCurrentContext": [ctypes.c_void_p],
        "aclrtDestroyContext": [ctypes.c_void_p],
        "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtDestroyStream": [ctypes.c_void_p],
        "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
        "aclrtFree": [ctypes.c_void_p],
        "aclrtMemset": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t],
        "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "aclrtPeekAtLastError": [ctypes.c_int],
        "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int],
        "simpler_prepare_run": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ],
        "simpler_launch_run_on_stream": [ctypes.c_void_p] * 4,
        "simpler_finalize_run": [ctypes.c_void_p] * 2,
        "get_last_scheduler_runtime_mode": [ctypes.c_void_p],
        "get_scheduler_mode_publish_count": [ctypes.c_void_p],
    }
    for name, arguments in declarations.items():
        function = getattr(lib, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    lib.get_last_scheduler_runtime_mode.restype = ctypes.c_uint32
    lib.get_scheduler_mode_publish_count.restype = ctypes.c_size_t
    return sorted(declarations)


def _probe_resident_timeout(device, failure, output, shape=(4, 2)):  # noqa: PLR0915 -- one child owns the borrowed context
    from simpler.task_interface import (  # noqa: PLC0415
        AddressSpace,
        ChipStorageTaskArgs,
        ChipTensor,
        DataType,
        TensorTransfer,
    )
    from test_kernel_mode_c_api import (  # noqa: PLC0415
        CallConfig,
        NativeRunDescriptor,
        _binaries,
        _load,
        _native_run_storage,
    )

    assert shape in _RESIDENT_SHAPES
    fixture = json.loads((output / "resident-fixture.json").read_text())
    _resident_timeout_profile(os.environ)
    lib = _load("a5", "onboard", "host_build_graph")
    _bind_resident_sdk(lib)
    assert lib.aclInit(None) == 0
    assert lib.aclrtSetDevice(device) == 0
    owner, caller = ctypes.c_void_p(), ctypes.c_void_p()
    assert lib.aclrtCreateContext(ctypes.byref(owner), device) == 0
    assert lib.aclrtSetCurrentContext(owner) == 0
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    backings = [ctypes.c_void_p(), ctypes.c_void_p()]
    sizes = (4 * 128, 16 * 8)
    for backing, size in zip(backings, sizes):
        assert lib.aclrtMalloc(ctypes.byref(backing), size, 0) == 0
        assert lib.aclrtMemset(backing, size, 0, size) == 0
    ctx = lib.create_device_context()
    assert ctx
    config = CallConfig()
    config.aicpu_thread_num = 2
    aicpu, aicore, dispatcher = _binaries("a5", "host_build_graph")
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    image = (output / "callable.bin").read_bytes()
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    carrier, run = _native_run_storage(lib)
    args = ChipStorageTaskArgs()
    for backing, size in zip(backings, sizes):
        args.add_tensor(
            ChipTensor.make(backing.value, (size // 8,), DataType.INT64, address_space=AddressSpace.DEVICE),
            transfer=TensorTransfer.NONE,
        )
    for scalar in (1, *shape, fixture["stall_ms"] if failure else 0):
        args.add_scalar(scalar)
    descriptor = NativeRunDescriptor(generation=1, dispatch_id=1, run_epoch=1)
    observed = _observe_resident_entry(
        lib,
        ctx,
        run,
        caller,
        args,
        config,
        descriptor,
        int(fixture["timeout_profile"]["SIMPLER_STREAM_SYNC_TIMEOUT_MS"]),
    )
    observed.update(failure=failure, shape=list(shape), blocks=1, output_verified=False)
    if failure:
        observed["cleanup"] = "isolated_child_exit_unconfirmed"
        print("RESIDENT_CALLER_STATUS " + json.dumps(observed), flush=True)
        assert carrier is not None and ctx and owner.value and caller.value and all(p.value for p in backings)
        _wait_for_resident_timeout_evidence(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]), shape)
        os._exit(0)
    assert observed["caller_sync_rc"] == 0
    records = (ctypes.c_int64 * (sizes[0] // 8))()
    summary = (ctypes.c_int64 * (sizes[1] // 8))()
    for host, backing, size in zip((records, summary), backings, sizes):
        assert lib.aclrtMemcpy(host, size, backing, size, 2) == 0
    _validate_resident_output(records, summary, shape)
    observed["output_verified"] = True
    assert lib.simpler_finalize_run(ctx, run) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    for backing in backings:
        assert lib.aclrtFree(backing) == 0
    assert lib.aclrtDestroyStream(caller) == 0
    assert lib.aclrtDestroyContext(owner) == 0
    assert lib.aclrtResetDevice(device) == 0
    assert lib.aclFinalize() == 0
    observed["cleanup"] = "released"
    print("RESIDENT_CALLER_STATUS " + json.dumps(observed), flush=True)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.parametrize(
    ("runtime", "error_origin"),
    [
        ("tensormap_and_ringbuffer", "aicpu"),
        ("tensormap_and_ringbuffer", "aicore"),
        ("host_build_graph", "aicore"),
    ],
)
@pytest.mark.parametrize("failure", [False, True], ids=["success", "reported-error"])
def test_kernel_error_reaches_caller_stream(runtime, error_origin, failure, request, tmp_path):
    arch = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            arch,
            runtime,
            error_origin,
            device,
            str(int(failure)),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "ASCEND_PROCESS_LOG_PATH": str(logs)},
    )
    (tmp_path / "native-caller.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        line.removeprefix("NATIVE_CALLER_STATUS ")
        for line in result.stdout.splitlines()
        if line.startswith("NATIVE_CALLER_STATUS ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    observed = json.loads(records[0])
    assert observed["runtime"] == runtime
    assert observed["error_origin"] == error_origin
    assert observed["failure"] is failure
    assert observed["peek_before"] == 0, observed
    assert observed["caller_sync_rc"] == (507018 if failure else 0), observed
    if error_origin == "aicore" and not failure:
        assert observed["output_marker"] == 42, observed
    if arch == "a5" and runtime == "host_build_graph":
        assert observed["scheduler_runtime_mode"] == _A5_LEGACY_GRAPH, observed
        assert observed["scheduler_mode_publish_count"] > 0, observed


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_completed_public_handle_observes_later_caller_error(runtime, request, tmp_path):
    arch = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "public-late-observer", arch, runtime, device, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "ASCEND_PROCESS_LOG_PATH": str(logs)},
    )
    (tmp_path / "public-late-observer.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line.removeprefix("PUBLIC_LATE_OBSERVER "))
        for line in result.stdout.splitlines()
        if line.startswith("PUBLIC_LATE_OBSERVER ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    observed = records[0]
    assert observed["runtime"] == runtime and observed["platform"] == arch
    assert observed["old_done_before_later_launch"] and observed["old_done_after_error"]
    assert observed["peek_before_wait"] == 0 and observed["wait_error"] == 507018
    assert observed["repeat_wait_error"] == 507018 and observed["distinct_caller_outputs"]
    assert observed["caller_resources_retained"]
    if arch == "a5" and runtime == "host_build_graph":
        assert observed["scheduler_runtime_mode"] == _A5_LEGACY_GRAPH, observed
        assert observed["scheduler_mode_publish_count"] > 0, observed


def _callable(arch, runtime, error_origin, failure, output):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(arch)
    children = []
    signature = []
    kernel_source = None
    kernel_binary = None
    if error_origin == "aicore":
        source = Path(__file__).with_name("support") / (
            "kernel_stream_graph.cpp" if runtime == "host_build_graph" else "kernel_stream_task.cpp"
        )
        kernel_source = (
            _ROOT / "tests/st/runtime_fatal_codes/kernels/aiv" / "kernel_async_completion_invalid.cpp"
            if failure
            else Path(__file__).with_name("support") / "kernel_stream_output.cpp"
        )
        kernel = compiler.compile_incore(
            source_path=str(kernel_source),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=list(compiler.get_orchestration_include_dirs(runtime)) + [str(_ROOT / "src/common")],
        )
        signature = [ArgDirection.INOUT]
        kernel_binary = extract_text_section(kernel)
        children = [(0, CoreCallable.build(signature=signature, binary=kernel_binary))]
    else:
        source = (
            _ROOT / "tests/st/automatic_reported_fatal.cpp"
            if failure
            else Path(__file__).with_name("support") / "kernel_stream_success.cpp"
        )
    binary = compiler.compile_orchestration(runtime_name=runtime, source_path=str(source), build_dir=str(output))
    chip = ChipCallable.build(
        signature=signature, func_name="aicpu_orchestration_entry", binary=binary, children=children
    )
    hashes = {
        "fixture_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "orchestration_sha256": hashlib.sha256(binary).hexdigest(),
    }
    hashes["callable_image_sha256"] = hashlib.sha256(
        ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    ).hexdigest()
    if kernel_source is not None:
        assert kernel_binary is not None
        hashes["kernel_fixture_sha256"] = hashlib.sha256(kernel_source.read_bytes()).hexdigest()
        hashes["core_binary_sha256"] = hashlib.sha256(kernel_binary).hexdigest()
    return chip, hashes


def _probe(arch, runtime, error_origin, device, failure, output):
    from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415
    from test_kernel_mode_c_api import (  # noqa: PLC0415
        CallConfig,
        NativeRunDescriptor,
        _binaries,
        _load,
        _native_run_storage,
    )

    chip, hashes = _callable(arch, runtime, error_origin, failure, output)
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    (output / "callable.bin").write_bytes(image)
    lib = _load(arch, "onboard", runtime)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtPeekAtLastError.argtypes = [ctypes.c_int]
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.simpler_prepare_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(CallConfig),
        ctypes.POINTER(NativeRunDescriptor),
    ]
    lib.simpler_launch_run_on_stream.argtypes = [ctypes.c_void_p] * 4
    lib.simpler_finalize_run.argtypes = [ctypes.c_void_p] * 2
    assert lib.rtSetDevice(device) == 0
    caller = ctypes.c_void_p()
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    ctx = lib.create_device_context()
    device_tensor = ctypes.c_void_p()
    if error_origin == "aicore":
        lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        lib.aclrtFree.argtypes = [ctypes.c_void_p]
        lib.aclrtMemset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t]
        lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        assert lib.aclrtMalloc(ctypes.byref(device_tensor), 4, 0) == 0
        assert lib.aclrtMemset(device_tensor, 4, 0, 4) == 0
    config = CallConfig()
    config.aicpu_thread_num = 2
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    assert (
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 1
        )
        == 0
    )
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    storage, run = _native_run_storage(lib)
    args = ChipStorageTaskArgs()
    if device_tensor.value is not None:
        args.add_tensor(ChipTensor.make(device_tensor.value, (1,), DataType.INT32, address_space=AddressSpace.DEVICE))
    descriptor = NativeRunDescriptor(generation=1, run_epoch=1)
    assert lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor)) == 0
    assert lib.simpler_launch_run_on_stream(ctx, run, caller, None) == 0
    # This is the first execution observer: neither native wait nor finalize ran.
    peek_before = lib.aclrtPeekAtLastError(0)
    started = time.monotonic()
    rc = lib.aclrtSynchronizeStreamWithTimeout(caller, 60000)
    elapsed = time.monotonic() - started
    mode = count = 0
    if arch == "a5" and runtime == "host_build_graph":
        lib.get_last_scheduler_runtime_mode.argtypes = [ctypes.c_void_p]
        lib.get_last_scheduler_runtime_mode.restype = ctypes.c_uint32
        lib.get_scheduler_mode_publish_count.argtypes = [ctypes.c_void_p]
        lib.get_scheduler_mode_publish_count.restype = ctypes.c_size_t
        mode = lib.get_last_scheduler_runtime_mode(ctx)
        count = lib.get_scheduler_mode_publish_count(ctx)
    marker = ctypes.c_int32(-1)
    if error_origin == "aicore" and not failure and rc == 0:
        assert lib.aclrtMemcpy(ctypes.byref(marker), 4, device_tensor, 4, 2) == 0
    print(
        "NATIVE_CALLER_STATUS "
        + json.dumps(
            {
                "runtime": runtime,
                "error_origin": error_origin,
                "scheduler_runtime_mode": mode,
                "scheduler_mode_publish_count": count,
                "output_marker": marker.value,
                "observer_seconds": elapsed,
                "failure": failure,
                "caller_sync_rc": rc,
                "peek_before": peek_before,
                **hashes,
            }
        ),
        flush=True,
    )
    if failure:
        # A fresh child samples exactly one error observer; teardown cannot add another.
        time.sleep(1)
        os._exit(0)
    assert lib.simpler_finalize_run(ctx, run) == 0
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
    assert lib.aclrtSynchronizeStreamWithTimeout(caller, 60000) == 0
    assert lib.aclrtDestroyStream(caller) == 0
    if device_tensor.value is not None:
        assert lib.aclrtFree(device_tensor) == 0
    assert storage is not None


def _public_observer_probe(arch, runtime, device, output):
    from simpler.buffer import Buffer  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415
    from test_kernel_mode_c_api import _load  # noqa: PLC0415

    lib = _load(arch, "onboard", runtime)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMemset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t]
    lib.aclrtPeekAtLastError.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    caller = ctypes.c_void_p()
    assert lib.aclrtCreateStream(ctypes.byref(caller)) == 0
    backings = [ctypes.c_void_p(), ctypes.c_void_p()]
    for backing in backings:
        assert lib.aclrtMalloc(ctypes.byref(backing), 4, 0) == 0
        assert lib.aclrtMemset(backing, 4, 0, 4) == 0
    assert backings[0].value != backings[1].value
    specs, source_hashes = [], []
    for failure, name in ((False, "success"), (True, "reported-error")):
        directory = output / name
        directory.mkdir()
        chip, hashes = _callable(arch, runtime, "aicore", failure, directory)
        specs.append(chip)
        source_hashes.append(hashes)
    worker = Worker(level=2, device_id=device, platform=arch, runtime=runtime)
    handles = [worker.register(spec) for spec in specs]
    worker.init(borrowed_context_generation=1)
    buffers, task_args = [], []
    for backing in backings:
        assert backing.value is not None
        buffer = Buffer.wrap(address=backing.value, nbytes=4, location=worker.device_location)
        buffers.append(buffer)
        args = TaskArgs()
        args.add_tensor(Tensor(buffer, shapes=(1,), dtype=DataType.INT32), TensorArgType.INOUT)
        task_args.append(args)
    first = worker.submit(handles[0], task_args[0], caller_stream=caller.value)
    deadline = time.monotonic() + 30
    while not first.done:
        assert time.monotonic() < deadline, "Healthy run did not reach its own completion boundary"
    second = worker.submit(handles[1], task_args[1], caller_stream=caller.value)
    mode, count = worker.scheduler_runtime_mode, worker.scheduler_mode_publish_count
    peek_before = lib.aclrtPeekAtLastError(0)
    observed_errors = []
    for _ in range(2):
        try:
            first.wait()
        except RuntimeError as error:
            assert "507018" in str(error), repr(error)
            observed_errors.append(507018)
        else:
            raise AssertionError("Completed handle cached success instead of observing its actual caller stream")
    print(
        "PUBLIC_LATE_OBSERVER "
        + json.dumps(
            dict(
                platform=arch,
                runtime=runtime,
                old_done_before_later_launch=True,
                old_done_after_error=first.done,
                peek_before_wait=peek_before,
                wait_error=observed_errors[0],
                repeat_wait_error=observed_errors[1],
                distinct_caller_outputs=backings[0].value != backings[1].value,
                caller_resources_retained=bool(second is not None and len(buffers) == len(task_args) == 2),
                scheduler_runtime_mode=mode,
                scheduler_mode_publish_count=count,
                source_hashes=source_hashes,
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    # A borrowed context with an unresolved failing successor retains every owner until process exit.
    os._exit(0)


if __name__ == "__main__" and sys.argv[1] == "resident-timeout":
    _probe_resident_timeout(
        int(sys.argv[2]),
        bool(int(sys.argv[3])),
        Path(sys.argv[4]),
        (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) == 7 else (4, 2),
    )
elif __name__ == "__main__" and sys.argv[1] == "public-late-observer":
    _public_observer_probe(sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]))
elif __name__ == "__main__":
    _probe(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), bool(int(sys.argv[5])), Path(sys.argv[6]))
