# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for PTO-ISA revision resolution and build metadata."""

import json
import os
import subprocess

import pytest

from simpler_setup import pto_isa

PIN_A = "a" * 40
PIN_B = "b" * 40
PIN_C = "c" * 40


def test_read_pto_isa_pin_reads_valid_pin(tmp_path):
    pin = tmp_path / "pto_isa.pin"
    pin.write_text(f"{PIN_A}\n")

    assert pto_isa.read_pto_isa_pin(pin) == PIN_A


def test_read_pto_isa_pin_missing_warns_and_returns_none(tmp_path, caplog):
    caplog.set_level("WARNING", logger="simpler_setup.pto_isa")

    assert pto_isa.read_pto_isa_pin(tmp_path / "missing.pin") is None
    assert "falling back to latest pto-isa" in caplog.text


def test_read_pto_isa_pin_empty_warns_and_returns_none(tmp_path, caplog):
    pin = tmp_path / "pto_isa.pin"
    pin.write_text("\n")
    caplog.set_level("WARNING", logger="simpler_setup.pto_isa")

    assert pto_isa.read_pto_isa_pin(pin) is None
    assert "is empty" in caplog.text


def test_read_pto_isa_pin_rejects_invalid_content(tmp_path):
    pin = tmp_path / "pto_isa.pin"
    pin.write_text("not-a-sha\n")

    with pytest.raises(RuntimeError, match="Invalid PTO-ISA pin"):
        pto_isa.read_pto_isa_pin(pin)


def test_resolve_pto_isa_commit_prefers_explicit_over_pin():
    assert pto_isa.resolve_pto_isa_commit(PIN_A) == PIN_A


def test_resolve_pto_isa_commit_uses_pin_by_default(monkeypatch):
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_C)

    assert pto_isa.resolve_pto_isa_commit() == PIN_C


@pytest.mark.parametrize("value", ["latest", "head", "none", ""])
def test_resolve_pto_isa_commit_explicit_unpinned_values_return_none(value):
    assert pto_isa.resolve_pto_isa_commit(value) is None


def test_ensure_pto_isa_root_uses_pin_instead_of_latest_for_existing_clone(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    checked_out = []
    updated = []

    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(
        pto_isa,
        "checkout_pto_isa_commit",
        lambda path, commit, verbose=False: checked_out.append(commit) or True,
    )
    monkeypatch.setattr(pto_isa, "_update_to_latest", lambda path, verbose: updated.append(path))
    monkeypatch.setattr(pto_isa, "_record_runtime_pto_isa", lambda root: None)

    assert pto_isa.ensure_pto_isa_root(update_if_exists=True) == str(clone_path.resolve())
    assert checked_out == [PIN_A]
    assert updated == []


def test_ensure_pto_isa_root_latest_still_updates_existing_clone(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    checked_out = []
    updated = []

    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(
        pto_isa,
        "checkout_pto_isa_commit",
        lambda path, commit, verbose=False: checked_out.append(commit) or True,
    )
    monkeypatch.setattr(pto_isa, "_update_to_latest", lambda path, verbose: updated.append(path))
    monkeypatch.setattr(pto_isa, "_record_runtime_pto_isa", lambda root: None)

    assert pto_isa.ensure_pto_isa_root(commit="latest", update_if_exists=True) == str(clone_path.resolve())
    assert checked_out == []
    assert updated == [clone_path]


def test_checkout_pto_isa_commit_rejects_non_git_clone(tmp_path, monkeypatch, caplog):
    calls = []

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append(args)
        return subprocess.CompletedProcess(["git", *args], returncode=128, stdout="", stderr="not a git repo")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)
    caplog.set_level("WARNING", logger="simpler_setup.pto_isa")

    assert not pto_isa.checkout_pto_isa_commit(tmp_path, PIN_A)
    assert calls == [["rev-parse", "--short", "HEAD"]]
    assert "Failed to read pto-isa HEAD" in caplog.text


def test_checkout_pto_isa_commit_marks_dubious_clone_safe_and_retries(tmp_path, monkeypatch):
    calls = []
    dubious = "fatal: detected dubious ownership in repository\nsafe.directory"
    safe_arg = f"safe.directory={tmp_path.resolve()}"
    normal_attempts = set()

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append((args, cwd, check))
        if (
            args
            in (
                ["rev-parse", "--short", "HEAD"],
                ["fetch", "origin"],
                ["checkout", PIN_A],
            )
            and tuple(args) not in normal_attempts
        ):
            normal_attempts.add(tuple(args))
            return subprocess.CompletedProcess(["git", *args], returncode=128, stdout="", stderr=dubious)
        if args == ["-c", safe_arg, "rev-parse", "--short", "HEAD"]:
            return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="bbbbbbb\n", stderr="")
        return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)

    assert pto_isa.checkout_pto_isa_commit(tmp_path, PIN_A)
    assert calls == [
        (["rev-parse", "--short", "HEAD"], tmp_path, False),
        (["-c", safe_arg, "rev-parse", "--short", "HEAD"], tmp_path, False),
        (["fetch", "origin"], tmp_path, False),
        (["-c", safe_arg, "fetch", "origin"], tmp_path, False),
        (["checkout", PIN_A], tmp_path, False),
        (["-c", safe_arg, "checkout", PIN_A], tmp_path, False),
    ]


def test_get_pto_isa_head_retries_dubious_ownership(tmp_path, monkeypatch):
    calls = []
    dubious = "fatal: detected dubious ownership in repository\nsafe.directory"
    safe_arg = f"safe.directory={tmp_path.resolve()}"

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append((args, cwd, check))
        if args == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(["git", *args], returncode=128, stdout="", stderr=dubious)
        if args == ["-c", safe_arg, "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(["git", *args], returncode=0, stdout=f"{PIN_A}\n", stderr="")
        return subprocess.CompletedProcess(["git", *args], returncode=1, stdout="", stderr="unexpected")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)

    assert pto_isa.get_pto_isa_head(str(tmp_path)) == PIN_A
    assert calls == [
        (["rev-parse", "HEAD"], tmp_path, False),
        (["-c", safe_arg, "rev-parse", "HEAD"], tmp_path, False),
    ]


def test_update_to_latest_retries_dubious_ownership(tmp_path, monkeypatch):
    calls = []
    dubious = "fatal: detected dubious ownership in repository\nsafe.directory"
    safe_arg = f"safe.directory={tmp_path.resolve()}"
    normal_attempts = set()

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append((args, cwd, check))
        if args in (["fetch", "origin"], ["reset", "--hard", "origin/HEAD"]) and tuple(args) not in normal_attempts:
            normal_attempts.add(tuple(args))
            return subprocess.CompletedProcess(["git", *args], returncode=128, stdout="", stderr=dubious)
        return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)

    pto_isa._update_to_latest(tmp_path, verbose=False)
    assert calls == [
        (["fetch", "origin"], tmp_path, False),
        (["-c", safe_arg, "fetch", "origin"], tmp_path, False),
        (["reset", "--hard", "origin/HEAD"], tmp_path, False),
        (["-c", safe_arg, "reset", "--hard", "origin/HEAD"], tmp_path, False),
    ]


def test_ensure_pto_isa_root_rejects_existing_clone_when_pin_checkout_fails(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    updated = []

    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(pto_isa, "checkout_pto_isa_commit", lambda path, commit, verbose=False: False)
    monkeypatch.setattr(pto_isa, "_update_to_latest", lambda path, verbose: updated.append(path))
    monkeypatch.setattr(pto_isa, "_record_runtime_pto_isa", lambda root: None)

    with pytest.raises(OSError, match="PTO-ISA not available"):
        pto_isa.ensure_pto_isa_root(update_if_exists=True)
    assert updated == []


def test_ensure_pto_isa_root_rejects_clone_when_pin_checkout_fails_after_clone(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    updated = []

    def clone_then_fail(target, commit, clone_protocol, verbose):
        (target / "include").mkdir(parents=True)
        return False

    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(pto_isa, "_clone", clone_then_fail)
    monkeypatch.setattr(pto_isa, "checkout_pto_isa_commit", lambda path, commit, verbose=False: False)
    monkeypatch.setattr(pto_isa, "_update_to_latest", lambda path, verbose: updated.append(path))
    monkeypatch.setattr(pto_isa, "_record_runtime_pto_isa", lambda root: None)

    with pytest.raises(OSError, match="PTO-ISA not available"):
        pto_isa.ensure_pto_isa_root(update_if_exists=True)
    assert updated == []


def test_write_pto_isa_build_metadata_records_actual_head(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: "a" * 40)

    pto_isa.write_pto_isa_build_metadata(tmp_path, str(tmp_path / "pto-isa"), requested_commit="latest")

    metadata = json.loads((tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).read_text())
    assert metadata["pto_isa_commit"] == "a" * 40
    assert metadata["requested_commit"] == "latest"


def test_write_pto_isa_build_metadata_rejects_unknown_head(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: "")

    with pytest.raises(RuntimeError, match="Point PTO_ISA_ROOT to a full pto-isa git checkout"):
        pto_isa.write_pto_isa_build_metadata(tmp_path, str(tmp_path / "pto-isa"))


def test_validate_runtime_pto_isa_accepts_matching_prefix(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "abcdef1234567890"}) + "\n"
    )
    monkeypatch.setenv("SIMPLER_RUN_PTO_ISA_COMMIT", "abcdef1")

    pto_isa.validate_runtime_pto_isa_compatible(tmp_path)


def test_validate_runtime_pto_isa_rejects_when_run_commit_unavailable(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "other_sha"}) + "\n"
    )
    monkeypatch.delenv("SIMPLER_RUN_PTO_ISA_COMMIT", raising=False)
    monkeypatch.delenv("PTO_ISA_ROOT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: None)

    with pytest.raises(RuntimeError, match="Cannot verify PTO-ISA runtime revision"):
        pto_isa.validate_runtime_pto_isa_compatible(tmp_path)


def test_validate_runtime_pto_isa_rejects_mismatch(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "build_sha"}) + "\n"
    )
    monkeypatch.setenv("SIMPLER_RUN_PTO_ISA_COMMIT", "run_sha")

    with pytest.raises(RuntimeError, match="PTO-ISA version mismatch"):
        pto_isa.validate_runtime_pto_isa_compatible(tmp_path)


def test_validate_runtime_pto_isa_uses_pin_for_non_git_env_root(tmp_path, monkeypatch, caplog):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "build_sha"}) + "\n"
    )
    monkeypatch.delenv("SIMPLER_RUN_PTO_ISA_COMMIT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "build_sha")
    monkeypatch.setenv("PTO_ISA_ROOT", str(tmp_path / "pto-isa"))
    (tmp_path / "pto-isa").mkdir()
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: "")

    caplog.set_level("WARNING", logger="simpler_setup.pto_isa")
    pto_isa.validate_runtime_pto_isa_compatible(tmp_path)
    assert "falling back to resolved PTO-ISA commit" in caplog.text


def test_validate_runtime_pto_isa_rejects_when_pin_missing_with_non_git_env_root(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "build_sha"}) + "\n"
    )
    monkeypatch.delenv("SIMPLER_RUN_PTO_ISA_COMMIT", raising=False)
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: None)
    monkeypatch.setenv("PTO_ISA_ROOT", str(tmp_path / "pto-isa"))
    (tmp_path / "pto-isa").mkdir()
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: "")

    with pytest.raises(RuntimeError, match="Cannot verify PTO-ISA runtime revision"):
        pto_isa.validate_runtime_pto_isa_compatible(tmp_path)


def test_ensure_pto_isa_root_records_runtime_commit_for_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PTO_ISA_ROOT", str(tmp_path))
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: "b" * 40)

    assert pto_isa.ensure_pto_isa_root(commit="c" * 40) == str(tmp_path)
    assert os.environ["SIMPLER_RUN_PTO_ISA_COMMIT"] == "b" * 40
    assert os.environ["SIMPLER_RUN_PTO_ISA_ROOT"] == str(tmp_path.resolve())
