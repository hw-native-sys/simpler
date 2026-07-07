# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for pinned PTO-ISA checkout resolution and build metadata."""

import json
import subprocess

import pytest

from simpler_setup import pto_isa

PIN_A = "a" * 40
PIN_B = "b" * 40
RUNTIME_A = "a2a3/onboard/host_build_graph"
RUNTIME_B = "a2a3/onboard/tensormap_and_ringbuffer"


def test_read_pto_isa_pin_reads_valid_pin(tmp_path):
    pin = tmp_path / "pto_isa.pin"
    pin.write_text(f"{PIN_A.upper()}\n")

    assert pto_isa.read_pto_isa_pin(pin) == PIN_A


def test_read_pto_isa_pin_rejects_missing_pin(tmp_path):
    with pytest.raises(RuntimeError, match="PTO-ISA pin not found"):
        pto_isa.read_pto_isa_pin(tmp_path / "missing.pin")


def test_read_pto_isa_pin_rejects_empty_pin(tmp_path):
    pin = tmp_path / "pto_isa.pin"
    pin.write_text("\n")

    with pytest.raises(RuntimeError, match="got an empty file"):
        pto_isa.read_pto_isa_pin(pin)


def test_read_pto_isa_pin_rejects_invalid_content(tmp_path):
    pin = tmp_path / "pto_isa.pin"
    pin.write_text("not-a-sha\n")

    with pytest.raises(RuntimeError, match="Invalid PTO-ISA pin"):
        pto_isa.read_pto_isa_pin(pin)


def test_clone_lands_on_pinned_commit(tmp_path, monkeypatch):
    target = tmp_path / "build" / "pto-isa"
    calls = []

    monkeypatch.setattr(pto_isa, "_is_git_available", lambda: True)

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append(args)
        return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    assert pto_isa._clone(target, PIN_A, verbose=False)
    assert calls == [
        ["clone", "--no-checkout", "https://github.com/hw-native-sys/pto-isa.git", str(target)],
        ["checkout", "--detach", "--force", PIN_A],
    ]


def test_ensure_pto_isa_root_reuses_pristine_checkout(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    checked = []

    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(pto_isa, "_clone", lambda *a, **k: pytest.fail("unexpected clone of pristine checkout"))
    monkeypatch.setattr(
        pto_isa,
        "_is_pristine_at_commit",
        lambda path, commit, verbose=False: checked.append((path, commit)) or True,
    )
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    assert pto_isa.ensure_pto_isa_root(verbose=True) == str(clone_path.resolve())
    assert checked == [(clone_path, PIN_A)]  # reused as-is, no checkout, no clone


def test_ensure_pto_isa_root_clones_when_missing(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    events = []

    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(
        pto_isa, "_is_pristine_at_commit", lambda *a, **k: pytest.fail("unexpected pristine check of missing checkout")
    )

    def fake_clone(path, commit, verbose=False):
        events.append(("clone", path, commit))
        (path / "include").mkdir(parents=True)
        return True

    monkeypatch.setattr(pto_isa, "_clone", fake_clone)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    assert pto_isa.ensure_pto_isa_root() == str(clone_path.resolve())
    assert events == [("clone", clone_path, PIN_A)]


def test_ensure_pto_isa_root_reclones_when_not_pristine(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    (clone_path / "include").mkdir(parents=True)
    events = []

    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(
        pto_isa,
        "_is_pristine_at_commit",
        lambda path, commit, verbose=False: events.append(("check", path, commit)) or False,
    )

    def fake_clone(path, commit, verbose=False):
        events.append(("clone", path, commit))
        return True

    monkeypatch.setattr(pto_isa, "_clone", fake_clone)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    assert pto_isa.ensure_pto_isa_root() == str(clone_path.resolve())
    # A dirty/wrong-revision cache is re-cloned fresh, never checked out in place.
    assert events == [("check", clone_path, PIN_A), ("clone", clone_path, PIN_A)]


def test_ensure_pto_isa_root_rejects_when_reclone_lands_wrong_commit(tmp_path, monkeypatch):
    clone_path = tmp_path / "build" / "pto-isa"
    (clone_path / "include").mkdir(parents=True)

    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_clone_path", lambda: clone_path)
    monkeypatch.setattr(pto_isa, "_is_pristine_at_commit", lambda path, commit, verbose=False: False)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_B)
    monkeypatch.setattr(pto_isa, "_clone", lambda path, commit, verbose=False: True)

    with pytest.raises(OSError, match="PTO-ISA not available"):
        pto_isa.ensure_pto_isa_root()


def test_is_pristine_at_commit_true_when_clean_and_at_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        assert args == ["status", "--porcelain"]
        return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)
    assert pto_isa._is_pristine_at_commit(tmp_path, PIN_A, verbose=False)


def test_is_pristine_at_commit_false_when_dirty(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        return subprocess.CompletedProcess(
            ["git", *args], returncode=0, stdout=" M docs/isa/TADDDEQRELU.md\n", stderr=""
        )

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)
    assert not pto_isa._is_pristine_at_commit(tmp_path, PIN_A, verbose=False)


def test_is_pristine_at_commit_false_when_wrong_head(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_B)
    monkeypatch.setattr(pto_isa, "_run_git", lambda *a, **k: pytest.fail("status must not run for a wrong HEAD"))

    assert not pto_isa._is_pristine_at_commit(tmp_path, PIN_A, verbose=False)


def test_land_on_commit_fetches_when_commit_is_missing(tmp_path, monkeypatch):
    calls = []

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append((args, cwd, check))
        if args == ["checkout", "--detach", "--force", PIN_A] and len([c for c in calls if c[0] == args]) == 1:
            return subprocess.CompletedProcess(["git", *args], returncode=1, stdout="", stderr="missing")
        return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    assert pto_isa._land_on_commit(tmp_path, PIN_A, verbose=False)
    assert calls == [
        (["checkout", "--detach", "--force", PIN_A], tmp_path, False),
        (["fetch", "origin"], tmp_path, False),
        (["checkout", "--detach", "--force", PIN_A], tmp_path, False),
    ]


def test_land_on_commit_retries_dubious_ownership_with_safe_directory(tmp_path, monkeypatch):
    calls = []
    dubious = "fatal: detected dubious ownership in repository\nsafe.directory"
    safe_arg = f"safe.directory={tmp_path.resolve()}"

    def fake_run_git(args, cwd=None, timeout=30, check=False):
        calls.append((args, cwd, check))
        if args == ["checkout", "--detach", "--force", PIN_A]:
            return subprocess.CompletedProcess(["git", *args], returncode=128, stdout="", stderr=dubious)
        return subprocess.CompletedProcess(["git", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pto_isa, "_run_git", fake_run_git)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    assert pto_isa._land_on_commit(tmp_path, PIN_A, verbose=False)
    assert calls == [
        (["checkout", "--detach", "--force", PIN_A], tmp_path, False),
        (["-c", safe_arg, "checkout", "--detach", "--force", PIN_A], tmp_path, False),
    ]


def test_write_pto_isa_build_metadata_records_pin_and_actual_head(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    pto_isa.write_pto_isa_build_metadata(tmp_path, str(tmp_path / "pto-isa"), [RUNTIME_A])

    metadata = json.loads((tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).read_text())
    assert metadata["schema_version"] == 3
    assert metadata["required_commit_from_pin"] == PIN_A
    assert metadata["actual_checkout_commit"] == PIN_A
    assert metadata["checkout_path"] == str((tmp_path / "pto-isa").resolve())
    assert metadata["runtime_artifacts"][RUNTIME_A]["required_commit_from_pin"] == PIN_A
    assert metadata["runtime_artifacts"][RUNTIME_A]["actual_checkout_commit"] == PIN_A


def test_write_pto_isa_build_metadata_preserves_other_runtime_entries(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "runtime_artifacts": {
                    RUNTIME_A: {
                        "required_commit_from_pin": PIN_B,
                        "actual_checkout_commit": PIN_B,
                    }
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_A)

    pto_isa.write_pto_isa_build_metadata(tmp_path, str(tmp_path / "pto-isa"), [RUNTIME_B])

    metadata = json.loads((tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).read_text())
    assert metadata["runtime_artifacts"][RUNTIME_A]["required_commit_from_pin"] == PIN_B
    assert metadata["runtime_artifacts"][RUNTIME_B]["required_commit_from_pin"] == PIN_A


def test_write_pto_isa_build_metadata_rejects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: PIN_B)

    with pytest.raises(RuntimeError, match="PTO-ISA checkout mismatch"):
        pto_isa.write_pto_isa_build_metadata(tmp_path, str(tmp_path / "pto-isa"), [RUNTIME_A])


def test_validate_runtime_pto_isa_current_pin_accepts_matching_metadata(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "runtime_artifacts": {
                    RUNTIME_A: {
                        "required_commit_from_pin": PIN_A,
                        "actual_checkout_commit": PIN_A,
                    },
                    RUNTIME_B: {
                        "required_commit_from_pin": PIN_B,
                        "actual_checkout_commit": PIN_B,
                    },
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)

    pto_isa.validate_runtime_pto_isa_current_pin(tmp_path, runtime_key=RUNTIME_A)


def test_validate_runtime_pto_isa_current_pin_rejects_stale_metadata(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "runtime_artifacts": {
                    RUNTIME_A: {
                        "required_commit_from_pin": PIN_B,
                        "actual_checkout_commit": PIN_B,
                    }
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)

    with pytest.raises(RuntimeError, match="Stale PTO-ISA runtime binaries"):
        pto_isa.validate_runtime_pto_isa_current_pin(tmp_path, runtime_key=RUNTIME_A)


def test_validate_runtime_pto_isa_current_pin_rejects_missing_runtime_entry(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "runtime_artifacts": {
                    RUNTIME_A: {
                        "required_commit_from_pin": PIN_A,
                        "actual_checkout_commit": PIN_A,
                    }
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)

    with pytest.raises(RuntimeError, match="has no entry for runtime"):
        pto_isa.validate_runtime_pto_isa_current_pin(tmp_path, runtime_key=RUNTIME_B)


def test_validate_runtime_pto_isa_current_pin_accepts_legacy_global_metadata(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 2, "required_commit_from_pin": PIN_A, "actual_checkout_commit": PIN_A}) + "\n"
    )
    monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: PIN_A)

    pto_isa.validate_runtime_pto_isa_current_pin(tmp_path, runtime_key=RUNTIME_A)
