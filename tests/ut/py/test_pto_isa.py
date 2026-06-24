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

import pytest

from simpler_setup import pto_isa


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
    monkeypatch.delenv("SIMPLER_PTO_ISA_COMMIT", raising=False)

    with pytest.raises(RuntimeError, match="Cannot verify PTO-ISA runtime revision"):
        pto_isa.validate_runtime_pto_isa_compatible(tmp_path)


def test_validate_runtime_pto_isa_rejects_mismatch(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "build_sha"}) + "\n"
    )
    monkeypatch.setenv("SIMPLER_RUN_PTO_ISA_COMMIT", "run_sha")

    with pytest.raises(RuntimeError, match="PTO-ISA version mismatch"):
        pto_isa.validate_runtime_pto_isa_compatible(tmp_path)


def test_validate_runtime_pto_isa_uses_explicit_env_for_non_git_env_root(tmp_path, monkeypatch, caplog):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "build_sha"}) + "\n"
    )
    monkeypatch.delenv("SIMPLER_RUN_PTO_ISA_COMMIT", raising=False)
    monkeypatch.setenv("SIMPLER_PTO_ISA_COMMIT", "build_sha")
    monkeypatch.setenv("PTO_ISA_ROOT", str(tmp_path / "pto-isa"))
    (tmp_path / "pto-isa").mkdir()
    monkeypatch.setattr(pto_isa, "get_pto_isa_head", lambda root: "")

    caplog.set_level("WARNING", logger="simpler_setup.pto_isa")
    pto_isa.validate_runtime_pto_isa_compatible(tmp_path)
    assert "falling back to resolved PTO-ISA commit" in caplog.text


def test_validate_runtime_pto_isa_rejects_latest_with_non_git_env_root(tmp_path, monkeypatch):
    (tmp_path / pto_isa.PTO_ISA_BUILD_METADATA).write_text(
        json.dumps({"schema_version": 1, "pto_isa_commit": "build_sha"}) + "\n"
    )
    monkeypatch.delenv("SIMPLER_RUN_PTO_ISA_COMMIT", raising=False)
    monkeypatch.setenv("SIMPLER_PTO_ISA_COMMIT", "latest")
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
