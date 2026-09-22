#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for payload restoration in simpler_setup.tools.core_swimlane."""

import json
import struct
import sys
from pathlib import Path
from typing import Optional

import pytest

from simpler_setup.scene_test import DUMP_ARGS_MODES
from simpler_setup.tools import core_swimlane


def _tensor_record(  # noqa: PLR0913
    *,
    slot=6,
    dtype="UINT8",
    shape=None,
    strides=None,
    start_offset=0,
    stage="before_dispatch",
    role="input",
    bin_offset=0,
    bin_size=4,
    truncated=False,
):
    shape = [4] if shape is None else shape
    strides = [1] if strides is None else strides
    return {
        "task_id": "0x0",
        "func_id": [0, 1, 1],
        "role": role,
        "stage": stage,
        "arg_index": slot,
        "kind": "tensor",
        "dtype": dtype,
        "shape": shape,
        "strides": strides,
        "start_offset": start_offset,
        "bin_offset": bin_offset,
        "bin_size": bin_size,
        "truncated": truncated,
    }


def _write_dump(
    tmp_path: Path,
    records,
    payload=b"",
    *,
    bin_file: Optional[str] = "args.bin",
    bin_format=None,
):
    dump_dir = tmp_path / "args_dump"
    dump_dir.mkdir()
    manifest = dump_dir / "args_dump.json"
    manifest.write_text(
        json.dumps(
            {
                "bin_format": bin_format
                or {
                    "type": "logical_contiguous",
                    "byte_order": "little_endian",
                },
                "bin_file": bin_file,
                # The fixture models a hybrid dump: every task in the manifest,
                # payload only for Arg::dump()-marked tensors.
                "dump_args_level": DUMP_ARGS_MODES["hybrid"],
                "args": records,
            }
        )
    )
    if bin_file is not None:
        (dump_dir / bin_file).write_bytes(payload)
    return manifest


def _reconstruct(manifest):
    _, tensor_count, kargs, _ = core_swimlane.reconstruct_task_args(manifest, [0, 1])
    return tensor_count, kargs


def test_restore_arg_loads_before_dispatch_payload_and_emits_memcpy(tmp_path):
    before = _tensor_record(bin_offset=0, bin_size=4)
    after = _tensor_record(
        stage="after_completion",
        role="inout",
        bin_offset=4,
        bin_size=4,
    )
    manifest = _write_dump(tmp_path, [after, before], b"\x01\x02\x03\x04\x09\x09\x09\x09")

    tensor_count, kargs = _reconstruct(manifest)
    core_swimlane.restore_arg_payloads(manifest, kargs, [6])

    assert kargs[0]["restore_data"] == b"\x01\x02\x03\x04"
    host = core_swimlane.emit_replay_host(tensor_count, kargs)
    assert "static const unsigned char hbuf0[]" in host
    assert "0x01, 0x02, 0x03, 0x04" in host
    assert "aclrtMemcpy(d_t0, t0Bytes, hbuf0, sizeof(hbuf0)" in host
    assert "aclrtMemset(d_t0" not in host


def test_restore_arg_scatters_logical_payload_into_strided_physical_view(tmp_path):
    logical = struct.pack("<4H", 10, 20, 30, 40)
    record = _tensor_record(
        dtype="UINT16",
        shape=[2, 2],
        strides=[3, 1],
        start_offset=1,
        bin_size=len(logical),
    )
    manifest = _write_dump(tmp_path, [record], logical)

    _, kargs = _reconstruct(manifest)
    core_swimlane.restore_arg_payloads(manifest, kargs, [6])

    assert struct.unpack("<6H", kargs[0]["restore_data"]) == (0, 10, 20, 0, 30, 40)


@pytest.mark.parametrize(
    ("record_updates", "message"),
    [
        ({"stage": "after_completion", "role": "output"}, "no before_dispatch payload"),
        ({"truncated": True}, "payload is truncated"),
        ({"bin_size": 3}, "does not match logical tensor size"),
    ],
)
def test_restore_arg_rejects_unusable_tensor_records(tmp_path, record_updates, message):
    record = _tensor_record(**record_updates)
    manifest = _write_dump(tmp_path, [record], b"\x01\x02\x03\x04")
    _, kargs = _reconstruct(manifest)

    with pytest.raises(ValueError, match=message):
        core_swimlane.restore_arg_payloads(manifest, kargs, [6])


def test_restore_arg_rejects_dump_without_payload(tmp_path):
    record = _tensor_record(bin_size=0)
    manifest = _write_dump(tmp_path, [record], bin_file=None)
    _, kargs = _reconstruct(manifest)

    with pytest.raises(ValueError, match="no bin_file"):
        core_swimlane.restore_arg_payloads(manifest, kargs, [6])


def test_restore_arg_rejects_unmarked_slot(tmp_path):
    record = _tensor_record(bin_size=0)
    manifest = _write_dump(tmp_path, [record])
    _, kargs = _reconstruct(manifest)

    with pytest.raises(ValueError, match=r"mark that tensor with CoreTaskArgs::dump"):
        core_swimlane.restore_arg_payloads(manifest, kargs, [6])


def test_restore_arg_rejects_missing_slot(tmp_path):
    manifest = _write_dump(tmp_path, [_tensor_record()], b"\x01\x02\x03\x04")
    _, kargs = _reconstruct(manifest)

    with pytest.raises(ValueError, match="slot 7 is not an arg"):
        core_swimlane.restore_arg_payloads(manifest, kargs, [7])


def test_restore_arg_rejects_unknown_bin_format(tmp_path):
    manifest = _write_dump(
        tmp_path,
        [_tensor_record()],
        b"\x01\x02\x03\x04",
        bin_format={"type": "physical", "byte_order": "little_endian"},
    )
    _, kargs = _reconstruct(manifest)

    with pytest.raises(ValueError, match="type=logical_contiguous"):
        core_swimlane.restore_arg_payloads(manifest, kargs, [6])


def test_restore_arg_rejects_scalar_slot(tmp_path):
    scalar = {
        "task_id": "0x0",
        "func_id": [0, 1, 1],
        "role": "scalar",
        "stage": "before_dispatch",
        "arg_index": 17,
        "kind": "scalar",
        "dtype": "INT64",
        "shape": [],
        "strides": [],
        "start_offset": 0,
        "value": 9,
        "bin_offset": 0,
        "bin_size": 0,
    }
    manifest = _write_dump(tmp_path, [scalar], b"")
    _, kargs = _reconstruct(manifest)

    with pytest.raises(ValueError, match="is a scalar"):
        core_swimlane.restore_arg_payloads(manifest, kargs, [17])


def test_get_or_run_dump_uses_hybrid_without_payload_selector(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(core_swimlane, "PROJECT_ROOT", tmp_path)

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        dump_dir = tmp_path / "outputs" / "run_3" / "args_dump"
        dump_dir.mkdir(parents=True)
        (dump_dir / "args_dump.json").write_text("{}")

    monkeypatch.setattr(core_swimlane.subprocess, "run", fake_run)
    manifest = core_swimlane.get_or_run_dump(tmp_path / "test_case.py", "a2a3sim", "sim", None)

    flag_index = commands[0].index("--dump-args")
    assert commands[0][flag_index + 1] == "hybrid"
    assert not any(value.startswith("--dump-args-payload") for value in commands[0])
    assert manifest.name == "args_dump.json"


def test_replay_cmake_includes_shared_platform_headers():
    cmake = core_swimlane.emit_cmakelists(
        "a2a3",
        "test",
        core_swimlane.ARCH_CONFIG["a2a3"],
        extra_include_dirs=["/sdk/asc/include"],
    )

    assert "${REPO_ROOT}/src/common/platform/include" in cmake
    assert '"/sdk/asc/include"' in cmake


def test_run_collect_forwards_msprof_timeout_in_minutes():
    script = core_swimlane.emit_run_collect(
        core_swimlane.ARCH_CONFIG["a2a3"],
        "/pto-isa",
        msprof_timeout=4,
    )

    assert "--timeout=4" in script
    assert "__MSPROF_TIMEOUT__" not in script


def test_cli_rejects_restore_and_set_on_the_same_slot(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core_swimlane",
            "--test",
            "unused.py",
            "--func-id",
            "0,1",
            "--restore-arg",
            "6",
            "--set-arg",
            "6=1",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        core_swimlane.main()

    assert "cannot use both --restore-arg and --set-arg" in capsys.readouterr().err


def test_cli_rejects_a5_payload_restoration(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core_swimlane",
            "--test",
            "unused.py",
            "--func-id",
            "0",
            "--platform",
            "a5sim",
            "--restore-arg",
            "6",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        core_swimlane.main()

    assert "does not support a5/a5sim" in capsys.readouterr().err


def test_default_tensor_initialization_remains_zero_fill():
    tensor = {
        "kind": "tensor",
        "slot": 0,
        "dtype": "UINT8",
        "shape": [4],
        "strides": [1],
        "start_offset": 0,
    }

    host = core_swimlane.emit_replay_host(1, [tensor])

    assert "aclrtMemset(d_t0, t0Bytes, 0, t0Bytes)" in host
