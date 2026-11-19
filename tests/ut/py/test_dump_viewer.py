# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recorded ArgsDump entries remain separately selectable and exportable."""

import json
import struct
import sys

import pytest

from simpler_setup.tools import dump_viewer


def _record(*, offset=0, stage="before_dispatch", task="0x0000000000000900"):
    return {
        "task_id": task,
        "func_id": [0],
        "role": "inout",
        "stage": stage,
        "arg_index": 0,
        "kind": "tensor",
        "dtype": "FLOAT32",
        "shape": [2],
        "strides": [1],
        "start_offset": 0,
        "is_contiguous": True,
        "bin_offset": offset,
        "bin_size": 8,
        "truncated": False,
    }


def _dump(tmp_path, records, *, recorded=True):
    directory = tmp_path / "args_dump"
    directory.mkdir()
    manifest = {
        "args": records,
        "bin_file": "args.saved.bin",
        "total_args": len(records),
        "dropped_records": None if recorded else 0,
    }
    if recorded:
        manifest.update(
            recorded_window=True,
            counts_unknown=True,
            dropped_records_complete=False,
            collection_verdict="aggregate_counts_unknown",
            host_discarded_args=0,
            metadata_discarded_records=0,
        )
    (directory / "args_dump.json").write_text(json.dumps(manifest))
    (directory / "args.saved.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    return directory


def test_recorded_export_preserves_repeated_task_stage_argument_occurrences(tmp_path, monkeypatch):
    directory = _dump(tmp_path, [_record(), _record(offset=8)])
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    dump_viewer.main()
    exported = list((directory / "txt").glob("*.txt"))
    assert len(exported) == 2
    assert {path.name for path in exported} == {
        "entry_000000_task_0x0000000000000900_before_inout0.txt",
        "entry_000001_task_0x0000000000000900_before_inout0.txt",
    }
    assert "[0] 1\n[1] 2" in (directory / "txt" / "entry_000000_task_0x0000000000000900_before_inout0.txt").read_text()
    assert "[0] 3\n[1] 4" in (directory / "txt" / "entry_000001_task_0x0000000000000900_before_inout0.txt").read_text()


def test_filtered_listing_keeps_indices_used_by_manifest_selection(tmp_path, monkeypatch, capsys):
    directory = _dump(tmp_path, [_record(), _record(offset=8, stage="after_completion")])
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--stage", "after"])
    dump_viewer.main()
    listed = [line.split()[0] for line in capsys.readouterr().out.splitlines() if "0x0000000000000900" in line]
    assert listed == ["1"]
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--stage", "after", "--index", "1"])
    dump_viewer.main()
    [exported] = (directory / "txt").glob("*.txt")
    assert "[0] 3\n[1] 4" in exported.read_text()


def test_recorded_listing_declares_unknown_aggregate_counts(tmp_path, monkeypatch, capsys):
    directory = _dump(tmp_path, [_record()])
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory)])
    dump_viewer.main()
    text = capsys.readouterr().out
    assert "Recorded window" in text
    assert "unknown" in text
    assert "aggregate_counts_unknown" in text


@pytest.mark.parametrize("changes", [{"bin_offset": 12}, {"bin_offset": -1}, {"bin_size": -8}])
def test_export_rejects_payload_ranges_outside_the_named_binary(tmp_path, monkeypatch, changes):
    directory = _dump(tmp_path, [_record() | changes])
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    with pytest.raises(ValueError, match="payload range"):
        dump_viewer.main()


def test_empty_recorded_window_is_a_valid_inspection(tmp_path, monkeypatch, capsys):
    directory = _dump(tmp_path, [])
    path = directory / "args_dump.json"
    manifest = json.loads(path.read_text())
    manifest["bin_file"] = None
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    dump_viewer.main()
    assert "No samples" in capsys.readouterr().out
    assert not (directory / "txt").exists()


def test_discarded_payload_is_not_reported_as_an_empty_tensor(tmp_path, monkeypatch, capsys):
    directory = _dump(tmp_path, [_record() | {"host_discarded": True, "bin_size": 0}])
    path = directory / "args_dump.json"
    manifest = json.loads(path.read_text())
    manifest.update(host_discarded_args=1, metadata_discarded_records=2, collection_verdict="loss")
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    dump_viewer.main()
    assert "host_discarded_args=1" in capsys.readouterr().out
    [exported] = (directory / "txt").glob("*.txt")
    assert "DATA DISCARDED" in exported.read_text()


def test_export_requires_the_manifest_named_payload_file(tmp_path, monkeypatch):
    directory = _dump(tmp_path, [_record()])
    (directory / "args.saved.bin").unlink()
    (directory / "args.bin").write_bytes(b"stale unrelated payload")
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    with pytest.raises(FileNotFoundError, match="args.saved.bin"):
        dump_viewer.main()


def test_ordinary_export_keeps_filename_and_inline_scalar(tmp_path, monkeypatch):
    scalar = _record() | {"kind": "scalar", "dtype": "INT64", "value": -7, "bin_size": 0}
    directory = _dump(tmp_path, [scalar], recorded=False)
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    dump_viewer.main()
    [exported] = (directory / "txt").glob("*.txt")
    assert exported.name == "task_0x0000000000000900_before_inout0.txt"
    assert "# value: -7" in exported.read_text()


def test_listing_distinguishes_discarded_payload_from_empty_tensor(tmp_path, monkeypatch, capsys):
    directory = _dump(tmp_path, [_record() | {"host_discarded": True, "bin_size": 0}, _record() | {"bin_size": 0}])
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory)])
    dump_viewer.main()
    rows = [line for line in capsys.readouterr().out.splitlines() if "0x0000000000000900" in line]
    assert "DISCARDED" in rows[0]
    assert "DISCARDED" not in rows[1]


def test_recorded_missing_payload_name_does_not_read_stale_default(tmp_path, monkeypatch):
    directory = _dump(tmp_path, [_record()])
    manifest_path = directory / "args_dump.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["bin_file"]
    manifest_path.write_text(json.dumps(manifest))
    (directory / "args.bin").write_bytes(struct.pack("<2f", 99, 100))
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    with pytest.raises(ValueError, match="bin_path is None"):
        dump_viewer.main()


@pytest.mark.parametrize("existing", [False, True])
def test_failed_export_does_not_publish_partial_output(tmp_path, existing):
    directory = _dump(tmp_path, [_record(offset=12)])
    target = directory / "txt" / dump_viewer.arg_filename(_record(), entry_index=0)
    target.parent.mkdir()
    if existing:
        target.write_text("previous complete export")
    with pytest.raises(ValueError, match="payload range"):
        dump_viewer.export_arg(_record(offset=12), directory / "args.saved.bin", directory, entry_index=0)
    assert target.read_text() == "previous complete export" if existing else not target.exists()
    assert list(target.parent.iterdir()) == ([target] if existing else [])


def test_write_failure_preserves_existing_export_and_removes_temporary_file(tmp_path, monkeypatch):
    directory = _dump(tmp_path, [_record()])
    target = directory / "txt" / dump_viewer.arg_filename(_record(), entry_index=0)
    target.parent.mkdir()
    target.write_text("previous complete export")

    def fail_write(arg, bin_path, out):
        out.write("partial")
        raise OSError("injected output failure")

    monkeypatch.setattr(dump_viewer, "write_arg", fail_write)
    with pytest.raises(OSError, match="injected output failure"):
        dump_viewer.export_arg(_record(), directory / "args.saved.bin", directory, entry_index=0)
    assert target.read_text() == "previous complete export"
    assert list(target.parent.iterdir()) == [target]


def test_ordinary_legacy_manifest_keeps_default_payload_name(tmp_path, monkeypatch):
    directory = _dump(tmp_path, [_record()], recorded=False)
    manifest_path = directory / "args_dump.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["bin_file"]
    manifest_path.write_text(json.dumps(manifest))
    (directory / "args.bin").write_bytes(struct.pack("<2f", 99, 100))
    monkeypatch.setattr(sys, "argv", ["dump_viewer", str(directory), "--export"])
    dump_viewer.main()
    [exported] = (directory / "txt").glob("*.txt")
    assert "[0] 99\n[1] 100" in exported.read_text()
