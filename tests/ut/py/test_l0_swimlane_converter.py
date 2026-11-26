# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from simpler_setup.tools.l0_swimlane_converter import convert, main


def test_converts_structured_intervals_without_raw_binary(tmp_path):
    records = {
        "format": "simpler_a5_l0_events_v1",
        "run_epoch": 42,
        "physical_core_id": 107,
        "system_counter_frequency_hz": 1_000_000_000,
        "core_frequency_hz": 2_000_000_000,
        "busy_interval_columns": ["anchor_syscnt", "start_cycles", "end_cycles", "lane", "block_id", "clip_flags"],
        "busy_intervals": [[1000, 5, 15, 1, 7, 0]],
    }
    (tmp_path / "core_107.events.json").write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "trace.json"

    assert convert(tmp_path, output) == 1
    trace = json.loads(output.read_text(encoding="utf-8"))
    assert trace["metadata"]["run_epoch"] == 42
    assert trace["metadata"]["capture_complete"] is None
    assert next(event for event in trace["traceEvents"] if event["ph"] == "X") == {
        "ph": "X",
        "cat": "a5.l0",
        "name": "L0 VECTOR",
        "pid": 100107,
        "tid": 1,
        "ts": 1.0025,
        "dur": 0.005,
        "args": {"physical_core_id": 107, "block_id": 7, "clip_flags": 0},
    }


def test_rejects_missing_core_frequency_without_mislabeling_cycles(tmp_path):
    records = {
        "format": "simpler_a5_l0_events_v1",
        "run_epoch": 42,
        "physical_core_id": 107,
        "system_counter_frequency_hz": 1_000_000_000,
        "core_frequency_hz": 0,
        "core_frequency_source": "unavailable",
        "busy_interval_columns": ["anchor_syscnt", "start_cycles", "end_cycles", "lane", "block_id", "clip_flags"],
        "busy_intervals": [[1000, 5, 15, 1, 7, 0]],
    }
    (tmp_path / "core_107.events.json").write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot convert core cycles to time"):
        convert(tmp_path, tmp_path / "trace.json")
    assert not (tmp_path / "trace.json").exists()


@pytest.mark.parametrize(
    "transport_complete,decode_complete,missing_core,expected_complete",
    [(False, True, False, False), (True, False, False, False), (True, True, False, True), (True, True, True, False)],
)
def test_trace_preserves_capture_completeness(
    tmp_path, capsys, transport_complete, decode_complete, missing_core, expected_complete
):
    records = {
        "format": "simpler_a5_l0_events_v1",
        "run_epoch": 42,
        "physical_core_id": 107,
        "system_counter_frequency_hz": 1_000_000_000,
        "core_frequency_hz": 2_000_000_000,
        "busy_interval_columns": ["anchor_syscnt", "start_cycles", "end_cycles", "lane", "block_id", "clip_flags"],
        "busy_intervals": [[1000, 5, 15, 1, 7, 0]],
        "stats": {"structurally_complete": decode_complete},
        "raw_bytes": 4,
    }
    manifest = {
        "run_epoch": 42,
        "system_counter_frequency_hz": 1_000_000_000,
        "aic_frequency_hz": 2_000_000_000,
        "aiv_frequency_hz": 1_200_000_000,
        "capture_complete": transport_complete,
        "raw_transport_complete": transport_complete,
        "cores": [{"physical_core_id": 107, "core_type": 1, "raw_bytes": 4}],
    }
    if missing_core:
        manifest["cores"].append({"physical_core_id": 106, "core_type": 2, "raw_bytes": 8})
    (tmp_path / "core_107.events.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "trace.json"

    assert convert(tmp_path, output) == 1
    trace = json.loads(output.read_text(encoding="utf-8"))
    assert trace["metadata"]["capture_complete"] is expected_complete
    assert trace["metadata"]["missing_event_cores"] == ([106] if missing_core else [])
    assert trace["metadata"]["raw_transport_complete"] is transport_complete
    assert trace["metadata"]["decode_complete"] is decode_complete
    assert all(event["ph"] == "X" for event in trace["traceEvents"])
    assert ("capture incomplete" in capsys.readouterr().err) is not expected_complete


def test_truncated_capture_keeps_intervals_and_marks_data_missing(tmp_path, capsys):
    records = {
        "format": "simpler_a5_l0_events_v1",
        "run_epoch": 42,
        "physical_core_id": 107,
        "system_counter_frequency_hz": 1_000_000_000,
        "core_frequency_hz": 2_000_000_000,
        "busy_interval_columns": ["anchor_syscnt", "start_cycles", "end_cycles", "lane", "block_id", "clip_flags"],
        "busy_intervals": [[1000, 5, 15, 1, 7, 0]],
        "raw_bytes": 32,
        "stats": {"structurally_complete": True},
    }
    manifest = {
        "run_epoch": 42,
        "system_counter_frequency_hz": 1_000_000_000,
        "aic_frequency_hz": 2_000_000_000,
        "aiv_frequency_hz": 1_200_000_000,
        "capture_complete": True,
        "raw_transport_complete": True,
        "cores": [{"physical_core_id": 107, "core_type": 1, "raw_bytes": 64}],
    }
    (tmp_path / "core_107.events.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "trace.json"

    assert convert(tmp_path, output) == 1
    trace = json.loads(output.read_text(encoding="utf-8"))
    metadata = trace["metadata"]
    assert metadata["capture_complete"] is False
    assert metadata["raw_transport_complete"] is False
    assert metadata["decode_complete"] is True
    assert metadata["raw_byte_mismatches"] == [
        {"physical_core_id": 107, "expected_raw_bytes": 64, "decoded_raw_bytes": 32}
    ]
    assert len(trace["traceEvents"]) == 1
    assert trace["traceEvents"][0]["ph"] == "X"
    assert trace["traceEvents"][0]["dur"] == 0.005
    warning = capsys.readouterr().err
    assert "L0 data missing; capture incomplete" in warning
    assert "core 107: expected 64 raw bytes, decoded 32" in warning


@pytest.fixture(scope="module")
def decoder_library(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[3]
    library = tmp_path_factory.mktemp("l0_decoder") / "decoder.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{root / 'src/a5/platform/include'}",
            f"-I{root / 'src/common/platform/include'}",
            str(root / "src/a5/platform/shared/host/l0_trace_records.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


@pytest.fixture
def raw_capture(tmp_path):
    raw = struct.pack(
        "<8I", 0xE00003E8, 0xE0000000, 0xE0000000, 0xE0000000, 0xF0000005, 0xF0010005, 0xF0000005, 0xEFFFFFFF
    )
    manifest = {
        "run_epoch": 42,
        "system_counter_frequency_hz": 1_000_000_000,
        "aic_frequency_hz": 1_500_000_000,
        "aiv_frequency_hz": 1_200_000_000,
        "capture_complete": True,
        "raw_transport_complete": True,
        "cores": [{"physical_core_id": 0, "raw_bytes": 32, "core_type": 1}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "core_0.bin").write_bytes(raw)
    return tmp_path


def test_cli_ignores_raw_files_outside_manifest(raw_capture, decoder_library, monkeypatch, capsys):
    (raw_capture / "core_1.bin").write_bytes(b"unrelated stale capture")
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    trace = json.loads((raw_capture / "l0_swimlane_trace.json").read_text())
    assert trace["metadata"]["capture_complete"] is True
    assert [e["args"]["physical_core_id"] for e in trace["traceEvents"]] == [0]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure", [{"capture_complete": False}, {"capture_error": 2}, {"producer_stopped": 0}])
def test_cli_keeps_good_core_when_another_core_lost_its_capture(
    raw_capture, decoder_library, monkeypatch, capsys, failure
):
    path = raw_capture / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["capture_complete"] = manifest["raw_transport_complete"] = False
    manifest["cores"].append({"physical_core_id": 1, "raw_bytes": 0, "core_type": 1, **failure})
    path.write_text(json.dumps(manifest))
    (raw_capture / "core_1.bin").write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    trace = json.loads((raw_capture / "l0_swimlane_trace.json").read_text())
    assert [event["args"]["physical_core_id"] for event in trace["traceEvents"]] == [0]
    assert trace["metadata"]["capture_complete"] is False
    assert trace["metadata"]["data_missing"] is True
    assert trace["metadata"]["incomplete_capture_cores"] == [1]
    assert "incomplete capture cores: [1]" in capsys.readouterr().err


@pytest.mark.parametrize("preserve_older_timestamp", [False, True])
def test_cli_rejects_cached_events_after_raw_capture_replacement(
    raw_capture, decoder_library, monkeypatch, capsys, preserve_older_timestamp
):
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    output = raw_capture / "l0_swimlane_trace.json"
    original = output.read_bytes()
    raw = raw_capture / "core_0.bin"
    original_stat = raw.stat()
    raw.write_bytes(struct.pack("<I", 0xE00007D0) + raw.read_bytes()[4:])
    modified = (
        original_stat.st_mtime_ns - 1_000_000_000
        if preserve_older_timestamp
        else (raw_capture / "core_0.events.json").stat().st_mtime_ns + 1_000_000_000
    )
    os.utime(raw, ns=(original_stat.st_atime_ns, modified))
    monkeypatch.setattr(sys, "argv", ["l0_swimlane_converter", str(raw_capture)])

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "--decoder-library" in capsys.readouterr().err
    assert output.read_bytes() == original


def test_cli_accepts_unchanged_raw_content_with_a_new_timestamp(raw_capture, decoder_library, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    output = raw_capture / "l0_swimlane_trace.json"
    original = output.read_bytes()
    raw = raw_capture / "core_0.bin"
    timestamp = raw.stat().st_mtime_ns + 1_000_000_000
    os.utime(raw, ns=(timestamp, timestamp))
    monkeypatch.setattr(sys, "argv", ["l0_swimlane_converter", str(raw_capture)])
    main()
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "field,value,expected_start,expected_duration",
    [
        ("aic_frequency_hz", 1_000_000_000, 1.01, 0.005),
        ("core_type", 2, 1 + 10 / 1200, 5 / 1200),
        ("system_counter_frequency_hz", 500_000_000, 2 + 10 / 1500, 5 / 1500),
    ],
)
def test_cli_refreshes_cached_clocks_from_manifest(
    raw_capture, decoder_library, monkeypatch, field, value, expected_start, expected_duration
):
    argv = ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    path = raw_capture / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "core_type":
        manifest["cores"][0][field] = value
    else:
        manifest[field] = value
    path.write_text(json.dumps(manifest))
    output = raw_capture / "trace.json"
    with pytest.raises(ValueError, match="clock.*--decoder-library"):
        convert(raw_capture, output)
    assert not output.exists()

    main()
    assert convert(raw_capture, output) == 1
    trace = json.loads(output.read_text())
    assert trace["metadata"]["system_counter_frequency_hz"] == manifest["system_counter_frequency_hz"]
    [interval] = trace["traceEvents"]
    assert interval["ts"] == pytest.approx(expected_start)
    assert interval["dur"] == pytest.approx(expected_duration)


def test_cli_requires_decode_for_unverified_legacy_cache(raw_capture, decoder_library, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    path = raw_capture / "core_0.events.json"
    data = json.loads(path.read_text())
    data.pop("raw_sha256")
    path.write_text(json.dumps(data))
    monkeypatch.setattr(sys, "argv", ["l0_swimlane_converter", str(raw_capture)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "--decoder-library" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value",
    [
        ("physical_core_id", -1),
        ("physical_core_id", 108),
        ("physical_core_id", 2**40),
        ("system_counter_frequency_hz", 0),
        ("aic_frequency_hz", -1),
        ("aiv_frequency_hz", 2**64),
    ],
)
def test_cli_rejects_values_outside_native_range(raw_capture, decoder_library, monkeypatch, capsys, field, value):
    path = raw_capture / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "physical_core_id":
        manifest["cores"][0][field] = value
    else:
        manifest[field] = value
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "out of range" in capsys.readouterr().err
    assert not (raw_capture / "core_0.events.json").exists()


def test_rejects_duplicate_physical_core_records(raw_capture, decoder_library, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    shutil.copyfile(raw_capture / "core_0.events.json", raw_capture / "core_backup.events.json")
    output = raw_capture / "duplicate.json"
    with pytest.raises(ValueError, match="duplicate physical core"):
        convert(raw_capture, output)
    assert not output.exists()


def test_concurrent_conversions_publish_independent_files(raw_capture, decoder_library, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    output = raw_capture / "concurrent.json"
    ready = Barrier(2)
    replace = os.replace

    def publish(source, target):
        if Path(target) == output:
            ready.wait(timeout=10)
        return replace(source, target)

    monkeypatch.setattr(os, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(convert, raw_capture, output) for _ in range(2)]
        assert [result.result(timeout=15) for result in results] == [1, 1]
    assert len(json.loads(output.read_text())["traceEvents"]) == 1
    assert not list(raw_capture.glob("*.tmp"))
    assert not list(raw_capture.glob(".*.tmp"))


def test_cli_preserves_unavailable_zero_frequency(raw_capture, decoder_library, monkeypatch, capsys):
    path = raw_capture / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["aic_frequency_hz"] = 0
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "frequency unavailable" in capsys.readouterr().err
    assert json.loads((raw_capture / "core_0.events.json").read_text())["core_frequency_hz"] == 0


def test_failed_raw_export_retains_existing_events(raw_capture, decoder_library, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(raw_capture), "--decoder-library", str(decoder_library)]
    )
    main()
    cached = raw_capture / "core_0.events.json"
    previous = cached.read_bytes()
    path = raw_capture / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cores"][0]["core_type"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert cached.read_bytes() == previous
    assert not list(raw_capture.glob(".l0-export-*"))


@pytest.mark.parametrize("missing_size", [None, 0])
def test_cli_missing_raw_file_does_not_reuse_old_intervals(
    tmp_path, decoder_library, monkeypatch, capsys, missing_size
):
    raw = struct.pack(
        "<8I", 0xE00003E8, 0xE0000000, 0xE0000000, 0xE0000000, 0xF0000005, 0xF0010005, 0xF0000005, 0xEFFFFFFF
    )
    manifest = {
        "run_epoch": 42,
        "system_counter_frequency_hz": 1_000_000_000,
        "aic_frequency_hz": 1_500_000_000,
        "aiv_frequency_hz": 1_200_000_000,
        "capture_complete": True,
        "raw_transport_complete": True,
        "cores": [{"physical_core_id": core, "raw_bytes": 32, "core_type": 1} for core in (0, 1)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for core in (0, 1):
        (tmp_path / f"core_{core}.bin").write_bytes(raw)
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(tmp_path), "--decoder-library", str(decoder_library)]
    )
    main()
    output = tmp_path / "l0_swimlane_trace.json"
    assert json.loads(output.read_text())["metadata"]["capture_complete"] is True
    assert capsys.readouterr().err == ""

    missing_raw = tmp_path / "core_1.bin"
    if missing_size is None:
        missing_raw.unlink()
    else:
        missing_raw.write_bytes(b"")
    main()
    trace = json.loads(output.read_text())
    intervals = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert [event["args"]["physical_core_id"] for event in intervals] == [0]
    assert trace["metadata"]["capture_complete"] is False
    assert trace["metadata"]["missing_event_cores"] == [1]
    assert all(event["ph"] == "X" for event in trace["traceEvents"])
    warning = capsys.readouterr().err
    assert "L0 data missing; capture incomplete" in warning
    assert "missing event cores: [1]" in warning

    monkeypatch.setattr(sys, "argv", ["l0_swimlane_converter", str(tmp_path)])
    main()
    trace = json.loads(output.read_text())
    assert trace["metadata"]["capture_complete"] is False
    assert trace["metadata"]["missing_event_cores"] == [1]
    assert [event["args"]["physical_core_id"] for event in trace["traceEvents"]] == [0]
    assert "L0 data missing" in capsys.readouterr().err

    (tmp_path / "core_0.bin").unlink()
    monkeypatch.setattr(
        sys, "argv", ["l0_swimlane_converter", str(tmp_path), "--decoder-library", str(decoder_library)]
    )
    main()
    trace = json.loads(output.read_text())
    assert trace["traceEvents"] == []
    assert trace["metadata"]["capture_complete"] is False
    assert trace["metadata"]["missing_event_cores"] == [0, 1]
    assert "missing event cores: [0, 1]" in capsys.readouterr().err
