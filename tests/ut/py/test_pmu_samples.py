# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PMU quality is consumed with raw rows, without invented replay identities."""

import csv
import json

import pytest

from simpler_setup.tools.pmu_samples import main, read_pmu_samples

_COLUMNS = [
    "thread_id",
    "core_id",
    "task_id",
    "func_id",
    "core_type",
    "pmu_total_cycles",
    "cycles",
    "event_type",
    "run_epoch",
]


def _artifact(tmp_path, *, rows=3):
    path = tmp_path / "pmu.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(_COLUMNS)
        for _ in range(rows):
            writer.writerow([0, 1, "0x0000000000000900", 2, 0, 4000, 7, 2, 710])
    metadata = {
        "recorded_window": True,
        "counts_unknown": True,
        "run_epoch": 710,
        "event_type": 2,
        "device_total_records": None,
        "device_dropped_records": None,
        "device_fatal": None,
        "written_records": rows,
        "transport_retired_buffers": None,
        "collector_transport_retired_buffers": 0,
        "collector_unknown_epoch_buffers": 0,
        "clamped_buffers": 0,
        "pmu_protection": "held",
        "csv_file": "pmu.csv",
        "columns": _COLUMNS,
        "collection_verdict": "aggregate_counts_unknown",
    }
    return path, metadata


def _write(path, metadata):
    path.with_name("pmu.collection.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("rows", [0, 3])
def test_recorded_quality_preserves_zero_and_duplicate_task_samples(tmp_path, rows, capsys):
    path, metadata = _artifact(tmp_path, rows=rows)
    _write(path, metadata)
    report = read_pmu_samples(path, recorded=True)
    assert len(report["records"]) == rows
    assert len({row["task_id"] for row in report["records"]}) == bool(rows)
    assert report["collection"]["device_total_records"] is None
    main([str(path), "--recorded"])
    assert json.loads(capsys.readouterr().out)["records"] == rows


@pytest.mark.parametrize(
    "change",
    [
        {"written_records": 1},
        {"run_epoch": 711},
        {"event_type": 4},
        {"device_total_records": 3},
        {"counts_unknown": False},
        {"collection_verdict": "write_failed"},
        {"columns": _COLUMNS[:-1]},
        {"collector_transport_retired_buffers": 1},
        {"collector_unknown_epoch_buffers": 1},
        {"clamped_buffers": 1},
        {"written_records": True},
        {"pmu_protection": "unknown"},
    ],
)
def test_recorded_reader_rejects_stale_fabricated_or_lossy_artifacts(tmp_path, change):
    path, metadata = _artifact(tmp_path)
    _write(path, metadata | change)
    with pytest.raises(ValueError):
        read_pmu_samples(path, recorded=True)


def test_recorded_metadata_is_required_and_ordinary_mode_cannot_hide_it(tmp_path):
    path, metadata = _artifact(tmp_path)
    assert len(read_pmu_samples(path)["records"]) == 3
    with pytest.raises(FileNotFoundError):
        read_pmu_samples(path, recorded=True)
    _write(path, metadata)
    with pytest.raises(ValueError, match="recorded=True"):
        read_pmu_samples(path)
