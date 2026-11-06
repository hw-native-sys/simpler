# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ScopeStats reports preserve incomplete collection evidence."""

import hashlib
import json
import re
import tracemalloc
from pathlib import Path

import pytest

from simpler_setup.tools.tmr.scope_stats_plot import process

_NORMAL_META = {
    "fatal": False,
    "dropped": 0,
    "total": 2,
    "task_window_max": [100],
    "heap_max": [1048576],
    "dep_pool_max": [100],
    "tensormap_max": 100,
}
_RECORDS = [
    {"site": "kernel.cpp:12", "ring": 0, "phase": "begin", "task_window_start": 0, "task_window_end": 10},
    {"site": "kernel.cpp:12", "ring": 0, "phase": "end", "task_window_start": 10, "task_window_end": 25},
]


def _render(tmp_path, metadata, records=_RECORDS):
    source = tmp_path / "scope_stats.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in [metadata, *records]) + "\n")
    [report] = process(source)
    return report.read_text()


@pytest.mark.parametrize("unknown", [{"counts_unknown": True}, {"collection_verdict": "partial_cut_unknown"}])
def test_unknown_collection_does_not_claim_healthy_device_counts(tmp_path, unknown):
    html = _render(
        tmp_path,
        {**_NORMAL_META, **unknown, "host_received_records": 3, "host_retained_records": 2},
    )

    assert "Healthy" not in html
    assert "Device statistics unknown" in html
    assert "Device total: unknown" in html
    assert "Device dropped: unknown" in html
    assert "Device fatal: unknown" in html
    assert "Host received: 3 records" in html
    assert "Host retained: 2 records" in html
    assert "<svg" in html
    assert "25 slots" in html


def test_partial_safe_collection_is_visible_even_without_device_drops(tmp_path):
    html = _render(
        tmp_path,
        {
            **_NORMAL_META,
            "total": 3,
            "collection_verdict": "partial_safe",
            "counts_unknown": False,
            "host_received_records": 3,
            "host_retained_records": 2,
        },
    )

    assert "Healthy" not in html
    assert "Partial collection" in html
    assert "Device dropped: 0" in html
    assert "Device fatal: false" in html
    assert "Host received: 3 records" in html
    assert "Host retained: 2 records" in html
    assert "Partial evidence" in html
    assert "<svg" in html


@pytest.mark.parametrize("with_pair", [True, False])
def test_unpaired_samples_remain_visible_without_inventing_scope_pairs(tmp_path, with_pair):
    records = [
        *(_RECORDS if with_pair else []),
        {**_RECORDS[0], "site": "unfinished.cpp:9"},
        {**_RECORDS[1], "site": "missing_begin.cpp:4"},
    ]
    html = _render(
        tmp_path,
        {
            **_NORMAL_META,
            "counts_unknown": True,
            "host_received_records": len(records),
            "host_retained_records": len(records),
        },
        records,
    )

    assert "2 unpaired records" in html
    assert "raw JSONL" in html
    assert "scope_stats.jsonl" in html
    if with_pair:
        assert "1 total paired scopes" in html
        assert "<svg" in html
        assert "25 slots" in html
    else:
        assert "0 total paired scopes" in html
        assert "<svg" not in html
        assert "No paired scopes found" in html


@pytest.mark.parametrize("capacity", [100, None])
def test_ordinary_unknown_metadata_preserves_samples_and_only_known_capacity(tmp_path, capacity):
    html = _render(
        tmp_path,
        {
            "fatal": None,
            "dropped": None,
            "total": None,
            "task_window_max": [capacity] if capacity is not None else None,
            "heap_max": None,
            "dep_pool_max": None,
            "tensormap_max": None,
            "counts_unknown": True,
            "collection_verdict": "partial_cut_unknown",
            "host_received_records": 2,
            "host_retained_records": 2,
        },
    )

    assert "Healthy" not in html
    assert "Device total: unknown" in html
    assert "Device dropped: unknown" in html
    assert "Device fatal: unknown" in html
    assert "<span>0 records</span>" not in html
    assert "<svg" in html
    assert "25 slots" in html
    if capacity is None:
        assert 'class="use-label">n/a</span>' in html
        assert "capacity use 25.00%" not in html
        assert "capacity use 0.00%" not in html
    else:
        assert "100 slots" in html
        assert "capacity use 25.00%" in html


def test_normal_seven_field_report_keeps_known_statistics_and_charts(tmp_path):
    html = _render(tmp_path, _NORMAL_META)

    assert "Healthy: no fatal, no dropped" in html
    assert "<span>2 records</span>" in html
    assert "Partial evidence" not in html
    assert "Device statistics unknown" not in html
    assert "unpaired records" not in html
    assert "<svg" in html
    assert "25 slots" in html
    assert "capacity use 25.00%" in html


@pytest.mark.parametrize("unknown", [{"counts_unknown": True}, {"collection_verdict": "partial_cut_unknown"}])
def test_unknown_counts_preserve_a_confirmed_fatal_latch(tmp_path, unknown):
    html = _render(tmp_path, {**_NORMAL_META, **unknown, "fatal": True})

    assert "Healthy" not in html
    assert '<span class="status danger">Fatal' in html
    assert "Device fatal: true" in html
    assert "Device fatal: unknown" not in html
    assert "fatal status are unknown" not in html
    assert "Device total: unknown" in html
    assert "Device dropped: unknown" in html
    assert "Partial evidence" in html
    assert "<svg" in html
    assert "25 slots" in html


def test_different_run_epochs_do_not_form_a_scope_pair(tmp_path):
    records = [
        {**_RECORDS[0], "run_epoch": 8200},
        {**_RECORDS[1], "run_epoch": 8201},
    ]
    html = _render(tmp_path, {**_NORMAL_META, "counts_unknown": True}, records)

    assert "2 unpaired records" in html
    assert "0 total paired scopes" in html
    assert "No paired scopes found" in html
    assert "<svg" not in html
    assert "25 slots" not in html


def test_interleaved_epochs_keep_their_own_scope_boundaries(tmp_path):
    records = [
        {**_RECORDS[0], "run_epoch": 8200},
        {**_RECORDS[0], "run_epoch": 8201, "task_window_start": 100, "task_window_end": 110},
        {**_RECORDS[1], "run_epoch": 8200},
        {**_RECORDS[1], "run_epoch": 8201, "task_window_start": 110, "task_window_end": 140},
    ]
    html = _render(tmp_path, {**_NORMAL_META, "total": 4}, records)

    assert "2 total paired scopes" in html
    assert "unpaired records" not in html
    assert "<svg" in html
    assert "<td>40 slots</td>" in html
    assert "<td>140 slots</td>" not in html
    assert "capacity use 40.00%" in html


def test_recorded_window_never_pairs_boundaries_across_replays(tmp_path):
    # The first replay lost its end; the next lost its begin at the same site.
    records = [
        {**_RECORDS[0], "run_epoch": 8200, "buf_seq": 0},
        {**_RECORDS[1], "run_epoch": 8200, "buf_seq": 1},
    ]
    html = _render(
        tmp_path,
        {
            **_NORMAL_META,
            "recorded_window": True,
            "fatal": None,
            "dropped": None,
            "total": None,
            "counts_unknown": True,
            "host_received_records": 3,
            "host_retained_records": 2,
        },
        records,
    )

    assert "Recorded collection window" in html
    assert "Raw boundary samples" in html
    assert "Host received: 3 records" in html
    assert "Host retained: 2 records" in html
    assert "Device total: unknown" in html
    assert "Device dropped: unknown" in html
    assert "Device fatal: unknown" in html
    assert "kernel.cpp:12" in html
    assert "begin" in html and "end" in html
    assert "8200" in html
    assert "100 slots" in html
    assert "same window identity" in html
    assert "Host storage did not retain 1 received samples; device loss is unknown" in html
    assert "total paired scopes" not in html
    assert "Top Peaks" not in html
    assert "25 slots" not in html
    assert "<svg" not in html


def test_recorded_window_preserves_all_samples_and_escapes_text(tmp_path):
    records = [
        {**_RECORDS[0], "site": "<source>&.cpp:12", "run_epoch": 82, "buf_seq": 0},
        {**_RECORDS[1], "site": "<source>&.cpp:12", "run_epoch": 82, "buf_seq": 0},
    ] * 3
    html = _render(
        tmp_path,
        {
            **_NORMAL_META,
            "recorded_window": True,
            "counts_unknown": True,
            "host_received_records": 6,
            "host_retained_records": 6,
        },
        records,
    )

    assert html.count("&lt;source&gt;&amp;.cpp:12") == 6
    assert "<source>" not in html
    assert "Host retained: 6 records" in html
    assert "Showing all 6 boundary samples in this JSONL" in html
    assert "Preview truncated" not in html
    assert "total paired scopes" not in html
    assert "capacity use" not in html


def test_empty_recorded_window_does_not_claim_measured_execution(tmp_path):
    html = _render(
        tmp_path,
        {
            **_NORMAL_META,
            "recorded_window": True,
            "counts_unknown": True,
            "host_received_records": 0,
            "host_retained_records": 0,
        },
        [],
    )

    assert "No retained boundary samples" in html
    assert "does not prove zero replays or zero loss" in html
    assert "Healthy" not in html
    assert "total paired scopes" not in html
    assert "<svg" not in html


def test_recorded_window_preserves_unknown_fields_without_zero_fill(tmp_path):
    html = _render(
        tmp_path,
        {"recorded_window": True, "host_received_records": 1, "host_retained_records": 1},
        [{"site": "scope.cpp:5", "phase": "begin", "run_epoch": 7}],
    )

    assert "scope.cpp:5" in html
    assert html.count("<td>unknown</td>") >= 10
    assert "0 slots" not in html
    assert "0 MiB" not in html
    assert "capacity use" not in html


def test_large_recorded_window_streams_a_bounded_preview(tmp_path, monkeypatch):
    source = tmp_path / "scope_stats.jsonl"
    with source.open("w") as output:
        output.write(json.dumps({"recorded_window": True, "total": None, "counts_unknown": True}) + "\n")
        for index in range(100_000):
            output.write(json.dumps({"site": f"sample-{index}.cpp:1", "phase": "begin", "run_epoch": 7}) + "\n")
    digest = hashlib.sha256(source.read_bytes()).digest()
    read_text = Path.read_text

    def refuse_whole_source_read(path, *args, **kwargs):
        assert path != source, "recorded JSONL must not be read into one string"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse_whole_source_read)
    tracemalloc.start()
    try:
        [report] = process(source)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    html = report.read_text()

    assert peak < 8 * 1024 * 1024
    assert report.stat().st_size < 1024 * 1024
    assert len(re.findall(r"<tr><td>\d+</td>", html)) == 1000
    assert "sample-0.cpp:1" in html
    assert "sample-999.cpp:1" in html
    assert "sample-1000.cpp:1" not in html
    assert "sample-99999.cpp:1" not in html
    assert "Showing the first 1,000 of 100,000 boundary samples in this JSONL" in html
    assert "Preview truncated" in html
    assert 'href="scope_stats.jsonl"' in html
    assert "Device total: unknown" in html
    assert hashlib.sha256(source.read_bytes()).digest() == digest


def test_recorded_preview_links_the_original_jsonl_from_another_directory(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    source = captures / "scope #&.jsonl"
    payload = json.dumps({"recorded_window": True, "host_retained_records": 99}) + "\n" + json.dumps(_RECORDS[0])
    source.write_text(payload)

    [report] = process(source, tmp_path / "reports")
    html = report.read_text()

    assert 'href="../captures/scope%20%23%26.jsonl"' in html
    assert "Host retained: 99 records" in html
    assert "Showing all 1 boundary samples in this JSONL" in html
    assert "Preview truncated" not in html
    assert source.read_text() == payload


def test_recorded_preview_still_rejects_malformed_json_beyond_the_preview(tmp_path):
    source = tmp_path / "scope_stats.jsonl"
    source.write_text(
        json.dumps({"recorded_window": True}) + "\n" + (json.dumps(_RECORDS[0]) + "\n") * 1001 + "not JSON\n"
    )

    with pytest.raises(json.JSONDecodeError):
        process(source)
