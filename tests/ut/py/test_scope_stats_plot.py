# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ScopeStats reports preserve incomplete collection evidence."""

import json

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
