#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import copy
import json
import sys

import pytest

from simpler_setup.scene_test import finalize_diagnostic_outputs
from simpler_setup.tools import swimlane_converter as sc


def _capture(frequency=1_000_000_000):
    def cycle(ns):
        return 1_000_000 + ns * frequency // 1_000_000_000

    return {
        "chip_swimlane_level": 4,
        "metadata": {
            "clock_freq_hz": frequency,
            "num_cores": 1,
            "core_types": ["aiv"],
            "orchestrator_source": "host",
            "host_orchestration_origin_ns": 1_000_000,
            "host_clock_domain_id": "test-host",
            "host_capture": {
                "status": "complete",
                "expected_records": 1,
                "recorded_records": 1,
                "dropped_records": 0,
                "error": None,
            },
        },
        "host_orchestrator_phases": [
            [{"submit_idx": 0, "task_id": 7, "start_host_ns": 1_000_000, "end_host_ns": 1_000_100}]
        ],
        "host_device_uploads": [
            {"phase": "arena_h2d", "start_host_ns": 1_000_200, "end_host_ns": 1_000_300, "detail": 0}
        ],
        "aicore_tasks": [[0, 7, 1, cycle(500), cycle(600), 0]],
        "scheduler_tasks": {"schema_version": 1, "producer": "aicpu", "records": [[0, 1, cycle(400), cycle(650)]]},
        "scheduler_records": {
            "schema_version": 1,
            "streams": [
                {
                    "platform": "a5",
                    "runtime": "host_build_graph",
                    "producer": "aicpu",
                    "scheduler_id": 0,
                    "worker_id": 0,
                    "core_type": "aicpu",
                    "physical_core_id": None,
                    "capture": {"committed": 1, "dropped": 0, "truncated": False},
                    "records": [
                        {
                            "start_cycles": cycle(300),
                            "end_cycles": cycle(700),
                            "loop_iter": 1,
                            "kind": "dispatch",
                            "tasks_processed": 1,
                            "task_id": None,
                        }
                    ],
                    "metrics": [],
                }
            ],
        },
    }


def _artifacts(tmp_path, *, frequency=1_000_000_000, logs=True, sidecar_pid=42):
    path = tmp_path / "chip_swimlane_records.json"
    path.write_text(json.dumps(_capture(frequency)))
    sidecar = {
        "schema_version": 1,
        "host_pid": sidecar_pid,
        "run_id": 1,
        "endpoint_dispatch_id": 3,
        "pipeline_slot": 0,
        "pipeline_generation": 1,
    }
    (tmp_path / "dispatch_identity.json").write_text(json.dumps(sidecar))
    log = tmp_path / "host.42.log"
    if logs:
        head = "[mono_ns=3000000][T0x1][TIMING] emit_host_span: [STRACE] v=1 pid=42 tid=42 inv=1 hid=abc"
        log.write_text(
            "\n".join(
                [
                    f"{head} depth=0 name=chip.run ts=999000 dur=1010000 run_id=1 dispatch_id=3 slot_id=0 generation=1",
                    f"{head} depth=1 name=chip.run.runner_run ts=2000000 dur=5000",
                    f"{head} depth=2 name=chip.run.runner_run.device_wall ts=0 dur=2000 clk=dev",
                    f"{head} depth=3 name=chip.run.runner_run.device_wall.sched ts=700 dur=500 clk=dev",
                ]
            )
            + "\n"
        )
    return path, log


def _convert(monkeypatch, path, *extra):
    monkeypatch.setattr(sys, "argv", ["swimlane_converter", str(path), *extra])
    assert sc.main() == 0


@pytest.mark.parametrize("frequency", [1_000_000_000, 50_000_000])
def test_single_capture_persists_alignment_and_reuses_it_without_logs(tmp_path, monkeypatch, frequency):
    path, log = _artifacts(tmp_path, frequency=frequency)
    original = json.loads(path.read_text())
    _convert(monkeypatch, path)
    enriched = json.loads(path.read_text())
    alignment = enriched["metadata"].pop("clock_alignment")
    assert alignment["status"] == "bounded"
    assert alignment["method"] == "span_containment_v1"
    assert alignment["host_anchor_ns"] == 2_000_750
    assert alignment["host_anchor_min_ns"] == 2_000_700
    assert alignment["host_anchor_max_ns"] == 2_003_800
    assert alignment["uncertainty_ns"] == 3100
    assert enriched == original
    data = sc.read_perf_data(path)
    assert data["timeline_metadata"]["layout"] == "containment_spliced"
    assert data["tasks"][0]["dispatch_time_us"] > 1000
    assert alignment["host_anchor_min_ns"] <= alignment["host_anchor_ns"] <= alignment["host_anchor_max_ns"]
    mapped = (
        alignment["host_anchor_ns"]
        + (original["scheduler_tasks"]["records"][0][2] - alignment["device_anchor_cycles"]) * 1_000_000_000 / frequency
    )
    assert data["tasks"][0]["dispatch_time_us"] == pytest.approx((mapped - 1_000_000) / 1000)
    saved = path.read_bytes()
    log.unlink()
    (tmp_path / "dispatch_identity.json").unlink()
    _convert(monkeypatch, path)
    assert path.read_bytes() == saved
    assert sc.read_perf_data(path)["tasks"] == data["tasks"]


@pytest.mark.parametrize(
    "condition", ["missing", "filtered", "wrong_pid", "wrong_dispatch", "wrong_host_time", "ambiguous"]
)
def test_single_capture_skips_alignment_without_matching_windows(tmp_path, monkeypatch, condition):
    path, log = _artifacts(tmp_path, logs=condition != "missing", sidecar_pid=99 if condition == "wrong_pid" else 42)
    if condition == "filtered":
        log.write_text("[WARN] profiling enabled\n")
    if condition == "wrong_dispatch":
        log.write_text(log.read_text().replace("dispatch_id=3", "dispatch_id=4"))
    if condition == "wrong_host_time":
        log.write_text(log.read_text().replace("ts=999000", "ts=2999000"))
    if condition == "ambiguous":
        log.write_text(log.read_text() + log.read_text().replace("inv=1", "inv=2"))
    original = copy.deepcopy(json.loads(path.read_text()))
    _convert(monkeypatch, path)
    data = sc.read_perf_data(path)
    assert not data["timeline_metadata"]["cross_domain_latency_available"]
    assert json.loads(path.read_text())["metadata"]["clock_alignment"]["status"] == "unavailable"
    result = json.loads(path.read_text())
    result["metadata"].pop("clock_alignment")
    assert result == original


def test_alignment_does_not_survive_changes_to_capture(tmp_path, monkeypatch):
    path, _ = _artifacts(tmp_path)
    _convert(monkeypatch, path)
    document = json.loads(path.read_text())
    document["aicore_tasks"][0][3] += 1
    path.write_text(json.dumps(document))
    assert not sc.read_perf_data(path)["timeline_metadata"]["cross_domain_latency_available"]


def test_unaligned_composite_keeps_dispatch_after_host_uploads(tmp_path):
    path, _ = _artifacts(tmp_path, logs=False)
    data = sc.read_perf_data(path)
    dispatch = data["aicpu_scheduler_phases"][0][0]["start_time_us"]
    assert dispatch >= 0.3
    assert not data["timeline_metadata"]["cross_domain_latency_available"]


def test_single_capture_accepts_explicit_log_path(tmp_path, monkeypatch):
    path, log = _artifacts(tmp_path)
    external = tmp_path / "test_output.log"
    log.rename(external)
    _convert(monkeypatch, path, "--host-log", str(external))
    assert json.loads(path.read_text())["metadata"]["clock_alignment"]["status"] == "bounded"


def test_legacy_capture_without_sidecar_matches_its_host_records(tmp_path, monkeypatch):
    path, _ = _artifacts(tmp_path)
    (tmp_path / "dispatch_identity.json").unlink()
    _convert(monkeypatch, path)
    assert json.loads(path.read_text())["metadata"]["clock_alignment"]["status"] == "bounded"


def test_alignment_preserves_record_formatting_and_only_adds_metadata(tmp_path, monkeypatch):
    path, _ = _artifacts(tmp_path)
    before = path.read_bytes()
    original = json.loads(before)
    _convert(monkeypatch, path)
    after = path.read_bytes()
    record = json.loads(after)["metadata"]["clock_alignment"]
    # Compact task rows must stay compact, even for a large capture.
    assert json.dumps(original["aicore_tasks"]).encode() in after
    assert len(after) - len(before) <= len(json.dumps(record).encode()) + 128


@pytest.mark.parametrize("logs", [True, False])
def test_scene_postprocessing_enriches_capture_and_reports_skips(tmp_path, caplog, logs):
    path, _ = _artifacts(tmp_path, logs=logs)
    finalize_diagnostic_outputs("alignment", tmp_path, chip_swimlane=4)
    assert json.loads(path.read_text())["metadata"]["clock_alignment"]["status"] == (
        "bounded" if logs else "unavailable"
    )
    if not logs:
        assert "clock alignment skipped" in caplog.text


@pytest.mark.parametrize("legacy", [False, True])
def test_device_only_capture_skips_automatic_containment(tmp_path, monkeypatch, legacy):
    path, _ = _artifacts(tmp_path)
    raw = json.loads(path.read_text())
    for key in ("orchestrator_source", "host_orchestration_origin_ns", "host_capture"):
        raw["metadata"].pop(key)
    raw.pop("host_orchestrator_phases")
    raw.pop("host_device_uploads")
    streams = raw["scheduler_records"]["streams"]
    streams[0]["runtime"] = "tensormap_and_ringbuffer"
    if legacy:
        raw["aicpu_scheduler_phases"] = [stream["records"] for stream in streams]
        raw.pop("scheduler_records")
    path.write_text(json.dumps(raw))
    original = path.read_bytes()

    def unexpected_containment(*args, **kwargs):
        pytest.fail("device-only single-file conversion must not calculate containment")

    monkeypatch.setattr(sc.containment, "host_windows", unexpected_containment)
    _convert(monkeypatch, path)
    assert path.read_bytes() == original
    assert "clock_alignment" not in sc.read_perf_data(path).get("timeline_metadata", {})


@pytest.mark.parametrize("source", ["runtime", "legacy_host", "host_upload_only"])
def test_hbg_alignment_recognizes_runtime_and_legacy_host_capture(tmp_path, monkeypatch, source):
    path, _ = _artifacts(tmp_path)
    raw = json.loads(path.read_text())
    if source == "runtime":
        for key in ("orchestrator_source", "host_capture"):
            raw["metadata"].pop(key)
        raw.pop("host_orchestrator_phases")
    elif source == "legacy_host":
        raw["aicpu_scheduler_phases"] = [stream["records"] for stream in raw.pop("scheduler_records")["streams"]]
    else:
        for key in ("orchestrator_source", "host_capture"):
            raw["metadata"].pop(key)
        raw.pop("host_orchestrator_phases")
        raw.pop("scheduler_records")
    path.write_text(json.dumps(raw))
    _convert(monkeypatch, path)
    assert json.loads(path.read_text())["metadata"]["clock_alignment"]["status"] == "bounded"


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_hbg_without_host_capture_skips_automatic_containment(tmp_path, monkeypatch, level):
    path, _ = _artifacts(tmp_path)
    raw = json.loads(path.read_text())
    raw["chip_swimlane_level"] = level
    for key in ("orchestrator_source", "host_orchestration_origin_ns", "host_capture"):
        raw["metadata"].pop(key)
    raw.pop("host_orchestrator_phases")
    raw.pop("host_device_uploads")
    if level < 3:
        raw.pop("scheduler_records")
    if level < 2:
        raw.pop("scheduler_tasks")
    path.write_text(json.dumps(raw))
    original = path.read_bytes()

    def unexpected_containment(*args, **kwargs):
        pytest.fail("HBG without Host capture must not calculate containment")

    monkeypatch.setattr(sc.containment, "host_windows", unexpected_containment)
    _convert(monkeypatch, path)
    assert path.read_bytes() == original


def test_level3_with_explicit_host_phase_capture_still_aligns(tmp_path, monkeypatch):
    path, _ = _artifacts(tmp_path)
    raw = json.loads(path.read_text())
    raw["chip_swimlane_level"] = 3
    path.write_text(json.dumps(raw))
    _convert(monkeypatch, path)
    assert json.loads(path.read_text())["metadata"]["clock_alignment"]["status"] == "bounded"
