#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from importlib import import_module

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l3_task_args, _outputs_dir, _sanitize_for_filename
from simpler_setup.tools.strace_timing import parse_spans


def _verify_runtime_clock_logs(monkeypatch, st_platform, after_finalize=None):
    """Check paired timing artifacts exist before any SceneTest converter runs."""
    scene_test_module = import_module("simpler_setup.scene_test")
    original = scene_test_module.finalize_diagnostic_outputs

    def finalize(case_label, output_prefix, **kwargs):
        for capture in output_prefix.rglob("chip_swimlane_records.json"):
            raw = json.loads(capture.read_text())
            logs = list(capture.parent.glob("host_clock_alignment.*.log"))
            has_host = raw["metadata"].get("orchestrator_source") == "host"
            if raw["chip_swimlane_level"] >= 3 and has_host:
                assert len(logs) == 1, "runtime must export timing logs before upper-level postprocessing"
                assert "clock_alignment" not in raw["metadata"]
                with logs[0].open() as stream:
                    spans = list(parse_spans(stream))
                assert len({(span.pid, span.inv) for span in spans}) == 1
                required = {"chip.run", "chip.run.runner_run", "chip.run.runner_run.device_wall"}
                if st_platform == "a5":
                    required.add("chip.run.runner_run.aicpu_launch")
                assert required <= {span.name for span in spans}
            else:
                assert not logs, "Device-only captures must not export Host alignment logs"
        result = original(case_label, output_prefix, **kwargs)
        if after_finalize is not None:
            after_finalize(case_label, output_prefix)
        return result

    monkeypatch.setattr(scene_test_module, "finalize_diagnostic_outputs", finalize)


FANOUT_WIDTH = 32
EXPECTED_PROFILED_TASKS = 1 + FANOUT_WIDTH + 1 + 1  # root + children + final + terminal dummy


@scene_test(level=2, runtime="host_build_graph")
class TestSchedulerPhases(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/scheduler_phases_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_noop.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_empty.cpp",
                "core_type": "aiv",
                "signature": [],
            },
        ],
    }

    CASES = [
        {
            "name": "resolve_dummy",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("input", torch.zeros(1, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        args.input[0] = 1

    def test_run(self, st_platform, st_worker, request, monkeypatch):
        _verify_runtime_clock_logs(monkeypatch, st_platform)
        outputs_dir = _outputs_dir()
        previous_outputs = (
            {path: path.stat().st_mtime_ns for path in outputs_dir.iterdir()} if outputs_dir.exists() else {}
        )
        super().test_run(st_platform, st_worker, request)
        level = self._effective_enable_chip_swimlane(request)
        if level == 0:
            return

        for case in self._matching_cases(st_platform, request):
            case_label = _sanitize_for_filename(f"TestSchedulerPhases_{case['name']}")
            matches = [
                path
                for path in outputs_dir.glob(f"{case_label}_*")
                if path not in previous_outputs or path.stat().st_mtime_ns > previous_outputs[path]
            ]
            assert matches, f"no output directory created for {case_label}"
            output_prefix = max(matches, key=lambda path: path.stat().st_mtime_ns)
            raw = json.loads((output_prefix / "chip_swimlane_records.json").read_text())
            aicore_rows = raw["aicore_tasks"]
            assert len(aicore_rows) == EXPECTED_PROFILED_TASKS, (
                f"task timing covers {len(aicore_rows)} tasks, expected {EXPECTED_PROFILED_TASKS} "
                "for the fanout/fanin DAG and terminal dummy"
            )

            if level >= 2:
                scheduler_tasks = raw["scheduler_tasks"]
                assert scheduler_tasks["producer"] == "aicore"
                scheduler_rows = scheduler_tasks["records"]
                assert len(scheduler_rows) == len(aicore_rows)
                # HBG builds these rows itself from host-side traces, so the
                # epoch it stamps has to agree with the AICore stream's.
                assert {int(row[6]) for row in aicore_rows} == {int(row[4]) for row in scheduler_rows}
                aicore_by_key = {(int(row[0]), int(row[2])): row for row in aicore_rows}
                assert {(int(row[0]), int(row[1])) for row in scheduler_rows} == set(aicore_by_key)
                for core_id, reg_task_id, dispatch_cycles, finish_cycles, _run_epoch in scheduler_rows:
                    aicore_row = aicore_by_key[(int(core_id), int(reg_task_id))]
                    assert 0 < dispatch_cycles <= aicore_row[3] <= aicore_row[4] <= finish_cycles
                lifecycle_records = raw["aicpu_lifecycle_records"]
                assert lifecycle_records, "AICPU lifecycle records are missing"
                thread_ids = [int(record["aicpu_thread_id"]) for record in lifecycle_records]
                assert thread_ids == list(range(len(thread_ids)))
                assert all("worker_id" not in record for record in lifecycle_records)
                for record in lifecycle_records:
                    assert 0 < record["handshake_start_cycles"] <= record["handshake_complete_cycles"]
                    assert 0 < record["context_publish_start_cycles"] <= record["context_publish_complete_cycles"]
                    assert 0 < record["bootstrap_wait_start_cycles"] <= record["bootstrap_complete_cycles"]
                    assert 0 < record["register_release_start_cycles"] <= record["register_release_end_cycles"]
                    assert 0 < record["exit_signal_start_cycles"] <= record["exit_signal_end_cycles"]
                    assert 0 < record["exit_wait_start_cycles"] <= record["exit_wait_end_cycles"]
                leader = lifecycle_records[0]
                assert 0 < leader["config_start_cycles"] <= leader["topology_complete_cycles"]
                assert all(
                    record["config_start_cycles"] == record["topology_complete_cycles"] == 0
                    for record in lifecycle_records[1:]
                )
            else:
                assert "scheduler_tasks" not in raw
                assert "aicpu_lifecycle_records" not in raw

            if level >= 3:
                streams = raw["scheduler_records"]["streams"]
                assert streams, "A5 HBG AICore Scheduler records are missing"
                assert all(stream["producer"] == "aicore" for stream in streams)
                assert all("runtime" not in stream for stream in streams)
                assert all(stream["capture"]["dropped"] == 0 for stream in streams)
                emitted_kinds = {record["kind"] for stream in streams for record in stream["records"]}
                required_kinds = {"bootstrap", "state_probe", "dispatch", "complete", "resolve", "refill", "idle"}
                assert required_kinds <= emitted_kinds, (
                    f"missing Scheduler kinds: {sorted(required_kinds - emitted_kinds)}"
                )
                assert not ({"fanin", "ready_claim", "ready_steal", "direct_refill"} & emitted_kinds)
                records = [record for stream in streams for record in stream["records"]]
                launch_kinds = {"dispatch", "worksteal", "refill"}
                launches_by_task = Counter(
                    int(record["task_id"])
                    for record in records
                    if record["kind"] in launch_kinds and record["task_id"] is not None
                )
                assert set(launches_by_task) == {int(row[1]) for row in aicore_rows}
                assert set(launches_by_task.values()) == {1}
                profiled_task_ids = set(launches_by_task)
                state_probe_task_ids = {
                    int(record["task_id"])
                    for record in records
                    if record["kind"] == "state_probe" and record["task_id"] is not None
                }
                assert state_probe_task_ids <= profiled_task_ids
                inbox_launch_task_ids = {
                    int(record["task_id"])
                    for record in records
                    if record["kind"] in {"dispatch", "worksteal"} and record["task_id"] is not None
                }
                assert inbox_launch_task_ids <= state_probe_task_ids, (
                    "Ready Inbox launches are missing state_probe records: "
                    f"{sorted(inbox_launch_task_ids - state_probe_task_ids)}"
                )
                refill_task_ids = {
                    int(record["task_id"])
                    for record in records
                    if record["kind"] == "refill" and record["task_id"] is not None
                }
                assert profiled_task_ids - state_probe_task_ids <= refill_task_ids, (
                    "tasks without state_probe were not launched by refill: "
                    f"{sorted(profiled_task_ids - state_probe_task_ids - refill_task_ids)}"
                )
                for stream in streams:
                    ordered = sorted(
                        stream["records"],
                        key=lambda record: (record["start_cycles"], record["end_cycles"]),
                    )
                    assert all(
                        previous["end_cycles"] <= current["start_cycles"]
                        for previous, current in zip(ordered, ordered[1:])
                    )
                probe_by_task = {
                    int(record["task_id"]): record
                    for record in records
                    if record["kind"] == "state_probe" and record["task_id"] is not None
                }
                launch_by_task = {
                    int(record["task_id"]): record
                    for record in records
                    if record["kind"] in launch_kinds and record["task_id"] is not None
                }
                assert all(
                    probe["end_cycles"] <= launch_by_task[task_id]["start_cycles"]
                    for task_id, probe in probe_by_task.items()
                )
            else:
                assert "scheduler_records" not in raw


def run_profiled_chip(orch, callables, task_args, config):
    chip_args, _ = _build_l3_task_args(task_args, callables.profiled_chip_sig)
    callables.keep(chip_args)
    orch.submit_next_level(callables.profiled_chip, chip_args, config, worker=0)


@scene_test(level=3, runtime="host_build_graph")
class TestConsecutiveClockAlignment(SceneTestCase):
    """A persistent ChipWorker produces and aligns two separate captures."""

    CALLABLE = {
        "orchestration": run_profiled_chip,
        "callables": [{"name": "profiled_chip", **TestSchedulerPhases.CALLABLE}],
    }
    CASES = [
        {
            "name": name,
            "platforms": ["a5"],
            "manual": True,
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {},
        }
        for name in ("first", "later")
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("input", torch.zeros(1, dtype=torch.int32).share_memory_()))

    def compute_golden(self, args, params):
        args.input[0] = 1

    def test_run(self, st_platform, st_worker, request, monkeypatch, tmp_path):
        level = self._effective_enable_chip_swimlane(request)
        alignment_enabled = level >= 3 and (level != 3 or os.environ.get("SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE"))
        captured_outputs = {}

        archive_root = tmp_path / "capture-archive"

        def retain_then_delete_first(case_label, output_prefix):
            output_prefix = output_prefix.resolve()
            if not captured_outputs:
                archived = archive_root / output_prefix.name
                shutil.copytree(output_prefix, archived)
                captured_outputs[case_label] = (archived, output_prefix)
                shutil.rmtree(output_prefix)
            else:
                captured_outputs[case_label] = (output_prefix, output_prefix)

        _verify_runtime_clock_logs(
            monkeypatch,
            st_platform,
            after_finalize=retain_then_delete_first if alignment_enabled else None,
        )
        super().test_run(st_platform, st_worker, request)
        if not alignment_enabled:
            return

        invocations = []
        for case in self._matching_cases(st_platform, request):
            label = _sanitize_for_filename(f"TestConsecutiveClockAlignment_{case['name']}")
            output_prefix, original_prefix = captured_outputs[label]
            captures = list(output_prefix.rglob("chip_swimlane_records.json"))
            assert len(captures) == 1
            capture = captures[0]
            raw = json.loads(capture.read_text())
            merged = json.loads((output_prefix / "l3_swimlane.json").read_text())
            alignment = raw["metadata"]["clock_alignment"]
            assert alignment["status"] == "bounded"
            rank_metadata = merged["metadata"]["ranks"]
            assert len(rank_metadata) == 1
            expected_input = original_prefix / capture.relative_to(output_prefix)
            assert rank_metadata[0]["input"] == str(expected_input)
            placement = rank_metadata[0]["placement"]
            pid, inv = placement["outer_pid"], placement["outer_inv"]
            assert pid != os.getpid(), "the producer must be a forked ChipWorker"
            log = capture.with_name(f"host_clock_alignment.{pid}.log")
            with log.open() as stream:
                spans = list(parse_spans(stream))
            assert {(span.pid, span.inv) for span in spans} == {(pid, inv)}
            assert placement["place_lo_ns"] >= placement["aicpu_launch_ns"]
            # The device anchor is the earliest device slice in both previews.
            device_events = [
                event
                for event in merged["traceEvents"]
                if event.get("ph") == "X"
                and event.get("cat") in {"aicpu_lifecycle", "aicpu_scheduler", "aicore_scheduler", "kernel"}
            ]
            assert device_events
            anchor_ns = merged["metadata"]["global_origin_ns"] + min(event["ts"] for event in device_events) * 1000
            assert abs(anchor_ns - alignment["host_anchor_ns"]) <= 1
            assert anchor_ns >= placement["aicpu_launch_ns"]
            invocations.append((pid, inv))
        assert len(invocations) == 2
        assert invocations[0][0] == invocations[1][0], "cases must reuse the same ChipWorker"
        assert invocations[0][1] != invocations[1][1], "each capture must select its own invocation"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
