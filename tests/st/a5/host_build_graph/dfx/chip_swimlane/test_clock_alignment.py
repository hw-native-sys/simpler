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
from importlib import import_module

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l3_task_args, _sanitize_for_filename
from simpler_setup.tools.strace_timing import parse_spans


def _verify_runtime_clock_logs(monkeypatch, st_platform, after_finalize=None):
    """Check paired timing artifacts exist before any SceneTest converter runs."""
    scene_test_module = import_module("simpler_setup.scene_test")
    original = scene_test_module.finalize_diagnostic_outputs

    def finalize(case_label, output_prefix, **kwargs):
        captures = output_prefix.rglob("chip_swimlane_records.json") if kwargs.get("chip_swimlane", 0) else ()
        for capture in captures:
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


ALIGNMENT_CALLABLE = {
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


def run_profiled_chip(orch, callables, task_args, config):
    chip_args, _ = _build_l3_task_args(task_args, callables.profiled_chip_sig)
    callables.keep(chip_args)
    orch.submit_next_level(callables.profiled_chip, chip_args, config, worker=0)


@scene_test(level=3, runtime="host_build_graph")
class TestConsecutiveClockAlignment(SceneTestCase):
    """A persistent ChipWorker produces and aligns two separate captures."""

    CALLABLE = {
        "orchestration": run_profiled_chip,
        "callables": [{"name": "profiled_chip", **ALIGNMENT_CALLABLE}],
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
                captured_outputs[case_label] = archived
                shutil.rmtree(output_prefix)
            else:
                captured_outputs[case_label] = output_prefix

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
            output_prefix = captured_outputs[label]
            captures = list(output_prefix.rglob("chip_swimlane_records.json"))
            assert len(captures) == 1
            capture = captures[0]
            raw = json.loads(capture.read_text())
            merged = json.loads((output_prefix / "l3_swimlane.json").read_text())
            alignment = raw["metadata"]["clock_alignment"]
            assert alignment["status"] == "bounded"
            rank_metadata = merged["metadata"]["ranks"]
            assert len(rank_metadata) == 1
            assert rank_metadata[0]["input"] == str(capture.relative_to(output_prefix))
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
