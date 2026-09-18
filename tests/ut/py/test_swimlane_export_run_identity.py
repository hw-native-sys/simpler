# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run identity survives the whole export → parse path.

The load-bearing case here drives the **real C++ exporter** and feeds its bytes
to the **real Python parser**. Asserting the parser against hand-written JSON
would pass even if the exporter emitted a different shape, which is the exact
drift this pair of layers is prone to: the identity is a trailing positional
column on one side and an index on the other.

The cpput binary is built by the "Build and run C++ unit tests" CI step, which
runs this file immediately after ctest for that reason. In the earlier
whole-suite pyut run the binary does not exist yet and the roundtrip skips; the
compatibility and error cases below do not need it and always run.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

from simpler_setup.tools.swimlane_converter import (
    HBG_RUNTIME,
    TMR_RUNTIME,
    generate_chrome_trace_json,
    read_perf_data,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPORT_SUBDIR = "chip_swimlane_export_identity"
# The collector is built once per runtime, since a platform source ships inside each
# runtime's image and the exporter names the runtime it was built for. The roundtrip
# runs under both: the exporter writes metadata.runtime and the parser decodes task
# ids by it, so a layout that only one of them agrees on shows up here and nowhere
# else in this file.
_COLLECTOR_RUNTIMES = (TMR_RUNTIME, HBG_RUNTIME)


def _export_two_runs(tmp_path: Path, runtime: str) -> Path:
    """Produce a two-run capture with the real exporter and return its JSON path."""
    binary = _REPO_ROOT / "tests" / "ut" / "cpp" / "build" / f"test_chip_swimlane_collector_{runtime}"
    if not binary.exists():
        pytest.skip(
            f"{binary} not built; the roundtrip runs in the C++ unit test CI step, "
            "after cmake --build tests/ut/cpp/build"
        )
    completed = subprocess.run(
        [str(binary), "--gtest_filter=*ExportsTwoRunsDistinctly*"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, f"exporter case failed:\n{completed.stdout}\n{completed.stderr}"
    path = tmp_path / _EXPORT_SUBDIR / "chip_swimlane_records.json"
    assert path.exists(), f"the exporter wrote no artifact under {tmp_path}"
    return path


@pytest.mark.parametrize("runtime", _COLLECTOR_RUNTIMES)
def test_two_runs_reusing_task_ids_stay_attributed(tmp_path, runtime):
    """Two runs, same core, same reg_task_ids, different times — each keeps its own."""
    records = read_perf_data(_export_two_runs(tmp_path, runtime))
    assert records["runtime"] == runtime, "the exporter named a runtime the parser did not read back"
    tasks = records["tasks"]
    assert tasks, "the parser produced no tasks from a real two-run capture"

    by_epoch = defaultdict(list)
    for task in tasks:
        by_epoch[task["run_epoch"]].append(task)

    assert sorted(by_epoch) == [7, 8], f"expected exactly runs 7 and 8, got {sorted(by_epoch)}"
    assert len(by_epoch[7]) == 3
    assert len(by_epoch[8]) == 3

    # The runs genuinely collide on identity: same task ids on the same core.
    # That is what makes the epoch load-bearing rather than decorative.
    ids_7 = sorted(t["task_id"] for t in by_epoch[7])
    ids_8 = sorted(t["task_id"] for t in by_epoch[8])
    assert ids_7 == ids_8, "the two runs did not reuse task ids, so this proves nothing"
    assert {t["core_id"] for t in tasks} == {0}

    # And they are separated in time, so a merged parse would be visible as one
    # run's span swallowing the other's.
    assert max(t["end_time_us"] for t in by_epoch[7]) < min(t["start_time_us"] for t in by_epoch[8])

    # No task may carry a duration built from one run's start and the other's end.
    for task in tasks:
        assert task["end_time_us"] >= task["start_time_us"]
        assert task["duration_us"] == pytest.approx(task["end_time_us"] - task["start_time_us"])


@pytest.mark.parametrize("runtime", _COLLECTOR_RUNTIMES)
def test_every_exported_row_carries_an_epoch(tmp_path, runtime):
    """The identity is per row, so the parser must never fall back to None here."""
    path = _export_two_runs(tmp_path, runtime)
    data = json.loads(path.read_text())
    assert data["metadata"]["runtime"] == runtime, "the exporter must name the runtime it was built for"
    assert len(data["aicore_tasks"]) == 6
    for row in data["aicore_tasks"]:
        assert len(row) == 7, f"aicore_tasks row is not seven columns: {row}"
        assert row[6] in (7, 8), f"row carries an unexpected epoch: {row}"
    for row in data["scheduler_tasks"]["records"]:
        assert len(row) == 5, f"scheduler_tasks row is not five columns: {row}"
        assert row[4] in (7, 8)


def _single_run_payload(aicore_rows, scheduler_rows):
    return {
        "chip_swimlane_level": 2,
        "metadata": {
            "runtime": TMR_RUNTIME,
            "clock_freq_hz": 50_000_000,
            "num_cores": 1,
            "core_types": ["aiv"],
        },
        "aicore_tasks": aicore_rows,
        "scheduler_tasks": {
            "producer": "aicpu",
            "records": scheduler_rows,
        },
    }


def test_legacy_capture_without_identity_parses_as_one_run(tmp_path):
    """A pre-identity file keeps its old single-run meaning and gains no fake epoch.

    Six-column AICore rows and four-column scheduler rows carry no identity at
    all. The parser reports that absence as ``None``; coercing it to 0 would be
    wrong because 0 is an epoch a device can really be given, so a legacy file
    would then claim to be run 0.
    """
    payload = _single_run_payload(
        [[0, 0x101, 1, 1010, 1015, 2], [0, 0x102, 2, 1020, 1025, 2]],
        [[0, 1, 1006, 1017], [0, 2, 1016, 1027]],
    )
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))

    tasks = read_perf_data(path)["tasks"]
    assert len(tasks) == 2
    assert {t["run_epoch"] for t in tasks} == {None}, "a legacy capture was given a fabricated epoch"
    assert sorted(t["task_id"] for t in tasks) == [0x101, 0x102]


def test_duplicate_identity_within_one_run_is_still_an_error(tmp_path):
    """The negative control: the epoch widens the key, it does not soften it.

    Two rows with the same (run_epoch, core_id, reg_task_id) are a real defect —
    one dispatch recorded twice — and must not be silently deduplicated just
    because the key now has three components.
    """
    payload = _single_run_payload(
        [[0, 0x101, 1, 1010, 1015, 2, 7], [0, 0x999, 1, 1030, 1035, 2, 7]],
        [[0, 1, 1006, 1017, 7]],
    )
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="duplicate aicore_tasks join key"):
        read_perf_data(path)


def test_same_ids_in_different_runs_are_not_a_duplicate(tmp_path):
    """The other half of that control: the identical key in two runs is legal."""
    payload = _single_run_payload(
        [[0, 0x101, 1, 1010, 1015, 2, 7], [0, 0x101, 1, 9010, 9015, 2, 8]],
        [[0, 1, 1006, 1017, 7], [0, 1, 9006, 9017, 8]],
    )
    path = tmp_path / "two_runs.json"
    path.write_text(json.dumps(payload))

    tasks = read_perf_data(path)["tasks"]
    assert len(tasks) == 2, "the same core+reg_task_id in two runs collapsed into one task"
    assert sorted(t["run_epoch"] for t in tasks) == [7, 8]


def _trace_tasks_for_two_runs():
    """Two runs of the same two-task chain: same ids, same core, run 8 far later."""
    tasks = []
    for epoch, base in ((7, 10.0), (8, 100.0)):
        for index, task_id in enumerate((0x101, 0x102)):
            start = base + index * 2.0
            tasks.append(
                {
                    "task_id": task_id,
                    "func_id": 0,
                    "core_id": 0,
                    "core_type": "aiv",
                    "start_time_us": start,
                    "end_time_us": start + 1.0,
                    "duration_us": 1.0,
                    "dispatch_time_us": start - 0.5,
                    "finish_time_us": start + 1.2,
                    "receive_time_us": start - 0.2,
                    "local_setup_us": 0.2,
                    "run_epoch": epoch,
                }
            )
    return tasks


def _dependency_flow_pairs(trace_path):
    """(start, finish) endpoint pairs for every dependency arrow in the trace."""
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    by_id = defaultdict(dict)
    for event in events:
        if event.get("cat") == "flow" and event.get("name") in ("dependency", "hb_violation"):
            by_id[event["id"]][event["ph"]] = event
    return [(pair["s"], pair["f"]) for pair in by_id.values() if "s" in pair and "f" in pair]


def test_two_runs_do_not_produce_cross_run_dependency_arrows(tmp_path):
    """The final trace keeps run boundaries, not just the parsed task list.

    deps.json is the run-independent static graph, so both runs of A -> B carry
    the same edge. Rendering from a task map merged across runs pairs run 1's
    producer with run 2's consumer: an arrow that moves forward in wall-clock
    time but describes a dependency that never existed.

    Export → parse cannot catch this; only the rendered trace can.
    """
    tasks = _trace_tasks_for_two_runs()
    out = tmp_path / "trace.json"
    generate_chrome_trace_json(tasks, str(out), deps_edges={0x101: [0x102]}, runtime_name=TMR_RUNTIME)

    pairs = _dependency_flow_pairs(out)
    assert pairs, "no dependency arrows were emitted, so this proves nothing"

    run7_end = max(t["end_time_us"] for t in tasks if t["run_epoch"] == 7)
    run8_start = min(t["start_time_us"] for t in tasks if t["run_epoch"] == 8)

    # No arrow may span the two runs.
    for start_event, finish_event in pairs:
        low = min(start_event["ts"], finish_event["ts"])
        high = max(start_event["ts"], finish_event["ts"])
        assert not (low <= run7_end and high >= run8_start), (
            f"dependency arrow spans both runs: ({low}, {high}); run 7 ends {run7_end}, run 8 starts {run8_start}"
        )

    # And each run must keep its own arrow in each view. This is the assertion
    # that actually discriminates: rendering from a map merged across runs makes
    # one task's two executions look like two SPMD subtasks of a single task, so
    # the anchor collapses to the earliest slice and the later run's arrow is
    # dropped rather than misdrawn. Counting per view per run catches that;
    # checking only that no arrow spans runs does not.
    per_view_per_run = defaultdict(set)
    for start_event, finish_event in pairs:
        view = start_event["pid"]
        run = 7 if max(start_event["ts"], finish_event["ts"]) <= run7_end else 8
        per_view_per_run[view].add(run)
    assert per_view_per_run, "no view emitted a dependency arrow"
    for view, runs in sorted(per_view_per_run.items()):
        assert runs == {7, 8}, f"view pid={view} rendered arrows for runs {sorted(runs)}, expected both"
    assert len(pairs) == 2 * len(per_view_per_run), (
        f"expected exactly one arrow per run per view, got {len(pairs)} across {len(per_view_per_run)} view(s)"
    )


def test_two_runs_do_not_report_a_happens_before_violation(tmp_path):
    """A cross-run pairing surfaces as a bogus hb_violation; there must be none."""
    out = tmp_path / "trace.json"
    generate_chrome_trace_json(
        _trace_tasks_for_two_runs(), str(out), deps_edges={0x101: [0x102]}, runtime_name=TMR_RUNTIME
    )

    events = json.loads(out.read_text())["traceEvents"]
    violations = [e for e in events if e.get("name") == "hb_violation"]
    assert not violations, f"cross-run pairing produced {len(violations)} bogus happens-before violation(s)"
