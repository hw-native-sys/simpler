#!/usr/bin/env python3
"""
Generate Perfetto swimlane JSON with dedicated lanes for:
  - Orchestrator (1 lane)
  - 3 Scheduler threads (3 lanes)
  - Each AIV core (individual lanes)
  - Each AIC core (individual lanes)

Usage:
    python3 tools/generate_full_swimlane.py outputs/perf_swimlane_XXXX.json
    python3 tools/generate_full_swimlane.py outputs/perf_swimlane_XXXX.json -o outputs/full_swimlane.json
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict


FUNC_ID_TO_NAME = {0: "QK", 1: "SF", 2: "PV", 3: "UP", 4: "AIC_HUB", 5: "AIV_HUB"}

PID_ORCHESTRATOR = 1
PID_SCHEDULER = 2
PID_AIC = 3
PID_AIV = 4


def assign_cores_to_threads(aic_ids, aiv_ids, num_threads=3):
    """Reproduce the C++ assign_cores_to_threads logic."""
    aic_sorted = sorted(aic_ids)
    aiv_sorted = sorted(aiv_ids)
    aic_per = len(aic_sorted) // num_threads
    aiv_per = len(aiv_sorted) // num_threads

    core_to_thread = {}
    for t in range(num_threads):
        for c in aic_sorted[t * aic_per:(t + 1) * aic_per]:
            core_to_thread[c] = t
        for c in aiv_sorted[t * aiv_per:(t + 1) * aiv_per]:
            core_to_thread[c] = t
    # Remainder cores go to last thread
    for c in aic_sorted[num_threads * aic_per:]:
        core_to_thread[c] = num_threads - 1
    for c in aiv_sorted[num_threads * aiv_per:]:
        core_to_thread[c] = num_threads - 1
    return core_to_thread


def generate_full_swimlane(tasks, output_path):
    events = []

    # Classify cores
    aic_ids = sorted({t["core_id"] for t in tasks if t["core_type"] == "aic"})
    aiv_ids = sorted({t["core_id"] for t in tasks if t["core_type"] == "aiv"})
    core_to_thread = assign_cores_to_threads(aic_ids, aiv_ids)

    # ── Process metadata ──
    for pid, name in [
        (PID_ORCHESTRATOR, "Orchestrator (AICPU Thread 3)"),
        (PID_SCHEDULER, "Scheduler Threads (AICPU 0-2)"),
        (PID_AIC, "AIC Cores"),
        (PID_AIV, "AIV Cores"),
    ]:
        events.append({"args": {"name": name}, "cat": "__metadata",
                        "name": "process_name", "ph": "M", "pid": pid})

    # ── Thread metadata ──
    # Orchestrator: single lane
    events.append({"args": {"name": "Orchestrator"}, "cat": "__metadata",
                    "name": "thread_name", "ph": "M", "pid": PID_ORCHESTRATOR, "tid": 0})

    # Scheduler: 3 lanes
    for t in range(3):
        events.append({"args": {"name": f"Scheduler {t}"}, "cat": "__metadata",
                        "name": "thread_name", "ph": "M", "pid": PID_SCHEDULER, "tid": t})

    # AIC cores
    for idx, cid in enumerate(aic_ids):
        events.append({"args": {"name": f"AIC_{cid}"}, "cat": "__metadata",
                        "name": "thread_name", "ph": "M", "pid": PID_AIC, "tid": cid})

    # AIV cores
    for idx, cid in enumerate(aiv_ids):
        events.append({"args": {"name": f"AIV_{cid}"}, "cat": "__metadata",
                        "name": "thread_name", "ph": "M", "pid": PID_AIV, "tid": cid})

    # Sort tasks by task_id for orchestrator ordering
    tasks_by_id = sorted(tasks, key=lambda t: t["task_id"])

    # Build task map for flow events
    task_map = {t["task_id"]: t for t in tasks}

    # ── Orchestrator lane ──
    # Estimate submission time: the orchestrator submits tasks sequentially.
    # Approximate submit_start as slightly before the earliest of:
    #   dispatch_time of this task or the previous task's submit_end.
    # For the first task, use dispatch_time - small_delta.
    orch_events = []
    prev_submit_end = 0
    min_dispatch = min(t.get("dispatch_time_us", 1e9) for t in tasks if t.get("dispatch_time_us", 0) > 0)

    for task in tasks_by_id:
        tid = task["task_id"]
        func_name = FUNC_ID_TO_NAME.get(task["func_id"], f"F{task['func_id']}")
        disp = task.get("dispatch_time_us", 0)

        # Heuristic: orchestrator submit window ≈ 9.5us/task (from orch profiling avg)
        orch_dur = 5.0  # estimated us per submit
        if prev_submit_end == 0:
            submit_start = min_dispatch - 50  # first task: 50us before first dispatch
        else:
            submit_start = prev_submit_end + 0.1

        submit_end = submit_start + orch_dur
        prev_submit_end = submit_end

        events.append({
            "name": f"{func_name}({tid})",
            "cat": "orchestrator",
            "ph": "X",
            "pid": PID_ORCHESTRATOR,
            "tid": 0,
            "ts": submit_start,
            "dur": orch_dur,
            "args": {"task_id": tid, "func": func_name, "core_id": task["core_id"]}
        })

    # ── Scheduler lanes ──
    # Group tasks by scheduler thread (heuristic: core ownership)
    for task in tasks:
        disp = task.get("dispatch_time_us", 0)
        fin = task.get("finish_time_us", 0)
        if disp <= 0 or fin <= 0:
            continue

        core_id = task["core_id"]
        sched_tid = core_to_thread.get(core_id, 0)
        func_name = FUNC_ID_TO_NAME.get(task["func_id"], f"F{task['func_id']}")
        task_id = task["task_id"]

        events.append({
            "name": f"{func_name}({task_id})",
            "cat": "scheduler",
            "ph": "X",
            "pid": PID_SCHEDULER,
            "tid": sched_tid,
            "ts": disp,
            "dur": fin - disp,
            "args": {
                "task_id": task_id,
                "core_id": core_id,
                "dispatch_us": disp,
                "finish_us": fin,
                "head_oh": task["start_time_us"] - disp,
                "exec": task["duration_us"],
                "tail_oh": fin - task["end_time_us"],
            }
        })

    # ── AIC / AIV core lanes ──
    event_id = 0
    task_to_eid = {}
    for task in tasks:
        core_id = task["core_id"]
        core_type = task["core_type"]
        pid = PID_AIC if core_type == "aic" else PID_AIV
        func_name = FUNC_ID_TO_NAME.get(task["func_id"], f"F{task['func_id']}")
        task_id = task["task_id"]
        ts = task["start_time_us"]
        dur = task["duration_us"]

        events.append({
            "name": f"{func_name}({task_id})",
            "cat": "kernel",
            "ph": "X",
            "id": event_id,
            "pid": pid,
            "tid": core_id,
            "ts": ts,
            "dur": dur,
            "args": {
                "task_id": task_id,
                "func_id": task["func_id"],
                "core_id": core_id,
                "duration_us": dur,
            }
        })
        task_to_eid[task_id] = event_id
        event_id += 1

    # ── Flow events (dependencies between core lanes) ──
    flow_id = 0
    for task in tasks:
        src_pid = PID_AIC if task["core_type"] == "aic" else PID_AIV
        src_tid = task["core_id"]
        src_ts_end = task["end_time_us"]

        for succ_id in task.get("fanout", []):
            succ = task_map.get(succ_id)
            if not succ:
                continue
            dst_pid = PID_AIC if succ["core_type"] == "aic" else PID_AIV
            dst_tid = succ["core_id"]
            dst_ts = succ["start_time_us"]

            events.append({"cat": "flow", "id": flow_id, "name": "dep",
                           "ph": "s", "pid": src_pid, "tid": src_tid,
                           "ts": src_ts_end - 0.01})
            events.append({"cat": "flow", "id": flow_id, "name": "dep",
                           "ph": "f", "pid": dst_pid, "tid": dst_tid,
                           "ts": dst_ts, "bp": "e"})
            flow_id += 1

    with open(output_path, "w") as f:
        json.dump({"traceEvents": events}, f, indent=2)

    print(f"Swimlane written: {output_path}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  AIC cores: {len(aic_ids)} ({aic_ids[0]}..{aic_ids[-1]})")
    print(f"  AIV cores: {len(aiv_ids)} ({aiv_ids[0]}..{aiv_ids[-1]})")
    print(f"  Events: {len(events)}")
    print(f"\nOpen https://ui.perfetto.dev/ and load {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate full swimlane Perfetto JSON")
    parser.add_argument("input", nargs="?", help="perf_swimlane_*.json file")
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args()

    if args.input is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
        candidates = sorted(outputs_dir.glob("perf_swimlane_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No perf_swimlane_*.json found in outputs/", file=sys.stderr)
            return 1
        input_path = candidates[-1]
        print(f"Auto-selected: {input_path.name}")
    else:
        input_path = Path(args.input)

    with open(input_path) as f:
        data = json.load(f)

    output_path = args.output or str(
        input_path.parent / f"perfetto_full_swimlane_{input_path.stem.split('_', 2)[-1]}.json"
    )

    generate_full_swimlane(data["tasks"], output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
