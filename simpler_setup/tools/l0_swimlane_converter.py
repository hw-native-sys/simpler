# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Convert runtime-exported L0 event records into a Chrome swimlane trace.

Usage: python -m simpler_setup.tools.l0_swimlane_converter path/to/l0_raw/run_1
The input is ``core_*.events.json``; this tool never reads the GM binary files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LANES = ("SCALAR", "VECTOR", "CUBE", "MTE1", "MTE2", "MTE3", "FIXP")


def convert(run_dir: Path, output: Path) -> int:
    paths = sorted(run_dir.glob("core_*.events.json"))
    if not paths:
        raise ValueError(f"no L0 event records found in {run_dir}")

    events: list[dict[str, object]] = []
    run_epoch: int | None = None
    system_hz: int | None = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "simpler_a5_l0_events_v1":
            raise ValueError(f"unsupported L0 records format: {path}")
        epoch = int(data["run_epoch"])
        sys_hz = int(data["system_counter_frequency_hz"])
        core_hz = int(data["core_frequency_hz"])
        core_id = int(data["physical_core_id"])
        if sys_hz <= 0:
            raise ValueError(f"missing system counter frequency in {path}")
        if core_hz <= 0:
            raise ValueError(f"core frequency unavailable in {path}; cannot convert core cycles to time")
        if run_epoch is not None and (epoch != run_epoch or sys_hz != system_hz):
            raise ValueError(f"mixed run identity or system clock in {run_dir}")
        run_epoch, system_hz = epoch, sys_hz
        if data.get("busy_interval_columns") != [
            "anchor_syscnt",
            "start_cycles",
            "end_cycles",
            "lane",
            "block_id",
            "clip_flags",
        ]:
            raise ValueError(f"unsupported busy interval columns: {path}")
        for anchor, start, end, lane, block_id, flags in data["busy_intervals"]:
            if not 0 <= lane < len(LANES) or end < start:
                raise ValueError(f"invalid busy interval in {path}")
            events.append(
                {
                    "ph": "X",
                    "cat": "a5.l0",
                    "name": f"L0 {LANES[lane]}",
                    "pid": 100000 + core_id,
                    "tid": lane,
                    "ts": anchor * 1_000_000 / sys_hz + start * 1_000_000 / core_hz,
                    "dur": (end - start) * 1_000_000 / core_hz,
                    "args": {"physical_core_id": core_id, "block_id": block_id, "clip_flags": flags},
                }
            )

    events.sort(key=lambda event: (event["ts"], event["pid"], event["tid"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    try:
        trace = {
            "traceEvents": events,
            "metadata": {
                "format": "simpler_a5_l0_swimlane_v1",
                "run_epoch": run_epoch,
                "system_counter_frequency_hz": system_hz,
            },
        }
        temp.write_text(
            json.dumps(trace, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return len(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing core_*.events.json")
    parser.add_argument("-o", "--output", type=Path, help="trace output path (default: run_dir/l0_swimlane_trace.json)")
    args = parser.parse_args()
    output = args.output or args.run_dir / "l0_swimlane_trace.json"
    try:
        count = convert(args.run_dir, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"l0_swimlane_converter: {error}\n")
    print(f"wrote {count} L0 intervals to {output}")


if __name__ == "__main__":
    main()
