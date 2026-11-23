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
The input is ``core_*.events.json``. Use ``--decoder-library`` with an L0-enabled
Host library to export ordered ``core_*.bin`` records using ``manifest.json`` first.
Physical GM snapshots (``core_*.gm.bin``) are not ordered record streams.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from collections.abc import Iterable
from pathlib import Path

LANES = ("SCALAR", "VECTOR", "CUBE", "MTE1", "MTE2", "MTE3", "FIXP")


def _all_complete(states: Iterable[bool | None]) -> bool | None:
    values = tuple(states)
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def convert(run_dir: Path, output: Path) -> int:
    paths = sorted(run_dir.glob("core_*.events.json"))
    if not paths:
        raise ValueError(f"no L0 event records found in {run_dir}")

    events: list[dict[str, object]] = []
    run_epoch: int | None = None
    system_hz: int | None = None
    decode_results: list[bool | None] = []
    exported_cores: set[int] = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "simpler_a5_l0_events_v1":
            raise ValueError(f"unsupported L0 records format: {path}")
        epoch = int(data["run_epoch"])
        sys_hz = int(data["system_counter_frequency_hz"])
        core_hz = int(data["core_frequency_hz"])
        core_id = int(data["physical_core_id"])
        exported_cores.add(core_id)
        decode_results.append(data.get("stats", {}).get("structurally_complete"))
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
    decode_complete = _all_complete(decode_results)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest and int(manifest["run_epoch"]) != run_epoch:
        raise ValueError(f"manifest and events belong to different runs in {run_dir}")
    expected_cores = {int(core["physical_core_id"]) for core in manifest.get("cores", []) if int(core["raw_bytes"]) > 0}
    missing_cores = sorted(expected_cores - exported_cores)
    unexpected_cores = sorted(exported_cores - expected_cores) if manifest else []
    statuses = (manifest.get("capture_complete"), manifest.get("raw_transport_complete"), decode_complete)
    capture_complete = False if missing_cores or unexpected_cores else _all_complete(statuses)
    metadata = {
        "format": "simpler_a5_l0_swimlane_v1",
        "run_epoch": run_epoch,
        "system_counter_frequency_hz": system_hz,
        "capture_complete": capture_complete,
        "raw_transport_complete": manifest.get("raw_transport_complete"),
        "decode_complete": decode_complete,
        "missing_event_cores": missing_cores,
        "unexpected_event_cores": unexpected_cores,
    }
    trace_events = list(events)
    if capture_complete is not True:
        trace_events.insert(
            0,
            {
                "ph": "i",
                "s": "g",
                "cat": "a5.l0.status",
                "name": "L0 capture incomplete" if capture_complete is False else "L0 capture status unknown",
                "pid": 0,
                "tid": 0,
                "ts": events[0]["ts"] if events else 0,
                "args": metadata,
            },
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    try:
        trace = {
            "traceEvents": trace_events,
            "metadata": metadata,
        }
        temp.write_text(
            json.dumps(trace, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return len(events)


def _export_raw_records(run_dir: Path, decoder_library: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cores = {int(core["physical_core_id"]): int(core["core_type"]) for core in manifest["cores"]}
    if not cores or min(cores) < 0:
        raise ValueError("manifest has no valid physical core IDs")
    core_types = (ctypes.c_uint32 * (max(cores) + 1))()
    for core_id, core_type in cores.items():
        core_types[core_id] = core_type
    library = ctypes.CDLL(str(decoder_library.resolve()))
    export = library.simpler_export_l0_event_records
    export.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    export.restype = ctypes.c_bool
    if not export(
        os.fsencode(run_dir.resolve()),
        int(manifest["run_epoch"]),
        int(manifest["aic_frequency_hz"]),
        int(manifest["aiv_frequency_hz"]),
        core_types,
        len(core_types),
    ):
        raise ValueError(f"raw L0 record export failed in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing core_*.events.json")
    parser.add_argument("-o", "--output", type=Path, help="trace output path (default: run_dir/l0_swimlane_trace.json)")
    parser.add_argument("--decoder-library", type=Path, help="L0-enabled Host library for offline raw record export")
    args = parser.parse_args()
    output = args.output or args.run_dir / "l0_swimlane_trace.json"
    try:
        if args.decoder_library is not None:
            _export_raw_records(args.run_dir, args.decoder_library)
        count = convert(args.run_dir, output)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"l0_swimlane_converter: {error}\n")
    print(f"wrote {count} L0 intervals to {output}")


if __name__ == "__main__":
    main()
