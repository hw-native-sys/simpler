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
Missing capture data retains the available intervals, prints a warning, and marks
the trace metadata incomplete with per-core byte counts.
A successful raw export removes cached event files not produced by that export.
Event-only archives remain usable without ``--decoder-library``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

LANES = ("SCALAR", "VECTOR", "CUBE", "MTE1", "MTE2", "MTE3", "FIXP")
MAX_PHYSICAL_CORES = 108  # A5 PLATFORM_MAX_CORES bounds the native core-type array.


def _all_complete(states: Iterable[bool | None]) -> bool | None:
    values = tuple(states)
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def convert(run_dir: Path, output: Path, *, record_paths: Iterable[Path] | None = None) -> int:
    paths = sorted(run_dir.glob("core_*.events.json") if record_paths is None else record_paths)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not paths and not manifest:
        raise ValueError(f"no L0 event records found in {run_dir}")

    events: list[dict[str, object]] = []
    run_epoch: int | None = None
    system_hz: int | None = None
    decode_results: list[bool | None] = []
    exported_cores: set[int] = set()
    decoded_sizes: dict[int, int | None] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "simpler_a5_l0_events_v1":
            raise ValueError(f"unsupported L0 records format: {path}")
        epoch = int(data["run_epoch"])
        sys_hz = int(data["system_counter_frequency_hz"])
        core_hz = int(data["core_frequency_hz"])
        core_id = int(data["physical_core_id"])
        if core_id in exported_cores:
            raise ValueError(f"duplicate physical core ID {core_id} in {path}")
        exported_cores.add(core_id)
        decoded_sizes[core_id] = int(data["raw_bytes"]) if "raw_bytes" in data else None
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
    decode_complete = _all_complete(decode_results) if decode_results else None
    if not paths:
        run_epoch = int(manifest["run_epoch"])
        system_hz = int(manifest["system_counter_frequency_hz"])
        if system_hz <= 0:
            raise ValueError(f"missing system counter frequency in {manifest_path}")
    if manifest and int(manifest["run_epoch"]) != run_epoch:
        raise ValueError(f"manifest and events belong to different runs in {run_dir}")
    expected_sizes = {
        int(core["physical_core_id"]): int(core["raw_bytes"])
        for core in manifest.get("cores", [])
        if int(core["raw_bytes"]) > 0
    }
    expected_cores = set(expected_sizes)
    missing_cores = sorted(expected_cores - exported_cores)
    unexpected_cores = sorted(exported_cores - expected_cores) if manifest else []
    mismatches = [
        {"physical_core_id": core, "expected_raw_bytes": expected_sizes[core], "decoded_raw_bytes": decoded_sizes[core]}
        for core in sorted(expected_cores & exported_cores)
        if decoded_sizes[core] is not None and decoded_sizes[core] != expected_sizes[core]
    ]
    unverified_sizes = sorted(core for core in expected_cores & exported_cores if decoded_sizes[core] is None)
    data_missing = bool(missing_cores) or any(
        mismatch["decoded_raw_bytes"] < mismatch["expected_raw_bytes"] for mismatch in mismatches
    )
    input_complete = False if missing_cores or unexpected_cores or mismatches else None if unverified_sizes else True
    transport_complete = (
        False if mismatches or (record_paths is not None and missing_cores) else manifest.get("raw_transport_complete")
    )
    capture_complete = _all_complete(
        (manifest.get("capture_complete"), transport_complete, decode_complete, input_complete)
    )
    metadata = {
        "format": "simpler_a5_l0_swimlane_v1",
        "run_epoch": run_epoch,
        "system_counter_frequency_hz": system_hz,
        "capture_complete": capture_complete,
        "raw_transport_complete": transport_complete,
        "decode_complete": decode_complete,
        "missing_event_cores": missing_cores,
        "unexpected_event_cores": unexpected_cores,
        "data_missing": data_missing,
        "raw_byte_mismatches": mismatches,
        "unverified_byte_count_cores": unverified_sizes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    )
    temp_path = Path(temp.name)
    try:
        trace = {
            "traceEvents": events,
            "metadata": metadata,
        }
        with temp:
            json.dump(trace, temp, separators=(",", ":"))
            temp.write("\n")
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)
    if capture_complete is False:
        details = [f"missing event cores: {missing_cores}"] if missing_cores else []
        details.extend(
            f"core {entry['physical_core_id']}: expected {entry['expected_raw_bytes']} raw bytes, "
            f"decoded {entry['decoded_raw_bytes']}"
            for entry in mismatches
        )
        warning = "L0 data missing; capture incomplete" if data_missing else "L0 capture incomplete"
        suffix = f" ({'; '.join(details)})" if details else ""
        print(f"l0_swimlane_converter: warning: {warning}{suffix}", file=sys.stderr)
    return len(events)


def _export_raw_records(run_dir: Path, decoder_library: Path) -> list[Path]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cores = {}
    for core in manifest["cores"]:
        core_id = core["physical_core_id"]
        if type(core_id) is not int or not 0 <= core_id < MAX_PHYSICAL_CORES:
            raise ValueError(f"physical core ID out of range: {core_id}")
        if core_id in cores:
            raise ValueError(f"duplicate physical core ID {core_id} in manifest")
        core_type = core["core_type"]
        if type(core_type) is not int or core_type not in (0, 1, 2):
            raise ValueError(f"invalid core type for physical core {core_id}")
        cores[core_id] = core_type
    if not cores:
        raise ValueError("manifest has no valid physical core IDs")
    frequencies = []
    for field in ("aic_frequency_hz", "aiv_frequency_hz"):
        value = manifest[field]
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError(f"{field} out of range for uint64: {value}")
        frequencies.append(value)
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
    paths = []
    with TemporaryDirectory(prefix=".l0-export-", dir=run_dir) as scratch:
        staging = Path(scratch)
        for core in cores:
            raw_path = run_dir / f"core_{core}.bin"
            if raw_path.is_file():
                (staging / raw_path.name).symlink_to(raw_path.resolve())
        if not export(
            os.fsencode(staging.resolve()),
            int(manifest["run_epoch"]),
            *frequencies,
            core_types,
            len(core_types),
        ):
            raise ValueError(f"raw L0 record export failed in {run_dir}")
        for source in sorted(staging.glob("core_*.events.json")):
            target = run_dir / source.name
            os.replace(source, target)
            paths.append(target)
    current_paths = set(paths)
    for path in run_dir.glob("core_*.events.json"):
        if path not in current_paths:
            path.unlink(missing_ok=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing core_*.events.json")
    parser.add_argument("-o", "--output", type=Path, help="trace output path (default: run_dir/l0_swimlane_trace.json)")
    parser.add_argument("--decoder-library", type=Path, help="L0-enabled Host library for offline raw record export")
    args = parser.parse_args()
    output = args.output or args.run_dir / "l0_swimlane_trace.json"
    try:
        record_paths = (
            _export_raw_records(args.run_dir, args.decoder_library) if args.decoder_library is not None else None
        )
        count = convert(args.run_dir, output, record_paths=record_paths)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"l0_swimlane_converter: {error}\n")
    print(f"wrote {count} L0 intervals to {output}")


if __name__ == "__main__":
    main()
