#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: E501
"""Read PMU task samples after diagnostic flush, including recorded-window quality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_PREFIX = ("thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles")


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or tuple(columns[:6]) != _PREFIX or columns[-2:] != ["event_type", "run_epoch"]:
            raise ValueError("invalid PMU CSV columns")
        if len(set(columns)) != len(columns):
            raise ValueError("duplicate PMU CSV columns")
        rows = list(reader)
    for row in rows:
        if set(row) != set(columns) or any(value is None for value in row.values()):
            raise ValueError("PMU row does not match its columns")
        try:
            for key, value in row.items():
                parsed = int(value, 16) if key == "task_id" and value.startswith("0x") else int(value)
                if parsed < 0:
                    raise ValueError
        except ValueError as error:
            raise ValueError("invalid PMU numeric row") from error
    return columns, rows


def read_pmu_samples(path: str | Path, *, recorded: bool = False) -> dict:
    """Read settled rows; recorded results require their collection descriptor.

    Finish diagnostic flush before reading. Repeated task IDs are samples, not
    duplicate rows. Successful publication does not prove device completeness.
    """
    path = Path(path)
    columns, rows = _read_rows(path)
    descriptor = path.with_name("pmu.collection.json")
    if not recorded:
        if descriptor.exists():
            raise ValueError("recorded PMU output requires recorded=True")
        return {"columns": columns, "records": rows, "collection": None}
    with descriptor.open(encoding="utf-8") as source:
        collection = json.load(source)
    if not isinstance(collection, dict):
        raise ValueError("invalid PMU collection descriptor")
    if collection.get("recorded_window") is not True or collection.get("counts_unknown") is not True:
        raise ValueError("recorded PMU output must declare unknown aggregate device counts")
    if any(
        collection.get(key, 0) is not None
        for key in ("device_total_records", "device_dropped_records", "device_fatal", "transport_retired_buffers")
    ):
        raise ValueError("recorded PMU output must not invent device or per-window transport counts")
    if collection.get("csv_file") != path.name or collection.get("columns") != columns:
        raise ValueError("PMU collection descriptor does not match the CSV")
    epoch = _count(collection.get("run_epoch"), "run_epoch")
    event = _count(collection.get("event_type"), "event_type")
    if not epoch or event not in (1, 2, 4, 5, 6, 7, 8):
        raise ValueError("invalid PMU window selection")
    if _count(collection.get("written_records"), "written_records") != len(rows):
        raise ValueError("PMU collection row count does not match the CSV")
    if any(int(row["run_epoch"]) != epoch or int(row["event_type"]) != event for row in rows):
        raise ValueError("PMU rows do not match the frozen window selection")
    for name in ("collector_transport_retired_buffers", "collector_unknown_epoch_buffers", "clamped_buffers"):
        if _count(collection.get(name), name) != 0:
            raise ValueError(f"PMU collection has observed loss or unattributed traffic: {name}")
    if collection.get("collection_verdict") != "aggregate_counts_unknown":
        raise ValueError("PMU collection was not successfully published")
    if collection.get("pmu_protection") not in ("held", "occupied", "refused", "absent"):
        raise ValueError("PMU collection has no driver protection observation")
    return {"columns": columns, "records": rows, "collection": collection}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--recorded", action="store_true", help="require and verify recorded-window quality metadata")
    args = parser.parse_args(argv)
    report = read_pmu_samples(args.csv, recorded=args.recorded)
    print(json.dumps({"records": len(report["records"]), "collection": report["collection"]}, indent=2))


if __name__ == "__main__":
    main()
