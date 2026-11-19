# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG host/device measurements from simpler's existing STRACE parser."""

import argparse
import json
import statistics
from pathlib import Path

from simpler_setup.tools.strace_timing import (
    _round_metrics,
    bucket_by_hid,
    count_record_heads,
    group_invocations,
    invocation_spans,
    parse_drop_summaries,
    parse_spans,
)


def read_invocations(log, warmup=5, rounds=100):
    if warmup < 0 or rounds < 1:
        raise ValueError("warmup must be nonnegative and rounds must be positive")
    lines = Path(log).read_text().splitlines()
    if parse_drop_summaries(lines):
        raise ValueError("STRACE reported dropped records; benchmark is incomplete")
    spans = list(parse_spans(lines))
    if count_record_heads(lines) != len(spans):
        raise ValueError("STRACE contains incomplete records")
    buckets = bucket_by_hid(group_invocations(invocation_spans(spans)))
    candidates = [items for items in buckets.values() if any(item.root() is not None for item in items)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one CSA callable, found {len(candidates)}")
    invocations = sorted(candidates[0], key=lambda item: (item.pid, item.inv))
    if len(invocations) != warmup + rounds:
        raise ValueError(f"Expected {warmup + rounds} rounds, found {len(invocations)}")
    return invocations


def summarize_values(values, name):
    if any(value <= 0 for value in values):
        raise ValueError(f"Nonpositive {name}")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "samples": values,
    }


def summarize(log, warmup=5, rounds=100):
    invocations = read_invocations(log, warmup, rounds)
    required = {"chip.run", "chip.run.runner_run.device_wall"}
    for invocation in invocations:
        names = invocation.by_name()
        if not required <= names.keys():
            raise ValueError(f"Missing HBG device timing markers in invocation {invocation.inv}")
        if not names["chip.run.runner_run.device_wall"].is_device:
            raise ValueError("Device timings must use the device clock domain")
    rows = [_round_metrics(invocation) for invocation in invocations[warmup:]]
    metrics = {
        name: summarize_values([row[column] for row in rows], name)
        for column, name in enumerate(("host_us", "device_us"))
    }
    return {"warmup": warmup, "rounds": rounds, "chip_swimlane_level": 0, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize HBG CSA host and device timings.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log, args.warmup, args.rounds), indent=2))
