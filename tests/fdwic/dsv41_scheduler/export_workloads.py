#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze nine task-graph slices from a completed HBG capture; standard library only."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def export(capture, flavor, output):
    deps_path = capture / "deps.json"
    if not deps_path.exists():
        deps_path = capture / "deps.json.gz"
    names_path = capture / "name_map.json"
    data = read(deps_path)
    names = read(names_path)["callable_id_to_name"]
    tasks = data["tasks"]
    labels = [[names[str(k)] for k in t["kernel_ids"] if k >= 0] for t in tasks]

    def positions(name):
        return [i for i, label in enumerate(labels) if name in label]

    start_name, attention_end, layer_end = (
        ("hc_collapse_norm", "hc_post", "hc_post_2")
        if flavor == "pypto_lib"
        else ("collapse_norm_grid", "expand_grid__2", "expand_grid")
    )
    starts, ends, finishes = positions(start_name), positions(attention_end), positions(layer_end)
    if not len(starts) == len(ends) == len(finishes) == 40:
        raise ValueError("capture must contain all 40 recognizable layer boundaries")
    ranges = {
        name: (starts[layer], ends[layer] + 1)
        for name, layer in [
            ("swa", 0),
            ("c2a_full", 2),
            ("c2a_reuse", 3),
            ("c1a_full", 20),
            ("c1a_reindex", 24),
            ("c1a_reuse", 21),
        ]
    }
    ranges.update(
        moe=(ends[0] + 1, finishes[0] + 1),
        engram=(finishes[0] + 1, starts[1]),
        layers_19_21=(starts[19], finishes[21] + 1),
    )
    summary = {}
    for name, (begin, end) in ranges.items():
        selected = tasks[begin:end]
        ids = {str(task["task_id"]): index for index, task in enumerate(selected)}
        internal = set()
        boundary = set()
        for edge in data["edges"]:
            if str(edge["succ"]) not in ids:
                continue
            if str(edge["pred"]) in ids:
                pair = (ids[str(edge["pred"])], ids[str(edge["succ"])])
                if pair[0] >= pair[1]:
                    raise ValueError(f"non-topological edge {pair}")
                internal.add(pair)
            else:
                boundary.add((str(edge["pred"]), str(edge["succ"])))
        directory = output / flavor / name
        directory.mkdir(parents=True, exist_ok=True)
        rows = ["# source_task_id kernel_names lane_mask block_num deferred"]
        executions = 0
        for index, task in enumerate(selected):
            mask = sum(1 << lane for lane, kid in enumerate(task["kernel_ids"]) if kid >= 0)
            label = "+".join(labels[begin + index]) or "control"
            deferred = int(any(n in {"dispatch_meta_wait", "combine_wait"} for n in labels[begin + index]))
            blocks = int(task["block_num"])
            rows.append(f"{task['task_id']}\t{label}\t{mask}\t{blocks}\t{deferred}")
            executions += blocks * mask.bit_count()
        (directory / "tasks.tsv").write_text("\n".join(rows) + "\n")
        (directory / "edges.tsv").write_text(
            "# predecessor successor (local task indexes)\n" + "".join(f"{p}\t{s}\n" for p, s in sorted(internal))
        )
        summary[name] = {
            "tasks": len(selected),
            "executions": executions,
            "edges": len(internal),
            "incoming_edges": len(boundary),
            "first_source_task": selected[0]["task_id"],
            "last_source_task": selected[-1]["task_id"],
            "task_table_sha256": hashlib.sha256((directory / "tasks.tsv").read_bytes()).hexdigest(),
            "edge_table_sha256": hashlib.sha256((directory / "edges.tsv").read_bytes()).hexdigest(),
        }
    manifest = {
        "source_deps_sha256": hashlib.sha256(deps_path.read_bytes()).hexdigest(),
        "source_names_sha256": hashlib.sha256(names_path.read_bytes()).hexdigest(),
        "source_tasks": len(tasks),
        "source_runtime": data["runtime"],
        "cases": summary,
        "boundary": (
            "Incoming edges are recorded for reference. "
            "Production replay runs the real prefix to produce boundary data."
        ),
        "preserved": [
            "logical task boundaries",
            "block counts",
            "AIC/AIV subtask masks",
            "unique internal dependency edges",
        ],
        "execution": (
            "These tables describe topology only. Use production/export_cases.py with the original C++ "
            "capture and source-built kernels to preserve tensors, communication, MIX and sync-start."
        ),
    }
    (output / flavor / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--flavor", required=True, choices=["pypto_lib", "pypto_lib_dsl"])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "workloads")
    args = parser.parse_args()
    export(args.capture, args.flavor, args.output)
