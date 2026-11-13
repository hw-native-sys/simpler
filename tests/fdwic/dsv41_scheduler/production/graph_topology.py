#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Extract a verified native Graph kernel DAG from a structural L4 capture."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_graph import audit

from simpler_setup.tools.hbg.native_graph import export_kernel_dependencies


def provenance(bundle):
    config = json.loads((bundle / "case.json").read_text())["graph_benchmark"]
    return {
        "source_level": 4,
        "repetitions": config["repetitions"],
        "callable_sha256": hashlib.sha256((bundle / "callable.bin").read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((bundle / "plan-rank0.bin").read_bytes()).hexdigest(),
    }


def project(bundle, run):
    case = json.loads((bundle / "case.json").read_text())
    if case["graph_benchmark"]["mode"] != "graph" or case["device_count"] != 1:
        raise ValueError("The current lower-bound model accepts single-device Graphs only")
    suite = Path(__file__).resolve().parents[1]
    audit(bundle, run, suite / "workloads" / case["flavor"] / case["case"] / "tasks.tsv")
    data = json.loads((run / "chip_swimlane_records.json").read_text())
    if data["chip_swimlane_level"] != 4:
        raise ValueError("Native Graph topology requires the structural L4 capture")
    section = data["aicore_protocol"]
    source = section["graph_source"]
    if (
        len(source["definitions"]) != 1
        or sum(task["kind"] == 1 for task in source["tasks"]) != case["graph_benchmark"]["repetitions"]
    ):
        raise ValueError("Graph definition or instance count changed")
    ids = {row["task_id"] for row in data["aicore_execution_identities"]["records"] if row["executed"]}
    result = export_kernel_dependencies(section, ids)
    result["baseline_provenance"] = provenance(bundle)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(project(args.bundle, args.run), indent=2) + "\n")
