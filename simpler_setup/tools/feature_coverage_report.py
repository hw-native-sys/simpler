# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reverify CI receipts and publish only verified feature counts."""

from __future__ import annotations

import argparse
import html
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from simpler_setup.test_environments import verify_result_files


def expected_partitions(environment, platform, manual):
    if environment == "torch_npu":
        return [("results", "all", manual, None)]
    partitions = [("main", "main" if platform == "a2a3" else "all", manual, 4)]
    if platform == "a2a3" and manual != "only":
        partitions += [("sdma", "sdma", "exclude", None), ("sdma_fault", "sdma_fault", "exclude", None)]
    return partitions


def verify_partition(plan_path, environment, platform, manual, partition, exclude_level):
    # Never consume a cached green summary after a missing or mismatched plan.
    coverage_path = plan_path.with_suffix(".coverage.json")
    coverage_path.unlink(missing_ok=True)
    plan = json.loads(plan_path.read_text())
    scope = plan["coverage"]["scope"]
    expected = {
        "environment": environment,
        "platform": platform,
        "manual": manual,
        "partition": partition,
        "runtime": None,
        "level": None,
        "exclude_level": exclude_level,
    }
    if scope != expected or plan["environment"] != environment or plan["platform"] != platform:
        raise ValueError(f"Result scope differs from the requested CI partition: {scope}")
    if environment == "default":
        manifest = json.loads(plan_path.with_suffix(".results.json").read_text())
        verify_result_files(
            plan_path, [Path(path) for path in manifest["files"]], dispatcher=True, retries=manifest["retries"]
        )
    else:
        verify_result_files(plan_path, [plan_path.with_suffix("").with_suffix(".xml")])
    return json.loads(coverage_path.read_text())


def _cell(value):
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def report(root, environment, platform, manual):
    lines = [f"## Feature coverage: {environment} / {platform}", ""]
    failed = False
    for name, partition, selection, exclude_level in expected_partitions(environment, platform, manual):
        lines += [f"### {name}", ""]
        try:
            summary = verify_partition(
                root / f"{name}.plan.json", environment, platform, selection, partition, exclude_level
            )
        except (ValueError, OSError, KeyError, TypeError, ET.ParseError) as error:
            lines += [f"**Not verified:** {_cell(error)}", ""]
            failed = True
            continue
        lines += [
            f"Verified receipts for source `{summary['source']}`; manual selection: `{selection}`.",
            "",
            "| Product / runtime / mode | Selected | Body entered | Passed | Skipped |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for group, counts in sorted(summary["groups"].items()):
            values = " | ".join(str(counts[key]) for key in ("selected", "executed", "passed", "skipped"))
            lines.append(f"| {_cell(group)} | {values} |")
        lines += ["", f"Attempts: {summary['attempts']}; recovered cases: {summary['recovered']}.", ""]
        for key, label in (("not_in_scope", "Outside this invocation"), ("not_delivered", "Not delivered")):
            if summary[key]:
                lines += [f"{label}: " + ", ".join(f"`{_cell(group)}`" for group in summary[key]) + ".", ""]
    lines += ["Body entry counts do not establish device completion independently of the test assertions.", ""]
    root.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(lines)
    (root / "coverage-summary.md").write_text(rendered, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as destination:
            destination.write(rendered)
    print(rendered)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--environment", choices=("default", "torch_npu"), required=True)
    parser.add_argument("--platform", choices=("a2a3", "a5"), required=True)
    parser.add_argument("--manual", choices=("exclude", "include", "only"), required=True)
    args = parser.parse_args()
    raise SystemExit(report(args.root, args.environment, args.platform, args.manual))


if __name__ == "__main__":
    main()
