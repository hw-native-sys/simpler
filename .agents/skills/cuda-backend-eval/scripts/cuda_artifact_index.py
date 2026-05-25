#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Index local CUDA backend benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_SMOKE_LABEL_RE = re.compile(r"^- Label: `([^`]+)`$", re.MULTILINE)


def _sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def _sorted_unique(values: set[Any]) -> list[Any]:
    return sorted(values, key=_sort_key)


def _read_artifact(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads((path / "cuda-benchmark.json").read_text())
    metadata = payload.get("metadata", {})
    results = payload.get("results", [])
    return {
        "path": str(path.relative_to(root)),
        "kind": "benchmark",
        "label": str(metadata.get("label", "unknown")),
        "machine": str(metadata.get("machine", "unknown")),
        "git_commit": str(metadata.get("git_commit", "unknown")),
        "result_count": len(results),
        "baselines": _sorted_unique({row.get("baseline", "unknown") for row in results}),
        "sizes": _sorted_unique({row.get("n", "unknown") for row in results}),
        "has_markdown": (path / "cuda-benchmark.md").exists(),
        "has_svg": (path / "cuda-benchmark.svg").exists(),
        "has_ratio_svg": (path / "cuda-benchmark-ratios.svg").exists(),
    }


def _smoke_label(report: str, fallback: str) -> str:
    match = _SMOKE_LABEL_RE.search(report)
    if match:
        return match.group(1)
    return fallback


def _read_smoke_artifact(path: Path, root: Path) -> dict[str, Any]:
    report = (path / "cuda-smoke-report.md").read_text()
    json_paths = sorted(path.glob("*.json"))
    payloads = [json.loads(json_path.read_text()) for json_path in json_paths]
    if len(payloads) > 1:
        machine = "combined"
    elif payloads:
        machine = str(payloads[0].get("machine", "unknown"))
    else:
        machine = "unknown"
    return {
        "path": str(path.relative_to(root)),
        "kind": "smoke",
        "label": _smoke_label(report, path.name),
        "machine": machine,
        "git_commit": str(payloads[0].get("git_commit", "unknown")) if payloads else "unknown",
        "result_count": len(payloads),
        "baselines": _sorted_unique({payload.get("runtime", "unknown") for payload in payloads}),
        "sizes": _sorted_unique({payload.get("n", "unknown") for payload in payloads}),
        "has_markdown": True,
        "has_svg": (path / "cuda-smoke-report.svg").exists(),
        "has_ratio_svg": False,
    }


def scan_artifacts(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.exists():
        return []
    benchmark_dirs = {path.parent for path in root.rglob("cuda-benchmark.json") if path.is_file()}
    smoke_dirs = {
        path.parent
        for path in root.rglob("cuda-smoke-report.md")
        if path.is_file() and path.parent not in benchmark_dirs
    }
    entries = [_read_artifact(path, root) for path in benchmark_dirs]
    entries.extend(_read_smoke_artifact(path, root) for path in smoke_dirs)
    return sorted(entries, key=lambda entry: entry["path"])


def _format_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _checkmark(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# CUDA Backend Artifact Index",
        "",
        "Generated from local CUDA benchmark and smoke report artifacts. Raw",
        "artifacts remain under `tmp/` and are intentionally not committed.",
        "",
        "| Path | Kind | Label | Machine | Commit | Results | Sizes | Baselines | Markdown | SVG | ratio SVG |",
        "| ---- | ---- | ----- | ------- | ------ | ------- | ----- | --------- | -------- | --- | --------- |",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['path']} | {entry['kind']} | {entry['label']} | {entry['machine']} | "
            f"{entry['git_commit']} | {entry['result_count']} | "
            f"{_format_list(entry['sizes'])} | {_format_list(entry['baselines'])} | "
            f"{_checkmark(entry['has_markdown'])} | {_checkmark(entry['has_svg'])} | "
            f"{_checkmark(entry['has_ratio_svg'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_index(root: Path, output: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = output or root / "index.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(scan_artifacts(root)))
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("tmp/cuda-backend"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = write_index(args.root, args.output)
    print(output)


if __name__ == "__main__":
    main()
