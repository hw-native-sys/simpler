#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Performance Data to Mermaid Diagram Converter

Converts performance data JSON (.json) to Mermaid flowchart format
for visualizing task dependencies.

Usage:
    python3 perf_to_mermaid.py  # Uses latest perf_swimlane_*.json in outputs/
    python3 perf_to_mermaid.py perf_swimlane_20260210_143526.json
    python3 perf_to_mermaid.py perf_swimlane_20260210_143526.json -o dependency_graph.md
    python3 perf_to_mermaid.py perf_swimlane_20260210_143526.json -k kernel_config.py
    python3 perf_to_mermaid.py perf_swimlane_20260210_143526.json --style compact
"""

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


def read_perf_data(filepath):
    """Read performance data from JSON file.

    Args:
        filepath: Path to input JSON file

    Returns:
        dict: Parsed performance data with keys:
            - version
            - tasks (list)

    Raises:
        ValueError: If JSON format is invalid
    """
    with open(filepath) as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["version", "tasks"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate version
    if data["version"] not in [1, 2]:
        raise ValueError(f"Unsupported version: {data['version']} (expected 1 or 2)")

    return data


def load_kernel_config(config_path):
    """Load kernel configuration from kernel_config.py file.

    Args:
        config_path: Path to kernel_config.py file

    Returns:
        dict: Mapping from func_id (as string) to function name

    Raises:
        ValueError: If file cannot be loaded or KERNELS definition is missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ValueError(f"Kernel config file not found: {config_path}")

    # Load the Python module dynamically.
    # kernel_config.py may import `task_interface` from the project's python/ directory,
    # so ensure it's on sys.path before executing the module.
    python_dir = str(Path(__file__).resolve().parent.parent / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract func_id to name mapping from KERNELS list
    if not hasattr(module, "KERNELS"):
        raise ValueError("kernel_config.py missing KERNELS definition")

    func_id_to_name = {}
    for kernel in module.KERNELS:
        if "func_id" not in kernel:
            print(f"Warning: Kernel entry missing 'func_id', skipping: {kernel}", file=sys.stderr)
            continue

        func_id = kernel["func_id"]

        if "name" not in kernel:
            print(
                f"Warning: Kernel entry for func_id={func_id} missing 'name', will use default naming",
                file=sys.stderr,
            )
            continue

        func_id_to_name[str(func_id)] = kernel["name"]

    return func_id_to_name


def generate_mermaid_flowchart(tasks, func_id_to_name=None, style="detailed", direction="LR", verbose=False):
    """Generate Mermaid flowchart from task data.

    Args:
        tasks: List of task dicts
        func_id_to_name: Optional dict mapping func_id to function name
        style: "detailed" or "compact" - controls node information density
        direction: "TD" (top-down) or "LR" (left-right) - controls flow direction
        verbose: Print progress information

    Returns:
        str: Mermaid flowchart diagram text
    """
    if verbose:
        print(f"Generating Mermaid flowchart (style: {style}, direction: {direction})...")
        print(f"  Tasks: {len(tasks)}")

    lines = []
    lines.append("```mermaid")
    lines.append(f"flowchart {direction}")
    lines.append("")

    # Generate node definitions
    for task in tasks:
        task_id = task["task_id"]
        func_id = task["func_id"]

        # Get function name
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
        else:
            func_name = f"Func_{func_id}"

        # Create node label based on style
        if style == "compact":
            # Compact: just task_id
            label = f"T{task_id}"
        else:
            # Detailed: func_name(task_id) format
            label = f"{func_name}({task_id})"

        # Node definition with label
        lines.append(f'    Task{task_id}["{label}"]')

    lines.append("")

    # Generate edges (dependencies)
    for task in tasks:
        task_id = task["task_id"]
        for succ_task_id in task["fanout"]:
            lines.append(f"    Task{task_id} --> Task{succ_task_id}")

    lines.append("")

    # Generate styling based on core_type using classDef and class
    # Build set of unique core_types
    unique_core_types = set(task["core_type"] for task in tasks)

    # Define color palette for core types
    core_type_colors = {
        "aic": "#66A3FF",  # Medium Blue for AIC
        "aiv": "#FFB366",  # Medium Orange for AIV
    }

    lines.append("    %% Styling by core type")

    # Define style classes
    for core_type in sorted(unique_core_types):
        color = core_type_colors.get(core_type, "#E0E0E0")  # Default gray if unknown
        lines.append(f"    classDef {core_type}Style fill:{color},stroke:#333,stroke-width:2px,color:#000")

    lines.append("")

    # Apply classes to nodes
    for core_type in sorted(unique_core_types):
        # Find all tasks with this core_type
        task_ids = [str(task["task_id"]) for task in tasks if task["core_type"] == core_type]
        task_list = ",".join(f"Task{tid}" for tid in task_ids)
        lines.append(f"    class {task_list} {core_type}Style")

    lines.append("```")

    return "\n".join(lines)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Convert swimlane performance JSON to Mermaid flowchart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                       # latest perf_swimlane_*.json under outputs/
  %(prog)s perf_swimlane_20260210_143526.json   # -> outputs/mermaid_diagram_20260210_143526.md
  %(prog)s perf_swimlane_20260210_143526.json -o custom_diagram.md
  %(prog)s perf_swimlane_20260210_143526.json -k kernel_config.py
  %(prog)s perf_swimlane_20260210_143526.json --style compact
  %(prog)s perf_swimlane_20260210_143526.json -v

View the Mermaid diagram:
  1. GitHub/GitLab Markdown preview
  2. https://mermaid.live/
  3. Editors with a Mermaid extension (e.g. VS Code)
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSON (.json). If omitted, use the newest perf_swimlane_*.json under outputs/",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output Markdown path (default: outputs/mermaid_diagram_<timestamp>.md)",
    )
    parser.add_argument(
        "-k",
        "--kernel-config",
        help="Path to kernel_config.py for func_id -> name mapping",
    )
    parser.add_argument(
        "--style",
        choices=["detailed", "compact"],
        default="detailed",
        help="Node detail: detailed (full) or compact (minimal)",
    )
    parser.add_argument(
        "--direction",
        choices=["TD", "LR"],
        default="TD",
        help="Flowchart direction: TD (top-down, default) or LR (left-right)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser


def _resolve_input_path(args):
    """Resolve input path, auto-selecting latest perf_swimlane_*.json if not specified."""
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input file not found: {input_path}", file=sys.stderr)
            return None
        return input_path

    outputs_dir = Path(__file__).parent.parent / "outputs"
    json_files = list(outputs_dir.glob("perf_swimlane_*.json"))
    if not json_files:
        print(f"Error: no perf_swimlane_*.json under {outputs_dir}", file=sys.stderr)
        print("Specify an input file or add .json files under outputs/", file=sys.stderr)
        return None

    input_path = max(json_files, key=lambda p: p.stat().st_mtime)
    if args.verbose:
        print(f"Auto-selected latest file: {input_path.name}")
    return input_path


def _resolve_output_path(args, input_path):
    """Determine output path from args or derive from input filename."""
    if args.output:
        return Path(args.output)

    input_stem = input_path.stem
    if input_stem.startswith("perf_swimlane_"):
        timestamp_part = input_stem[len("perf_swimlane_") :]
    else:
        timestamp_part = datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir / f"mermaid_diagram_{timestamp_part}.md"


def main():
    args = _build_parser().parse_args()

    input_path = _resolve_input_path(args)
    if input_path is None:
        return 1

    try:
        if args.verbose:
            print(f"Reading performance data: {input_path}")
        data = read_perf_data(input_path)
        if args.verbose:
            print("\n=== Performance data ===")
            print(f"  Version: {data['version']}")
            print(f"  Tasks: {len(data['tasks'])}")
            print()

        func_names = {}
        if args.kernel_config:
            if args.verbose:
                print(f"Loading kernel config: {args.kernel_config}")
            func_names = load_kernel_config(args.kernel_config)
            if args.verbose:
                print(f"  Loaded {len(func_names)} func_id name mappings")
                for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                    print(f"    func_id={func_id}: {name}")
                print()

        output_path = _resolve_output_path(args, input_path)

        mermaid_text = generate_mermaid_flowchart(
            data["tasks"],
            func_names,
            args.style,
            args.direction,
            args.verbose,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Task Dependency Graph\n\n")
            f.write(f"Generated from: `{input_path.name}`\n\n")
            f.write(mermaid_text)

        print("\nConversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
