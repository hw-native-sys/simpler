#!/usr/bin/env python3
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

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import importlib.util


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
    with open(filepath, "r") as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["version", "tasks"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate version
    if data["version"] != 1:
        raise ValueError(f"Unsupported version: {data['version']} (expected 1)")

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

    # Load the Python module dynamically
    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract func_id to name mapping from KERNELS list
    if not hasattr(module, "KERNELS"):
        raise ValueError(f"kernel_config.py missing KERNELS definition")

    func_id_to_name = {}
    for kernel in module.KERNELS:
        if "func_id" not in kernel:
            print(
                f"Warning: Kernel entry missing 'func_id', skipping: {kernel}",
                file=sys.stderr,
            )
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


def generate_mermaid_flowchart(
    tasks, func_id_to_name=None, style="detailed", direction="LR", verbose=False
):
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
        print(
            f"Generating Mermaid flowchart (style: {style}, direction: {direction})..."
        )
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
        lines.append(
            f"    classDef {core_type}Style fill:{color},stroke:#333,stroke-width:2px,color:#000"
        )

    lines.append("")

    # Apply classes to nodes
    for core_type in sorted(unique_core_types):
        # Find all tasks with this core_type
        task_ids = [
            str(task["task_id"]) for task in tasks if task["core_type"] == core_type
        ]
        task_list = ",".join(f"Task{tid}" for tid in task_ids)
        lines.append(f"    class {task_list} {core_type}Style")

    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert swimlane performance JSON to Mermaid flowchart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s                                       # 使用 outputs/ 中最新的 .json 文件
  %(prog)s perf_swimlane_20260210_143526.json   # 输出：outputs/mermaid_diagram_20260210_143526.md
  %(prog)s perf_swimlane_20260210_143526.json -o custom_diagram.md
  %(prog)s perf_swimlane_20260210_143526.json -k kernel_config.py
  %(prog)s perf_swimlane_20260210_143526.json --style compact
  %(prog)s perf_swimlane_20260210_143526.json -v

生成的 Mermaid 图可以：
  1. 直接在 GitHub/GitLab Markdown 中渲染
  2. 在 https://mermaid.live/ 中可视化
  3. 在支持 Mermaid 的编辑器中查看（如 VS Code + Mermaid 插件）
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="输入 JSON 文件 (.json)。如果未指定，使用 outputs/ 目录中最新的 perf_swimlane_*.json 文件",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出 Markdown 文件（默认：outputs/mermaid_diagram_<timestamp>.md）",
    )
    parser.add_argument(
        "-k",
        "--kernel-config",
        help="kernel_config.py 文件路径，用于 func_id 到函数名的映射",
    )
    parser.add_argument(
        "--style",
        choices=["detailed", "compact"],
        default="detailed",
        help="节点信息密度：detailed（详细，包含核心和时间）或 compact（紧凑，仅函数名）",
    )
    parser.add_argument(
        "--direction",
        choices=["TD", "LR"],
        default="TD",
        help="流程图方向：TD（从上到下，默认）或 LR（从左到右）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    # If no input file specified, find the latest .json file in outputs/ directory
    if args.input is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
        json_files = list(outputs_dir.glob("perf_swimlane_*.json"))

        if not json_files:
            print(
                f"错误：在 {outputs_dir} 中未找到 perf_swimlane_*.json 文件",
                file=sys.stderr,
            )
            print("请指定输入文件或确保 outputs/ 中存在 .json 文件", file=sys.stderr)
            return 1

        # Get the most recently modified file
        input_path = max(json_files, key=lambda p: p.stat().st_mtime)

        if args.verbose:
            print(f"自动选择最新文件：{input_path.name}")
    else:
        input_path = Path(args.input)

    # Check input file exists
    if not input_path.exists():
        print(f"错误：输入文件不存在：{input_path}", file=sys.stderr)
        return 1

    try:
        # Read performance data JSON
        if args.verbose:
            print(f"读取性能数据：{input_path}")

        data = read_perf_data(input_path)

        if args.verbose:
            print(f"\n=== 性能数据 ===")
            print(f"  版本：{data['version']}")
            print(f"  任务数量：{len(data['tasks'])}")
            print()

        # Load function name mapping from kernel_config.py if provided
        func_names = {}
        if args.kernel_config:
            if args.verbose:
                print(f"加载 kernel config：{args.kernel_config}")
            func_names = load_kernel_config(args.kernel_config)
            if args.verbose:
                print(f"  加载了 {len(func_names)} 个函数名映射")
                for func_id, name in sorted(
                    func_names.items(), key=lambda x: int(x[0])
                ):
                    print(f"    func_id={func_id}: {name}")
                print()

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Extract timestamp from input filename
            input_stem = input_path.stem  # filename without extension

            # Try to extract timestamp from filename
            if input_stem.startswith("perf_swimlane_"):
                timestamp_part = input_stem[
                    len("perf_swimlane_") :
                ]  # e.g., "20260210_143526"
            else:
                # Fallback: use current timestamp
                timestamp_part = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate output filename and save to outputs/ directory
            outputs_dir = Path(__file__).parent.parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)  # Ensure outputs directory exists
            output_path = outputs_dir / f"mermaid_diagram_{timestamp_part}.md"

        # Generate Mermaid diagram
        mermaid_text = generate_mermaid_flowchart(
            data["tasks"], func_names, args.style, args.direction, args.verbose
        )

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Task Dependency Graph\n\n")
            f.write(f"生成自：`{input_path.name}`\n\n")
            f.write(mermaid_text)

        print(f"\n✓ 转换完成")
        print(f"  输入：{input_path}")
        print(f"  输出：{output_path}")

        return 0

    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
