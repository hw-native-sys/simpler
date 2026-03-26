# Swimlane 性能分析工具

本目录包含 PTO Runtime 的性能分析工具。

## 工具列表

- **[swimlane_converter.py](#swimlane_converterpy)** - 转换为 Chrome Trace Event 可视化格式
- **[sched_overhead_analysis.py](#sched_overhead_analysispy)** - Scheduler 开销分析（Tail OH 分解）
- **[perf_to_mermaid.py](#perf_to_mermaidpy)** - 转换为 Mermaid 依赖图
- **[benchmark_rounds.sh](#benchmark_roundssh)** - 批量运行 examples 并报告每轮耗时
- **[device_log_resolver.py](#device_log_resolverpy)** - Device log 路径解析库

---

## swimlane_converter.py

将性能分析数据 JSON 文件转换为 Chrome Trace Event 格式，以便在 Perfetto 中可视化。

### 功能概述

`swimlane_converter.py` 将 PTO Runtime 的性能分析数据（`perf_swimlane_*.json`）转换为可在 Perfetto 跟踪查看器（https://ui.perfetto.dev/）中可视化的格式。同时提供按函数分组的任务执行统计分析，并在解析到 device log 时输出 scheduler overhead deep-dive 报告。

### 基本用法

```bash
# 自动检测 outputs/ 目录中最新的性能分析文件
python3 tools/swimlane_converter.py

# 指定输入文件
python3 tools/swimlane_converter.py outputs/perf_swimlane_20260210_143526.json

# 指定输出文件
python3 tools/swimlane_converter.py outputs/perf_swimlane_20260210_143526.json -o custom_output.json

# 从 kernel_config.py 加载函数名映射
python3 tools/swimlane_converter.py outputs/perf_swimlane_20260210_143526.json \
    -k examples/host_build_graph/paged_attention/kernels/kernel_config.py

# 使用指定 device id 自动选择 device log（device-<id>）
python3 tools/swimlane_converter.py outputs/perf_swimlane_20260210_143526.json -d 0

# 详细模式（用于调试）
python3 tools/swimlane_converter.py outputs/perf_swimlane_20260210_143526.json -v
```

### 命令行选项

| 选项 | 简写 | 说明 |
|------|------|------|
| `input` | | 输入 JSON 文件（perf_swimlane_*.json）。如果省略，使用 outputs/ 中最新的文件 |
| `--output` | `-o` | 输出 JSON 文件（默认：outputs/merged_swimlane_<timestamp>.json） |
| `--kernel-config` | `-k` | kernel_config.py 文件路径，用于函数名映射 |
| `--device-log` | | 设备日志文件/目录/glob 覆盖输入（优先级最高） |
| `--device-id` | `-d` | 指定 device id，从 `device-<id>` 目录自动选择日志 |
| `--verbose` | `-v` | 启用详细输出 |

### device log 选择优先级

`swimlane_converter.py` 和 `sched_overhead_analysis.py` 使用一致的解析规则（由 `device_log_resolver.py` 提供）：

1. `--device-log`（文件/目录/glob）显式覆盖
2. `-d/--device-id` 对应 `device-<id>` 目录
3. 自动扫描 `device-*`，选择最接近 perf 时间戳的 `.log`

log root 解析顺序：
- `$ASCEND_WORK_PATH/log/debug/`
- `~/ascend/log/debug/`（fallback）

### 输出内容

工具生成三类输出：

#### 1. Perfetto JSON 文件

可在 Perfetto 中可视化的 Chrome Trace Event 格式 JSON 文件：
- 文件位置：`outputs/merged_swimlane_<timestamp>.json`
- 打开 https://ui.perfetto.dev/ 并拖入文件即可可视化

#### 2. 任务统计信息

按函数分组的统计摘要（打印到控制台），包含 Exec/Latency 对比和调度开销分析：

- **Exec**：AICore 上的 kernel 执行时间（end_time - start_time）
- **Latency**：AICPU 视角的端到端延迟（finish_time - dispatch_time，包含 head OH + Exec + tail OH）
- **Head/Tail OH**：调度头部/尾部开销
- **Exec_%**：Exec / Latency 百分比（kernel 利用率）

解析到 device log 时，还会输出 Sched CPU（AICPU scheduler 线程实际 CPU 时间 per task）和 Exec/Sched_CPU 比率。

#### 3. Scheduler overhead deep-dive（自动）

当 device log 成功解析后，`swimlane_converter.py` 会直接调用 `sched_overhead_analysis` 的分析逻辑，并在同一次运行中输出：

- Part 1: Per-task time breakdown
- Part 2: AICPU scheduler loop breakdown
- Part 3: Tail OH distribution & cause analysis

### 与 run_example.py 集成

启用性能分析运行测试时，转换器会自动调用：

```bash
# 运行测试并启用性能分析 - 测试通过后自动生成 merged_swimlane.json
python examples/scripts/run_example.py \
    -k examples/host_build_graph/vector_example/kernels \
    -g examples/host_build_graph/vector_example/golden.py \
    --enable-profiling
```

测试通过后，工具将：
1. 自动检测 outputs/ 中最新的 `perf_swimlane_*.json`
2. 从 `-k` 指定的 kernel_config.py 加载函数名
3. 把运行时有效 device id（`-d`）透传给 `swimlane_converter.py`
4. 自动解析 device log 并输出选择策略
5. 生成 `merged_swimlane_*.json` 用于可视化
6. 将任务统计与 scheduler overhead deep-dive 报告打印到控制台

---

## sched_overhead_analysis.py

分析 AICPU scheduler 的调度开销，定量分解 Tail OH（任务完成到 scheduler 确认之间的延迟）的来源。

### 功能概述

`sched_overhead_analysis.py` 从两个数据源进行分析：
1. **Perf profiling 数据**（`perf_swimlane_*.json`）：提取每个 task 的 Exec / Head OH / Tail OH 时间分解
2. **设备日志**（device log）：解析 AICPU scheduler 线程的循环分解（scan / complete / dispatch / idle）、锁竞争和 fanout 统计

支持三种 device log 格式：
1. **New two-level tree**（`PTO2_SCHED_PROFILING=1`）：`=== Scheduler Phase Breakdown: total=Xus, Y tasks ===`，后跟各 phase 行
2. **Legacy detailed**（`PTO2_SCHED_PROFILING=1`）：`completed=X tasks in Yus (Z loops, W tasks/loop)`，后跟 `--- Phase Breakdown ---` 及带 fanout/fanin/pop 统计的 phase 行
3. **Summary**（`PTO2_SCHED_PROFILING=0`）：`Scheduler summary: total_time=Xus, loops=Y, tasks_scheduled=Z`

### 基本用法

```bash
# 自动选取最新的 perf 数据和设备日志
python3 tools/sched_overhead_analysis.py

# 指定 device id 自动选取 device-<id> 日志
python3 tools/sched_overhead_analysis.py --perf-json outputs/perf_swimlane_20260210_143526.json -d 0

# 指定文件
python3 tools/sched_overhead_analysis.py \
    --perf-json outputs/perf_swimlane_20260210_143526.json \
    --device-log ~/ascend/log/debug/device-0/device-*.log
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `--perf-json` | perf_swimlane_*.json 文件路径。省略时自动选取 outputs/ 中最新的文件 |
| `--device-log` | 设备日志文件/目录/glob 覆盖输入（优先级最高） |
| `-d, --device-id` | 指定 device id，从 `device-<id>` 自动选取日志 |

### 输出内容

分三部分输出：

- **Part 1：Per-task time breakdown** — Exec / Head OH / Tail OH 各占 Latency 的百分比
- **Part 2：AICPU scheduler loop breakdown** — 各 scheduler 线程的循环统计、各阶段（scan / complete / dispatch / idle）耗时占比、锁竞争和 fanout/fanin/pop 统计
- **Part 3：Tail OH distribution & cause analysis** — Tail OH 分位数分布（P10~P99）、scheduler 循环迭代耗时与 Tail OH 的关联分析、主导 phase 的数据驱动洞察

---

## perf_to_mermaid.py

将性能分析数据转换为 Mermaid 流程图格式，可视化任务依赖关系。

### 功能概述

`perf_to_mermaid.py` 将 PTO Runtime 的性能分析数据（`perf_swimlane_*.json`）转换为 Mermaid 流程图格式。生成的 Markdown 文件可以：
- 在 GitHub/GitLab 中直接渲染
- 在 https://mermaid.live/ 中查看
- 在支持 Mermaid 的编辑器中查看（如 VS Code + Mermaid 插件）

### 基本用法

```bash
# 自动检测 outputs/ 目录中最新的性能分析文件
python3 tools/perf_to_mermaid.py

# 指定输入文件
python3 tools/perf_to_mermaid.py outputs/perf_swimlane_20260210_143526.json

# 指定输出文件
python3 tools/perf_to_mermaid.py outputs/perf_swimlane_20260210_143526.json -o diagram.md

# 从 kernel_config.py 加载函数名映射
python3 tools/perf_to_mermaid.py outputs/perf_swimlane_20260210_143526.json \
    -k examples/host_build_graph/paged_attention/kernels/kernel_config.py

# 使用紧凑样式（仅显示任务ID和函数名）
python3 tools/perf_to_mermaid.py outputs/perf_swimlane_20260210_143526.json --style compact

# 指定流程图方向（从左到右）
python3 tools/perf_to_mermaid.py outputs/perf_swimlane_20260210_143526.json --direction LR

# 详细模式
python3 tools/perf_to_mermaid.py outputs/perf_swimlane_20260210_143526.json -v
```

### 命令行选项

| 选项 | 简写 | 说明 |
|------|------|------|
| `input` | | 输入 JSON 文件（perf_swimlane_*.json）。如果省略，使用 outputs/ 中最新的文件 |
| `--output` | `-o` | 输出 Markdown 文件（默认：outputs/mermaid_diagram_<timestamp>.md） |
| `--kernel-config` | `-k` | kernel_config.py 文件路径，用于函数名映射 |
| `--style` | | 节点样式：`detailed`（默认，包含函数名和任务ID）或 `compact`（仅任务ID）|
| `--direction` | | 流程图方向：`TD`（从上到下，默认）或 `LR`（从左到右）|
| `--verbose` | `-v` | 启用详细输出 |

### 输出内容

生成包含 Mermaid 流程图的 Markdown 文件：

#### Detailed 样式（默认）

```mermaid
flowchart TD

    Task0["QK(0)"]
    Task1["SF(1)"]
    Task2["PV(2)"]
    Task3["UP(3)"]
    Task4["QK(4)"]
    Task5["SF(5)"]
    Task6["PV(6)"]
    Task7["UP(7)"]
    Task8["QK(8)"]
    Task9["SF(9)"]
    Task10["PV(10)"]
    Task11["UP(11)"]
    Task12["QK(12)"]
    Task13["SF(13)"]
    Task14["PV(14)"]
    Task15["UP(15)"]

    Task0 --> Task1
    Task1 --> Task2
    Task2 --> Task3
    Task3 --> Task7
    Task4 --> Task5
    Task5 --> Task6
    Task6 --> Task7
    Task8 --> Task9
    Task9 --> Task10
    Task10 --> Task11
    Task11 --> Task15
    Task12 --> Task13
    Task13 --> Task14
    Task14 --> Task15

    %% Styling by core type
    classDef aicStyle fill:#66A3FF,stroke:#333,stroke-width:2px,color:#000
    classDef aivStyle fill:#FFB366,stroke:#333,stroke-width:2px,color:#000

    class Task0,Task2,Task4,Task6,Task8,Task10,Task12,Task14 aicStyle
    class Task1,Task3,Task5,Task7,Task9,Task11,Task13,Task15 aivStyle
```

---

## benchmark_rounds.sh

批量运行预定义的 examples，解析 device log 中的 timing 行并报告每轮耗时。

### 功能概述

`benchmark_rounds.sh` 遍历 `EXAMPLES` 数组中配置的测试用例（位于 `tests/device_tests/tensormap_and_ringbuffer/` 下），依次调用 `run_example.py` 运行每个 example，然后从生成的 device log 中提取 `orch_start` / `orch_end` / `sched_end` 时间戳计算每轮 elapsed 时间。

当前预配置的 examples：
- `alternating_matmul_add`
- `benchmark_bgemm`
- `paged_attention_unroll`
- `batch_paged_attention`
- `paged_attention`

### 基本用法

```bash
# 使用默认参数（device 0, 10 rounds）
./tools/benchmark_rounds.sh

# 指定 device 和 rounds
./tools/benchmark_rounds.sh -d 4 -n 20

# 额外参数透传给 run_example.py
./tools/benchmark_rounds.sh -d 0 -n 5 --case 1
```

### 命令行选项

| 选项 | 简写 | 说明 |
|------|------|------|
| `--device` | `-d` | Device ID（默认：0） |
| `--rounds` | `-n` | 每个 example 的运行轮数（默认：10） |
| `--help` | `-h` | 显示帮助信息 |

所有未识别的参数会透传给 `run_example.py`。

### 输出内容

对每个 example 输出：
- 每轮的 Elapsed 时间（微秒）
- 平均耗时和总轮数

最终输出汇总：passed / failed 数量。

### Device log 解析

脚本通过以下方式定位 device log：
- 优先使用 `$ASCEND_WORK_PATH/log/debug/device-<id>/`
- Fallback 到 `~/ascend/log/debug/device-<id>/`
- 在运行前快照已有 log 文件，运行后等待新 log 文件出现（最多 15 秒）

---

## device_log_resolver.py

Device log 路径解析库，被 `swimlane_converter.py` 和 `sched_overhead_analysis.py` 共同使用。

### 功能概述

`device_log_resolver.py` 提供确定性的 device log 路径解析逻辑，支持三种选择优先级：

1. **显式路径**（`--device-log`）：支持文件、目录、glob 模式
2. **Device ID**（`--device-id`）：从 `<log_root>/device-<id>/` 选择最新 `.log`
3. **自动扫描**：遍历所有 `device-*` 目录，选择与 perf 时间戳最接近的 `.log`

### 主要函数

| 函数 | 说明 |
|------|------|
| `get_log_root()` | 返回 log root 路径（`$ASCEND_WORK_PATH/log/debug/` 或 `~/ascend/log/debug/`） |
| `infer_device_id_from_log_path(log_path)` | 从路径中推断 device id（如 `device-0`） |
| `resolve_device_log_path(device_id, device_log, perf_path)` | 按优先级解析 device log 路径，返回 `(Path, strategy_string)` |

### 使用方式

该模块不作为独立命令行工具使用，而是被其他工具导入：

```python
from device_log_resolver import resolve_device_log_path

log_path, strategy = resolve_device_log_path(
    device_id="0",
    device_log=None,
    perf_path=Path("outputs/perf_swimlane_20260210_143526.json"),
)
```

---

## 共同配置

### 输入文件格式

分析工具共用相同的输入格式 - PTO Runtime 生成的 `perf_swimlane_*.json` 文件：

```json
{
  "version": 1,
  "tasks": [
    {
      "task_id": 0,
      "func_id": 0,
      "core_id": 0,
      "core_type": "aic",
      "start_time_us": 100.0,
      "end_time_us": 250.5,
      "duration_us": 150.5,
      "fanout": [1, 2],
      "fanout_count": 2
    }
  ]
}
```

### Kernel Config 格式

要在输出中显示有意义的函数名，需要提供 `kernel_config.py` 文件：

```python
KERNELS = [
    {
        "func_id": 0,
        "name": "QK",
        # ... 其他字段
    },
    {
        "func_id": 1,
        "name": "SF",
        # ... 其他字段
    },
]
```

工具从 `KERNELS` 列表中提取 `func_id` 到 `name` 的映射。

---

## 工具选择建议

### 使用 swimlane_converter.py 当你需要：
- 查看详细的时间线执行视图
- 分析任务在不同核心上的调度情况
- 查看精确的执行时间和时间间隔
- 获取任务执行的统计信息
- 专业的性能分析和优化

### 使用 perf_to_mermaid.py 当你需要：
- 快速查看任务依赖关系
- 在文档中嵌入依赖图
- 在代码审查中分享依赖结构
- 不需要时间线细节，只关注拓扑结构
- 在 GitHub/GitLab 中直接查看

### 使用 benchmark_rounds.sh 当你需要：
- 批量运行多个 examples 并对比耗时
- 获取每轮的 elapsed 时间统计
- 在硬件上做端到端性能回归测试

### 推荐工作流

```bash
# 1. 运行测试获取性能数据
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --enable-profiling

# 2. 生成 Perfetto 可视化（自动）
# → outputs/merged_swimlane_*.json

# 3. 生成 Mermaid 依赖图
python3 tools/perf_to_mermaid.py -k ./kernels/kernel_config.py

# 4. 批量 benchmark（硬件上）
./tools/benchmark_rounds.sh -d 0 -n 20

# 5. 分析结果
# - 详细性能分析：Perfetto (https://ui.perfetto.dev/)
# - 依赖关系概览：Mermaid 图（GitHub/编辑器）
# - 统计摘要：控制台输出
```

---

## 故障排查

### 错误：找不到 perf_swimlane_*.json 文件
- 确保使用 `--enable-profiling` 标志运行了测试
- 检查 outputs/ 目录是否存在并包含性能分析数据

### 警告：Kernel entry missing 'func_id' or 'name'
- 检查 kernel_config.py 文件格式
- 确保所有 KERNELS 条目都有 'func_id' 和 'name' 字段

### 错误：Unsupported version
- 工具仅支持版本 1 的性能分析数据格式
- 使用最新的 runtime 重新生成性能分析数据

### 错误：Perf JSON missing required fields for scheduler overhead analysis
- 该错误表示输入的 `perf_swimlane_*.json` 缺少 deep-dive 分析需要的字段（通常是 `dispatch_time_us` / `finish_time_us`）
- `swimlane_converter.py` 的基础转换可继续成功，但 deep-dive 会跳过或失败
- 处理路径：
  1. 使用 `--enable-profiling` 重新跑一次，生成新的 `outputs/perf_swimlane_*.json`
  2. 重新执行 `swimlane_converter.py` 或 `sched_overhead_analysis.py`
  3. 检查 JSON 中每个 task 是否包含 `dispatch_time_us` 和 `finish_time_us`

### benchmark_rounds.sh 无 timing 数据
- 确保运行时启用了 profiling（`PTO2_PROFILING` 环境变量）
- 检查 device log 目录是否可访问
- 确认 log 中包含 `orch_start` / `orch_end` / `sched_end` 时间戳行（需要 `PTO2_PROFILING=1`）

### Mermaid 图在 GitHub 上不显示
- 确保文件是 `.md` 扩展名
- 检查 Mermaid 语法是否正确
- GitHub 有时需要刷新才能渲染 Mermaid 图

---

## 输出文件说明

| 文件 | 工具 | 用途 | 格式 |
|------|------|------|------|
| `perf_swimlane_*.json` | Runtime | 原始性能分析数据 | JSON |
| `merged_swimlane_*.json` | swimlane_converter.py | Perfetto 可视化 | Chrome Trace Event JSON |
| `mermaid_diagram_*.md` | perf_to_mermaid.py | 依赖关系图 | Markdown + Mermaid |

---

## 相关资源

- [Perfetto Trace Viewer](https://ui.perfetto.dev/)
- [Mermaid Live Editor](https://mermaid.live/)
- [Mermaid 文档](https://mermaid.js.org/)
