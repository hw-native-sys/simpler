# CSA 的 AICPU / AICore 调度性能对照

## 结论与范围

2026-10-26，同一 A5、同一 CSA 负载、关闭完整泳道插桩的对照结果：

| 中位数 | AICPU     | AICore    | 增量                   | AICore / AICPU |
| ------ | --------: | --------: | ---------------------: | -------------: |
| Device | 2.133 ms  | 22.664 ms | +20.531 ms（+962.30%） | 10.623         |
| Host   | 77.408 ms | 98.907 ms | +21.499 ms（+27.77%）  | 1.278          |

**当前受测 AICore 版本明显回退，不能称为性能提升。**
两组均通过三轮、每轮七项输出检查；54 个业务 kernel 的运行时代码长度和
FNV1a64 指纹逐项相同。没有改变模型、输入、block 数或 kernel。

本次没有运行时切换调度器的开关，比较的是两个固定 runtime 版本，
不是同一二进制切换一个配置。尚未将 Device 增量拆分到启动、Scalar 调度、
同步及退出，不把 20.531 ms 全部归因于某条 atomic 或某个 helper。

## 固定条件

| 项目           | 配置                                                           |
| -------------- | -------------------------------------------------------------- |
| 模型           | `pypto-lib-csa/models/deepseek_v4_pro/decode_attention_csa.py` |
| 模型提交       | `d1de019cb5e84bcb26b28d9c53756c8ac0fb78e8`                     |
| PyPTO 生成版本 | `a0ff351c519a580e9523c9ba889b662459680f3f`                     |
| 运行方式       | 复用冻结 HBG 独立用例，不重新生成 kernel / golden              |
| 负载           | DeepSeek V4 Pro，B=4，S=2，seed=0                              |
| position_ids   | `[8192,8193,0,1,2,3,3,4]`                                      |
| 工作量         | 54 个 kernel；79 个有 kernel 的逻辑任务；2308 条物理执行       |
| AICPU runtime  | `f5cc679bbf3d7d2058ca453aa6bf0b6686cff989`                     |
| AICore runtime | `98215088b941820778a22e397d0f526fa9f3bd8a`                     |
| PTO ISA        | `418de18dde8fa33773dc5c384fdfe5506463b628`，两组一致           |
| 硬件           | 设备 0，Ascend950PR_958b，32 AIC / 64 AIV                      |
| 工具链         | CANN 9.1.0、CCEC 15.0.5、Torch 2.12.0+cpu                      |
| 环境           | 独立 detached worktree / venv / runtime 构建，同卡串行采样     |
| 采样           | 每组 3 轮正确性检查，再独立进程 5 轮预热 + 100 轮计时          |
| 计时           | `chip_swimlane_level=0`，使用既有 STRACE 时钟记录              |

每轮恢复相同 cache/state 初值，不是连续推进自回归 token 的测量。
输入、golden、源文件清单复用
[CSA fixture](../../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/manifest.json)。
两组清单 SHA256 均为 `d1511befaae48a3b692a723320a076064e2ebb1bced193175866c692f9909b45`。

## 原始样本与统计

单位 μs；P10/P90 为排序样本的线性插值。正式样本均为 100 个。

| 窗口   | 调度方式 | 最小值    | P10       | 中位数    | P90        | 平均值    | 最大值     |
| ------ | -------- | --------: | --------: | --------: | ---------: | --------: | ---------: |
| Device | AICPU    | 2007.707  | 2090.895  | 2133.493  | 2188.056   | 2134.166  | 2223.852   |
| Device | AICore   | 21820.826 | 22142.438 | 22664.058 | 23626.022  | 22782.737 | 24544.193  |
| Host   | AICPU    | 76966.950 | 77152.857 | 77407.935 | 77853.134  | 78053.972 | 89265.455  |
| Host   | AICore   | 97847.536 | 98167.968 | 98906.585 | 103565.779 | 99902.004 | 119661.172 |

- Device：`chip.run.runner_run.device_wall` 的设备时钟窗口，包含设备调度、执行和启停，
  不是 kernel 时间之和。
- Host：`chip.run`，包含构图及搬运等，不能当作纯调度耗时。
- L4 独立采集，只用于确认来源和协议；上表不使用 L4 插桩时长。

[comparison.json](perf_data/2026-10-26-csa/comparison.json) 保存全部正式样本、统计、
runtime SHA256、54 个 kernel 的指纹，以及调度来源与协议审计摘要。
仅将机上绝对路径换成描述性占位符；原始报告仍保留在本地归档。

[aicpu.strace.txt](perf_data/2026-10-26-csa/aicpu.strace.txt) 和
[aicore.strace.txt](perf_data/2026-10-26-csa/aicore.strace.txt) 各保存 105 轮的
Host / Device 原始标记，保留时间、时钟域和调用身份，去掉外围日志与无关阶段。
从这些精简文件重新解析，正式样本及统计与完整日志逐项一致。

## 正确性、来源与干扰

| 检查                       | AICPU               | AICore                     |
| -------------------------- | ------------------- | -------------------------- |
| 正式采样前输出检查         | 3 × 7 项通过        | 3 × 7 项通过               |
| L4 deps / timing 输出检查  | 两轮各 7 项通过     | 两轮各 7 项通过            |
| scheduler producer         | `aicpu`             | `aicore`                   |
| 物理执行记录               | 2308                | 2308（864 AIC / 1444 AIV） |
| Host capture               | 107/107，无丢弃     | 107/107，无丢弃            |
| 显式 block 身份 / 原生协议 | legacy 不提供此能力 | 全部通过                   |
| 依赖顺序审计               | 不冒称新协议验证    | 133 条检查通过             |
| L4 与计时 runtime 二进制   | SHA256 一致         | SHA256 一致                |
| 正式采样的其他 A5 CLI 测试 | 未观察到            | 未观察到                   |

第一轮 AICPU bench 与另一 session 的同卡测试矩阵重叠，已排除。
矩阵退出后重新测量，正式顺序为 AICPU → AICore；每 0.25 秒观察其他 A5 CLI 进程，
两组均无重叠记录。本机没有 `task-submit`，没有硬件独占锁，
进程观察也不证明不存在所有外部负载。

泳道图、raw、完整日志、受测二进制和被排除样本仅保留在本地
`reports/csa_scheduler_compare_20261026.oikNk1/`，不纳入本次提交。
结果只覆盖此固定 CSA 负载，不外推到其他 batch、模型或多卡。

## 复核与复跑

在仓库根目录启用项目 venv，直接复用已有分析器，不新增计时解析实现：

```bash
CSA_CASE=examples/a5/host_build_graph/deepseek_v4_pro_csa
CSA_RESULTS=tests/fdwic/pa_hbg/spmd/perf_data/2026-10-26-csa
python "$CSA_CASE/analyze_perf.py" "$CSA_RESULTS/aicpu.strace.txt"
python "$CSA_CASE/analyze_perf.py" "$CSA_RESULTS/aicore.strace.txt"
```

重新上板时，分别在上述两个固定提交的独立 worktree 中，按
[CSA 运行说明](../../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md)
准备独立 venv、CANN 和冻结数据，执行：

```bash
python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py l4 -d 0
python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py bench -d 0
```

AICore 的 L4 命令须增加 `--require-aicore-scheduler`，不得接受 legacy 回退。
两组串行使用同一设备，不得同时采样；后续使用新的结果目录，不能覆盖本次证据。
