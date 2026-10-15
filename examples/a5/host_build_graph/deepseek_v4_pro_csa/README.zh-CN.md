# DeepSeek V4 Pro CSA：HBG 独立调度验证用例

本目录提供原生 `SceneTestCase` 的 **HBG 独立用例**。冻结 kernel、参数 ABI、数据清单、七项 golden、运行和分析工具均在本目录内，不导入其他 runtime 的用例，也不依赖 PyPTO 或模型仓库。首次使用需另行导入固定数据，数据文件不进入 Git。

**当前边界：HBG 已在 A5 跑通，但当前主线对这个 CSA 使用的是 AICPU 调度，不是 AICore 分布式调度。** `main@946f49ac` 的 resident v0 只接受 `logical_block_num == 1` 且不要求 sync-start 的任务。CSA 有多 block 的 SPMD 任务，整图会选择 `LEGACY_UNSUPPORTED_SHAPE`；本机 L4 记录的 `scheduler_tasks.producer` 和五个 scheduler stream 均为 `aicpu`。

因此，这份用例是后续扩展、调试 HBG 分布式调度的固定验收负载；**当前结果不能当作分布式调度的性能基线**。本次没有修改 runtime、拆任务、减少 SPMD block 数或改变 kernel 来绕过限制。

## 1. 入口与适配范围

| 文件 | 用途 |
| ---- | ---- |
| `test_deepseek_v4_pro_csa.py` | `@scene_test(level=2, runtime="host_build_graph")` 独立入口 |
| `workload.py`、`kernel_config.py` | 本目录冻结 kernel 的注册、52 项参数 ABI 和源码一致性检查 |
| `kernels/vendor/orchestration/attention_csa_test.cpp` | HBG host 构图入口，可直接跟踪 `rt_submit_*` |
| `kernels/vendor/aic/`、`kernels/vendor/aiv/` | 54 个逐字节保留的生成 kernel |
| `fixture.py`、`manifest.json` | 本目录数据导入、58 个数据文件与 55 个 C++ 源码的 SHA256 校验 |
| `golden.py`、`_validation.py`、`_compressor_reference.py` | 原 CSA 七项比较规则及 Torch 参考函数 |
| `run.py` | 正确性、性能、L1/L4，以及实际调度路径校验 |
| `analyze_perf.py` | HBG 的 Host/Device 计时统计 |
| `baseline_20261014.json` | 本机统计、逐轮样本、版本摘要和采集完整性结果 |

负载为 Pro、B=4、S=2、seed=0，`position_ids=[8192,8193,0,1,2,3,3,4]`。共 52 个参数、54 个注册 kernel、79 个实际执行的逻辑任务、2308 条物理执行记录。

来源为 `hw-native-sys/pypto-lib` 的 `restore/deepseek-v4-pro-a5-decode-csa` 分支、提交 `d1de019cb5e84bcb26b28d9c53756c8ac0fb78e8`，入口 `models/deepseek_v4_pro/decode_attention_csa.py`。本目录 `manifest.json` 记录完整来源、原清单摘要、原构图源码摘要及 HBG 源码摘要。输入和 golden 不重新生成，54 个 kernel 不改写。

只有三项适配：

1. C++ 构图中的 `Tensor` 显式写成 `simpler::hbg::Tensor`，避免与 HBG 公共头引入的其他 Tensor 类型混淆。
2. 30 处 `launch_spec.set_core_num(...)` 改用 HBG 的 `set_block_num(...)`。参数表达式、逻辑 block 数、提交顺序和 kernel ID 均保持原样。
3. `hc_attn_scale` 使用 `TensorTransfer.NONE`，保留 host 可读地址。HBG 构图读取其三个标量；它不是设备 kernel 输入。其余参数传输方式和每轮 INOUT 初值恢复不变。

CPU 测试会逆向还原上述两项 C++ 文本适配，并检查还原后的 SHA256 与原文件一致；同时检查 52 个参数中只有 `hc_attn_scale` 改变传输方式。另有隔离子进程禁止导入 TMR 用例，确保 HBG 不依赖本地未提交文件。

触发当前回退的第一项任务为 `task_id=7`、`kernel_ids=[1,2,-1]`、`block_num=4`；整份图包含 60 个多 block 任务，最大的 `block_num=512`。判定位置是 `src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h` 的 `scheduler_resident_v0_task_shape_supported()`，整图回退位置是 `host/runtime_maker.cpp` 的 `create_scheduler_state()`。把 block 数直接改成 1 会改变计算，不能作为分布式路径接通的修复。

## 2. 环境与复跑命令

命令均在 **simpler 仓库根目录**执行，使用独立 `.venv-csa`、Torch `2.12.0+cpu` 和固定数据。新的工作目录先创建环境和编译当前 checkout；不需要重新运行 PyPTO 生成 kernel：

```bash
python3 -m venv --system-site-packages .venv-csa
source .venv-csa/bin/activate
python -m pip install 'torch==2.12.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'scikit-build-core>=0.10' 'nanobind>=2,<3' 'cmake>=3.20' pytest cloudpickle numpy
# 加载本机 CANN set_env.sh 后构建。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CMAKE_BUILD_PARALLEL_LEVEL=24 MAX_JOBS=24 \
  python -m pip install --no-build-isolation --no-deps -e .
```

Torch 仅用于 CPU 数据和 golden；`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 避免加载系统中版本不匹配的 Torch NPU 插件，不会关闭 simpler 的设备执行。切换提交或修改 runtime/platform C++ 后，需重新执行 editable 安装；运行前会检查本地 checkout、Torch 2.12 和 native binding 的构建提交。

`data/` 包含 58 个文件，共 418,374,879 字节，约 399 MiB；它和 `.o/.so` 均被本目录 `.gitignore` 排除。首次使用，从原始已验证产物或另一份相同快照导入，目标是 **HBG 本目录的独立副本**：

```bash
CSA_HBG=examples/a5/host_build_graph/deepseek_v4_pro_csa
python "$CSA_HBG/fixture.py" \
  ../pypto-lib-csa/build_output/_jit_attention_csa_test_20261014_111102_es2p49y4/data
python "$CSA_HBG/fixture.py"
```

导入器验证全部文件的大小和 SHA256，拒绝覆盖已有但内容不同的文件。`data/in/` 是输入及 INOUT 初值，`data/out/` 是原模型 Torch golden，不能用设备输出替换。导入后无需保留来源目录；缺少数据时 pytest 明确跳过，`run.py` 直接报错，不会把跳过当成功。

```bash
source .venv-csa/bin/activate
# 先按本机实际路径加载 CANN 的 set_env.sh。
# 若 ASCEND_HOME_PATH 已设置：
source "$ASCEND_HOME_PATH/set_env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CSA_HBG=examples/a5/host_build_graph/deepseek_v4_pro_csa

# 连续三轮，逐轮检查全部七项输出。
python "$CSA_HBG/run.py" check -d 0

# 先做三轮正确性检查，再用独立进程运行 5 轮预热 + 100 轮性能。
python "$CSA_HBG/run.py" bench -d 0

# 依赖图与计时分进程采集，两次都检查 golden。
python "$CSA_HBG/run.py" l1 -d 0
python "$CSA_HBG/run.py" l4 -d 0

# 分布式调度验收门禁：必须有 L4 证据，且 producer 必须为 aicore。
python "$CSA_HBG/run.py" l4 -d 0 --require-aicore-scheduler
```

最后一条命令在当前主线会明确失败，原因是 AICPU 回退，而不是 CSA 数值错误。后续分布式调度器支持本负载后，它应通过；不能删除这道检查来宣称分布式路径已经接通。`check`、`bench`、L1 不具备判断 scheduler producer 的证据，不能与此门禁等同。

最小入口与预编译命令：

```bash
python "$CSA_HBG/test_deepseek_v4_pro_csa.py" \
  -p a5 -d 0 --manual include --rounds 3

python -m simpler_setup.tools.scene_test_compile \
  "$CSA_HBG/test_deepseek_v4_pro_csa.py" \
  --platform a5 --manual include --compile-workers 24
```

用例仅支持 A5，标记为 `manual`。每次命令保存到唯一的 `outputs/csa-hbg-{模式}-*/`，最终打印 `result.json`。记录包含当前 checkout、HBG 三个 runtime 二进制的 SHA256、HBG 构图源码 SHA256、本目录 fixture 摘要、实际命令及日志。修改 runtime C++ 后需重建：

```bash
CMAKE_BUILD_PARALLEL_LEVEL=24 MAX_JOBS=24 \
  python -m pip install --no-build-isolation --no-deps -e .
```

七项检查为 `x_out`、`kv_cache`、`cmp_kv`、`compress_state`、`inner_compress_state`、`idx_kv_cache`、`idx_kv_scale`，保持原 CSA 的容差、映射和未映射区域保护规则。`cmp_kv` 先独立验证 compressor state，再以通过校验的实际 state 计算量化 cache 参考值；不放松 state 校验。性能阶段显式跳过 golden，正确性由先行三轮提供；所有轮次恢复相同 cache/state 初值，不是连续推进的自回归 token。

## 3. 性能与泳道口径

HBG 主指标是 `device_us`，来自 `chip.run.runner_run.device_wall` 的设备时钟记录，已经按计数器频率换算为 µs。它包含设备运行窗口，不等于纯 kernel 时间之和，也不包含 host 构图时间。

`host_us` 是较大的 host 调用窗口，包含构图、数据搬运等；当前每轮传输大量固定数据，不能把它当纯调度耗时。HBG 没有与 TMR 等价的设备 orchestrator 区间，本用例不输出伪造的 `effective_us`、`orch_us`、`sched_us`，也不将 HBG `device_us` 与 TMR `effective_us` 直接相减来计算收益。

L1/L4 仅解释行为，不代替无泳道插桩性能。校验器会对照 `deps.json` 检查逐任务 block/subtask 数与 2308 条物理记录；L4 还检查调度记录关联、scheduler phase 丢失/截断、host 构图采集是否完整，并写出：

```json
{
  "scheduler_producer": "aicpu",
  "distributed_scheduler_verified": false,
  "clock_alignment_status": "bounded"
}
```

`bounded` 表示 host 与 device 时间轴有带不确定范围的对齐，不是精确同钟。脚本会保留 `host_clock_alignment.*.log` 等辅助文件，并使用仓库的标准转换器生成 `L1.json` / `L4.json`，可导入 Perfetto；不要丢弃这些文件后自行把 host 和设备绝对时间相减。

## 4. 本机验证与后续使用

2026-10-14，`main@946f49acbbd4df72b333784cfad60b1573e70b40`，设备 0，`Ascend950PR_958b`，32 AIC + 64 AIV，设备计数器 1 GHz，Torch `2.12.0+cpu`，CANN `9.2.0-weekly-20260715`。AICPU 拓扑查询按 runtime 现有策略回退为五线程，不能据此确认它们的物理 DIE 归属。

HBG 三轮 × 七项输出全部通过。无泳道插桩的 100 轮结果如下，单位 µs；这是 **HBG + AICPU 回退路径** 的一次基线，不是优化收益结论。

| 指标 | 最小值 | 中位数 | 平均值 | 最大值 |
| ---- | -----: | -----: | -----: | -----: |
| `device_us` | 2040.359 | 2131.0075 | 2134.79915 | 2249.551 |
| `host_us` | 102962.932 | 113156.443 | 115308.16656 | 151511.088 |

逐轮样本和结果索引保存在 [baseline_20261014.json](baseline_20261014.json)。性能原始结果为 `outputs/csa-hbg-bench-xws0owx0/result.json`。L1/L4 路径及完整性记录也见该基线文件。

- L1：`outputs/csa-hbg-l1-4hy53mry/L1.json`，79 个逻辑任务、2308 条物理记录。
- L4：`outputs/csa-hbg-l4-q95jbmo_/L4.json`，执行记录齐全，107 条 host 构图记录完整，scheduler producer 为 `aicpu`。
- 严格分布式门禁：`outputs/csa-hbg-l4-ctk1wscv/result.json`，两次七项 golden 均通过后，因 AICPU 回退按预期返回失败。

该历史基线的 `fixture_manifest_sha256` 保留采集时的原清单摘要，与本目录清单中的 `source_manifest_sha256` 对应；独立 HBG 清单包含 runtime 和 HBG 构图摘要，因此清单本身的 SHA256 不同。历史性能样本和二进制摘要不重写为当前版本，也不把目录独立化当作性能优化。

后续顺序：修改 HBG 调度器并重建 → `check` → `l4 --require-aicore-scheduler` → `bench`。门禁通过后，再用多次无插桩测量判断分布式调度改动的收益；保持 kernel、输入、工具链、设备和采集级别不变。这个入口只覆盖单卡内的 CSA 固定负载，不表示已验证多卡调度或完整网络。

CPU 合约测试不需要设备或大数据文件：

```bash
python -m pytest -q \
  tests/ut/py/test_deepseek_v4_pro_csa_hbg.py \
  tests/ut/py/test_deepseek_v4_pro_csa_hbg_fixture.py \
  tests/ut/py/test_scene_test_golden_hooks.py
```

2026-10-15 的独立用例复验中，上述 60 项检查全部通过。仅从 Git 待提交树导出 HBG 用例和两个专项测试、不包含 TMR 用例或数据的干净目录中，53 项专项测试也全部通过。测试覆盖无 TMR 目录依赖、冻结源码、接口适配、host-only 参数、七项 golden、错误计时记录、采集不完整以及 AICPU 回退门禁。

同日使用基于 `7e19db4df679278d8bc4f9eea06c8e8fda4a492e` 的 feature 分支 runtime 在设备 0 复验：

- `outputs/csa-hbg-check-rdhhgfi_/result.json`：三轮 × 七项输出全部通过。
- `outputs/csa-hbg-l4-5dlwq5b0/L4.json`：依赖与计时两次采集的七项输出均通过，79 个逻辑任务、2308 条物理记录、107 条 host 构图记录完整。
- 对该实测 L4 执行严格 producer 校验，因 `aicpu` 回退按预期拒绝；没有把数值通过认作 AICore 调度通过。
- 2026-10-14 的 HBG 历史 benchmark 由本目录分析器重算，100 轮样本及统计完全一致；本次未重新采样性能基线。
