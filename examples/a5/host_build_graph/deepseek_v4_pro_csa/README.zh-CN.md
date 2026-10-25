# DeepSeek V4 Pro CSA：HBG 独立调度验证用例

本目录提供原生 `SceneTestCase` 的 **HBG 独立用例**。冻结 kernel、参数 ABI、数据清单、七项 golden、运行和分析工具均在本目录内，不导入其他 runtime 的用例，也不依赖 PyPTO 或模型仓库。首次使用需另行导入固定数据，数据文件不进入 Git。

**当前边界：原样 CSA 的七项 golden、显式 block/slot 身份和原生协议因果审计已在 A5 通过严格 L4 门禁。** 普通 C/V 与 GM Mix 支持原 N 分波；无 kernel Dummy 只推进逻辑依赖，不伪造物理执行。创建用例时的 `main@946f49ac` 仅支持单 block，历史结果仍属于 AICPU 回退，不能改写为本版证据。sync_start 和性能比较尚未完成。最新实现边界见 [SPMD 设计](../../../../tests/fdwic/pa_hbg/a5-hbg-spmd-design.md)，逐步实现和实测集中于 [实现过程](../../../../tests/fdwic/pa_hbg/spmd/implementation-progress.md)。

因此，这份用例是后续扩展、调试 HBG 分布式调度的固定验收负载；**历史 AICPU 结果不能当作分布式调度的性能基线**。用例不通过拆任务、减少 SPMD block 数或改变 kernel 绕过限制。

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

历史验收中首先触发回退的任务为 `task_id=7`、`kernel_ids=[1,2,-1]`、`block_num=4`；整份图包含 60 个多 block 任务，最大的 `block_num=512`。当前判定位置是 `src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h` 的 `scheduler_resident_task_shape_supported()`，普通 C/V/Mix 均可原 N 执行；sync_start 仍由 `host/runtime_maker.cpp::create_scheduler_state()` 整图回退。把 block 数直接改成 1 会改变计算，不能作为分布式路径接通的修复。

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

# 分布式调度验收：L4、AICore 来源、执行身份和原生协议因果证明必须齐全。
python "$CSA_HBG/run.py" l4 -d 0 --require-aicore-scheduler
```

最后一条命令已在当前 GM Mix 实现上通过；旧分支 AICPU 回退仍必须失败。不能删除来源、记录数量、依赖因果或采集完整性检查来使它通过。`check`、`bench`、L1 不具备完整协议证明，不能与此门禁等同。历史仅带执行身份、没有 `aicore_protocol` 的 L4 捕获仍可读取，但也不能通过当前严格门禁。

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

## 5. 真实 CSA 的调度 TDD（2026-10-23）

入口为 [test_csa_business.py](../../../../tests/fdwic/pa_hbg/csa/test_csa_business.py)，直接继承本目录的原生用例，复用全部 54 个 kernel、52 项 ABI 和七项比较器。**没有换成探针、缩减 block 数、重新生成 golden，或修改生产 runtime。** 用例参数只改变经过验证的物理存储布局；性能比较仍用原来的 `run.py bench`，不能混合不同布局的计时。

业务数值与调度来源是两道独立门禁：

| 验收组 | 本次结果 | 后续通过条件 |
| ------ | -------- | ------------ |
| `TestCSABusiness` | A5 上 12 个用例 × 3 轮全部通过，252 项输出校验 | 所有输出继续符合原比较规则 |
| `TestCSAResident` | A5 上 3 个用例均数值通过，再因 `mode=4` 失败 | 数值通过，且每次调用新发布 resident 模式 |
| `run.py l4 --require-aicore-scheduler` | 数值和采集完整性通过，因 `producer=aicpu` 失败 | L4 的 scheduler producer 为 `aicore` |

不能给后两组加 `xfail`、跳过 golden 或接受回退来使它们变绿。`TestCSABusiness` 允许 legacy，是稳定的业务基线，不是 resident 的替代证明。所有硬件用例保持 `manual`，避免在没有固定数据的机器上误启动大负载。

### 业务变体及 oracle 边界

[business_cases.py](business_cases.py) 定义 12 个用例：原始基线；main state、inner state、KV、compressed KV、index cache 五组分别反转；全部物理页反转、循环移动、固定种子打乱；index 页单独循环移动；index scale 写入扰动；全页打乱与 scale 扰动组合。

- 每组页重排同时更新 pool、block table 和 slot mapping；普通 KV 还更新 `window_swa_indices`，index payload/scale 始终使用同一个映射。`-1` 哨兵、页内偏移、FP8 位模式和原有别名关系不变，golden 仅做相同物理页重排。
- 原始 `idx_kv_scale` 的输入恰好等于输出。只看它数值通过，不能证明写入发生。扰动用例将**本次应写的两个槽位**初值置为 2，仍使用原 golden，迫使 `kv_and_cache_write` 正确覆盖；其他槽位保持原样。
- 仍严格检查六个 pool 的未映射区域，不允许数值容差掩盖越界写。未触碰 compressor state 中原本合法的 `-Inf` 也必须保留，不能对整个 pool 一概要求有限值。
- 每轮由现有 SceneTest L2 循环恢复七个输出缓冲区的初值。另有 CPU 测试直接驱动该循环，在三次设备调用的替身入口逐项断言已恢复，避免只用“重复输出相同”推断 reset 正确。
- **不把 batch 任意排列当作等价变换。** 快照中请求 0 的 position 8192 和请求 2 的历史 position 0 都映射到 state 物理页 2，涉及行 8、9 的跨请求读写。真机单独复测 batch 反转时，`x_out`、`cmp_kv`、`idx_kv_cache` 不一致；旧 golden 不能直接按 batch 重排复用。该探索不是 scheduler 回归证据，最终测试拒绝这种未获支持的变换，并锁定两个 pool 的别名事实。

### 真实图与边界断言

[graph_contract.json](graph_contract.json) 来源于本次 A5 HBG 的 `deps.json`，保留原始采集 SHA256、commit 和输入清单摘要。图包含 79 个执行任务、2 个 Dummy、133 条去重后的图内依赖边，60 个多 block 任务、2308 条执行记录（AIC 864 / AIV 1444）。Host 已完成的分配节点另列为 `host_ready_refs`，不伪造它们的设备执行。

[graph_contract.py](graph_contract.py) 在 `run.py l1/l4` 中检查原始 kernel 组合、block 数、依赖和 tensor view，并检查逐任务 AIC/AIV 数量、单次正数 epoch 和包括 Dummy 在内的依赖。A5 resident 的新 capture 通过 `aicore_execution_identities` 提供入口采集的 `block_idx` / `subtask_slot`，并按原图逐项核对 2308 个唯一身份。L4 另以 `aicore_protocol` 记录实际准入、成员、领取、许可、组握手和完成；统一审计器基于本核序号和跨核发布／观察建立因果偏序，不把跨核时间戳当作全局顺序。冻结图须与独立 dep-gen 的任务、kernel、原 N 和 fanin 完全一致。严格入口同时要求 `block_identity_verified=true` 与 `protocol_verified=true`；历史缺失相应对象时保持未验证，不能按条数补造证明。

CPU TDD 分为两层：

- 不需要大数据：`test_csa_business_variants.py`、`test_csa_capture_contract.py`、`test_csa_graph_contract.py`、`test_csa_scene_contract.py` 和 `hbg_spmd_contracts/test_csa_protocol.py`。逐条删除 133 条边、逐项缩减 60 个多 block 任务、去掉 Mix 的 AIV、破坏 16 个 pack 的 tensor view 都必须失败；完整 DAG 在 1/4/32 cluster 和不同 ready 顺序下执行。专测 Mix 512 的首块/波次边界/末块拖尾、最后一条 lane 的可见性、8 路 merge、16 路 projection 汇合、Dummy 以及普通分波不能误标成 sync_start。
- 需要固定数据但不需要卡：[test_csa_snapshot.py](../../../../tests/fdwic/pa_hbg/csa/test_csa_snapshot.py)。检查真实七项 golden、全部布局变换、各 pool 首末写入/未映射行破坏、NaN/Inf、8 个 token 的最终输出损坏和 scale 漏写。此文件缺数据会失败，不会静默跳过。

本次先写失败测试，确认旧采集校验遗漏“执行总数相同但核类型错误、epoch 混杂/为零、消费者越过 Dummy 提前执行”四个反例，再修正用例侧校验器。协议模型测试变绿只说明设计断言自洽，不说明生产 Scalar 调度器已实现。

本次新增 460 项 CPU／快照检查全部通过；与既有 SPMD、CLA、CSA 和 golden hook 测试合并回归，共 1021 项通过。5 个相关 C++ 测试目标、Ruff、Markdown 和 Python 3.9 语法检查通过。

### 复跑及实测记录

在仓库根目录使用 Torch 2.12 的 `.venv-csa`，先确认已为当前 checkout 重编 native binding。以下不新增环境开关、pytest 参数或调度配置：

```bash
source .venv-csa/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# 12 个真实业务用例，每个 3 轮；同一 Worker 连续复用。
python tests/fdwic/pa_hbg/csa/test_csa_business.py \
  -p a5 -d 0 --manual include --case TestCSABusiness:: --rounds 3

# 故意保留的 resident 红灯；不是纯协议模型。
python tests/fdwic/pa_hbg/csa/test_csa_business.py \
  -p a5 -d 0 --manual include --case TestCSAResident::

# 本次新增的 CPU 和真实快照测试。
python -m pytest -q tests/ut/py/test_csa_*.py \
  tests/ut/py/hbg_spmd_contracts/test_csa_protocol.py \
  tests/fdwic/pa_hbg/csa/test_csa_snapshot.py

# 原有入口保持可用；L4 严格门禁仍拒绝回退。
python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py l4 \
  -d 0 --require-aicore-scheduler
```

本次设备 0 为 `Ascend950PR_958b`，当前 checkout 为 `f5cc679bbf3d7d2058ca453aa6bf0b6686cff989`，Torch `2.12.0+cpu`。没有使用缺失的 `npu-smi` 或 `task-submit`，通过 CANN 查询确认 A5 后直接上卡，未获取设备锁。结果索引见 [results_20261023.json](../../../../tests/fdwic/pa_hbg/csa/results_20261023.json)，其中区分通过、预期红灯和排除的 batch 变换探索。

原始日志保存在 `outputs/csa-business-a5-20261023-validated.log`（36 次调用，epoch 1–36）、`outputs/csa-resident-a5-20261023.log`、`outputs/csa-hbg-check-t5gdhdhv/`、`outputs/csa-hbg-l4-y96qddiu/`、`outputs/csa-hbg-l4-9zolp_hy/` 和 `outputs/csa-hbg-l1-ouk4nqas/`。后续改调度器时先确保业务组没有正确性回归，再将 resident 和 L4 来源门禁从红变绿；不要据此声称 sync_start 已获得硬件覆盖。

## 6. GM Mix 接线后的验收边界

2026-10-25 的原样严格 L4 已通过，结果为 `outputs/csa-hbg-l4-fdakpyh4/result.json`：
79 个有 kernel 任务、2308 条执行（AIC 864／AIV 1444），133 条依赖时序检查，
Host 构图 107/107、无 dropped/truncated，scheduler producer 为 `aicore`。
deps/timing 两轮各七项数值比较均通过；输入、54 个 kernel、52 项 ABI、原 N 和校验标准未变。
两个无 kernel Dummy 不占物理 trace，但仍在逻辑完成与依赖链中。

上述历史结果保留 `block_identity_verified=false`，当时的 schema 没有显式 block_idx/subtask_slot。
当前入口已采集显式身份和原生协议事件，严格 L4 的身份与因果审计均通过，
结果同时为 `block_identity_verified=true` 和 `protocol_verified=true`。
仅带显式身份的历史结果不因此获得完整协议证明；Perfetto metadata 保留本轮原生审计摘要。
原有 kernel 依赖时序检查同时保留为额外保守门禁，并报告 `dependency_timing_guard=true`，不能代替协议证明。
首错误身份与退出冷日志已接线；sync_start 和无插桩性能比较仍待完成。所有失败定位和逐步复验仅集中记录于
[实现过程](../../../../tests/fdwic/pa_hbg/spmd/implementation-progress.md)，不修改本页历史基线数值。
