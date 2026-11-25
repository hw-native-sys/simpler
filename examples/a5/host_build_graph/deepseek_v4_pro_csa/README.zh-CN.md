# DeepSeek V4 Pro CSA：HBG 独立调度验证用例

本目录包含冻结的 C++ kernel、参数 ABI、数据清单、七项 golden、运行与分析工具，
不依赖 PyPTO、模型仓库或其他 runtime 的用例。首次使用需要单独导入固定数据；
数据、编译产物和泳道图不进入 Git。

当前 A5 `host_build_graph` 使用每核日历执行 CSA，包括普通 C/V、MIX 和多 block
任务。本用例覆盖七项输出校验、每个 block 的角色与执行区间，以及 L4 调度采集。
本 CSA 未设置 `sync_start`，同步启动由独立测试覆盖。

## 1. 冻结负载与入口

负载为 Pro、B=4、S=2、seed=0，`position_ids=[8192,8193,0,1,2,3,3,4]`。
包含 52 个参数、54 个注册 kernel、79 个执行任务、2 个 Dummy、2308 条物理执行
（AIC 864 / AIV 1444），60 个多 block 任务，最大 N=512。

来源为 `hw-native-sys/pypto-lib` 的 `restore/deepseek-v4-pro-a5-decode-csa` 分支，
提交 `d1de019cb5e84bcb26b28d9c53756c8ac0fb78e8`，入口
`models/deepseek_v4_pro/decode_attention_csa.py`。本目录 `manifest.json` 保存来源
及源码、数据摘要；54 个计算 kernel、输入和 golden 保持原样。

| 文件 | 用途 |
| ---- | ---- |
| `test_deepseek_v4_pro_csa.py` | 原生 SceneTestCase，A5 HBG 独立入口 |
| `workload.py`、`kernel_config.py` | kernel 注册、52 项 ABI、冻结源码检查 |
| `kernels/vendor/orchestration/attention_csa_test.cpp` | HBG Host 构图 |
| `kernels/vendor/aic/`、`kernels/vendor/aiv/` | 冻结的原始计算 kernel |
| `fixture.py`、`manifest.json` | 58 个数据文件、55 个 C++ 源文件及 SHA256 校验 |
| `golden.py`、`_validation.py`、`_compressor_reference.py` | 七项输出比较及 CPU 参考 |
| `business_cases.py` | 物理页布局、index scale 写入哨兵变体 |
| `graph_contract.json`、`graph_contract.py` | 原图、依赖、tensor view 与逐 block 执行合同 |
| `run.py`、`analyze_perf.py` | 正确性、性能、L1/L4 与调度来源验证 |

HBG 适配只有三项：构图中的 Tensor 显式使用 `simpler::hbg::Tensor`；
30 处 `set_core_num` 改为 `set_block_num`，表达式和任务不变；
Host 读取的 `hc_attn_scale` 使用 `TensorTransfer.NONE`，它不是设备 kernel 输入。
CPU 测试逆向还原前两项并检查原源码摘要，同时看护唯一的 Host-only 参数。
不得通过减少 block、替换通信 kernel 或缩减计算负载通过测试。

## 2. 环境和数据

以下命令在仓库根目录执行。新工作目录使用自己的环境并构建当前 checkout：

```bash
python3 -m venv --system-site-packages .venv-csa
source .venv-csa/bin/activate
python -m pip install 'torch==2.12.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'scikit-build-core>=0.10' 'nanobind>=2,<3' 'cmake>=3.20' pytest cloudpickle numpy
# 按本机实际路径加载 CANN 的 set_env.sh。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps -e .
```

Torch 仅用于 CPU 数据和 golden；关闭 backend autoload 是为避免系统 Torch NPU
插件版本冲突，不会关闭 simpler 的设备执行。切换提交或修改 runtime/platform C++
后重新构建；runner 会检查本地 checkout、Torch 2.12 和 binding 的构建提交。

数据共 418,374,879 字节，约 399 MiB，包含输入、INOUT 初值和原模型 golden。
从另一份相同快照导入本目录的独立副本，不要求原模型仓库继续存在：

```bash
CSA_HBG=examples/a5/host_build_graph/deepseek_v4_pro_csa
python "$CSA_HBG/fixture.py" /path/to/verified/csa/data
python "$CSA_HBG/fixture.py"
```

导入器逐文件检查大小和 SHA256，拒绝覆盖内容不同的文件。`data/in/` 是输入和初值，
`data/out/` 是 CPU golden，不能用设备输出替换。缺数据时普通 scene pytest 会跳过，
`run.py` 报错；跳过不是验收通过。专项 `test_csa_snapshot.py` 缺数据直接失败。

## 3. 正确性、性能和泳道图

上卡前按仓库流程确认 A5 并预约空闲设备。设置 CPU 线程数后执行：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CSA_HBG=examples/a5/host_build_graph/deepseek_v4_pro_csa

# 三轮，每轮校验全部七项输出。
python "$CSA_HBG/run.py" check -d 0

# 先三轮校验，再独立进程进行 5 轮预热和 100 轮无泳道插桩性能测量。
python "$CSA_HBG/run.py" bench -d 0

# 依赖图与计时分进程采集，两轮均检查 golden。
python "$CSA_HBG/run.py" l1 -d 0
python "$CSA_HBG/run.py" l4 -d 0
```

七项输出为 `x_out`、`kv_cache`、`cmp_kv`、`compress_state`、
`inner_compress_state`、`idx_kv_cache`、`idx_kv_scale`。保留原容差和未映射区域保护。
`cmp_kv` 先独立验证 compressor state，再用已验证的实际 state 计算量化参考值。
性能阶段不重复 golden，由前置三轮验证；每轮恢复相同初值，不是连续自回归 token。

每次命令产生唯一的 `outputs/csa-hbg-*/`，记录 checkout、fixture、构图源码、
runtime 二进制 SHA256、实际命令和日志，最后打印 `result.json`。L1/L4 可由标准
转换器导入 Perfetto，不应覆盖旧采集。独立原始入口仍可使用：

```bash
python "$CSA_HBG/test_deepseek_v4_pro_csa.py" -p a5 -d 0 --manual include --rounds 3
```

严格 L4 校验日历运行的调度来源和记录完整性。AICore 调度记录与 Worker 精确关联，
2308 个 task/block/slot 身份逐项对应原图。依赖边由独立 dep-gen 冻结并校验，
Worker 区间用于检查这些依赖的执行顺序。Host 及调度阶段记录必须完整。
结果包含 `block_identity_verified=true` 和 `scheduler_producer=aicore`。

## 4. 业务变体与回归测试

[业务测试夹具](../../../../tests/support/a5/host_build_graph/csa/business.py) 复用全部原 kernel、
参数和七项比较器。`TestCSABusiness` 的 12 个用例检查业务语义，并要求每次执行均有新的
日历调度模式发布。

物理页反转、循环移动和固定种子打乱会同步修改 pool、block table、slot mapping，
普通 KV 同步修改 `window_swa_indices`，index payload/scale 始终使用同一映射。
写入哨兵将本轮应写的两个 index scale 槽初值置为 2，防止“初值等于 golden”掩盖漏写。
未映射区域、FP8 位模式和合法的 `-Inf` 不变。不能任意重排 batch：快照包含跨请求
物理 state 别名，旧 golden 不支持这种等价变换。

```bash
python -m pytest -q tests/ut/py/a5/test_csa_*.py
```

CPU 测试涵盖删除依赖边、缩减 block、漏掉 Mix lane、损坏 tensor view、错绑身份、
陈旧来源和每轮七个缓冲区恢复。

## 5. 性能结论和口径

最新保留优化的历史无泳道插桩 Device 中位数约 **2083.5975 us（2.084 ms）**。
同卡 A/B/A/B、每臂 200 个样本，相对同轮负 Ready 探测复用前的 2245.264 us
下降 7.20%。完整条件、逐批统计和复跑方法见 [性能说明](https://github.com/hw-native-sys/simpler/blob/1ea06c2aee35993c288887582cb7f802b2724546/tests/fdwic/pa_hbg/spmd/performance.md)。
这不是本次文档整理新测出的收益，也没有达到 1.5 ms。

`device_us` 是完整 `chip.run.runner_run.device_wall` 窗口，不是首末 kernel span；
`host_us` 还包括 Host 构图、数据搬运等，不能当作纯调度开销。泳道插桩有额外成本，
不能拿 L1/L4 窗口代替无插桩性能，也不能与 TMR 的 `effective_us` 直接相减。
Host/device 的 `bounded` 时钟对齐带不确定范围，不能视为精确同钟。

已有 [2026-10-14 基线文件](baseline_20261014.json) 属于旧版本 AICPU 回退的历史资料，
对照日历性能时，使用两次实测，固定输入、kernel、工具链、设备和计时范围，并先完成严格正确性验收。
