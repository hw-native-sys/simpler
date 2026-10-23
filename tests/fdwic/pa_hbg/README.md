# PA HBG：Batch 4

用于观察 A5 Scalar 调度的小规模 Paged Attention 示例。计算规模沿用
`simpler_dev/examples/a5/fully_distributed_within_core/paged_attention_unroll`
的常用 Case1，仅把 `batch` 从 256 改为 **4**。

Python 用例定义完整保存在 [test_pa_hbg.py](test_pa_hbg.py)：包括 `CALLABLE`、
B4 参数、输入参数适配和 golden 校验入口，不导入或继承 `tests.st` 中的测试类。
C++ 编排完整保存在本目录的
[paged_attention_orch.cpp](kernels/orchestration/paged_attention_orch.cpp)，
`CALLABLE` 直接指向此文件，不再使用 `tests/st` 中的编排源码。
QK/SF/PV/UP 计算 kernel 通过源码路径复用当前
[HBG PA unroll kernels](../../st/a5/host_build_graph/paged_attention_unroll/kernels)，
输入生成与 golden 公式复用 `simpler_setup.goldens.paged_attention`。不引入旧版
FDWIC runtime 或旧版 shared/private TensorMap 配置。

## 计算与调度流程

参数为：`batch=4`、`num_heads=16`、`kv_head_num=1`、`head_dim=128`、
`block_size=128`、`context_len=8192`、`max_model_len=32768`、`dtype=bfloat16`。

每个 batch 先通过 Alloc 分配在线 softmax 状态 `mi / li / oi`，再将 64 个 KV
block 作为一个 unroll group，提交以下计算链：

```text
QK（AIC）：query × key，计算 attention score
  └─ SF（AIV）：softmax 准备，得到局部统计量及权重
       └─ PV（AIC）：权重 × value，计算局部输出
            └─ UP（AIV）：结合 SF 的统计量，更新 Alloc 分配的状态并写出最终结果
```

四个 batch 共 4 次 Alloc 和 16 次计算 kernel 提交；不同 batch 的数据互不依赖。
上图描述计算顺序，具体依赖由原 HBG 编排的 tensor 参数和 IN/OUT/INOUT 语义建立，
本示例不重新定义依赖。

HBG 在 host 侧构建任务图和 TensorMap，再交给当前 runtime 执行。该用例提交的
是普通单 block AIC/AIV 任务，属于当前 A5 resident Scalar scheduler 支持的范围，
不是多核重复 replay 编排的旧版 FDWIC 实现。可用三级泳道图中的
`scheduler_tasks.producer` 检查实际调度来源是否为 `aicore`。

## 运行

从仓库根目录执行，使用已安装的当前 runtime。下面的命令会自动编译并加载本目录的
`paged_attention_orch.cpp`，构建 B4 任务图并执行，无需手动编译 CPP 或重建 runtime。
CPP 导出的是 HBG 编排入口 `aicpu_orchestration_entry`，运行在 host 侧；该名字是沿用
的入口符号，并不表示在 AICPU 上编排。它不是带 `main()` 的独立可执行程序。

```bash
source .venv/bin/activate

# A5 真机：运行并与 PyTorch golden 比较。
python tests/fdwic/pa_hbg/test_pa_hbg.py -p a5 -d 0

# A5 真机：额外采集调度泳道图和依赖图。
python tests/fdwic/pa_hbg/test_pa_hbg.py -p a5 -d 0 --enable-chip-swimlane 3 --enable-dep-gen

# CPU 功能模拟；模拟器时间不能用作 A5 性能数据。
python tests/fdwic/pa_hbg/test_pa_hbg.py -p a5sim
```

也可以使用 pytest：

```bash
python -m pytest tests/fdwic/pa_hbg/test_pa_hbg.py --platform a5 --device 0 --pto-session-timeout 1200
```

默认启用数值校验，容差与原用例一致：`rtol=atol=1e-3`。诊断文件由统一测试框架
写到仓库根目录的 `outputs/`，本目录不保存测试生成物。这个 B4 用例用于验证功能
和观察调度流程，不替代 B256 的性能基线。

## SPMD 设计参考

[A5 HBG SPMD 设计实现参考](a5-hbg-spmd-design.md)复用完成驱动的依赖推进，
运行时统一确定参与者顺序，非参与核不受约束，本核完成后允许跨 task 重叠。
覆盖动态领取建议、AIC 有限协调的 GM Mix、完成聚合和 corner case；先验收原样 CSA，再开放 sync_start。
这是设计参考，不改变当前 B4 用例或 runtime 的已实现能力。

准备开发时看 [SPMD 编码契约](spmd/implementation-contract.md) 和
[测试入口及红绿状态](spmd/test-plan-and-status.md)。新增的 SPMD 验收位于 `spmd/`，
包含故意保留的未实现项红灯；只跑 B4 时请使用上面的精确文件路径，不递归执行整个目录。

针对 `examples/a5/cla-swimlane.json` 另有 [CLA 形状驱动的 HBG TDD 用例](spmd/test_cla_hbg.py)：
保留原图的多 block、Dummy、双分支和八路汇合结构；运行与红绿标准见上面的测试说明。

真实 CSA 业务验收使用 [csa/test_csa_business.py](csa/test_csa_business.py)，直接复用
[HBG CSA 的原 kernel、输入和七项 golden](../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md)。
业务基线与 resident 红灯分开选取，不能用 CLA 探针或 CPU 模型替代这里的 A5 业务验证。

## 验证记录（2026-10-15）

- 使用本目录独立 CPP 的 `a5sim` standalone：开启三级泳道图和依赖采集，
  golden 校验通过；编排源码与原 HBG 版本内容一致。
- `a5sim` pytest：不开泳道图，`1 passed`。
- Python 用例不导入任何 `tests.st` 模块；编排路径指向本目录独立 CPP，
  四个计算 kernel 的源码路径有效。
- 依赖图中 QK/SF/PV/UP 各 4 个、均为单 block；20 条依赖边与原编排一致，
  没有跨 batch 依赖。
- 模拟泳道图中 `scheduler_tasks.producer=aicore`，16 条计算任务记录齐全，
  scheduler stream 没有丢弃或截断记录；生成物保留在
  `outputs/TestPaHbgB4_B4_20261015_082214/`。
- Ruff 和 clang-format 检查通过。未做 A5 真机验证：当前环境缺少 `npu-smi`，
  `onboard-arch-precheck` 无法确认设备型号；本记录不提供硬件性能结论。
