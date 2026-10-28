# CSA Ready 记录开销修复与 L1/L4 对照

日期：2026-10-28。基线为 `ded082f20fc88f2e4fc093f7e106c1c50c32666e`。
本次只修正详细泳道图采集对 Ready 发布的扰动，不改变调度协议，也不作为低于 1.5ms 的验收。
完整样本、源码／二进制身份和 kernel 指纹见[对照数据](perf_data/2026-10-28-ready-trace/comparison.json)。

## 修复边界

`scheduler_ready.h` 的 bootstrap append、普通 append 和 direct-resolve 三个路径，
原来在 level≥3 时遍历后继的全部 block 和 active subtask，重复写相同的
`ready_transition_cycles`。普通 append 每次还执行 cache 失效、barrier、写回及 barrier，
全部记录完成后才发布真正的 Ready 队列。t23 的 512 个 block、两个 active subtask
因此在一个 Scalar 上串行写 1024 条记录。

当前 Host 导出、原生协议导出和泳道图 reader 均不读取该字段的值，本次直接删除三处循环，
没有新增 task 级时间戳或事后补造 Ready 时刻。逻辑任务的依赖推进、Ready FIFO、direct candidate、
准入及三 lane 组协议均保持不变。

保留 `SchedulerTaskTrace` 的 384B 布局和该保留字，`SchedulerTaskControl` 仍为 512B，
原生协议仍为 v5。completion 对此 cache line 的维护没有删除：同一条 line 中仍有
`aicore_entry_cycles`、`handshake_publish_cycles`、`register_release_cycles` 等有效字段。
Resolve 本身的起止时间、执行身份和协议事件均保留。

## 同卡实测

使用本机 A5 的 32C/64V、设备 0，Torch 2.12，原样 B4/S2/seed0 CSA。
前后 54 个链接后 kernel 的大小和 FNV1a64 指纹完全一致；每臂 bench/L1/L4 的源码和三件
runtime 二进制身份一致。正式采集期间没有并行编译、测试或设备查询，六次运行的 A5 CLI
竞争监测均为空。无 task-submit，直接使用本机设备；该监测不证明系统级独占。

任务窗口取真实 kernel 的最早开始到最晚结束，不使用包含 local setup 的 Worker View 条。

| 指标 | 修复前 | 修复后 |
| ---- | ------ | ------ |
| L1 kernel 任务窗口 | 2.288619ms | 2.229120ms |
| L4 kernel 任务窗口 | 3.750828ms | 2.799637ms |
| L4 相对 L1 的增幅 | 63.89% | 25.59% |
| 无泳道图完整 Device 中位数 | 2.4156225ms | 2.3883845ms |

每臂无泳道图测量为三轮七项 golden，然后 5 次预热加 100 个正式样本。
Host 中位数为 80.7751695→83.473111ms，不声称 Host 收益。
L1/L4 每个等级每臂各一次独立捕获，不能把上述窗口比值当作统计稳定保证。
level0/1 本来就不执行删除的循环，其小幅变化不能直接记为本次修复收益，
也不能将 L4 节省的 0.951191ms 从无泳道图的约 2.4ms 中扣除。

| L4 区间 | 修复前（us） | 修复后（us） |
| ------- | ------------ | ------------ |
| Resolve(t22) | 433.413 | 0.556 |
| Resolve(t31) | 57.905 | 3.622 |
| t22 最后返回→t23 首次开始 | 467.287 | 38.644 |
| t31 最后返回→t32 首次开始 | 93.697 | 40.272 |
| t56 所有 kernel 前驱最晚返回→首次开始 | 145.911 | 41.613 |

t56 的多个前驱完成顺序会变化：本次 L4 基线最晚返回者为 t52，修复后为 t49，
因此没有固定拿历史 Resolve(t53) 比较，也不单凭编号认定哪个 Resolver 发布了 t56。
修复后 L4 仍比 L1 长，其他详细记录开销不在本次修改范围。

## 正确性与回归

- 新增 195 个参数化 C++ 用例，覆盖三种发布路径及 Mix direct candidate 退回队列，
  包括合法纯 C／纯 V／四种 Mix mask、N=1/2/512、L1/L3/L4 和 inline。
  检查非空完整 trace 区域保持不变，同时验证实际 FIFO 顺序、唯一领取、direct claim 和唤醒计数。
- 修复前新增用例 132 失败、63 通过，既有 completion 的更新断言另有 1 失败；
  修复后全部通过。最终 21 个 C++ 目标共 1316 个 GoogleTest 定义全部通过。
- 协议、执行身份、因果、时钟及泳道图转换相关 Python 测试 3479 项通过。
- HBG 仿真回归：55 项通过、29 项取消选择；并行入口的父 pytest 只汇报取消选择项，
  通过数来自 36 个 standalone、1 个 L3 和 18 个 L2 子用例。
- HBG 真机回归：76 项通过、4 项跳过、4 项取消选择。范围为 A5 HBG 的 examples 和 tests/st，
  排除 level4、manual 和 `resident_failure`；本次未重跑完整故障注入矩阵。
- 原样 CSA 的 L1/L4 均通过七项 golden。严格 L4 保留 79 个 kernel task、2308 个执行身份、
  133 条依赖、46733 个协议事件和 96 个 worker；Host 记录 107/107，无丢失。

日志位于仓库 `outputs/csa-readytrace-*`。初次基线因上次提交后的 Python 扩展版本戳过期而拒绝，
按正常流程重编绑定后重新采集；拒绝的尝试不计入测量。
两次逐参数隔离的真机整体验收在完成前主动中断，均不计入通过数；最终通过的是带
`--runtime host_build_graph -p no:xdist` 的完整重跑，日志为 `outputs/csa-readytrace-a5-runtime.log`。

## 泳道图与复跑

原始记录、Perfetto JSON、每轮样本与完整日志保留在工作区
`reports/csa_poursoul_20261027.Osnaam/artifacts/`，从仓库根目录以 `../reports/` 访问：

- 修复前：`readytrace-baseline-l1/L1.json`、`readytrace-baseline-l4/L4.json`。
- 修复后：`readytrace-after-l1/L1.json`、`readytrace-after-l4/L4.json`。
- 无泳道图：`readytrace-baseline-bench-b/`、`readytrace-after-bench-a/`。

启用项目 venv 和 CANN、重建当前 A5 HBG runtime 后，在仓库根目录复用现有入口：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py l4 -d 0 --require-aicore-scheduler
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py l1 -d 0
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py bench -d 0
```

各运行串行执行，不覆盖已有归档；完整 Device 计时与插桩任务窗口分开报告。
