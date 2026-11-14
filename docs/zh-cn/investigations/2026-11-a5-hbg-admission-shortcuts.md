# A5 HBG admission 批量发布与 claim seal 信号替换实验

**日期**：2026-11-10
**结论**：下列候选均已撤回；不能作为已验证的性能优化。2026-11-13 补充阶段 CAS 实验。

## 批量发布最多八条 admission

候选一次准备最多八条 cooperative task 的 admission record，再以一次共享锁和一次 tail
发布提交连续前缀，试图减少逐任务提交开销。模型 kernel、图和输入保持一致。

293 项 C++ 检查和真实 A5 decode 数值校验通过。相同设备、相同 dev 构建配置下，
当时记录的 device_wall 中位数为 28.035174 → 27.8586445 ms，下降约 0.63%，
没有建立足够收益。四个生产/测试文件已恢复，未保留该优化。

这些数据采集于性能口径统一之前，**不是当前 device L1 的验收数据**，也不能累加到
后续优化收益。未来若重新尝试，必须使用同一 host 对齐方案，对全部 18 个
`dsv41_scheduler` 用例进行成对 L1 测试。

## 用 st_dev 替换 claim seal 的 atomic 交权

候选试图利用最后一个有效 claimer 的单写者属性，把 `claim_seal` 的 atomic 发布及
读者的强确认改为现有 bypass store/load 封装。dev 编译约 149.7 秒，290 项 C++
检查及八个 A5 功能测试目标通过，但这些结果不足以证明内存协议可靠。

硬件依据来自工作区 `simpler-fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`：

- A5 scalar cache 不具备 CPU 多核 coherent cache 语义；交权必须单独论证。
- 原子控制字与 ordinary store/DCCI 的 payload 必须隔离，不能按单个变量推断整条 cacheline。
- `st_dev` 的 repeated-store 精确终值回退已经在单 AIV 上复现，跨核不是必要条件。
- 单写者、bypass、末尾 DSB，以及业务用例偶尔通过，均不构成新通知协议的充分证据。

该手册的探针工具链为 CANN 9.1；本轮安装目录虽名为 `cann-9.3.0`，实际包版本为
9.2.0。没有当前工具链、同构访问时序的精确探针证明新协议成立。因此撤回两文件修改，
保留原 atomic 发布、强确认及屏障，不把该候选计入收益。没有把已有业务中的其他
`st_dev` 使用一起修改。

只有查清 ISA/实现契约，并用相同芯片、编译器和业务时序的真实硬件压力探针补齐证据后，
才考虑重开；仅凭单写者推理不得重试。

## 证据与测量边界

本地实验根目录为 `logs/ds41-layers19-21-parity-20261109/`：

- `rejected-admission-batch8.patch` 和对应候选构建、功能与计时日志。
- `rejected-claim-seal-signal.patch`、`seal-dev-comparison.json`、`validation-seal-dev/`。
- `suite18/aligned-runner-manifest.json` 及 runner 源码差异记录 host 测量修正；
  Host 在 prepare 后对齐两 rank，再启动 device。此修正降低启动偏斜，不能算调度器收益。

以下是 2026-11-10 当时的口径，已被后续完整 device L0 口径替代：
以开启 L1 profiling 时的 AICPU 设备端起止时间验收，包含业务 kernel 前后的调度开销；
每轮取两 rank 耗时的最大值，再取十轮中位数，预热一次。必须明确时间戳覆盖的入口与退出边界；
现有 `AicpuPhase::RunWall` 包围 `aicpu_execute()`，不含最外层入口门控和返回包装。
分别报告九个 PyPTO-lib、九个 DSL 及全部十八项中位数之和，不把 host 时间、不同编译配置、
首末业务 kernel 窗口或跨核 phase 累计值混为完整 AICPU 耗时。

## admission 阶段 CAS 直接认领（2026-11-13）

SWA/lib 的 L3 关键链中，13 个首发间隙合计 432.792 µs，其中 admission
claim 至日志发布为 128.355 µs，发布至首个 payload 准备为 89.145 µs。
候选针对前一段，将独立 `admission_claim_lock` 改为对 `admission_state`
执行 `INITIALIZED → PREPARED` CAS，以成功返回授予唯一 pending owner，
省去一次解锁。任务状态强确认、bootstrap、ready 领取、全局 admission
顺序、MIX、sync_start 和异步完成条件保持原合同。

阶段状态所在行整体改为 atomic-only；bootstrap 与 ADMITTED 发布也同步
改用 atomic store 和原有屏障，未混用 ordinary/bypass store。移除旧锁
字段时同步更新 DFX producer、reader 和既有用例。业务 kernel 未改。

两版在同一设备分配中逐 case 对照，分别预热一次、计时十次；采用完整
`chip.run.runner_run.device_wall` 均值。EP2 每轮取两个 rank 的最大值。

| 场景 | 全包对照 / ms | 阶段 CAS / ms | 变化 |
| ---- | ------------: | ------------: | ---: |
| 14 组非通信，单卡 device 2 | 20.4873413 | 20.5654060 | +0.3810% |
| 4 组通信，devices 4、5 | 27.7815134 | 28.4255266 | +2.3181% |

全部 18 组、两版数值校验通过，但没有综合性能收益。原 main `2f61d0c6c`
的非通信汇总为 19.9902648 ms；不能改用已回退的全包版替代最终验收基线。

同一 device 2 的另一次 L3 成对采集中，25 次 admission claim 的中位耗时
从 3.337 降至 2.239 µs；t22→t29 的完整间隙却为 29.526→29.603 µs。
这证明局部指令路径确实缩短，但不足以说明完整交接或 L0 获益。L3 含
profiling 扰动，不能将其局部差值乘任务数来预报 L0。

既有 C++ 检查 2036 项、Python 检查 1066 项及三个选定的 A5
Graph/MIX/defer_wait 组合场景通过；14 个 A5 IR 编译单元检查无发现。
额外的既有 `test_single_block_sync_request_preserves_execution[7]` L4
场景在 device 5 出现两个 AIC 已进入 kernel 却未返回，随后 scheduler
timeout。此记录不能证明死锁或 CCEC 编译错误的根因。

同卡全包对照的两次设备快照均显示七个调用全部返回、三个 task 完成、
scheduler error 为零；但 Host finalize 返回 -1000，未走完数值和 DFX
断言。恢复匹配 reader 后仍复现，不能再将该 Host 错误归因于 schema
版本不匹配，也不能将整项对照写成通过。该错误与候选 AIC 未返回应分开
定位。以上检查不构成保留无收益候选的理由，生产改动已全部撤回。

证据位于仓库相对路径 `artifacts/build-execute-separation/`：

- `benchmark18-admission-cas/{results,summary}.json`：全部样本和分组结果。
- `admission-cas-L3-pair/claim-analysis.json`：同卡 L3 claim 与完整间隙。
- `admission-cas-rejected.patch`、`runtime-admission-cas/`：撤回差异和冻结产物。
- `functional-admission-cas-l4/`、`functional-admission-cas-l4-control/`：L4 两版日志。

下一步应依据完整关键路径处理准备、发布、成员接收和动态 block 分配，
不能把本实验包装成已取得收益的优化。
