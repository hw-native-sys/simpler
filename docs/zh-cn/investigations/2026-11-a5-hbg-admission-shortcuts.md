# A5 HBG admission 批量发布与 claim seal 信号替换实验

**日期**：2026-11-10
**结论**：两种候选均已撤回；不能作为已验证的性能优化。

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

后续只以开启 L1 profiling 时的 AICPU 设备端起止时间验收，包含业务 kernel 前后的调度开销；
每轮取两 rank 耗时的最大值，再取十轮中位数，预热一次。必须明确时间戳覆盖的入口与退出边界；
现有 `AicpuPhase::RunWall` 包围 `aicpu_execute()`，不含最外层入口门控和返回包装。
分别报告九个 PyPTO-lib、九个 DSL 及全部十八项中位数之和，不把 host 时间、不同编译配置、
首末业务 kernel 窗口或跨核 phase 累计值混为完整 AICPU 耗时。
