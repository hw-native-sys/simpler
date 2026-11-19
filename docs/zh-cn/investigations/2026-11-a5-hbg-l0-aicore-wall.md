# A5 HBG L0 AICore wall

## 计时定义

L0 直接记录实际执行的 AICore kernel 首尾，覆盖 AIC、AIV 和 MIX。
每个核保留首次 kernel begin 和最近一次 kernel end，最终归并为：

```text
aicore_wall = max(各核最后一个 kernel end) - min(各核第一个 kernel begin)
```

窗口包含首尾之间的调度空洞和等待。无实际 kernel 的核不参与归并；
全部为空或出现半写、倒置的时间对时，不发布有效 AICore wall。
Percounter 使用相同的实际 kernel 首尾口径及空闲核处理规则；
历史版本的调度循环窗口包含启动和收尾，不能直接与本口径比较。

## 采集与发布

- 首次执行标志和最新结束值保存在调度循环的局部变量中，结束值在退出循环时写回。
- 无法预知哪个 kernel 最后执行，因此每次实际 kernel 返回仍需读取一次设备计数器。
- 每个核通过现有 64 字节 Handshake 发布首尾，完成 DCCI 和写屏障后才确认退出。
- AICPU 在原有退出报告读取处累计分区首尾，最终只合并各分区结果。
  无新增逐 kernel 日志、共享抢占原子或集中重读全体报告。
- 仅在启用 Device phase capture 且归并得到有效 kernel 窗口时，Host 才输出
  `chip.run.runner_run.aicore_wall` STRACE；全空运行不输出该事件。
  `dur` 单位为 ns，并携带 `aicore_start_cycle`、`aicore_end_cycle`、`aicore_cnt_hz`。
  现有 rounds 表可读取 `AICore (us)` / `Avg AICore`。

AICore 计数器与 AICPU 计数器按各自时钟解释。
双卡先分别计算每卡窗口，再逐轮取较慢卡的耗时；不跨卡相减原始计数器。

## 验收方法与结果

下述实测验收只覆盖 HBG。Percounter 的计时边界已统一，
其新增采集开销和当前组合的设备正确性仍需独立验证，不能沿用这些结论。

验收指标是新增采集导致的 L0 **Device wall 平均值**膨胀不超过 1%。
开启、关闭版使用相同 DFX ABI、调度实现、业务 kernel 和输入；关闭版撤回首尾采集，
开启版包含本次合入的退出报告归并改动。两版共用当前工作树的调度编译边界。
不能用 DFX 编译关闭版直接搭配原有 orchestration：两种模式的 `CoreTaskArgs` 布局不同。

固定次序为关—开—开—关、开—关—关—开。每批 5 次预热、10 次测量，
两版各 40 次，全部样本参与平均。每轮保留原输入复位、prefix 及 golden 校验。
本地 Host 测试驱动使用整段文件读取和单字节重复值批量填充，缩短计时外准备；
没有重编业务 kernel，也没有减少测量次数或跳过 golden。

所有设备由 `task-submit --device auto --device-num N` 分配，记录真实 `TASK_DEVICE`。
同时最多使用两张卡。本次验收按用户确定的范围只采用 8 组单卡对照。
双卡测试不再继续，未完成的双卡记录不纳入验收，也不据此声称通信场景满足 1%。

**性能验收通过：完成 8/8 组，已完成用例最大平均膨胀 +0.238%。**

新版 C2A 的同二进制空对照差异为 +0.606%（设备 0）。
负差异不直接解释为优化收益；结论只覆盖本次环境、二进制及输入组合，
不保证任意其他用例或调度实现均满足 1%。单位为 ms，膨胀按
`(开启 / 关闭 - 1) × 100%` 计算。

| Case | Off Device (ms) | On Device (ms) | Change | AICore (ms) | Device |
| ---- | --------------: | -------------: | -----: | ----------: | ------ |
| pypto_lib__c2a_full | 1.519289 | 1.507049 | -0.806% | 1.334781 | 1 |
| pypto_lib_dsl_all_sync64__c2a_full | 1.789331 | 1.782929 | -0.358% | 1.635343 | 1 |
| pypto_lib_dsl__c2a_full | 1.762494 | 1.757994 | -0.255% | 1.607489 | 1 |
| latest__swa | 0.506548 | 0.504574 | -0.390% | 0.367410 | 0 |
| latest__c2a_full | 0.658210 | 0.653590 | -0.702% | 0.482049 | 1 |
| latest__c1a_reindex | 0.619216 | 0.617746 | -0.237% | 0.467590 | 0 |
| latest__c1a_reuse | 0.512819 | 0.514038 | +0.238% | 0.359751 | 0 |
| latest__engram | 1.171659 | 1.164426 | -0.617% | 1.023406 | 0 |

## 正确性检查

- CPU 覆盖跨分区首尾归并、无实际工作的核、半写时间对、全空运行及旧运行状态清理。
- 退出报告归并测试在分区退休后擦除原始报告，确认最终归并使用已保存的分区快照。
- SWA 实机校验中，汇总首尾与 578 次实际 kernel 执行记录的最小 begin、最大 end 完全一致。
- rounds 表的 Python 测试确认 AICore wall 与 Device wall 分列。

本地证据位于 `artifacts/dfx-budget/`，包括运行库构建及 IR 检查、
`l0-wall-register-envelope/`、`l0-wall-fast-paired/`、`l0-wall-fast-aa/`
以及测试驱动来源记录 `fast-host-runner-build/provenance.json`。

## 历史数据及复用边界

原 Host 驱动下，新版 C2A 曾出现 +5.124% 的开关差异，但同二进制也出现 +2.371%，
超出 1% 分辨要求。这些完整原始记录保留在 `l0-wall-register-balanced/`、
`l0-wall-aa-c2a-balanced/`，未删除或择优计入当前结果。
其他早期试验见 `l0-wall-abba/`、`l0-wall-local-abba/`、`l0-wall-retirement-abba/`；
未完成对照不用于验收。`native-endpoint-check/` 等早期双卡记录没有 Device 起跑同步，
仅作为未同步历史保留。

`l0-wall-complete.patch` 汇总了独立的 L0 首尾采集修改，可应用于原基础提交，
不包含通用编译优化；该独立组合只检查了补丁可应用性，没有另行测算性能。
`l0-wall-fast-paired/pypto_lib__moe/` 是已结束的双卡试验，仅保留原始记录，不用于验收。
