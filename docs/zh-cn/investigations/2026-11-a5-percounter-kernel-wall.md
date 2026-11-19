# A5 Percounter L0 kernel 首尾性能

本记录来自 fwic-aicore session：实现提交 `fd4d4863a`，完整测量记录提交 `150ad0ebb`。
后续实现或主线同步不能直接沿用此表作为重新测量结果。文末原始产物路径位于该 session 的独立工作区。

## 口径与实现

`AICore wall = max(各核最后一个实际 kernel end) - min(各核第一个实际 kernel begin)`。
此窗口包含 kernel 首尾之间的调度、等待和通信空洞，不含首个 kernel 之前的建图、
启动准备，也不含最后一个 kernel 之后的退出和日志发布。未执行 kernel 的核不参与归并。
依赖占位任务、空 callable 和 predicate 跳过的任务不改变窗口。

percounter 的旧窗口覆盖整个 lane 循环及日志发布，不能作为同口径的 AICore wall 比较。
新实现复用已有首尾报告和 AICPU 退出归并：首个实际 kernel 读取 begin，每次返回更新本地 end，
最后通过 Handshake 发布。L1–L4 的任务记录与窗口复用相同时间戳。

## 测试方式

- 保留全部 16 个配置，复用原业务二进制和输入；仅重编 percounter 运行库。
- L0 两版使用相同 DFX ABI、调度代码和输入，区别为旧 lane 窗口与新 kernel 首尾采集。
- 每批 5 次预热、10 次正式测量，固定关—开—开—关、开—关—关—开次序。
  每版各 40 次，全部样本取平均；Device wall 和 AICore wall 来自同轮 L0。
- 每轮保留输入复位、prefix 与 golden。新版 MoE 保留原生 Host `get_tensor_data`
  路径以及 0/6、6/0、72/72 三组正确性检查。
- 双卡复用 fwic-graph 的 Device 起跑同步实现。两版均在同一 Device stream 上，
  先完成双方 epoch rendezvous，再启动被测运行；同步不计入 Device wall。
  校验每轮双方 epoch 和 gate end 不晚于本卡 kernel begin。
- 双卡分别计算各卡时长，再逐轮取两个 rank 的较大值，最后平均；不跨卡相减原始计数器。
- 所有任务通过 `task-submit --device auto --device-num N` 分配，记录实际 `TASK_DEVICE`。
  按用户授权同时最多占用四张卡，两个双卡任务并行。

仅对发生 507033 且尚无 Device wall 样本的设备初始化失败，允许在同批次、同分配设备上重试；
保留失败目录及日志，不对性能数值或 golden 失败择优重跑。

## 结果

已完成 16/16 组；表内均为平均值。Device wall 与 AICore wall 来自同轮 L0。

开关对照的 Device wall 均值差异最大为 +3.365%。
15/16 组平均膨胀低于 1%；新版原生 MoE 为 +3.365%，未满足此门槛。
该项保留实测值，不将开关差异全部解释为采集自身成本，也不宣称全量通过 1% 验收。

| Case | Implementation | Device wall (ms) | AICore wall (ms) | Device |
| ---- | -------------- | ---------------: | ---------------: | ------ |
| swa | 新版 PyPTO-lib | 0.345225 | 0.235504 | 0 |
| c2a_full | 初版 PyPTO-lib | 1.198839 | 1.089796 | 1 |
| c2a_full | DSL 半同步 | 1.804725 | 1.717217 | 1 |
| c2a_full | DSL 全同步≤64 | 2.244259 | 2.148534 | 1 |
| c2a_full | 新版 PyPTO-lib | 0.417736 | 0.303815 | 0 |
| c1a_reindex | 新版 PyPTO-lib | 0.408711 | 0.293938 | 0 |
| c1a_reuse | 新版 PyPTO-lib | 0.347321 | 0.234839 | 0 |
| moe | 初版 PyPTO-lib | 3.054713 | 2.977887 | 0,1 |
| moe | DSL 半同步 | 1.932475 | 1.846731 | 2,3 |
| moe | DSL 全同步≤64 | 1.990939 | 1.896224 | 0,1 |
| moe | 新版 PyPTO-lib | 0.955222 | 0.878857 | 2,3 |
| engram | 新版 PyPTO-lib | 1.037688 | 0.952684 | 1 |
| layers_19_21 | 初版 PyPTO-lib | 15.505018 | 15.422858 | 0,1 |
| layers_19_21 | DSL 半同步 | 10.848655 | 10.759697 | 0,1 |
| layers_19_21 | DSL 全同步≤64 | 12.132530 | 12.042482 | 2,3 |
| layers_19_21 | 新版 PyPTO-lib | 4.653437 | 4.573585 | 2,3 |

## 验证与证据

CPU 测试覆盖首次 begin、后续 end、空任务和 predicate 跳过以及任务记录与窗口的时间戳一致性。
两组 percounter C++ 测试通过；设备 IR 检查 6/6 编译单元通过。
SWA L1 实机的 578 次 kernel 记录，其最早 begin 与最晚 end 和汇总值完全相等。

本地原始记录：`artifacts/dfx-budget/percounter-wall-paired/`；
首尾校验：`percounter-wall-kernel-envelope/`；运行库：`percounter-wall-off/`、
`percounter-wall-kernel/`；同步实现及二进制校验：`percounter-wall-gate-{off,on}/`。
早期未做 Device 起跑同步的双卡记录不纳入本次结果。

未完成的早期尝试保存在 `percounter-wall-incomplete-history/`，不与完整对照混算。
