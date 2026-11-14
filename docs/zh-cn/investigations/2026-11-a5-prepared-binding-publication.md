# A5 共享执行包的静态 Binding 写回消减实验

**2026-11-13 原实验、2026-11-14 保留定向接收观察的重测均已撤回：未建立 L0 综合收益。**

## 候选与计量

Build/Execute 分离实现 `aabab11ec` 已由 Host 初始化每个执行单元的
`block_idx/block_num`，执行核仍重复填写这两个字段及 `sub_block_id`，
然后写回 Binding 的两条 cacheline 并执行 DSB。
候选将 `sub_block_id` 也按 logical lane 在 Host 初始化，删除上述三个
device store、两次 Binding DCCI 和随后的 DSB。PreparedCall 的接收刷新、
async 入口、MIX 会合和 trace 发布保持原实现。

性能复用 `dsv41_scheduler`，一次预热、十次计时，L0 指标为
`chip.run.runner_run.device_wall`。attention 使用单卡。
以下是 14 组非通信 case 的均值之和，不是一次 forward 时间：

| 版本 | 汇总 / ms | 相对共享包版本 |
| ---- | --------: | -------------: |
| 冻结 main `2f61d0c6c` | 19.9902648 | — |
| 共享包、定向 DCCI | 20.5123812 | — |
| 静态 Binding 候选 | 20.6135803 | +0.4934% |

候选在 device 1 完成，14 组数值校验通过。共享包对照此前使用 device 3，
SWA/lib 使用 device 0；这些结果没有控制设备与时序差异，不能证明小幅
回退的因果来源，但没有提供保留候选的收益证据。没有采用无变化阈值。

## 功能边界与撤回理由

既有 dispatch、MIX、ordered execution、layout 共 1719 项 C++ 检查通过；
三个选定 A5 Graph/MIX/sync_start/defer_wait 组合用例通过。
A5 IR 检查覆盖 14 个编译单元，零发现；runtime 构建耗时 200.06 秒。

MoE/lib 在 devices 2、3 的首轮出现多个 AIC 同一 PC 的 fixpipe 参数错误
169，随后才发生调度超时；未得到四组通信性能数据，不能将候选写成
18 组功能通过。随后原全包版本在同一对 devices 2、3 完成一次预热和十次
计时，数值校验通过（`static-binding-moe-baseline-control/`）。这一区别仍未
证明错误由 Binding 引起，也不能将候选失败归为环境问题。

源码审查未发现 Host 初始化遗漏或后续覆盖，冻结 Host 二进制也包含静态
字段初始化。但删除 Binding 写回同时失去了其 cacheline 维护；kernel
改为直接首次读取 Host 数据，而包 acquire 只刷新 PreparedCall。此前
PreparedCall 的实测已说明不能只靠启动 DCI 与“地址未复用”推定读取可靠。
这项证据对应 `aabab11ec` 的 Device Build：builder 在 resident 启动后写入
Call 并发布 BUILT，接收方启动时的 DCI 早于这些写入，不能替代交权后的观察。
它不能直接证明“Host 在本轮启动前完整发布、run 内不再改写”的另一种合同失败。
现有失败快照没有 Binding 字段，不能证明失败时 kernel 读到的 block 数。

候选没有建立性能收益，且仍有未归因的功能失败，因此整项撤回。若重开，
应先保留 Binding 定向 cache 观察，单独验证静态字段写入是否可以取消，
再以一致设备和完整 L0 对照评估；不能把 L3 中被删除的区间直接视为收益。

## 本地证据

相对于仓库 `artifacts/build-execute-separation/`：

- `static-binding-rejected.patch`：完整候选差异。
- `benchmark18-static-binding/`：14 组数据、原始日志及 MoE 失败记录。
- `functional-static-binding/result.json`：三个组合功能用例。
- `cpp-static-binding-results.json`、`ir-static-binding.log`：CPU 与编译器检查。
- `runtime-static-binding/` 与 `runtime-targeted-dcci/`：候选和对照产物。

## 2026-11-14：保留 Binding 接收观察后的重测

重测以 `e5693e9ef31` 为基线（运行时生产代码等同 `c1d0f3f8668`）。
该实现的 `scheduler_receive_execution_token` 已对 Call 和 Binding 两条
cacheline 定向 DCCI，并执行尾 DSB 与 compiler barrier；这些操作全部保留。
候选只把固定 `sub_block_id` 前移到 Host，删除设备重复赋值及 slot stage
对应的两次 Binding 写回和尾 DSB。动态 async context、实际消费者的数据观察、
入口 store barrier、完成发布和协作 ACK 不变。普通 AIV 使用 logical subtask，
MIX 的正式成员映射保证 physical lane 与 subtask 一致；没有模型特判。

所有 18 组均使用相同冻结业务 bundle/输入，各臂一次预热、十次 L0 测量。
单卡对照自动分配 device 0，通信对照自动分配 devices 1、2；按 case 交替两臂
先后顺序。以下仍是非通信用例均值之和，不是一个 forward 的耗时：

| 非通信 L0 / ms | 基线 | 候选 | 变化 |
| -------------- | ---: | ---: | ---: |
| PyPTO-lib，7 组 | 9.7314084 | 9.7640963 | +0.336% |
| DSL，7 组 | 10.9355728 | 10.9794412 | +0.401% |
| 全部，14 组 | 20.6669812 | 20.7435375 | +0.370% |

各 case 比值的几何均值变化为 +0.011%。没有建立综合收益，整项撤回。
SWA/lib 改善 3.32%，但 C1A full/lib、C1A full/DSL、C1A reindex/lib
分别回退 4.33%、3.64%、5.92%；不能选择性保留有利样本，也没有应用无变化阈值。

本次所有 18 组数值校验通过，包括 MoE 与第 19–21 层的两种实现，未复现旧实验的
fixpipe 169。新旧基线和接收协议不同，这只能说明当前候选通过了本次覆盖，不能
据此宣布旧失败由 Binding 引起或已经查清根因。
另有 5 个 C++ 套件、60 项 DFX Python 检查及 17 项 A5 Graph/MIX/sync_start/
defer_wait/prepared-call 实机检查通过；2 项 sim-only 检查跳过。

独立 C2A reuse/lib 同卡对照的 L0 为 1277.084 → 1286.353 µs，L3 单次为
1484.162 → 1472.020 µs。另采 L1，筛选所有可追溯计算前驱已在前一 kernel 返回前
结束的跨 task 普通执行：基线 48 个窗口的最小值/中位数是 5.872/7.5485 µs；
候选 57 个窗口是 5.295/6.803 µs。没有 <5 µs 样本。该筛选不证明 token 已预持有
或全部资源/异步许可满足，样本集合也不同，不能代替快入口完整验收或 L0 收益。

A5 构建 338.73 秒。IR 扫描覆盖 15/16 个 TU，零 findings；未解码的 vendor SIMT
wrapper 分配段与 canonical relocations 和前版一致，不能称为完整 clean scan。
数据布局未增加 Call/Binding/Control/SSBUF 字段；Host 每个 execution 新增一次
静态赋值，未单独测量其 CPU 成本。

本地证据位于 `artifacts/execution-token-implementation/`：
`static-binding/experiment.patch`、`static-binding-snapshot/`、
`static-binding-18/{configuration,matrix,comparison}.json`、`static-binding-18/report.md`，
以及 `c2a-reuse-static-binding/`。两版 trace 由各自冻结 source 的 converter 转换。
