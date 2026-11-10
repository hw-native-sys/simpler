# A5 HBG 异步后端与资源生命周期

[返回总览](../a5-hbg-async-completion.md)。延迟逻辑完成、资源持有、数据可见性是三个相互关联的要求，分别验证，优先复用现有机制。

## 1. 条件观察

COUNTER 为 INT32 单调通知，expected 在 `[0, INT32_MAX]`，比较遵循有符号 `value >= expected`。负的当前值（包括 `INT32_MIN`、`-1`）表示尚未满足，返回 PENDING；不得当作无符号大值提前完成，也不得把合法负初值当作坏记录。非法 expected、类型或对齐仍拒绝。观察使用合法对齐的精确 32-bit device load，不清零通知，不用 atomic-add-zero 读取外部地址。一次检查返回 READY/PENDING/FAILED，内部不循环等待。

SDMA 使用当前 SDK 公共完成记录 API 返回的 post-done 地址和完整 post id。复用已有 session、postId 延续及内部槽复用保护，不解析私有 SQ 布局、不自行清零共享完成记录。workspace 完整范围采用公共 `kSdmaWorkspaceBytes`，但 allocation 范围不代表其中每个地址都是完成记录。

提交资格与观察资格分开验证。参与调度的 AIC/AIV 均须能观察支持的条件；不能把仅在 AIV 上可用的提交 helper 原样搬到 AIC。编译与反汇编只证明指令形态，真实可见性仍需硬件用例。

## 2. 提交与登记

SDMA adapter 保持现有调用接口。提交前检查形状、字节数和明确的零工作，并为可能产生的完成记录预留足够容量；提交后使用公共记录数量/索引 API 登记实际条件，归还未用预留。非空 TGET 得到零 handle、未知完成类型、登记溢出或 helper 失败必须锁存错误，不把它们当成零登记成功。

A5 SDK 的 TPUT 使用同步 MTE 回退，搬运完成后合法返回 SDMA 类型的零 handle。adapter 依据实际操作类型接受该结果，将本次预留按零条条件结束，不查询完成记录、不发布 CHECK。`postFailed`、错误引擎和既有首错仍必须失败；不能将 TPUT 的零 handle 规则推广到 TGET，也不能把同步回退表述为异步 SDMA PUT。

SDK 默认以物理 `get_block_idx()` 选择 channel group，同一 cluster 的两个 AIV 共享该值。resident 为每个实际 cluster 保存一个独立 cache line 的提交互斥字；所有 task 和 Graph 实例共享这组 run 内状态。取得互斥权后才构造 session，完成提交、SDK postId/SQ 状态持久化及条件登记后交权。不同 cluster 可以并行，同 cluster 的不同 workspace 保守串行提交。

基础 adapter 使用 SDK 默认的单队列 session；一次非空 TGET 导出一条公共完成记录，同一物理执行可通过多次请求登记多条记录。直接改变共享 workspace 的队列分组不属于该提交互斥保证：多队列 session 必须按实际物理队列的重叠范围保护提交，并保持 workspace 的分组一致，不能仅沿用每 cluster 一个互斥字就宣称支持。

SDK 保存 SQ tail 时既写入 GM，也更新提交核的 scalar cache。交权前必须完成临时 pipeline，并写回这些元数据缓存行；否则旧 owner 的后续 session 缓存维护可能覆盖新 owner 已推进的 SQ 状态。基础 adapter 使用 SDK 同类的全 DCache 写回及设备屏障，随后原子释放互斥字；这不代表 DMA payload 已完成。

互斥仅覆盖提交，不等待本次 DMA 完成。成功提交释放互斥字，异常退出将其置为失败，后续提交不得继续使用可能不一致的 session 状态。争用等待使用设备时钟限制为 5 秒，超时使当前登记失败，不抢占或清除其他核的所有权；核硬故障仍由既有 run 退出流程处理。跨 run 的同一 domain 使用继续依赖既有 worker 执行顺序和 fence，run 内互斥不能替代它们。

defer_wait 不会使提交队列满、上述提交争用、SyncAll 或 kernel 内轮询自动让出；kernel 必须正常返回，调度器才重新取得执行资源。CHECK 本身始终有界且不等待提交互斥字。验收必须包含同一 cluster 双 AIV 连续提交，验证公共完成记录的 postId 不重复及全部数据结果，不能只测试单个提交者。

## 3. 复用已有 owner

| 资源 | 既有持有机制与所需最后访问 |
| ---- | -------------------------- |
| HBG heap、staging、调度 image | run 持有，直到所有逻辑完成、resolve 排空及参与核退休 |
| 调用参数 backing | 已有 source/import 引用及 run 记录，覆盖执行与数据交付 |
| 通信 context/window/workspace | CommDomain 使用登记和 run fence；scalar `device_context()` 同样登记使用 |
| DMA 源 | 引擎最后一次读取完成 |
| DMA 目标 | 最后写入和相关消费者读取完成 |
| signal、完成记录 | 检查者及 backend/peer 都不再访问 |

延迟 TaskDone、后继解析和 resolved_task_count 会自然延长既有 run 持有时间；kernel 返回或单 task 完成不会单独释放 HBG heap。不能把 scalar context 当成无人保活，也不能用一份新身份快照代替实际引用。

`owner_task_id` 表示依赖/生产来源，可用于现有记录索引，不等于 allocator 或通信 owner。Graph 的 PARAM、SUB_TASK 和 view 几何已经提供绑定线索；需要消费范围时从现有表示解析运行内旁表，不修改 TensorData 或 Graph 格式。

## 4. 复用与可见性

generation 只识别旧引用，不能阻止旧 DMA 写入已经复用的地址。资源保持分配也不自动保证 payload/signal 未被覆盖。复用必须等待旧访问和实际消费者结束：run 内由正确的 DAG/后端约束保证，跨 run/peer 由既有 fence 或明确的完成确认保证。

COUNTER 的 `value >= expected` 不区分 payload 的版本，也不能阻止 peer 提前写入下一轮数据。生产者须等待上一轮实际消费者的完成确认，或使用独立缓冲槽；重置通知值也须发生在旧读写和检查都退休之后。这是业务通信协议的复用前提，run/domain 的持有只保证 backing 不被释放。验证重置时使用与通知值独立的 payload 标签，避免再次使用 expected=1 时把旧数据误认作本轮结果。

依赖满足不等于所有核 cache 已更新。优先验证已有 backend 发布、实际消费者读取指令和缓存维护路径。若这些已提供获取保证，不增加多余准备协议；否则只为缺失的读取范围和核增加有界获取。观察核的 invalidate 不能代替消费核的获取。

## 5. 错误和关闭

首错停止新的正常发布和业务准入，但不能假定已提交 DMA/peer 访问随之停止。不能靠清零 pending 或强置计算关闭、目录为空、超时、普通 stream drain 或核 ACK 宣告外部访问结束。

登记溢出、非法完成条件和登记协议错误分别保留既有 `ASYNC_WAIT_OVERFLOW`、`ASYNC_COMPLETION_INVALID`、`ASYNC_REGISTRATION_FAILED` 状态，经 AICPU 映射到现有运行结果；不能统一折叠为参数错误。错误冷路径同时记录名称及任务身份，超时记录 `SCHEDULER_TIMEOUT`。这些诊断不参与正常调度决策。

正常路径复用现有 run/domain 退休。异常路径须核实当前后端究竟提供什么停止或完成保证；仅覆盖其他引擎的 drained 判断不能证明 SDMA 停止。证明不足时复用现有资源隔离与保留机制，禁止提前释放和复用，并返回原错误。

不为本功能预设新的通用关闭回执 ABI。确有现有错误路径无法表达的缺口时，先给出可复现的实际访问问题，再在对应 owner/backend 做局部修复。2/4 rank 验收必须覆盖 peer 最后访问，不能用本地 reset 冒充远端停止证明。
