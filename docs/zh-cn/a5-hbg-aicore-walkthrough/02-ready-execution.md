# 2. 依赖、有序供给与 kernel 执行

[总索引](README.md) · [上一段：初始化](01-initialization.md) · [下一段：协作与完成](03-cooperative-completion.md)

## 图视图与候选

本核的 SchedulerGraphView 只保存本轮只读地址、extent、数量和版本。
顶层任务从 80 B TaskImageStorage 读取 ID/kernel、参数计数、pool delta 与 predicate；
native child 从 template/instance 获取形状和参数地址。
16 B SchedulerTaskMetadata 为每 expanded node 保存 flags、active mask、kernel IDs、block 数和 trace prefix。

Host 按现有数据和 Graph 控制依赖，以 `(level, -b_level, node_ref)` 排序，再投影为纯 AIC 目录和含 MIX 的 AIV 目录。
每个 logical kernel node 出现一次，目录运行中只读；本核游标在确认当前任务供给关闭后推进。
跨引擎后继领票前确认前驱供给已关闭；领票不等于依赖完成，执行仍等待真实 DONE。
一次查询最多推进八项；本核已持有头部任务时不领该任务第二块，也不越过尚未关闭的头部。

## 从具体 Call 到 fanin 证明

每个真实 block/lane 都有已发布 Call 和 Binding。普通任务通过有效 block ticket
取得一个具体执行 token，验证函数、任务、epoch、block/lane 和 generation。
每核可持有两份普通自取记录和两份按 bank 索引的 MIX 成员记录；取得 Call 责任不等于取得 kernel 执行许可。

```text
查询本核目录游标并确认前驱供给
    -> 取得一个具体 execution 的 Claim
    -> 接收 immutable Call 和稳定 Binding
    -> WAIT_FANIN：按 producer 顺序确认 canonical DONE
    -> 记录正前缀，下次从第一个未证明的 producer 继续
    -> 在原位置标记 READY，再按该位置执行
```

completion_state 的 total/remaining_compute/debt 必须合法。
非零 bypass 提示可以保守地继续等待；DONE 或非法提示必须 atomic 确认。
Graph 外部 fanin 经过 completion_refs，child-local index 加实例 child_begin；root 依赖 ENTER。
不存在每条边 remaining decrement、wake-list 或 per-successor 通知。

自取窗口被等待依赖的项占满时，保留两份自取责任及准备状态；原核在依赖满足后继续执行。
执行责任从首次有效领取到退休不转移，不再发布 PARKING/PARKED 或让其他核重领。
等待期间继续服务独立 async/control 和已有协作责任，MIX 成员使用各自的 bank 记录。

## 本核 slot 与最终门禁

每份执行沿用取得责任时的自取或 MIX 成员位置，直到本核退休。
slot 保存 task/block/lane、prepared handle、predicate 和缓存观察状态，Call args 不随 slot 复用被覆盖。
local ready mask/head 对已准备 slot 维持本地 FIFO；直接执行路径可以跳过一次排队，不能跳过依赖或资源检查。

参数描述符在依赖满足后观察。predicate 支持的形状为单 block/lane；
false 跳过函数，同时保留这次执行必须完成的贡献、trace 和退休责任。
最终 READY→EXECUTING 之前仍检查 epoch、Claim owner、shared admission、MIX/group 与 sync_start 许可。

executor 取得 Call.function 和 Call.args，业务函数的签名为 kernel(args)。
args[48]/args[49] 指向该 execution 稳定的 LocalContext/GlobalContext。
LocalContext 给逻辑 block_idx/block_num，GlobalContext 给物理 AIV lane 及本轮预绑定 async group 地址。
函数不接收本核私有 slot、admission state 或另一个 worker 的局部指针。

## 主循环与计时

主循环交错服务控制 scan、token 依赖、有序供给、cooperative obligations、固定 MIX 组、async service 和本地 slot。
一个候选被阻塞，不能让已有责任失去服务机会；目录空提示也不能代替 run 完成证明。

进入 kernel 前暂存 ready-observe；返回后暂存 kernel start/end。
三个自然对齐 uint64_t 共 24 B，退休时合并到 256 B GM TaskTrace。
GM trace 四行分别保存 publication、kernel timing、dispatch/completion、execution identity。
首错、trace 有效位与业务执行状态分开；核返回函数不自动授权复用 slot 或清退整个 run。

具体布局见[数据布局](../design/a5-hbg-graph-task/data-layout.md)，
有序供给与固定归属合同见[execution token](../design/a5-hbg-graph-task/ready-queue-design.md)。
