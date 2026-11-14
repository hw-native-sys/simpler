# 2. 依赖、目录发现与 kernel 执行

[总索引](README.md) · [上一段：初始化](01-initialization.md) · [下一段：协作与完成](03-cooperative-completion.md)

## 图视图与候选

本核的 SchedulerGraphView 只保存本轮只读地址、extent、数量和版本。
顶层任务从 80 B TaskImageStorage 读取 ID/kernel、参数计数、pool delta 与 predicate；
native child 从 template/instance 获取形状和参数地址。
16 B SchedulerTaskMetadata 为每 expanded node 保存 flags、active mask、kernel IDs、block 数和 trace prefix。

Host 将纯 AIC 和所有 MIX 分入 AIC catalog，纯 AIV 分入 AIV catalog。
每个 logical kernel node 出现一次，目录运行中只读；各核只推进自己的发现游标。
没有 shared head、动态 enqueue 或跨独立任务的 FIFO 保证。
每次最多扫描八项，随后继续其他责任，即使这次目录观察未找到可执行项。

## 从具体 Call 到 fanin 证明

每个真实 block/lane 都有已发布 Call 和 Binding。普通任务通过 block ticket 和 Call control CAS
取得一个具体执行 token，验证函数、任务、epoch、block/lane 和 generation。
每核可持有四个 token；取得 Call 责任不等于取得 kernel 执行许可。

```text
发现 ref
    -> 取得一个具体 execution 的 Claim
    -> 接收 immutable Call 和稳定 Binding
    -> WAIT_FANIN：按 producer 顺序确认 canonical DONE
    -> 记录正前缀，下次从第一个未证明的 producer 继续
    -> 全部依赖满足后才转入本核 READY slot
```

completion_state 的 total/remaining_compute/debt 必须合法。
非零 bypass 提示可以保守地继续等待；DONE 或非法提示必须 atomic 确认。
Graph 外部 fanin 经过 completion_refs，child-local index 加实例 child_begin；root 依赖 ENTER。
不存在每条边 remaining decrement、wake-list 或 per-successor 通知。

token 窗口被等待依赖的项占满时，只有发现 ready replacement 才能将旧项经 PARKING→PARKED 发布。
其他核用有界 probe 和周期 sweep 重领，所有权变更都通过 owner/generation CAS。
parked-count 只是提示，不能禁止权威扫描或用来证明图已结束。

## 本核 slot 与最终门禁

每核两个 40 B dispatch slot，只由本核准备和执行。
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

主循环交错服务控制 scan、token 依赖、目录发现、cooperative obligations、MIX replacement、async service 和本地 slot。
一个候选被阻塞，不能让已有责任失去服务机会；目录空提示也不能代替 run 完成证明。

进入 kernel 前暂存 ready-observe；返回后暂存 kernel start/end。
三个自然对齐 uint64_t 共 24 B，退休时合并到 256 B GM TaskTrace。
GM trace 四行分别保存 publication、kernel timing、dispatch/completion、execution identity。
首错、trace 有效位与业务执行状态分开；核返回函数不自动授权复用 slot 或清退整个 run。

具体布局见[数据布局](../design/a5-hbg-graph-task/data-layout.md)，
可执行的发现/转移合同见[execution token](../design/a5-hbg-graph-task/ready-queue-design.md)。
