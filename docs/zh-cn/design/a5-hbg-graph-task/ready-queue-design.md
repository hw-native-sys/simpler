# A5 HBG 工作目录与 execution token

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前 DONE 使用 CntNotify SRAM，其他拟修改后端分别标明。

当前 resident 按不可变 engine catalog 有序发放工作，由每个 AIC/AIV 管理自己的执行责任和两个 slot。
Graph child 与普通顶层 kernel 使用同一套有序供给协议。历史共享 ready 队列的测量保存在
[当时的验收报告](../../../../tests/docs/a5-hbg-graph-task/two-ready-queues/README.zh-CN.md)，不作为当前机制的说明。

## 两份不可变目录 {#engine-catalogs}

Host 按 expanded node 的 metadata 分类，纯 AIC 和所有 MIX 放进 AIC catalog，纯 AIV 放进 AIV catalog。
双 AIV MIX 仍由 AIC coordinator 发现；实际 active mask 和 resource mask 由 MIX 协议解释。
每个 executable logical node 在其目录中出现一次，目录项不按 block/lane 放大。
ENTER、EXIT、inline node 单独计为控制节点，不进入这两份目录。

Host 按现有数据与 Graph 控制依赖，以 `(level, -b_level, node_ref)` 排序，两个目录保持同一全局序的相对位置。
`SchedulerTaskCatalogHeader`、两份 descriptor 和紧凑 ref 数组在 launch 前发布，运行中只读。
每核保存私有发行游标与当前项的静态信息。当前任务全体有效票发完或合法终结后，依据该 task 的共享供给证明推进私有游标，不再 CAS 争用一个目录共享头部。
同类前驱由已关闭前缀证明供给进度；跨引擎前驱另查实际供给状态。目录顺序不构成完成屏障。
每次最多推进八项；本核同 task 的 block 未退休时，不领第二块，也不绕过未关闭的头部。

## Call 与执行责任

Host 预构造每个真实执行的 immutable Call 和 invocation Binding；kernel 参数向量不保存在可覆盖的核内 slot 中。
一个任务可有多个 block/lane Call，每个 Call 配一个 128 B control sector。
phase、固定 owner 与普通票的 TICKET_OWNED 标记同处 state 字；run epoch 由本次 Call 与稳定 backing 绑定。
普通票记录由 owner 独占维护，协作控制仍使用原子状态转换，两者不能混为一次新的所有权竞争。

普通执行由 `next_block` 的有效 FAA 票取得唯一 execution token，不再追加一次 BUILT→CLAIMED CAS。
固定 owner 发布执行诊断、验证本轮 epoch、任务身份、block/lane 和函数地址，提前完成合法 Binding 准备。
取得 token 不等于允许读取业务输入。
每核最多保留四个 token，fanin 观察规则见[canonical gate](dependency-counters.md)。

```text
查询私有目录位置、当前 task 的共享供给证明及跨引擎前驱供给
  -> 取得具体 block/lane 的执行责任
  -> 观察并校验已发布 Call 与 Binding
  -> WAIT_FANIN：逐步证明所有 producer DONE
  -> READY slot：观察参数描述符、求 predicate
  -> 最终 admission / epoch / MIX / sync_start 门禁
  -> EXECUTING：调用 Call.function(Call.args)
  -> seal async、归还真实贡献、发布 trace、退休资源
```

MIX 和 sync_start 使用各自的 group reservation/admission owner；
它们不因普通 token 的提前领取而改变成员集合或最终执行许可。

两份目录只描述 engine 分类和拓扑顺序，不限制 claim 计数器分地址。
当前不同 task 的 `next_block` 已分开，单个 SPMD 内仍竞争一个地址；后续必要 claim
应按独立票区间分散并保持 128B 对齐及间距，保留不重复、不遗漏和前项全局发完票的证明。
64B DCache 隔离不足以避免 L2 侧争用。分地址不能改变已领取 block 的固定 owner，
也不能用预定执行核、`ld_dev` 后 `st_dev` 或 STARS event 代替 atomic claim。
claim 与后续操作存在时序依赖时须显式 DSB，不能依赖 FAA/CAS 自带发布顺序。

## 四个 token 与两个 slot

两种容量表达不同资源：token 可以保存等待依赖的执行责任，slot 保存即将执行的准备状态。
每核两个 40 B slot，只被本核准备和执行；local ready mask/head 对已准备 slot 维持本地顺序。
直接执行路径可省去一次本地排队，不能省掉 fanin 或最终资源门禁。

token 窗口被等待依赖的执行占满时，保留四份具体责任，由原核在条件满足后继续。
有效执行票不退款、不转交其他核；普通 block 退休后才恢复本核对同 task 的领取资格。
MIX 保留正式组和固定的未来组；未来组等待依赖时不提前占用伙伴执行资源，也不跨簇重领。

## 前进性与清退

主循环交错执行 token 观察、控制 scan、cooperative discovery、MIX/sync_start、async service 和本地 slot 执行。
一个 pending token 或 group reservation 不能让核停止服务已承担的责任。
不能用目录空提示或某次 scan 未找到 ready 项证明图已结束。

kernel 返回后 seal async，真实计算贡献与 async debt 按 canonical gate 结算。
先取得本轮的 kernel/trace/resource owner 退休证明，再复用 slot；最终仍经过 EXIT、ACK 和 post-close release。
错误时取消尚未执行的责任，在途执行按失败清退协议收敛，不能伪造正常 DONE。

实现入口为 `scheduler_plan_task_catalog`、`scheduler_refill_execution_tokens`、
`scheduler_query_issue_head`、`scheduler_service_controls` 和 `scheduler_fill_own_slots`。
缓存维护、参数所有权及 固定 MIX 组 的约束见
[build/execute 合同](build-execute-separation.md)，运行全流程见[协议](protocol.md)。
