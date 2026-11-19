# A5 HBG 工作目录与 execution token

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前 DONE 使用 CntNotify SRAM，其他拟修改后端分别标明。

当前 resident 按不可变 engine catalog 有序发放工作，由每个 AIC/AIV 管理原位执行记录。
Graph child 与普通顶层 kernel 使用同一套有序供给协议。历史共享 ready 队列的测量保存在
[当时的验收报告](../../../../tests/docs/a5-hbg-graph-task/two-ready-queues/README.zh-CN.md)，不作为当前机制的说明。

## 两份不可变目录 {#engine-catalogs}

Host 按 expanded node 的 metadata 分类，纯 AIC 进入 AIC catalog，纯 AIV 和所有 MIX 进入 AIV catalog。
每簇 AIV0 代领 MIX；AIV1 遇到未关闭的 MIX 头项，须等该项供给关闭后才能领取后面的纯 AIV 工作。
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

普通单块由 `next_block` 的有效 FAA 票确定归属；普通多块通过四个独立的 128 B 分区号段领取，在最终分区关闭时汇总供给进度。
固定 owner 发布执行诊断、验证本轮 epoch、任务身份、block/lane 和函数地址，提前完成合法 Binding 准备。
取得 token 不等于允许读取业务输入。
每核有两个自取位置，同一任务最多占其中一个，fanin 观察规则见[canonical gate](dependency-counters.md)。

```text
查询私有目录位置、当前 task 的共享供给证明及跨引擎前驱供给
  -> 取得具体 block/lane 的执行责任
  -> 观察并校验已发布 Call 与 Binding
  -> WAIT_FANIN：逐步证明所有 producer DONE
  -> 在原记录发布最终 payload、取得入口许可，按实际位置和 epoch 执行
  -> EXECUTING：调用 Call.function(Call.args)
  -> seal async、归还真实贡献、发布 trace、退休资源
```

MIX 和纯核 `sync_start` 共用 Host 的排名约束。MIX 的发布者、活动成员、记账者可以不同，实际承诺和退休按 worker 与本代身份记录。两类来源使用各自的执行记录。详见[协作任务的 Host 准备与 Device 校验](admission-preparation.md)。

## 执行记录与控制进展

每核有两份自取记录，以及按 SSBUF bank 索引的两份 MIX 成员记录。取得 block 或收到本代描述符后可以提前准备 Call；最终 payload、执行、完成与退休沿用同一条来源记录。等待 fanin 或组内伙伴时保留已取得的责任，未满足条件的记录不授予执行许可。一次只有一个 kernel 在本核执行。

供给、依赖和协作成员的进度各有证明。私有游标只在前项供给关闭后继续分发；原子有效票授予唯一 owner；真正的数据后继还须等逻辑 DONE。异步 CHECK、控制节点和簇内已有责任继续按有界服务推进。窗口深度的实测见[调度设计](build-execute-separation.md)，运行全流程见[协议](protocol.md)。
