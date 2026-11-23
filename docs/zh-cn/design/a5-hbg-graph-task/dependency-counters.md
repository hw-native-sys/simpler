# A5 HBG canonical completion gate

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前 DONE 使用 CntNotify SRAM，其他拟修改后端分别标明。

resident 将逻辑完成作为单调依赖事实，使用计数器、SET 通知或 canonical GM gate 表达。
Host 发布完整 fanin 视图；设备不会为每条边维护 remaining 或挂接 wake-list，多个后继共享非消费式观察。
本页描述当前实现，配套的工作发现见[目录与 execution token](ready-queue-design.md)。

## 一个完成事实，三个观察后端

`scheduler_observe_task_completion` 按以下顺序选择已绑定的完成来源：

| 后端 | DONE 判据 | 发布与汇总 |
| ---- | --------- | ---------- |
| 单调 CntNotify 完成计数器 | 读值达到任务的累计 threshold，且未超过整链 limit | P 物理返回与可能异步 lane 的 Q 条件完成使用 INC 写 1；canonical GM 后续按任务一次归档，不阻塞 fanin ready |
| 独占 SET 通知 | 0 为 PENDING，1 为 DONE，其他值非法 | 非计数器任务先通过 GM 完成门合并贡献，再由唯一完成者发布 1 |
| canonical GM gate | completion_state 合法且 computation/debt 均归零 | CPU/sim 使用；可靠 bypass 读取，不以 atomic fetch-add(0) 再确认 |

计数器只沿真实 DAG 祖先链共享，每任务的阈值固定且一轮内始终有效，不给独立任务增加依赖。
显式同步 lane 只贡献 P；可能异步 lane 还需 Q，本次没有登记条件时也必须结清其 Q。
普通空组可在返回时连续发送 P、Q 两条单位 INC。MIX 的每个活动成员成功返回时发送自己的 P；
可能异步的 MIX 成员在空组封口时发送 Q，非空组由最终 CHECK owner 发送 Q。每轮 READY 前清零链计数，资源由 retained publication 保留。

每个 expanded node 仍保留 `SchedulerTaskControl::completion_state`。它在 GM 路径裁定逻辑完成，
在计数器路径保存归档结果：

```text
bits 63:48 = total：真实 block × active lane 的总贡献数 T
bits 47:32 = remaining_compute：尚未返回的真实执行数 C
bits 31:0  = debt：compute debt 加 task 级 has_async 门闩 D
initial(T) = (T, T, T)
DONE       = total 合法且 C == 0 且 D == 0
```

T 为非零且不超过 INT16_MAX。GM 路径的真实返回或合法 predicate skip 通过 RMW 合并贡献，
不允许各核读后再用普通 store 覆盖；未执行 kernel 的资源 lane 不额外贡献。
非空异步组仅由首注册者设置一次 has_async，真实 registered/pending 计数保留在 async join。
最后物理贡献可只关闭 computation join，由唯一 FINISH 在条件满足后清除剩余门闩。
原子完成字按 128 B L2 侧粒度隔离，所在区域没有 cached 整行写回。

多个后继可观察同一完成来源。第一次绑定前驱时保存地址，未就绪即返回有界推进循环；
正向 DONE 可按本轮 task 身份复用，未完成值不能永久缓存。
生产者输出发布、消费者获取及 kernel 入口所需的屏障和缓存接收保留；
计数值或原子操作本身不替代这些跨操作顺序，也不授予 task claim 所有权。
A5 真机在 Host 准备阶段校验全部申请的 CntNotify。资源不足、SET 地址无效或未按 32 B 对齐、INC 地址与 SET + 8 不匹配，均终止本次准备。
详见 [CntNotify 生命周期](build-execute-separation.md#cntnotify-done)。

## 依赖访问与有界观察

顶层 fanin 从紧凑镜像读取。native Graph 的外部依赖经过 `completion_refs` 映射；
child 的 Graph-local index 映射到 `child_begin + local`，root 依赖 ENTER，EXIT 依赖所有 sink。
引用必须合法并满足 producer_ref < consumer_ref；损坏的 ref/format 进入首错处理。

普通 execution token 可以在取得 Call 责任后等待 fanin。token 保存已经确认 DONE 的正前缀，
下次只从尚未证明的第一个 producer 继续；不能把一个 pending 提示缓存为完成事实。
fanin 完成后，token 才转入私有 READY slot。最终 admission、epoch 和资源门禁仍在 kernel 前检查。

ENTER、EXIT 和 inline 控制节点不占 kernel slot。每次控制切片最多检查八个节点，
逐个观察 producer 的完成来源。全部 DONE 后，唯一 CAS 将该控制节点从初态改成完成态，
只有胜者发布 DONE 通知并累计一份本地 logical completion。其他核及消费者观察同一份非消费式通知；GM 回退节点观察 gate。
控制节点之间不递归传播；下一轮 scan 提供后继进展。

## Graph 完成与资源退休

EXIT 等待全部 sink 的逻辑 DONE，不能用记录顺序最后一个 child 代替 sink 集合。
任一 sink 的 async debt 未清，EXIT 就不能完成；Graph shell 的外部消费者因此继续等待。
实际执行的 kernel、block/lane 返回与资源握手依然由各自 owner 退休。

后继观察前驱的 DONE，AICPU 观察所有终点的 DONE。L4 从退休记录统计 `resolved_task_count`；若设备任务所用的任一计数器缺少有效退休快照，该值为 JSON `null`。
CntNotify 路径的 P 可以先于整组 FIN 到齐。常规上板运行中，AIV0 在复用 bank 或退出时检查参与核的本代 FIN。CPU/sim 的 GM 路径由记账成员收齐本代活动成员的成功 FIN 后批量扣减物理贡献。清退还须确认 bank 退休及整个 run 的 EXIT/ACK。
Host 必须取得本轮的清退证明后才能回收 backing。

实现入口为 `scheduler_observe_task_completion`、`scheduler_observe_fanin_prefix`、
`scheduler_progress_graph_controls`、`scheduler_publish_fanout` 和 GM 内联 FINISH。
相关测试覆盖非法 gate、重复贡献、控制 CAS 唯一性、跨核依赖、async waiter 进展及 run 复用。
