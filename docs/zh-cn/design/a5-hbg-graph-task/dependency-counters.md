# A5 HBG canonical completion gate

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前 DONE 使用 CntNotify SRAM，其他拟修改后端分别标明。

默认 resident 使用 producer 的 canonical gate 裁定逻辑完成，由唯一完成者发布 CntNotify SRAM 通知。
Host 发布完整 fanin 视图；设备不会为每条边维护 remaining 或挂接 wake-list，多个后继共享一份非消费式 DONE 通知。
本页描述当前实现，配套的工作发现见[目录与 execution token](ready-queue-design.md)。

## 一个完成事实

每个 expanded node 有一个 `SchedulerTaskControl::completion_state`：

```text
bits 63:48 = total：真实 block × active lane 的总贡献数 T
bits 47:32 = remaining_compute：尚未返回的真实执行数 C
bits 31:0  = debt：compute debt 加尚未结清的 async debt D
initial(T) = (T, T, T)
DONE       = total 合法且 C == 0 且 D == 0
```

T 为非零且不超过 INT16_MAX。每个真实 kernel 返回或合法 predicate skip
贡献一次 RMW，同时减去一个 computation 和一份 compute debt。
未执行 kernel 的 MIX coordinator/resource lane 不贡献完成数。
异步条件以额外 debt 阻止提前 DONE；最后一次真实返回可以只关闭 computation join，
由唯一 async FINISH 在条件满足后结清剩余 debt。

后继用可靠 `ld_dev.b32` 读取 CntNotify SET 区域，0 为 PENDING、1 为 DONE。SDK 资源不足的节点保留 canonical GM gate 读取，CPU/sim 也采用 GM；两者均不经 atomic fetch-add(0) 重复确认。
单调 DONE 可以按本轮 task 身份私有复用；未完成值只说明这次观察未就绪，不能永久缓存。
完成贡献仍需 RMW 合并多核更新，不能改为各核读后 `st_dev` 覆盖。
生产者结果发布到完成贡献、消费者观察 DONE 到读取业务输入的顺序均须显式 DSB；
atomic 本身不提供这些跨操作顺序，DSB 也不代替实际消费者的缓存获取。
原子完成字按 128B L2 侧粒度隔离，所在区域没有 cached 整行写回。

多个后继核观察同一 producer 的 SRAM 通知，不再与该 producer 的 GM 完成贡献争用同一地址。
每次未就绪即返回有界推进循环，不新增紧密死循环；第一次绑定前驱时保存通知地址，
正向证明仍可私有复用。STARS 不替代多写者完成计数或 task claim；本轮也不使用无返回值的硬件 ADD。
每轮 AICPU 在 READY 之前复位，资源随 retained publication 持有。详见
[CntNotify 生命周期与回退](build-execute-separation.md#cntnotify-done)。

## 依赖访问与有界观察

顶层 fanin 从紧凑镜像读取。native Graph 的外部依赖经过 `completion_refs` 映射；
child 的 Graph-local index 映射到 `child_begin + local`，root 依赖 ENTER，EXIT 依赖所有 sink。
引用必须合法并满足 producer_ref < consumer_ref；损坏的 ref/format 进入首错处理。

普通 execution token 可以在取得 Call 责任后等待 fanin。token 保存已经确认 DONE 的正前缀，
下次只从尚未证明的第一个 producer 继续；不能把一个 pending 提示缓存为完成事实。
fanin 完成后，token 才转入私有 READY slot。最终 admission、epoch 和资源门禁仍在 kernel 前检查。

ENTER、EXIT 和 inline 控制节点不占 kernel slot。每次控制切片最多检查八个节点，
逐个观察 producer gate。全部 DONE 后，唯一 CAS 将该控制节点从初态改成完成态，
只有胜者发布 DONE 通知并累计一份本地 logical completion。其他核及消费者观察同一份非消费式通知；GM 回退节点观察 gate。
控制节点之间不递归传播；下一轮 scan 提供后继进展。

## Graph 完成与资源退休

EXIT 等待全部 sink 的逻辑 DONE，不能用记录顺序最后一个 child 代替 sink 集合。
任一 sink 的 async debt 未清，EXIT 就不能完成；Graph shell 的外部消费者因此继续等待。
实际执行的 kernel、block/lane 返回与资源握手依然由各自 owner 退休。

DONE 是依赖事实；`resolved_task_count` 是批量提交的 run 汇总，二者不可互换。
观察到 DONE 不代表该 producer 的 trace、MIX release 或整个 run 的 EXIT/ACK 已完成。
Host 必须取得本轮的清退证明后才能回收 backing。

实现入口为 `scheduler_observe_task_completion`、`scheduler_execution_token_observe_fanin`、
`scheduler_service_controls`、`scheduler_contribute_subtask` 和 async FINISH。
相关测试覆盖非法 gate、重复贡献、控制 CAS 唯一性、跨核依赖、async waiter 进展及 run 复用。
