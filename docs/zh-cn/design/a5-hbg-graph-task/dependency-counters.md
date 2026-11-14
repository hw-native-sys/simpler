# A5 HBG canonical completion gate

默认 resident 使用 producer 的 canonical gate 证明依赖完成。
Host 发布完整 fanin 视图；设备不会为每条边维护 remaining，也不会挂接 wake-list 或通知后继。
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

`ld_dev` 的非零剩余提示只用于判定本轮仍在等待。任何 DONE 或非法提示都通过
atomic fetch-add(0) 再确认；过期的 bypass 值不能授权消费业务输入。
原子完成字所在区域没有 cached 整行写回。

## 依赖访问与有界观察

顶层 fanin 从紧凑镜像读取。native Graph 的外部依赖经过 `completion_refs` 映射；
child 的 Definition-local index 映射到 `child_begin + local`，root 依赖 ENTER，EXIT 依赖所有 sink。
引用必须合法并满足 producer_ref < consumer_ref；损坏的 ref/format 进入首错处理。

普通 execution token 可以在取得 Call 责任后等待 fanin。token 保存已经确认 DONE 的正前缀，
下次只从尚未证明的第一个 producer 继续；不能把一个 pending 提示缓存为完成事实。
fanin 完成后，token 才转入私有 READY slot。最终 admission、epoch 和资源门禁仍在 kernel 前检查。

ENTER、EXIT 和 inline 控制节点不占 kernel slot。每次控制切片最多检查八个节点，
逐个观察 producer gate。全部 DONE 后，唯一 CAS 将该控制节点从初态改成完成态，
只有胜者累计一份本地 logical completion。其他核及消费者直接观察同一个 gate。
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
