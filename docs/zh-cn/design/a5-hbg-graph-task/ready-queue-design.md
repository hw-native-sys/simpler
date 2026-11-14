# A5 HBG 工作目录与 execution token

当前 resident 用不可变 engine catalog 发现工作，由每个 AIC/AIV 管理自己的执行责任和两个 slot。
Graph child 与普通顶层 kernel 使用同一套发现协议。历史共享 ready 队列的测量保存在
[当时的验收报告](../../../../tests/docs/a5-hbg-graph-task/two-ready-queues/README.zh-CN.md)，不作为当前机制的说明。

## 两份不可变目录 {#engine-catalogs}

Host 按 expanded node 的 metadata 分类，纯 AIC 和所有 MIX 放进 AIC catalog，纯 AIV 放进 AIV catalog。
双 AIV MIX 仍由 AIC coordinator 发现；实际 active mask 和 resource mask 由 MIX 协议解释。
每个 executable logical node 在其目录中出现一次，目录项不按 block/lane 放大。
ENTER、EXIT、inline node 单独计为控制节点，不进入这两份目录。

`SchedulerTaskCatalogHeader`、两份 descriptor 和紧凑 ref 数组在 launch 前发布，运行中只读。
每核有自己的扫描游标，目录没有 shared head、enqueue、出队顺序或跨独立任务的 FIFO 保证。
一次发现切片最多检查八项；未发现 runnable 工作也必须让出本轮服务机会，继续处理控制和 cooperative/async 责任。

## Call 与执行责任

Host 预构造每个真实执行的 immutable Call 和 invocation Binding；kernel 参数向量不保存在可覆盖的核内 slot 中。
一个任务可有多个 block/lane Call，每个 Call 配一个 128 B 原子 control sector。
phase、owner 和 generation 同处一个原子字，避免分别观察三个字段而拼出不存在的所有权。

普通执行通过 block ticket 与 Call 的 BUILT→CLAIMED CAS 取得唯一 execution token，
然后验证本轮 epoch、任务身份、block/lane 和函数地址。取得 token 不等于允许读取业务输入。
每核最多保留四个 token，fanin 观察规则见[canonical gate](dependency-counters.md)。

```text
发现一个目录项
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

## 四个 token 与两个 slot

两种容量表达不同资源：token 可以保存等待依赖的执行责任，slot 保存即将执行的准备状态。
每核两个 40 B slot，只被本核准备和执行；local ready mask/head 对已准备 slot 维持本地顺序。
直接执行路径可省去一次本地排队，不能省掉 fanin 或最终资源门禁。

token 窗口被等待依赖的执行占满时，发现一个可运行替代者后可将旧 token
经 PARKING→PARKED 发布。其他核通过有界 probe 与周期 sweep 重新领取；
parked-count 只是提示，不能取消权威 sweep。owner/generation CAS 防止重复领取或旧 owner 归还。

## 前进性与清退

主循环交错执行 token 观察、控制 scan、cooperative discovery、MIX/sync_start、async service 和本地 slot 执行。
一个 pending token 或 group reservation 不能让核停止服务已承担的责任。
不能用目录空提示或某次 scan 未找到 ready 项证明图已结束。

kernel 返回后 seal async，真实计算贡献与 async debt 按 canonical gate 结算。
先取得本轮的 kernel/trace/resource owner 退休证明，再复用 slot；最终仍经过 EXIT、ACK 和 post-close release。
错误时取消尚未执行的责任，在途执行按失败清退协议收敛，不能伪造正常 DONE。

实现入口为 `scheduler_plan_task_catalog`、`scheduler_refill_execution_tokens`、
`scheduler_park_prepared_execution`、`scheduler_service_controls` 和 `scheduler_fill_own_slots`。
缓存维护、参数所有权及 MIX replacement 的约束见
[build/execute 合同](build-execute-separation.md)，运行全流程见[协议](protocol.md)。
