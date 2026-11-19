# 3. MIX、sync_start、完成与退出

[总索引](README.md) · [上一段：发现与执行](02-ready-execution.md) · [下一段：Graph 与 async](04-native-async.md)

## 资源参与与实际贡献

普通 AIC/AIV 可以有多 block 并分 wave 执行；block 数是逻辑 grid，不能拿物理 cluster 数替代。
MIX 的 resource mask 与 active mask 相同，只有掩码内的 lane 执行函数并报告 FIN。
例如双 AIV MIX 由 AIV0 发布描述符，AIV0 和 AIV1 各执行一次，AIC 不参与。
在 AIC+AIV1 的组合中，AIV0 仍负责发布，计算与 FIN 由 AIC 和 AIV1 承担。

## MIX 和 sync_start 的所有权

每簇的 AIV0 领取一个 MIX block，将 task、block、rank 和参与掩码发布到本簇的 SSBUF。
两个 bank 都被占用时，代理在本核保留一份已分发的待发布组。
成员两次读到相同的有效代后，按掩码接收自己的 Call，并保留本代直到完成。
成员通过依赖、顺序及启动许可后进入 kernel。记账者收齐本代所有参与成员的 FIN，代理才可复用这个 bank。
代理分别保留已领取组、两个 bank 最近发布的组及发布前的取消记录。各执行核分别记录已接收、进入 kernel 和物理返回，支持定位发布前取消及 kernel 挂起。

sync_start 为一个任务的 blocks 建立同驻 cohort，使用全 run 的 ordered cooperative admission。
metadata 不决定运行时 order key；Graph 内外共享同一顺序域。
普通执行也必须通过最终 admission gate，提前接收 Call 不会提前获得进入 cohort 的许可。
每个成员只在其真实 group、block 和 lane 下承担义务；busy 核不能为后台发现丢弃已接手的义务。

## Canonical 完成与诊断退休

每 node 的 completion_state 由 total T、remaining computation C 和 debt D 组成：
初态为 (T,T,T)，每个真实执行返回同时减少 C 和 compute debt。
async 注册保留额外 debt，直到条件满足后的唯一 FINISH 结清；只有合法 C=D=0 才是 DONE。
最后的计算返回可能只关闭 computation join，并由后续 FINISH 完成逻辑结算。

完成路径的顺序是：seal async registration，归还 Claim 的真实返回责任，
结算 canonical contribution，整理并发布 trace，退休 group/slot 资源。
trace.valid 在有效计时行 writeback 与 barrier 后原子发布，不能被之后的 cached 整行回写覆盖。
同一个核持有 slot 直到这些退休步骤结束；LocalContext/GlobalContext 及 args 仍属于稳定的 execution Binding。

DONE 与 run 汇总不同。消费者直接观察 producer gate，
每个逻辑任务只由最终记账者报告一次完成。AICPU 监督本轮计数，全核退休后将其写入 resolved_task_count。
控制节点通过独立有界 scan、fanin proof 和唯一 CAS 完成，没有后继通知或同步 fanout walk。
逻辑依赖完成也不代替 MIX lane release、trace retirement 或全核清退。

## 完成监督与双向清退

AICPU 监督本轮预期/汇总数量、首错和 dump service；不再逐任务派发。
首错禁止新成功执行，未执行责任可以取消，在途 kernel 的真实资源责任仍必须退休。
失败 terminal 同时保留首个执行错误与清退结果，不能拿后来的关闭失败覆盖首错。

清退按寄存器窗口的唯一 owner 分区发送 EXIT，核观察后发布 nonce/ACK，
AICPU 等待本轮 ACK 并完成 teardown，随后开放 post-close release。
核看到 release 后才返回。缺失报告或缺失 ACK 都不是已经退出的证明。
本轮确实初始化的 PMU/DFX 状态才由该 run 结算；错误启动不访问未建立的诊断所有权。

最终的清退证明决定 backing 能否归还池，不能只检查 DONE 或 resolved_task_count。
若证明失败，poison 并保留借用存储，阻止后续 epoch 覆盖它。
