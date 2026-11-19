# A5 Graph task 调度与发布协议

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前 DONE 使用 CntNotify SRAM，其他拟修改后端分别标明。

本文描述当前 resident 执行合同。节点、布局和 Host 地址来源见[数据布局](data-layout.md)；
[目录与 token](ready-queue-design.md)说明工作发现，[canonical gate](dependency-counters.md)说明依赖完成。
所有参与者在一次 run 中使用同一 epoch、worker 身份和稳定 backing。

## GM 操作与时序边界

按统一硬件工作模型，`ld_dev/st_dev` 经 L2，atomic 走 HBM；两类请求在 SoC 总线上均不保序。
发布者完成数据写入/缓存维护后，须在通知发布前执行相应 DSB；消费者读到合法通知后，
须在依赖它的数据读取或下一状态操作前完成相应 DSB 与必要缓存获取。
成功 atomic 只证明该次竞争结果，不自动发布此前其他地址的数据。
可合并没有中间依赖的一批操作的屏障，不能将有中间消费者的排序推迟到 run 末尾。

唯一 owner 的通知可按字段合同采用 `st_dev`；多写者 claim、完成汇合和交权仍由 atomic 裁定。
独立竞争单元按 128B 隔离，64B 仅描述 Scalar DCache 维护。
GM 同址读取也有争用，故同一次推进中的合法观察值应带身份传递，不在下层重复读或 FAA(0)确认。
CntNotify DONE 与后续其他 STARS Wait 均须保持本页的 epoch、MIX/sync_start、defer_wait 与退休语义，
加 `nop` 限制空轮询入口带宽；当前 SPR 握手不构成通用 event 分配机制。

## 从设备入口到 launch gate

Host 在 orchestration 完成后发布紧凑任务镜像、Graph readonly 计划、完整 Call/Binding 和控制区。
启动 descriptor 给出 resident mode、worker context base 和可信 extent；AICore 在握手前不访问尚未发布的 scheduler backing。

1. AIC/AIV 报告物理身份与 run epoch。无效 descriptor 进入失败启动清退。
2. AICPU 接收完整的 AIC/两 AIV cluster，校验拓扑及实际执行需求，填入每个 worker 的 context。
3. AICPU 发布 context 地址和 READY。每个 worker acquire 后校验 view、epoch、extent、catalog、prepared layout 和 cluster peers。
4. Worker 核对冻结目录和实际拓扑；每簇 AIV0 在首轮补给前初始化本簇 SSBUF。
5. 各物理 lane 有界地准备首批工作，AIC 和 AIV1 分别报告自己的完成位。
6. AIV0 完成自己的补给并观察到两个伙伴的完成位后，把本簇三个 worker 计入本轮总数。最后到达的簇发布 `bootstrap_complete`，任何首轮准备失败都走错误路径。
7. AICPU 打开唯一 DMB launch gate。该 gate 打开前，已取得 Call 或 READY slot 的核也不能调用 kernel。

READY 是配置发布点。它取代重复的角色标记，不能用零索引表示某个核尚未初始化。
run 复用重新观察当前 epoch 的 context/Call，不沿用前一轮的核内配置。

## Graph 展开与控制进展

shell 在 Host 展开成 ENTER、children、EXIT：外部 fanin 指向 ENTER，root 依赖 ENTER，
child 依赖映射到实例内 ref，EXIT 等待所有 sink，外部消费者把 shell 引用映射到 EXIT。
Graph 运行时不动态增添 topology，也不在设备构建 child 参数。

ENTER、EXIT 和 inline node 没有 business Call、计算成员责任或 kernel 执行记录。
控制切片每次最多检查八个 ref，观察全部 producer 的 canonical gate 后以唯一 CAS 完成节点。
完成者累计 logical completion，消费者后续 scan 观察其 gate；不存在 control inbox 或后继通知链。
即使自取与 MIX 执行记录都在等待，也要继续服务控制切片。

## 参数观察与执行

Host 每个真实 block/lane 生成一个 immutable Call：函数地址、完整 args、任务身份、epoch 和 Binding 地址。
Binding 的 LocalContext 与 GlobalContext 分占 cache line；逻辑 block geometry 和 async group 地址在 launch 前绑定。
默认的一个自取 token 可以提前取得 Call；依赖满足后原地发布最终 payload。

kernel 最终入口前仍需验证自己的 epoch、Claim 固定 owner 和跨来源排序条件及实际资源承诺。
predicate 在支持的单 block/lane 形状下按依赖顺序求值；false 跳过函数并保留正常的贡献与退休责任。
损坏参数、非法 predicate 和不支持的 shape 在 Host 拒绝或进入设备首错处理，不能降级成普通完成。

业务输入的 async 观察发生在实际 consumer，并在依赖满足后执行。
已发布 Call 的接收与业务输入可读性是两个独立的证明，不能从前者推导后者。
完整的 cache 观察与参数唯一写者规则见[build/execute 合同](build-execute-separation.md)。

## MIX 与 sync_start

每簇 AIV0 代领 MIX，发布本代 SSBUF 描述符与单调排序边界。活动 lane 收到对应的任务和 block 后准备自己的 Call，确认前驱逻辑完成并服从较早的冲突责任后发布 ARRIVED。活动成员到齐才发布最终 payload 并运行。返回后各活动 lane 发布 FIN；实际记账成员收齐本代成功 FIN，才提交组的物理信用。失败进入错误清退。

`sync_start` 的每个 block 使用不同执行位置。不可撤销的承诺检查 Host 生成的每 lane 排序阈值和未退休的冲突责任；较早返回的成员可先服务独立工作。实际依赖仍等 TaskDone，描述符复用仍等前代退休。

## 完成、诊断与清退

真实执行返回后 seal async registration，并按固定 owner 退休执行责任。
普通 TICKET_OWNED 记录由 owner 按数据发布合同维护；最后成功返回可复用同次完成贡献证明退休，
不要求额外的 RETURNED 原子。MIX/cohort 仍按其协作状态与 ACK 合同发布返回。
compute return 同时减少 canonical gate 的 computation 和 compute debt；async FINISH 清除额外 debt。
合法零 gate 向消费者证明 DONE；resolved run 汇总可以稍后批量提交。
trace 和 MIX/group retirement 仍由拥有者完成后，slot 才可复用。

每执行的任务 trace 为 256 B：publication、kernel timing、dispatch/completion、execution identity 四行。
核内 staged trace 为三个时间戳、共 24 B。没有 completion inbox ID、旧 refill 或未写入的 resolver timing 行。
有效计时行写回并完成 barrier 后才原子发布 valid；atomic valid 所在行不能被后续 cached 整行回写。

L4 因果诊断统一采用 `hbg-native-protocol`，不按版本号分流。native Graph 额外携带独立的 graph source/node map；
共同记录 worker 身份、真实贡献、Call publication、最终 permit 和 trace retirement。
ready_source 只接受目录发现 0 和 cooperative/group 3，不接受已无生产者的 direct-resolve/refill 事实。
解析器只接受当前格式；历史 JSON 不能用默认值伪装成一份当前协议证明。

AICPU 监督汇总完成、首错、dump service 和退出，不派发 kernel。
首错禁止新的成功执行；未执行责任可取消，在途工作保持失败清退所需所有权。
每个寄存器窗口只有一个清退 owner，缺失报告/ACK 都不是已经退出的证明。
所有参与核完成 EXIT、ACK 及 post-close release 后，Host 才能回收本轮 GM backing。
