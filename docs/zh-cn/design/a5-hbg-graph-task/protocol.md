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
4. 所有核都参与 bootstrap，以 worker_index/runtime_worker_count 划分 node 区间。
   不另设 scheduler 角色、索引或数量；诊断 activity 也按 worker 身份寻址。
5. cluster coordinator 清理本轮 signal；各分区初始化必要的 admission 记录。
   `bootstrap_scan_complete` 之前不执行控制完成，也不进入 kernel。
6. scan barrier 后初始化两个 FREE slot，执行有限控制 scan 并准备首批 execution。
   最后一个 worker 发布 `bootstrap_complete`。
7. AICPU 打开唯一 DMB launch gate。该 gate 打开前，已取得 Call 或 READY slot 的核也不能调用 kernel。

READY 是配置发布点。它取代重复的角色标记，不能用零索引表示某个核尚未初始化。
run 复用重新观察当前 epoch 的 context/Call，不沿用前一轮的核内配置。

## Graph 展开与控制进展

shell 在 Host 展开成 ENTER、children、EXIT：外部 fanin 指向 ENTER，root 依赖 ENTER，
child 依赖映射到实例内 ref，EXIT 等待所有 sink，外部消费者把 shell 引用映射到 EXIT。
Graph 运行时不动态增添 topology，也不在设备构建 child 参数。

ENTER、EXIT 和 inline node 没有 business Call、admission membership 或 kernel slot。
控制切片每次最多检查八个 ref，观察全部 producer 的 canonical gate 后以唯一 CAS 完成节点。
完成者累计 logical completion，消费者后续 scan 观察其 gate；不存在 control inbox 或后继通知链。
即使普通 token 和 slot 全满、MIX reservation 正在等待，也要继续服务控制切片。

## 参数观察与执行

Host 每个真实 block/lane 生成一个 immutable Call：函数地址、完整 args、任务身份、epoch 和 Binding 地址。
Binding 的 LocalContext 与 GlobalContext 分占 cache line；逻辑 block geometry 和 async group 地址在 launch 前绑定。
四个普通 token 可以提前取得这些 Call，依赖满足后再转入两个本核 slot。

kernel 最终入口前仍需验证自己的 epoch、Claim 固定 owner 和共享 admission/resource 状态。
predicate 在支持的单 block/lane 形状下按依赖顺序求值；false 跳过函数并保留正常的贡献与退休责任。
损坏参数、非法 predicate 和不支持的 shape 在 Host 拒绝或进入设备首错处理，不能降级成普通完成。

业务输入的 async 观察发生在实际 consumer，并在依赖满足后执行。
已发布 Call 的接收与业务输入可读性是两个独立的证明，不能从前者推导后者。
完整的 cache 观察与参数唯一写者规则见[build/execute 合同](build-execute-separation.md)。

## MIX 与 sync_start

普通 MIX 使用每 cluster 的资源协议。active mask 决定真实 kernel lane，resource mask 还包含 AIC coordinator。
AIC 发现 MIX 并保留对应 block 的 group reservation；AIV 按已经发布的具体 group 参加，不能自由执行其他 cluster 的 slot。
coordinator 写回 descriptor 并 barrier 后发布对应 lane 的 assigned_seq；arrived_seq/released_seq 记录各 lane 的进出责任。
GROUP_RESERVED 可以等待 fanin，等待期间固定保留原组责任，不转移或跨簇重领。
资源等待期间继续处理 async/control 进展，并在执行前重查 固定 owner、fanin 与 admission；原组的执行责任保持不变。

sync_start 使用全 run 的 cooperative admission 顺序与显式成员集合。
一次 cohort 只覆盖一个 logical task 的 blocks，不覆盖整个 Graph，也不给每个 Graph 单独建顺序域。
独立普通执行仍须通过最终 shared admission gate；准备和预留不能提前越过 cohort。
MIX 资源 lane 的 release 负责资源退休，不增加一个未执行 kernel 的物理完成贡献。

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
