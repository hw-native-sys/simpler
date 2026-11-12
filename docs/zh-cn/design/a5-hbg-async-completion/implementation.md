# A5 HBG 异步完成实施方案

[返回总览](../a5-hbg-async-completion.md)。实现以可运行的非阻塞等待为首要闭环，复用既有注册、参数、Graph 和资源 owner。

## 1. 修改边界

| 模块 | 职责 |
| ---- | ---- |
| A5 HBG Host planner/runtime maker | 根据现有图预留完整记录容量，只初始化和上传稳定映射、header 及必要控制字前缀 |
| kernel context 与登记 helper | 保留地址式登记接口，写入本执行的稳定记录，检查容量与首错 |
| 执行返回与 completion/MIX 收尾 | 封口登记，释放实际执行资源，汇合计算释放和实际 pending 条件 |
| CHECK | 任意参与核领取条件记录，每次有界检查，保存进度并交回执行机会 |
| FINISH 与 resolver | 唯一发布 TaskDone，分段路由后继，排空发布后贡献 resolved_task_count |
| 已有 run/domain owner | 保持 heap、staging、导入 backing、通信 context/window/workspace 至实际退休 |
| 后端与消费者获取 | 验证实际 SDMA 提交、条件读取和消费者缓存路径，局部修补具体缺口 |

公共 Host 注册、ChipWorker 参数传输、A2/A3、TMR、TensorData 和 Graph Definition 不因 resident defer_wait 增加新协议。共享问题只有在目标用例证明已有路径不足时才单独修复。

Host 临时计划按逻辑 task 保存设备任务描述、条件区起点、每 block 的条件区步长和三个 lane 的容量。规划以 checked arithmetic 计算完整图像边界，初始化时才展开 block/lane，写入连续的固定 group header。临时描述数量随逻辑 task 数增长；设备记录数量仍随物理执行数增长，记录顺序、offset、容量、初始化前缀和上传范围保持既定布局。初始化不能触及尚未公开的 token/进度后缀。

## 2. 首个闭环

1. waiter 通过既有 helper 登记 COUNTER 后正常返回。
2. 返回路径发布稳定 group，释放 slot 及局部 block/MIX 义务；逻辑任务仍未完成。
3. 后提交的同类核 publisher 获得执行机会并更新 signal。
4. CHECK 有界地观察到条件满足，与计算返回义务汇合。
5. FINISH 发布唯一逻辑完成并放行消费者；消费者实际读取通知对应的数据。

首要测试取实际同类执行资源数 C，分别提交 C−1/C/C+1 个 waiter，然后提交 publisher。publisher 不依赖 waiter；消费者依赖 waiter。必须证明 publisher 可运行、未通知时消费者未执行、通知后消费者结果正确。仅有编译、同步回归或 mock 条件测试不算闭环完成。

## 3. 服务与完成顺序

整个 run 尚未发布异步工作时跳过空目录扫描和额外消费者获取；尚未激活的空闲核每 16 个服务机会探测一次，业务返回、本核错误及消费者获取不受此空闲预算延迟。活动开始后，普通业务与控制工作共享服务额度：最多 4 次业务返回后服务一次 CHECK/FINISH；空闲和准入受阻时仍提供控制服务。CHECK/FINISH 轮转，每次最多领取一个记录，基础工作预算为 8。PENDING 或失败 CAS 不计为真实进展，不延长 watchdog。

物理执行返回与逻辑任务完成独立。SPMD 的全部 block、MIX 的全部 active lane 必须按真实释放顺序记账；零登记及合法 skip 也必须封口并释放执行资源。只有非零 group 增加 pending；计算释放全部完成后关闭登记，最后一次合法原子汇合才有 FINISH 激活权。

全部成员均零登记的普通、SPMD、MIX task 沿用既有同步完成和本核后继路径，不发布独立 FINISH 工作。计算释放复用原有汇合，SPMD 必须在 trace 发布和 slot 释放后贡献 completed_blocks，MIX 必须先收齐 release。非 ordered 快路径只能在真实最后计算返回后判断是否曾登记；不能根据先返回成员的零登记或暂时 pending=0 推断整个任务同步。

零登记执行不初始化任何 CHECK/FINISH 后缀状态；首次非零发布必须初始化其全部进度和 lease，目录零位保护未初始化存储。

最终准入和消费者本核获取之后，执行核只解析一次 task/block/lane 到条件组的映射；绑定保存在本次 kernel 调用的局部变量中，返回时直接复用。返回仍核对 EXECUTING 状态、run 与组身份、任务及 block/lane、记录边界、开放预留和重复封口，不依赖 kernel 可改写的上下文来恢复绑定。seal 和启用的参数抓取完成后经过返回门禁，才直接退休本核 slot。

无活动记录的服务片段不领取任何责任，由 resident 主循环继续观察全局错误；本核登记错误和活跃片段不能被此路径跳过。resolver 在片段入口、后继选择及队列发布边界检查全局错误，局部游标转换不重复读取；失败时仍保留已取得但未排空的责任，不贡献成功完成计数。

普通完成、FINISH 与 Graph 控制节点共用一个 resolver 主体。普通完成使用调用内游标与大预算，直接处理到底，保留单 fanin 快路、同类核 direct refill 及 MIX 前缀发布；不会转为独立 FINISH 工作。FINISH 和 Graph 控制服务使用保存游标与小预算，跨核移交不携带栈上 direct candidate。已收集的 ready 前缀必须在后续长 fanin 扫描可能让出之前发布，避免可运行后继被扫描过程扣留。

FINISH 保存完整 resolver 游标及最多 8 项发布批次，跨核恢复不重放已发生的队列发布。终态记录和目录清理完成后，才贡献 run 完成计数；正常退出不能越过尚未收尾的异步记录。

## 4. 完整覆盖顺序

| 验证层 | 必须覆盖 |
| ------ | -------- |
| 基础 COUNTER | 延迟通知、C−1/C/C+1、独立业务推进、无提前消费者、0/1/64 条条件 |
| 执行形状 | 普通 AIC/AIV、SPMD、MIX 3/5/6/7、sync-start、predicate true/skip、Graph 多实例及普通任务混排 |
| 真实后端 | A2/A3 现有通信用例适配 A5 HBG；2/4 rank；COUNTER 与 SDMA 的实际完成记录 |
| 资源与可见性 | run/domain 持有、跨 run 复用、同 run channel/session 并发、实际消费者旧 cache |
| 异常 | 登记溢出、非法记录、提交失败、超时、核退出失败、pending close、DMA 最后访问证明 |
| 回归与性能 | 新增和受影响原有 A5 HBG 用例通过，再执行同机同配置基线性能对照 |

已有 API 或后端已经保证的行为用调用链和实际用例验证；缺口修复限定在对应层，不把所有安全要求合并成新的通用资源协议。

## 5. 交付与诊断

设计、实现与验收保持一致。开发记录和原始日志独立存放，文档不携带过程复述。阶段完成后及时提交、总结并继续后续工作；清理已有历史时按用户要求合并提交，之后按可验证单元组织。

复用已有 DFX 输出和协议观测入口。必要事件关联 run/task/block/lane、登记封口、实际执行释放、条件满足、逻辑完成及 resolve 排空；不在热路径逐次打印 PENDING，也不为了观测扩展用户可见接口。

FINISH 的 TaskDone actor 在既有 `flags` 字段中设置 bit 31，表示延迟完成。cooperative task 的 order 从已发布的 admission 记录读取，不依赖完成核仍持有原 cohort。校验器仅对带此标记的完成事件允许跨核 actor，并检查全部 block 贡献、执行 trace 发布先于逻辑完成以及后继依赖；admission、inline 和 Graph control 不能使用该标记。该诊断不替代外部完成条件和 payload 可见性的功能验证。
