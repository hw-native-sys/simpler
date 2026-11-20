# A5 HBG 异步完成实施方案

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](../a5-hbg-graph-task/build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；当前实现、拟修改布局和未来 STARS 后端分别标明。

[返回总览](../a5-hbg-async-completion.md)。实现以可运行的非阻塞等待为首要闭环，复用既有注册、参数、Graph 和资源 owner。

## 1. 修改边界

| 模块 | 职责 |
| ---- | ---- |
| A5 HBG Host planner/runtime maker | 根据现有图预留完整记录容量，只初始化和上传稳定映射、header 及必要控制字前缀 |
| kernel context 与登记 helper | 保留地址式登记接口，写入本执行的稳定记录，检查容量与首错 |
| 执行返回与 completion/MIX 收尾 | 封口登记，贡献物理子任务完成，分别汇合计算与实际 pending 条件并收尾执行资源 |
| CHECK | 任意参与核领取条件记录，每次有界检查，保存进度并交回执行机会 |
| 异步完成 | GM join 的唯一 owner 内联发布 DONE；CntNotify 路径由最终 CHECK owner 发布非空组的条件贡献 Q |
| 已有 run/domain owner | 保持 heap、staging、导入 backing、通信 context/window/workspace 至实际退休 |
| 后端与消费者获取 | 验证实际 SDMA 提交、条件读取和消费者缓存路径，局部修补具体缺口 |

公共 Host 注册、ChipWorker 参数传输、A2/A3、TMR、TensorData 和 Graph Definition 不因 resident defer_wait 增加新协议。共享问题只有在目标用例证明已有路径不足时才单独修复。

Host 临时计划按逻辑 task 保存设备任务描述、条件区起点、每 block 的条件区步长和三个 lane 的容量。规划以 checked arithmetic 计算完整图像边界，初始化时才展开 block/lane，写入连续的固定 group header。临时描述数量随逻辑 task 数增长；设备记录数量仍随物理执行数增长，记录顺序、offset、容量、初始化前缀和上传范围保持既定布局。初始化不能触及尚未公开的 token/进度后缀。

## 2. 首个闭环

1. waiter 通过既有 helper 登记 COUNTER 后正常返回。
2. 返回路径发布稳定 group，释放 slot 及局部 block/MIX 义务；逻辑任务仍未完成。
3. 后提交的同类核 publisher 获得执行机会并更新 signal。
4. CHECK 有界地观察到条件满足，与计算返回义务汇合。
5. GM 路径的 join owner 内联发布 DONE，或 CntNotify 路径的最终 CHECK owner 发布 Q，使计数达到任务阈值。消费者在依赖完成后读取数据。

首要测试取实际同类执行资源数 C，分别提交 C−1/C/C+1 个 waiter，然后提交 publisher。publisher 不依赖 waiter；消费者依赖 waiter。必须证明 publisher 可运行、未通知时消费者未执行、通知后消费者结果正确。仅有编译、同步回归或 mock 条件测试不算闭环完成。

## 3. 服务与完成顺序

本轮未发布异步工作时，空闲核每 16 个服务机会探测一次活动标志。活动开始后，累计 4 次业务返回触发异步服务；空闲或准入受阻时也检查服务门。CHECK 从第一个活动服务片段开始，之后每两个片段获得一次机会，每次最多领取一个记录、检查 8 个条件。单纯的 PENDING 或 CAS 碰撞不会刷新 watchdog。

每个实际执行在返回时先封口登记。CntNotify 路径上，普通/SPMD 和 MIX 的每个成功成员各贡献一个物理 P。可能异步的 lane 还需一个条件 Q：空组在封口或返回尾部结清，非空组由最终 CHECK owner 结清。MIX 记账成员收齐本代活动成员的成功 FIN 后退役 bank。

GM 完成门的物理总数是 block 数乘 active lane 数。普通/SPMD 的物理贡献发生在本核返回时；MIX 由记账成员收齐本代成功 FIN 后按 active lane 数一次贡献。只有非空条件组增加 join 的 pending，首个非空组还为任务设置一个异步门闩。最后一笔物理贡献在没有门闩时直接完成任务，有门闩时关闭计算端。关闭计算端和最后一个条件组结清中，完成汇合的那次原子操作授予唯一的 `FINISH_OWNER`。

最终准入和消费者本核获取之后，执行核只解析一次 task/block/lane 到条件组的映射；绑定保存在本次 kernel 调用的局部变量中，返回时直接复用。返回仍核对 EXECUTING 状态、run 与组身份、任务及 block/lane、记录边界、开放预留和重复封口，不依赖 kernel 可改写的上下文来恢复绑定。seal 和启用的参数抓取完成后经过返回门禁，才直接退休本核 slot。

GM 的 `FINISH_OWNER` 在取得所有权的调用中通过 CAS 将 `(T, 0, 1)` 转为 `(T, 0, 0)` 并发布 DONE。后继各自检查 fanin，AICPU 观察 Host 规划的所有终点。CHECK 的条件检查游标只用于尚未完成的条件组，失败进入本轮首错和清退流程。

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

复用已有 DFX 输出和协议观测入口。必要事件关联 run/task/block/lane、登记封口、实际执行释放、条件满足、逻辑完成及资源退休；不在热路径逐次打印 PENDING，也不为了观测扩展用户可见接口。

GM 内联 FINISH 的 TaskDone actor 在既有 `flags` 字段中设置 bit 31，表示延迟完成。cooperative task 的 order 从已发布的 admission 记录读取，不依赖完成核仍持有原 cohort。校验器仅对带此标记的完成事件允许跨核 actor，并检查全部物理子任务贡献、封口登记先于 computation close 以及后继依赖；执行 trace 和 MIX release 的收尾由资源退休与 EXIT 顺序约束；admission、inline 和 Graph control 不能使用该标记。该诊断不替代外部完成条件和 payload 可见性的功能验证。
