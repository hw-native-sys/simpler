# A5 HBG resident：按执行顺序走读

本文从设备入口开始，描述当前 Host 预构造 Call、AIC/AIV 自行调度的 resident 实现。
Graph view/native image 为 v6/v5，因果诊断普通/native Graph 格式为 v22/v23。
工作发现使用不可变目录和 execution token，依赖使用 canonical completion gate。

## 阅读顺序

1. [结构与生命周期](00-data-structure-scopes.md)：输入、GM backing、核内状态及各自所有者。
2. [初始化](01-initialization.md)：设备入口、物理报告、READY、bootstrap 和 DMB。
3. [发现与执行](02-ready-execution.md)：fanin、四个 token、两个 slot 与函数调用。
4. [协作、完成与退出](03-cooperative-completion.md)：MIX/sync_start、canonical debt、trace 和退休。
5. [Graph 与 async](04-native-async.md)：实例展开、边界参数和延迟逻辑完成。

精确字段和尺寸统一以[数据布局](../design/a5-hbg-graph-task/data-layout.md)为准，
调度不变量见[执行协议](../design/a5-hbg-graph-task/protocol.md)。

## 端到端执行链

| 顺序 | 执行者 | 动作 |
| ---- | ------ | ---- |
| 1 | Host | 完成 orchestration，pack 顶层镜像，展开 Graph，绑定全部 Call 和 invocation Context |
| 2 | AICPU 与 AIC/AIV | AICPU 清本轮 discovery/gate；AIC/AIV 从平台入口进入 aicore_execute 并报告身份 |
| 3 | AICPU | 接受本轮物理报告，校验完整 cluster、执行需求、epoch 和寄存器归属 |
| 4 | AICPU | 填 WorkerContext，先发布地址，再发布 READY |
| 5 | 每核 | acquire READY，观察配置，校验 graph view、prepared layout、catalog 和 peers |
| 6 | 每核 | 用 worker_index 分区 bootstrap，初始化 admission；到达 scan barrier |
| 7 | 每核 | 服务控制节点、准备首批 slot；最后一核发布 bootstrap_complete |
| 8 | AICPU | 打开 DMB launch gate，核获得执行权限 |
| 9 | 每核 | 发现 Call、观察依赖、准备本核 slot，通过最终 admission/资源门禁，调用 kernel |
| 10 | 每核 | seal async，归还真实贡献，写回 trace，退休 group/slot |
| 11 | 各参与者 | 服务 async 和控制节点，批量汇总 logical completion；AICPU 监督完成或首错 |
| 12 | AICPU 与每核 | EXIT、nonce/ACK、teardown 和 post-close release；之后才能返回及回收 backing |

代码入口为 `aicore_executor.cpp::aicore_execute`、`AicoreLifecycle`、
`resident_scheduler_impl.h::run_resident_scheduler` 和 `run_ready_dispatch_loop`。
两份普通/native 编译入口共享这些合同；无效启动不能切换到其他执行器。
