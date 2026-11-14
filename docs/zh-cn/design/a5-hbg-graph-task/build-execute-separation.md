# HBG 调度重构：有序供给、固定 execution token 与连续执行

**状态：固定归属、有序发票、Host Kahn 分层排序和本卡通知排序已接入。** Host 根据已有数据、Graph 控制和可确定的本卡通知关系排序。Device 保留前项全局发完有效票再开放后项的约束，不要求前项整体执行完成。普通 HBG 与 native Graph 的本卡通知排序已有实机证据；不得据此宣称远端通知已经发起。跨卡通知保留原用法，不增加 Host 跨 rank 收集、同步或参与要求。

本文以 `atomic_probe/pa_scheduler/cross_core_only_scheduler` 的实际执行流程为参考，只保留当前实现与拟修改合同。目标顺序是 **Host 补齐数据与通知排序约束 → 拓扑分层并在层内排序 → 投影为两类任务目录 → Device 按序发完前项有效票再推进 → 固定 owner 等待并执行**。不能仅移植四 token 和提前领取，却遗漏 probe 的有序取票。HBG 的 Build 已由 Host 完成，不补一套 Device Build 调度器。通用排序的效果按八组非通信 L0 总和验证。

修改范围为 Simpler 调度器、必要的 Host 布局、内部协议与 DFX，以及经授权的 PyPTO 内部通知元数据适配；不修改用户 kernel、前端语义或参数 ABI。继续使用这一份统一文档；开发按完整功能纵向分阶段本地提交，未经允许不 push。

调度优化必须由通用任务状态、依赖和资源合同驱动，不按模型名、特定 task 编号、shape 或固定图拓扑选择策略。SWA 等单例的泳道图用于提出和验证机制假设，不能替代通用性验收。

最终必须完全废弃原 ready_queue 调度逻辑：不能在 token 旁边保留 ready 发布/领取作为备用执行入口。保留 AIC/MIX、AIV 两个不可变有序目录，不增加分片；每类共享发行前沿只在当前 task 全局发完有效票或合法终结后前移。它约束新票供给，不阻止已有 token 执行；跨引擎依赖另外检查前驱供给进度，不新增串行化两类独立任务的全机单游标。任务 ready 时不入队，也不重新竞争执行权。普通、MIX、sync_start 统一遵守供给顺序，仅保留各自资源与成员协议。迁移时删除独立轮转发现、转交执行权、旧 ready 中转及只服务这些路径的布局、DFX 和测试，不留双轨开关。

## 1. 目标执行循环与当前差距

### 目标的准确含义

有已发布且适合本核类型的工作、本核有补给机会时，提前持有具体待执行单元。前一 kernel 返回并完成必要的结果发布后，若已有 pending 的全部执行条件满足，就直接进入下一 kernel，不再先做一轮全局发现、重新分配 block、构造参数和 READY slot 发布。

这里的 pending 必须含具体 `node_ref/block/lane`、唯一执行票和所有权。只保存 task ID、fanin view 或一个“可提前准备”的 hint，不满足要求。最有价值的 pending 已完成 Claim 和参数接收，停在 `WAIT_FANIN`；只有票号、尚未接收参数的 `TICKETED` 不能计为同一种准备程度。

“有 pending”不等于“有 ready”，也不等于任意时刻都能填满窗口。元数据未发布、真实依赖未满足、MIX 伙伴忙碌、sync 成员尚未到齐时，必须保留等待语义。没有下一项可用 token 的补给路径单独计时，不能混成快入口已达标。

分别测量两个间隙：

- `G_worker`：同一核下一次 kernel 开始减前一次 kernel 结束，回答本核能否快速接续。
- `G_dep`：后继首个 kernel 开始减真正最后一个计算前驱的最后 kernel 结束，回答跨核依赖交接是否接近 5 µs。异步场景另记录最后逻辑条件满足到后继开始，不能把物理返回当作逻辑完成。

同时记录同一 SPMD 任务的全部 block 起跑跨度、实际参与核数、每核执行次数和尾部。仅首个 block 早启动、其余 block 多执行一轮，不算达到目标。5 µs 是拟达到的交接预算，整体收益仍由完整 L0 实测判断。

### 先确定每核运行的主循环

```text
启动：接收 Host 已发布的镜像，初始化本核 token 窗口
循环：
    处理已观察到的终态；履行已承诺的 MIX/cohort 责任
    优先推进已有 token，单次最多访问四槽：
        TICKETED → 唯一 Claim → 接收固定数据 → WAIT_FANIN
        WAIT_FANIN → 检查未确认的依赖；不满足则查看其他 token
        条件满足 → 最终许可 → kernel → 必要完成发布 → 查看下一 token
    到服务预算边界：推进一片独立 async/control 服务
    有相应空位时：查询本类共享头部及其跨引擎前驱供给进度
    头部可领且本核符合条件：领取一块；暂不能领则服务已有责任，不自行跳到后项
    头部全局发完有效票或合法终结：推进共享头部；已有 token 不等整个 task 完成
    本核责任退休：若头部仍有余块且本核可领，继续领取，不能改填后序新票
    全窗口等待：保留四份责任，观察前驱；继续服务独立 async/control 和已承担的协作责任
    无可推进项：有界轮询；不持有 lease 等待外部条件
```

普通 token 的 fanin 已满足时，先用本核 admission 的 current/pending 字段做否定过滤：任一非空就保留 token，不填写和发布必定无法执行的 slot。不提前调用共享许可检查，也不缓存正向许可；本地字段为空仍须在实际 kernel 入口检查共享 admission/MIX 状态。已有 slot 遇到新的跨核发布时，原有本地回收路径继续有效。

一个已有 token 完成后可以解锁本核另一个 token，返回应首先沿这条局部链继续。有序取票是补给入口；已有 token 的执行不能以先推进 issue_head 为前置条件。MIX/cohort 的已承诺责任、终态处理和到期的控制服务仍按各自协议处理。

已有未来 MIX 组也遵循这一规则：正式组服务之后，先观察并尝试 ARM 已持有的 `cluster_pending`，再发现新 cohort/MIX。若因已提交 cohort 的 frontier、hold 或 slot 占用而不能 ARM，继续履行该 cohort；ordered 服务有进展后至多再尝试一次已有 pending。没有发生新领取时，不在同轮发现结束后重复检查同一 pending。该优先级不绕过会合许可，也不能因为 pending 的 fanin 未满足就停止控制服务或合法的新任务供给。

普通任务成功返回后，可以保留本核已确认的 fanin 作为续领提示，但它没有独立发票权。旧 block 退休后，必须重新确认本类共享头部；提示命中这个任务且仍有余块时才可续领，不能因提示省事而越过尚未发完的前项。已有 ready pending 优先执行，predicate 和最终入口检查仍逐次执行；MIX、sync_start 保留各自的组/成员协议。

必须区分两条真实路径：

```text
已接收的 pending：前一 kernel 返回 → 必要完成发布
                 → 剩余依赖/最终许可 → 下一 kernel

缺少 pending：   有序供给检查 → FAA 取票 → Claim CAS → 接收参数
                 → 等待依赖 → 最终许可 → kernel
```

目标是让第一条覆盖足够多的实际交接。第二条也计入总耗时和间隙分布，不能只挑第一条报告小于 5 µs。取票、Claim、参数接收必须在后继自己的入口之前完成；若要让它们与前驱计算重叠，还必须在前驱完成前得到 Scalar 执行机会。

### probe 实际提供了什么

参考目录为工作区 `simpler-fdwic-swimlane-deps/tests/atomic_probe/pa_scheduler/cross_core_only_scheduler/`；硬件合同参考同仓 `tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`。HBG 不移植 ordinary probe 的 Device Build、TensorMap 插入、Submit 或任务生成。

| probe 机制 | 实际入口 | 对 HBG 的含义 |
| ---------- | -------- | ------------- |
| Build 使用共享递增票号 | `pa_scheduler_core.h::TakeSharedBuildTicket` | 借鉴确定的任务顺序；HBG 已在 Host 构造，不复制 Device Build 发票 |
| Execute 从同类共享游标领取连续 ordinal | `pa_scheduler_core.h::TakeSharedExecTicket` | 所有同类领取者共享供给进度，不能各自选择目录起点 |
| Host 以最终设备地址预绑定不可变参数/context | `common/prebuilt_dispatch_host.h::PrepareStaticExecutionBindings` | 保留并完善现有 Host Call，不再用 Device 重建已知常量 |
| 每核 4 个真实 token，每批 2 个 Execute ordinal | `shared_exec_protocol.h::kExecTokensPerWorker/kExecTicketBatchSize`；`pa_scheduler_core.h::TakeSharedExecTicket` | 借鉴有界前视；不机械照搬批量 2，避免宽 SPMD 囤票 |
| 先绑定具体 task，再领取共享 cell | `BindSharedExecTicketToToken`、`ProgressCrossCoreActiveToken` | 将具体执行票的发放前移到 fanin 就绪之前 |
| 已知 BUILT 字直接做一次 Claim CAS | `pa_exec_adapter.h::PrebuiltPaExecBinding` | 初次领取使用 Host 发布的通用初态，消除 BUILT/PREPARED 双重尝试 |
| Claim 后接收参数，未 ready 时保留 `WaitingFanin` | `shared_exec_protocol.h::ClaimAndBindPreparedExecPayload` | pending 持有具体执行责任，不再只是 task/fanin 观察 |
| 检查剩余 fanin 并在同一 helper 转入执行 | `TryMarkExecutionTokenEngineInflight` | ready 后不再领取、重复确认同一批依赖或重新发布本核 READY |
| 先推进已有 token，有空位再取票 | `ProgressCrossCoreOwnedTokens/ProgressCrossCoreExec` | 将 pending 快入口放在发现/补给之前 |

probe 有 AIC、AIV 两个集中 Execute 游标；其 FAA 保证各引擎内 ordinal 发放有序，不保证跨引擎的物理调用或完成时间也有序。`common/host_prepare.cpp::PrepareExecutionImage` 先在 Host 构造镜像，`common/only_scheduler_host.h` 明确检查 Device Build 票数为零。HBG 借鉴有序发票和固定 owner，另补通用 DAG 的跨引擎前驱检查及 SPMD block 规则，不能把 probe 特定图的验证外推到所有图。

每批 FAA 与每 task Claim CAS 是不同操作，“单 CAS”只描述 cell Claim，不包含取票、依赖和完成原子。其固定 PA 控制字推导依赖特定任务形状，HBG 必须由通用 Host 计划提供初态，不能复制 `task_id % 5` 等模型规则。

probe 先把一批 ordinal 绑定为 `WaitingBuilt`，再逐槽 Claim、acquire、检查依赖和同步执行；不是先把四槽全部准备好才执行。如果第一槽已经 ready，它可以先执行，第二槽此时仍未完成 Claim。因此不能把“批量领了两张票”直接当作“下一个 kernel 已准备好”。已经停在 `WaitingFanin` 的下一项才具有无新票、无新 Claim 的快接续路径。

同 Scalar 执行同步 kernel 时不补 token，也没有别核写它的私有 token；前视来自此前的调度机会。本文保留这种机会性前视，不承诺强制满窗口，也不为补满窗口延迟已有 ready 工作。HBG 的不可变构造由 Host 完成，其他空闲 Scalar 可以推进依赖、控制服务和自己的 token；接收核自己的 DCache 观察和动态许可仍由接收核完成。

### 当前 HBG 与目标的对应

下面区分已接入的机制与待纠正、仍需验收的合同。实现、编译和测量前须重新冻结实际 HEAD、diff、输入和二进制；本文不把旧采集标成新版本结果。

| 部分 | 当前实现 | 本次目标 |
| ---- | -------- | -------- |
| Host 任务顺序 | `scheduler_order_task_catalog` 按 `(level, -b_level, node_ref)` 排序现有数据、Graph 控制及可确定的本卡通知边 | 补齐实际通知关系并逐边校验，保持任务身份与业务依赖不变 |
| Device 发票顺序 | `scheduler_issue_head` 仅返回本类共享头部；未发完就不看后项 | 保留，目录改用 Host 重排后的拓扑序；本核不可领不授权跳项，发完票不等于执行完成 |
| 跨引擎依赖 | `scheduler_issue_fanin_ready` 在领票前确认前驱供给已关闭 | 保留数据前驱供给检查，补齐通知排序边与 Graph 控制检查；实际执行仍等待真正 DONE |
| Host 镜像 | 每执行单元已有不可变 Call 和独立 Binding | 保留；固定定位信息尽量在 Host 完成 |
| 普通候选 | 4 个具体 execution token，含 block、lane、所有权和阶段 | 验证 pending 覆盖率与整体 block 供给，避免只有首块交接变快 |
| 普通 SPMD 领取 | task 内 `next_block` 发票，初次 Claim 后接收数据并等待依赖 | 单核同 task 最多一个未退休 block；A0 退休后 A1 若可本核领取，须优先于后序新票 |
| 共享准备状态 | Host 发布 BUILT Call，执行者 Claim/接收；Device 重复 builder 已删除 | 保持唯一执行责任与错误退出合同 |
| 调度循环 | 已有 token 优先；满窗保留责任，不再改领替代项 | 验证补给、完成与独立控制服务均能推进，并降低交接开销 |
| 普通 MIX | 每簇有正式组及一份固定未来组，从 AIC/MIX 头部取票 | 保留统一头部，使用重排后的目录；已有组不转移，未来组不提前占用伙伴执行资源 |
| sync_start | 独立有序 continuation 与动态成员准入 | 成员确定后由原成员履行，不绕过真实到达 |
| defer_wait | 共享等待记录与独立 CHECK/FINISH 保留；普通与 native Graph 的本卡后置通知用例已有实机证据 | 通知任务 P 排在 W 前，等待前保证通知路径已可独立推进；保留晚到通知和固定 owner |
| 完成 | N16/C16/D32 权威门控 | 保留，不另造一份普通 DONE/ready 权威 |

## 2. 从 task 有序开始：供给、预绑定、等待与退休

### 先区分构造、发票与完成

| 状态 | 含义 | 不代表什么 |
| ---- | ---- | ---------- |
| BUILT | Host 已发布该实例的完整不可变 Call | 没有指定执行核，也不代表依赖满足 |
| ISSUED_ALL | task 的全部有效 block 已取得不可撤销的执行责任 | 不要求这些 block 已开始或完成计算 |
| DONE | 计算贡献、合法 skip 和异步条件按原协议全部结清 | 只能由权威完成协议确认，不能由发票游标推断 |

`ISSUED_ALL` 是本文的逻辑判定，不要求新增一份完成状态。普通任务复用有效 `next_block` 发票事实，sync_start 复用真实成员领取及 claim seal；MIX 按逻辑 block 的整组责任判断。执行者取得有效票后必须有限步保存身份、完成 Claim 或报告终态，不能取得票后又等待其他任务来决定是否保留。

本方案保留 `ISSUED_ALL` 作为后项领取门槛：Host 先形成满足依赖约束的顺序，Device 按这个顺序把前项全部有效执行票分配出去，再开放后项。这里的“全局”是该 task 的所有 block 都已有人负责，不是只看本核已领到一块，也不是要求全机只有一个发票游标。对于普通依赖 `A → B`，A 全体发票后即可提前领取 B、接收参数；B 的 kernel 仍等待 A 的逻辑 DONE。因此有序发票与重叠准备、并行计算相容，不能把发票门槛误称为执行完成屏障。

### Host 拓扑分层与层内优先级

**保证前驱排在后继前面，必须在 Host 排序时完成，并在目录发布前逐边验证。** 选择 Kahn 式拓扑分层，优先展开同层的独立分支；层内再按剩余依赖链长度排序。参考 [Kahn 原论文](https://doi.org/10.1145/368996.369025) 与 [拓扑分层定义](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_generations.html)。

排序输入是已实例化的 DAG 及必要的通知排序边。`level` 表示从源节点出发的最早拓扑层，`b_level` 表示到终点的最长剩余依赖链；初版均按边数计算，不假设已知 kernel 时长：

```text
level(v)   = 0                                      无前驱时
           = 1 + max(level(p), p 是 v 的前驱)          其他情况
b_level(v) = 0                                      无后继时
           = 1 + max(b_level(s), s 是 v 的后继)        其他情况
排序键      = (level 升序, b_level 降序, 原 node_ref 升序)
```

同层节点之间没有依赖；每条排序约束 `A → B` 都必须满足 `level(A) < level(B)` 和最终 `rank(A) < rank(B)`。这是拓扑分层，不是按最短路径距离做普通 BFS。Kahn 的零入度集合只用于 Host 规划，不恢复 Device ready_queue。

以讨论中的局部依赖为例，省略其余前置条件：

```text
t4 ──→ t8 → t9 → t10（MIX）
 └──→ t11 → t12

层 0：t4
层 1：t8、t11
层 2：t9、t12
层 3：t10

一种全局顺序：t4、t8、t11、t9、t12、t10
AIC/MIX 投影：t8、t11、t10
AIV 投影：   t4、t9、t12
```

**层号只是排序优先级，不形成执行屏障。** t9 的真实前驱完成后就可以执行，不必等待独立的 t11 完成；同层也不要求同时启动或等待全层资源。Host 排序提供并行机会，实际执行仍受数据就绪、核类型、block 数和成员许可约束，不宣称该启发式对任意图都是最优。

### Host 发布前的硬约束

排序位置 `rank` 与原任务身份 `node_ref` 分离，不改 task ID、kernel、Tensor/scalar 绑定、predicate 或完成身份，不预定物理执行核。两个不可变目录必须是同一个全局序的有序投影，不能分别独立决定优先级。

1. 复用现有数据依赖合法性检查。现有 `scheduler_fanin_ref` 要求数据 producer ref 小于 consumer ref；非法引用、自依赖等不能靠排序掩盖。Host 已完成节点沿用合法豁免，不重读其失效的历史输入。
2. **已有信息能确定的本卡通知生产/发起任务 P 必须排在对应 defer_wait W 前：`rank(P) < rank(W)`。** 多个通知来源分别建边。通知排序边与数据 fanin 分开表示；不为达到顺序而修改数据版本，也不把它误解释成“P 必须逻辑完成后 W 才能登记”。具体运行时与跨卡约束见第 4 节。
3. 对数据依赖、通知排序边及 Graph 控制边共同检查无环性，并验证排序后每条边的 rank。通知边可能连接原提交序中较晚的 P 与较早的 W，不能用 node_ref 大小代替 rank 校验，也不能把通知边直接塞入要求逻辑 DONE 的原数据 fanin。
4. AIC/MIX、AIV 两个目录无遗漏、无重复，保持全局序的相对位置。普通和 sync_start 的同类节点共用目录；Vector-only MIX 仍由 AIC coordinator 领组票，不再从 AIV 目录重复领取。
5. Graph ENTER/EXIT、control/inline 的真实约束必须参与规划；这些节点不因排序变成普通业务执行票。保留实例身份、元数据发布和动态激活条件；sync_start 排序不替代资源准入、真实成员领取与 seal。
6. Host 预计算静态约束、目录位置及必要的动态检查目标，Device 不重复计算层号或验证整张拓扑。每类关闭前缀证明其前方供给已结束，未关闭头部约束该类新票；跨引擎前驱与通知关系保留实际供给/进展检查。目录 rank、已发票与 DONE 分开，不用本类头部代替另一类状态。
7. 静态排序、实际领取、真实完成分别验证。尤其不能只凭 `rank(P) < rank(W)` 就认定远端通知已经发起；跨引擎、跨卡进展还需相应的运行时或协议证明。

Host 必须先完成已知约束图与目录校验，再发布和 H2D。非法图或成环时报出具体节点关系，不发布部分目录让 Device 补救。远端来源或缺失通知元数据不等于非法图，不能为此新增运行拒绝。现有 ref 顺序可用于计算原数据图的层号；加入通知边后须针对完整约束图重新确认拓扑序，不能继续假定 ref 升序满足新增边。不增加逐 block 大表，block 仍由 task-local `next_block` 分配。

### 与 percounter、列表调度的关系

| 方法 | 主要依据 | 本方案用途 |
| ---- | -------- | ---------- |
| Kahn 拓扑分层 | 优先安排最早拓扑层的独立分支 | Host 主排序规则 |
| percounter 的 b-level | 最长剩余依赖链优先、原 ID 稳定打破同分 | 借鉴为层内次序；纯 b-level 作为冻结性能对照 |
| HEFT | 预计计算/通信成本与最早完成时间，并选择处理器 | 后续资源模型的参考；不直接采用其静态物理核映射 |

percounter 的实现位于 `src/a5/runtime/percounter/host/region_plan.h::build_tables` 与 `host/graph_template.h::build_graph_template`。它对普通节点按 b-level 降序，同分按 ID；长支路可能先深入几步再处理独立短支路。本方案把正向层号置于 b-level 之前，两者不能混称为同一个策略。

[HEFT 原论文](https://disco.ethz.ch/courses/fs14/seminar/paper/Jochen/4.pdf) 可作为异构资源调度参考。本方案初版只用静态图信息，不要求 kernel 耗时模型；实际核继续由 Device 动态领取确定。比较“拓扑分层＋层内 b-level”与纯 b-level 时，保持 kernel、输入及资源协议一致，以八组非通信 L0 总和判断，不按模型选择策略，也不在生产中保留历史兜底开关。

### Device 沿拓扑目录发票，前项全体发完再推进

保留 `scheduler_query_issue_head` 的共享头部语义。**依赖图给出偏序，Host 把它排成满足约束的确定顺序；Device 不临时重新选序。** 同类目录中，前项所有有效执行票发完或按协议合法终结，才能领取后项。跨引擎没有依赖的目录项可以各自推进；有依赖时还须确认另一类前驱已全体发票或合法终结。实际调用另外等待数据 DONE、Graph 激活及资源许可。

| 当前情况 | 处理 |
| -------- | ---- |
| 本类头部有余块，本核有相应位置且未持有该 task | 领取头部的一块，立即保存固定 owner |
| 本核已持有头部 SPMD 的未退休 block，头部尚未全体发票 | 不领同 task 第二块，也不越过头部；推进已有 token、协作及独立控制服务 |
| 本核 block 已安全退休，头部仍有余块 | 重新取得该 task 的领取资格，继续补前序余块 |
| 头部全局发完票，部分 block 仍在执行或等待 | 推进头部，可以提前领取后项；不等待前项整体 DONE |
| 普通窗口或相应组/成员位置已满 | 不增加隐藏责任、不迁移 owner；服务已有工作 |
| 跨引擎前驱未发完、元数据未发布或必要控制/通知约束未满足 | 暂不领取该候选，继续相应进展服务；本类头部不能替代这些证明 |
| 候选是 MIX 或 sync_start | 遵守同一目录，再执行各自组/成员协议；完整组责任或真实 claim seal 才能关闭供给 |

```text
有界补给：
    查询本类共享头部；已发完或合法终结的项按预算推进
    检查跨引擎前驱供给，以及 Graph/通知的必要门槛
    本核有相应位置、未持有该 task 且协议允许：领取一个有效单元
    本核暂不能领：返回服务已有责任，不用私有 cursor 跳到后项
    有 ready 的已持有任务：不为继续发新票而延迟它
```

“前视”仍指在前驱计算完成之前持有后继 execution token，不表示越过尚未发完票的目录项。`SchedulerIssueSnapshot` 可缓存单次 fill 已确认的头部，头部推进或已有协议要求失效时须重新查询；不引入逐核扫描后项的选择器。共享进度用返回型原子确认，`ld_dev` 仅作保守提示；不得用 hint 授予 owner 或释放资源。继续参考 `atomic_probe/ATOMIC_USAGE_GUIDE.md` 的同址争用与持续轮询进展证据。

这是一项比“只按真实依赖边先后发票”更强的供给约束：同一目录里的独立任务也被 Host 排成先后。本轮保留它以简化责任与进展证明，把独立分支尽量靠前的工作放在 Host 排序中。前例的 AIC/MIX 目录改成 `t8、t11、t10` 后，t8 发完票就能开始 t11 的供给/准入，不再等 t10 发完；不要求 t8 整体执行完成。若 t11 本身是 sync_start，真实成员领取仍须满足它自己的依赖和资源许可。

### 接受的性能代价与本轮不改善的原因

**保留前项全体发票后再领取后项，接受超出物理核数的宽 SPMD 可能推迟独立后项的代价；本轮不为此引入跳项机制。** 这里用 `N` 表示 task 的逻辑 block 数，用 `K` 表示实际配置的对应引擎资源池的物理核数；AIC、AIV 分别计算，不能将两类核相加作为普通 task 的容量。

当 `N > K`，每核同 task 最多持有一个未退休 block，最多 K 张票可同时被持有；必须有部分 block 执行并退休后，才能继续分配余票。例如 4 个 AIC 已各持有 A 的一块、A 共 8 块且等待前驱 P，目录中后面的独立任务 B 即使已经 ready，也不能领取。A 的余票未发完期间可能出现空闲计算资源，损失原本可与 P 重叠的 B 执行机会。这是说明机制的例子，不是实测性能结论；若核正在满负荷执行 A，也不能将 B 等候直接等同于计算资源浪费。

**当前目标网络的多数业务通常按 SPMD block 数不大于对应物理核数组织，即 `N <= K`。** 这是本轮设计采用的业务常态。在各核有领取机会且协议允许时，全部 block 可以分别交给不同核，无须等待同 task 的 block 退休来腾出下一张票的资格；因此上例由 `N > K` 必然引入的多轮发票，不是当前主要优化对象。

本轮优先保留明确的领取顺序、固定 owner 和简单的进展证明，通过 Host 拓扑分层改善独立分支次序，并优化常见路径的交接开销。不为较少出现的超宽 SPMD 增加 Device 动态跳项、额外候选搜索或已领取任务迁移。超宽普通 SPMD 仍须正确执行，这项业务常态不变成 `N > K` 的输入拒绝条件；sync_start 超过同时驻留容量的拒绝规则另行保留。

`N <= K` 也不保证随时发完票：部分核暂忙、窗口已满、MIX 成员约束或 sync_start admission 仍可能推迟领取。这些实际等待继续在 L3 中区分，不能据此宣称已经实现最大并发。当前验收仍按八组非通信 L0 总和；只有后续代表性业务证明超宽 SPMD 的供给等待成为主要瓶颈时，再重新评估是否值得改变这项取舍。

### 前序余块与四槽固定等待

以非 sync 的 A 有 A0、A1 两块，B/C/D/E 都依赖整个 A 为例。**当 A1 尚未被任何核领取时，窗口 `[A0, B, C, D]` 不符合本方案的供给规则**，因为 B/C/D 不应已获得执行票。一核执行时，合法过程为：

```text
先领 A0：  [ A0, 空, 空, 空 ]    A1 尚未领取，不能跳到 B
A0 退休：  [ 空, 空, 空, 空 ]
再领 A1：  [ A1, 空, 空, 空 ]    A 全体发票，此后可以按序领取 B/C/D
后继窗口： [ A1, B, C, D ]       仅示意允许的持票状态，不强制延迟 ready 的 A1 来填窗
A 全部完成后，B/C/D 才能执行；其 owner 始终不变
```

多核时，A1 可以由另一核提前领取；确认 A 已全体发票后，持有 A0 的核也可领取 B，无须等 A0 退休。若 A0 退休时 A 仍有未分配余块，继续从 A 领取，不改填 E。若要求 sync_start，Host 先检查同时驻留需求：超过资源池总容量就拒绝；总容量足够而暂时忙碌则由 admission 等待，不能让部分成员先进入等待全部成员的 kernel。

这条规则排除了“后继窗口挤掉尚未分配的计算前驱”的示例，但不独自证明全部活性。仍须保证已发票责任得到服务、MIX/cohort 资源协议不成环、Graph 控制可推进，以及通知来源已纳入约束。固定 owner、四份普通责任上限、合法 skip 和错误退休合同保持不变。

### 不可变 Call 与可写运行态分开

一个纯核执行单元的身份为 `(run_epoch, node_ref, block_idx, subtask_slot, execution_index)`。`node_ref` 区分 native Graph 实例，不能用原始 source TaskId 或 Definition ID 替代。MIX 的分配单位是整个逻辑 block，其 active lanes 各有独立 Call。

| 信息 | 构造位置与使用时机 |
| ---- | ------------------ |
| 函数设备入口、Tensor descriptor 地址、scalar、固定 shape、Call/Binding 地址、block 身份 | Host 以最终设备基址构造并检查；Tensor 内容未生成不妨碍绑定稳定地址 |
| async image/bytes/run_epoch、group offset/index 等固定定位信息 | 同一 run 内按执行单元预计算；不等于登记了 deferred condition |
| 实际 execute_owner、可能依赖物理 AIV lane 的 `sub_block_id` | 归属确定后设置；不能把逻辑 slot 一概当物理 lane |
| predicate、Graph 激活、MIX/cohort 当次许可 | Device 执行边界检查，不读取尚未生成的数据来提前决定 |
| async count/error/entries、SEALED、credit、CHECK/FINISH | 真实执行与返回协议推进，不在预取时创建信用或完成通知 |
| DFX 序号/时间戳、args dump 和运行期可写标志 | 真实阶段产生，不伪造预绑定事件 |

沿用 `scheduler_prepare_host_calls()` 和既有 metadata/H2D/launch 发布链。函数入口使用注册时的 resolved entry，不能把 CoreCallable 对象地址当函数地址。固定字段只检查一次；执行者仍需通过正确的接收协议读取它们。

入口解析限定在 A5 HBG 的 Host Call 准备路径，复用原四参数 `set_callable_tables()` 提供的对象地址表。
A5 真机按注册器的既有布局取 `CoreCallable::binary_data_offset()` 对应的代码地址，不解引用设备对象；
A5 模拟器读取 Host 对象中的 `resolved_addr()`。A2/A3、TMR 和公共 Runtime 的绑定接口保持原样。

普通路径删除 `next_prepare_block`、Device `PREPARING/PREPARED` 和相应重复构造职责。Host 发布后的 Call 已是完整执行数据；领取者完成定向观察和 token 私有缓存即可。真正需要实际执行者填写的可写 context 在最终入口有限步填写，不能把整个 Binding 宣称为只读。

静态字段前移和取消 cache 维护必须分开论证。既有 [Binding 发布调查](../../investigations/2026-11-a5-prepared-binding-publication.md) 没有建立“Host 初始化后可省掉 Binding 接收”的依据；本方案保留所需 Binding 定向观察，并为其可写字段规定唯一 writer。不能仅因移除重复 Device store，就顺带删除保护其他字段可见性的 DCCI/DSB。

### 普通初次执行票

通过有序供给检查后，task-local `next_block` 每次 FAA 取 **1 个**有效 block，再用 Host 已知初态做一次 Claim CAS。不同 task 的状态分散，但同一宽 SPMD 的 `next_block` 和同类 issue_head 仍是共享观察点，不能宣称原子竞争已经消失。头部切换的 CAS、供给 fanin 观察、越界尝试都另计，不能用“一 FAA 加一 CAS”表示全部调度成本。

- 新票选择检查本核是否已有同 `(run_epoch, node_ref)` 的 current/inflight/pending；存在则该 task 暂不可本核再领，头部尚未全体发票时不能自行检查后项。退休后若头部仍有余块，继续领取前序任务；头部全体发票后才开放后项。
- 有效票立即保存到 token，Claim/acquire 重试不重新取票；只有越过 block 上界的超额票可以按既有合同退款。
- Host 已发布、初次唯一票上的 Claim 失配是错误或已确认的终态取消，不是普通竞争落败。
- `next_block >= block_num` 只证明新票已经发完，不证明 kernel 完成。已领取单元仍由各自 owner 等待、执行与退休，发票耗尽不得提前结束运行。
- 静态初态由通用计划生成并校验 run、单元范围及布局版本，Device 不预读可变 cell 来猜初态，不提前读取未获许可的 payload。

### 四 token 是有界前视，不是四份物理资源预约

每 worker 初始设置 4 个普通执行 token，包含 current/inflight；同核最多一个同步 kernel 在执行，其余槽可保存其他任务尚未执行的具体单元。同一普通 SPMD 任务不能同时占用两个位置。token 保存身份、Call/Binding 指针、执行票/owner generation、已确认 fanin 前缀和私有阶段，不复制四份 512B 参数包。

token 与现有两个 dispatch slot 分开。slot 可继续承载当前执行、协作责任和 DFX 身份，但普通 pending 不要求先占据物理 slot。不能通过把 `SCHEDULER_PENDING_SLOT_COUNT` 从 2 改成 4 来代替此协议。具体存储选型以小型 owner-private 状态为准，不将 128B GM 对齐传播到局部栈对象。

若通过现有 dispatch slot 复用 kernel/完成实现，该记录必须引用同一执行责任并保持原 executor；不能 token 和 slot 各保存一个可独立完成的 owner。token 的 `COMPLETING` 结束必须晚于该责任的最后一次必要使用，DFX 不能依赖已经复用的槽位恢复旧身份。

以下名称是目标协议，不表示当前源码已有同名状态：

| token 阶段 | 持有的责任 | 允许的推进 |
| ---------- | ---------- | ---------- |
| EMPTY | 无 | 可以接收新票 |
| TICKETED | 具体单元的唯一新票；尚未完成 Claim | 不可执行、不可遗忘；有限步完成 Claim 或终态处理 |
| ACQUIRING | 已有唯一执行权，正在观察只读 Call/固定定位信息 | 原核完成接收，不得改变 executor |
| WAIT_FANIN | 单元已领取并接收；依赖或 Graph 激活尚未齐全 | 保留原位置，检查本核其他 token |
| ELIGIBLE | 已满足当前观察到的条件，尚待最终资源/动态许可 | 满足最终许可才进入；不把此状态视为永久许可 |
| INFLIGHT | 实际调用已经开始 | 只能由原核返回、记账并退休，不可提前回收 |
| COMPLETING | kernel 返回或合法 predicate skip，正在归还责任 | 完成必要发布后释放 token，不从复用槽找旧身份 |

```mermaid
flowchart LR
    H[Host 发布完整 Call] --> T[TICKETED]
    T --> C[唯一 Claim 与参数接收]
    C --> W[WAIT_FANIN]
    W -->|条件满足| E[最终许可]
    E --> K[直接执行 kernel]
    K --> R[必要完成与资源退休]
    R --> N[继续检查已有 token]
```

### 执行归属固定，依赖等待保留原位置

执行票一旦领取，具体 `(run_epoch, node_ref, block, lane)` 就归领取核负责，直到执行退休或按错误协议终止。等待依赖、暂时缺少资源、窗口已满，都不能撤销执行归属、退回有效票或让其他核重新认领。

准备职责与执行职责分开：准备者可以提供不可变 Call，不能因准备了数据而自动取得执行权。当前 HBG 的静态 Call 由 Host 预构造；设备侧第一次领取执行票才确定 executor。已经确定的 executor 不因其他核空闲而改变。

- 有 ready 的已有 token：优先执行；未 ready 的 token、已确认 fanin 前缀及已接收参数留在本核原位置。
- 有空位置：从本类共享头部领取；同一 SPMD 仍最多持有一个未退休 block，不能因本核去重而越过未关闭头部。
- 四个位置都在等待：保留四份责任，继续观察前驱与服务独立控制路径，不领取第五份普通责任。
- 本核 block 返回并安全退休：释放其位置并复查头部；同一 SPMD 仍有余块就继续领取，不等待整个 task 完成。
- 不增设窗口外待办列表，不以切换存储位置的方式隐性扩大前视容量。

任务目录与 issue_head 只用于分配尚未领取的执行单元。已经领取的单元由原核直接观察依赖，不进入全局再次发现或竞争执行权的流程。删除 PARK 状态、共享重领索引和超时换 owner 的处理；本核 token/dispatch 表示切换不改变 executor，也不能重复计入容量。

四位置等待不等于自动证明全图活性。前驱、Graph 控制节点和异步通知者必须能够由尚未完成的执行或独立服务推进；若当前发票顺序可能让全部窗口等待一个尚未分配的普通通知任务，应从未领取任务的供给顺序与准入条件解决。禁止以转移已领取任务、丢弃责任或扩充隐性第五槽作为补救。

## 3. 快速接续的原子成本、服务边界与 SPMD 分布

### 原子操作分别算在哪条路径

| 操作 | 普通路径的目标成本 | 是否应留在已接收 token 的接续间隙 |
| ---- | ------------------ | --------------------------------- |
| 有序供给、选择本地空位 | 头部进度、前驱供给、本核去重与控制/通知门槛；前沿推进 CAS 和竞争另计 | 否；已有可执行 token 不依赖新的补给 |
| 初次分配具体 block | 一次有效 `next_block` FAA；超额竞争单列 | 否；应已持票 |
| 初次授予单元所有权 | 一次已知初态 Claim CAS，无预读加双重试探 | 否；应已完成 Claim |
| 接收固定 Call/Binding | 必要范围的 DCCI、尾 DSB 和私有元数据缓存 | 否；等待期间保留已完成的接收，不因等待重新竞争执行权 |
| 检查依赖和最终许可 | 只推进未确认的 fanin 前缀；动态许可按实际合同观察 | 是；不能用旧观察替代新出现的条件 |
| `WAIT_FANIN → INFLIGHT` | 唯一 owner 的私有状态转换，不新增领取 FAA/CAS 或本核 ready 队列原子 | 是；同一 helper 继续调用 |
| kernel 返回及完成发布 | 保留执行权退休、N/C/D 贡献、输出发布和必要协作操作 | 是；不能把完成成本藏在 kernel 时间内 |

“最小原子开销”的本次具体目标是消除重复竞争：普通单元首次有效分配的主体是一 FAA 加一 Claim CAS，之后已有 token 就绪时不再次领取。新增有序供给的共享观察、头部推进以及依赖、完成、错误和协作原子必须完整统计，不把主体计数当作整条路径的总成本。

### 最终入口与有界控制服务

一次 token 扫描最多访问四槽。有后槽完成解锁前槽时可再扫，但在不补新票的这段内最多完成四个已有 token；不能嵌入无界补给循环。最多四个普通 kernel 返回，或四次无 kernel 的推进轮次后，必须提供一次有界 async/control 服务；普通新任务发现仅在本核有空位置时进行。已公布的 MIX/cohort 责任按资源协议优先，不等待这个普通预算。

从 `WAIT_FANIN` 成功检查到 kernel 调用由同一入口完成：复用已确认前缀，不再次检查相同依赖；沿用错误观察合同，终态一旦被观察到就禁止新领取或调用。predicate、Graph 激活和会变化的资源许可仍须有效。实际执行责任和必要的动态 context 建立完成后，直接调用，返回时从该执行记录退休。

成功路径不再强制执行 `scheduler_publish_dispatch_slot → fill 返回 → ready mask 重扫 → 再领取`。没有可用 pending、资源不允许或需要控制服务时走慢路径；等待或执行表示的本核切换不改变 executor。两条路径共用 kernel 调用与完成实现，不遗漏 async writer、合法 skip、输出发布或错误处理。

从串行时间看，已接收 pending 的 `G_worker` 仍包含“前项必要完成 + token 选择 + 剩余依赖观察 + 最终许可/动态 context + 调用入口”，到期的控制服务及已承诺的协作责任也要计入。提前领取只移走发现、取票、Claim 和固定数据接收，不能消除这条剩余链。

`G_dep` 沿真正最后前驱的完成发布、后继观察与调用入口计算；跨核可重叠的工作不相加。实现后若这条剩余链仍超过 5 µs，必须回到泳道细分其 atomic、DCCI/DSB 和普通访存，不能用“四 token 已完成”代替性能达标。

### 普通 SPMD 的分布约束

默认规则用于非 sync 的纯 AIC/AIV；MIX 和 sync 的票据另见下一节。

- 每个 worker 对同一 `(run_epoch, node_ref)` 的 current/inflight、未退休 dispatch slot 和普通 pending 合计最多持有一个 block。限制与 block 数、兼容 worker 数及已发票数无关；Fresh 取票和完成后的续领均遵守同一判定。
- 持有 A.block0 时，本核不能再领 A.block1；A 全局发完票后，在其他门槛和剩余容量允许时，可以按 Host 顺序提前领取 B.block0，不必等 A.block0 或整个 A 完成。A.block0 退休后若 A 仍有未分配余块，先领取其余块。不同 Graph 实例按 node_ref 区分，不按 kernel ID 去重。
- 每次只领取一个新 block；成功后先推进现有 token，不连续批领同任务多个 block。领取一个 block 不等于该 task 发完，不能据此推进 issue_head。
- 实际执行核由动态取票决定，Host 不固定 block→worker，也不按 prepare_owner、prefix 取模或缓存亲和性分配执行权。
- 本核 block 返回并完成必要记账、执行责任退休后，可以继续领取同 task 的其他 block；不等待整个 SPMD DONE、其他核返回或 task 级异步债务归零。例如 64 个 AIV 执行 128 个 block，第一块退休的核即可领取第二块，不增加全局轮次屏障。
- 等待期间始终计入原核持有的责任；只有本核执行退休或错误终止才能结束这份责任。等待不构成再次领取同一 SPMD 另一 block 的资格。

同一任务的各 block 共用不可变 fanin，成功返回后的续领可沿用本核已验证的依赖前缀，负向结果不跨轮缓存。Claim、Call/Binding 接收、predicate 和最终资源许可仍逐单元保留；MIX、sync_start 走独立协议。全窗口等待时保持全部普通执行责任，async/control 继续获得独立服务机会。

单核同任务单 block 与窗口上限不证明跨 task 的负载完全均衡。多个不同未来任务可能集中在少数核；必须结合等待跨度、空闲核、起跑分布和每核执行次数验证。不能仅以 Claim 变快、局部 cache 命中增加或 pending 数增多证明整体收益。

## 4. MIX、sync_start、defer_wait、Graph 与内存边界

普通 token 的前视能力不能直接替换其他责任。各类等待占用的资源如下：

| 类别 | 提前持有的内容 | 正式执行前不能提前占用 | 等待时的进展方式 |
| ---- | -------------- | ---------------------- | ---------------- |
| 普通纯 AIC/AIV | 本核窗口内的具体 execution token | 执行流水线、协作成员或伙伴 slot | 保留原位置与 owner，查看其他已有 token |
| 普通 MIX | coordinator 的一份未来组 token；可选只读准备公告 | stage/armed/arrived、伙伴 slot、cohort hold | 保留已确定的组归属，独立服务当前正式组 |
| sync_start | 独立有序 continuation 和静态准备信息 | 未获许可的成员票、seal、hold | 按顺序服务真实 admission；成员确定后由原成员履行 |
| defer_wait | 共享等待记录、每核一份独立 async 服务 continuation | 已返回的等待 task 的普通执行 token | 单个服务槽有界 CHECK/FINISH 后让出机会；通知任务可达性另行保证 |

任何类别都不允许本核同时执行两个同步 kernel。独立 continuation 的含义是等待记录和服务机会不被普通窗口耗尽，不是另外增加一套可以并发调用 kernel 的执行器。

### 普通 MIX：提前持组，不提前占伙伴

MIX 以 `(run, node_ref, block, group_generation)` 为组级 pending；AIC coordinator 仅从 AIC/MIX 共享头部领取组票，active lanes 不能分别从普通 AIV 目录抢同一组的未来执行。Vector-only MIX 仍由 AIC 协调，AIC 不产生虚假的业务调用或完成贡献。

每簇最多一份未来 group pending，独立于正在服务的正式 group；同 task 的 current/pending 组仍最多一份。满足前驱发票条件后，MIX 可在逻辑依赖 ready 前取得组票，但只保留固定组责任，不设置 stage/armed、arrived、cohort hold 或业务 slot。保留单独 preparation announcement，伙伴有空时可以观察不可变 Call；该公告不授予执行权，不允许两个准备者并发写动态 Binding。

已有未来组的依赖未满足时，保留 GROUP_RESERVED、已确定的 coordinator/cluster 和已完成准备。不能因另一个组 ready 就撤销这份责任并改领替代项；已确定的真实 lane 也不能迁移到其他核。

未来组等待不阻止独立 async/control 与已有正式组服务，但不授权领取第二份未来组。独立 MIX 通知者的可达性必须由未分配任务的供给顺序及准入保证，不能用让出已领取组来弥补。

只有未来组位置空闲且当前头部是普通 MIX 时才领下一组；头部是纯 AIC 或 sync_start 就服务对应入口，不能用 MIX 专属 cursor 跳过去。MIX 的全部逻辑 block 已取得整组责任后才允许头部前进，不能把第一个组 reservation 当作整 task 已发完。等待已有组时直接重查其依赖，不执行跨簇重领。

正式路径保留 `scheduler_cluster_mix_try_arm/service/retire` 的资源裁决，消费已持有的 block/owner，不二次发票。执行顺序为：确认逻辑依赖 → 检查 hold/frontier → stage → 再检查 → 发布 armed generation → 必需 lane 到达 → 执行 → 全部返回后退休。只协调实际 active lanes 和 coordinator，不扩大为无关成员。

已 ARMED 且本 lane 尚未履行的正式责任优先于普通 token 补给。`scheduler_cluster_mix_needs_service` 按本地 PUBLISHED/ARRIVED/RUNNING 或尚未观察的新 generation 判定服务优先级；它不授予执行许可。`scheduler_fill_own_slots` 保留已有 READY/direct slot 的排空入口，然后优先再服务正式组一次，不插入普通 progress/refill。resident 到期预补给也服从这个判定，coordinator 的 RELEASED 先得到 retire 服务。已经 finished 的 peer 不因同一代 staged mask 尚未清零而重复承担到达责任，可以继续有界准备。future reservation 不触发这项限制；async/control、错误检查及现有 slot 回收仍按原路径推进，不能通过跳过整个主循环来实现优先级。

组归属或真实执行成员一旦确定，等待期间保持不变。stage 的有界发布和失败清理不授权转交已领取任务；armed/arrived 后不迁移或拆组。部分 lane 仍在运行时不覆盖 SSBUF generation，也不让它们与后续 block 配对。

组所有权与 active lane 的执行责任不是一回事；不能将普通单 lane 的“一次 CAS”承诺套到整个 MIX。组级/每 lane 的实际授权和交接操作须分别统计，保留同一 lane 只执行一次的验证。

### sync_start：独立 continuation，真实成员仍按许可认领

每 worker 为有序 cohort 保留独立 continuation，不让四个普通 token 占满后阻止 admission 服务。Host 已知每个 block 的 Call 可以提前准备；未来 cohort 的候选不计入真实成员数、claim seal、hold 或启动完成数。独立的是服务位置；候选和真实成员仍遵守 Host 排序及共同 admission 协议，不另造任意乱序的协作任务供给链。

先在 Host 检查 sync task 的同时驻留需求：超过资源池总容量应拒绝；总容量足够而暂时忙碌，按 admission 等待。sync 不沿普通新票路径提前分配真实成员票；真实依赖满足、资源与 admission 允许后，实际可用 worker/cluster 动态认领成员，完成参数接收和会合。发布候选或日志不等于全部成员已领取，目录头部越过该项须有真实 claim seal；不能用独立 continuation 绕过这项供给门槛。各核不能分别占据不同 cohort 的部分成员互相等待。已承担责任由原成员履行，不用预先指定的核代替真实到达。

保留 full/partial 容量检查、共同 cohort sequence、最后有效成员 seal、局部退休及 frontier。普通 pending 不是资源许可：入口必须服从现有 stage/hold/armed 仲裁。已正式公布的普通 MIX 与已有 cohort 双重责任按同一裁决服务；相邻 cohort 不能形成循环等待，也不新增“上一整个 cohort 全部完成才开始下一个”的全局屏障。

协作场景的 5 µs 要区分静态接收、实际成员到达和必需同步。不能为了缩短泳道直接跳过成员许可。

### defer_wait：通知先安排，等待独立服务

**通知生产/发起任务 P 必须排在对应的独立 defer_wait W 之前：`rank(P) < rank(W)`。** W 的消费者 B 再按真实数据依赖排在 W 后。多个通知来源分别形成约束；通知边参与 Host 分层、b-level 计算和最终 rank 校验，不能只写成“通知通常能到达”。

本次约束只针对独立的 defer_wait task；不允许把 defer_wait 与计算 kernel 混合。普通 kernel 无须为此补齐所有内部行为的元数据，未知普通 kernel 不应阻止运行。通知生产者内部若进行同步等待，可以继续占用执行核；调度器不接管该等待、不增设让核或迁移补救，也不因它等待 waiter 登记而自动添加反向排序边。通知工作获分配不保证其 kernel 内的同步等待结束，这类业务协议的进展由 kernel 自身负责。

这条约束区分三个时刻：

| 时刻 | 要求 |
| ---- | ---- |
| Host 安排通知任务 | P 及其前置路径先于 W；通知排序边不改写原数据 fanin 和任务 ID |
| Device 进入等待路径 | 必须保证对应通知工作已经取得可独立推进的执行机会或已经提交；不能让 W/其消费者先占满窗口，再指望未分配的 P 获得位置 |
| 通知值实际到达 | 可以早于或晚于 W 登记；条件真实满足后才允许 FINISH 发布 W 的逻辑完成 |

“先安排”不等于“通知值必须提前到达”，也不要求 P 的整个计算/传输已经结束。Device 不能只凭 Host rank 宣称 P 已经运行：通知工作已领取且资源协议保证其进展，或外部发送/传输已经提交，均须有相应的权威证据；仅有候选、未就绪 hint 或 async 容量不算证明。无法建立该证据时，异步适配仍未完成。

P 是本地可执行 task 时，W 领取前还须确认 P 已全体发票或合法终结；同类可由统一目录头部证明，跨引擎须增加明确的通知供给前驱检查。这项检查属于发票与进展协议，不向 W 的数据 fanin 注入“等待 P 逻辑 DONE”的新语义。P 已发票后仍需遵守既有资源服务规则，保证通知能实际推进。

Host 的职责限于加工已有图、参数和依赖信息。跨卡通知沿用原通信协议：发送卡只有 notify、接收卡只有 defer_wait 完全合法；两边不必各自同时具有 write 和 wait。未参与此次通信的 rank 不需要提供计划、空 manifest 或参与任何新增同步。禁止为排序增加 allgather、对端计划交换、全域屏障或跨卡 run 编号对齐。

排序位置只在本卡目录内有意义。对远端来源，Host 不补本地虚假前驱，也不宣称远端已经发起。W 按原机制登记并物理返回、释放业务 slot；独立 CHECK/FINISH 观察实际通知，消费者仍等 W 的逻辑 DONE。元数据缺失或来源不能由现有信息确定时，不构造推测边、不读取设备业务数据，也不拒绝原本的调用方式。

本卡同一计数器上的重复 SET 只有在真实数据/Graph 依赖证明其先后且值不下降时，才选择最早达到阈值的生产任务。ADD 的初值或贡献关系不明、reset/epoch 不明时保留原条件等待，不根据 expected 虚构 epoch。已知循环仍由 Kahn 检查报告。

本设计只处理独立的 defer_wait task W；不假设用户业务 kernel 内任意登记等待后继续阻塞。每核复用一份 `SchedulerAsyncLocalState` 作为独立服务槽，单次只持一个 CHECK/FINISH 工作项；它不占四个普通 token，也不是第五个业务执行槽。W 登记并物理返回后归还执行位置，逻辑完成仍等真实条件结清。

```text
P 及其前置路径先被安排，并保证通知工作能独立推进
    → W 登记 defer_wait 并物理返回，释放执行位置
    → B 可以在合法前视窗口内等待 W，保持原 owner
    → 通知早到或随后到达，CHECK 观察真实条件
    → 唯一 FINISH 结清 W 的信用，B 才获得实际执行许可
```

独立等待槽保证持续检查；可确定的本卡 P→W 约束防止通知生产任务被等待者挤掉。远端通知不依赖新增 Host 门槛。保持已知约束图无环，不靠转交已领取的 B、扩充隐藏槽或把 defer_wait 改成 wait 自救。发现循环时，应报告具体数据/控制/通知关系，不能静默改写用户语义。

`runtime_maker.cpp` 从生成物的 `simpler_hbg_notification_plan_v1` 与 `simpler_hbg_notification_kernel_flags_v1` 读取已有通知和等待的内部描述，用真实绑定的 Tensor/scalar、block/lane 及只读 Host 通信域信息定位本卡关系。async 布局容量不是 may-defer 标记，不能用容量或 kernel 名猜 P/W。普通 opaque kernel 不新增执行前提；元数据不完整仅表示对应排序关系未知。

本次开发允许补齐 PyPTO→Simpler 的代码生成适配：编译器从已有通知和 defer_wait IR 提取内部元数据，由 Host 规划消费。用户脚本、前端语义及 kernel 计算保持不变；通知元数据不能通过实际运行 kernel、读取尚未生成的业务数据或按函数名推断来获得。

保留 `DeferredPredecessorLeavesIndependentCounterPublisherRunnable` 反例：其原提交序把 P 放在 W 的多个消费者之后。修复验证应由真实 Host 规划读取通知关系、产生 P 先于 W 的顺序，再观察实际执行及完成，不能手工预置 P 已完成或删除失败断言。验证 P 已安排而通知仍晚到，以及物理返回与逻辑完成的区别。跨卡单向通信保持原路径，不为 Host 排序扩展通信测试矩阵；源码次序改变不等于实机通过。

`scheduler_service_async_slice()` 每次处理一个 CHECK 或 FINISH；CHECK 最多观察 8 个条件，未满足时保存进度、归还 lease 并轮转。一个本地服务槽不限制全图只能有一个 W：多个等待记录留在共享表，不能逐个占满普通窗口。任何核不得带 CHECK/FINISH lease 进入业务 kernel；持续命中快入口时也按第 3 节预算服务控制路径。

复用现有 N16/C16/D32 完成字：物理返回归还相应份额，非空异步组的信用由唯一 FINISH 结清。取票耗尽、kernel 返回、P 已安排都不能伪造 W 的逻辑 DONE；保留条件登记与最后返回交错的 join 证明，以及真实归零原子的 DFX 因果身份。

### native Graph、predicate 与安全退出

Host 预构造每个实例的 Call、边界 alias/view、scalar 和 heap 地址；Token 用 run + node_ref 区分实例。拓扑分层覆盖实例实际数据、通知和 ENTER/CHILD/EXIT 控制关系；排序不改原 ref 和参数绑定。child 只有在元数据已发布、必要的 ENTER/激活条件满足时才可领取；执行仍等待真实计算前驱 DONE。普通前驱全体发票不能替代实例激活或控制/通知门槛，目录投影必须保留这些约束的检查位置，不能因只看可执行节点而丢掉实例边界。外部消费者仍遵守 EXIT 合同；未发布实例不发票、不读取占位指针，不补 Device 动态生成旁路。

predicate 仍在合法执行边界读取；false 分支按既有整任务/组语义完成 skip，贡献恰好一次，不能让 MIX 某 lane 跳过而伙伴进入等待。提前取得固定 async 定位信息不代表创建 writer 的可写状态、登记条件或发布信用。

错误、取消和 EXIT 按责任清点：TICKETED 新票、OWNED pending、正式 MIX/cohort、INFLIGHT、完成贡献和 async lease 都必须有最终归宿。只能按现有错误协议终止责任，不能转交其他核重跑同一 execution；inflight 等真实返回后退休。Host 不能以 cursor 发完或某一核本地窗口为空就回收 Call/Binding/Graph heap。

需要同时证明“所有执行单元已实际返回或按失败合同终止”“没有遗留的已领取责任”“组资源和 async/control 已安全退休”。run/实例身份及协作 generation 不回绕，越界或旧身份不能当作正常重复消费。内部布局与导出协议变更时同步更新 Host、Device、reader 和测试。

### 缓存与存储合同

- 共享控制按 **128B atomic-only 隔离**设计；64B 是 Scalar DCache 操作粒度。控制行内不混入 ordinary store 或 DCCI 数据。
- 初次 Claim 成功后，对 Call 和需要接收的固定数据做定向 DCCI、尾 DSB 与 compiler barrier；数据观察可提前到 WAIT_FANIN，不能因为 Host 已构造就省掉。
- 每核 token 只由固定 owner 写，可使用 owner-private 存储；初次执行授权和共享完成仍使用原子协议。本核等待状态不作为跨核交权信号。
- 可写动态 Binding 延后到最终许可后由固定 executor 填写；准备者不能修改属于另一 executor 的可写上下文。等待期间保留参数与观察进度，必要的缓存可见性协议不因固定归属而省略。
- 提前观察参数地址不等于提前观察业务 Tensor 内容。受前驱或 async 保护的数据仍在实际依赖满足后执行必要的数据观察。
- 沿用启动路径的 DCache 清理合同，不引入每 task 全 cache DCI，不将 DCCI 误当按地址 invalidate-only；SSBUF 使用其专用 release/acquire 屏障。
- 共享记录、Call 和 Binding 在 run 内不复用地址，直到安全退休。记录 token、SSBUF、Host 初始化和实际 H2D 成本，不新增用于跨核再次认领的元数据或复制 probe 的固定大 arena。

## 5. 开发步骤、DFX 与验收

### 按目标协议分阶段完成

先完成本设计，再继续工作区的生产改动。**第一项交付必须是 task 有序供给，不能先继续修局部抢票、转交或替代任务机制。** 以下为纵向交付顺序；已有清理改动随相应协议整合，每阶段通过所列边界再本地提交，不将设计修正当作代码或实机结果。

| 步骤 | 依赖 | 主要文件与入口 | 可审查的完成条件 |
| ---- | ---- | -------------- | ---------------- |
| 1. Host 约束图、分层与目录 | 当前 Call、数据依赖、通知对应关系及 Graph 实例映射 | `host/task_catalog_plan.h::scheduler_plan_task_catalog`；`host/dependency_plan.h::scheduler_validate_dependencies`；`scheduler_task_catalog.h`；必要的通知规划元信息 | 完整约束图无环；按 `(level, -b_level, node_ref)` 排序；逐边 rank 校验，P 在 W 前；生成两个一致投影，不改业务身份，不引入层完成屏障 |
| 2. 按拓扑目录有序发票与固定 owner | 1；所有入口一起接入 | `scheduler_issue.h`；`scheduler_execution_tokens.h`；`scheduler_cluster_mix.h`；`scheduler_cooperative_discovery.h`；`scheduler_admission.h` | 同类头部全体发票才推进，跨引擎前驱供给检查完整；不因本核去重跳项；发票门槛不变成全任务完成门槛；四槽、单 task 单 block、固定 owner 和协作门槛成立 |
| 3. 既有快速执行与完成路径接合 | 1–2 | `scheduler_prepared_execution.h::scheduler_acquire_execution_token`；`scheduler_execution_tokens.h::scheduler_progress_execution_tokens`；`scheduler_dispatch.h::scheduler_fill_own_slots`；`aicore/resident_scheduler_impl.h`；`scheduler_completion.h` | 已有 token 不等发票头部；Claim/acquire 和必要完成各一次；同核表示切换不重复持责；窗口满时仍服务控制路径 |
| 4. Graph 和 standalone defer_wait 边界 | 1–3 | `host/native_graph_plan.cpp`；Graph 控制入口；`host/async_completion_plan.h`；`scheduler_async_finish.h::scheduler_service_async_slice` | 实例激活/通知门槛完整；P先于W排序且等待前通知链已可推进；本卡早到/晚到通知及安全退出验证，远端原协议不变；独立等待槽不占普通窗口 |
| 5. 删除残留、同步协议及测量 | 各步骤同步删除，最后核对 | `scheduler_types.h`；`scheduler_prepared_layout.h`；`host/scheduler_protocol_export.h`；`simpler_setup/tools/hbg/`；相关已有回归 | 无 PARK/跨核重领、旧 scan cursor/round、ready 中转或重复 builder；内部版本及所有生产者/消费者一致；L0 与 L3 证据对应冻结源码 |

步骤 1 的 Host 排序及校验不能单独宣称“有序调度完成”，步骤 2 的入口统一是放行整个运行时的必要条件。partial 实现不能通过暂时禁用 MIX、sync_start、Graph 或 defer_wait 来冒充完整通过。保留 Host 预绑定、原任务身份和业务依赖；排序只修改目录优先级及必要的供给计划，不重做 Device Build，也不添加模型专用捷径。

上表 Host/AICore 路径以 `src/a5/runtime/host_build_graph/` 为根，未写目录的调度器头文件位于其 `runtime/scheduler/`；`simpler_setup/` 以仓库根为准。列出的是当前改造入口，不要求继续沿用全部旧函数名。保留必要协作路径不等于保留旧普通路径兜底；各功能通过后移除它对应的旧分支。提交前冻结实际源码并核对上游同文件改动，不能复用文档中某个过去 PR 状态代替当前核对。

使用当前 a5-onboard 构建流程，只构建所需运行时。遵守仓内 CCEC 函数跳转约束，对实际编译目录运行 A5 IR 检查，尤其关注过度对齐局部对象地址逃逸到 outlined call；不通过全量强制 inline 掩盖问题。检查结果、人工豁免范围和实机验证分别记录，不把未覆盖的编译单元写成全部通过。

### 功能验证使用明确事件

复用现有 ordered execution、gm_mix、async_control、condition_boundaries、Graph 和协议导出测试；只补新的责任边界，不为每个 helper 增加镜像测试。CPU 测试用显式 gate/ACK 构造状态，A5 验证真实数值、身份及退出；不以睡眠或相对运行速度证明某次交错发生过。

| 场景 | 必须观察到的结果 |
| ---- | ---------------- |
| Host 顺序 | 数据与通知约束组成完整无环图；验证正向层号、层内 b-level 和稳定 ID 次序，逐边 rank 合法；两个目录无遗漏/重复，非法输入和排序启动前拒绝 |
| 分层无执行屏障 | t9 的前驱完成时，即使同上一层的独立 t11 尚未完成也可执行；不增加全层完成计数或屏障 |
| 静态检查与动态进度 | 关闭前缀表示本类供给进度，不等于 DONE；不能替代另一引擎状态；Host 已完成豁免、Graph 激活及通知门槛不丢失 |
| 前项全体发票 | A1 尚未被任何核领取时，依赖 A 的 B 不得持票；A0 退休后可补 A1，不领后项 E；A 全体发票而尚未 DONE 时，B 可以持票但不能执行 |
| 跨引擎取票 | 两类目录各自推进，后继领取前确认跨引擎前驱已全体发票；无依赖的另一类项无需等待；执行仍等待真实 DONE |
| 宽 SPMD | 4 个核/8 个 block 等已有小拓扑验证：同 task 未退休 block 去重；退休后继续补头部余块；全体发票才推进后项，不等待全部计算完成 |
| sync 容量 | 同时驻留需求超过总容量时在启动前拒绝；只是暂时忙碌则等待准入，不把两者混为一谈 |
| unready 提前领取 | gate 保持前驱未完成时，后继已持票并进入 WAIT_FANIN，kernel 未进入；释放后恰好执行一次 |
| 已有 pending 接续 | 后续 token 在前一 kernel 返回前已完成 Claim/acquire；返回后不再取新票或 Claim，同入口执行 |
| SPMD 与跨 task 分布 | 同 task 的 current+pending 去重；无有效票退款；快核可继续取剩余工作；不因前视制造多余尾波 |
| 全窗口 WAIT | 显式保持四项前驱未完成；四份责任、位置和准备状态不变，不领取第五项；前驱完成后由各自原核执行 |
| MIX 未来组等待 | 未 ready 的未来组保持归属与准备；其他正式组和独立服务继续推进；不换组、不跨簇重领、不在 arming 时二次发票 |
| 领取后的唯一执行核 | 同一 execution 从第一次有效票到退休始终是同一 executor；其他核不能执行；错误终止不转成异核重跑 |
| 通知先于等待 | 可确定的本卡通知来源 P 满足 rank(P)<rank(W)；验证其实际通知路径先获得推进机会；W登记后通知才到达仍可正确FINISH；跨卡不以本地排序伪造远端状态，不预置P完成冒充修复 |
| 快路径持续命中 | CHECK/FINISH 与 Graph control 得到有界服务；普通窗口有空位置时继续新票发现，不饿死已有等待责任 |
| MIX/cohort 交错 | ordinary/MIX/sync 遵守统一目录头部与各自资源协议，完整组责任或真实 seal 才关闭供给；已承诺 cohort 次序不被破坏；复用现有 lane、stage/hold 和相邻 cohort 覆盖 |
| Graph 与 predicate | 多实例不串参；ENTER/EXIT 不提前；false 分支合法退休，未发布元数据不被读取 |
| async 与退出 | 条件早到/晚到/登记中满足均不提前 DONE；物理返回释放资源；所有未退休责任阻止成功退出 |

### L3 必须展示真正的前视和串行链

事件统一关联 run/node/block/lane/token/固定 owner，并能还原 Host 层号/rank、目录位置、共享头部及跨引擎前驱供给进度；区分头部未发完、本核去重、initial ticket、Claim/acquire、WAIT_FANIN、最终许可、kernel、完成、MIX/cohort 和 async/control。显示本核原位置等待跨度，但不将等待当作持续 Scalar 工作。

当前有序供给的三个阶段映射到 `scheduler_query_issue_head`、`scheduler_issue_fanin_ready`、`scheduler_task_supply_closed`。保留这些阶段的供给语义，新增 Host 排序及通知边时同步其身份和等待原因；不得把等待发完票显示为等待计算完成。atomic、ld_dev、DCCI/DSB 沿现有 primitive 记录展开，缓存命中不伪造共享查询事件，Host 导出和 Python 消费者一起更新。

必须回答：

- 每次进入 kernel 时，已有多少具体 pending？多少已完成 Claim/acquire，多少只是 TICKETED？
- 对前序 A 的余块，本核退休后是否先补 A1？首张后项票是否晚于 A 全体有效票？头部供给等待有多少来自宽 SPMD、本核去重或真实成员准入，Host 排序是否让独立分支更早获得机会？
- 对通知 P→W，Host rank 是否满足约束？W进入等待前有什么实际进展证据？通知到达和W逻辑完成是否分别记录？
- 返回后的下一次调用来自已有 pending 还是临时补给？若缺少 pending，原因是前序合法票优先、控制/通知门槛、窗口容量、同 task 去重、资源会合，还是没有获得补给机会？
- 参数接收是否早于前驱完成、早于本核前一次 kernel 返回？后继进入时还缺哪些操作？
- `G_worker` 和 `G_dep` 中是否出现新票、重复 Claim、目录扫描或 READY 中转？快/慢路径分别占多少？
- issue_head 观察/推进 CAS、首次取票 FAA、Claim CAS、供给/执行依赖确认和完成原子分别多少？已持有任务就绪后是否再次取票或 Claim？不能用一个总 atomic 数替代分类。
- 所有 ld_dev、atomic、DCCI、DSB 独立标识；residual 包含普通访存、控制和 profiling，不当作纯 Scalar 指令时间。
- token 扫描含有多少 kernel 调用？inclusive TokenScan 不能当成可移除的调度耗时。
- SPMD 的参与核数、全部 block 起跑跨度、每核执行次数和第二轮尾部是否变差？是否有闲核拿不到已预留的工作？
- 未领取任务的供给、前驱和通知者的可达性、组资源退休与 CHECK/FINISH 是否可证明进展？

延续任务计时与 Scalar observer 的编译分离；必要事件的原始时间戳保留，导出/编码尽量不干扰中间计算调度。记录丢失、截断或不完整因果关系必须显式失败，不能用缺失事件表示零耗时。

`G_worker/G_dep` 使用真实 kernel 起止时间戳，不能用包含 setup/receive 的显示条边界代替。按任务对列出各次采集的间隙、低于 5 µs 的比例和最坏值，并同时列出有 pending 的覆盖率；中位数低于 5 µs 不代表所有交接达标。协作等待、真实依赖等待和调度开销分别归因，但原始间隙必须完整保留。

### 性能与保留规则

- 整体性能目标仍是相对本次重构前冻结的 `2f61d0c6c` 提升 40% 以上；分阶段对已验证父版本的收益不能相加替代这项最终对照。验收集切换到下述八组后，原始 runtime 也须使用相同八组 kernel、输入和计时方式重新配对，不能沿用旧用例集合的总和或把阶段 baseline 当成最终分母。5 µs 交接与整体 L0 收益分别验证，任何一项的局部样本不代表目标完成。
- 最终性能以完整 device L0 `chip.run.runner_run.device_wall` 为准，覆盖 AICPU 设备入口到退出；每组 1 次预热、10 次计时。L1/L3 用于定位，不用首末业务 kernel 包络代替 L0。
- 性能验收使用 `dsv41_scheduler` PR #3087 的 16 组配对对照，保持各组原 kernel、输入、图、sync_start 配置一致。16 组均须通过数值验证；8 组非通信场景的 L0 device 均值之和决定是否保留，同时披露逐项回退。八组通信场景记录性能，但不作为性能保留门槛；Graph、MIX、sync_start、defer_wait 的通用功能另按上述合同验证。单个 SWA 或其他模型的收益不能替代全套结果。
- 所有硬件任务使用 `task-submit --device auto --device-num N`，在任务内使用并记录实际 `TASK_DEVICE`；attention 单卡，同一分配内做配对对照，记录先后顺序。
- L1/L3 也区分预热与正式采集；不从单条交接里扣除整次 L1−L0，不累加跨核或嵌套 span 冒充串行耗时。5 µs 判断同时报告低扰动观测、采集开销和未达标原因。
- 对照完整目标方案前后，并保留各独立改动的归因。综合有收益才保留性能改动，1% 收益也记录；无 ±5% 忽略区间。有收益的独立修改分别本地 commit，未经允许不 push。
- Host 准备、实际 H2D、GM/SSBUF 和峰值内存单列；不能把工作搬出 L0 区间后不报告。历史测量与详细采集留在实验结果中，不回填本设计作为实施完成状态。

验收需要同时证明：具体 pending 确实提前持有、已有工作能直接接续、SPMD 分布没有以尾波换局部收益、协作与异步语义完整，以及完整 L0 有实际收益。四个 token、一次 CAS 或某个局部 span 变短，都不能单独替代这些结果。

非通信考核清单：初版、DSL 半同步、DSL 全同步的 `c2a_full` 三组，加新版 PyPTO-lib 的 `swa`、`c2a_full`、`c1a_reindex`、`c1a_reuse`、`engram` 五组。各组均 1 次预热、10 次正式测量。`layers_19_21` 的大图回退单独修复并报告，不混入八组之和。
