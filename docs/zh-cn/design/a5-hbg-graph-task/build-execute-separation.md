# HBG 调度设计：提前准备、最小领取与轻量等待

**状态：固定归属、有序发票和 Host Kahn 分层排序已接入。** Host 根据现有数据与 Graph 控制依赖排序；Device 在前项全局发完有效票后开放后项，不要求前项整体执行完成。本卡通知配对和 PyPTO 通知元数据导出暂缓，后续由真实场景驱动评估，见 [#3096](https://github.com/hw-native-sys/simpler/issues/3096)。现有通信协议和延后完成机制保持不变。

本文以 `atomic_probe/pa_scheduler/cross_core_only_scheduler` 的实际执行流程为参考，只保留当前实现与拟修改合同。供给顺序是 **Host 读取现有数据与 Graph 控制依赖 → 拓扑分层并在层内排序 → 投影为两类任务目录 → Device 按序发完前项有效票再推进 → 固定 owner 等待并执行**。不能仅移植四 token 和提前领取，却遗漏有序取票。HBG 的不可变 Call 由 Host 构造，Device 仍有接收、owner 相关准备和动态入口工作；不能将这三者统称为“Build 已全部在 Host 完成”，也不照搬 probe 的任务生成、TensorMap 或 Device Build 队列。调度优化按 12 组 AICore L0 均值总和验收；功能正确且明确减少冗余 atomic／ld_dev 的候选允许不超过0.5%的总和回退，超过0.5%则撤回。

修改范围为 A5 HBG 调度器、必要的 Host 布局、内部协议与 DFX；不新增 PyPTO 通知接口，不修改用户 kernel、前端语义或参数 ABI。

调度优化必须由通用任务状态、依赖和资源合同驱动，不按模型名、特定 task 编号、shape 或固定图拓扑选择策略。SWA 等单例的泳道图用于提出和验证机制假设，不能替代通用性验收。

**当前保留标准（2026-11-16 更新）：** CntNotify 扩展逐项对比上一保留版本的12组 AICore L0 均值总和；总和回退超过0.5%即撤回，不超过0.5%可以保留，并记录每组变化。MIX 握手的 +0.224% 总和回退在保留范围内。

性能主指标统一为 HBG 与 percounter 的 **AICore L0 wall**：各实际执行核的最晚 kernel end 减最早 kernel begin，包含中间调度和依赖等待，排除首末 kernel 外的 AICPU 启停开销。每组预热一次、正式测量十次；双卡每轮先 Device 会合，各卡按自己的时钟计算，再逐轮取 rank 最大值并求均值。缺失或非法端点直接判无效，不回退到 Device wall 或旧 percounter 调度循环窗口；历史数据保留原口径，不重新贴标。

原全局 ready_queue 调度逻辑已由任务目录与固定 token 取代，不能将 ready 发布/领取恢复为备用执行入口。保留 AIC/MIX、AIV 两个不可变有序目录，不扩增目录数量；这不限制 task 内必要 claim atomic 按 128B 分地址，目录拓扑与计数器布局是两回事。各核私有发行游标只在当前 task 全局发完有效票或合法终结后前移，共享的是每个 task 的供给证明。它约束新票供给，不阻止已有 token 执行；跨引擎依赖另外检查前驱供给进度，不新增串行化两类独立任务的全机单游标。任务 ready 时不进入全局队列，也不重新竞争执行权；本核 ready mask/direct slot 只是已有责任的执行表示。普通、MIX、sync_start 统一遵守供给顺序，仅保留各自资源与成员协议。

<!-- markdownlint-disable-next-line MD033 -->
<a id="a5-hardware-principles"></a>

## A5 硬件与调度统一原则

以下七条采用 **2026-11-16 用户给定的 A5 硬件工作模型**，作为本设计及其他 A5 调度设计的共同前提。100 ns+、约四拍是该前提给出的量级，不是本次新增实测或跨芯片、跨负载的固定延迟保证。历史探针保留原始条件与结果；其旧工程建议若与本节冲突，后续设计按本节修订。当前实现与拟修改方案仍须分别标明。

| 编号 | 硬件前提 | 对设计的直接约束 |
| ---- | -------- | ---------------- |
| 1 | `ld_dev/st_dev` 绕过 Scalar DCache、进入 L2 cache；atomic 不进入 cache，走 HBM 原子路径，单次并发增加 100 ns+ 量级开销 | Wait 用可靠 `ld_dev` 观察；单写者通知可用 `st_dev`；不能用 FAA(0)给同一读值再做一次确认。atomic 留给真正需要的竞争裁定与多写者合并 |
| 2 | SoC 总线对 `ld/st_dev` 和 atomic 均不保序 | 同核操作间有时序依赖，就在相应依赖边界显式加 DSB；成功 CAS/FAA、源码先后和 C++ `memory_order` 名字都不能代替硬件屏障 |
| 3 | STARS 有 event/notify SRAM 资源，不受 GM 同址读写的并发串行瓶颈限制 | 逻辑 DONE 已采用 CntNotify SET/load；其他 Wait 后端按资源合同推进，它不自动授予 task 的唯一执行权 |
| 4 | A5 上基于 GM 的同地址读取、写入都有并发串行代价 | 既要减少同址 atomic，也要控制同址 `ld_dev/st_dev` 的扇入和请求密度；不能把 atomic 改成多核无限紧轮询后就宣称争用消失 |
| 5 | 动态领取 task 仍须由 atomic claim 裁定；对保留下来的竞争性 claim，分地址 atomic 是消除同址争用的方案 | 优先降低同地址 atomic 并发，再减少总次数。独立竞争单元按 **128B L2 侧隔离粒度**布局，不能仅相隔 64B DCache 行；不改成固定核预分配或非原子的读后写抢票 |
| 6 | STARS SRAM 并发排队约四拍，可近似忽略，但入口带宽有限 | STARS 等待循环须以 `nop` 限制空轮询请求速率，避免无限紧循环耗尽入口；同时保留有界控制服务和实际唤醒延迟观测 |
| 7 | AICore SPR 等寄存器即使可映射访问，也缺少通用 event 的分配、释放方案 | 在资源分配、归属、复用与释放机制完成前，不把 SPR 当作通用 task event 池；既有固定用途寄存器协议不等于具备该能力 |

第 5 条的“唯一方案”针对 **必须保留的动态 claim 的同地址竞争**：继续优化它需要分散地址，不能用普通读写取代唯一裁决。删除冗余操作、提前 Build、缩短入口和改进 Wait 仍然有价值，但不提供另一种动态所有权原语。分地址也不能只是换几个变量名：需明确合法票区间、不重复不遗漏、同一 task 全部票发完的证明、跨地址发行顺序及 owner 固定不变，且不能在每次分散 claim 后再访问一个全局共享队头或汇总原子，把热点重新引入。

### 三阶段怎样应用这些前提

- **Build：尽可能并发、尽可能提前。** owner 无关的地址、参数与静态校验提前完成，owner 相关准备在归属确定后推进；批量数据写入和缓存维护在其发布边界完成 DSB，再发布可观察通知。不可变 Call 的准备不包含尚未生成的业务 Tensor 数据。
- **领取：atomic 只裁定一次必要的唯一责任。** 同一有效票已授予的普通执行权不再二次 CAS；仍须竞争的 claim 按 task/独立票区间分地址并以 128B 隔离。STARS event、可靠 `ld_dev` 和目录 hint 都不能取代该 atomic 裁决。
- **Wait：观察已定位条件，不重新 Build 或 Claim。** GM 路径用 `ld_dev`，单写者通知根据协议用 `st_dev`；观察值携带 task/epoch/sequence 身份传给下层复用。满足条件后按消费者获取合同执行 DSB 与必要缓存观察，最终 admission/MIX/sync_start/退出许可仍独立检查。逻辑 DONE 已映射到 CntNotify SRAM，其他条件通知继续按各自协议处理。

DSB 只针对同核前后操作的实际顺序依赖；跨核通知本身不要求发送后追加 DSB，也不等待接收方何时看到。单写者也不自动等于可以用 st_dev：需要 RMW、原子交权或不可分割联合状态发布的字段仍保留 atomic。DSB 不是跨核会合，也不使另一个核的 DCache 自动更新。发布数据→发布状态、读到状态→使用数据、依赖前次返回值的后续 atomic，都须写清排序边界；可以合并同一批无相互依赖操作后的屏障，不能把它移到已有中途消费者的最后一次运行结束。普通 cached 数据与 atomic-only 控制区继续隔离，避免 DCCI 把旧脏数据覆盖到控制字。

### STARS 与 SPR 的演进边界

当前逻辑 DONE 优先使用沿真实祖先链共享的 CntNotify 累计阈值；非计数器任务在有资源时使用每节点独占 SET 通知，否则直接观察 canonical GM gate。MIX 根据既有模式使用 SSBUF，或为 sync_start MIX 的远端 assigned/arrived/released 使用独立 SET/read 通知。CntNotify 随运行镜像持有并在每次 READY 之前复位；其他 STARS 接入仍需绑定已有 task/epoch/sequence，明确 event 申请、初始化、先通知后等待的处理、完成后清理、复用及异常清退；不能漏通知、把上一轮通知认作本轮完成，或在旧观察者仍访问时复用资源。通知与 payload 可见性仍按各后端合同排序，SRAM 低排队代价不免除这些责任。

STARS 的 `nop` 是入口限流，不是让有工作可做的 Scalar 无条件停顿：空轮询受控，已有可执行任务、合法 Build、defer_wait 服务和 EXIT 检查继续获得进展。具体间隔由入口请求率与发现延迟决定，不预置模型专用常数。SPR 在具备上述资源管理机制以前不作为该方案的替代品；A3 上的寄存器可达性实验也不能直接作为 A5 映射可用性的证明。

### 当前实现相对原则的差距

| 对照项 | 已有实现 | 后续要处理的差距 |
| ------ | -------- | ---------------- |
| 地址隔离与动态领取 | `SchedulerTaskControl` 各控制字、`SchedulerPreparedControl` 已按 128B 隔离；普通有效 FAA 票直接授予 owner | 同一 SPMD 的 `next_block` 仍是共享地址；跨 task 已分散不等于宽 SPMD 内已无争用。分地址 claim 是后续方案，不能写成已实现 |
| 观察与发布 | fanin 使用可靠 `ld_dev`；claim_seal/admission_state 使用无隐式 DSB 的 st_dev，日志/尾及错误信号保留既有发布屏障 | 继续按字段删除重复读与纯读取 FAA(0)，评估单写者通知；多写者计数与 CAS 仍保留 RMW |
| 同核顺序 | 现有 payload、Binding 和完成发布有显式屏障 | 逐条核对状态→数据、数据→通知、claim→后续依赖操作；不能以某 helper 使用 atomic 或形参带 `memory_order` 宣称整条协议已保序。本次未完成全量屏障充分性证明 |
| 等待争用 | 私有 fanin 前缀、地址绑定、MIX 观察值复用与有界服务已减少部分查询 | 逻辑 DONE 已迁移到 CntNotify；其他共享条件及 GM 回退仍可能同址争用，尚无通用 SPR event 池 |

## 1. 目标执行循环与当前差距

### Build、领取、Wait 的统一原则

这里的 task 原则覆盖每个执行单元实际调用前所需的工作，不把函数名或现有状态枚举当成阶段定义。普通任务领取的是具体 block/lane；MIX 领取的是逻辑 block 的组责任，sync_start 另有真实成员准入。

| 原则 | 应实现的行为 | 必须一起衡量的代价 |
| ---- | ------------ | ------------------ |
| Build：并发、提前 | 将地址解析、参数组织、metadata、固定 context、所需接收及发布，放到各自最早合法时机；owner 无关部分由 Host 或合法准备者提前完成，owner 相关部分在固定归属后、依赖等待期间完成 | 准备结果真正早于前驱完成的覆盖率、还留在启动入口的工作、准备但未使用的包、Host/H2D/GM/本核存储成本 |
| 领取：慎用 atomic | 动态唯一 owner 由 atomic claim 裁定；必要 claim 按 128B 分地址降低同址竞争，多写者合并保留 RMW，普通有效 FAA 票已经授予的责任不再竞争第二次 | 同地址并发请求优先于总次数；有效票、超额票/退款、协作准入、完成贡献分别计数；提前固定 owner 可能造成的等待 |
| Wait：已定位条件的轻量观察 | DONE 使用 CntNotify st_dev.b32/ld_dev.b32，GM 回退与其他条件保留各自协议；复用正向证明，不重做静态 Build 或 FAA(0)确认。每次未就绪返回推进循环，不新增空转循环 | 失败轮询次数和请求密度、条件满足后的发现延迟、发布者进展，以及夹在等待循环中的必要控制工作 |

三者可以交错，不能强制排成三个互不重叠的流水段：

```text
Host：固定信息构造、Call 发布 ───────────────────────────────┐
Device：有序供给 → 唯一票/成员责任 → 本核接收与固定准备 → Wait │
         其他执行单元的准备、kernel 和控制服务可以与之重叠     │
                    条件与本核资源齐备 → 最终入口 → kernel ←─┘
                                          ↓
                           结果发布、完成贡献、资源退休
                                          ↓
                               后继观察逻辑完成并执行
```

“Build 完成”至少区分 **静态 Call 已发布**、**本执行者所需的固定准备已完成**、**动态入口条件已满足**。predicate、实际输入可见性、当次资源许可、async 运行期状态不能仅凭参数地址稳定就提前确认。接收方的本核 cache 维护也不能由另一 Scalar 的预取自动代替。另一个核执行 kernel 时，本核可以推进合法准备；同一 Scalar 正在同步执行 kernel 时，不能假定调度循环仍在并行运行。

固定 owner 不允许转移，但准备时机和领取时机仍是两个独立选择。四 token 是有界容量，不是“越早占满越好”的证明。若已领取 B 的核正在执行较长的独立 A，B 可能在依赖满足后继续等该 owner；这项延迟必须保留在结果里，不能只统计 Claim 指令或 Wait 函数。现有有序供给、四槽和单核同 SPMD 一份未退休责任仍是约束；调整领取时机属于后续独立候选，不隐式改变这些规则。

Wait 中“地址已知”也不等于所有 task 共享一个全局地址：当前每个前驱有 canonical completion 和可用时的 CntNotify 通知。GM 回退或其他共享条件的多个消费者仍可能竞争同址；SRAM 也有入口带宽限制。可靠读取不代表无限带宽；更短的单轮检查可能增加单位时间的请求量。负向观察不能作为永久证明，降低查询频率也不能连带降低实际 kernel 入口的 EXIT/错误检查。

完成发布是三条原则的闭环。执行资格要求已有 owner、固定准备完成、依赖逻辑完成、本核可执行、协作许可有效且无终止条件同时成立；之后还有不可提前的入口操作。前驱已返回但完成贡献尚未发布，或 defer_wait 债务尚未结清时，后继仍不可执行。因此不能仅凭这三个局部阶段变短就宣称达到整图最优或 5 µs 目标。

### 目标的准确含义

有已发布且适合本核类型的工作、本核有补给机会时，提前持有具体待执行单元。前一 kernel 返回并完成必要的结果发布后，若已有 pending 的全部执行条件满足，就直接进入下一 kernel，不再先做一轮全局发现、重新分配 block、构造参数和 READY slot 发布。

这里的 pending 必须含具体 `node_ref/block/lane`、唯一执行票和所有权。只保存 task ID、fanin view 或一个“可提前准备”的 hint，不满足要求。当前普通 `WAIT_FANIN` 已完成 Call 校验、Binding 接收、固定 metadata 与 sub_block_id 准备及其发布；仍可能等待依赖、predicate、实际输入观察和最终资源许可。只有票号、尚未完成接收和固定准备的 `TICKETED` 不能计为同一种准备程度；MIX 的预校验、正式 lane 接收与到齐许可也不能混称为一次 Build 完成。

“有 pending”不等于“有 ready”，也不等于任意时刻都能填满窗口。元数据未发布、真实依赖未满足、MIX 伙伴忙碌、sync 成员尚未到齐时，必须保留等待语义。没有下一项可用 token 的补给路径单独计时，不能混成快入口已达标。

分别测量两个间隙：

- `G_worker`：同一核下一次 kernel 开始减前一次 kernel 结束，回答本核能否快速接续。
- `G_dep`：后继首个 kernel 开始减真正最后一个计算前驱的最后 kernel 结束，回答跨核依赖交接是否接近 5 µs。异步场景另记录最后逻辑条件满足到后继开始，不能把物理返回当作逻辑完成。

同时记录同一 SPMD 任务的全部 block 起跑跨度、实际参与核数、每核执行次数和尾部。仅首个 block 早启动、其余 block 多执行一轮，不算达到目标。5 µs 是拟达到的交接预算，整体收益仍由完整 L0 实测判断。

### 先确定每核运行的主循环

```text
启动：接收 Host 已发布的镜像，初始化本核 token 窗口
循环：
    处理退出、错误以及上一 kernel 必须完成的结果发布
    到服务预算边界：只推进有界的 async/control 服务，不领取新任务
    已有 READY slot：直接到最终入口许可检查
    已承诺的 MIX/cohort：推进会合、准备与退休；产出可执行 slot 就立即返回执行
    已持有的普通 token：观察未确认依赖；找到可执行项就立即返回执行
    当前没有可执行项：推进新 admission/MIX 发现和普通有序补给
    每个补给分片完成后：重新检查已有 slot、协作责任与普通 token
    全窗口等待：保留固定归属，继续有界地服务通知、控制和已有协作责任
    有进展但无可执行项：继续下一片补给；不因为固定分片数而进入无意义退避
    无可推进项：有界轮询，不持有 lease 等待外部条件
```

### 执行优先与控制进展的边界

默认原则是 **已有可执行任务优先，空闲时补给**。“可执行”必须同时满足本核固定 owner、数据依赖、predicate、MIX 资源与 sync_start 许可；只完成 Call/Binding 准备还不能跳过 ACK 或最终入口检查。

工作分成三类，不共用一个“维护”分支：

1. 必要控制：上一 kernel 的完成发布、已有组的到达/退休、到期的 defer_wait CHECK 与已取得责任的直接 FINISH、Graph 控制、错误与退出处理。这些动作不得被连续计算无限延期，按现有有限服务预算推进；服务预算到期不构成领取新票的理由。
2. 已有执行责任：先读取本地 READY 标记，推进已有协作责任，再检查至多四个普通 token。得到 direct slot 或本地 READY 后，立刻退出发现/补给路径；实际 kernel 入口继续进行动态许可检查，并且只发布一次执行许可。READY 或 direct slot 被拒绝后，必须推进已有协作与控制责任再重选，不能因为本地 READY 标记一直存在而饿死解除许可的路径。
3. 新供给：只有没有选出可执行项时，才检查本核目录头及共享供给证明、检查发行前驱、领取新票或提出新 admission。已有未就绪责任不禁止合法预取；不得回到“只要存在 current/pending 就停止全部发现”的策略。

普通 fast path 不得因为累计执行达到服务预算而被强制切到 refill。除到期的必要控制服务外，已有 READY slot 不得先走一轮 Graph 控制和完整 fill 才被选中。正式 SSBUF MIX 与已提交 cohort 的资源限制保持有效；被入口拒绝的普通 slot 仍按现有恢复/等待合同处理，不能转移 owner。

补给分片是让出检查的边界，不是限制空闲核只检查一个条目。本核有待重新检查的执行/协作责任时，发行头每次最多推进一个已关闭目录项，发行 fanin 每次最多确认一条前驱；保留发行头和已确认 fanin 前缀，下轮继续推进。完全空闲的核仍可按现有预算连续扫描目录，不增加无效的服务绕行。每次新普通票只接收一个 block 后返回选择路径；活跃核在下一次领取前重新考虑已有工作。前序供给约束、同一 SPMD 单核最多一个未退休 block 和通知先于 wait 的排序要求不变。

实施顺序：先拆分已有协作服务与新任务发现，再把统一的已有执行选择放到主循环和补给分片之间，最后将服务预算与 refill 解耦。复用现有 DFX 函数区间，新增实际函数时同步命名；不新增共享 ready 状态或缓存正向执行许可。

功能验证覆盖 READY 直接选择、伙伴未到齐时仍能预取、已有 MIX 到齐后优先启动、发行头跨分片继续推进以及 async/control 的有界服务。性能仍按同一冻结输入的 12 组 L0 总和判断；L3 用于检查发行查询是否仍阻挡已有执行，不能把组内起跑跨度直接当成 L0 收益。

常规补给路径中，普通 token 的 fanin 已满足时，先用本核 admission 的 current/pending 字段做否定过滤：任一非空就保留 token，不填写和发布必定无法执行的 slot。不提前调用共享许可检查，也不缓存正向许可；本地字段为空仍须在实际 kernel 入口检查共享 admission/MIX 状态。cohort 领取前的有限窗口使用下述独立许可，不能从此否定过滤反推正向许可。已有 slot 遇到新的跨核发布时，原有本地回收路径继续有效。

`pending_task_ref` 仅表示尚未确认提交的本地候选，不拥有共享资源。`scheduler_refresh_local_obligation` 确认提交 tail、日志 task、epoch、order 和 ADMITTED 后，若记录与本地候选相同，就同时清除该重复候选；不能再要求取得全局提交锁来清除它。真实 current obligation、成员领取、seal 与 kernel 入口许可继续独立生效；不同任务的候选和未验证发布不得清除。

一个已有 token 完成后可以解锁本核另一个 token，返回应首先沿这条局部链继续。有序取票是补给入口；已有 token 的执行不能以先推进 issue_head 为前置条件。MIX/cohort 的已承诺责任、终态处理和到期的控制服务仍按各自协议处理。

### sync_start 领取前的有限就绪窗口

仅延后本核提出新 admission，不能阻止其他核提交 cohort。`scheduler_can_delay_cohort_claim` 在本核尚未取得有效票、未确认无票、pending 为空时，允许已持有的就绪普通 token 先执行。current 必须为 `sync_start`；GM MIX 还要求本核 `gm_mix.phase == IDLE`。CLAIMED、PUBLISHED、ARRIVED、RUNNING 或 RELEASED 的 GM 组不使用此窗口。

- `scheduler_fill_ordered_slots` 观察 cohort、恢复本核普通 READY slot 后，在 GM MIX 服务或纯 cohort 取票之前冻结现有四 token 槽的候选掩码。每槽只考虑一次，后续补给不加入窗口；拒绝后恢复原 token 也不重建掩码。
- 仅推进依赖已满足的普通 token。未就绪槽立即移出本次候选掩码，但保留原 owner 和执行责任；不等待它变 ready。每次最多选择一个 kernel，物理返回后重新进入正常主循环，窗口耗尽即恢复原 cohort 服务。
- 窗口不发新票、不补位、不越过有序供给头；常规补给中的 current/pending 否定过滤不放宽。prepared handle、已确认 fanin、predicate 与业务输入观察继续沿用原合同。
- 每个选中 slot 保存当前 admission 序号。实际入口重新验证 `slot.preclaim_order == admission.scan_cursor`、领取前状态、运行错误与本 lane 的 SSBUF staged/named 状态；不缓存正向许可供后续 kernel 复用。
- 本核已观察 cohort 并持有 `sync_hold`，不得提前清除 hold。AIC 尚未领取当前 GM block；AIV 即使已有远端 `assigned_seq`，本地 IDLE 时仍未 observe descriptor、报告 arrived 或取得执行 slot。coordinator 必须等待该 lane 到达，并等 release 后才能复用 descriptor，有限普通执行不改变组身份或允许提前 ARM。
- Graph 按当前 run 的 expanded task 身份区分实例，不能使用 Definition ID 代替。`defer_wait` 登记任务返回后仍保留逻辑债务；CHECK 和 Graph control 保持原有有界服务预算，FINISH 由 join 唯一 owner 当次直接完成。

进展依据是有限已有窗口与不可撤销的本核组接收边界：最多四项，每项只检查一次，不等待未满足依赖，不接收新的优先项。有限普通 kernel 返回后才领取或观察当前 GM 组，不会占住本核已接收的协作组再等待普通任务。若业务 kernel 内还有额外阻塞同步，其进展仍须满足既有 kernel 合同，不能仅凭 DAG ready 推导必然返回。

DFX 的普通 HBG/native Graph 当前采用 v26/v27，使用 PRECLAIM 标志及已观察 cohort 序号。审计要求所有更早本核成员责任已关闭、当前成员已观察，普通执行的 `trace_published` 早于本成员关闭；取得 block 的 AIC 还须早于本核 `claim_before`，AIV 须早于本核 `descriptor_ready`。不能用远端 AIC 的领取时刻禁止该 AIV 的合法窗口。MIX block 的 trace 索引按 active lane 数展开；原 SSBUF、执行身份及组生命周期审计保持完整。常规普通入口仍要求严格的已关闭前缀。

L4 证明资源归属和执行先后；冻结候选窗口的边界由本地掩码及确定性用例验证，不把未记录的普通取票时点当作 raw 证据。复用已有 `preclaim_mask`、`preclaim_initialized` 和 slot 序号，不新增共享原子、GM 表或另一套 grace 状态。功能验证覆盖 READY 恢复、AIV 已 assigned 未 observe、领取或 observe 后拒绝、掩码不重建及异步债务未清。

代价是最多四个已有有限 kernel 可能推迟 MIX 最后成员到达；这有界但不保证模型总耗时改善。L3 用来判断已有普通任务是否被后到 MIX 切分，不能把单个 task 起跑跨度的缩短当作模型收益。保留按同输入完整 12 组 AICore L0 总和决定；双卡两版每轮均先 device 会合，再进入原执行流程，逐轮取 rank 的 AICore 窗口最大值后计算十轮均值。

### 已有未来组与同任务续领

已有未来 MIX 组也遵循这一规则：正式组服务之后，先观察并尝试 ARM 已持有的 `cluster_pending`，再发现新 cohort/MIX。若因已提交 cohort 的 frontier、hold 或 slot 占用而不能 ARM，继续履行该 cohort；ordered 服务有进展后至多再尝试一次已有 pending。没有发生新领取时，不在同轮发现结束后重复检查同一 pending。该优先级不绕过会合许可，也不能因为 pending 的 fanin 未满足就停止控制服务或合法的新任务供给。

普通任务成功返回后，可以保留本核已确认的 fanin 作为续领提示，但它没有独立发票权。旧 block 退休后，必须重新确认本核目录前沿；提示命中这个任务且仍有余块时才可续领，不能因提示省事而越过尚未发完的前项。已有 ready pending 优先执行，predicate 和最终入口检查仍逐次执行；MIX、sync_start 保留各自的组/成员协议。

必须区分两条真实路径：

```text
已接收的 pending：前一 kernel 返回 → 必要完成发布
                 → 剩余依赖/最终许可 → 下一 kernel

缺少普通 pending：有序供给检查 → FAA 取票/固定 owner → 登记诊断
                  → 接收与固定准备 → 等待依赖 → 最终许可 → kernel
```

目标是让第一条覆盖足够多的实际交接。第二条也计入总耗时和间隙分布，不能只挑第一条报告小于 5 µs。取票、责任登记、参数接收和固定准备必须在后继自己的入口之前完成；若要让它们与前驱计算重叠，还必须在前驱完成前得到 Scalar 执行机会。MIX/cohort 另行统计其真实原子交权，不套用普通票的领取成本。

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

下面是按三条原则核对的源码事实及差距，不代表所有目标已经验收，也不是新的性能测量。实现、编译和测量前须冻结实际 HEAD、diff、输入和二进制；rebase 或未提交修正后的状态不得沿用旧版本的通过结论。

| 原则与落点 | 当前实现 | 仍需处理或核实的差距 |
| ---------- | -------- | -------------------- |
| Build／Host：`scheduler_prepare_host_calls` | 按执行单元生成完整不可变 Call，固定地址和 async 定位在启动前发布 | 每单元512B Call 的重复内容、H2D 和 GM 成本仍存在；共享模板只能作为成本候选，不能凭节省空间就改成更重的临执行组装 |
| Build／普通核：`scheduler_execution_token_receive`、`scheduler_prepare_token_binding` | 接收 Binding、固定 metadata、sub_block_id 及写回在 WAIT_FANIN 之前完成 | 已持票的固定准备仍需本核 Scalar 获得运行机会；四槽不能证明每次交接都已有准备完成的后继 |
| Build／MIX：`scheduler_prepare_cluster_lane`、`scheduler_gm_mix_prepare_lane` | SSBUF 提前阶段校验 Call；GM MIX lane 可在等待伙伴到齐时保留已接收 handle | 预校验不等于正式组接收/Binding 发布全部结束；仍有正式 acquire 时的 Call 校验、retained metadata 读取及到齐后的发布，需要按组生命周期识别可复用部分 |
| Build／最终入口：`scheduler_dispatch_execution_token`、`resident_scheduler_impl.h` | 普通派发复用 metadata 和固定 Binding，不再重复写回；动态入口操作仍保留 | token 转 slot、predicate、资源许可、实际输入观察、async context、args dump 与调用入口仍在就绪后的路径上；先区分语义必需项与还能提前的准备 |
| 领取／普通票：`scheduler_refill_execution_tokens`、`scheduler_claim_execution_token<true>` | 一次有效 next_block FAA 固定 owner；执行记录以单 owner 诊断方式发布，不再二次 Claim CAS | 同一宽 SPMD 的 next_block 仍是共享热点；预过滤、越界退款、诊断 DCCI/DSB 和 supply seal 仍有成本；冷取消及退休推导必须一致识别合法 TICKET_OWNED 状态 |
| 领取／目录：`scheduler_bind_issue_entry`、`scheduler_issue_fanin_ready` | 私有游标、固定 entry metadata/fanin 和已确认前缀已缓存 | 头部未发完会挡后项，跨引擎供给与动态 seal 仍需观察；不得为增加提前领取率破坏拓扑供给和同核单 block 约束 |
| Wait／普通依赖：`scheduler_observe_fanin_prefix` | 当前前驱地址已绑定，已完成前缀及 DONE 证明可复用；未完成时继续可靠读取 | DONE 的 GM 同址轮询已迁到 SRAM，GM 回退和其他条件仍有读竞争；`scheduler_fill_own_slots` 的不同推进点最多三次调用 token 扫描，多个 token 也可能等待同一前驱。需看整体请求密度，不能只看一次 Wait helper |
| Wait／协作：`scheduler_observe_ordered_progress`、`scheduler_gm_mix_check_acks` | ACK 正向掩码和本次 assignment 观察已复用；SSBUF 保留专用访问合同 | 组身份校验、admission 状态、未到达伙伴和最终许可仍需推进；不能把普通完成观察直接当成 MIX/sync_start 放行 |
| Wait／控制服务：`scheduler_service_async_slice`、Graph control | 必要控制按预算服务，defer_wait 物理返回与逻辑债务分离 | 不是 percounter 旧分析中的“每轮从头扫完整条链”；仍需量化实际服务占比、目录检查、到期时机与 ready 任务被延迟的程度 |
| 发布／完成：`scheduler_contribute_returns`、async join/FINISH | C/D/N 和唯一 FINISH 责任仍由真正的多写者原子协议决定 | kernel 返回到逻辑 DONE 的必要发布仍在依赖关键路径；不能将所有写侧改成普通 store 或 st_dev |
| 发布／单写者信号：`scheduler_publish_signal` | admission 日志、commit_tail、错误信号已有 st_dev + DSB 路径 | 这是实际代码现状，不是全部写侧可替换的依据；与 atomic_probe 的保守使用规则仍须逐地址、位宽、重复写及交接模式对齐证据 |

因此下一步应从泳道图确认 **准备覆盖率不足、就绪后剩余准备、重复观察、必要完成发布** 各自占多少，再选择改动；不重新引入 ready_queue、复制完成通知或默认扩大 token 窗口。与 percounter 的讨论可用于提出候选，不能把其具体缺陷或优化收益直接套到 HBG。

### Host 静态初始化与 cluster 启动汇聚

Host 初始化 admission 静态事实，Device 将 SSBUF 初始化及首 fill 完成按 cluster 汇聚，取消全图 bootstrap 扫描及全核 scan 会合；最终 `bootstrap_complete → AICPU DMB` 执行门保持不变，任何核首 fill 失败都不能放行业务 kernel。此项不改变 steady-state 发票、owner、MIX、sync_start 或 async 完成协议。

- Host 在现有 `scheduler_init_spmd_task_records` 中先验证整图、容量及布局，再写当前 epoch、member_count 和 INITIALIZED。纯 AIC 成员数为 C，纯 AIV 为 2C；MIX 为 `C * popcount(active_mask | AIC)`，但 sync MIX 的同时驻留 block 位置仍只有 C。动态 order_key、ADMITTED、日志、票据及 arrival/release 继续由 Device 产生。epoch 沿用原 uint64_t 全范围，不充当跨进程唯一 nonce；AICPU 实际拓扑发现及一致性校验必须保留。
- 通用 layout 路径按实际 cluster 数 C 分配 `alignas(128) SchedulerBootstrapCluster`，每项只有一个运行期 atomic-only 状态字。普通-only 和空图也必须分配；保留本轮 Host 重置、H2D/READY 观察、运行结束 fence 后复用的完整合同。summary 的 offset 使用已有尾部空间；首次使用须检查对齐、完整范围和区域归属，不能把高对齐类型按值放到 AIC/AIV 栈上。
- 状态位固定为 INIT=1、AIV lane1 DONE=2、AIV lane2 DONE=4。有 MIX 时 Host 置0，本簇 AIC 清 SSBUF、执行尾 DSB 后，以一次 FAA(+1) 发布 INIT并确认旧值0；AIV以负向 bypass提示加返回型FAA(0)强确认 INIT，再DSB后读取SSBUF。无 MIX 时 Host直接置1，AIC不再写INIT，以免覆盖先到AIV的DONE位。运行期不使用ordinary store、st_dev或DCCI修改gate行。
- 每个AIV仅在本核有界首fill成功后，经过DSB，一次FAA增加自己的DONE位；返回旧值必须含INIT、无自己的位且无未知位。AIC自身首fill成功后，以负向提示加FAA(0)强确认本簇状态恰好7，再DSB并对既有 `bootstrap_arrived_count` 唯一贡献+3。旧全局计数须为3的倍数且不超过worker_count−3；只有到达原worker_count阈值的贡献者发布bootstrap_complete。
- 恰好一次来自已验证的一AIC两AIV拓扑和每worker唯一的启动成功尾，不把加位放入重试循环；FAA后的旧值检查不能替代这个控制流证明。INIT及peer等待均保留worker错误、watchdog和EXIT处理，失败不贡献DONE或+3。首fill继续有界且不执行业务kernel，不得在其中等待前驱kernel完成或伙伴进入业务执行。
- 删除Device静态admission writer、全图bootstrap扫描和旧scan计数/完成字段，同步删除 `scheduler_service_controls` 的旧scan门。Graph控制改由本轮Host/READY初始化的调用前置条件保护，真实激活、fanin、epoch和CAS完成条件不变。不得给废弃字段写常量维持幽灵门；不更改 `awaits_bootstrap()`、AICPU最终放行及EXIT/ACK/退休责任。

本轮新空间为128C字节，32簇为4096B；集中到齐FAA由2W次变为C次，满卡从192次变为32次。各簇INIT、DONE和强确认仍有局部原子成本，不能把集中次数差当作总指令减少数或墙钟收益。现有SSBUF访问屏障、Host/READY、Call 的启动观察及 Binding 接收屏障继续保留；新状态发布前后的DSB不代替payload DCCI。

开发依次同步Host/layout、Device cluster协议、Graph控制门、诊断及现有fixture。最少功能验证包括无MIX时AIV先完成而AIC后到、旧SSBUF/相同epoch残留不能提前放行、任一lane失败不产生global complete、恰好两peer完成才+3、空图及Graph/async跨run复用；优先扩展既有用例。审查实际AIC/AIV IR后再上板。L2只定位bootstrap区间变化；保留由同输入12组AICore L0总和裁决，双卡两版每轮先device会合再计时。首个kernel之前的bootstrap缩短不计入主指标；此项对首波准备和随后调度交错的影响必须在新口径下实测，不能沿用旧RunWall收益。

### Cooperative discovery 的已确认依赖前缀

协作候选等待尚未完成的前驱时，本核保留此前已证明 DONE 的连续 fanin 前缀，避免每次 discovery 都从第零项重新做共享读取和原子确认。证据来自实际 kernel 之间的协作准入路径；启动阶段的缩短不计入这项优化。

- 私有状态仅包含当前 expanded `task_ref` 和 `next_fanin`。候选身份变化时清零；每次 run 重建 `SchedulerLocalState`，不得跨 run 或不同 Graph instance 复用证明。
- 继续使用 `scheduler_observe_fanin_prefix`。只有既有权威完成协议已经确认 DONE、计算贡献及异步债务均已结清的前驱才能推进；Host 已完成节点沿用既有静态证书。pending 提示、kernel 返回、供给关闭和 claim seal 都不能代替 DONE。
- 保留 issue head、错误检查、fanin view 验证、admission stage/commit、MIX 与 sync_start 成员责任和最终 kernel 许可。缓存只省去本核重复确认，不发新票、不占新 slot、不转移 owner、不推进目录，也不代替真正消费数据时的 DCCI/DSB。
- 不复用 `issue_next_fanin`：它证明供给关闭；也不提前填写 admission 的 pending/current 身份。当前义务与正在观察的下一候选仍是独立状态。
- 功能验证覆盖“前缀已 DONE、末项仍有异步债务、重复观察、末项完成后准入，以及切换候选后重置”。性能由统一 AICore L0 的外部对照验证；少一次原子确认不直接等同全图收益。

### 新就绪协作候选的及时发布

`scheduler_fill_cooperative_slots` 在本次 discovery 新产生 ready pending 后，立即执行原末尾的那一次 `scheduler_service_cooperative_slots`，随后返回调用者；不先发现下一项普通 MIX 或执行其预准备。已有 pending 的 discovery 早退会清除本次 progress，因此 `progress && pending非空` 只识别新 stage，不让持续抢锁失败的旧 pending 一直绕过其他服务。

- 前移整个既有 service，而不是直接调用 commit。service 内已有 SSBUF 责任、pending ARM、必要的组服务及 ARRIVED/RUNNING 早退保持优先，之后才按原协议发布 admission。
- 每次 pass 仍只有原来的那一次 service；不新增观察锁、不抢未 ready 任务、不重复 CAS，也不保留任何跨返回的发布锁。原 `failed`、`direct_slot` 和进展后的 issue snapshot 失效处理必须保留。
- 无可执行 slot 或发布重试时也返回正常调度循环；下一轮仍可推进既有 token、其他候选及普通 MIX。仅后移准备未来工作，不取消准备或更改固定 owner。
- 代价是下一项普通 MIX 的预准备可能延后，不能用一个 admission 局部窗口变短证明全图收益。外部性能对照按完整 12 组 AICore L0 总和判定；功能验证需要确认延后的准备能恢复且已有 SSBUF 责任不被跨过。

### GM MIX 在准备之后观察伙伴到齐

`e97cbc0a4` 的 Binding 提前写回、组内 metadata 缓存及同次 context 检查消减已按用户要求撤回。当前保留下面的原有 Call 准备与 ACK 观察机制；不因该候选总和回退低于1%而自动恢复。测量和撤回依据见[调查记录](../../../investigations/2026-11-a5-hbg-mix-binding-preparation.md)。

参与计算的 GM MIX lane 在 ARRIVED 阶段先完成本核 Call acquire/validate，或验证已保留的 FILLING 句柄，再执行原有的唯一一次 ACK 检查。这样可观察到准备期间伙伴的到达，不把准备前的 `all=false` 带回其他任务的发现路径。不增加重查循环、等待预算或每轮原子次数。

- 不执行计算的协调 AIC（`subtask_slot == UINT8_MAX`）继续先检查伙伴并按原协议释放；不为它创建无用的 payload 或执行句柄。
- 未到齐时保留原 FILLING/CLAIMED 身份并返回；后续轮次不重复 acquire。到齐后仍检查当前 run、descriptor 身份及 block，再发布 READY；sync_start 的共同许可和最终 kernel 入口检查不变。
- 外层观察复用已校验 ACK 和同次推进的 assignment 快照；精确 sequence、身份及失败位检查均保留。本 lane 尚未 release，同一组的合法 ACK 不会回退或跳到后代；future sequence 仍是错误。
- 非法 ACK 可能在取得未执行句柄后才被发现，由原取消/错误退出路径处理，仍不得发布 READY、执行 kernel 或伪造完成贡献。若 payload 和 ACK 同时损坏，首个诊断错误的优先顺序可能改变。
- 功能验证使用显式事件让伙伴在真实 payload 准备期间到达，验证本次服务可直接发布，及仍未到齐时继续保留句柄；不得用 sleep 或模拟时长推测到达。DFX 的到齐证据仍早于 slot publication，允许早已存在的“先准备、后到齐”顺序。

### 接续派发与 MIX 完成贡献

普通 SSBUF MIX 的计算 lane 先完成首次 Call/Binding 接收与准备，再执行本轮唯一一次 arrived 检查；保留 FILLING 的后续服务不重复 acquire。不执行计算的协调 lane 继续直接检查到齐。正式 generation、成员 mask、资源承诺与入口许可不变；不能把准备前的负向快照用于准备后的发布决定。

普通补给复用同一个单 token 派发函数：新票完成接收和 fanin 观察后，可直接尝试派发该 token，并把选中的 slot 返回外层。已有 token 扫描也调用此函数，避免新增一次全窗口扫描。无法派发时仍服务已承诺协作与到期控制；最后入口重新检查共享许可，不缓存正向资源许可。

MIX 完成由唯一协调者按组汇聚。每个计算 lane 完成真实返回、writer seal 和发布屏障后，只提交成功信用与资源释放；全部资源 lane 退役且全部计算 lane 成功时，协调者在 GM 路径一次减去 `K * SCHEDULER_COMPLETION_RETURN_UNIT`；计数器路径按 K 发送 INC 写 1 贡献 P。`K` 是 active lane 数，不包含 vector-only MIX 的非计算协调 AIC。

- 成功和失败复用已有 ACK：SSBUF 的成功值为本组奇数 generation，失败值为 generation+1；GM 的低 63 位保留 sequence，最高位标记失败。到达 ACK 不变，序号校验和 DFX 导出始终使用解码后的 sequence。失败仍可收敛资源，但不能计为正常 DONE。
- 每个 group 使用本核 `completion_submitted` 标记唯一提交，占用现有结构填充空间。新 group 初始化此位；重复服务和错误退出不能重复减计数。共享协议空间不增长。
- GM 路径的计算份额 C 与异步债务 D 保持独立：聚合只归还各 lane 的计算信用，条件信用仍由唯一 FINISH 结清。计数器路径则由条件封口或最终 CHECK 发布 Q，并以累计阈值确认完成。仅在 GM 路径，最后计算组返回与最后条件组满足无论谁先发生，都由 join 唯一选出一个 FINISH owner 并完成 canonical 结账。
- L4 分开记录各 lane 的物理返回和 owner 的组贡献。GM 组票覆盖连续 K 份贡献，最后组产生的 DONE 带 `COMPLETION_FROM_GROUP` 标记并复用该组真实区间；不为每个 lane 伪造独立原子贡献；计数器路径记录实际 P/Q 区间，不生成返回式 FAA ticket。
- 协调者在发布下一组前完成本组贡献。资源释放 ACK 不等于成功信用，Host 退休仍等待计算和异步责任全部结束。

该协议复用组内已有的退役检查，但最后贡献需要协调者再次得到执行机会；减少原子次数不能直接证明全图收益，仍按 12 组 AICore L0 总和验收。

普通 MIX 已由协调者确认 fanin，参与 lane 从正式 ARM 继承该证明。普通 SPMD 的首次完成观察使用绑定的完成来源：完整计数器映射优先比较累计阈值；无计数器时读独占 SET 通知；两者均无映射时才读取 canonical GM completion word。可靠 `ld_dev` 分别证明合法阈值已达、SET=1 或合法 `C=D=0`，随后进入本核 DONE 缓存，不追加 `FAA(0)`。DONE 在本轮任务身份内不可撤销；跨 run 初始化、各后端贡献协议及发布与消费屏障保持。供给 seal 不能代替 DONE。

GM MIX 的 assigned_seq 与未确认 arrived/released ACK 直接消费可靠 ld_dev 的完整64位值，不追加 FAA(0)。descriptor 的 run/task/order/cluster/group/lane-predecessor 校验及读取前失效/屏障保留；ACK 只接受当前精确组序列，旧值等待、未来值报错，released 原值中的失败位仍参与成功判定。外层 observation 对 coordinator ACK 执行精确序列校验并更新已有 arrivals_confirmed/releases_confirmed；部分伙伴尚未到达时也保留已确认前缀，正式服务只消费缓存或补读缺失成员。新组沿用原清零规则；释放失败位只锁存，全部资源释放后仍由原 retire 拒绝完成信用。外层显式区分 WAIT/READY/ERROR，正式 group_acks DFX 与最终许可检查留在内层。assigned_seq 使用本次 fill 的局部快照，绑定 epoch/task/order/group/phase，只有首个 service 可消费且使用后清除，身份变化时重新读取。所有发布、资源退休和完成屏障保持原合同。

### 已接入的共享观察机制

以下机制已经构成当前基线，不重复列为待实施优化。后续候选依次与保留版本比较，独立核对 12 组 AICore L0 总和；任务归属、kernel、前端语义及跨卡启动同步保持原合同。

1. **私有目录游标**：删除共享 `next_task`、其 FAA 确认和推进 CAS，以及 Host 的游标分配/初始化。各核从目录零位开始，只有确认本项 supply seal 或合法终态后才在本核越过该项。最后有效领取者发布 seal 后可直接推进自己的游标。跨引擎前驱供给检查、本 task 单未完成 block 和协作/异步服务独立保留。落后核每轮有界追赶，不让目录扫描阻塞已有可执行工作。
2. **GM MIX 单次确认与组内许可**：既有 coordinator 通过可靠读取确认到齐集合，经 SSBUF 发布本组 sequence 许可，伙伴消费许可后继续原 slot 准备及执行检查。arrival/release 的正向 ACK 按本组私有 mask 复用，本核发布构成本核证明；新组清空 mask，失败 release 保持 sticky。coordinator 必须先发布许可再进入 kernel，inactive AIC 同样履行协调责任。组许可占既有簇 SSBUF 区8B，不是 Task DONE 副本；退出 drain 也使用同一许可协议，缺失 SSBUF 视为协议错误。全局 sync_start 成员登记、真正的数据可见性屏障和 canonical completion 保持。 coordinator 完成本核关闭检查及发布屏障后，推进既有 SSBUF admission frontier；伙伴消费该关闭前缀。DFX 区分 GM 读取和 SSBUF 许可消费，不为缓存命中制造 FAA 记录。历史实验与当前验证见 [调查文档](../../../investigations/2026-11-a5-hbg-gm-mix-arrival-permit.md)。
3. **任务状态观察**：四份完成通知的实现已撤回，当前不保留该四份 GM 副本、Host 扇出阈值或分组读取路径；完成来源使用计数器阈值、单份 SET 通知或 GM gate。先减少同一地址上的 atomic 请求，包括并发争抢和对同一事实的重复确认，再减少 atomic 总调用次数；通过明确 task、执行单元、协作组及异步检查责任来处理，不以复制完成状态规避争用。已有实验保留为[调查记录](../../../investigations/2026-11-a5-hbg-replicated-completion-notices.md)，不作为当前实现。

第一项的正确性依赖于闭合前缀的归纳：核领取目录第 i 项前，已经分别确认所有更早项关闭；各核游标无需相等。目录用于有序发票，kernel 入口仍独立等待 DONE 与资源许可。依赖确认的复用只能继承已经成立的事实，不能通过供给关闭或提前领取制造就绪。

GM 路径实际发布 canonical DONE 的核，可将该次返回式原子已经确认的结论写入现有16项 task 私有证明缓存；计数器路径使用完整阈值的真实观察或本核已持有的 DONE 证明。后续供给/依赖检查直接复用。普通返回与 MIX 组贡献只在 `TASK_LAST` 时记 DONE；控制节点完成 CAS、异步 FINISH CAS 只在成功后记 DONE。`PARTIAL`、`TASK_LAST_DEFERRED`、失败的 CAS、单独的 slot/资源退休都不授予该证明。缓存仍按本轮完整 task 身份区分，碰撞只遗忘结论，不增加容量或共享通知；既有 pending_completed、组资源关闭、错误检查与 DFX 记录独立保留。

### 控制节点的本核前驱检查进度

普通 HBG 镜像在 Host 完成原图排序后，可折叠无计算的 inline 汇合节点：只有零或一个活跃消费者、且合并后的消费者 fanin 不超过现有128项上限时才折叠。消费者继承被折叠节点的全部前驱并去重；物理 kernel、MIX、sync_start、真实 defer_wait 任务都保留。多个消费者共享的汇合点保留，以免复制依赖检查；native Graph 的 ENTER/EXIT/CONTROL_DUMMY 继续遵循其既有实例与控制合同。

折叠先在 Host 暂存图上计算，再将全部存活 fanin 行重排到原 fanin 池；只处理零/单消费者保证边数不增加，不增长 Device 存储，不覆盖 tensor/scalar 区域。若原图共享 fanin 存储而独立重排无法容纳，则保留该镜像。只修改未发布的 resident 副本，Host Tensor owner/last_consumer 与原分配计划保留。原物理目录、rank 和拓扑层号保留，新增直连边必须仍满足前驱先于消费者。被折叠节点按 Host 完成节点记账，更新 control_count 与 inline_completed_count；消费者已接回真实前驱，因此真实 async 债务仍阻止后继执行。末端无消费者的汇合点可折叠，但整次运行仍等待全部真实任务完成。

未折叠的 inline/native control 沿原有目录与每轮8项预算轮转。每核使用16项控制节点专用私有缓存，保存 task_ref 对应的已确认 DONE 的 fanin 前缀，避免多个未完成 control 轮转时丢掉彼此进度。槽位内完整身份必须匹配；hash 碰撞只遗忘旧证明，不能将旧前缀授予新节点。零进度不驱逐其他节点的有效前缀，扫描普通节点或跳过已完成 control 不清空缓存。新 run 清零全部条目；缓存遗忘只造成重新确认，不改变数据依赖、Graph 激活、最终完成 CAS 或错误检查。

合法图的 task_count 不超过 INT32_MAX，fanin count 不超过32位字段范围，因此每项用64位编码 task_ref+1 与 next_fanin，零值表示空条目。缓存共128B/核，仅在本核使用，不新增共享通知或原子操作。

等待中的 control 不绑定服务核，也不阻止扫描其他 control；不新增共享选主、通知副本或 ready 队列。前缀只保存当前完成来源的有效 DONE 证明；可靠 `ld_dev` 读到合法计数阈值、SET=1 或 GM `C=D=0` 可建立该证明，PENDING、供给关闭和参数就绪均不能代替它。操作减少与12组 L0 门禁分别验证。

### 收敛同址原子访问的实施约束

优先减少同一地址上的 atomic 请求，再减少 atomic 总调用次数，并减少重复 ld_dev 观察。保留唯一 canonical 完成字、固定执行 owner、有序供给和已有协作语义；不复制完成通知，也不通过扩散地址替代职责设计。每个候选独立比较12组 AICore L0 均值之和，并展示逐项变化。已证明消除了冗余 atomic／ld_dev 且功能合同保持时，允许总和回退不超过0.5%仍保留；回退超过0.5%则撤回。须明确区分操作数量减少和已测得的性能收益，不能隐藏新增的普通访存、DCCI/DSB 或其他地址的竞争。

1. **观察与处理职责**：异步 CHECK 的目录用于发现候选，lease 每次只取一份可靠 `ld_dev` 快照，带 generation 的 CAS 才授予处理责任。非 RUNNABLE 留待后续扫描，非法状态直接报错，不保留交替强复核标志或追加 FAA(0)。旧 generation 不能越过 CAS；CAS 返回更新后的 RUNNABLE 时最多再尝试一次。目录游标仍越过已访问条目，保持扫描预算与公平性；私有状态仍64B，FINISH 不使用目录或 lease。CHECK 每次服务有界，返回业务执行前释放责任，不持 lease 进入可能同步等待的 kernel。 非空组登记 canonical 完成债务时，可靠读取作为 CAS 期望值；不满足有效性、C>0 或债务上界时直接拒绝，不追加 FAA(0)。CAS 冲突后复用返回值，只有 CAS 成功才取得信用，不能用无检查的 FAA(+1)替代溢出校验。
2. **稳定事实复用**：在同一 run 和同一任务义务中复用已确认的供给关闭、完成及依赖前缀。每个证明必须有明确的来源和失效点；错误、epoch、实际执行许可继续在必要边界检查。先验证 admission 的合法 supply seal 能否直接证明最终有效登记数，再复用已有本核 DONE 缓存。
3. **执行责任退休**：普通 execution token 的真实执行者复用最后一次贡献 FAA 的返回值，在 C 归零时省略该执行单元的 RETURNED 记录发布；PARTIAL 和失败返回仍由固定 owner 发布诊断状态，MIX/cohort 保持原原子退休合同。C=0 仅证明计算执行责任结清，D>0 时 task 仍未完成。记录继续提供挂起定位；Host 同时导出 raw phase、有效 phase 和推导依据，AICPU 错误快照也沿原 cache invalidate 观察记录，不据此授予执行或修改状态。checked return/fail/cancel 拒绝已推导退休的旧句柄。PreparedLayout 的任务控制区 offset、结构大小和身份验证不变；诊断写回与原有完成/退出屏障分别保留。

## 2. 从 task 有序开始：供给、预绑定、等待与退休

### 先区分构造、发票与完成

| 状态 | 含义 | 不代表什么 |
| ---- | ---- | ---------- |
| BUILT | Host 已发布该实例的完整不可变 Call | 没有指定执行核，也不代表依赖满足 |
| ISSUED_ALL | task 的全部有效 block 已取得不可撤销的执行责任 | 不要求这些 block 已开始或完成计算 |
| DONE | 计算贡献、合法 skip 和异步条件按原协议全部结清 | 只能由权威完成协议确认，不能由发票游标推断 |

`ISSUED_ALL` 是供给关闭证明，与逻辑 DONE 分离。普通任务及普通 SSBUF MIX 的最后有效领取者利用 `next_block` 返回值，在已有独立 atomic line 的 `claim_seal` 发布完整 block 数；sync_start 继续由真实成员登记完成路径发布 seal。观察者确认正向 seal 后复用关闭事实，不反复读写正在领取的 `next_block`。MIX 的票仍代表逻辑 block 的整组责任，原始票号不等于真实成员到齐。执行者取得有效票后必须有限步保存身份、完成 Claim 或报告终态，不能取得票后又等待其他任务来决定是否保留。

普通 token 的最后有效票在唯一 Claim 成功、handle 已记录固定 owner 后发布 seal，再接收与校验不可变 Call/Binding。此时本核的 token 仍是 TICKETED，Claim handle 已登记，接收成功才进入 WAIT_FANIN；其他核可以领取并准备后项，执行仍等待前驱 canonical DONE。接收失败须记录错误并撤销未启动的本地执行责任，不能完成前驱或放行依赖消费者。Claim/Receive 分开计时；需要完整接收的 MIX/cohort 入口仍组合调用两步。普通 MIX 与 sync_start 保留各自真实责任登记的发布点。

普通纯核 token 与普通 SSBUF MIX 的 coordinator 共用本核当前发行项的无票证明。两者在首次可靠 `ld_dev` 观察 next_block 已达到 block 数，或首次越界取票完成退款后记录。之后同项不再重复分配或强确认无票，已有 group/pending 仍按原 owner 服务。该事实不代表正式供给关闭、成员登记或 DONE；目录仍等待原有 claim seal/完成证明。continuation、refill 与普通 MIX discovery 共用此事实，发行项切换及新 run 时清空，不改变 sync_start GM MIX 的成员取票协议。此项以已证明减少的重复原子访问作为保留依据，当前 L0 总和回退不超过0.5%仍可保留，超过0.5%则撤回；不把操作数量下降等同于实测耗时收益，后续继续分析访问时序与竞争。

部分驻留 `sync_start` 的每个候选位置直接领取一次 `claim_tickets`，由返回票号确定是否拥有 block；`sync_claimed/ticket_exhausted` 防止该位置重复取票。领取前不另外读取有效领取计数和 claim seal。实际有效领取仍单独递增 `next_block`，最后领取者发布 seal；没有获得 block 的位置仍通过原关闭检查确认有效供给完成，不能用原始票号代替该证据。满驻留路径保持一次有效领取原子操作，不使用额外票号。

本方案保留 `ISSUED_ALL` 作为后项领取门槛：Host 先形成满足依赖约束的顺序，Device 按这个顺序把前项全部有效执行票分配出去，再开放后项。这里的“全局”是该 task 的所有 block 都已有人负责，不是只看本核已领到一块，也不是要求全机只有一个发票游标。对于普通依赖 `A → B`，A 全体发票后即可提前领取 B、接收参数；B 的 kernel 仍等待 A 的逻辑 DONE。因此有序发票与重叠准备、并行计算相容，不能把发票门槛误称为执行完成屏障。

### Host 拓扑分层与层内优先级

**保证前驱排在后继前面，必须在 Host 排序时完成，并在目录发布前逐边验证。** 选择 Kahn 式拓扑分层，优先展开同层的独立分支；层内再按剩余依赖链长度排序。参考 [Kahn 原论文](https://doi.org/10.1145/368996.369025) 与 [拓扑分层定义](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_generations.html)。

排序输入是已实例化的数据与 Graph 控制依赖 DAG。`level` 表示从源节点出发的最早拓扑层，`b_level` 表示到终点的最长剩余依赖链；初版均按边数计算，不假设已知 kernel 时长：

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

普通图在 Host 私有镜像发布前，先固定完整原图的 Kahn 排序和两个任务目录，再精简执行检查的 fanin，并按既有 rank 稳定升序排列。精简仅适用于 A5 resident 普通图：live task 在执行或 predicate 跳过之前均须确认全部 fanin DONE，故另一个直接前驱的逻辑 DONE 可以证明其传递前驱 DONE。Host 用每 task 一个64位字记录前64个 task ref 内的祖先，只删除已由其他直接前驱证明的边；超出窗口的边、未被覆盖的重复边、原 fanin 存储 offset 和任务目录保持。完整输入先校验，非法边不能被精简掩盖。若 live fanin 行共享或重叠存储，则整份镜像保持原 fanin，不执行原地精简或排序，避免改写另一任务的依赖。

HostDone 节点不传播历史祖先，因为它的输入可能已失效且未经 Device 确认。defer_wait 的 DONE 仍要求 C=0、D=0，不能用计算返回、PUBLISHED 或通知提示替代。普通任务的供给关闭也沿原有前驱供给约束传递，删边不提前越过尚未供给的祖先。native Graph 的 ENTER/EXIT 与共享 Definition 暂不精简；AICPU early-dispatch 不满足相同 DONE 推导合同，也不采用此转换。运行时 DFX 的协议图导出精简后实际上传的 fanin。

1. 复用现有数据依赖合法性检查。现有 `scheduler_fanin_ref` 要求数据 producer ref 小于 consumer ref；非法引用、自依赖等不能靠排序掩盖。Host 已完成节点沿用合法豁免，不重读其失效的历史输入。
2. 对现有数据依赖及 Graph 控制边检查无环性，并验证排序后每条边的 rank；不改写原数据 fanin 的逻辑完成语义。
3. AIC/MIX、AIV 两个目录无遗漏、无重复，保持全局序的相对位置。普通和 sync_start 的同类节点共用目录；Vector-only MIX 仍由 AIC coordinator 领组票，不再从 AIV 目录重复领取。
4. Graph ENTER/EXIT、control/inline 的真实约束必须参与规划；这些节点不因排序变成普通业务执行票。保留实例身份、元数据发布和动态激活条件；sync_start 排序不替代资源准入、真实成员领取与 seal。
5. Host 预计算静态约束和目录位置，Device 不重复计算层号或验证整张拓扑。每类关闭前缀证明其前方供给已结束，未关闭头部约束该类新票；跨引擎前驱保留实际供给检查。目录 rank、已发票与 DONE 分开，不用本类头部代替另一类状态。
6. 静态排序、实际领取、真实完成分别验证；目录位置不表示通知已经发起或到达。

Host 必须先完成现有依赖图与目录校验，再发布和 H2D。非法图或成环时报出具体节点关系，不发布部分目录让 Device 补救。不增加逐 block 大表，block 仍由 task-local `next_block` 分配。

### 与 percounter、列表调度的关系

| 方法 | 主要依据 | 本方案用途 |
| ---- | -------- | ---------- |
| Kahn 拓扑分层 | 优先安排最早拓扑层的独立分支 | Host 主排序规则 |
| percounter 的 b-level | 最长剩余依赖链优先、原 ID 稳定打破同分 | 借鉴为层内次序；保留正向层号的进展约束 |
| HEFT | 预计计算/通信成本与最早完成时间，并选择处理器 | 后续资源模型的参考；不直接采用其静态物理核映射 |

percounter 的实现位于 `src/a5/runtime/percounter/host/region_plan.h::build_tables` 与 `host/graph_template.h::build_graph_template`。它对普通节点按 b-level 降序，同分按 ID；长支路可能先深入几步再处理独立短支路。本方案把正向层号置于 b-level 之前，两者不能混称为同一个策略。

[HEFT 原论文](https://disco.ethz.ch/courses/fs14/seminar/paper/Jochen/4.pdf) 可作为异构资源调度参考。本方案初版只用静态图信息，不要求 kernel 耗时模型；实际核继续由 Device 动态领取确定。无完整通知关系时，纯 b-level 可能把通知生产者放到等待后继之后，见[固定 owner 下的排序反例](../../../investigations/2026-11-a5-hbg-blevel-fixed-owners.md)。其他排序候选也须先通过该进展约束，再保持 kernel、输入及资源协议一致，以 12 组 L0 总和判断，不按模型选择策略，也不在生产中保留历史兜底开关。

### Device 沿拓扑目录发票，前项全体发完再推进

`scheduler_query_issue_head` 使用本核目录游标及已关闭前缀。**依赖图给出偏序，Host 把它排成满足约束的确定顺序；Device 不临时重新选序。** 同类目录中，前项所有有效执行票发完或按协议合法终结，才能领取后项。跨引擎没有依赖的目录项可以各自推进；有依赖时还须确认另一类前驱已全体发票或合法终结。实际调用另外等待数据 DONE、Graph 激活及资源许可。

| 当前情况 | 处理 |
| -------- | ---- |
| 本类头部有余块，本核有相应位置且未持有该 task | 领取头部的一块，立即保存固定 owner |
| 本核已持有头部 SPMD 的未退休 block，头部尚未全体发票 | 不领同 task 第二块，也不越过头部；推进已有 token、协作及独立控制服务 |
| 本核 block 已安全退休，头部仍有余块 | 重新取得该 task 的领取资格，继续补前序余块 |
| 头部全局发完票，部分 block 仍在执行或等待 | 推进头部，可以提前领取后项；不等待前项整体 DONE |
| 普通窗口或相应组/成员位置已满 | 不增加隐藏责任、不迁移 owner；服务已有工作 |
| 跨引擎前驱未发完、元数据未发布或必要控制约束未满足 | 暂不领取该候选，继续相应进展服务；本类头部不能替代这些证明 |
| 候选是 MIX 或 sync_start | 遵守同一目录，再执行各自组/成员协议；完整组责任或真实 claim seal 才能关闭供给 |

```text
有界补给：
    查询本核目录前沿；已发完或合法终结的项按预算推进
    检查跨引擎前驱供给，以及 Graph 的激活与控制门槛
    本核有相应位置、未持有该 task 且协议允许：领取一个有效单元
    本核暂不能领：返回服务已有责任，不用私有 cursor 跳到后项
    有 ready 的已持有任务：不为继续发新票而延迟它
```

“前视”仍指在前驱计算完成之前持有后继 execution token，不表示越过尚未发完票的目录项。`SchedulerIssueSnapshot` 可缓存单次 fill 已确认的头部，头部推进或已有协议要求失效时须重新查询；不引入逐核扫描后项的选择器。供给 seal 与无票状态通过可靠 `ld_dev` 观察，实际取票和交权仍使用原子更新；不得用 hint 授予 owner 或释放资源。任务完成观察独立使用可靠 `ld_dev` 的合法终态快照，不重复执行 `FAA(0)`。继续参考 `atomic_probe/ATOMIC_USAGE_GUIDE.md` 的同址争用与持续轮询进展证据。

这是一项比“只按真实依赖边先后发票”更强的供给约束：同一目录里的独立任务也被 Host 排成先后。本轮保留它以简化责任与进展证明，把独立分支尽量靠前的工作放在 Host 排序中。前例的 AIC/MIX 目录改成 `t8、t11、t10` 后，t8 发完票就能开始 t11 的供给/准入，不再等 t10 发完；不要求 t8 整体执行完成。若 t11 本身是 sync_start，真实成员领取仍须满足它自己的依赖和资源许可。

### 接受的性能代价与本轮不改善的原因

**保留前项全体发票后再领取后项，接受超出物理核数的宽 SPMD 可能推迟独立后项的代价；本轮不为此引入跳项机制。** 这里用 `N` 表示 task 的逻辑 block 数，用 `K` 表示实际配置的对应引擎资源池的物理核数；AIC、AIV 分别计算，不能将两类核相加作为普通 task 的容量。

当 `N > K`，每核同 task 最多持有一个未退休 block，最多 K 张票可同时被持有；必须有部分 block 执行并退休后，才能继续分配余票。例如 4 个 AIC 已各持有 A 的一块、A 共 8 块且等待前驱 P，目录中后面的独立任务 B 即使已经 ready，也不能领取。A 的余票未发完期间可能出现空闲计算资源，损失原本可与 P 重叠的 B 执行机会。这是说明机制的例子，不是实测性能结论；若核正在满负荷执行 A，也不能将 B 等候直接等同于计算资源浪费。

**当前目标网络的多数业务通常按 SPMD block 数不大于对应物理核数组织，即 `N <= K`。** 这是本轮设计采用的业务常态。在各核有领取机会且协议允许时，全部 block 可以分别交给不同核，无须等待同 task 的 block 退休来腾出下一张票的资格；因此上例由 `N > K` 必然引入的多轮发票，不是当前主要优化对象。

本轮优先保留明确的领取顺序、固定 owner 和简单的进展证明，通过 Host 拓扑分层改善独立分支次序，并优化常见路径的交接开销。不为较少出现的超宽 SPMD 增加 Device 动态跳项、额外候选搜索或已领取任务迁移。超宽普通 SPMD 仍须正确执行，这项业务常态不变成 `N > K` 的输入拒绝条件；sync_start 超过同时驻留容量的拒绝规则另行保留。

`N <= K` 也不保证随时发完票：部分核暂忙、窗口已满、MIX 成员约束或 sync_start admission 仍可能推迟领取。这些实际等待继续在 L3 中区分，不能据此宣称已经实现最大并发。当前验收按 12 组 L0 总和（含通信）；只有后续代表性业务证明超宽 SPMD 的供给等待成为主要瓶颈时，再重新评估是否值得改变这项取舍。

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

这条规则排除了“后继窗口挤掉尚未分配的计算前驱”的示例，但不独自证明全部活性。仍须保证已发票责任得到服务、MIX/cohort 资源协议不成环、Graph 控制可推进，异步完成条件仍按原协议处理。固定 owner、四份普通责任上限、合法 skip 和错误退休合同保持不变。

### 不可变 Call 与可写运行态分开

一个纯核执行单元的身份为 `(run_epoch, node_ref, block_idx, subtask_slot, execution_index)`。`node_ref` 区分 native Graph 实例，不能用原始 source TaskId 或 Graph ID 替代。MIX 的分配单位是整个逻辑 block，其 active lanes 各有独立 Call。

| 信息或操作 | 最早合法时机 | 由谁准备、如何保存 | 留到实际入口的部分 |
| ---------- | ------------ | ------------------ | ------------------ |
| 函数入口、Tensor descriptor 地址、scalar、固定 shape、Call/Binding 地址、block 身份 | Host 已知本 run 的最终基址、图和参数时 | `scheduler_prepare_host_calls` 发布不可变 Call；数据未生成不妨碍地址绑定 | 业务 Tensor 内容的有效性与可见性，不由 Call 提前证明 |
| async image/bytes/run_epoch、group offset/index | 实例及执行单元已确定时 | Host 预绑定固定定位 | 当次 writer/entries/count、实际登记和 seal，不提前创建信用 |
| 目录 metadata、fanin view、当前等待对象地址 | 对应已发布节点首次合法解析时 | 本核 issue binding、token、pending group 或 cooperative wait 中保存 | 动态 supply seal、完成字、activation 等仍按合同观察 |
| 唯一 execute_owner、普通 sub_block_id、固定 Binding 的接收和发布 | 有效票固定 owner 后 | 普通 `scheduler_prepare_token_binding` 在 WAIT_FANIN 前完成；保留必要 DCCI/DSB | 当前 owner 是否空闲以及瞬时资源许可 |
| MIX lane 的准备 | 各自的 group/generation 与准备授权成立时 | 静态预校验和正式 acquire/prepare 分开；只在已证明的组生命周期内复用 | 动态到齐、ARM/GO、lane 资源和正式执行许可 |
| predicate、Graph 激活、async 保护的业务输入 | 对应数据与控制前驱已满足后 | 实际消费者在合法边界读取和观察 | 当前准备后仍可能发生的状态变化、EXIT/错误 |
| DFX 时间戳、args dump、运行期可写 context | 对应真实事件发生时 | 真实执行者生成；诊断不授予业务许可 | 不能用预填的时间戳或旧 context 代替本次执行事实 |

沿用 `scheduler_prepare_host_calls()` 和既有 metadata/H2D/launch 发布链。函数入口使用注册时的 resolved entry，不能把 CoreCallable 对象地址当函数地址。固定字段只检查一次；执行者仍需通过正确的接收协议读取它们。

入口解析限定在 A5 HBG 的 Host Call 准备路径，复用原四参数 `set_callable_tables()` 提供的对象地址表。
A5 真机按注册器的既有布局取 `CoreCallable::binary_data_offset()` 对应的代码地址，不解引用设备对象；
A5 模拟器读取 Host 对象中的 `resolved_addr()`。A2/A3、TMR 和公共 Runtime 的绑定接口保持原样。

普通路径删除 `next_prepare_block`、Device `PREPARING/PREPARED` 和相应重复构造职责。Host 发布后的 Call 已是完整执行数据；领取者复用本轮不可变 Call，完成 Binding 定向观察和 token 私有缓存即可。真正需要实际执行者填写的可写 context 在最终入口有限步填写，不能把整个 Binding 宣称为只读。

静态字段前移和取消 cache 维护必须分开论证。既有 [Binding 发布调查](../../investigations/2026-11-a5-prepared-binding-publication.md) 没有建立“Host 初始化后可省掉 Binding 接收”的依据；本方案保留所需 Binding 定向观察，并为其可写字段规定唯一 writer。不能仅因移除重复 Device store，就顺带删除保护其他字段可见性的 DCCI/DSB。

### 普通初次执行票

通过有序供给检查后，task-local `next_block` 每次 FAA 取 **1 个**有效 block，返回的有效票直接固定 owner。`scheduler_claim_execution_token<true>` 名称中的 Claim 表示身份与诊断登记，当前普通路径不再执行第二次 Claim CAS。不同 task 的状态分散，但同一宽 SPMD 的 `next_block` 和供给 seal 仍是共享观察点，不能宣称原子竞争已经消失。供给 seal、fanin 观察、超额票/退款和 owner 诊断发布都另计，不能用“一 FAA”表示全部调度成本。MIX/cohort 的原子交权另有资源合同。

- 新票选择检查本核是否已有同 `(run_epoch, node_ref)` 的 current/inflight/pending；存在则该 task 暂不可本核再领，头部尚未全体发票时不能自行检查后项。退休后若头部仍有余块，继续领取前序任务；头部全体发票后才开放后项。
- 有效票立即保存到 token，后续身份登记或接收不能重新取票；只有越过 block 上界的超额票可以按既有合同退款。
- Host 已发布、初次唯一票上的诊断初态或执行身份失配是错误或已确认的终态取消，不是普通竞争落败。
- `next_block >= block_num` 只证明新票已经发完，不证明 kernel 完成。已领取单元仍由各自 owner 等待、执行与退休，发票耗尽不得提前结束运行。
- 静态初态由通用计划生成并校验 run、单元范围及区段边界。普通 owner 对独占诊断记录的旧值检查只用于拒绝生命周期失配，不承担另一场跨核领取竞争；不能凭普通读取授予共享 cell 的所有权。

### 四 token 是有界前视，不是四份物理资源预约

每 worker 初始设置 4 个普通执行 token，包含 current/inflight；同核最多一个同步 kernel 在执行，其余槽可保存其他任务尚未执行的具体单元。同一普通 SPMD 任务不能同时占用两个位置。token 保存身份、Call/Binding 指针、执行票/固定 owner、已确认 fanin 前缀和私有阶段，不复制四份 512B 参数包。

token 与现有两个 dispatch slot 分开。slot 可继续承载当前执行、协作责任和 DFX 身份，但普通 pending 不要求先占据物理 slot。不能通过把 `SCHEDULER_PENDING_SLOT_COUNT` 从 2 改成 4 来代替此协议。具体存储选型以小型 owner-private 状态为准，不将 128B GM 对齐传播到局部栈对象。

若通过现有 dispatch slot 复用 kernel/完成实现，该记录必须引用同一执行责任并保持原 executor；不能 token 和 slot 各保存一个可独立完成的 owner。token 的 `COMPLETING` 结束必须晚于该责任的最后一次必要使用，DFX 不能依赖已经复用的槽位恢复旧身份。

以下名称是目标协议，不表示当前源码已有同名状态：

| token 阶段 | 持有的责任 | 允许的推进 |
| ---------- | ---------- | ---------- |
| EMPTY | 无 | 可以接收新票 |
| TICKETED | 具体单元的唯一新票；接收与固定准备未结束 | 不可执行、不可遗忘；有限步登记、接收并准备或报告终态 |
| WAIT_FANIN | 单元已领取且本核固定准备完成；依赖或 Graph 激活可能未齐全 | 保留原位置，推进当前未完成条件并检查本核其他 token |
| ELIGIBLE | 已满足当前观察到的条件，尚待最终资源/动态许可 | 满足最终许可才进入；不把此状态视为永久许可 |
| INFLIGHT | 实际调用已经开始 | 只能由原核返回、记账并退休，不可提前回收 |
| COMPLETING | kernel 返回或合法 predicate skip，正在归还责任 | 完成必要发布后释放 token，不从复用槽找旧身份 |

```mermaid
flowchart LR
    H[Host 发布静态 Call] --> T[有效 FAA 票固定 owner]
    T --> C[登记诊断与接收及固定准备]
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
- 有空位置：从本核目录前沿领取；同一 SPMD 仍最多持有一个未退休 block，不能因本核去重而越过未关闭头部。
- 四个位置都在等待：保留四份责任，继续观察前驱与服务独立控制路径，不领取第五份普通责任。
- 本核 block 返回并安全退休：释放其位置并复查头部；同一 SPMD 仍有余块就继续领取，不等待整个 task 完成。
- 不增设窗口外待办列表，不以切换存储位置的方式隐性扩大前视容量。

任务目录与 issue_head 只用于分配尚未领取的执行单元。已经领取的单元由原核直接观察依赖，不进入全局再次发现或竞争执行权的流程。删除 PARK 状态、共享重领索引和超时换 owner 的处理；本核 token/dispatch 表示切换不改变 executor，也不能重复计入容量。

四位置等待不等于自动证明全图活性。前驱和 Graph 控制节点仍须得到服务，通信进展遵守原协议；发现真实进展问题后须依据具体依赖与运行记录分析。禁止以转移已领取任务、丢弃责任或扩充隐性第五槽作为补救。

## 3. 快速接续的原子成本、服务边界与 SPMD 分布

### 原子操作分别算在哪条路径

| 操作 | 普通路径的目标成本 | 是否应留在已接收 token 的接续间隙 |
| ---- | ------------------ | --------------------------------- |
| 有序供给、选择本地空位 | 头部进度、前驱供给、本核去重与控制门槛；关闭证明与发票竞争另计 | 否；已有可执行 token 不依赖新的补给 |
| 初次分配具体 block | 一次有效 `next_block` FAA；超额竞争单列 | 否；应已持票 |
| 初次授予单元所有权 | 普通有效 FAA 票直接确定 owner；执行记录为本核诊断，不再追加 Claim CAS；cooperative 保留原子交权 | 否；应已持票并完成接收 |
| 读取固定 Call／接收 Binding | Call 复用本轮启动观察；Binding 两行 DCCI、尾 DSB 和私有元数据缓存 | 否；等待期间保留已完成的接收，不因等待重新竞争执行权 |
| 检查依赖和最终许可 | 只推进未确认的 fanin 前缀；动态许可按实际合同观察 | 是；不能用旧观察替代新出现的条件 |
| `WAIT_FANIN → INFLIGHT` | 唯一 owner 的私有状态转换，不新增领取 FAA/CAS 或本核 ready 队列原子 | 是；同一 helper 继续调用 |
| kernel 返回及完成发布 | 保留执行权退休、N/C/D 贡献、输出发布和必要协作操作 | 是；不能把完成成本藏在 kernel 时间内 |

普通单元首次有效分配由一条 FAA 授予不可撤销的 block/lane 票，后续接收不再独立竞争执行记录。票限定的执行记录标记 TICKET_OWNED，整个128B区域仅由该 owner 更新，使用普通 store、单行 DCCI、尾 DSB 发布 CLAIMED/RETURNED/FAILED 诊断；该区域没有其他 Device writer 或执行许可读者，不承担跨核交权。AICPU 错误快照保留原定向 invalidate，Host 在所有核退休后读取，仍保留原有 owner、phase 和完成推导。取消先核对 owner 和当前状态，失败不产生完成贡献；普通最后返回的 C 归零推导仍成立。MIX/cohort 的记录继续原 CAS/Exch 协议，两类访问不作用于同一执行记录；TaskControl 的发票、C/D/N、异步 lease/FINISH 及实际资源许可保持原子语义。不使用 st_dev，也不把“一个 writer”泛化为任意共享控制可取消原子。

普通 token refill 与 ordinary MIX 在实际申请新票之前，用一次可靠 `ld_dev(next_block)` 判定 `next_block >= N`。有效票不可撤销；越界申请的回滚只撤回超额部分，因此该确认足以设置本核当前 task 的 `issue_ticket_exhausted`。它不能设置 SUPPLY_CLOSED、发布 seal、推进目录或放行 kernel。continuation 的耗尽观察写入同一个本核耗尽位，紧接的 refill 复用本次 fill 的 issue snapshot，不重复读取 next_block，也不因此额外重扫发行头；下一轮 fill 使用新的 snapshot。refill 自己首次确认耗尽时仍清除本轮旧 snapshot；切换 task 时清除耗尽位。未确认耗尽仍走原一次申请 FAA，竞争导致的越界仍须回滚；不改变共享游标的最终精确值。continuation 的可选提示检查保持原语义，负向提示只把工作交回正式 refill，不另做原子确认。

供给游标检查分别读取一次 seal 与 next_block，不追加 `FAA(0)`。正向 seal 之后保留读侧 DSB，再检查 next_block，保证跨字观察顺序；seal 非法、游标越界或 seal 与游标不一致仍报错。实际取票和退款原子不变。

异步计算关闭入口由最后计算返回授权，所有注册已经 seal；调用方直接传入最后 physical FAA 的完成字结果，不重复读取 completion。deferred 路径在 close RMW 前执行 DSB，利用该 RMW 返回的 join 字校验并裁定唯一 FINISH owner，不追加 join 的 ld_dev。同步 checked 入口没有 close RMW，在读取 join 校验零登记之前保留 DSB。屏障排序本核最后贡献与后续共享操作，不能由贡献之前的屏障替代；并发 CHECK、payload 发布及其他数据可见性合同继续保留。

### 本轮不可变 Call 的缓存观察

Host 在启动前构造并上传全部 Call，包括当前 native Graph 的 child 实例。metadata 上传通过 entry fork 事件先于 AICore 启动；每次 resident 入口执行原有 DSB + DCI，清除该核上轮缓存。Call 在本轮内没有 Device 写者，普通接收与 MIX 提前准备均直接读取已发布 Call，不再逐次遍历512B执行DCCI，不新增参数extent或额外屏障。所有 epoch/task/execution/block/lane、参数数量、function和固定context指针校验保留。

Binding 接收仍观察 local/global context 两行，保留接收尾 DSB 与 compiler barrier。LocalContext 的 block 几何只由 Host 初始化，Device 准备阶段不再对未修改的 LocalContext 二次写回；GlobalContext 的动态字段仍按原时点写回，发布尾 DSB 保留。defer_wait 的业务 tensor 数据仍由实际消费者在依赖满足后观察；Call 地址和参数不可变不代表其指向的数据已经产出。Graph 激活、Tensor/view观察、args_dump边界和最终执行许可均保持。跨run复用依赖重新上传及每次resident入口的清缓存，不能把本轮证明带到下一轮。

### 本核复用已确认的任务完成

每个 worker 在本次 run 的私有状态中保留16个任务证明，共128B，按 task ref 取模索引，编码为 task 身份与 SUPPLY_CLOSED/DONE 等级。正向 seal 通过可靠 `ld_dev` 观察、或最后有效票 owner 的真实发布可建立供给关闭证明；可靠观察计数器阈值、SET=1 或 GM `C=D=0`，以及 GM 完成 RMW 返回的合法 DONE，均可建立 DONE。同一 task 的证明不降级，供给关闭不能放行 kernel；同一不可变运行镜像内重复检查可复用对应等级。供给检查和执行 fanin 共享本核证明，不把发完票、物理返回或负向 `ld_dev` 提示当作完成。

本核尚无对应证明时，供给检查先观察绑定的完成来源，按计数器、SET、GM 的顺序选择已有映射；只有可靠 `ld_dev` 观察到该后端的合法 DONE 才直接建立可同时服务供给及执行 fanin 的证明。尚未完成则继续检查原供给 seal，允许已发完票但仍在计算或等待异步条件的任务开放后项领取。`C=0,D>0` 仍为未完成，不能把物理返回当作 DONE。此顺序不增加缓存或共享状态；已 sealed 但未完成的任务会多读一次完成来源提示，必须在全套 L0 对照中计入这项成本。

只有已确认的供给关闭或 DONE 才写入对应证明；供给证明不能冒充完成，非法状态不增加证明。索引碰撞只遗忘旧证明，不授予新任务完成。每次 resident 启动重新构造私有状态，Graph 展开后的 task ref 仍保留实例身份，不能跨 run 或任务身份复用。首次 DONE 观察使用可靠 `ld_dev`，不再追加原子读；各后端的完成贡献协议、payload 可见性和最终 MIX/sync_start/predicate/错误许可检查均保留。此缓存不增加共享 ready 状态，也不转移执行权。

### 最终入口与有界控制服务

本核目录前沿保留当前 entry 的已校验 metadata、控制字地址及按需解码的 fanin view。当前项仍被 supply seal 或跨引擎前驱阻塞时，复用这些不可变事实并从原确认前缀继续；动态 completion/seal 仍按原合同读取。普通续领、refill 与 cooperative 发现复用相同 entry；切换目录项重新绑定，不能把静态绑定当成发票或执行许可，也不新增全局发布计数器。

普通 token、普通 MIX 的待准备组和 cooperative 发现，各自缓存当前未完成前驱的 task 身份、fanin 序号、前驱 ref 和完成来源定位（计数器描述符、SET 通知或 GM 控制字）。首次解码保留原有范围及 HostDone 检查，重复等待直接观察该来源；不缓存负向完成结果。前驱确认 DONE 后清除定位，切换 task 或新 run 时原私有状态失效；native Graph 的映射 ref 保留实例身份。该定位不授予执行或转移 owner，错误、动态许可和控制服务预算保持。

普通 token 在 Claim 及 Binding 接收成功后缓存固定 metadata，填写已确定的 sub_block_id，并完成原有两行 Binding 写回及尾 DSB，然后进入 WAIT_FANIN。该准备不占物理 dispatch slot，不读取 predicate 或尚未产出的业务输入。依赖满足后复用上述准备，只填本核 slot 并保留最终许可检查；因协作准入而恢复为本核 token 时保留已发布 Binding，重新绑定不可变 metadata。MIX/cohort 的组发布及动态 async context、输入观察仍按原执行边界进行。

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
| defer_wait | 共享等待记录、每核一份独立 async 服务 continuation | 已返回的等待 task 的普通执行 token | 单个服务槽按预算 CHECK，拿到 join 唯一责任后立即收尾；通信继续遵守原协议 |

任何类别都不允许本核同时执行两个同步 kernel。独立 continuation 的含义是等待记录和服务机会不被普通窗口耗尽，不是另外增加一套可以并发调用 kernel 的执行器。

### 普通 MIX：提前持组，不提前占伙伴

MIX 以 `(run, node_ref, block, group_generation)` 为组级 pending；AIC coordinator 仅从 AIC/MIX 本核目录前沿领取组票，active lanes 不能分别从普通 AIV 目录抢同一组的未来执行。Vector-only MIX 仍由 AIC 协调，AIC 不产生虚假的业务调用或完成贡献。

每簇最多一份未来 group pending，独立于正在服务的正式 group；同 task 的 current/pending 组仍最多一份。满足前驱发票条件后，MIX 可在逻辑依赖 ready 前取得组票，但只保留固定组责任，不设置 stage/armed、arrived、cohort hold 或业务 slot。保留单独 preparation announcement，伙伴有空时可以观察不可变 Call；该公告不授予执行权，不允许两个准备者并发写动态 Binding。

已有未来组的依赖未满足时，保留 GROUP_RESERVED、已确定的 coordinator/cluster 和已完成准备。不能因另一个组 ready 就撤销这份责任并改领替代项；已确定的真实 lane 也不能迁移到其他核。

未来组等待不阻止独立 async/control 与已有正式组服务，但不授权领取第二份未来组。未领取任务仍遵守既有目录与资源协议，不能用让出已领取组来换取后序票。

只有未来组位置空闲且当前头部是普通 MIX 时才领下一组；头部是纯 AIC 或 sync_start 就服务对应入口，不能用 MIX 专属 cursor 跳过去。MIX 的全部逻辑 block 已取得整组责任后才允许头部前进，不能把第一个组 reservation 当作整 task 已发完。等待已有组时直接重查其依赖，不执行跨簇重领。

正式路径保留 `scheduler_cluster_mix_try_arm/service/retire` 的资源裁决，消费已持有的 block/owner，不二次发票。执行顺序为：确认逻辑依赖 → 检查 hold/frontier → stage → 再检查 → 发布 armed generation → 必需 lane 到达 → 执行 → 全部返回后退休。只协调实际 active lanes 和 coordinator，不扩大为无关成员。

已 ARMED 且本 lane 尚未履行的正式责任优先于普通 token 补给。`scheduler_cluster_mix_needs_service` 按本地 PUBLISHED/ARRIVED/RUNNING 或尚未观察的新 generation 判定服务优先级；它不授予执行许可。`scheduler_fill_own_slots` 保留已有 READY/direct slot 的排空入口，然后优先再服务正式组一次，不插入普通 progress/refill。resident 到期预补给也服从这个判定，coordinator 的 RELEASED 先得到 retire 服务。已经 finished 的 peer 不因同一代 staged mask 尚未清零而重复承担到达责任，可以继续有界准备。future reservation 不触发这项限制；async/control、错误检查及现有 slot 回收仍按原路径推进，不能通过跳过整个主循环来实现优先级。

组归属或真实执行成员一旦确定，等待期间保持不变。stage 的有界发布和失败清理不授权转交已领取任务；armed/arrived 后不迁移或拆组。部分 lane 仍在运行时不覆盖 SSBUF generation，也不让它们与后续 block 配对。

组所有权与 active lane 的执行责任不是一回事；不能将普通单 lane 的“一次 CAS”承诺套到整个 MIX。组级/每 lane 的实际授权和交接操作须分别统计，保留同一 lane 只执行一次的验证。

### sync_start：独立 continuation，真实成员仍按许可认领

每 worker 为有序 cohort 保留独立 continuation，不让四个普通 token 占满后阻止 admission 服务。Host 已知每个 block 的 Call 可以提前准备；未来 cohort 的候选不计入真实成员数、claim seal、hold 或启动完成数。独立的是服务位置；候选和真实成员仍遵守 Host 排序及共同 admission 协议，不另造任意乱序的协作任务供给链。

先在 Host 检查 sync task 的同时驻留需求：超过资源池总容量应拒绝；总容量足够而暂时忙碌，按 admission 等待。sync 不沿普通新票路径提前分配真实成员票；真实依赖满足、资源与 admission 允许后，实际可用 worker/cluster 动态认领成员，完成参数接收和会合。发布候选或日志不等于全部成员已领取，目录头部越过该项须有真实 claim seal；不能用独立 continuation 绕过这项供给门槛。各核不能分别占据不同 cohort 的部分成员互相等待。已承担责任由原成员履行，不用预先指定的核代替真实到达。

提交候选前以可靠 `ld_dev` 读取该 task 的 admission 状态。ADMITTED 在当前 run 内不可撤销，观察到后检查当前 run，即可清除本核重复 pending，不追加 `FAA(0)`，也不争用全局 commit_lock。其他状态继续通过 commit_lock 的 CAS 取得发布权，并在取得锁后重新可靠读取 admission 状态和 commit_tail；不得复用锁前快照。ADMITTED 可以先于 commit_tail，所以清理提案不接纳协作义务、不发成员票，仍等待 committed prefix；它可以解除本核普通 token 的 pending 阻挡，但实际入口仍按原 tail/资源协议检查。退出排空对重复 pending 使用相同的 ADMITTED 读取规则；record/log/ADMITTED/tail 的发布次序、atomic 发布、cache 屏障及错误和 epoch 检查保持。

保留 full/partial 容量检查、共同 cohort sequence、最后有效成员 seal、局部退休及 frontier。普通 pending 不是资源许可：入口必须服从现有 stage/hold/armed 仲裁。已正式公布的普通 MIX 与已有 cohort 双重责任按同一裁决服务；相邻 cohort 不能形成循环等待，也不新增“上一整个 cohort 全部完成才开始下一个”的全局屏障。

sync_start 成员完成本核唯一 block 后，纯核的 slot 全空、或 MIX coordinator 的组已回到 IDLE 且 slot 全空，即可复用当前义务的 `sync_claimed` 关闭本核责任。该位只在有效领取计数 RMW 通过范围校验后设置，接纳新义务和退休时清零；不增加私有或共享状态。它只证明本位置不再领取第二个 block，不证明全局发票结束，因此不能设置 SUPPLY_CLOSED、替代正式 claim seal 或开放后项。未领取 block 的位置仍须强确认全局游标达到 N；MIX 伙伴仍观察协调者实际发布的关闭前缀。run error、epoch、slot/group、sequence 检查保留。

DFX 用 `MEMBER_FROM_CLAIM` 区分本核领取证明与新鲜游标读取，前者的 `closed_cursor` 为无读取哨兵，claim-seal/cursor-seen 为零。审计必须找到该成员唯一的真实 block claim，连接 claim→本地 closed，不制造 closed 时的全局计数值，也不增加 closed→其他成员随后 claim 的虚假时序边。零工作成员和 MIX 伙伴不能冒用此证明。

sync MIX 关闭交接统一到已有 SSBUF `sync_frontier[0]`，删除按 task/cluster 分配的 GM close seal、协调者发布 CAS 和伙伴确认 FAA。AIC 是每个 MIX 的协调成员，包括不参与计算的 vector-only MIX；它只在本组真实退休、slot 全空且当前错误/epoch 校验通过后关闭本地义务。零工作协调者仍先确认全局发票结束。关闭前保留 GM 发布屏障，再推进自身 frontier；这只表示 admission 前缀已退休，不能推出 task DONE。

启动路径先确认本轮 cluster INIT，保证旧 SSBUF 已清除；epoch 字段本身不替代 INIT。伙伴只有在当前 run 的协调者 frontier 大于本地 order 时才能关闭，且继续检查自身 group/slot 已退休；读到超过 admission 容量的值必须报错。共享 mailbox 已被后项复用时，重新取得同一关闭前缀证明再退休旧项，不能误消费后项 descriptor。已不参加后续 lane mask 的落后伙伴也可从单调前缀证明旧项退休，不需要每 task 保留另一份 GM 通知。

DFX 记录协调者的关闭及伙伴观察到的 frontier，关联真实的协调者关闭前缀，删除 GM seal 地址、stride 和发布/确认语义。不能把一次 SSBUF 读取画成 atomic，也不能把本核关闭记录错误地放在伙伴已观察到的发布之后。Host 布局、初始化、运行时引用、导出与协议审计同时维护这一份关闭状态。

协作场景的 5 µs 要区分静态接收、实际成员到达和必需同步。不能为了缩短泳道直接跳过成员许可。

### defer_wait：物理返回与逻辑完成分离

独立的 defer_wait task 登记条件后物理返回，释放普通执行位置。每核复用一份 `SchedulerAsyncLocalState` 服务共享等待记录，单次只持一个 CHECK 工作项。计数器路径在条件完成时贡献 Q；GM 路径的 FINISH 由 join FAA 唯一选出并立即处理。该服务不占四个普通 token，也不是第五个业务执行槽。多个等待记录保留在共享表，不逐个占满普通窗口。

```text
defer_wait 登记条件并物理返回，释放执行位置
    → 消费者可以在合法前视窗口内等待，保持原 owner
    → CHECK 观察实际通知条件
    → 计数器路径贡献 Q；GM 路径由 join FAA 选出唯一 FINISH_OWNER 并结清信用
    → 消费者在其真实依赖全部完成后执行
```

非空组只有首次成功 seal 的执行者可以初始化 CHECK lease。完成条件 payload、适用后端的债务/join 及检查进度发布后，唯一 sealer 用单写者 st_dev 初始化 RUNNABLE；显式 DSB 保证初始化先于目录 bit 发布。旧 lease 存储未初始化，不读取或校验旧值；重复 seal 必须在此前由组的 OPEN 状态检查拒绝，不得覆盖 RUNNABLE/OWNED lease。CHECK 领取、让出、完成继续使用原有带 generation 的 CAS。

`scheduler_service_async_slice()` 每两个活跃服务机会允许一次 CHECK，首个机会可检查；保留既有 CHECK 频率，删除 FINISH 后不加倍争抢。CHECK 最多观察8个条件，未满足时保存进度、归还 lease 并轮转。任何核不得带 CHECK lease 进入业务 kernel；持续命中快入口时也按第3节预算服务控制路径。

以下 join/FINISH 与 N16/C16/D32 归零步骤适用于未映射完成计数器的 GM 路径；计数器任务通过 P/Q 达到累计阈值后直接就绪，再执行 canonical 归档。

最后计算返回或最后条件组完成的 join FAA 返回 `NONE / WAITING / FINISH_OWNER / ERROR`。唯一 owner 携带本次原子返回值推导的 join 终态及 task/run 身份，在同一调用链中直接检查 canonical completion 并 CAS 到 DONE；不再发布 FINISH lease/目录，不重新竞争责任，也不将 outcome 留到下一次 kernel。CHECK 侧先终结自身 lease、清除 CHECK 目录，再处理 join。owner 产生后的错误必须使 run 失败，不能重试或丢弃责任；最终 CAS 成功者恰好增加一次 pending_completed 并记录真实 DFX 完成因果。计算返回和 MIX 信用外层不能重复计数。

FINISH owner 从本轮不可变 task metadata 的 block 数与活跃 lane 数得到总贡献数 N，由已确认的 join 结果验证存在合法非零登记；canonical debt 只保存单个 has_async 门闩，预期 completion 为 `(N, C=0, D=1)`。校验 N 与 join 后，直接 CAS 该预期值到 `(N, 0, 0)`，不先 FAA(0) 再读取同一个完成字。CAS 返回值是唯一成功依据；剩余计算未归零、债务不符、总数不符或重复 FINISH 均失败且不修改原状态。保持原有发布屏障、run/实例身份校验及 DFX 因果，不把静态 N 或 join 本身当作任务 DONE。

FINISH 目录、lease、扫描游标及独立服务轮换同时删除，Host 分配、Device 校验和内部布局同步更新。`SchedulerAsyncLayout` 仍128B，公开 AsyncCtx、AsyncWriter、AsyncGroupData 和条件记录 ABI 不变；不改模型 kernel 或前端表达。

异步目录更新先用可靠 `ld_dev` 取得初始 compare 值，成功、重复及重试均以 CAS 返回的旧值裁定，不再先做 FAA(0)。这是多写者状态更新，读取结果不授予修改权；当前疑似重复也通过 identity CAS 裁定。CAS 失败后复用返回的完整目录字，只改变本记录对应 bit。一次 run 内每 bit 仍只有 0→1→0，清除后不复用；当前保守上界为128次 CAS，覆盖初次 compare 失配、其余63位各两次变化及最后成功。可靠读取不代表该值在后续 CAS 时仍未被其他写者改变。保留调用者的 payload/终态发布屏障，不新增共享存储。

复用现有 N16/C16/D32 完成字：物理返回归还相应份额，非空异步组的信用由唯一 FINISH 结清。取票耗尽、kernel 返回和目录顺序都不能伪造逻辑 DONE；保留条件登记与最后返回交错的 join 证明，以及真实归零原子的 DFX 因果身份。

Host 不推断本卡或跨卡 notify/defer_wait 对应关系，不新增通知排序边、通信上下文快照或 PyPTO 元数据导出。kernel 内同步和跨卡通信继续遵守原协议；若真实场景证明进展不足，再按 [#3096](https://github.com/hw-native-sys/simpler/issues/3096) 分析，不能把人工 self-notify 用例当作当前模型需求。

### native Graph、predicate 与安全退出

Host 预构造每个实例的 Call、边界 alias/view、scalar 和 heap 地址；Token 用 run + node_ref 区分实例。拓扑分层覆盖实例实际数据和 ENTER/CHILD/EXIT 控制关系；排序不改原 ref 和参数绑定。child 只有在元数据已发布、必要的 ENTER/激活条件满足时才可领取；执行仍等待真实计算前驱 DONE。普通前驱全体发票不能替代实例激活或控制门槛，目录投影必须保留这些约束的检查位置，不能因只看可执行节点而丢掉实例边界。外部消费者仍遵守 EXIT 合同；未发布实例不发票、不读取占位指针，不补 Device 动态生成旁路。

predicate 仍在合法执行边界读取；false 分支按既有整任务/组语义完成 skip，贡献恰好一次，不能让 MIX 某 lane 跳过而伙伴进入等待。提前取得固定 async 定位信息不代表创建 writer 的可写状态、登记条件或发布信用。

错误、取消和 EXIT 按责任清点：TICKETED 新票、OWNED pending、正式 MIX/cohort、INFLIGHT、完成贡献和 async lease 都必须有最终归宿。只能按现有错误协议终止责任，不能转交其他核重跑同一 execution；inflight 等真实返回后退休。Host 不能以 cursor 发完或某一核本地窗口为空就回收 Call/Binding/Graph heap。

需要同时证明“所有执行单元已实际返回或按失败合同终止”“没有遗留的已领取责任”“组资源和 async/control 已安全退休”。run/实例身份及协作 generation 不回绕，越界或旧身份不能当作正常重复消费。内部布局与导出协议变更时同步更新 Host、Device、reader 和测试。

### 缓存与存储合同

“Wait 的写侧可以考虑 st_dev”必须先拆成逐字段协议。决定方式的是该值由谁更新、谁依据它获得什么权限、是否复用以及整条 cacheline 的访问合同；不能只看名字里有 ready/done，也不能只数 writer。

| 字段或区域 | 当前写入职责与方式 | 当前读取方式／对后续评估的约束 |
| ---------- | ------------------ | ------------------------------ |
| 不可变 Call、固定图 metadata | Host 在启动前构造及发布，run 内无 Device writer | 启动观察后只读；稳定地址不保证业务 Tensor 内容已可用 |
| 普通 TICKET_OWNED 执行诊断 | 有效票的固定 owner 使用普通 store + DCCI + DSB | 同 owner 生命周期检查；AICPU/Host 诊断观察；不用于另一核取得执行许可 |
| `next_block`、部分 sync 的 `claim_tickets` | 多核 FAA 分配或合并，必要时处理超额票 | 读快照可以减少无效尝试，实际分配仍由 RMW 裁定 |
| 完成计数器、`completion_state`、async join／lease／FINISH | 计数器路径用 INC 聚合 P/Q；GM 路径用 CAS/FAA 合并债务和裁定唯一责任 | 优先观察计数器阈值，其次 SET，最后 GM；GM 归档不阻塞计数器就绪；所有权与多写者合并不能改成普通读后覆盖 |
| `claim_seal`、`admission_state` | 最后有效领取者、提交锁持有者分别使用 scheduler_store_signal 的 st_dev；各字段独占128B，run 内无其他 RMW writer | FAA/登记→seal 的前置 DSB、状态→commit_tail 的已有 DSB 保留；不因跨核通知本身添加写后 DSB，供给关闭不等于计算 DONE |
| GM MIX assigned/arrived/released 序号 | 当前保留 atomic store/Exch | ACK 行混有多个 writer 的字段，released_seq 还联合编码序号与失败位；本轮不机械混入 st_dev，后续须单独论证原子发布需求和整行访问合同 |
| admission `log[tail]`、`commit_tail`、错误 status | `scheduler_publish_signal` 已使用 st_dev + DSB；提交锁或错误认领决定活跃 writer | 按统一硬件原则和逐字段发布合同审查；不能将这些单写者路径外推为可替换所有多写者 RMW |
| SSBUF group/generation/frontier | 既有专用 SSBUF store/load 与 release/acquire | 不属于 GM st_dev/ld_dev 替换范围，必须保留其硬件及组生命周期合同 |

对候选 st_dev 发布，必须先给出数据→通知的顺序、reader 的 payload 可见性、整行不被旧脏缓存覆盖、精确位宽/对齐、重复写及 writer 交接、epoch/sequence 复用边界。DSB 只建立相应本核内存顺序，不把普通覆盖写变成多写者 RMW。官方 [WriteGmByPassDCache 说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/scalar_compute/WriteGmByPassDCache_ISASI.md) 明确要求处理旧脏行写回覆盖及 DCCI 与 bypass write 的顺序。

工作区 `simpler-fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md` 的原始结果继续作为证据，旧的全面限制 st_dev 的工程建议不再覆盖本节统一原则。重复写仅尾部 DSB 的失败，不能推广为显式满足每条时序依赖的 st_dev 发布都不可用；逐写 DSB 的通过样本也不能授权覆盖多写者更新。既有 [claim_seal 替换调查](../../investigations/2026-11-a5-hbg-admission-shortcuts.md) 保留当时撤回的结果，新候选按实际 writer、发布顺序和生命周期判断并做相应验证，不改写历史为成功，也不把旧结论作为永久禁用理由。

- 共享 RMW 控制按 **128B atomic-only 隔离**设计；64B 是 Scalar DCache 操作粒度。RMW 行内不混入 ordinary store 或 DCCI 数据。普通有效票独占的执行诊断区域属于 owner 数据，整条区域不再有 RMW 或跨核许可观察，独立使用 DCCI 发布；不能与 cooperative 控制记录混用。
- Call 复用 Host 上传与启动清缓存建立的本轮只读可见性；初次 Claim 成功后，Binding 继续两行 DCCI、尾 DSB 与 compiler barrier。该接收可提前到 WAIT_FANIN，但不能据此省略前驱计算及 defer_wait 数据的执行前观察。
- 每核 token 只由固定 owner 写，可使用 owner-private 存储；初次执行授权和共享完成仍使用原子协议。本核等待状态不作为跨核交权信号。
- 执行器真实返回时，普通有效票的 PARTIAL 和失败记录按 owner 数据发布；MIX/cohort 的真实返回继续 atomicExch，组 reserve/arm/acquire 与对应取消保持 CAS。普通最后一次成功返回可复用贡献 FAA 的 C 归零证明，省略该单元的 RETURNED 发布；D>0 时 task 仍未逻辑完成。检查入口继续拒绝错误 owner、状态不符和已推导退休的旧句柄。完成贡献前的 DSB、失败返回在组 ACK 前的 DSB 均保留，执行责任退休不替代任务 C/D/N 完成计数。
- 可写动态 Binding 延后到最终许可后由固定 executor 填写；准备者不能修改属于另一 executor 的可写上下文。等待期间保留参数与观察进度，必要的缓存可见性协议不因固定归属而省略。
- 提前观察参数地址不等于提前观察业务 Tensor 内容。受前驱或 async 保护的数据仍在实际依赖满足后执行必要的数据观察。
- 沿用启动路径的 DCache 清理合同，不引入每 task 全 cache DCI，不将 DCCI 误当按地址 invalidate-only；SSBUF 使用其专用 release/acquire 屏障。
- 共享记录、Call 和 Binding 在 run 内不复用地址，直到安全退休。记录 token、SSBUF、Host 初始化和实际 H2D 成本，不新增用于跨核再次认领的元数据或复制 probe 的固定大 arena。

## 5. 开发步骤、DFX 与验收

### 按三条原则审查及实施候选

Host 拓扑排序、两个有序目录、固定 owner、四 token 及协作/异步边界已经构成基础约束，不重新从“实现 task 有序”开始。下面是后续候选的审查顺序；本次原则和源码核对不自动代表开始下一轮性能开发。

| 步骤 | 核对或拟修改内容 | 主要入口 | 可审查的完成条件 |
| ---- | ---------------- | -------- | ---------------- |
| 1. 明确阶段和字段角色 | 列出每项准备的最早时机、所有权授予点、等待条件与发布合同 | `host/prepared_execution_plan.h`；`scheduler_prepared_execution.h`；`scheduler_memory.h` | 静态 Build、本核准备和最终入口分开；对每个共享字段明确 writer、reader、生命周期和真正权限 |
| 2. 审查 Build 覆盖率及启动尾部 | 对普通与 MIX 分别识别可提前工作、可复用的校验/metadata及真实动态工作 | `scheduler_execution_tokens.h`；`scheduler_gm_mix.h`；`scheduler_cluster_mix.h`；`aicore/resident_scheduler_impl.h` | 准备确实早于前驱完成，已有准备得到复用；不提前占伙伴、读未就绪输入或授予动态许可；统计未使用准备及存储成本 |
| 3. 审查领取和生命周期成本 | 分类有效 FAA、超额票、组 CAS、独占诊断发布和完成贡献 | `scheduler_issue.h`；`scheduler_admission.h`；`scheduler_prepared_execution.h`；`scheduler_completion.h` | 指明减少的是哪个地址的何种请求；固定 owner、取消、失败、退休推导、AICPU 错误快照及 Host 导出保持一致，不能只验证成功快路径 |
| 4. 审查 Wait 与必要控制 | 区分单次前驱观察、整轮重复扫描、MIX/async/control 服务和最终检查 | `scheduler_ready.h`；`scheduler_dispatch.h`；`scheduler_async_finish.h`；Graph control | 同时记录失败轮询密度与完成发现延迟，服务预算保持有界和公平；不能重用过期负向观察或节流 kernel 入口终止检查 |
| 5. 冻结并逐项裁决 | 功能、IR、L3 因果及12组配对 L0，记录 Host/H2D/内存成本 | 既有测试、导出器和 `dsv41_scheduler` | 使用同一输入与计时边界；完整披露逐项回退和保留理由；单独本地提交保留项 |

表中未写目录的头文件位于 `src/a5/runtime/host_build_graph/runtime/scheduler/`。任何候选都必须保持 MIX、sync_start、Graph、defer_wait；不能通过禁用这些路径、转移已确定 owner 或新增模型特判来获得数字。新 st_dev 发布按统一原则及逐字段合同实施，不能机械替换多写者 RMW；分地址 claim 和 STARS Wait 分别按上述所有权与资源边界开发。提交前重新核对上游和本地状态，不使用过去 PR 的状态代替当前源码验证。

使用当前 a5-onboard 构建流程，只构建所需运行时。遵守仓内 CCEC 函数跳转约束，对实际编译目录运行 A5 IR 检查，尤其关注过度对齐局部对象地址逃逸到 outlined call；不通过全量强制 inline 掩盖问题。检查结果、人工豁免范围和实机验证分别记录，不把未覆盖的编译单元写成全部通过。

### 功能验证使用明确事件

复用现有 ordered execution、gm_mix、async_control、condition_boundaries、Graph 和协议导出测试；只补新的责任边界，不为每个 helper 增加镜像测试。CPU 测试用显式 gate/ACK 构造状态，A5 验证真实数值、身份及退出；不以睡眠或相对运行速度证明某次交错发生过。

| 场景 | 必须观察到的结果 |
| ---- | ---------------- |
| Host 顺序 | 现有数据与 Graph 控制依赖组成无环图；验证正向层号、层内 b-level 和稳定 ID 次序，逐边 rank 合法；两个目录无遗漏/重复，非法输入和排序启动前拒绝 |
| 分层无执行屏障 | t9 的前驱完成时，即使同上一层的独立 t11 尚未完成也可执行；不增加全层完成计数或屏障 |
| 静态检查与动态进度 | 关闭前缀表示本类供给进度，不等于 DONE；不能替代另一引擎状态；Host 已完成豁免、Graph 激活及控制门槛不丢失 |
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
| 延后完成 | 等待登记后释放执行位置；真实条件满足后才发布逻辑完成；保留早到、晚到和登记中满足的交错验证 |
| 快路径持续命中 | CHECK/FINISH 与 Graph control 得到有界服务；普通窗口有空位置时继续新票发现，不饿死已有等待责任 |
| MIX/cohort 交错 | ordinary/MIX/sync 遵守统一目录头部与各自资源协议，完整组责任或真实 seal 才关闭供给；已承诺 cohort 次序不被破坏；复用现有 lane、stage/hold 和相邻 cohort 覆盖 |
| Graph 与 predicate | 多实例不串参；ENTER/EXIT 不提前；false 分支合法退休，未发布元数据不被读取 |
| async 与退出 | 条件早到/晚到/登记中满足均不提前 DONE；物理返回释放资源；所有未退休责任阻止成功退出 |

### L3 必须展示真正的前视和串行链

事件统一关联 run/node/block/lane/token/固定 owner，并能还原 Host 层号/rank、各核目录位置及跨引擎前驱供给进度；区分头部未发完、本核去重、initial ticket、Claim/acquire、WAIT_FANIN、最终许可、kernel、完成、MIX/cohort 和 async/control。显示本核原位置等待跨度，但不将等待当作持续 Scalar 工作。

当前有序供给的三个阶段映射到 `scheduler_query_issue_head`、`scheduler_issue_fanin_ready`、`scheduler_task_supply_closed`。保留这些阶段的供给语义，Host 重排目录时同步其身份和等待原因；不得把等待发完票显示为等待计算完成。atomic、ld_dev、DCCI/DSB 沿现有 primitive 记录展开，缓存命中不伪造共享查询事件，Host 导出和 Python 消费者一起更新。

另按 `(run, node_ref, block, lane, execution_index)` 关联三条原则的证据：静态 Call 发布、本核有效票、固定准备完成、等待对象/条件、最终许可及真实 kernel 开始。当前普通 token 的 `acquired_cycles` 在固定 Binding 准备之后取得，不能将其解释成单条 FAA/CAS 的完成时间。已有 DFX 缺少某个端点时注明不可判定，不用 ready 显示条的边界冒充事件；新增记录只在对应 profiling 模式中采集，不给 L0 添加追踪成本。

Build 同时展示准备工作量和它与前驱计算的重叠；领取同时展示同地址参与核数及实际 RMW 次数；Wait 同时展示地址/身份、轮询次数、请求密度和条件满足后的发现延迟。未采集地址或返回值时，L3 只能按函数归因，不能声称已经完成逐地址证明。完成贡献和动态入口仍算在真实交接中，跨核重叠 span 不累加成串行节省。

必须回答：

- 每次进入 kernel 时，已有多少具体 pending？多少已完成 Claim/acquire，多少只是 TICKETED？
- 对前序 A 的余块，本核退休后是否先补 A1？首张后项票是否晚于 A 全体有效票？头部供给等待有多少来自宽 SPMD、本核去重或真实成员准入，Host 排序是否让独立分支更早获得机会？
- defer_wait 的物理返回、条件满足和逻辑完成是否分别记录？
- 返回后的下一次调用来自已有 pending 还是临时补给？若缺少 pending，原因是前序合法票优先、控制门槛、窗口容量、同 task 去重、资源会合，还是没有获得补给机会？
- 参数接收是否早于前驱完成、早于本核前一次 kernel 返回？后继进入时还缺哪些操作？
- `G_worker` 和 `G_dep` 中是否出现新票、重复 Claim、目录扫描或 READY 中转？快/慢路径分别占多少？
- 目录关闭确认、首次取票 FAA、Claim CAS、供给/执行依赖确认和完成原子分别多少？已持有任务就绪后是否再次取票或 Claim？不能用一个总 atomic 数替代分类。
- 所有 ld_dev、atomic、DCCI、DSB 独立标识；residual 包含普通访存、控制和 profiling，不当作纯 Scalar 指令时间。
- token 扫描含有多少 kernel 调用？inclusive TokenScan 不能当成可移除的调度耗时。
- SPMD 的参与核数、全部 block 起跑跨度、每核执行次数和第二轮尾部是否变差？是否有闲核拿不到已预留的工作？
- 未领取任务的供给、前驱的可达性、组资源退休与 CHECK/FINISH 是否可证明进展？

延续任务计时与 Scalar observer 的编译分离；必要事件的原始时间戳保留，导出/编码尽量不干扰中间计算调度。记录丢失、截断或不完整因果关系必须显式失败，不能用缺失事件表示零耗时。

`G_worker/G_dep` 使用真实 kernel 起止时间戳，不能用包含 setup/receive 的显示条边界代替。按任务对列出各次采集的间隙、低于 5 µs 的比例和最坏值，并同时列出有 pending 的覆盖率；中位数低于 5 µs 不代表所有交接达标。协作等待、真实依赖等待和调度开销分别归因，但原始间隙必须完整保留。

### CntNotify：逻辑完成通知与后继等待 {#cntnotify-done}

当前完成观察按计数器、SET 通知、canonical GM gate 的顺序选择后端；每个任务只使用其已发布描述符指定的完成事实。

- 具备完整映射的计算任务使用单调完成计数器。Host 只让真实 DAG 祖先链共享同一对象，
  每任务保存累计阈值。物理返回 P 和可能异步 lane 的条件完成 Q 分别贡献；显式同步 lane 只有 P。
  `ld_dev.b32` 读值达到该任务阈值即证明逻辑完成，不等待 canonical GM 的后续归档。
  超过整链 limit 的值属于协议错误；超过旧任务阈值但未超过 limit 可以来自后续合法贡献。
- 未使用计数器的任务，包括控制节点和计数器映射不足的任务，在有 SET 资源时使用独占的
  二值 DONE 通知。其 GM 完成门仍合并计算与异步债务，唯一完成者发布 `st_dev.b32(1)`；
  后继非消费式读取，0 为 PENDING、1 为 DONE。只有物理返回而异步条件未清不能发布 DONE。
- SET 资源也不足时，消费者直接读取 canonical GM completion；CPU/sim 使用该路径。
  SDK 创建资源耗尽允许回退，其他 SDK 错误使准备失败。Host 分别记录计划与实际映射覆盖。
- CntNotify ADD 不返回旧值，不能据此领取唯一执行票。当前 A5 后端持有 SDK 提供的 SET/read
  映射并校验 32 B 对齐，再使用同一 slot 的 +0x08 INC 地址；只发送已实测的写 1 命令。
  此硬件合同限于下面记录的芯片和软件栈，不是通用 SDK 地址保证。
  有效 SET 映射若不是 32 B slot 基址，则不派生 INC 地址，整链转入 SET/GM 完成路径；
  该能力回退不吞掉 SDK 创建、取 ID 或映射失败。
- 地址描述符独立于 GM atomic-only 控制行，随 prepared publication 和 capture/replay lease 保留。
  每轮 AICPU 在任何 READY 发布之前清零每条计数链；SET 通知按节点是否已由 Host 完成初始化为
  DONE 或 PENDING。尾部屏障保证复位先于 READY，不依赖 Host 再次 bind。
- 后继第一次绑定前驱时保存完成来源，同次等待不重复查地址表；正向 DONE 证明可按本轮任务身份缓存。
  未就绪即返回原有有界推进循环。完成观察不转移执行所有权，也不替代 kernel 入口、
  Tensor 数据可见性或 MIX/sync_start 的许可与退休检查。
- SRAM 访问不做 DCCI。业务输出发布、异步完成和消费者获取所需的 DSB 保留，
  不给每次通知写机械添加屏障。MIX 的 P 贡献晚于 coordinator 收齐全部资源 release。
- 验证实际 AIC/AIV 访问、重复复位、P/Q 区分和资源回退；性能采用下面的 12 组
  AICore L0 配对标准。启动复位和 Host 准备成本单列，不计作消失的工作。

### GM MIX 的 CntNotify 握手

sync_start MIX 的远端 AIV assigned/arrived/released 使用独立 SET/read 通知。
GM descriptor 和已有 SSBUF 协议保留；每个 cluster 的六个映射地址放在 mailbox
只读区域，不与 ACK 原子行混用。非 sync_start MIX 不申请这批资源。
Host 以全部 MIX block 数作为每簇序号上界，上界小于2^31且六个映射齐备才启用；
其余使用完整64-bit GM协议。SRAM release的bit31映射回原bit63失败位，不能截断序号或吞掉失败。
每轮 AICPU 在 READY 前清零，发布、接收与资源退休边界的既有DSB保留。

GM MIX coordinator 的自身 ARRIVED/RELEASED 只更新本地 confirmed 与 failure 状态；只有远端 lane 发布共享 ACK。唯一 ACK 汇合者已经持有自身到达/退休的直接证明，不需要再给自己写 GM 通知。远端到齐、完整 resource 退出、必要 DSB 和 DFX 的真实逻辑事件仍保留。

已发布的 `claim_seal=count` 直接提供终态游标，不再重读 `next_block`；DFX 的终态游标由 seal 证明，不表示额外 GM 读取。admission 接收方通过 committed tail、日志及 record 的 epoch/order 确认发布，不再查询同一任务的 ADMITTED；提交端去重与 kernel 入口许可不变。最后一次 physical FAA 将完整 post-gate 私有值传给 async close，close 使用必需 RMW 的返回值校验 join，正常 deferred 路径不再重读这两个字；同步 checked 入口仍保留必要的零登记校验。

async 登记的 canonical gate 只保存一个 task 级 `has_async` 门闩，真实 registered/pending 数量保留在现有 join。每个非空组照常执行一次 join.register FAA，并复用其返回值识别首注册者；只有首注册者执行 canonical 校验 CAS。它设置门闩之前不能提交自己的 physical return，所以其他组先返回或完成 CHECK 也不能误关 task。正常 gate 的 debt 为 remaining + has_async，close 校验门闩与合法非零 registered，FINISH CAS 从 `encode(total,0,1)` 转为 DONE；不新增等待 owner 或资源，零登记任务保持原同步路径。DFX 的 completion_debt 表示物理余量加门闩，不能当作剩余异步 group 数。

CHECK lease 在目录发布之前只有唯一 sealer，可用单写者 st_dev 初始化 RUNNABLE；保留初始化后的 DSB，随后才发布目录位。发布后的 claim、release、finish 仍使用原子交权，不能将初始化证明扩展到多写者更新。

成功领取 CHECK 后，只重新接收 AsyncGroupData 的首64B可写行和 progress 行，两条 DCCI 之后统一 DSB，再读取两者字段；header 后64B由 Host 初始化且整轮不可变，复用启动观察。目录与 lease、条件读取、CCU 事件消费及消费者 tensor 获取边界均保留。

### 按 lane 分类完成能力

`async_capability` 是 kernel 的静态完成合同，随 CoreCallable 保存并进入缓存身份。当前兼容阶段在注册入口把缺失声明、`None`、条件字段为零的显式 `UNKNOWN`，以及旧序列化格式的全零声明，统一转换为 `DEFERRED(max_conditions=64, condition_types=3)`。不得把未知能力当作同步，也不能由某次运行没有登记条件推断为同步。转换在持有的副本上完成，缓存身份使用转换后的声明，不修改调用者的原始缓冲区。非法类型、版本、字段及边界仍报错；显式 UNKNOWN 携带非零条件字段也不接受。

公共 `set_callable_tables()` 保持四参数接口。A5 HBG 在准备阶段通过 HostApi，按当前 callable ID 借用注册器持有的只读能力表，并校验其长度与函数表一致；重新绑定或查询失败都会清除旧能力引用。能力表随 callable 注册生命周期保留，不在每次准备时复制。TMR 不获取该视图，也不参与能力表绑定。

Host 和 Device 内部只保留 `SYNCHRONOUS`、`DEFERRED` 两态，不恢复 UNKNOWN 调度分支。注册入口验证并归一化每个 kernel，包括本次未使用的入口。Host 在 Graph 展开后只按本次实际可执行的 leaf kernel、每个活动 lane 选择容量；未使用的合法 DEFERRED 声明不会给纯同步图增加异步布局。补齐上游准确声明仍有价值：未知入口按最大容量保留异步上下文和 P/Q 完成协议，只有显式 SYNCHRONOUS 才能享受零布局和同步快速路径。

- `SYNCHRONOUS`：Host 容量为零，无 group、lease 和条件数组；Device 跳过 begin、条件数量检查和 seal，只提交物理完成 P。
- `DEFERRED`：Host 按已验证的条件上界和类型声明分配；Device 返回时提交 P，空登记或最终 CHECK 提交 Q。

纯同步图的 async layout 大小为零，不准备异步上下文，不探测 activity，也不运行 CHECK 服务。混合图只为可能异步的 lane 保留资源；同步 lane 不因同 task 或同图中其他 lane 的能力而分配异步组。预备 context 明确写入零 async 字段和无效 group 索引，重放不能沿用先前执行的登记状态。

CntNotify 每任务目标贡献为 `block_num × (active_lane_count + possible_async_lane_count)`。同步普通执行返回仅写一次 INC；可能异步的普通执行零登记时可合并提交 P/Q 两个信用。有异步登记时，P/Q 分开提交。MIX 的 Q 只由具有异步组的 lane 提交，P 仍由 coordinator 在所有资源真实退休后按活动 lane 数提交。例：AIC 同步、两个 AIV 可能异步，每 block 为 P3+Q2，共5，不是6。

SYNCHRONOUS 只描述自身完成语义，不表示前驱数据已经可见。前驱逻辑 DONE、正常依赖获取和必要的缓存/顺序操作仍然成立。Host 沿真实依赖传播 DEFERRED 祖先信息，只对确有此类祖先的任务设置输入获取标记；其他任务不查询全图 activity。所有执行计划都必须基于入口归一化后的能力完成这个计算，内部缺失或无效能力仍拒绝准备。带标记的消费者保留实际执行核上的获取。不能因为消费者自身同步就删除异步前驱的数据可见性保证。

验证关注纯同步图零布局、混 lane 信用总数、DEFERRED 的零登记/实际 CHECK、UNKNOWN 和旧调用归一化、无效声明被拒绝和重放。未知入口即使本次没有实际登记也走空 seal；真实登记仍等待 CHECK 提交 Q，不能提前发布 DONE。性能仍使用同一份业务二进制与注册元数据，对照12组 AICore L0 wall；声明来源和实际生效覆盖需要一并记录，不把未传入 runtime 的源码分类当成优化已生效。

### CntNotify 资源复用与控制状态选型

DeviceRunner 按 pipeline slot 和用途保留通知池。只有没有 publication/capture lease
持有的池才可复用；live/captured 使用者占用的池独立保留，不能在其退出前复位给新运行。
每轮 Device 的 READY 前复位保持不变。普通关闭在 context 有效时显式释放映射与handle，
失败保留供重试；确认设备 reset 后只 abandon，析构不调用失效context上的SDK。

控制信号优先迁移到 CntNotify，并按完整状态字的读、写、初始化及复用合同选型：
单写者值/序号用SET；exactly-once贡献的数量汇合可用ADD，不同条件也可用不进位的权重ADD；
需要条件身份和幂等重复时可用BIT_WR掩码置位。当前实机已验证SET及BIT_WR，
ADD和BIT_CLR的正式直接映射仍被当前SDK/驱动拒绝；ADD已通过下述SET slot+0x08偏移访问实测，BIT_CLR未作同等证明。
只有依赖返回值裁定唯一责任的CAS/FAA等继续使用GM atomic；同一锁若仍用GM CAS，
不能单独把它的解锁写搬到SRAM。普通payload、Tensor数据、GM descriptor不属于控制通知替换。
原64-bit状态的范围、失败位和代际语义必须保留，不能截断成32-bit。

AICore以32-bit ld_dev/st_dev访问通知，资源地址决定SET或BIT_WR操作；
AICPU在启动阶段通过映射地址写值并在READY之前DSB。通知地址不做DCCI，
同核存在数据发布顺序依赖时保留相应DSB，跨核通知本身不机械新增屏障。

### 2026-11-17：CntNotify ADD 偏移访问实测依据

`fwic-dsl` 已在 Ascend950DT_9581、CANN 9.3.0、driver 25.6.rc2.b023 上实测：持有 SDK 返回的 SET/read 映射，按 `slot = SET_address & ~0x1f`、`ADD_address = slot + 0x08` 得到同一通知对象的 ADD 地址，以 32-bit `st_dev` 写入。正式 ADD 地址接口仍返回 507899；该错误不表示硬件 ADD 不可用，也不需要更新驱动后才能使用本次已验证的偏移方式。

| 验证 | 结果 |
| ---- | ---- |
| 单核初值7，ADD写1一次 | 8 |
| 96核各写1一次，64轮 | 每轮96 |
| 96核各写1八次，64轮 | 每轮768 |
| 换第二、第三个对象各测64轮 | 每轮96，其他对象哨兵不变 |

自动设备5，任务 `task_20261117_163922_212564423280`；使用硬件 SyncAll 对齐。256个满核轮次均有全核写入区间重叠，共67,585次ADD写1、24,672条逐核记录，无丢计数或邻居对象变化。此探针每次写后执行DSB，只证明操作数1；恢复调度器的合并ADD(2)、MIX聚合ADD(2/3)及批量尾屏障之前需补对应实测，不能从本结果直接外推。

SET映射与handle覆盖整个运行生命周期；派生ADD地址不独立map/unmap。SDK报告的4字节是该资源接口返回长度，本次偏移可访问性由上述特定环境实测支撑，不扩写为其他芯片的公开SDK保证。原始源码/记录位于工作区 `cnt-notify-add-offset-20261117/`，本仓证据副本为 `artifacts/cnt-add-enable-20261117/reference-*`。恢复先前的单调计数链，继续保留当前MIX自身ACK省略、已有证明复用，以及GM回退路径的单次async标记；不恢复bitmap或run内bank交接。

补充同构实测（`artifacts/cnt-add-enable-20261117/probe/`）：向偏移地址写2保持初值不变，写3一次只触发加一，不能解释为通用delta加法。固定命令值1连续写8次、每核批尾一次DSB，96核64轮全部为768，邻居哨兵正常。只重编Host增加的无溢出高位对照也通过：`0x7ffffff0 + 96 = 0x80000050`、`0xffffff00 + 96 = 0xffffff60`。安装目录名为cann-9.3.0，但补充探针读取的CANN/runtime版本头为9.2.0，版本信息按原始记录保留，不仅根据目录命名判断。以上是功能结果，不是调度性能对比。

### 当前 ADD 完成链实现

Host 规划按拓扑秩从近到远遍历真实前驱，遇到最近且容量足够的当前链尾即复用；零贡献控制节点仍可作为祖先路径。辅助存储为 O(V)，不构造逐任务祖先矩阵，也不扫描与当前任务无关的链。独立任务直接新建链；复杂图的搜索成本仍取决于实际访问的祖先和边，不宣称所有 DAG 都是线性时间。

- Host 将严格真实DAG祖先链映射到同一单调计数器，每任务保存固定累计阈值；每execution具有P物理贡献，只有可能异步的lane再具有Q条件贡献。Host用64-bit累计，超过UINT32_MAX前拆链。独立任务不共享链，也不添加执行依赖。
- 已知同步普通返回贡献1；可能异步的普通执行在本次零登记时贡献2，实际发送两次32-bit写1；有defer时P、Q各发送一次写1。MIX可能异步lane在无条件封口或最终CHECK时贡献Q，coordinator在完整资源release后按active lane数贡献P。不能把提前准备、发票结束或仅物理返回当作仍有条件的任务逻辑完成。
- `scheduler_notify_add(credits)` 是逻辑批量贡献接口，实际只写已验证的命令1。数据发布前的DSB、消费者数据观察、kernel入口与退出的既有屏障保留；通知自身不机械增加逐次或尾DSB。读到完整阈值证明贡献已到达；探针批尾DSB的结果与此协议论证分开，实际无额外通知尾DSB的整网路径另行实测。
- 后继直接读计数器并比较自己的阈值；计数器在一轮中只增不减，旧阈值永久有效，无bitmap、旧快照、run内reset或bank转交。每轮READY前仅由AICPU清零链头一次，publication/capture lease继续保护生命周期。
- canonical GM账本按task一次有界归档，不阻塞fanin ready、固定owner或资源退休。固定归档者可复用本核当前task的真实DONE证明，避免再次读取同一计数器；未命中仍观察阈值，SUPPLY_CLOSED不能代替DONE，error检查和唯一结账/退出统计不变。DFX记录真实P/Q贡献区间、累计阈值和退出计数；不伪造FAA返回ticket，也不以归档时刻代替逻辑ready。
- 由持有的SET映射校验32B slot并派生+0x08地址，派生地址不独立map/unmap；容量不足的整链仍用GM路径。GM回退保留近期的自身ACK删除、已有观察证明复用和首次登记者单has_async。Host覆盖日志使用benchmark可见的TIMING级，记录`backend=SET-slot-offset`及计划/实际链数、eligible/mapped任务数。
- 正式验收要求实际非零ADD覆盖，记录完整与部分覆盖；GM回退成功不算ADD通过。继续以12组AICore L0配对总和和0.5%回退门槛判断性能，不从硬件探针直接推导整网收益。

本轮实际ADD启用验证：12组均确认eligible计算任务完整映射；11组数值通过，原生MoE沿用仅计时。AICore L0配对总和20.2296068→20.2704963 ms（+0.202%），符合0.5%保留门槛；SWA为0.2577120→0.2486034 ms（−3.534%），新版第19–21层为3.0837522→3.1455712 ms（+2.005%）。该结果不表示各case均加速。完整样本、实际设备、覆盖证据保存在 `artifacts/cnt-add-enable-20261117/RESULTS.md` 与JSON中。

<!-- markdownlint-disable-next-line MD033 -->
<a id="performance-retention-policy"></a>

### 性能与保留规则

每项修改先对应到实测泳道图中的具体串行链，再查 `atomic_probe/ATOMIC_USAGE_GUIDE.md` 与同构探针的硬件证据。明确可减少的共享访存、等待或重复准备及保留的交权边界；不按函数 inclusive 时间直接估算可省开销，也不凭模型名称试参数。

- 当前优化目标先减少同地址 atomic 请求，再减少 atomic 总调用及重复 ld_dev。每项策略比较同一组输入下12个 AICore L0 均值的总和及逐项变化：功能正确、明确消减冗余 atomic／ld_dev 且总和回退不超过0.5%时可保留，回退超过0.5%则撤回。计数减少和耗时收益分别提供证据，不能将新增 DCCI/DSB、普通读写或其他地址竞争隐去。percounter 作为定位参照，比较结论须由可比版本、输入及计时边界的实际对照支持。5 µs 交接用于分析关键路径，不能替代12组指标。
- 最终性能以 AICore L0 `chip.run.runner_run.aicore_wall` 为准：在无 L1/L3 插桩的运行中，直接取所有实际 AIC/AIV kernel 的最晚 end 减最早 begin，包含首尾之间的调度空洞、依赖和通信等待，排除头尾 AICPU 开销。每组 1 次预热、10 次计时；无 kernel 的核不参与归并，缺失或无效时间戳不能用 Device wall 代替。HBG、基线及 percounter 必须核实同一首尾边界；percounter 原调度循环窗口不得直接冒充 kernel 首尾窗口。L1/L3 只用于定位。
- 双卡用例每轮先做跨卡 Device 同步，再开始上述 L0 计时。测试启动门必须位于 AICPU 计时入口之前：两卡完成本轮入队后才允许设备会合，随后执行原有 AICPU/AICore 流程；同步门自身等待不计入 AICore L0；只有首末实际 kernel 之间的计算、调度空洞及通信等待计入，AICPU 启停与首末 kernel 之外的准备、退出不计入主指标。单卡不加门。每轮取两个 rank 的本卡耗时最大值，再对十轮求均值；不得跨卡相减硬件计数器或从旧样本减去估计的启动差。基线、候选与 percounter 均须使用同一门；旧的未同步双卡数据另标口径，不能直接作为新口径基线。
- 性能验收使用 `dsv41_scheduler` 当前清单的 12 组配对对照，保持各组原 kernel、输入、图、sync_start 配置一致。11组回放须通过数值验证；新版原生 MoE 沿用已约定的72/72性能专用模式，明确标记 `numerical_validation:false`，不能把执行成功当作数值通过；全部 12 组（含 MoE 与 layers_19_21 通信场景）的 AICore L0 均值之和决定是否保留，同时披露逐项回退；Graph、MIX、sync_start、defer_wait 的通用功能另按上述合同验证。单个 SWA 或其他模型的收益不能替代全套结果。
- 所有硬件任务使用 `task-submit --device auto --device-num N`，在任务内使用并记录实际 `TASK_DEVICE`；attention 单卡，同一分配内做配对对照，记录先后顺序。
- L1/L3 也区分预热与正式采集；不从单条交接里扣除整次 L1−L0，不累加跨核或嵌套 span 冒充串行耗时。5 µs 判断同时报告低扰动观测、采集开销和未达标原因。
- 对照完整目标方案前后，并保留各独立改动的归因。常规性能改动以综合收益决定保留，1% 收益也记录；已证明消减冗余 atomic／ld_dev 的结构性修改适用上述不超过0.5%回退的保留例外，超过0.5%则撤回；回退如实记录。无 ±5% 忽略区间。有收益的独立修改分别本地 commit，未经允许不 push。
- Host 准备、实际 H2D、GM/SSBUF 和峰值内存单列；不能把工作搬出 L0 区间后不报告。历史测量与详细采集留在实验结果中，不回填本设计作为实施完成状态。

验收需要同时证明：具体 pending 确实提前持有、已有工作能直接接续、SPMD 分布没有以尾波换局部收益、协作与异步语义完整，以及同口径 AICore L0 有实际收益。四个 token、一次 CAS 或某个局部 span 变短，都不能单独替代这些结果。

12 组中的非通信清单：初版、DSL 半同步、DSL 全同步的 `c2a_full` 三组，加新版 PyPTO-lib 的 `swa`、`c2a_full`、`c1a_reindex`、`c1a_reuse`、`engram` 五组。各组均 1 次预热、10 次正式测量。另外 DSL 半同步与新版 PyPTO-lib 各自的 MoE 与 `layers_19_21` 共四组通信配置同样计入保留指标，合计12组。
