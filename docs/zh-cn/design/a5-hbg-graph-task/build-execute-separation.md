# HBG 调度设计：提前准备、最小领取与轻量等待

全局自取与簇内 MIX 共用执行入口。每核从原自取记录或本 lane 的 MIX 记录执行；每簇 AIV0 发布 MIX 描述符，Host 计算的 rank 和每 lane 阈值约束互相冲突的资源承诺。任务形状和启动检查见[协作任务的 Host 准备与 Device 校验](admission-preparation.md)。

在 A5 HBG 的 L3 图上，普通自取路径依次可见查目录、领执行票、本核准备、fanin 检查和执行。MIX 的 AIV0 在本簇有空闲 SSBUF 槽位时领取一个 block 的执行票，随即发布本代描述符。活动成员按各自的 lane 接收 Host 预建的 Call。泳道图中 `CLAIM_EXECUTE` 表示领执行票，`DISPATCH · notify_mix_members` 表示簇内通知。执行许可、对后继发布完成贡献和本核退役分别计时。各阶段和短调用的显示规则见[芯片泳道图说明](../../../dfx/chip-swimlane-profiling.md#a5-hbg-scalar-detail-at-level-3)。

设计参考 `atomic_probe/pa_scheduler/cross_core_only_scheduler` 的有序领取与固定归属。Host 根据现有数据和 Graph 控制依赖生成稳定拓扑序，提前构造不可变 Call。Device 按序分发，接收本执行者的数据并准备，条件满足后原地执行。性能裁决沿用[7 组 AICore L0 口径](#performance-retention-policy)。

修改范围为 A5 HBG 调度器、必要的 Host 布局、内部协议与 DFX；不新增 PyPTO 通知接口，不修改用户 kernel、前端语义或参数 ABI。

调度优化必须由通用任务状态、依赖和资源合同驱动，不按模型名、特定 task 编号、shape 或固定图拓扑选择策略。SWA 等单例的泳道图用于提出和验证机制假设，不能替代通用性验收。

**当前保留标准（2026-11-20 更新）：** CntNotify 扩展逐项对比上一保留版本的 7 组 AICore L0 均值总和；总和回退超过0.5%即撤回，不超过0.5%可以保留，并记录每组变化。此前 MIX 握手的 +0.224% 是历史 12 组结果。

性能主指标统一为 HBG 与 percounter 的 **AICore L0 wall**：各实际执行核的最晚 kernel end 减最早 kernel begin，包含中间调度和依赖等待，排除首末 kernel 外的 AICPU 启停开销。每段预热后计时 5 轮，每个用例中两个版本各取 10 轮；双卡每轮先 Device 会合，各卡按自己的时钟计算，再逐轮取 rank 最大值并求均值。缺失或非法端点直接判无效，不回退到 Device wall 或旧 percounter 调度循环窗口；历史数据保留原口径，不重新贴标。

<!-- markdownlint-disable-next-line MD033 -->
<a id="a5-hardware-principles"></a>

## A5 硬件与调度统一原则

**调度器负责分配任务，以及通知执行者任务已到达、通知后继依赖已满足。调度状态按实际参与范围选择存放位置。**

- 能在本核完成的状态转换，所需留存的状态优先放在 SSBUF。对于在同一簇内就能结束的 MIX 协作，AIV0 在本簇 SSBUF 发布分发信息，各参与成员只写自己的 FIN 字段，AIV0 复用槽位时检查这些字段是否对应本代。
- 需要让其他核观察的任务到达、依赖就绪和完成进度，优先通过 CntNotify 传递。例如每个完成的 block 向计数器写一次 INC，后继直接观察任务阈值。有限的通知槽可沿有依赖的任务链复用，同时在途的独立任务须避免槽位冲突。
- 调度热路径原则上只在取得 task 本身及其参数时读取 GM，例如读取 Host 预建的 Call、tensor 地址和 scalar。已由 SSBUF 或 CntNotify 表达的状态，避免再写入 GM 供其他核轮询或二次确认。动态领取需要由返回值唯一确定 block 归属时，保留用于这次裁决的 GM atomic。

以下七条采用 **2026-11-16 用户给定的 A5 硬件工作模型**，作为本设计及其他 A5 调度设计的共同前提。100 ns+、约四拍是该前提给出的量级，不是本次新增实测或跨芯片、跨负载的固定延迟保证。历史探针保留原始条件与结果；其旧工程建议若与本节冲突，后续设计按本节修订。当前实现与拟修改方案仍须分别标明。

| 编号 | 硬件前提 | 对设计的直接约束 |
| ---- | -------- | ---------------- |
| 1 | `ld_dev/st_dev` 绕过 Scalar DCache、进入 L2 cache；atomic 不进入 cache，走 HBM 原子路径，单次并发增加 100 ns+ 量级开销 | Wait 用可靠 `ld_dev` 观察；单写者通知可用 `st_dev`；不能用 FAA(0)给同一读值再做一次确认。atomic 留给真正需要的竞争裁定与多写者合并 |
| 2 | SoC 总线对 `ld/st_dev` 和 atomic 均不保序 | 同核操作间有时序依赖，就在相应依赖边界显式加 DSB；成功 CAS/FAA、源码先后和 C++ `memory_order` 名字都不能代替硬件屏障 |
| 3 | STARS 有 event/notify SRAM 资源，不受 GM 同址读写的并发串行瓶颈限制 | 逻辑 DONE 已采用 CntNotify SET/load；其他 Wait 后端按资源合同推进，它不自动授予 task 的唯一执行权 |
| 4 | A5 上基于 GM 的同地址读取、写入都有并发串行代价 | 既要减少同址 atomic，也要控制同址 `ld_dev/st_dev` 的扇入和请求密度；不能把 atomic 改成多核无限紧轮询后就宣称争用消失 |
| 5 | 动态领取 task 仍须由 atomic claim 裁定；对保留下来的竞争性 claim，分地址 atomic 是消除同址争用的方案 | 优先降低同地址 atomic 并发，再减少总次数。独立竞争单元按 **128B L2 侧隔离粒度**布局，不能仅相隔 64B DCache 行；动态路径保留唯一裁决；下文定义的刚好一波由物理位置唯一确定 owner |
| 6 | STARS SRAM 并发排队约四拍，可近似忽略，但入口带宽有限 | STARS 等待循环须以 `nop` 限制空轮询请求速率，避免无限紧循环耗尽入口；同时保留有界控制服务和实际唤醒延迟观测 |
| 7 | AICore SPR 等寄存器即使可映射访问，也缺少通用 event 的分配、释放方案 | 在资源分配、归属、复用与释放机制完成前，不把 SPR 当作通用 task event 池；既有固定用途寄存器协议不等于具备该能力 |

第 5 条的“唯一方案”针对 **必须保留的动态 claim 的同地址竞争**：继续优化它需要分散地址，不能用普通读写取代唯一裁决。删除冗余操作、提前 Build、缩短入口和改进 Wait 仍然有价值，但不提供另一种动态所有权原语。分地址也不能只是换几个变量名：需明确合法票区间、不重复不遗漏、同一 task 全部票发完的证明、跨地址发行顺序及 owner 固定不变，且不能在每次分散 claim 后再访问一个全局共享队头或汇总原子，把热点重新引入。

### A5 双计算 die 的局部性与满波 MIX 放置

在 32 簇的 Ascend950DT_9581 上，握手记录中 C0–15 和 C16–31 分属两个计算 die，每簇有一个 AIC 和两个 AIV。Host 规划固定归属的满波 MIX 时，运行时按簇号映射逻辑 block。簇数 `K` 为不小于 8 的 2 的幂、任务有 AIC 和至少一个 AIV 时，映射为 `(c & 2) ? (c ^ (K / 2)) : c`。该置换保留低位分组，每个 block 仍有唯一的簇负责。其余形状按原簇号分配。[映射实现](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h)

代理发布、成员领取和尾段 Call 的提前绑定使用同一映射，依赖、入场和完成仍按逻辑 block 记账。这样可把相邻分块分摊到两个 die。其他 SKU 的逻辑簇与物理 die 对应关系应从 resident 握手记录确认。

### 普通 SPMD 的分区领取

普通非 sync_start SPMD（含非 sync_start MIX 的 group 领取）使用四个独立的 128B 计数器，分别分配 `part + 4*k` 的 block。空闲核优先访问自己的分区，耗尽后检查其他分区；执行 owner 仍由成功的原子领取确定，不做 Host 固定分配、不转移已领取责任。同一核同一 task 最多持有一个未完成 block 的约束不变。

每个非空分区由最后一个有效领取者报告关闭。已映射 CntNotify 时，领取者在绑定自己的执行记录或发布 MIX 的 SSBUF 描述符后置位，执行 DSB，再读取本任务的位掩码。普通非同步任务的分区报告者看到掩码全齐后，可将 `next_block` 和 GM `claim_seal` 依次写为 N。多个报告者写入相同终值。MIX 等协同路径仍通过 `CAS(next_block, 0, N)` 选出唯一发布者。

未映射时，GM 路径在领取末票时将该分区容量累加到 `next_block`。使总量达到 N 的领取者在绑定执行记录或发布 MIX 描述符后发布同一 `claim_seal`。两条路径都保证全部 block 已有 owner 才宣布供给关闭。分区越界领取不回退，也不产生执行权。单 block 任务保留单计数器路径。

每个 task 增加四个 128B 分区；每个多 block task 额外最多四次分区关闭汇聚。必须计入这项空间和原子开销，以全套 7 组 AICore L0 总和验证同址争用降低是否覆盖成本；不能从 L3 的跨核累计耗时直接推算 L0 收益。

### 三阶段怎样应用这些前提

- **Build：尽可能并发、尽可能提前。** owner 无关的地址、参数与静态校验提前完成，owner 相关准备在归属确定后推进；批量数据写入和缓存维护在其发布边界完成 DSB，再发布可观察通知。不可变 Call 的准备不包含尚未生成的业务 Tensor 数据。
- **领取：atomic 只裁定一次必要的唯一责任。** 同一有效票已授予的普通执行权不再二次 CAS；仍须竞争的 claim 按 task/独立票区间分地址并以 128B 隔离。STARS event、可靠 `ld_dev` 和目录 hint 都不能取代该 atomic 裁决。
- **Wait：观察已定位条件，不重新 Build 或 Claim。** GM 路径用 `ld_dev`，单写者通知根据协议用 `st_dev`；观察值携带 task/epoch/sequence 身份传给下层复用。满足条件后按消费者获取合同执行 DSB 与必要缓存观察，MIX、sync_start 和退出许可仍独立检查。逻辑 DONE 已映射到 CntNotify SRAM，其他条件通知继续按各自协议处理。

DSB 只针对同核前后操作的实际顺序依赖；跨核通知本身不要求发送后追加 DSB，也不等待接收方何时看到。单写者也不自动等于可以用 st_dev：需要 RMW、原子交权或不可分割联合状态发布的字段仍保留 atomic。DSB 不是跨核会合，也不使另一个核的 DCache 自动更新。发布数据→发布状态、读到状态→使用数据、依赖前次返回值的后续 atomic，都须写清排序边界；可以合并同一批无相互依赖操作后的屏障，不能把它移到已有中途消费者的最后一次运行结束。普通 cached 数据与 atomic-only 控制区继续隔离，避免 DCCI 把旧脏数据覆盖到控制字。

### STARS 与 SPR 的演进边界

当前逻辑 DONE 优先使用沿真实祖先链共享的 CntNotify 累计阈值；A5 上板的设备控制节点使用每节点独占 SET 通知，CPU/sim 可观察 canonical GM gate。sync_start MIX 使用簇内 SSBUF 传递 assigned/arrived/released 握手。CntNotify 随运行镜像持有并在每次 READY 之前复位；其他 STARS 接入仍需绑定已有 task/epoch/sequence，明确 event 申请、初始化、先通知后等待的处理、完成后清理、复用及异常清退；不能漏通知、把上一轮通知认作本轮完成，或在旧观察者仍访问时复用资源。通知与 payload 可见性仍按各后端合同排序，SRAM 低排队代价不免除这些责任。

STARS 的 `nop` 是入口限流，不是让有工作可做的 Scalar 无条件停顿：空轮询受控，已有可执行任务、合法 Build、defer_wait 服务和 EXIT 检查继续获得进展。具体间隔由入口请求率与发现延迟决定，不预置模型专用常数。SPR 在具备上述资源管理机制以前不作为该方案的替代品；A3 上的寄存器可达性实验也不能直接作为 A5 映射可用性的证明。

## 1. 执行循环

### Build、领取、Wait 的统一原则

这里的 task 原则覆盖每个执行单元实际调用前所需的工作，不把函数名或现有状态枚举当成阶段定义。普通任务领取的是具体 block/lane；MIX 领取的是逻辑 block 的组责任，sync_start 另有成员的实际承诺与退休。

| 原则 | 应实现的行为 | 必须一起衡量的代价 |
| ---- | ------------ | ------------------ |
| Build：并发、提前 | 将地址解析、参数组织、metadata、固定 context、所需接收及发布，放到各自最早合法时机；owner 无关部分由 Host 或合法准备者提前完成，owner 相关部分在固定归属后、依赖等待期间完成 | 准备结果真正早于前驱完成的覆盖率、还留在启动入口的工作、准备但未使用的包、Host/H2D/GM/本核存储成本 |
| 领取：慎用 atomic | 动态唯一 owner 由 atomic claim 裁定；必要 claim 按 128B 分地址降低同址竞争，多写者合并保留 RMW，普通有效 FAA 票已经授予的责任不再竞争第二次 | 同地址并发请求优先于总次数；分别统计有效票、分区超额票、单 block 超额票退款、协作准入和完成贡献；提前固定 owner 可能造成的等待 |
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

Wait 中“地址已知”也不等于所有 task 共享一个全局地址：当前每个前驱有 canonical completion 和可用时的 CntNotify 通知。GM 回退或其他共享条件的多个消费者仍可能竞争同址；SRAM 也有入口带宽限制。可靠读取不代表无限带宽；更短的单轮检查可能增加单位时间的请求量。负向观察不能作为永久证明，降低查询频率也不能连带降低实际 kernel 入口的 EXIT/错误检查。

完成发布是三条原则的闭环。执行资格要求已有 owner、固定准备完成、依赖逻辑完成、本核可执行、协作许可有效且无终止条件同时成立；之后还有不可提前的入口操作。前驱已返回但完成贡献尚未发布，或 defer_wait 债务尚未结清时，后继仍不可执行。因此不能仅凭这三个局部阶段变短就宣称达到整图最优或 5 µs 目标。

### 目标的准确含义

有已发布且适合本核类型的工作、本核有补给机会时，提前持有具体待执行单元。前一 kernel 返回并完成必要的结果发布后，若已有 pending 的全部执行条件满足，就直接进入下一 kernel，不再先做一轮全局发现、重新分配 block、构造参数和 READY slot 发布。

这里的 pending 必须含具体 `node_ref/block/lane`、唯一执行票和所有权。只保存 task ID、fanin view 或一个“可提前准备”的 hint，不满足要求。普通 token 进入 `WAIT_FANIN` 时已校验 Call，接收 Binding 并保存固定 metadata。依赖满足后，owner 设置 sub_block_id 并发布 GlobalContext，然后检查最终执行许可。只有票号、尚未完成接收和固定准备的 `TICKETED` 不能计为同一种准备程度；MIX 的预校验、正式 lane 接收与到齐许可也不能混称为一次 Build 完成。

pending 可以等待尚未发布的元数据、真实计算依赖、本组 MIX 伙伴或较早的协作资源承诺。目标协议中，本组许可已成立且其余成员的进展有保证时，本核可以先进入 `sync_start` kernel。没有可补给单元时，单独记录发现和等待耗时。

分别测量两个间隙：

- `G_worker`：同一核下一次 kernel 开始减前一次 kernel 结束，回答本核能否快速接续。
- `G_dep`：后继首个 kernel 开始减真正最后一个计算前驱的最后 kernel 结束，回答跨核依赖交接是否接近 5 µs。异步场景另记录最后逻辑条件满足到后继开始，不能把物理返回当作逻辑完成。

同时记录同一 SPMD 任务的全部 block 起跑跨度、实际参与核数、每核执行次数和尾部。仅首个 block 早启动、其余 block 多执行一轮，不算达到目标。5 µs 是拟达到的交接预算，整体收益仍由完整 L0 实测判断。

<!-- markdownlint-disable-next-line MD033 -->
<a id="a5-slot-sources"></a>

### 两种 slot 来源与同一个执行入口

| 来源 | 谁取得责任，记录存在哪里 | 执行和归还 |
| ---- | ------------------------ | ---------- |
| 全局自取 | 符合类型的空闲核从目录为普通或纯核 `sync_start` 任务动态领取 block。核在自己的 W 个位置之一保存任务实例、block/lane、prepared handle、已确认 fanin 和领取身份 | 同一位置从领取一直保留到物理执行、必要的完成贡献和本地退休。选中后传引用给执行器 |
| MIX 分发 | 每簇 AIV0 为整个逻辑 block 领取一次组责任。实际成员从本簇 SSBUF 获得同一代分发，在原组记录及本 lane 的接收状态下准备和等待 | 成员按组身份直接进入同一个执行器。每个实际 lane 只归还自己的执行责任，组按掩码归并 |

一次选择只产生短期引用 `(来源, 原位置, 身份)`，运行中以 `(run_epoch, 实例 node_ref, block, lane)` 标识真实 execution，MIX 再带 cluster 和 generation。选择和 kernel 入口在同一次推进中完成。执行器按来源引用直接访问原记录。许可暂未满足时，原记录和 owner 保持有效，下个有界机会继续。predicate、业务输入获取、动态 context、错误检查和最终许可仍在各自有效边界执行。

W 表示每核可持有的全局自取且未退休的 block 数。当前为 **W=1**。比较窗口深度时使用同一执行路径。正在执行的自取 block 占其中一份，来自 MIX 的 block 使用自己的组记录。同一核同时只执行一个 kernel。AIV0 为 MIX 领取的是组责任，即使它也是计算成员，同一 block 也只走 MIX 来源一次。SSBUF 描述符、prepared Call/Binding、DFX 轨迹和异步 CHECK 记录各有自己的寿命，不计入 W。结构验证后只改变 **W** 做容量对照。本核持有 A 等待前驱 X 时，若独立的 B 已允许分发，深度至少为 2 时可提前领取 B。W=3 和 W=4 还分别能保留一份、两份已允许分发的 block。W=1 须等 A 退休才能自取 B。各深度均保持固定 owner。

```text
每核循环
    完成上一单元必要的归还，处理错误、退出和到期的 CHECK / Graph 服务
    推进已承诺的组，观察本核自取记录及 SSBUF 中属于本 lane 的记录
    有合法候选 → 读取原记录，按共同 rank 检查资源承诺，直接执行及归还
    暂不能执行 → 保留原位置，服务能解除阻塞的伙伴、控制节点或前驱
    自取位置有空且前序分发允许 → 从全局目录领一块
    本簇组记录可用且 MIX 分发允许 → AIV0 代领并发布，自取窗口满时仍能推进
```

已满足条件的候选优先于补给。新分发、收到一代发布、前驱完成或资源释放后，只重访受影响的候选。控制服务保持独立预算。所有自取位置都在等待时，成员仍可通过 SSBUF 处理早序 MIX，登记核仍能检查自己挂起的条件组。如果占据最后一个自取位置会妨碍更早纯核 cohort 分发，就要在取得该位置前落实顺序与容量约束。设计不通过转移已领取 block 解围。

### probe 实际提供了什么

参考目录为工作区 `simpler-fdwic-swimlane-deps/tests/atomic_probe/pa_scheduler/cross_core_only_scheduler/`；硬件合同参考同仓 `tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`。HBG 不移植 ordinary probe 的 Device Build、TensorMap 插入、Submit 或任务生成。

| probe 机制 | 实际入口 | 对 HBG 的含义 |
| ---------- | -------- | ------------- |
| Build 使用共享递增票号 | `pa_scheduler_core.h::TakeSharedBuildTicket` | 借鉴确定的任务顺序；HBG 已在 Host 构造，不复制 Device Build 任务分发 |
| Execute 从同类共享游标领取连续 ordinal | `pa_scheduler_core.h::TakeSharedExecTicket` | 所有同类领取者共享供给进度，不能各自选择目录起点 |
| Host 以最终设备地址预绑定不可变参数/context | `common/prebuilt_dispatch_host.h::PrepareStaticExecutionBindings` | 保留并完善现有 Host Call，不再用 Device 重建已知常量 |
| 每核 4 个真实 token，每批 2 个 Execute ordinal | `shared_exec_protocol.h::kExecTokensPerWorker/kExecTicketBatchSize`；`pa_scheduler_core.h::TakeSharedExecTicket` | 借鉴有界前视；不机械照搬批量 2，避免宽 SPMD 囤票 |
| 先绑定具体 task，再领取共享 cell | `BindSharedExecTicketToToken`、`ProgressCrossCoreActiveToken` | 将具体执行票的发放前移到 fanin 就绪之前 |
| 已知 BUILT 字直接做一次 Claim CAS | `pa_exec_adapter.h::PrebuiltPaExecBinding` | 初次领取使用 Host 发布的通用初态，消除 BUILT/PREPARED 双重尝试 |
| Claim 后接收参数，未 ready 时保留 `WaitingFanin` | `shared_exec_protocol.h::ClaimAndBindPreparedExecPayload` | pending 持有具体执行责任，不再只是 task/fanin 观察 |
| 检查剩余 fanin 并在同一 helper 转入执行 | `TryMarkExecutionTokenEngineInflight` | ready 后不再领取、重复确认同一批依赖或重新发布本核 READY |
| 先推进已有 token，有空位再取票 | `ProgressCrossCoreOwnedTokens/ProgressCrossCoreExec` | 将 pending 快入口放在发现/补给之前 |

probe 有 AIC、AIV 两个集中 Execute 游标；其 FAA 保证各引擎内 ordinal 发放有序，不保证跨引擎的物理调用或完成时间也有序。`common/host_prepare.cpp::PrepareExecutionImage` 先在 Host 构造镜像，`common/only_scheduler_host.h` 明确检查 Device Build 票数为零。HBG 借鉴有序任务分发和固定 owner，另补通用 DAG 的跨引擎前驱检查及 SPMD block 规则，不能把 probe 特定图的验证外推到所有图。

每批 FAA 与每 task Claim CAS 是不同操作，“单 CAS”只描述 cell Claim，不包含取票、依赖和完成原子。其固定 PA 控制字推导依赖特定任务形状，HBG 必须由通用 Host 计划提供初态，不能复制 `task_id % 5` 等模型规则。

probe 先把一批 ordinal 绑定为 `WaitingBuilt`，再逐槽 Claim、acquire、检查依赖和同步执行；不是先把四槽全部准备好才执行。如果第一槽已经 ready，它可以先执行，第二槽此时仍未完成 Claim。因此不能把“批量领了两张票”直接当作“下一个 kernel 已准备好”。已经停在 `WaitingFanin` 的下一项才具有无新票、无新 Claim 的快接续路径。

同 Scalar 执行同步 kernel 时不补 token，也没有别核写它的私有 token；前视来自此前的调度机会。本文保留这种机会性前视，不承诺强制满窗口，也不为补满窗口延迟已有 ready 工作。HBG 的不可变构造由 Host 完成，其他空闲 Scalar 可以推进依赖、控制服务和自己的 token；接收核自己的 DCache 观察和动态许可仍由接收核完成。

### 控制节点的本核前驱检查进度

普通 HBG 镜像在 Host 完成原图排序后，可折叠无计算的 inline 汇合节点：只有零或一个活跃消费者、且合并后的消费者 fanin 不超过现有128项上限时才折叠。消费者继承被折叠节点的全部前驱并去重；物理 kernel、MIX、sync_start、真实 defer_wait 任务都保留。多个消费者共享的汇合点保留，以免复制依赖检查；native Graph 的 ENTER/EXIT/CONTROL_DUMMY 继续遵循其既有实例与控制合同。

折叠先在 Host 暂存图上计算，再将全部存活 fanin 行重排到原 fanin 池；只处理零/单消费者保证边数不增加，不增长 Device 存储，不覆盖 tensor/scalar 区域。若原图共享 fanin 存储而独立重排无法容纳，则保留该镜像。只修改未发布的 resident 副本，Host Tensor owner/last_consumer 与原分配计划保留。原物理目录、rank 和拓扑层号保留，新增直连边必须仍满足前驱先于消费者。被折叠节点按 Host 完成节点记账，更新 control_count 与 inline_completed_count；消费者已接回真实前驱，因此真实 async 债务仍阻止后继执行。末端无消费者的汇合点可折叠，但整次运行仍等待全部真实任务完成。

未折叠的 inline/native control 沿原有目录与每轮8项预算轮转。每核使用16项控制节点专用私有缓存，保存 task_ref 对应的已确认 DONE 的 fanin 前缀，避免多个未完成 control 轮转时丢掉彼此进度。槽位内完整身份必须匹配；hash 碰撞只遗忘旧证明，不能将旧前缀授予新节点。零进度不驱逐其他节点的有效前缀，扫描普通节点或跳过已完成 control 不清空缓存。新 run 清零全部条目；缓存遗忘只造成重新确认，数据依赖、Graph 激活和错误检查沿用既有约束。

具有独立 SRAM DONE 通知的控制节点确认完整 fanin 和合法的 GM 初态后，经过发布屏障并幂等地 SET(1)。等待者、设备终点检查与超时进度检测都可以通过这份通知观察完成。普通运行由 SET 完成记账。缺少通知资源的控制节点沿用 GM CAS。原生协议采集另以 CAS 记录唯一记账者，DFX 的 `CONTROL_NOTIFY_ACCOUNTING` 时间戳对应这次记账；多个核可能先后发布同一个 DONE。

合法图的 task_count 不超过 INT32_MAX，fanin count 不超过32位字段范围，因此每项用64位编码 task_ref+1 与 next_fanin，零值表示空条目。缓存共128B/核，仅在本核使用，不新增共享通知或原子操作。

等待中的 control 不绑定服务核，也不阻止扫描其他 control；不新增共享选主、通知副本或 ready 队列。前缀只保存当前完成来源的有效 DONE 证明；可靠 `ld_dev` 读到合法计数阈值、SET=1 或 GM `C=D=0` 可建立该证明，PENDING、供给关闭和参数就绪均不能代替它。操作减少与 7 组 L0 门禁分别验证。

## 2. 从 task 有序开始：供给、预绑定、等待与退休

### 先区分构造、任务分发与完成

| 状态 | 含义 | 不代表什么 |
| ---- | ---- | ---------- |
| BUILT | Host 已发布该实例的完整不可变 Call | 没有指定执行核，也不代表依赖满足 |
| ISSUED_ALL | task 的全部有效 block 已取得不可撤销的执行责任 | 不要求这些 block 已开始或完成计算 |
| DONE | 计算贡献、合法 skip 和异步条件按原协议全部结清 | 只能由权威完成协议确认，不能由任务分发游标推断 |

`ISSUED_ALL` 是 task 的分发关闭证明。对动态分发，单 block 任务由第一次有效领取确定关闭者。普通静态 DAG 中可映射的动态多 block 任务由各非空分区的最后领取者，在本核绑定或本簇 SSBUF 发布后置位 CntNotify；写后 DSB 再读本任务掩码。普通非同步任务允许读到全齐的报告者重复发布相同终值。MIX 等协同路径仍由一次成功的 `CAS(next_block, 0, N)` 选出唯一发布者。未映射的任务沿用 GM 分区容量累加。两条路径的目录观察者都读取原有 GM `claim_seal`。`sync_start` 只计入真实有效领取，越界尝试不能推动 seal。MIX 一次领取固定一整个逻辑 block 的组责任。无论哪条路径，seal 都不证明各伙伴已经看到描述符或进入 kernel；本簇发布可见性由第 4 节的 SSBUF 边界证明。

普通 token 的最后有效票在唯一 Claim 成功、handle 已记录固定 owner 后发布 seal，再接收与校验不可变 Call/Binding。此时本核的 token 仍是 TICKETED，Claim handle 已登记，接收成功才进入 WAIT_FANIN；其他核可以领取并准备后项，执行仍等待前驱 canonical DONE。接收失败须记录错误并撤销未启动的本地执行责任，不能完成前驱或放行依赖消费者。Claim/Receive 分开计时；需要完整接收的 MIX/cohort 入口仍组合调用两步。普通 MIX 与 sync_start 保留各自真实责任登记的发布点。

本核在当前目录项已确认领取耗尽后，继续服务原有责任。耗尽观察只排除再次竞争这项分发；只有完整分发关闭或合法终结才能推进目录。切换项或 run 时清空任务专属的观察。

纯核 `sync_start` 在实体位置上取得有效票后，记录本核已领取的任务实例。早返回的核也不能为该实例再领一块；剩余位置仍可完成分发。MIX 的 `sync_start` 按簇限制同一实例的领取，并由活动 lane 各自承诺和退休。

本方案保留 `ISSUED_ALL` 作为后项领取门槛：Host 先形成满足依赖约束的顺序，Device 按这个顺序把前项全部有效执行票分配出去，再开放后项。这里的“全局”是该 task 的所有 block 都已有人负责，不是只看本核已领到一块，也不是要求全机只有一个任务分发游标。对于普通依赖 `A → B`，A 全部完成任务分发后即可提前领取 B、接收参数；B 的 kernel 仍等待 A 的逻辑 DONE。因此有序任务分发与重叠准备、并行计算相容，不能把任务分发门槛误称为执行完成屏障。

### Host 拓扑分层与层内优先级

**保证前驱排在后继前面，必须在 Host 排序时完成，并在目录发布前逐边验证。** 选择 Kahn 式拓扑分层，优先展开同层的独立分支；层内再按剩余依赖链长度排序。参考 [Kahn 原论文](https://doi.org/10.1145/368996.369025) 与 [拓扑分层定义](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_generations.html)。

Host 根据已实例化的数据与控制依赖计算两个层号。`level` 是从源节点出发的最早拓扑层，`b_level` 是到终点的最长剩余依赖链。两者均按边数计算。

```text
level(v)   = 0                                      无前驱时
           = 1 + max(level(p), p 是 v 的前驱)          其他情况
b_level(v) = 0                                      无后继时
           = 1 + max(b_level(s), s 是 v 的后继)        其他情况
基础排序键  = (level 升序, b_level 降序, 原 node_ref 升序)
```

对于普通展开图，若 deferred 组数和 Device 控制节点数均为零，且图中没有 predicate，Host 可在基础顺序上局部调整。对每个宽 SPMD 任务 S，分别估算 AIC、AIV 的并发占用。MIX 先取 `min(block 数, 簇数)`，再乘每个 block 的相应计算 lane 数。32 簇上的 1C1V 即使有多波，也至多同时占用 32 个 AIV。纯 AIV 任务可以使用每簇两路 AIV，在同一设备上至多占用 64 个。纯任务和 MIX 均可作为 S，也可带 `sync_start`。每个窄任务 B 只考察其后在同一引擎上需求达到半数执行核的第一个 S。若先遇到另一个 MIX 或 `sync_start`，则停止。若 B 与最近的 S 能在该引擎同时放下，B 保持原位，也不会再为更远的宽任务让位。若多个 B 满足条件，S 选择在原顺序中离它最近的一个，把 B 移到 S 的紧后方。

B 必须是该引擎上的普通纯任务，无 `sync_start`，且需求小于容量的一半。只有 B 和 S 的需求之和超过该引擎容量、S 的 `b_level` 大于 B、两者的原始 `level` 相差至多一层，才允许移动。B 移动时不能越过自己的任何直接后继。每个 S 至多移动一个 B，每个 B 至多移动一次。Host 逐边验证前驱仍排在后继之前，再按新 rank 重建各 lane 的发放门槛。协作任务间保持原相对顺序。

Host 将 B 较短的剩余依赖链当作它可能可以晚发的依据。`b_level` 的单位是边数。验证实际效果时，离线对比相同输入形状、kernel 和 block 配置的两版 profiling。按 Device 时间戳计算 B 最后一块完成到其最早直接后继开始的间隔，B 为终点时取到本轮测量结束的间隔。对 S 列出其各个 block 最早和最晚的入核时刻、最早和最晚的完成时刻，同时检查 B、S 及其他工作在同一引擎上实际重叠的区间。观察到的间隔也包含等核、其他依赖及调度延迟，单凭它不能保证 B 可以推迟同样长的时间。Host 绑定时只使用上述静态信息。

MIX 成员首次接收描述符时校验任务并保存本代的 `sync_start`、异步输入标记和本 lane 的发放门槛。随后使用已保存的值检查准入，执行前仍须观察实际前驱完成和当前发放进度。

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
目标 AIC 自取目录：t8、t11
目标 AIV 目录：   t4、t9、t12、t10（t10 仅 AIV0 代领，成员从 SSBUF 接收）
```

**层号只是排序优先级，不形成执行屏障。** t9 的真实前驱完成后就可以执行，不必等待独立的 t11 完成；同层也不要求同时启动或等待全层资源。Host 排序提供并行机会，实际执行仍受数据就绪、核类型、block 数和成员许可约束，不宣称该启发式对任意图都是最优。

### Host 发布前的硬约束

排序位置 `rank` 与原任务身份 `node_ref` 分离，不改 task ID、kernel、Tensor/scalar 绑定、predicate 或完成身份，不预定物理执行核。两个不可变目录必须是同一个全局序的有序投影，不能分别独立决定优先级。

普通图在 Host 私有镜像发布前，先固定完整原图的 Kahn 排序和两个任务目录，再精简执行检查的 fanin，并按既有 rank 稳定升序排列。精简仅适用于 A5 resident 普通图：live task 在执行或 predicate 跳过之前均须确认全部 fanin DONE，故另一个直接前驱的逻辑 DONE 可以证明其传递前驱 DONE。Host 用每个任务 128 位的位图记录此前 128 个 task ref 内的祖先，只删除已由其他直接前驱证明的边；超出窗口的边、未被覆盖的重复边、原 fanin 存储 offset 和任务目录保持。完整输入先校验，非法边不能被精简掩盖。若 live fanin 行共享或重叠存储，则整份镜像保持原 fanin，不执行原地精简或排序，避免改写另一任务的依赖。

HostDone 节点不传播历史祖先，因为它的输入可能已失效且未经 Device 确认。defer_wait 的 DONE 仍要求 C=0、D=0，不能用计算返回、PUBLISHED 或通知提示替代。普通任务的供给关闭也沿原有前驱供给约束传递，删边不提前越过尚未供给的祖先。native Graph 的 ENTER/EXIT 与共享 Definition 暂不精简；AICPU early-dispatch 不满足相同 DONE 推导合同，也不采用此转换。运行时 DFX 的协议图导出精简后实际上传的 fanin。

1. 复用现有数据依赖合法性检查。现有 `scheduler_fanin_ref` 要求数据 producer ref 小于 consumer ref；非法引用、自依赖等不能靠排序掩盖。Host 已完成节点沿用合法豁免，不重读其失效的历史输入。
2. 对现有数据依赖及 Graph 控制边检查无环性，并验证排序后每条边的 rank；不改写原数据 fanin 的逻辑完成语义。
3. 两个目录保持同一全局 rank 的相对顺序，每个可分发 task 恰有一个领取入口。纯 AIC 进入 AIC 目录，纯 AIV 和所有 MIX 进入 AIV 目录。AIV1 可以跳过 MIX 的领取操作，但要遵守本目录的分发前缀和本 lane 的 SSBUF 参与义务。Host 的掩码剪枝只消除无冲突的资源排序检查。
4. Graph ENTER/EXIT、control/inline 的真实约束必须参与规划；这些节点不因排序变成普通业务执行票。保留实例身份、元数据发布和动态激活条件；sync_start 排序不替代资源准入、真实成员领取与 seal。
5. Host 预计算静态约束和目录位置，Device 不重复计算层号或验证整张拓扑。每类关闭前缀证明其前方供给已结束，未关闭头部约束该类新票；跨引擎前驱保留实际供给检查。目录 rank、已任务分发与 DONE 分开，不用本类头部代替另一类状态。
6. 静态排序、实际领取、真实完成分别验证；目录位置不表示通知已经发起或到达。

Host 必须先完成现有依赖图与目录校验，再发布和 H2D。非法图或成环时报出具体节点关系，不发布部分目录让 Device 补救。不增加逐 block 大表，普通多 block 任务从四个固定大小的分区计数器取票，单 block 任务从 `next_block` 取票。

### 与 percounter、列表调度的关系

| 方法 | 主要依据 | 本方案用途 |
| ---- | -------- | ---------- |
| Kahn 拓扑分层 | 优先安排最早拓扑层的独立分支 | Host 主排序规则 |
| percounter 的 b-level | 最长剩余依赖链优先、原 ID 稳定打破同分 | 借鉴为层内次序；保留正向层号的进展约束 |
| HEFT | 预计计算/通信成本与最早完成时间，并选择处理器 | 后续资源模型的参考；不直接采用其静态物理核映射 |

percounter 的实现位于 `src/a5/runtime/percounter/host/region_plan.h::build_tables` 与 `host/graph_template.h::build_graph_template`。它对普通节点按 b-level 降序，同分按 ID；长支路可能先深入几步再处理独立短支路。本方案把正向层号置于 b-level 之前，两者不能混称为同一个策略。

[HEFT 原论文](https://disco.ethz.ch/courses/fs14/seminar/paper/Jochen/4.pdf) 可作为异构资源调度参考。本方案初版只用静态图信息，不要求 kernel 耗时模型；动态分发由 Device 裁定实际核，刚好一波按下文的 ID 映射。无完整通知关系时，纯 b-level 可能把通知生产者放到等待后继之后，见[固定 owner 下的排序反例](../../../investigations/2026-11-a5-hbg-blevel-fixed-owners.md)。其他排序候选也须先通过该进展约束，再保持 kernel、输入及资源协议一致，以 7 组 L0 总和判断，不按模型选择策略，也不在生产中保留历史兜底开关。

### Device 沿拓扑目录任务分发，前项全体发完再推进

`scheduler_find_issuable_task` 使用本核目录游标及已关闭前缀。**依赖图给出偏序，Host 把它排成满足约束的确定顺序；Device 不临时重新选序。** 同类目录中，前项所有有效执行票发完或按协议合法终结，才能领取后项。跨引擎没有依赖的目录项可以各自推进；有依赖时还须确认另一类前驱已全部完成任务分发或合法终结。实际调用另外等待数据 DONE、Graph 激活及资源许可。

| 当前情况 | 处理 |
| -------- | ---- |
| 本类头部有余块，本核有相应位置且未持有该 task | 领取头部的一块，立即保存固定 owner |
| 本核已持有头部 SPMD 的未退休 block，头部尚未全部完成任务分发 | 不领同 task 第二块，也不越过头部；推进已有 token、协作及独立控制服务 |
| 本核 block 已安全退休，头部仍有余块 | 重新取得该 task 的领取资格，继续补前序余块 |
| 头部全局完成任务分发，部分 block 仍在执行或等待 | 推进头部，可以提前领取后项；不等待前项整体 DONE |
| 普通窗口或相应组/成员位置已满 | 不增加隐藏责任、不迁移 owner；服务已有工作 |
| 跨引擎前驱未发完、元数据未发布或必要控制约束未满足 | 暂不领取该候选，继续相应进展服务；本类头部不能替代这些证明 |
| 候选是 MIX 或 sync_start | 按目录和组/成员协议推进。固定映射或动态分发的有效 seal 证明供给关闭，入场单独检查冲突责任 |

```text
有界补给：
    查询本核目录前沿；已发完或合法终结的项按预算推进
    检查跨引擎前驱供给，以及 Graph 的激活与控制门槛
    本核有相应位置、未持有该 task 且协议允许：领取一个有效单元
    本核暂不能领：返回服务已有责任，不用私有 cursor 跳到后项
    有 ready 的已持有任务：不为继续发新票而延迟它
```

“前视”仍指在前驱计算完成之前持有后继 execution token，不表示越过尚未完成任务分发的目录项。`SchedulerIssueSnapshot` 可缓存单次 fill 已确认的头部，头部推进或已有协议要求失效时须重新查询；不引入逐核扫描后项的选择器。供给 seal 与无票状态通过可靠 `ld_dev` 观察，实际取票和交权仍使用原子更新；不得用 hint 授予 owner 或释放资源。任务完成观察独立使用可靠 `ld_dev` 的合法终态快照，不重复执行 `FAA(0)`。继续参考 `atomic_probe/ATOMIC_USAGE_GUIDE.md` 的同址争用与持续轮询进展证据。

目录内的独立任务也遵守既定分发顺序。上例中 AIC 目录的 t8 完成分发后就能开始 t11 的分发，t11 的计算仍等待其真实前驱；t10 的 MIX 由 AIV0 在另一目录按序代领。Host 层号用于发现并发，cohort 的 rank 另外约束冲突资源的承诺次序。

### 接受的性能代价与本轮不改善的原因

**保留前项全部完成任务分发后再领取后项，接受超出物理核数的宽 SPMD 可能推迟独立后项的代价；本轮不为此引入跳项机制。** 这里用 `N` 表示 task 的逻辑 block 数，用 `K` 表示实际配置的对应引擎资源池的物理核数；AIC、AIV 分别计算，不能将两类核相加作为普通 task 的容量。

当 `N > K`，每核同 task 最多持有一个未退休 block，最多 K 张票可同时被持有；必须有部分 block 执行并退休后，才能继续分配余票。例如 4 个 AIC 已各持有 A 的一块、A 共 8 块且等待前驱 P，目录中后面的独立任务 B 即使已经 ready，也不能领取。A 的余票未发完期间可能出现空闲计算资源，损失原本可与 P 重叠的 B 执行机会。这是说明机制的例子，不是实测性能结论；若核正在满负荷执行 A，也不能将 B 等候直接等同于计算资源浪费。

**当前目标网络的多数业务通常按 SPMD block 数不大于对应物理核数组织，即 `N <= K`。** 这是本轮设计采用的业务常态。在各核有领取机会且协议允许时，全部 block 可以分别交给不同核，无须等待同 task 的 block 退休来腾出下一张票的资格；因此上例由 `N > K` 必然引入的多轮任务分发，不是当前主要优化对象。

本轮优先保留明确的领取顺序、固定 owner 和简单的进展证明，通过 Host 拓扑分层改善独立分支次序，并优化常见路径的交接开销。不为较少出现的超宽 SPMD 增加 Device 动态跳项、额外候选搜索或已领取任务迁移。超宽普通 SPMD 仍须正确执行，这项业务常态不变成 `N > K` 的输入拒绝条件；sync_start 超过同时驻留容量的拒绝规则另行保留。

`N <= K` 也不保证随时完成任务分发：部分核暂忙、窗口已满、MIX 成员约束或 sync_start 的位置承诺 仍可能推迟领取。这些实际等待继续在 L3 中区分，不能据此宣称已经实现最大并发。本轮满核特化按新版 PyPTO-lib 7 组 AICore L0 wall 总和验收（含通信）；只有后续代表性业务证明超宽 SPMD 的供给等待成为主要瓶颈时，再重新评估是否值得改变这项取舍。

**动态分发时 AIV1 在 MIX 目录头的等待。** AIV0 和 AIV1 各自推进同一 AIV 目录的私有游标。遇到未封闭的 MIX 头项时，只有各簇 AIV0 可以代领组；AIV1 仍可服务已发布给它的 MIX 或本核已有责任，但无法越过该项领取后面的纯 AIV 工作。全部 block 已被领取并发布 `claim_seal == block_num` 后才可前移；已确认的逻辑 DONE 也证明供给关闭。宽 MIX 后接独立纯 AIV 工作时，这段等待可能推迟后者的领取。定位时应将 MIX 最后一张票、seal 观察和下一个 AIV 任务的领取对齐；这段等待的耗时需逐图测量。

### 不可变 Call 与可写运行态分开

一个纯核执行单元的身份为 `(run_epoch, node_ref, block_idx, subtask_slot, execution_index)`。`node_ref` 区分 native Graph 实例，不能用原始 source TaskId 或 Graph ID 替代。MIX 的分配单位是整个逻辑 block，其 active lanes 各有独立 Call。

| 信息或操作 | 最早合法时机 | 由谁准备、如何保存 | 留到实际入口的部分 |
| ---------- | ------------ | ------------------ | ------------------ |
| 函数入口、Tensor descriptor 地址、scalar、固定 shape、Call/Binding 地址、block 身份 | Host 已知本 run 的最终基址、图和参数时 | `scheduler_prepare_host_calls` 发布不可变 Call；数据未生成不妨碍地址绑定 | 业务 Tensor 内容的有效性与可见性，不由 Call 提前证明 |
| async image/bytes/run_epoch、group offset/index | 实例及执行单元已确定时 | Host 预绑定固定定位 | 当次 writer/entries/count、实际登记和 seal，不提前创建信用 |
| 目录 metadata、fanin view、当前等待对象地址 | 对应已发布节点首次合法解析时 | 本核 issue binding、token、已发布的组记录或 cooperative wait 中保存 | 动态 supply seal、完成字、activation 等仍按合同观察 |
| 唯一 execute_owner、普通 sub_block_id、固定 Binding 的接收和发布 | 有效票固定 owner 后 | `scheduler_bind_ticketed_execution` 通过 `scheduler_claim_for_execution<true>` 绑定已领票的单元并接收固定 Binding；fanin 满足后，`scheduler_publish_dispatch_slot` 设置 sub_block_id，写回 GlobalContext 并执行 DSB | 当前 owner 是否空闲以及瞬时资源许可 |
| MIX lane 的准备 | 各自的 group/generation 与准备授权成立时 | AIV0 领到执行票后在本次调用里用 `scheduler_notify_mix_members` 通知成员。静态预校验和正式 acquire/prepare 分开；只在已证明的组生命周期内复用 | 动态到齐、ARM/GO、lane 资源和正式执行许可 |
| predicate、Graph 激活、async 保护的业务输入 | 对应数据与控制前驱已满足后 | 实际消费者在合法边界读取和观察 | 当前准备后仍可能发生的状态变化、EXIT/错误 |
| DFX 时间戳、args dump、运行期可写 context | 对应真实事件发生时 | 真实执行者生成；诊断不授予业务许可 | 不能用预填的时间戳或旧 context 代替本次执行事实 |

沿用 `scheduler_prepare_host_calls()` 和既有 metadata/H2D/launch 发布链。函数入口使用注册时的 resolved entry，不能把 CoreCallable 对象地址当函数地址。固定字段只检查一次；执行者仍需通过正确的接收协议读取它们。

入口解析限定在 A5 HBG 的 Host Call 准备路径，复用原四参数 `set_callable_tables()` 提供的对象地址表。
A5 真机按注册器的既有布局取 `CoreCallable::binary_data_offset()` 对应的代码地址，不解引用设备对象；
A5 模拟器读取 Host 对象中的 `resolved_addr()`。A2/A3、TMR 和公共 Runtime 的绑定接口保持原样。

静态字段前移和取消 cache 维护必须分开论证。既有 [Binding 发布调查](../../investigations/2026-11-a5-prepared-binding-publication.md) 没有建立“Host 初始化后可省掉 Binding 接收”的依据；本方案保留所需 Binding 定向观察，并为其可写字段规定唯一 writer。不能仅因移除重复 Device store，就顺带删除保护其他字段可见性的 DCCI/DSB。

### 刚好一波时按 ID 分发

Host 在 Graph 实例展开后，根据本次启动实际使用的簇数 `K` 为每个符合条件的节点标记 `FIXED_OWNER`。无论是否带 `sync_start`，映射相同。

| 形态 | 逻辑 block 数 | 第 `c` 簇拥有的 block |
| ---- | ------------: | --------------------- |
| 纯 C | `K` | AIC 执行 `c` |
| 纯 V（单个逻辑 V subtask） | `2K` | 两个物理 AIV 分别执行 `2c`、`2c+1` |
| 1C1V | `K` | AIV0 发布 `c`，AIC 与原掩码指定的 V 执行 |
| 1C2V | `K` | AIV0 发布 `c`，三个成员各执行同一 block 的一个 subtask |

这些任务在 Host 建图时已确定全部 owner。跨引擎后继可以提前领取；其 kernel 仍须观察真实的逻辑完成，包括异步等待。每个纯核绑定自己的 block 后才推进私有目录。本簇 AIV0 发布对应的 SSBUF 描述符后才能推进覆盖该 MIX 的边界。AIV1 可提前领取目录后项，入场仍受本簇边界和较早的冲突责任约束。带 `sync_start` 的纯核在本核还持有较早普通工作时，先退休较早工作。

静态分发不触碰 GM 分区计数器、`next_block` 和 `claim_seal`。L4 按实际 cluster、lane、block 和发布次序核对责任。部分占用、多波、纯双 V 的 MIX 和条件任务继续使用动态分发。固定分配的成本是忙核返回前无人代执行它的 block。

### 动态分发下普通 block 的首次领取

通过有序供给检查后，一次成功的动态任务分发固定一个 block 的 owner。普通多 block SPMD 使用四个 128B 分区领取，单 block 可沿单计数器路径。第一次有效领取后将身份写入原自取位置；入核不再领取或转存。分区关闭、全任务 `claim_seal`、fanin 观察和 owner 诊断的成本分别核算。

- 新票选择检查本核是否已有同 `(run_epoch, node_ref)` 的 current/inflight/pending；存在则该 task 暂不可本核再领，头部尚未全部完成任务分发时不能自行检查后项。退休后若头部仍有余块，继续领取前序任务；头部全部完成任务分发后才开放后项。
- 有效领取立即记录在原位置。单 block 路径退回争用产生的超额票；多 block 分区的超额票不产生 owner，也不回退，继续探测其他分区。
- Host 已发布、初次唯一票上的诊断初态或执行身份失配是错误或已确认的终态取消，不是普通竞争落败。
- 对四分区任务，目录通过 GM `claim_seal == N` 确认 `ISSUED_ALL`。已映射时，普通非同步任务在位图全齐后允许幂等发布，协同路径由 CAS 选出发布者。未映射时，由分区容量累加到 N 的领取者发布。已领取的执行责任仍由原 owner 推进。
- 静态初态由通用计划生成并校验 run、单元范围及区段边界。普通 owner 对独占诊断记录的旧值检查只用于拒绝生命周期失配，不承担另一场跨核领取竞争；不能凭普通读取授予共享 cell 的所有权。

### 自取窗口的深度与执行责任

W 表示每个物理 worker 能同时持有的未退休自取 block 数。当前采用 W1。每核另有一份独立的 MIX 成员记录，对应本簇的 SSBUF bank。执行中的 block 也占其来源位置。自取窗口深度只改变待运行工作的持有量，单核一次仍执行一个 kernel。

2026-11-18 的同结构 W1–W4 对照每组同设备预热 1 次、测量 10 次，按轮换顺序运行。7 项 AICore L0 均值总和为 W1 18.068514 ms、W4 18.234993 ms，W1 低 0.91%。其中 11 个回放用例逐版通过数值校验，原生 MoE 的 72/72 只记录性能运行。单个 SWA 的 W1/W4 为 0.209540/0.213708 ms，原生 MoE 为 0.604940/0.604284 ms。当时依据这批负载选择了 W1；其他图的依赖宽度、MIX 比例和控制服务会改变最合适的深度。

普通执行槽以一个本核状态记录从领取到退休的全过程。有效票将槽置为 `TICKETED`，接收 Host 预建的 Call 和 Binding 后进入 `FILLING`。前驱和谓词检查完成后发布 `READY`，最终顺序许可通过才转入 `EXECUTING`，完成记账后回到 `FREE`。绑定失败时保留票和已获得的 handle，供错误清理定位。

下表描述各阶段的协议责任。

| 自取位置的阶段 | 持有的责任 | 允许的推进 |
| -------------- | ---------- | ---------- |
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

任务目录与 issue_head 只用于分配尚未领取的执行单元。已经领取的单元由原核直接观察依赖，不进入全局再次发现或竞争执行权的流程。本核唤醒后继续核对原 owner 与记录身份。

等待中的执行记录保留责任。前驱、Graph 控制节点与通信仍需各自取得进展，窗口深度及未退休执行的实际等待可在泳道图中分别核查。

## 3. 快速接续的原子成本、服务边界与 SPMD 分布

### 原子操作分别算在哪条路径

| 操作 | 普通路径的目标成本 | 是否应留在已接收 token 的接续间隙 |
| ---- | ------------------ | --------------------------------- |
| 有序供给、选择本地空位 | 头部进度、前驱供给、本核去重与控制门槛；关闭证明与任务分发竞争另计 | 否；已有可执行 token 不依赖新的补给 |
| 初次分配具体 block | 单 block 从 `next_block` 取票；多 block 从分区计数器取票，分区最后一个有效领取者汇总容量 | 否，已持有记录可直接接续 |
| 初次授予单元所有权 | 普通任务的有效 FAA 确定 owner。MIX 由代理领取组任务，SSBUF 有效代和参与掩码确定各成员的执行权 | 否；应已持票并完成接收 |
| 读取固定 Call／接收 Binding | Call 复用本轮启动观察；Binding 两行 DCCI、尾 DSB 和私有元数据缓存 | 否；等待期间保留已完成的接收，不因等待重新竞争执行权 |
| 检查依赖和最终许可 | 只推进未确认的 fanin 前缀；动态许可按实际合同观察 | 是；不能用旧观察替代新出现的条件 |
| `WAIT_FANIN → INFLIGHT` | 唯一 owner 的私有状态转换，不新增领取 FAA/CAS 或本核 ready 队列原子 | 是；同一 helper 继续调用 |
| kernel 返回及完成发布 | 保留执行权退休、N/C/D 贡献、输出发布和必要协作操作 | 是；不能把完成成本藏在 kernel 时间内 |

普通单元首次有效分配由一条 FAA 授予不可撤销的 block/lane 票，后续接收不再独立竞争执行记录。票限定的执行记录标记 TICKET_OWNED，整个128B区域仅由该 owner 更新，使用普通 store、单行 DCCI、尾 DSB 发布 CLAIMED/RETURNED/FAILED 诊断；该区域没有其他 Device writer 或执行许可读者，不承担跨核交权。AICPU 错误快照保留原定向 invalidate，Host 在所有核退休后读取，仍保留原有 owner、phase 和完成推导。取消先核对 owner 和当前状态，失败不产生完成贡献；普通最后返回的 C 归零推导仍成立。MIX 的代理分发与成员执行分别记录。AIV0 在每簇分别记录最近领取的组、bank 最近发布的组及发布前取消的组。新领取的组等待 bank 释放时，仍可定位在途的已发布组。SSBUF 的有效代发布后，各活跃成员在本核 handle 中记录 CLAIMED 和 ENTERED，物理返回时向 GM 发布 RETURNED。GM 中的 BUILT 表示尚无返回记录，成员可能在等待依赖或已进入 kernel。Host 对此状态的执行归属和入口证据均输出 null。开启 L4 协议采集时，成员实际接收已发布的组后额外向 GM 写入 CLAIMED，跨过依赖和协同许可、即将调用 kernel 时写入 ENTERED。超时快照据此区分已接收但未入场、已到达调用边界、已返回三个阶段。MIX 成员在最终入口许可处读取 SSBUF 代号并执行 DSB_ALL。关闭采集、PMU 和参数 dump，且本成员没有异步登记或受保护输入时，实际执行的同步 MIX 可直接进入 kernel。其他入口保留 DSB_DDR，FIN 和退出各自维持发布顺序。成员接收失败或调用前取消只处理本核 handle，真实错误由现有 run_error 通报；GM 不记录这些阶段。发布前取消时各成员保持 BUILT。诊断记录只供观察，调度以有效组分发、稳定 SSBUF 代、参与掩码和最终许可为准。

领取阶段沿用各 task 的分地址竞争。核可以复用本实例已确认的耗尽证明以减少探测；没有取到有效 block 不能推导本簇所有 MIX 已发布，也不能替代真实分发关闭。这样接续一个已持有单元时，无须再为它竞争一次。

供给游标读取已发布的合法 seal 后复用分发关闭证明。尚无关闭证明时，按分配器合同观察实际游标；局部无余量、组内发布和依赖 DONE 有各自不同的读取语义。

异步计算关闭入口由最后计算返回授权，所有注册已经 seal；调用方直接传入最后 physical FAA 的完成字结果，不重复读取 completion。deferred 路径在 close RMW 前执行 DSB，利用该 RMW 返回的 join 字校验并裁定唯一 FINISH owner，不追加 join 的 ld_dev。同步 checked 入口没有 close RMW，在读取 join 校验零登记之前保留 DSB。屏障排序本核最后贡献与后续共享操作，不能由贡献之前的屏障替代；并发 CHECK、payload 发布及其他数据可见性合同继续保留。

### 本轮不可变 Call 的缓存观察

Host 在启动前构造并上传全部 Call，包括当前 native Graph 的 child 实例。metadata 上传通过 entry fork 事件先于 AICore 启动；每次 resident 入口执行原有 DSB + DCI，清除该核上轮缓存。Call 在本轮内没有 Device 写者，普通接收与 MIX 提前准备均直接读取已发布 Call，不再逐次遍历512B执行DCCI，不新增参数extent或额外屏障。所有 epoch/task/execution/block/lane、参数数量、function和固定context指针校验保留。

Binding 接收仍观察 local/global context 两行，保留接收尾 DSB 与 compiler barrier。LocalContext 的 block 几何只由 Host 初始化，Device 准备阶段不再对未修改的 LocalContext 二次写回；普通任务在 fanin 满足后写回 GlobalContext 并执行 DSB，随后检查最终执行许可。defer_wait 的业务 tensor 数据仍由实际消费者在依赖满足后观察；Call 地址和参数不可变不代表其指向的数据已经产出。Graph 激活、Tensor/view观察、args_dump边界和最终执行许可均保持。跨run复用依赖重新上传及每次resident入口的清缓存，不能把本轮证明带到下一轮。

### 本核复用已确认的任务完成

每个 worker 在本次 run 的私有状态中保留16个任务证明，共128B，按 task ref 取模索引，编码为 task 身份与 SUPPLY_CLOSED/DONE 等级。正向 seal 通过可靠 `ld_dev` 观察、或最后有效票 owner 的真实发布可建立供给关闭证明；可靠观察计数器阈值、SET=1 或 GM `C=D=0`，以及 GM 完成 RMW 返回的合法 DONE，均可建立 DONE。同一 task 的证明不降级，供给关闭不能放行 kernel；同一不可变运行镜像内重复检查可复用对应等级。供给检查和执行 fanin 共享本核证明，不把完成任务分发、物理返回或负向 `ld_dev` 提示当作完成。

本核尚无对应证明时，供给检查先观察绑定的完成来源，按计数器、SET、GM 的顺序选择已有映射；只有可靠 `ld_dev` 观察到该后端的合法 DONE 才直接建立可同时服务供给及执行 fanin 的证明。尚未完成则继续检查原供给 seal，允许已完成任务分发但仍在计算或等待异步条件的任务开放后项领取。`C=0,D>0` 仍为未完成，不能把物理返回当作 DONE。此顺序不增加缓存或共享状态；已 sealed 但未完成的任务会多读一次完成来源提示，必须在全套 L0 对照中计入这项成本。

只有已确认的供给关闭或 DONE 才写入对应证明；供给证明不能冒充完成，非法状态不增加证明。索引碰撞只遗忘旧证明，不授予新任务完成。每次 resident 启动重新构造私有状态，Graph 展开后的 task ref 仍保留实例身份，不能跨 run 或任务身份复用。首次 DONE 观察使用可靠 `ld_dev`，不再追加原子读；各后端的完成贡献协议、payload 可见性和最终 MIX/sync_start/predicate/错误许可检查均保留。此缓存不增加共享 ready 状态，也不转移执行权。

### 最终入口与有界控制服务

本核目录前沿缓存 entry 的固定 metadata、控制字地址及已解析的 fanin。普通补给与同任务续领使用同一目录头，MIX 代领使用其合法的 AIV0 目录位置。更换任务、Graph 实例或 run 时重新绑定。最终承诺消费的是两种来源的排序与资源证明，见[SSBUF 边界](#a5-mix-floor)。

普通 token 和 MIX 成员各自持有一个执行位置。本核的 fanin 游标指向第一个尚未确认的前驱。首次解码校验前驱 ref，优先复用本轮已缓存的 DONE 证明。本核仍欠该 MIX 前驱的成员贡献时，后继继续等待。需要继续确认时，首次观察先识别 HostDone，再为设备前驱绑定完成来源。遇到 PENDING 时，执行位置保存前驱 ref 和本轮的完成来源。对于 CntNotify，位置缓存读地址、任务完成阈值和整链上限，重试时仅重新读取实时计数，超过整链上限即报错。SET 通知以 0 和 1 表示等待和完成，其余值报错。GM 后端重新读取已绑定的控制字；确认后清除绑定并推进游标。游标到达末尾即表示全部前驱满足。位置换绑或新一轮运行会清空等待状态，native Graph 的 ref 始终保留实例身份。任务分发另行检查前驱的分发进度。

全局自取成功后，原位置保留固定 metadata、已接收的 Call/Binding 和已发布的本核字段，随后等待 fanin。就绪时直接使用该位置；因冲突暂缓进入时继续持有同一份 handle。MIX 的身份来自 SSBUF，执行时仍须观察实际输入并建立必要的 async context。

一次选择至多检查 W 个自取位置。一次 kernel 返回可能解锁另一份原地责任，优先重访已有候选。最多四个普通 kernel 返回或四次无 kernel 的推进轮次后，提供一次有界 async/control 服务。容量实验只变 W，保持该服务预算和 MIX 退休规则一致。

`WAIT_FANIN` 的条件满足后，选择器传递原持有位置及其身份。同一次推进中验证本核资源许可和本轮实际输入可见性，再调用 kernel；MIX 用其已接收的本代记录和 handle 执行同一入口。资格暂不成立时继续独立控制服务，后续按原记录重查。

任何执行来源均可直接到达 kernel 与归还入口。删除强制发布本地 READY、复制为 dispatch slot 以及许可被拒后从 slot 搬回 token 的路径。合法 predicate skip、异步登记、完成信用、DFX 身份和错误退出都使用原 execution 身份。

从串行时间看，已接收 pending 的 `G_worker` 仍包含“前项必要完成 + token 选择 + 剩余依赖观察 + 最终许可/动态 context + 调用入口”，到期的控制服务及已承诺的协作责任也要计入。提前领取只移走发现、取票、Claim 和固定数据接收，不能消除这条剩余链。

`G_dep` 沿真正最后前驱的完成发布、后继观察与调用入口计算；跨核可重叠的工作不相加。实现后若这条剩余链仍超过 5 µs，必须回到泳道细分其 atomic、DCCI/DSB 和普通访存，不能用“四 token 已完成”代替性能达标。

### 普通 SPMD 的分布约束

默认规则用于非 sync 的纯 AIC/AIV；MIX 和 sync 的票据另见下一节。

- 每个 worker 对同一普通 `(run_epoch, node_ref)` 至多持有一个未退休 block，执行中的 block 也计入。新任务分发和同任务续领共用此检查；MIX 以实际组成员和 generation 追踪归属。
- 持有 A.block0 时，本核不能再领 A.block1；A 全局完成任务分发后，在其他门槛和剩余容量允许时，可以按 Host 顺序提前领取 B.block0，不必等 A.block0 或整个 A 完成。A.block0 退休后若 A 仍有未分配余块，先领取其余块。不同 Graph 实例按 node_ref 区分，不按 kernel ID 去重。
- 每次只领取一个新 block；成功后先推进现有 token，不连续批领同任务多个 block。领取一个 block 不等于该 task 发完，不能据此推进 issue_head。
- 实际执行核由动态取票决定，Host 不固定 block→worker，也不按 prepare_owner、prefix 取模或缓存亲和性分配执行权。
- 本核 block 返回并完成必要记账、执行责任退休后，可以继续领取同 task 的其他 block；不等待整个 SPMD DONE、其他核返回或 task 级异步债务归零。例如 64 个 AIV 执行 128 个 block，第一块退休的核即可领取第二块，不增加全局轮次屏障。
- 等待期间始终计入原核持有的责任；只有本核执行退休或错误终止才能结束这份责任。等待不构成再次领取同一 SPMD 另一 block 的资格。

同一任务的各 block 共用不可变 fanin，成功返回后的续领可沿用本核已验证的依赖前缀，负向结果不跨轮缓存。Claim、Call/Binding 接收、predicate 和最终资源许可仍逐单元保留；MIX、sync_start 走独立协议。全窗口等待时保持全部普通执行责任，async/control 继续获得独立服务机会。

单核同任务单 block 与窗口上限不证明跨 task 的负载完全均衡。多个不同未来任务可能集中在少数核；必须结合等待跨度、空闲核、起跑分布和每核执行次数验证。不能仅以 Claim 变快、局部 cache 命中增加或 pending 数增多证明整体收益。

## 4. MIX、sync_start、defer_wait、Graph 与内存边界

### 任务形态、代理与完成

| 任务形态 | 每个 block 使用的资源 | 不同 block 的分散与入口约束 |
| -------- | --------------------- | --------------------------- |
| 纯 AIC 或纯 AIV | 一个符合类型的核 | 普通 SPMD 允许一个核顺序执行多个 block |
| 纯核 `sync_start` | 一个符合类型的核 | N 个 block 需要 N 个不同的核。本核早返回也不能再领本实例的另一块 |
| 普通 MIX | 同一 cluster 中实际掩码指定的一组 lane | 组内资源按同一 generation 交接，一个 cluster 可顺序执行多个 block |
| MIX 加 `sync_start` | 每个 block 仍需一组匹配的 lane | N 个 block 需要 N 个可同时兑现的组位置。每组取得组内许可即可进入，其他组可稍后进入 |

AIV0 是每个 cluster **唯一的 MIX 领取者和 SSBUF 描述符发布者**，覆盖普通 MIX 和带 `sync_start` 的 MIX。一个逻辑 block 只领取一次，AIV1 和 AIC 依据正式发布取得自己的成员责任。实际参与掩码决定会合、执行与 FIN：在 `AIV0+AIV1` 的纯向量组中，AIC 可以继续无冲突工作；在 `AIC+AIV1` 的组中，AIV0 负责发布，但没有计算、到达或 FIN 信用。Host 分开编码“谁可代领”和“谁参与计算/占用硬件”，容量检查按真实核类型、组拓扑与互斥资源计算。

descriptor 携带 run、实例、rank、block、active mask、generation 及组发布序号。各成员确认本核 fanin 和已领取的早序任务。有活动 lane 受更早纯核 `sync_start` 约束时，各成员发布 ORDER_READY 和本代 ARRIVED，核对全组到齐后入场。无冲突时成员在本核检查通过后直接 READY。CHECK 在调度循环中服务，依赖异步任务的后继继续等待其逻辑完成。L4 在独立入场时仅记录本 lane 的到达。使用 CntNotify 的任务由每个执行成员在成功返回时发布一份物理完成，然后各自发布带代际的 FIN。AIV0 在复用记录或退出时核对参与成员的本代 FIN。协议采集另外记录该代的退休证明。CPU/sim 的 GM 路径由记账成员收齐 FIN 后归并物理完成。组逻辑完成无需等不参与计算的代理再次得到调度。

各活动 lane 独立读取同一份 fanin，用本核游标保留已经确认的前缀。提前入场时，较早进入的 AIC 可能在 kernel 内等待 AIV 的跨核事件，而 AIV 正在执行已获准的普通任务。一直到 peer 抵达事件，这段等待都会占用 AIC。Host 的排序门槛约束较早的冲突义务；它不能给无关 kernel 的剩余时长或成员到达偏差设定上界。

正常上板时，AIV0 直接核对本代活动成员的 FIN。协议采集和 CPU/sim 的 GM 路径另外由记账成员 release 发布 `accounted_generation`。AIV0 确认退休条件后，用 `scheduler_cluster_store_release` 把 `armed_generation` 改为偶数。该 helper 在 store 之前执行 DSB；紧接 store 的第二个 DSB 使旧代失效的写入先于该 bank 的描述符覆盖。可能仍在读取旧奇数代的观察者会在读取字段后再次核对 generation。新描述符填完后，AIV0 才 release 发布下一代奇数。

<!-- markdownlint-disable-next-line MD033 -->
<a id="a5-mix-floor"></a>

### 全设备协作 rank 与每簇一份 SSBUF 单调边界

Host 在每个设备 run 内生成拓扑相容且唯一的协作 `rank`。普通 MIX 同样会承诺组内资源，也纳入冲突排序。各核使用同一顺序在自己资源域中的投影。全局自取侧依据目录与分发关闭证明推进本地游标；AIV0 为 MIX 分发侧发布本簇唯一的 `mix_floor`。两个来源都能提前发现和准备。首次**不可撤销的资源承诺**要检查双方：一旦某个动作使本 lane 无法再服务更早的冲突任务，该动作即为检查边界。只固定 block 的 owner、仍能履行更早义务的提前领取可留在边界之前。

`mix_floor = F` 表示：**在本次 run，本簇以后不会再发布 rank 小于 F 的 MIX 记录**。相同 rank 的不同 block 用组序号和 generation 区分。本簇记录在上一代退休后才可复用。F 在本轮内只增不减。启动握手隔离上一轮并清除旧记录后设置初值；未来可能出现 MIX 时初值不能高于最早可能的 rank。终值须有目录与 Graph 供给已经封闭的证明，已发布的组仍按各自身份退休。

1. **Host 剪枝。** 对候选 rank `r` 和它会承诺的资源，Host 用参与掩码和硬件互斥关系排除确定无冲突的早序项。设剩余早序 MIX 的最大 rank 为 `h`，只需检查 `F > h`；集合为空就省去这次 SSBUF 读取。对动态归属采用可能参与的资源集合，不能用“还没收到”推断不参与。资源独立时仍须遵循既有的数据、Graph 激活和目录供给约束。
2. **AIV0 发布。** AIV0 通过 Host 的相邻 MIX 索引按序推进 F。本簇领取执行票时就在同一次调用中发布描述符。观察到该项供给关闭，或本簇已领到该同步任务的唯一 block 后，才可将 F 提高到下一项。同任务续领也受该界限约束。先写描述符并按 SSBUF 协议发布 generation/序号，再 release 发布提高后的 F。只在 F 前进时写。即使本簇本次未领到 MIX，AIV0 观察到实际供给关闭后仍须发布进度。GM `claim_seal` 本身不能证明本簇记录已可见。
3. **两侧仲裁。** 自取来源要跨过边界时，先 acquire 最新 F，再核对它覆盖的已发布组、自己的参与位与 generation。不能先看到空 mailbox、再用新 F 追认“没有旧组”。MIX 来源反向检查本地自取目录：较早的冲突 `sync_start` 若仍可需要本核，且本核尚未履行其义务，就先推进它；已关闭分发且没有本核遗留义务，或已静态排除冲突，才可越过。两个域使用同一个 rank；本核游标保存在本地，只有 MIX 来源的 F 发布到 SSBUF。
4. **允许抢跑。** 若 M10 可能需要本核，承诺 S11 前须看见 `F >= 11` 并处理已发布的 M10。首次进入 S11 的门槛不含 `F > 11` 或 S11 的 `claim_seal == block_num`。`sync_start` 允许各核错开起跑，前提是剩余成员仍有互不冲突的位置和取得它们的进展路径。早返回即可归还执行位置；若本核仍可能重访该实例，只保留已领资格的标记，直至目录/成员前缀能证明本核不会再领第二块。

两个来源同时可选时，先兑现已作出的本代承诺，再按 rank 选择冲突候选。同 rank 的 MIX 块在本簇按发布序号交接，覆盖可复用记录前先确认前代已退休。正向 F 证明在本 run 内缓存，只在所需边界高于缓存时刷新；未满足时保留责任并推进其他合法工作。Host 已排除冲突的路径，因保序新增的共享读取应为零。SSBUF 按已有 release/acquire 协议处理有先后关系的读写，保序不增加每 block 的 GM atomic 或 CntNotify。

### 冲突 rank 的进展证明

以能通过静态容量检查、依赖满足且可分发的最小冲突 rank 为例。每个参与者要么可以服务它，要么在执行能自行返回的先前工作，要么仍在履行严格更早的义务。高 rank 的承诺不能夺走它所需的最后一个合格核或组。对于同 rank，纯核成员各领一块；MIX 成员等待同一组的真实成员完成资源交接，组内不会因槽位复用看到不同 generation。由此可以从最早未兑现的义务向前推进。该论证以算子按既有同步合同返回、外部条件有真实生产者、控制服务有界为前提；只检查静态核数不能成立。

例如 M10 需要 C、V0、V1，S11 的两个 block 需要两个不同的 V。即使 V0 先见 S11、V1 先见 M10，V0 在 S11 上的不可撤销承诺也要等待 M10 的较早冲突义务处理完，或取得本簇排除它的证明。若 S11 的 kernel 内有合法的成员同步，缺少这条规则时，V0 可在 S11 内等 V1，而 V1 同时在 M10 等 V0。

AIV0 还有一项软件上的**发布责任**。进入可能等待伙伴的 kernel 之前，它必须先完成伙伴取得许可所依赖的本簇描述符和边界发布。即使某个早序 MIX 不包含 AIV0 的计算 lane，也不得仅凭计算掩码剪掉这项发布责任。尚不能给出证明就暂缓承诺，同时推进分发、已有组和 CHECK。各簇独立兑现同一 rank 约束，不设全设备的逐块到达屏障。

`claim_seal` 证明前项供给关闭，`mix_floor` 约束本簇未来可发布的 MIX，ARRIVED 按本组入场条件提供到达证明，FIN 约束本代实际成员的退休，逻辑 DONE 释放数据后继。每项证明都有独立的生产者和消费点。

### defer_wait：物理返回与逻辑完成分离 {#deferred-physical-and-logical-completion}

执行 defer_wait 的物理核封口非空条件组后释放执行位置，并负责该组直至结清。当前组的索引、待满足条件位图和游标保存在核内。条件地址和阈值位于 Host 为该 execution 预留的稳定记录中。

```text
defer_wait 登记条件并物理返回，释放执行位置
    → 消费者可以在合法前视窗口内等待，保持原 owner
    → CHECK 观察实际通知条件
    → 计数器路径贡献 Q；GM 路径由 join FAA 选出唯一 FINISH_OWNER 并结清信用
    → 消费者在其真实依赖全部完成后执行
```

同核有多个等待组时，多出的组使用各自预留的 GM 记录组成 FIFO。每次最多检查一个组的八个条件。若还有其他组，则把未完成的组放回队尾。单个 COUNTER 首次检查未达标时，登记核保存已校验的计数器地址和阈值。后续未达标的 CHECK 每次只读取一次该地址。达标时复用本次采样，检查错误状态并核验描述符，再提交条件信用。组切换、追加等待组或启动新一轮运行时清除缓存。含多个条件的组在登记和条件满足时将待满足位图发布到该组的 GM 进度记录，AICPU 通过位图变化观察条件进展。仅轮询而未满足任何条件时，当前组的检查游标仍可在本核推进。调度主循环继续周期性检查错误。

本核有待检查组时，在进入下一个业务 kernel 前给一次检查机会，空闲时同样推进。本核队列为空时直接返回。登记核执行长 kernel 期间，已到达的信号会在它回到调度循环后观察。

以下 join/FINISH 与 N16/C16/D32 归零步骤适用于未映射完成计数器的 GM 路径；计数器任务通过 P/Q 达到累计阈值后直接就绪，再执行 canonical 归档。

最后计算返回或最后条件组完成的 join FAA 返回 `NONE / WAITING / FINISH_OWNER / ERROR`。唯一 owner 携带本次原子返回值推导的 join 终态及 task/run 身份，在同一调用链中直接检查 canonical completion 并 CAS 到 DONE；不再发布 FINISH lease/目录，不重新竞争责任，也不将 outcome 留到下一次 kernel。登记核确认等待位图已清空后，先从本地移出已完成的组并提升队头（队列为空时清除当前组），再结算该组的 join。owner 产生后的错误必须使 run 失败，不能重试或丢弃责任；最终 CAS 成功者发布逻辑完成，并在采集时记录真实 DFX 完成因果。计算返回和 MIX 信用外层不能重复计数。

FINISH owner 从本轮不可变 task metadata 的 block 数与活跃 lane 数得到总贡献数 N，由已确认的 join 结果验证存在合法非零登记；canonical debt 只保存单个 has_async 门闩，预期 completion 为 `(N, C=0, D=1)`。校验 N 与 join 后，直接 CAS 该预期值到 `(N, 0, 0)`，不先 FAA(0) 再读取同一个完成字。CAS 返回值是唯一成功依据；剩余计算未归零、债务不符、总数不符或重复 FINISH 均失败且不修改原状态。保持原有发布屏障、run/实例身份校验及 DFX 因果，不把静态 N 或 join 本身当作任务 DONE。

FINISH 目录、lease、扫描游标及独立服务轮换同时删除，Host 分配、Device 校验和内部布局同步更新。`SchedulerAsyncLayout` 仍128B，公开 AsyncCtx、AsyncWriter、AsyncGroupData 和条件记录 ABI 不变；不改模型 kernel 或前端表达。

复用现有 N16/C16/D32 完成字：物理返回归还相应份额，非空异步组的信用由唯一 FINISH 结清。取票耗尽、kernel 返回和目录顺序都不能伪造逻辑 DONE；保留条件登记与最后返回交错的 join 证明，以及真实归零原子的 DFX 因果身份。

Host 不推断本卡或跨卡 notify/defer_wait 对应关系，不新增通知排序边、通信上下文快照或 PyPTO 元数据导出。kernel 内同步和跨卡通信继续遵守原协议；若真实场景证明进展不足，再按 [#3096](https://github.com/hw-native-sys/simpler/issues/3096) 分析，不能把人工 self-notify 用例当作当前模型需求。

### native Graph、predicate 与安全退出

协作 rank 按 **run 和展开后的实例节点** 分配。同一 Definition 的不同调用保留独立成员身份。能在 Host 确定的实例使用稳定 rank 和参与掩码；尚未激活的有界区段须预留有序范围，或让发布前缀停在最早未决位置。实例元数据及控制条件发布后，只能在预留范围内解析或沿已证明稳定的前缀追加。可按既有 predicate/控制语义证明跳过的区段才能封闭。禁止在已经公布的 F 之前补入新的冲突 rank。仅使用已有图、实例和激活信息，不引入跨卡 Host 交换。

持有未决区段的候选不独占其生产者或 Graph control 所需资源。本轮无法给出可靠次序或进展证明的 Graph 形态，是移除旧日志的阻断条件，不能以跳过该形态或把占位节点视为 DONE 来验收。

Host 预构造每个实例的 Call、边界 alias/view、scalar 和 heap 地址；Token 用 run + node_ref 区分实例。拓扑分层覆盖实例实际数据和 ENTER/CHILD/EXIT 控制关系；排序不改原 ref 和参数绑定。child 只有在元数据已发布、必要的 ENTER/激活条件满足时才可领取；执行仍等待真实计算前驱 DONE。普通前驱全部完成任务分发不能替代实例激活或控制门槛，目录投影必须保留这些约束的检查位置，不能因只看可执行节点而丢掉实例边界。外部消费者仍遵守 EXIT 合同；未发布实例不分配任务、不读取占位指针，不补 Device 动态生成旁路。

predicate 仍在合法执行边界读取；false 分支按既有整任务/组语义完成 skip，贡献恰好一次，不能让 MIX 某 lane 跳过而伙伴进入等待。提前取得固定 async 定位信息不代表创建 writer 的可写状态、登记条件或发布信用。

错误、取消和 EXIT 按责任清点：TICKETED 新票、OWNED pending、正式 MIX/cohort、INFLIGHT、完成贡献和本核挂起的异步组 都必须有最终归宿。只能按现有错误协议终止责任，不能转交其他核重跑同一 execution；inflight 等真实返回后退休。Host 不能以 cursor 发完或某一核本地窗口为空就回收 Call/Binding/Graph heap。

需要同时证明“所有执行单元已实际返回或按失败合同终止”“没有遗留的已领取责任”“组资源和 async/control 已安全退休”。run/实例身份及协作 generation 不回绕，越界或旧身份不能当作正常重复消费。内部布局与导出协议变更时同步更新 Host、Device、reader 和测试。

### 缓存与存储合同

“Wait 的写侧可以考虑 st_dev”必须先拆成逐字段协议。决定方式的是该值由谁更新、谁依据它获得什么权限、是否复用以及整条 cacheline 的访问合同；不能只看名字里有 ready/done，也不能只数 writer。

| 数据 | 写入与观察 |
| ---- | ---------- |
| Host Call、固定 Binding | Host 发布静态字段，领取者按当前 run 接收；可写 Binding 由实际 owner 在相应发布边界写回并排序 |
| 分区计数、`next_block`、`claim_seal` | 有效票由原子操作授予；最后领满任务的路径发布供给关闭，其他 worker 可复用已确认的关闭证明 |
| SSBUF 描述符、generation、`mix_floor` | AIV0 以簇内 release/acquire 发布。各活动 lane 只写自己的 ARRIVED、FIN 序号，AIV0 复用 bank 时核对本代全部成功 FIN |
| 第一错误及退出状态 | 错误由实际认领者发布。退出要求实际在途责任已清退、参与核握手完成 |

对候选 st_dev 发布，必须先给出数据→通知的顺序、reader 的 payload 可见性、整行不被旧脏缓存覆盖、精确位宽/对齐、重复写及 writer 交接、epoch/sequence 复用边界。DSB 只建立相应本核内存顺序，不把普通覆盖写变成多写者 RMW。官方 [WriteGmByPassDCache 说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/scalar_compute/WriteGmByPassDCache_ISASI.md) 明确要求处理旧脏行写回覆盖及 DCCI 与 bypass write 的顺序。

工作区 `simpler-fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md` 的原始结果继续作为证据，旧的全面限制 st_dev 的工程建议不再覆盖本节统一原则。重复写仅尾部 DSB 的失败，不能推广为显式满足每条时序依赖的 st_dev 发布都不可用；逐写 DSB 的通过样本也不能授权覆盖多写者更新。既有 [claim_seal 替换调查](../../investigations/2026-11-a5-hbg-admission-shortcuts.md) 保留当时撤回的结果，新候选按实际 writer、发布顺序和生命周期判断并做相应验证，不改写历史为成功，也不把旧结论作为永久禁用理由。

- 共享 RMW 控制按 **128B atomic-only 隔离**设计；64B 是 Scalar DCache 操作粒度。RMW 行内不混入 ordinary store 或 DCCI 数据。普通有效票独占的执行诊断区域属于 owner 数据，整条区域不再有 RMW 或跨核许可观察，独立使用 DCCI 发布；不能与 cooperative 控制记录混用。
- Call 复用 Host 上传与启动清缓存建立的本轮只读可见性；初次 Claim 成功后，Binding 继续两行 DCCI、尾 DSB 与 compiler barrier。该接收可提前到 WAIT_FANIN，但不能据此省略前驱计算及 defer_wait 数据的执行前观察。
- 每核 token 只由固定 owner 写，可使用 owner-private 存储；初次执行授权和共享完成仍使用原子协议。本核等待状态不作为跨核交权信号。
- `scheduler_complete_lane_execution` 接收 resident executor 当前持有的 slot。执行器在调用前检查 READY、CLAIMED 和 owner，并核对最终 entry permit；MIX 额外核对 bank 的 armed generation 和 run epoch。获得许可后，本核将该 slot 切为 EXECUTING。此后的 block/subtask 范围、lane 与核类型的对应、sync_start 属性及 MIX generation 都沿用同一次领取与准备确定的身份。kernel 只接收 GM 参数和上下文，本核调用期间不运行调度循环，其他核也不能改写它的私有 slot。
- 完成 helper 依赖上述执行器不变量，不重新把这些字段与任务元数据逐一比对，也不为任意传入的 READY slot 提供独立状态检查。完成入口只用于同一个 slot 的真实返回或合法 predicate skip；调用后封口或抓取失败走只退役、无成功贡献的分支。新增调用者须先建立同样的状态与所有权前提。
- 执行器真实返回时，普通有效票的 PARTIAL 和失败记录按 owner 数据发布；MIX 成员记录自己的物理返回，再在 SSBUF 发布自己的本代 FIN。普通最后一次成功返回可复用贡献 FAA 的 C 归零证明，省略该单元的 RETURNED 发布；D>0 时 task 仍未逻辑完成。检查入口继续拒绝错误 owner、状态不符和已推导退休的旧句柄。完成贡献前的 DSB、失败返回在组 ACK 前的 DSB 均保留，执行责任退休不替代任务 C/D/N 完成计数。
- 可写动态 Binding 延后到最终许可后由固定 executor 填写；准备者不能修改属于另一 executor 的可写上下文。等待期间保留参数与观察进度，必要的缓存可见性协议不因固定归属而省略。
- 提前观察参数地址不等于提前观察业务 Tensor 内容。受前驱或 async 保护的数据仍在实际依赖满足后执行必要的数据观察。
- 沿用启动路径的 DCache 清理合同，不引入每 task 全 cache DCI，不将 DCCI 误当按地址 invalidate-only；SSBUF 使用其专用 release/acquire 屏障。
- 共享记录、Call 和 Binding 在 run 内不复用地址，直到安全退休。记录 token、SSBUF、Host 初始化和实际 H2D 成本，不新增用于跨核再次认领的元数据或复制 probe 的固定大 arena。

## 5. DFX 与验收

<!-- markdownlint-disable-next-line MD033 -->
<a id="a5-slot-rollout"></a>

### 协作顺序与来源记录的验收

用真实的 descriptor 发布、成员承诺、返回和资源复用建立事件先后。普通图和 native Graph 均按 `run_epoch/task_ref/block/lane` 关联原记录。改变窗口深度时保持实际输入、kernel、活动掩码与等待协议一致。

| 场景 | 必须观察到的结果 |
| ---- | ---------------- |
| Host 顺序 | 现有数据与 Graph 控制依赖组成无环图；验证正向层号、层内 b-level 和稳定 ID 次序，逐边 rank 合法；两个目录无遗漏/重复，非法输入和排序启动前拒绝 |
| 分层无执行屏障 | t9 的前驱完成时，即使同上一层的独立 t11 尚未完成也可执行；不增加全层完成计数或屏障 |
| 静态检查与动态进度 | 关闭前缀表示本类供给进度，不等于 DONE；不能替代另一引擎状态；Host 已完成豁免、Graph 激活及控制门槛不丢失 |
| 前项全部完成任务分发 | A1 尚未被任何核领取时，依赖 A 的 B 不得持票；A0 退休后可补 A1，不领后项 E；A 全部完成任务分发而尚未 DONE 时，B 可以持票但不能执行 |
| 跨引擎取票 | 两类目录各自推进，后继领取前确认跨引擎前驱已全部完成任务分发；无依赖的另一类项无需等待；执行仍等待真实 DONE |
| 宽 SPMD | 4 个核/8 个 block 等已有小拓扑验证：同 task 未退休 block 去重；退休后继续补头部余块；全部完成任务分发才推进后项，不等待全部计算完成 |
| sync 容量 | 同时驻留需求超过总容量时在启动前拒绝；只是暂时忙碌则等待准入，不把两者混为一谈 |
| unready 提前领取 | gate 保持前驱未完成时，后继已持票并进入 WAIT_FANIN，kernel 未进入；释放后恰好执行一次 |
| 已有 pending 接续 | 后续 token 在前一 kernel 返回前已完成 Claim/acquire；返回后不再取新票或 Claim，同入口执行 |
| SPMD 与跨 task 分布 | 同 task 的 current+pending 去重。有效票固定 owner。单 block 退回竞争产生的超额票，多 block 分区保留超额计数并继续探测。快核可继续领取剩余工作 |
| 全窗口 WAIT | 分别填满深度为 1、2、3、4 的自取窗口。已领取单元和 owner 不动，仍能从 SSBUF 推进较早的 MIX，Graph/CHECK 保持进展。最后一个自取位置不能截断尚需本核服务的早序自取义务 |
| MIX 等待 | 已领 block 保留簇归属，早序资源冲突要求全组到齐时核对本代所有活动成员的 ARRIVED，其余组通过本地顺序检查后直接 READY；等待期间继续服务现有可推进的责任 |
| 领取后的唯一执行核 | 同一 execution 从第一次有效票到退休始终是同一 executor；其他核不能执行；错误终止不转成异核重跑 |
| 延后完成 | 等待登记后释放执行位置；真实条件满足后才发布逻辑完成；保留早到、晚到和登记中满足的交错验证 |
| 快路径持续命中 | CHECK/FINISH 与 Graph control 得到有界服务；普通窗口有空位置时继续新票发现，不饿死已有等待责任 |
| M10/S11 颠倒到达 | 显式扣住更早的 MIX 发布，使 V0 先见 S11、V1 先见 M10。两向仲裁保持较早义务可推进；分别覆盖空记录、本簇 bank 占用时推迟领票、无本簇分配、相同 rank 新 generation 及旧 run 边界 |
| Graph 与 predicate | 多实例不串参；ENTER/EXIT 不提前；false 分支合法退休，未发布元数据不被读取 |
| async 与退出 | 条件早到/晚到/登记中满足均不提前 DONE；物理返回释放资源；所有未退休责任阻止成功退出 |

复用上述场景重点检查以下边界，选用显式 gate、计数和执行身份作为判据：

- **代理和成员掩码。** `AIV0+AIV1` 在 AIC 运行无冲突任务时仍能领取并执行。`AIC+AIV1` 核验 AIV0 只发布和最终回收，它在发布后忙碌不延误实际成员的组完成。`AIC+AIV0` 和 `AIC+AIV0+AIV1` 核验实际成员各贡献一次。除真实共享资源外，无冲突候选不读取 MIX 边界。
- **早进入和分散。** 给纯核 `sync_start` 留一个尚未领取的 block，确认先到成员可进入且返后不能再领；其余成员即使早到者在 kernel 内等它们也能进入。对 sync MIX 验证一个组可在末组登记前起跑、N 组分散及本组共享资源已安全交接。超容量拒绝和部分容量暂忙分别覆盖。
- **Graph 和终止。** 两个实例引用同一 Definition，延迟其中一个区段的激活或显式 skip，边界不能越过未决早序任务。跨实例的迟到成员、失败、退出及 SSBUF 复用不得串代。`defer_wait` 未满足时执行位置可复用，逻辑后继直到真实条件满足才执行。

### L3 必须展示真正的前视和串行链

事件统一关联 run/node/block/lane/token/固定 owner，并能还原 Host 层号/rank、各核目录位置及跨引擎前驱供给进度；区分头部未发完、本核去重、initial ticket、Claim/acquire、WAIT_FANIN、最终许可、kernel、完成、MIX/cohort 和 async/control。显示本核原位置等待跨度，但不将等待当作持续 Scalar 工作。

当前有序供给的三个阶段映射到 `scheduler_find_issuable_task`、`scheduler_check_predecessor_supply`、`scheduler_task_supply_closed`。保留这些阶段的供给语义，Host 重排目录时同步其身份和等待原因；不得把等待完成任务分发显示为等待计算完成。atomic、ld_dev、DCCI/DSB 沿现有 primitive 记录展开，缓存命中不伪造共享查询事件，Host 导出和 Python 消费者一起更新。

另按 `(run, node_ref, block, lane, execution_index)` 关联三条原则的证据：静态 Call 发布、本核有效票、固定准备完成、等待对象/条件、最终许可及真实 kernel 开始。当前普通 token 的 `acquired_cycles` 在固定 Binding 准备之后取得，不能将其解释成单条 FAA/CAS 的完成时间。已有 DFX 缺少某个端点时注明不可判定，不用 ready 显示条的边界冒充事件；新增记录只在对应 profiling 模式中采集，不给 L0 添加追踪成本。

目标 DFX 用 `(run_epoch, 实例, rank, block, lane, 来源, generation)` 连接领取、原位准备、MIX 发布、F 发布/观察、首次不可撤销承诺、kernel 进出、贡献与退休。F 的观察必须能对应到已发生的发布；Host 剪掉的检查不伪造设备读事件。统计两种来源的选择、窗口满且另有可分发项的次数、占用峰值、本组会合，以及早序义务未满足的实际原因。核对准备与执行使用同一来源记录。L3 展示达阈值的实际函数及原语，L4 审计因果关系，L0 不写这些跟踪记录。

Build 同时展示准备工作量和它与前驱计算的重叠；领取同时展示同地址参与核数及实际 RMW 次数；Wait 同时展示地址/身份、轮询次数、请求密度和条件满足后的发现延迟。未采集地址或返回值时，L3 只能按函数归因，不能声称已经完成逐地址证明。完成贡献和动态入口仍算在真实交接中，跨核重叠 span 不累加成串行节省。

必须回答：

- 每次进入 kernel 时，已有多少具体 pending？多少已完成 Claim/acquire，多少只是 TICKETED？
- 对前序 A 的余块，本核退休后是否先补 A1？首张后项票是否晚于 A 全体有效票？头部供给等待有多少来自宽 SPMD、本核去重或真实成员准入，Host 排序是否让独立分支更早获得机会？
- defer_wait 的物理返回、条件满足和逻辑完成是否分别记录？
- 返回后的下一次调用来自已有 pending 还是临时补给？若缺少 pending，原因是前序合法票优先、控制门槛、窗口容量、同 task 去重、资源会合，还是没有获得补给机会？
- 参数接收是否早于前驱完成、早于本核前一次 kernel 返回？后继进入时还缺哪些操作？
- `G_worker` 和 `G_dep` 中是否出现新票、重复 Claim、目录扫描？快/慢路径分别占多少？
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
- 上板时，需要在设备上结清的控制节点使用独占的
  二值 DONE 通知。计算任务由 GM 完成门合并计算与异步债务，唯一完成者发布 `st_dev.b32(1)`。
  Graph/inline 控制节点确认全部 fanin 已完成且 canonical 状态合法后，可以幂等 SET 放行后继，随后由 CAS 胜者唯一记账。
  后继非消费式读取，0 为 PENDING，1 为 DONE。只有物理返回而异步条件未清不能发布 DONE。
- CPU/sim 可直接读取 canonical GM completion。A5 真机申请 CntNotify 失败时，Host 立即终止构图。Host 分别记录计划与实际映射覆盖。
- CntNotify ADD 不返回旧值，不能据此领取唯一执行票。当前 A5 后端持有 SDK 提供的 SET/read
  映射并校验 32 B 对齐，再使用同一 slot 的 +0x08 INC 地址；只发送已实测的写 1 命令。
  此硬件合同限于下面记录的芯片和软件栈，不是通用 SDK 地址保证。
  SET 基址必须按 32 B 对齐。地址未对齐或 INC 映射无效时，Host 在发布调度镜像前报错。
- 地址描述符独立于 GM atomic-only 控制行，随 prepared publication 和 capture/replay lease 保留。
  每轮 AICPU 在任何 READY 发布之前清零每条计数链；SET 通知按节点是否已由 Host 完成初始化为
  DONE 或 PENDING。尾部屏障保证复位先于 READY，不依赖 Host 再次 bind。
- 后继第一次绑定前驱时保存完成来源，同次等待不重复查地址表；正向 DONE 证明可按本轮任务身份缓存。
  未就绪即返回原有有界推进循环。完成观察不转移执行所有权，也不替代 kernel 入口、
  Tensor 数据可见性或 MIX/sync_start 的许可与退休检查。
- SRAM 访问不做 DCCI。业务输出发布、异步完成和消费者获取所需的 DSB 保留，
  MIX 成员在自己的成功返回路径写入 CntNotify 的 P；AIV0 在复用或退出时检查本代活动 lane 的 FIN。
- 验证实际 AIC/AIV 访问、重复复位、P/Q 区分和资源回退；性能采用下面的 7 组
  AICore L0 配对标准。启动复位和 Host 准备成本单列，不计作消失的工作。

### 异步条件与控制进度

async 登记的 canonical gate 只保存一个 task 级 `has_async` 门闩，真实 registered/pending 数量保留在现有 join。每个非空组照常执行一次 join.register FAA，并复用其返回值识别首注册者；只有首注册者执行 canonical 校验 CAS。它设置门闩之前不能提交自己的 physical return，所以其他组先返回或完成 CHECK 也不能误关 task。正常 gate 的 debt 为 remaining + has_async，close 校验门闩与合法非零 registered，FINISH CAS 从 `encode(total,0,1)` 转为 DONE；不新增等待 owner 或资源，零登记任务保持原同步路径。DFX 的 completion_debt 表示物理余量加门闩，不能当作剩余异步 group 数。

每个非空组由唯一登记者封口，并由该核检查和发布条件信用。同核 FIFO 的溢出记录有独立缓存行，入队后保证其内容可跨业务调用保存。

每个核对本核登记的组轮转。CntNotify 继续用于后继依赖的完成计数和普通 DONE 通知。

Host 根据折叠控制节点、展开 Graph 后的实际依赖找出全部设备终点。AICPU 在 bootstrap 后依次检查终点的 ADD 阈值、SET 通知或 GM 完成字。所有终点逻辑完成才请求 EXIT，异步条件和独立分支都包含在这个判定中。已完成的终点可以跳过。整轮进度不再占用 CntNotify 槽，正常调度期间也无需按核分区轮询中间任务。

长链可能持续执行而终点尚未变化。看门狗达到半个超时预算后才抽查中间进展。L3/L4 在退休时采集计数链诊断，L4 由最终记录计算 `resolved_task_count`；设备任务所映射的计数器只要有一个缺少有效快照，该值即为 JSON `null`。所有核退休后才能回收资源。

当前组的条件条目由本核登记和封口。检查时读取真实通知地址。CCU 的读清顺序、SDMA 的阈值语义以及消费者获取输入的可见性边界照常生效。

### Deferred completion 登记

Host 为每个参与 kernel 的 block 和 lane 准备完成记录。每条记录最多登记 64 个条件。设备在 kernel 返回时读取本次实际登记。没有登记时提交执行和完成两个信用。有登记时提交执行信用，等实际信号满足后再提交完成信用。一个 1C2V block 的目标是六个信用。

Host 沿展开后的依赖传播输入可见性要求。后继观察到前驱逻辑完成后，仍在实际执行核上获取输入。

### CntNotify 资源复用与控制状态选型

DeviceRunner 按 pipeline slot 和用途保留通知池。只有没有 publication/capture lease
持有的池才可复用；live/captured 使用者占用的池独立保留，不能在其退出前复位给新运行。
每轮 Device 的 READY 前复位保持不变。普通关闭在 context 有效时显式释放映射与handle，
失败保留供重试；确认设备 reset 后只 abandon，析构不调用失效context上的SDK。

控制信号优先迁移到 CntNotify，并按完整状态字的读、写、初始化及复用合同选型：
单写者值/序号用SET；exactly-once贡献的数量汇合可用ADD，不同条件也可用不进位的权重ADD；
需要条件身份和幂等重复时可用BIT_WR掩码置位、BIT_CLR掩码清位。当前实机已验证SET、BIT_WR及同槽偏移访问的ADD、BIT_CLR。
ADD和BIT_CLR的正式直接映射仍被当前SDK/驱动拒绝；接口失败不能用来推断同槽硬件操作不可用。
只有依赖返回值裁定唯一责任的CAS/FAA等继续使用GM atomic；同一锁若仍用GM CAS，
不能单独把它的解锁写搬到SRAM。普通payload、Tensor数据、GM descriptor不属于控制通知替换。
原64-bit状态的范围、失败位和代际语义必须保留，不能截断成32-bit。

AICore以32-bit ld_dev/st_dev访问通知，资源地址决定SET或BIT_WR操作；
AICPU在启动阶段通过映射地址写值并在READY之前DSB。通知地址不做DCCI，
同核存在数据发布顺序依赖时保留相应DSB，跨核通知本身不机械新增屏障。

### 2026-11-17：CntNotify ADD 偏移访问实测依据

本机已确认每个通知占32B槽：`slot = SET_address & ~0x1f`，SET/read为`slot+0x00`、ADD为`slot+0x08`、BIT_WR为`slot+0x10`、BIT_CLR为`slot+0x18`。Host必须先成功取得并保留SET映射，验证返回地址非零、长度至少4B及槽对齐；不能把其他SoC的寄存器跨度直接套入此布局。四种访问均为32位，派生地址不独立申请或释放，生命周期由原通知handle和映射共同保证。

BIT_CLR按`x &= ~mask`工作，已验证单bit清除、保留其他bit、96核并发清除、不同bit并发置/清、重复幂等及同资源重放。独立探针使用真实SyncAll建立相位边界，8场景×16轮×96核×2次重放共24,576次观测，错误、超时及相邻通知污染均为零；证据在`artifacts/cnt-three-20261117/bit-clr/`。这证明所测硬件操作可用，整网性能仍需在实际条件发布路径验证。同一bit的置位与清位仍须由业务协议规定先后，不能把无返回值的操作当作唯一owner裁定。

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
- 已知同步普通返回贡献1；可能异步的普通执行在本次零登记时贡献2，实际发送两次32-bit写1；有defer时P、Q各发送一次写1。每个 MIX lane 在成功返回且输出可见后各自贡献一次 P；可能异步的 lane 在无条件封口或最终 CHECK 时另贡献 Q。不能把提前准备、任务分发结束或仅物理返回当作仍有条件的任务逻辑完成。
- `scheduler_notify_increment` 每次向 SET + 8 写入一次 32 位命令 1。普通执行封口时没有登记异步条件，且该执行已预留条件 credit 时，再写一次。数据发布前的DSB、消费者数据观察、kernel入口与退出的既有屏障保留；通知自身不机械增加逐次或尾DSB。读到完整阈值证明贡献已到达；探针批尾DSB的结果与此协议论证分开，实际无额外通知尾DSB的整网路径另行实测。
- 后继直接读计数器并比较自己的阈值；计数器在一轮中只增不减，旧阈值永久有效，无bitmap、旧快照、run内reset或bank转交。每轮READY前仅由AICPU清零链头一次，publication/capture lease继续保护生命周期。
- canonical GM账本按task一次有界归档，不阻塞fanin ready、固定owner或资源退休。固定归档者可复用本核当前task的真实DONE证明，避免再次读取同一计数器；未命中仍观察阈值，SUPPLY_CLOSED不能代替DONE，error检查和唯一结账/退出统计不变。DFX记录真实P/Q贡献区间、累计阈值和退出计数；不伪造FAA返回ticket，也不以归档时刻代替逻辑ready。
- 由持有的SET映射校验32B slot并派生+0x08地址，派生地址不独立map/unmap；A5 上板申请不足会使 Host 准备失败。GM回退保留近期的自身ACK删除、已有观察证明复用和首次登记者单has_async。Host覆盖日志使用benchmark可见的TIMING级，记录`backend=SET-slot-offset`及计划/实际链数、eligible/mapped任务数。
- 正式验收要求实际非零ADD覆盖，记录完整与部分覆盖；GM回退成功不算ADD通过。继续以 7 组 AICore L0配对总和和0.5%回退门槛判断性能，不从硬件探针直接推导整网收益。

2026-11-17 的历史 ADD 启用验证：12 组均确认eligible计算任务完整映射；11组数值通过，原生MoE沿用仅计时。AICore L0配对总和20.2296068→20.2704963 ms（+0.202%），符合0.5%保留门槛；SWA为0.2577120→0.2486034 ms（−3.534%），新版第19–21层为3.0837522→3.1455712 ms（+2.005%）。该结果不表示各case均加速。完整样本、实际设备、覆盖证据保存在 `artifacts/cnt-add-enable-20261117/RESULTS.md` 与JSON中。

<!-- markdownlint-disable-next-line MD033 -->
<a id="performance-retention-policy"></a>

### 性能与保留规则

每项修改先对应到实测泳道图中的具体串行链，再查 `atomic_probe/ATOMIC_USAGE_GUIDE.md` 与同构探针的硬件证据。明确可减少的共享访存、等待或重复准备及保留的交权边界；不按函数 inclusive 时间直接估算可省开销，也不凭模型名称试参数。

- 当前优化目标先减少同地址 atomic 请求，再减少 atomic 总调用及重复 ld_dev。每项策略比较同一组输入下 7 个 AICore L0 均值的总和及逐项变化：功能正确、明确消减冗余 atomic／ld_dev 且总和回退不超过0.5%时可保留，回退超过0.5%则撤回。计数减少和耗时收益分别提供证据，不能将新增 DCCI/DSB、普通读写或其他地址竞争隐去。percounter 作为定位参照，比较结论须由可比版本、输入及计时边界的实际对照支持。5 µs 交接用于分析关键路径，不能替代 7 组指标。
- 最终性能以 AICore L0 `chip.run.runner_run.aicore_wall` 为准：在无 L1/L3 插桩的运行中，直接取所有实际 AIC/AIV kernel 的最晚 end 减最早 begin，包含首尾之间的调度空洞、依赖和通信等待，排除头尾 AICPU 开销。每个用例在同一设备分配内按 A×5、B×5、B×5、A×5 的顺序对照。A 为基线，B 为候选。每段先预热再计时 5 轮，预热排除在外。报告四段均值，并用每个版本各 10 轮的均值计算变化；无 kernel 的核不参与归并，缺失或无效时间戳不能用 Device wall 代替。HBG、基线及 percounter 必须核实同一首尾边界；percounter 原调度循环窗口不得直接冒充 kernel 首尾窗口。L1/L3 只用于定位。
- 双卡用例每轮先做跨卡 Device 同步，再开始上述 L0 计时。测试启动门必须位于 AICPU 计时入口之前：两卡完成本轮入队后才允许设备会合，随后执行原有 AICPU/AICore 流程；同步门自身等待不计入 AICore L0；只有首末实际 kernel 之间的计算、调度空洞及通信等待计入，AICPU 启停与首末 kernel 之外的准备、退出不计入主指标。单卡不加门。每轮取两个 rank 的本卡耗时最大值，再对十轮求均值；不得跨卡相减硬件计数器或从旧样本减去估计的启动差。基线、候选与 percounter 均须使用同一门；旧的未同步双卡数据另标口径，不能直接作为新口径基线。
- 性能验收使用 `dsv41_scheduler` 当前清单的 7 组配对对照，保持各组原 kernel、输入、图、sync_start 配置一致。6 组回放须通过数值验证；新版原生 MoE 沿用已约定的72/72性能专用模式，明确标记 `numerical_validation:false`，不能把执行成功当作数值通过；全部 7 组（含 MoE 与 layers_19_21 通信场景）的 AICore L0 均值之和决定是否保留，同时披露逐项回退；Graph、MIX、sync_start、defer_wait 的通用功能另按上述合同验证。单个 SWA 或其他模型的收益不能替代全套结果。
- 所有硬件任务使用 `task-submit --device auto --device-num N`，在任务内使用并记录实际 `TASK_DEVICE`；attention 单卡，同一分配内做配对对照，记录先后顺序。
- L1/L3 也区分预热与正式采集；不从单条交接里扣除整次 L1−L0，不累加跨核或嵌套 span 冒充串行耗时。5 µs 判断同时报告低扰动观测、采集开销和未达标原因。
- 对照完整目标方案前后，并保留各独立改动的归因。常规性能改动以综合收益决定保留，1% 收益也记录；已证明消减冗余 atomic／ld_dev 的结构性修改适用上述不超过0.5%回退的保留例外，超过0.5%则撤回；回退如实记录。无 ±5% 忽略区间。有收益的独立修改分别本地 commit，未经允许不 push。
- Host 准备、实际 H2D、GM/SSBUF 和峰值内存单列；不能把工作搬出 L0 区间后不报告。历史测量与详细采集留在实验结果中，不回填本设计作为实施完成状态。

比较容量时固定执行结构。每行列出 W1、W2、W3、W4 四档同卡耗时，报告 7 项总和和各档相对 W4 的变化。单列 `G_worker/G_dep`、block 起跑跨度、实际参与核数/组数和域内进展；首次 kernel 提前不能代替 7 组 L0 均值总和改善。

当前清单使用 `pypto_lib_latest` 的七个原生用例。单卡运行 `swa`、`c2a_full`、`c1a_reindex`、`c1a_reuse`、`engram`，双卡运行 `moe` 和 `layers_19_21`。用例集以 [list_cases.py](../../../../tests/fdwic/dsv41_scheduler/production/list_cases.py) 的输出为准。
