# CSA 全核计算空窗优化：证据与实验

日期：2026-10-28。用户已将目标从 2ms 修订为原样 CSA、无泳道图的完整 Device 中位数低于 1.5ms，
最终按[实施计划](csa-sub2ms-optimization-plan.md)做五批与完整正确性验收；当前尚未达成。
不改 54 个业务 kernel、七项 golden、2308 个执行身份、三 lane GM 合同或计时范围。

## 新基线的空窗

基线是 [Ready 记录修复](csa-ready-profiling-20261028.md)后的工作树，
相对 `ded082f20f` 的生产 diff 为 `6cdda7d76f25be61e22419720ba7f223736f3664e80f873444cd4c3da8a53612`。
同卡单批 5 次预热加 100 次正式样本的完整 Device 中位数为 2.3883845ms。
原始 L1/L4 保存在工作区 `reports/csa_poursoul_20261027.Osnaam/artifacts/readytrace-after-l1/`
和 `readytrace-after-l4/`，从仓库根目录通过 `../reports/` 访问。

用现有 `read_perf_data` 解析真实 kernel 起止时间，合并全部核的执行区间，再取补集。
范围裁到首个 kernel 开始和最后一个 kernel 返回，不含启动与退出。

| 指标 | L1 | L4 |
| ---- | -- | -- |
| kernel 任务窗口 | 2229.120us | 2799.637us |
| 无业务 kernel 的间隙 | 84 段／673.891us | 58 段／982.958us |
| 跨 task 间隙 | 48 段／626.174us | 40 段／877.112us |
| 同 task 间隙 | 36 段／47.717us | 18 段／105.846us |
| 最大 15 段总和 | 465.785us | 649.614us |

L1 最大间隙是 t56→首个投影任务 58.979us，随后是 t46→后继 41.436us；
多个串行链转换在 23–33us。边界任务编号只代表区间两端，不自动证明直接依赖。
L4 的 96 个 Scheduler Idle 区间交集为零，最多同时 95 个：这里存在真实的全核计算空窗，
但仍有少数 Scalar 在 Resolve、准入、领取或发布，不是所有调度者都无事可做。

两个值得直接试验的因果线索：

- t24 的首块在 1010.598us 已完成 payload 发布，同一 Scalar 又去领取、准备第二块，
  第二块到 1023.992us 才发布，首块 entry 为 1028.653us。t56 也有同样顺序。
  这些是 L4 原生事件，不把其差值全算成 level0 可省时间。
- t56 的最后 kernel 返回为 1998.425us，逻辑 DONE 为 2004.070us；
  Resolve 遍历 17 条后继依赖，其中 16 路投影可以就绪；本次耗时 35.115us，
  首个后继 kernel 到 2071.796us 才开始。
  这是剩余真实控制工作及插桩，不再是已删掉的 1024 次 Ready 字段写入。

## 外部 review 的取舍

参考 [poursoul 对 ded082f20f 的 comment](https://github.com/nalinaly/simpler/commit/ded082f20f#commitcomment-206984781)，
逐项与当前代码和测试合同对照：

- 扫描期成本值得优化，但已有 `current_task_ref` 早退和每轮拓扑缓存；正常不是每次 poll
  重验同一记录。CSA 有 61 条协同记录，首次扫描量约为 96×61，而非用 79 个 kernel task 计算。
  reader 拒绝发布后坏记录仍是现行测试合同，不能把完整校验只留在 debug。
- Mix 的 block 局部三 lane 资源域与 task 候选成员集合不是一个概念。当前任何 cluster 都可能
  领取，成员必须覆盖其所有 lane；文档已澄清。没有授权改变三 lane，本轮不做成员收窄或 TPipe 资源域变更。
- 全局序分片需要重新解决跨类型 Mix 的同序与 order_key 身份，本轮不作为第一项改动。
- 现有 L4 原生协议已在 kernel 调用前发布执行身份和 entry 时间，先复用该证据，不另加重复热路径记录。
  非返回 kernel 的超时、现场保留和恢复仍须专门验收，不能以有记录代替完整诊断结论。
- 当前 bootstrap 和主循环的 Mix victim 起点都默认成零，不只是 bootstrap；作为独立初始化问题跟进。
- Host 分布单独记录，不把漂移简单视为已解释的噪声，也不将 Host 耗时替代 Device 目标。
- 读／写类型区分可作为后续规范改进，但示意 wrapper 返回同样的整数并不能单独阻止错误判定；
  允许同一类型同时做 bypass 与原子写也不能证明 cache line 隔离。本轮不为性能目标做大范围类型迁移。

## 按证据推进的候选

1. **准入记录观察合并，保留完整校验。** 从已经验证的布局直接确定 bitmap 紧邻 header 的地址，
   同时 invalidate 两条 line 后做一次 barrier；扫描只调用一次完整 record validator。
   坏 offset 仍拒绝，不能拿 record 内未验证的指针做 cache 操作。没有 ABI 或协议变更。
2. **纯 C/V 首个 READY 块优先执行。** discovery、completion 和入口门禁仍照常服务，
   仅避免为同一协同 task 填第二槽而阻塞已有可执行块。不改 Mix、不增在飞额度、不缓存许可。
3. **同一有界动作复用已确认的耗尽结果。** 不把 raw 尾票或 `ticket_exhausted` 当有效 cursor=N；
   独立 close、跨调用、sync_start 部分 cursor 和错误入口保留强确认。
4. **更大改动后置。** 若上述收益不足，再评估领取确认信号或 cluster close-seal 隔离。
   当前 seal 为密集 uint64_t，可能争用但尚无独立归因；布局改变必须同步 Host/reader 并显式保留历史语义。

旧的 ACK 全量预筛、仅布局 admission_state 隔离和 GM seq/ACK/seal 单写者实验已有负结果，
不在没有新证据时重复。每个候选单独重建和测量，不以预计节省代替实测。

### 准入记录观察：保留小幅等价简化

新增三个 C++ 保护测试，覆盖第二 bitmap word、成员／非成员、冷／暖拓扑和极端非法 offset；
基线 112 个 admission 用例通过。候选的 admission／ordered_execution／gm_mix 三个 C++ 目标通过。
这属于等价优化，保护测试在基线也应通过，不伪称业务 RED。
两平台各 29 项原生两行可见性探针、285 项 Python oracle、严格 CSA L4 全部通过。
L4 保留 2308 个执行身份、133 条依赖和 46733 个协议事件，七项 golden 通过。

按 A/B/A/B 串行复测，每批三轮七项 golden、5 次预热加 100 次正式样本：

| 批次 | Device 中位数（us） |
| ---- | ------------------- |
| A1：Ready 修复基线 | 2388.3845 |
| B1：合并观察 | 2366.9705 |
| A2：恢复基线 | 2405.8580 |
| B2：恢复候选 | 2381.8930 |

每臂 200 样本，合并中位数 2396.518→2374.5355us，改善约 0.92%。
两对分别减少约 21.414us 和 23.965us；属于小幅收益，不能替代更大的跨任务空窗优化。
没有扩大控制区或减少校验，保留该简化。四批 kernel 指纹相同、CLI 竞争观察为空。
生产 diff 为 `d18851835eb3db6900ab5c2340d194b8a0977268d95e6eb2dcd2341a7b50cdc0`；
测量摘要、每批身份和原始样本归档校验和见[扫描对照摘要](perf_data/2026-10-28-idle/scan-summary.json)。

### 首个 READY 优先：保留

仅在纯 C/V 协同任务的填槽循环增加 `local_ready_mask == 0` 条件，已有准备好的块时不继续取块；
discovery、completion、关闭检查及执行入口重新授权保持原路径，普通非协同任务与 Mix 不改变。
新增 8 个 C/V 参数化测试，在旧策略上全部失败；改变策略后通过。
旧的双槽预填测试驱动改为在两次实际执行间调用 fill，依赖、唯一完成、predicate、身份和 LocalDone
断言保留；另调整一项 inline 服务测试，使它仍验证两块都完成后才释放后继。
相关四个 C++ 目标通过。两平台各 44 个普通有序场景和 47 个 sync_start 场景通过；
pytest 显示每平台两个 scene class 通过，其 test_run 遍历全部匹配 CASES，不是只跑两个 shape。
严格 CSA L4 保留全部 2308 个执行身份、133 条依赖和 46733 个协议事件，七项 golden 通过。

A/B/A/B 每臂 200 个正式样本：扫描基线合并中位数 2382.2805us，候选 2267.266us，改善约 4.83%。
四批中位数依次为 2381.893、2273.1985、2383.177、2259.8695us。
生产 diff 为 `51178495e4f769d65a5b73128dad421445cc8533fa2345ba0c4d2d3b5246d57a`，
完整来源见[首块优先对照摘要](perf_data/2026-10-28-idle/first-ready-summary.json)。

L4 中 t24/t56 每 worker 首块 publication→entry 中位数分别从 18.4225/18.5675us
降到 4.635/4.520us，证实命中“首块等待第二次领取”的路径。
L4 全局计算空窗从 1034.709 降到 913.955us；新 L1 窗口为 2084.570us，空窗为 600.907us。
这些是独立捕获，不能按插桩差值精确拆算无泳道图收益。t24/t56 的 L4 自身跨度略增，
因此不把首块提前等同于整 task 同幅缩短。t56 扇出 Resolve 仍约 35.8us，尚未解决。

### Mix扇出早发：让后继提前离开全核计算空窗

按最新要求，先处理全体业务 kernel 都未执行的区间，再处理其他热路径。
首块优先后的 L1 中，t56 到首个投影仍有 57.159us 空窗；L4 对应为 71.893us，
其中 Resolve 为 35.838us。t46 的八路扇出也是同类候选，其 L4 Resolve 为 14.309us。

冻结图中，t56 的 16 路投影各有三个实际前驱：t56、Host 创建者 t2、各自的 alloc 节点。
第 17 个后继 t105 有 19 个前驱，此时还必须继续等待 pack 等节点。
不能把业务源码中的一条显式依赖误当成冻结图的全部依赖，也不能将 t105 提前发布。

早发前的实现先遍历整个已摘下的等待链，按类型在本地构建 Ready 链，最后统一入队。
本轮验证的最小改动是：仅对 Mix，在至少两个后继已真正就绪、且等待链仍有剩余时，
提前发布一次前缀；剩余仍批量发布。前缀与后缀进入同一个 coordinator inbox，
已发布链不再由 resolver 改写，且禁止后缀重新走 direct Mix 旁路而越过前缀。
不泛化纯 AIV：其入队目标由 batch 首节点的奇偶决定，拆批可能改变目标队列。

验证重点为旧队列 FIFO、前缀消费与后缀发布并发、剩余依赖迁移、direct 回退、
前缀前／后的错误终止以及完整执行身份。无新增生产 hook 或开关。
收益必须同时在 L1/L4 空窗位置与无插桩完整 Device 上验证，尚未测量前不计入已保留收益。

新增5项Ready测试在旧策略上4失败、1通过；实施后Ready、admission、ordered_execution、gm_mix、
inline_services五个完整C++目标通过，并发／迁移等测试重复50轮通过。
另补充坏fanin与真实三lane入口门禁测试：先合法准备READY，再在发布Mix前缀后遇到坏后继，
已领取前缀不能继续准入，先前合法获得过的kernel许可也不能重用；首次错误身份保持。
这里不承诺整次Resolve具有事务性：错误被发现**之前**，已发布的合法前缀可以执行；
发现错误后停止新准入与kernel入口，已发布前缀不回滚，未发布后缀不再发布。

同布局、仅增加早发的严格L4初验（`idle-close128-after-l4` → `idle-mixprefix-after-l4`）：

| 指标 | 无早发 | 早发 |
| ---- | ------ | ---- |
| t56后首个kernel前的全核计算空窗 | 74.505us | 39.643us |
| t56末kernel到全部16个pack末kernel | 468.071us | 436.501us |
| 首投影到全部pack末kernel | 393.566us | 396.858us |
| 全局无kernel间隙总和 | 942.806us | 899.801us |
| 首kernel到末kernel窗口 | 2729.811us | 2688.731us |

新捕获的首准入1973.171us、首kernel1994.193us，均早于Resolve结束1996.458us；
Resolve结束前已有两个投影准入、七次kernel调用启动，验证了预期重叠。
前移并非全部转移到尾部，但从首投影算起的尾部跨度略增3.292us，也一并保留。
两份记录均通过严格协议、2308执行身份、46733协议事件和133条依赖审计，54个kernel指纹一致。
这是单次插桩捕获的对照，不替代无插桩性能验收。

同布局无插桩A/B/A/B已完成，每批3轮×7项golden、5预热+100正式样本：

| 批次 | Device中位数（us） | Host中位数（us） |
| ---- | ------------------ | ---------------- |
| A1 `idle-close128-after-a` | 2248.3570 | 78495.7465 |
| B1 `idle-mixprefix-after-a` | 2224.5775 | 78313.5670 |
| A2 `idle-mixprefix-before-a` | 2245.7985 | 87607.4760 |
| B2 `idle-mixprefix-after-b` | 2229.9340 | 78318.3380 |

每臂200样本合并Device中位数2246.699→2227.8845us，减少18.8145us，约0.84%。
两对分别减少23.7795us与15.8645us；54个链接后kernel指纹一致，所有golden通过，
CLI竞争观察均为空。Host的A2有明显漂移，最大470093.886us，不宣称稳定Host收益。
此结论是在128B实验布局上的独立prefix对照；未将布局本身的小幅差异算成已确认收益。
四批身份、分位数及当前8B验证入口见[Mix前缀对照摘要](perf_data/2026-10-28-idle/mix-prefix-summary.json)。

新L1（`idle-mixprefix-after-l1`）窗口2065.166us、全局计算空窗569.111us，
其中t56→首投影26.241us。对照旧first-ready L1的布局不同，只作为趋势，不单独归因prefix。
源码提交后，边界测试曾因Python扩展仍带旧commit版本戳在collection阶段被拒绝；
该次未运行设备，不计为通过；扩展重编后，空Ready复用版本已完成下文记录的真机与仿真回归。

### 提前确认当前Mix组完成：负收益，已撤回

排除t46/t56大扇出后，first-ready的L1最大十段全局计算空窗仍合计260.775us。
L4中26个Mix task最后贡献组的阶段中位数为：最后active返回到最后资源lane release
4.449us、全三lane release到C观察确认6.342us、确认到block完成贡献1.216us。
这里只把原生事件当定位证据，不把包含插桩的时间直接算成无插桩可省时间。

该实验基线的fill先做新候选发现，再处理当前已RELEASED的Mix组；确认完成晚，也推迟后继Ready。
当时的候选保持已有pending发布优先，随后提前尝试当前C协调者的退休，再正常执行discovery：

- 复用frontier和ordinary槽停放检查，坏状态不能先贡献完成数。
- 三lane ACK仍走原有query和强确认；TASK_LAST仍由原完成路径唯一发布DONE/Resolve。
- 提前路径只退休，不提前close、claim或发布下一组；discovery的两份credit预算保留。
- WAIT只在本轮首次后段服务跳过重复退休，下一轮重新确认，不缓存跨轮许可。
- 尚未发现的独立坏候选可能晚于当前有效block退休；错误一经发布，后续贡献、领取和入口仍停止。

24项新增参数化测试覆盖四种Mix mask。在原顺序下16项确定性失败、8项保护通过，
包括同轮后继准入、旧队列FIFO、满credit回退、frontier/park/错误、迟到ACK和不提前领取下一组。
重复WAIT的无副作用调用次数没有现成统计或hook，不伪称动态覆盖，另以控制流审查保证。
候选实现时，新增24项测试和六个关联C++目标全部通过。
四项旧测试的“同轮后继不能READY”曾替换为READY必须已ADMITTED、
cooperative、block/payload身份正确，原有完整exactly-once循环保留；撤回候选时旧断言一并恢复。

恢复8B/v5后的独立基线为 `idle-prefix8-before-retire-l4` 与
`idle-prefix8-before-retire-a`，完整Device单批2250.447us。其L4全局空窗921.875us，
最后贡献组全release→C观察中位6.267us，全task最后raw kernel返回→DONE中位12.512us。
这份prefix基线中还出现t79已全release后、C确认前提交t70准入的直接交错，
故不沿用更早记录中“该区间没有成功commit”的观察来概括所有版本。
退休候选 `idle-retire-first-after-l4` 通过严格v5身份/依赖审计，但完整Device为2380.2695us，
恢复基线 `idle-retire-first-before-a` 为2251.311us，候选反而慢约5.73%。
L4的最后贡献组全release→C确认中位降到4.435us，却把同task退休→下一claim中位
从1.657us拉长到5.433us；同task内全核计算空窗增加70.931us。
因此撤回生产和绑定该顺序的测试，完整补丁与测试日志保留；
详见[提前退休负结果](../../../../docs/zh-cn/investigations/2026-10-csa-early-mix-retirement.md)。

### 空Ready结果复用：保留

不再移动退休与后续领取的相对顺序。现有poll已经逐个读取适用的own inbox和对应偷取目录；
若它们都为空，无须在同一fill再次进入discovery反复清目录bit、barrier和重读空inbox。
该候选没有新增GM判空，只把完整负探测结果作为本轮栈上的布尔输出。

只有无pending、无candidate、无local READY、全部适用队列确认为空且配置合法时才复用。
坏scheduler_count/index等情况仍走原领取校验；提示不授予ownership或执行许可，不改变任何共享状态。
新的Ready并发发布可以延后到下一轮重新探测，每个fill重新读own inbox，不能仅依赖目录或缓存跨轮结果。
当前Mix服务顺序、三lane强ACK、retire→next-claim/publish连续性及依赖门禁保持不变。

新增接口用“同接口、始终输出false”的保守适配做可运行对照；11项测试中8项hint契约失败、
3项保护通过，业务推进断言均通过。这不是旧原API的业务RED，也不拿成功pop计数伪造空扫描次数。
最终实现的11项测试和六个关联C++目标全部通过。A5 HBG examples、tests/st及FDWIC边界范围
88通过、4跳过、4取消选择；A5sim同范围67通过、29取消选择。没有将这些计数冒充完整故障矩阵。
进一步的21个C++目标与3479项Python协议、身份、因果、时钟、early-enqueue及转换测试全部通过。

同卡A/B/A/B、每臂200个正式样本，完整Device合并中位数2245.264→2083.5975us，
改善约7.20%。四批中位数依次2251.311、2084.9525、2239.7165、2080.9475us。
每批三轮七项golden通过，54个链接后kernel指纹一致，CLI竞争观察为空；
Host批次漂移明显，不声称Host收益。批次身份、分位数和结果校验和见
[空Ready复用摘要](perf_data/2026-10-28-idle/ready-probe-summary.json)。

同布局L4的全局空窗921.875→833.188us，窗口2711.793→2506.685us；
同task退休→下一claim中位1.6575→1.730us，未出现提前退休实验的周转停顿，
t23同cluster相邻claim周期40.534→34.9255us，C/V入口偏斜2.886→0.835us。
新L1（`idle-ready-probe-after-l1`）窗口1909.368us、全局计算空窗526.240us。
该候选有可保留的实测收益，但完整Device仍高于1.5ms，不能结项。

### 耗尽关闭预筛：撤回

在纯 C/V 无 READY、没有 sync_claimed/ticket_exhausted 且本地形状／类型有效时，
query≥N 只触发原有强 close；只有其 FAA0 与全部检查通过才推进。未关闭则回原 claim 路径。
新增 8 项等价保护测试在基线通过，候选四个相关 C++ 目标、两平台各 91 个边界和严格 CSA L4 通过。
候选单批完整 Device 中位数 2284.839us，随后恢复基线为 2262.5875us，未见收益，撤回生产分支。
保留耗尽安全测试，不新增关闭证明对象、跨轮缓存、字段、ABI 或原生协议版本。
详见[撤回记录](../../../../docs/zh-cn/investigations/2026-10-csa-close-prefilter.md)。

### cluster close-seal 地址隔离：初验后归档，暂不保留

原有每 task 的 cluster seal 连续按8B排列，32组只占256B。该候选仅将已有每个 seal 的
存储单元隔离到128B，保留全部FAA0/store/barrier、三lane与成员／关闭合同，不新增领取完成信号。
固定CSA有26个Mix，832个seal，实际planner测试核对总状态增量为99,840B。
涉及Host/Device布局和原生记录，显式v6提供必填close_seal_stride=128，旧v1–v5仍严格按8B审计。
相关21个C++目标、3707项Python测试与历史v1–v5真实记录通过；新增v6严格CSA L4通过。
原始记录为 `artifacts/idle-close128-after-l4/`，单批完整Device为2248.357us，
在 `artifacts/idle-close128-after-a/`。这不是交替复测结论，暂不计入已保留收益。
其L4全局无kernel空窗942.806us，未比首块优先记录的913.955us更短，
所以不能用单批Device的小幅差值宣称已解决全核空窗；主线转向上述扇出早发实验。

该备选的生产、Python、C++测试改动已分别完整保存在工作区
`reports/csa_poursoul_20261027.Osnaam/close128-{production,python,cpp}-candidate.patch`，
并从工作树撤回。当前仍使用8B seal与v5协议；前面的v6测试和测量只描述该已归档候选。
撤回不影响Ready记录修复、扫描合并、首块优先及prefix代码/测试。
接下来的prefix和退休顺序实验需在恢复的8B布局上重新验证，不冒用v6捕获作为当前二进制证据。
