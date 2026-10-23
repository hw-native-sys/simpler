# A5 HBG SPMD 边界条件和验证

[返回设计总览](../a5-hbg-spmd-design.md)；协议见 [运行时有序准入](ordered-execution-protocol.md)
和 [block 与 Mix 执行](block-and-mix-execution.md)。
功能验收同时证明：执行身份不重不漏、参与者能够推进、非参与者不被阻塞、
没有额外的全任务完成屏障。golden 通过或总执行数相同都不能单独证明这些性质。

可执行入口、矩阵覆盖边界和本次验证状态见 [测试入口和验收状态](test-plan-and-status.md)；
字段与接线设计见 [编码契约](implementation-contract.md)。不要把下表的待实现项当成已经验证。

## 必须保留的正反例

### 普通任务先验收跨 task 重叠

普通 A 有两个 block，分别由两核不可撤销地领取；B 与 A 没有数据依赖。
一个 A block 快返回，另一个故意延迟，快核关闭本核 A 后应能进入 B。

~~~text
核 0：领取 A.block0 → 完成并发布 → 关闭 A → B 开始
核 1：领取 A.block1 → 继续执行 ─────────────────→ A 返回
~~~

断言至少存在 B 的 kernel_start 早于 A 的最后一个 kernel_end。
增加真实 A→B 依赖后，B 必须等 A 整体完成；这区分顺序门槛与数据依赖。
该测试属于普通多 block 阶段，不应等 sync_start 实现后才做。

sync_start 阶段追加同类测试：A、B 都在 kernel 内等待各自参与者，
A 全部进入后故意延迟一个 block 的尾部。快核可进入 B 等待，慢核先完成 A 再进入 B。

### ready 的 B 可以先于 A

A 提交较早但等待 P，独立 B 已 ready。要求找到 B 先获准执行并进入 kernel、
A 此时尚未 ready 的记录。随后 A 获得更后 order_key，各核遵守同一结果。
不能用各核自己观察 ready 的顺序代替共同发布，也不能要求 task_ref 单调执行。

### 非参与者可以越过 A

A 是纯 AIC 协同任务，B 是无数据／资源依赖的 AIV 工作。
AIV 应可以在 A 执行期间运行 B。另测部分交叠的成员集合：
只有相交的成员按共同顺序推进，不能用全局最慢核前沿挡住所有人。

### 必须能识别的错误协议

- 各核任意挑 A/B，或相交成员建立 A→B、B→A 的相反顺序：必须找到循环等待反例。
- 两核承载三个相互等待的 block：必须在启动前拒绝；两个 pending slots 不是两倍容量。
- 普通 A 有六个 block，本核只做一个便进入 B，其他未领取 block 无人处理：检查必须失败。
- C 进入 A.block0、同组 V 进入 A.block1：身份检查失败，不能只比较 task_ref。
- 已预填的后序普通 slot 或 direct refill 跳过必需 Mix：即使最终数值正确也违反准入合同。

## 边界条件矩阵

普通多 block／Mix 和原样 CSA 先验收；C23、C24、C55 的共同驻留语义在后续
sync_start 阶段验收，前阶段保留未支持形状的拒绝／回退检查。
多 block predicate、Graph、异步和工作转移未开放时验证明确拒绝，不宣称已支持。

### 形状与执行覆盖

| 编号 | 构造 | 必须成立的结果 |
| ---- | ---- | -------------- |
| C01 | block 数为 1、2、3、7 | 逻辑编号恰好执行一次，block_num 始终为原值 |
| C02 | block 数为 P−1、P、P+1、2P+1、512 | 普通任务正确分波，不被全部 block 同时占核的 demand 检查挡住 |
| C03 | block 数为 0、负值、超过接口范围 | Host 转换和乘积检查拒绝，不进入 kernel |
| C04 | N 乘 active_subtasks 超过接口或存储上限 | 不因字段窄化变成合法小值 |
| C05 | 纯 AIC 与纯 AIV | 使用各自类型容量，Vector block 不误配为 cluster 编号 |
| C06 | mask 3、5、6、7 | 子槽、函数、物理 lane 正确；mask 6 的协调 AIC 不算 kernel |
| C07 | N 小于成员数 | 零工作成员确认领取耗尽后关闭；真正非成员直接跳过 |
| C08 | 领取记录或共同驻留映射遗漏、重复、交叠 | 对应协议／入口检查失败，不靠总完成数掩盖 |
| C09 | 漏一个 block，另一个重复执行 | 执行身份集合检查失败，即使总数和某些输出相同 |
| C10 | 发布的成员与 discovery 不符或缺核 | kernel 放行前失败，不在线缩减已发布成员 |
| C11 | 不连续物理 core ID | 不以物理 ID 直接索引紧凑 worker 表 |
| C12 | tensor view、共享 workspace、合法 reduction | 保留原 ABI，不重复加 block 偏移或擅自私有化数据 |

### 顺序与依赖

| 编号 | 构造 | 必须成立的结果 |
| ---- | ---- | -------------- |
| C13 | 独立 A/B，A 一个 block 慢返回 | 快核关闭自己的 A 后进入 B，不等待 TaskDone(A) |
| C14 | 同负载增加 A→B 真依赖 | B 不早于 A 成功终态及必要数据发布 |
| C15 | A/B ready 对不同核可见顺序相反 | 按统一发布的 order_key 执行，不按观察先后自定顺序 |
| C16 | 两 task 成员集合仅部分相交 | 相交成员同序；无义务核不受全局前沿门槛约束 |
| C17 | 链、菱形、多 fanin、跨类型依赖 | 沿完成驱动路由正确推进，每个 task 只释放一次依赖 |
| C18 | 一个任务释放大量后继 | 不丢唤醒、不重复发布，不饿死本地或远端工作 |
| C19 | 在途普通 kernel 尚占用 Mix 相关资源 | 先完成必要资源收尾，不因其无 SPMD 标志就覆盖 |
| C20 | 所有 block 已领取，最后一个仍很慢 | 分配封闭不等于 TaskDone，依赖消费者不得提前读结果 |
| C21 | predicate true、false 或求值失败 | 已支持形状保持语义；未支持组合明确拒绝，不能各 lane 分歧 |
| C22 | Dummy 有前驱、Host inline Alloc、空图 | 正确推进服务终态，不伪造 kernel 执行，不提前退出 |

### 协同资源与生命周期

| 编号 | 构造 | 必须成立的结果 |
| ---- | ---- | -------------- |
| C23 | 纯 C/V sync_start，早晚到达混合 | 早到核可开始，晚到核不被后序相交任务截走 |
| C24 | 共同驻留需求等于容量或超过容量 | 前者可推进，后者启动前拒绝 |
| C25 | Mix 的 C、V 参数准备和返回速度不同 | 同 block 配对，各核自备 context，资源合法复用 |
| C26 | 两个 cluster，一个 A 组慢返回 | 快 cluster 关闭本地 A 后可进入 B，不等远端 A |
| C27 | 非执行 lane 的旧 kernel 占用相关资源 | 新 Mix 不覆盖旧资源，资源义务不计作 kernel |
| C28 | 连续 Mix block 重用硬件事件和缓冲 | 上一 block 通信收尾不污染下一 block |
| C29 | 旧 run_epoch 或 slot_generation 的到达／完成 | 不作为本轮事件，不释放新工作 |
| C30 | 序号、代次接近上限或完成数异常增长 | 明确拒绝／安全重建，不静默回绕 |
| C31 | block、发布者或组协调者失败 | 运行失败，不补成功计数、不放行依赖后继 |
| C32 | kernel 阻塞，其他核暂无任务 | 报告真实未完成身份，不提前回收 GM／参数 |
| C33 | 重复执行同图、不同 N、不同 shape | 初始化及缓存视图完整，不吃上一轮状态 |
| C34 | 未支持的 Graph 或异步 kernel | 明确拒绝／回退；resident 验收判失败 |
| C35 | 某核参数 slot 复用于不同 block | kernel 看到自己的 context，不被远端或后续覆写 |
| C36 | 输出可见性与完成标志交错 | 逐条满足发布链，不能仅凭 counter 到阈值读取 |
| C37 | 动态转移给已关闭 A 的核 | 拒绝转移，不能破坏 LocalDone |
| C38 | 同一 task 重新入队或队头出现旧身份 | 检测节点生命周期／代次错误，不能沿用单次入队的 ABA 假设 |

### 运行时准入与新代码接入

| 编号 | 构造 | 必须成立的结果 |
| ---- | ---- | -------------- |
| C39 | A 等 P，B 已 ready 且独立 | B 可先发布并执行；未 ready 的 A 不占据阻塞序号 |
| C40 | 多核同时申请同一／不同 ready task | 同 task 只发布一次，不同 task 的顺序对所有成员一致 |
| C41 | 发布者暂停在记录准备与尾部发布之间 | 不读半记录、不越过空洞；错误走失败协议 |
| C42 | A 仍在执行时 B 已 ready | 允许发布 B，不把准入锁持有到 A 完成 |
| C43 | 纯 AIC 的 A 与无关 AIV 的 B | B 可在 A 结束前执行，不添加无关 AIV 成员 |
| C44 | direct refill 和两个 slot 已填后序候选 | 保存候选但禁止越序，能让出较早义务所需的临时资源 |
| C45 | 普通队列持续有工作，Mix lane 已被承诺 | 必需 Mix 有进展，不能无限优先执行普通 slot |
| C46 | 一个核已跳过 A，再向 A 增加该成员 | 拒绝成员扩张，不接受迟到参与义务 |
| C47 | cursor 到 N，某领取者尚在参数准备 | 唯一归属不可撤销时快核可关闭；存在转交事务时不能只看 cursor |
| C48 | Mix AIC 已领取但暂停在发布组身份之前 | AIV 不因暂时无 dispatch 就关闭 A，也不错误进入后序组 |
| C49 | cluster 关闭标志与最后组发布交错 | 关闭携带最后序号，AIV 不漏组，旧记录不被覆盖 |
| C50 | 日志和队列暂空，前驱仍在运行 | worker 继续服务，不能当作 EOF 或成功退出 |
| C51 | 运行模式 latch 残留或 hook 未接线 | 本轮发布计数必须增加，不能凭旧值通过来源门禁 |
| C52 | 首阶段普通多 block 与仍未支持形状混合 | 按当阶段能力矩阵验证 resident／拒绝／回退，不盲改全部 marker |
| C53 | 调度层没有可用 SSBUF 通道 | 目标 GM 路径仍可初始化、执行和退出，不残留空指针门禁 |
| C54 | 准入临界区持有者准备进入 kernel／等待成员 | 合同检查阻止；不得把短提交锁变成全任务锁 |
| C55 | sync_start 的成员仍在较早 task，部分成员先空闲 | 允许固定未来位置并先开始，不要求所有核先空闲或先 ARRIVED |
| C56 | mask 6 的 AIC 无 kernel，但尚有组协调义务 | 协调者不得提前关闭或进入后序阻塞 kernel，不增加执行数 |

## 验证实现要求

### 协议与现有回归

使用 Host/CPU 可控交错测试完成路由、统一准入、成员投影、唯一领取、局部结束和等待关系。
保留错误协议的反例，不能只有顺利串行执行；测试同时检查预期和实际执行身份集合。
资源等待、组协调、服务节点单独记录，不能混入 kernel 次数。
不为测试新增生产环境变量或宏，优先复用 fixture 和模拟原语。

- [scheduler 合同测试](../../../../tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_scheduler_contracts.cpp)
- [dispatch、refill 与 generation 测试](../../../../tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_scheduler_dispatch.cpp)
- [ready 和 wake 测试](../../../../tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_scheduler_ready.cpp)
- [单 block Mix 与 rendezvous](../../../../tests/st/a5/host_build_graph/single_block_mix/test_single_block_mix.py)
- [fanout imbalance](../../../../tests/st/a5/host_build_graph/fanout_imbalance/test_fanout_imbalance.py)
- [现有多 block legacy 门禁](../../../../tests/st/a5/host_build_graph/spmd_legacy_fallback/test_spmd_legacy_fallback.py)

新增普通 SPMD 支持后，最后一个用例的“多 block 必须 legacy”预期要按支持范围更新；
同时保留尚未支持的 sync_start／Graph 回退看护，不能只删掉失败的来源断言。

### 调度来源与设备验证

复用 Worker.scheduler_runtime_mode 和 scheduler_mode_publish_count，
检查本轮发生新发布，且选择 resident。完整执行验收还要核对 trace 的 aicore producer。

现有自动 fixture 位于 tests/st/a5/host_build_graph/conftest.py；
tests/fdwic 和 examples 下的独立入口不会自动继承它。应接入同一 hook 合同，
不能只加一个 marker 就认为门禁生效，也不能只依赖 pytest 而遗漏 standalone。
CSA 继续保留 --require-aicore-scheduler，不另造同义配置。

CPU 模拟用于协议与 golden，不用于推断设备耗时。真机重点验证 GM 发布、
缓存可见性、实际物理配对和 kernel 收尾；超时不自动等同于已证明死锁。
本机可直接使用设备，缺少 npu-smi／task-submit 不等于无卡；
无独占隔离时记录条件，不把被其他进程干扰的数据当作稳定基线。

可迁移已有 kernel 与输入，但不能把 legacy 通过记录算作新 resident 已验证：

- [纯 AIV 多 block](../../../../tests/st/a5/tensormap_and_ringbuffer/spmd_multiblock_aiv/test_spmd_multiblock_aiv.py)
- [Mix 多 block](../../../../tests/st/a5/tensormap_and_ringbuffer/spmd_multiblock_mix/test_spmd_multiblock_mix.py)
- [sync_start](../../../../tests/st/a5/tensormap_and_ringbuffer/spmd_sync_start/test_spmd_sync_start.py)
- [饥饿场景](../../../../tests/st/a5/tensormap_and_ringbuffer/spmd_starvation/test_spmd_starvation.py)
- [HBG SPMD PA](../../../../tests/st/a5/host_build_graph/spmd_paged_attention/test_spmd_paged_attention.py)

### 原样 CSA 是普通多 block 阶段的出口

使用 [冻结 CSA 用例](../../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md)
的原输入、原 54 个注册 kernel、原 block 数及七项 golden。
固定样本预期 2308 个核执行身份：纯 AIC 31、纯 AIV 611、Mix 1666；
79 个有 kernel 的逻辑任务之外还有 2 个无 kernel 依赖节点。
实际验收按本轮图推导集合，不硬编码这些数量，也不混用逻辑任务／block／子槽分母。

分别采集不带完整插桩的功能／性能结果和完整身份的诊断记录。
trace 缺失、截断或来源无法证明时，相应检查失败，不能静默 skip。
不得通过减小 N、替换通信 kernel 或回退 AICPU 来通过此阶段。

## 泳道和性能口径

执行记录至少关联 run_epoch、task_ref、order_key、block_idx、subtask_slot、
worker、物理组、generation 和 kernel 起止时间。
另记录准入发布、成员／角色、领取范围、LocalDone、block 完成与 TaskDone，
使“允许重叠”和“没有迟到成员”都能审计。

等待需区分数据依赖、前序成员义务、准入发布、block 领取及 Mix 局部资源。
不能把所有等待合成一个 poll span，也不能只记录 task_ref 就声称执行顺序一致。

~~~text
trace_index = task_trace_base
            + block_idx * active_subtask_count
            + active_rank(subtask_slot)
~~~

Host 检查乘积和前缀溢出。不同 block 不共用旧单 block trace 索引；
跨 worker 时序不能用各核各自归零的时钟比较。按本轮确认的频率换算设备周期。

性能分开报告 Host 准备／上传、设备执行及主要 kernel 间隙，固定 kernel、输入、
核集合、工具链、计时边界和插桩条件。原 only-scheduler 负载不能直接与真实 CSA 相减。
重点测准入提交、block cursor、GM 组握手、完成聚合和 direct refill 命中变化，
不预设动态领取比静态分配快，或复用 ready 就必然没有额外开销。

128B 地址隔离不消除同一个 cursor 的真实原子竞争；软件 shard 不等于 DIE。
批量发布优化不得把未发布贡献或准入空洞带入后序阻塞 kernel。

## 有限状态抽象检查

以下模型已保存为 [explorer.py](../../../../tests/ut/py/hbg_spmd_contracts/explorer.py)，
对应 [test_interleavings.py](../../../../tests/ut/py/hbg_spmd_contracts/test_interleavings.py)。
本次重新运行核对了下列状态数；遍历预算不足将失败，不计作证明通过。

2026-10-23 的枚举模型包含：ready 后一次性准入、成员集合、block 唯一领取、
已领取／已进入／已返回状态、每核 LocalDone 和真实任务依赖。
普通 block 可独立返回；协同 block 在全部同伴进入后允许返回。
LocalDone 要求本核无在途工作，普通模式还要求不可撤销的领取范围耗尽。

| 场景 | 可达状态数 | 结果 |
| ---- | ---------- | ---- |
| A/B 均 ready，任一共同顺序 | 117 | 无终止前死锁，存在跨 task 重叠 |
| 增加真实 A→B 依赖 | 42 | 无终止前死锁，不允许 B 提前 |
| A 等 P，ready 的 B 可先执行 | 473 | 存在 A 未 ready 时 B 已进入的执行 |
| 不参与 A 的核独立执行 B | 103 | 无终止前死锁，存在独立推进 |
| 六个普通 block 动态分给两核，再进入 B | 4799 | 无终止前死锁，存在跨 task 重叠 |
| A、B、C 成员分别为 01、12、02 | 1768 | 任一共同发布顺序均无终止前死锁 |
| 错误地允许各核任意选 A/B | 373 | 找到 4 个非终态无进展状态 |
| 三个共同驻留 block 映射到两核 | 10 | 找到 1 个非终态无进展状态 |
| 错误地做完一个普通 block 就离开 | 7023 | 找到 52 个非终态无进展状态 |

模型枚举所有可行准入、领取、进入、返回及局部关闭动作，不把空转 poll 当成进展。
它把共同发布和所有权绑定视为原子步骤，测试中的任务统一表示为准入节点；
没有验证实际发布临界区、普通单 block 旁路、Mix GM 组协议、缓存、失败、predicate
或 kernel 本身。上述结果不替代矩阵中的实现测试和 A5 真机验证。
