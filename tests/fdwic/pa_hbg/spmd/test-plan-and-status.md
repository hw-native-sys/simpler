# A5 HBG SPMD 测试入口和验收状态

对应 [编码契约](implementation-contract.md) 与 [C01–C56 边界矩阵](corner-cases-and-validation.md)。
以下原始验收基线为 `f5cc679bbf3d`，记录日期 2026-10-23。
2026-10-25 起的生产编码结果在 [实现过程](implementation-progress.md) 持续记录；
其中 payload 前置修复已使六种 mask 的 N=1 仿真各三轮通过，不覆盖本页历史失败证据。
**模型通过不等于生产 SPMD 已实现；原生验收中的红灯必须保留，不能用回退或 skip 消掉。**

真实 HBG CSA 另有 [业务 TDD 入口](../csa/test_csa_business.py)，保留 54 个原 kernel 与七项 golden。
覆盖范围、A5 实跑记录和独立 resident 红灯见 [CSA 中文说明](../../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md)。

## 一、四层检查分别证明什么

| 层次 | 文件 | 检查对象与边界 |
| ---- | ---- | -------------- |
| CPU 合同模型 | [model.py](../../../ut/py/hbg_spmd_contracts/model.py)、[types.py](../../../ut/py/hbg_spmd_contracts/types.py) | 身份、成员、动态领取、局部顺序、完成与代次；不是运行时实现 |
| 交错与独立审计 | [explorer.py](../../../ut/py/hbg_spmd_contracts/explorer.py)、[trace_audit.py](../../../ut/py/hbg_spmd_contracts/trace_audit.py) | 穷举小状态空间、保留错误反例；独立检查事件，不调用模型状态转换来判对错 |
| 现有 C++ 原语 | [test_hbg_spmd_primitives.cpp](../../../ut/cpp/a5/runtime/host_build_graph/test_hbg_spmd_primitives.cpp) | 真正调用仓库 ready-pop／generation helper，固定其当前语义；不是新调度器 |
| 原生 HBG 入口 | [test_spmd_contract.py](test_spmd_contract.py)、[kernels](kernels) | 编译 Host 编排和 AIC/AIV kernel，执行实际 runtime，检查本轮 resident 来源、每个身份、原 N、依赖结果 |

CPU 包中另外有 [test_native_fixture.py](../../../ut/py/hbg_spmd_contracts/test_native_fixture.py)，
检查原生用例的 golden、来源门禁、保护区和错误检测器本身；不能把它计作设备执行。

协议模型把共同发布／唯一领取视为原子动作，所有有 kernel 的节点均表示为准入节点。
状态枚举不模拟设备 cache、失败核停机或真实 GM mailbox；随机 DAG 检查使用固定 seed，
只是可复现的多形状完成顺序检查，不宣称枚举了这些 DAG 的全部并发交错。
状态空间预算耗尽时直接失败，不返回“未发现死锁所以通过”。

### C++ 的四个具体回归

- GenericPopSnapshotDoesNotPinTheExpectedTask：A 被取走后，旧 endpoints 调用 generic pop 会取 B。
- TailAppendChangesEndpointsWithoutChangingTheHead：尾部追加改变 endpoints，不应误认为队头必定改变。
- AnEmptySnapshotDoesNotProveCurrentQueueQuiescence：旧空快照不是当前队列空或全图结束的证据。
- OwnAndRemoteMixGenerationsAreIndependentDomains：own 槽与远端 Mix 的 generation 不能当共同顺序。

这里验证的是**原语使用前提**。如果主方案一次 pop 准入，就不需要为了测试新增条件摘头接口；
如果改成延迟摘头，必须另加 expected-task 操作及其竞态测试。

## 二、已经造好的边界与尚欠设备证明的部分

| 边界组 | 现有可执行测试 | 仍需随生产编码完成 |
| ------ | -------------- | ------------------ |
| C01–C11：N、mask、容量、身份、稀疏核 | test_shapes：合法／非法 N、32767 乘积边界、mask 4 规范化、sync 容量、拓扑重复／缺失、trace 索引 | Host 实际拒绝路径；discovery 与 placement 一致性；native 全矩阵 |
| C12：view 和原 ABI | 原生不同 task 使用不相交 view，消费者读取完整 view | 原样 CSA workspace／reduction／TPipe |
| C13–C16、C39、C42–C43 | test_order_and_completion、test_interleavings：快核进入 B、真依赖禁止、B 先于未 ready A、非成员前进、交叠成员 | 对真实 kernel 起止记录证明重叠与顺序 |
| C17–C18、C22 | test_graphs：宽度 2/7/32/65/128 的 fanout/fanin、20 个固定 seed DAG；无 kernel 服务和空图 | 生产 wake-list 原子竞态与公平性；Host inline 不重复计数 |
| C19–C20、C25–C28、C47–C49、C56 | model：一组一个身份、全部局部资源到达、mask 6 协调者、未发布最后组、cluster 独立关闭 | GM mailbox 逐步暂停、真实通信收尾、seal 与 mailbox 复用、inactive lane 释放 |
| C23–C24、C55 | sync_once、容量、早到／晚到、过载和 A/B 反序死锁反例 | 有界 barrier kernel、不同物理位置和真正的满核任务；当前原生矩阵未包含 sync_start |
| C29–C33、C35–C38 | stale lease、generation／epoch 上限、重复完成、输出可见性、错误不放行、重跑换 shape | A5 cache 发布、硬件错误静止与清理；真实重新绑定的所有槽初始化 |
| C40–C41、C46、C50 | 准入去重、记录容量、准备不占可见洞、冻结成员、暂空非 EOF | 短锁和普通 entry permit 的线性化竞态、发布者失败 |
| C44–C45、C54 | 三种 source 的模型门禁、双通道循环等待反例、两个 pending 槽不是两个位置 | 实际 own/Mix/direct refill 统一 gate、让出预填 payload、控制服务饥饿 |
| C51 | 新旧 publish_count、mode 0/3/4/非法值拒绝；原生每次比较同时验来源 | 接完整 trace 的实际 producer，不只依赖 hook |
| C21、C34、C52–C53 | 保留现有能力边界；本次不假装实现 Graph/predicate/async/无 SSBUF | 对应生产 gate 的正反用例、GM 路径去除 SSBUF 初始化依赖 |

test_trace_audit 对缺事件、重复事件、错 task/block/slot/N/order/generation/worker、
跨 cluster 配对、提前 TaskDone、遗漏 LocalDone、旧 epoch、错误来源、截断、
缺字段和逆序时间分别造坏数据，要求失败。
重复事件用例会重新编号事件时间，防止仅因 timestamp 重复而“误以为抓住了协议错误”。

### 三类反例必须一直保留

1. 两个互相等待的 task 在不同核上反序进入：模型找到非终态无进展路径。
2. 普通 task 只做一块就关闭、剩余块无人负责：模型找到遗留工作。
3. Mix 最后一个 block 已领取但尚未发布，AIV 不能把 cursor=N 当作可离开；
   另以实际 C++ helper 验证“原始 generation 可比较”的假设不成立。

### CLA 真实泳道驱动的增量测试

来源为本地 `examples/a5/cla-swimlane.json`（原始大文件不入库，来源门禁需另行提供），SHA-256：
`a33e497915b499f2ee13b9ae8182f4cb358d62106f8d358bbae9bc67fb30d747`。
文件是 TMR 的 Perfetto 导出，不是 HBG resident 协议 trace。
捕获没有记录生成版本；本页的 f5cc679 是测试目标代码基线，不代表捕获的生成提交。
[只读投影器](../../../ut/py/hbg_spmd_contracts/cla_trace.py)复用现有 TMR TaskId 解码，
生成约 55 KB 的 [冻结夹具](fixtures/cla_workload.json)，不复制整张图，也不修改原文件。
原图含 101 个节点：56 个有 kernel、43 个 alloc、2 个 Dummy；
736 次核执行（176 AIC、560 AIV），199 条去重后的显示依赖边。
Scheduler View 的 736 个镜像不是额外执行；flow 箭头数量也不是逻辑边数量。

| 实测特征 | 针对性检查 |
| -------- | ---------- |
| N 包含 5/6/10/12/20/24，最大 32 | 整份图在 1/2/4/32 cluster、不同 ready 顺序下运行，完整身份仍为 736 |
| r2t17 的 kernel 时长 0.454–23.512 μs；r2t20 为 0.642–40.082 μs | 31/32 已完成不得释放消费者；故意延迟 block 0/15/31，不以最大编号代表最后完成 |
| r3t10 为 32 个 1C1V block | 32 个 AIC 先返回不等于 32 个 Mix block 完成；快 cluster 可进入独立后序任务 |
| 两路 gather32 → 两路 Mix12 → 八路 V8/Mix4 → Mix32 汇合 | 两分支都要满足；八路中任一路长尾都不得提前释放最终消费者 |
| r1t34、r2t22 是有真实前驱的 Dummy | 不计 kernel；前驱检查穿透无 kernel 节点，不能跳过其 kernel fanin |
| r1t0/r2t0/r3t0 重用 local id；八路 proj_o 使用相同函数 | 保留完整 task 身份；HBG ref 按提交顺序映射，不能按裸 ID 或函数合并 |
| 367.338 μs 时共享队列全零，仍有 21 个 kernel 在途 | 空 ready queue 不是 EOF；必须等未完成身份及控制义务收敛 |
| r2t16 的箭头代表块先于最后一个块结束 | 把消费者移到代表块结束之后、真正最后返回之前，审计必须失败 |

flow 322 的 `hb_violation` 仅在 Scheduler View：前驱 finish 为 329.922 μs，
后继 dispatch 为 319.455 μs；Worker View 前驱全部 kernel 已于 318.861 μs 结束，
后继 kernel 最早 340.895 μs 才开始。因此不能仅凭箭头名字宣布逻辑依赖违规。
这些时间只固定捕获事实，不作为模拟器或未来调度器的耗时阈值。

文件没有提供 block_idx、逻辑 subtask_slot、sync_start、物理 cluster 配对、WAIT/RETAIN
以及原始 collector 的 dropped/truncated 证明；投影器不补猜这些字段。
1C1V 仅按观察到的形状规范化为 mask 3，另造 mask 5 的合法放置变体，
不是声称捕获证明了原来的 AIV 逻辑槽。显示边在测试中保守作为先行约束，
不声称所有显示边都是原 runtime 的 readiness 边，也不把它冒充原样 CLA 数学计算。

主要检查分别在 [来源和坏数据测试](../../../ut/py/hbg_spmd_contracts/test_cla_capture.py)、
[调度合同测试](../../../ut/py/hbg_spmd_contracts/test_cla_protocol.py)、
[原生夹具自身检查](../../../ut/py/hbg_spmd_contracts/test_cla_native_fixture.py)。

## 三、原生用例结构与使用方法

### 90 个参数场景

mask 为 1、2、3、5、6、7；每种取
N=1、2、3、7、31、32、33、63、64、65、96、128、129、512，共 84 个场景。
另加每种 mask 下 17 个独立 task、每 task 3 block 的槽复用场景，共 90 个。
32 AIC／64 AIV 的容量两侧、2P+1 和 CSA 的 N=512 都在矩阵中；
其他拓扑仍按实际容量追加 P−1/P/P+1，不能认为这些常数覆盖所有设备。

每个基本场景先提交两个独立 task，再提交读取完整输出的单 AIV 消费者。
每个 task/block/逻辑子槽拥有 128B 记录：
第一条 64B line 只原子累加访问次数，第二条 line 写 block+1、原 N、slot+1、task+1。
计数 line 不做普通写回，避免旧 cache line 覆盖计数。
block 0 也写非零身份，漏执行不能靠初始零混过去。
每 task 还有保护记录，分别计数缺失 context 指针与非法 block/N/slot；
正常访问不会触碰它。

消费者在真实 HBG fanin 满足后校验全部 active 身份、次数及保护区，
写回成功数／错误数。最后 block 延迟写记录，用于放大依赖过早释放问题。
Host 对全部输出精确比较，inactive 槽和 padding 必须保持零。
此测试的原子和消费者校验有明显开销，**不是调度性能基准**。

kernel 读取 block/N 使用现有 get_block_idx(args)/get_block_num(args)。
仅在调用前检查 ABI 后缀指针是否为空，避免当前实现缺 context 时把合同失败变成设备异常；
不补造 context，不改为相信输入 scalar 的 N，也不篡改原 block 数。

### CLA 的七个 TDD 场景

[test_cla_hbg.py](test_cla_hbg.py)复用现有 128B 记录、设备原子包装和 SceneTestCase。
`cla_full` 保留 56 个 kernel task、2 个 Dummy、88 条非 alloc 边和 736 次核执行；
43 个 alloc 作为已经完成的 Host 输入准备，不让 Scalar 重做分配。
其余场景为 `cla_tail_first`、`cla_full_aiv1`、`cla_two_gathers`、`cla_eight_way_join`、
`cla_cross_ring`、`cla_irregular_shapes`。缩小子图会收缩被省略的中间节点，保留可达约束。

每个 kernel 在入口读取**所有检查前驱的全部 block/active slot**，错误写入自己的记录，
后续前驱补齐不能抹掉这项错误；Dummy 的 kernel 前驱会传递到检查列表。
数据表读取使用显式依赖，避免全表读被 TensorMap 自动加成虚假的串行依赖。
这些 probe kernel 不是原 CLA kernel，不验证真实 TPipe/数值计算，也不作性能基准。

TDD 推进顺序：先保持来源／协议／夹具检查绿色；再逐项消除原生红灯，
只有本轮 resident 发布、精确执行身份、前驱哨兵同时通过才算转绿。
禁止删除 mode 断言、减少原 N 或用 xfail 掩盖缺口。不能把 `mode=4` 的正确结果算作成功。

### 运行命令

从仓库根目录，用当前项目 venv；源码更新后若原生扩展旧，先按仓库流程增量重编译。

~~~bash
source .venv/bin/activate

# CPU 合同、交错、审计器及用例自身检查。
python -m pytest tests/ut/py/hbg_spmd_contracts -q

# C++ 实际原语 + 既有 ready/dispatch/contracts/bootstrap 回归。
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build -j 8 --target \
  test_a5_hbg_spmd_primitives test_a5_hbg_scheduler_ready \
  test_a5_hbg_scheduler_dispatch test_a5_hbg_scheduler_contracts \
  test_a5_hbg_scheduler_bootstrap
ctest --test-dir tests/ut/cpp/build --output-on-failure \
  -R '^(test_a5_hbg_spmd_primitives|test_a5_hbg_scheduler_(ready|dispatch|contracts|bootstrap))$'

# 当前纯 C/V 的单 block 绿色基线。
python -m pytest tests/fdwic/pa_hbg/spmd/test_spmd_contract.py \
  --platform a5sim --case mask1_n1 --case mask2_n1 --pto-session-timeout 600 -q

# 目前应失败：不能接受 N=2 回退到 legacy。
python -m pytest tests/fdwic/pa_hbg/spmd/test_spmd_contract.py \
  --platform a5sim --case mask1_n2 --pto-session-timeout 600 -q

# 初始基线暴露单 block Mix context 问题；修复后应通过，不是多 block 门禁。
python -m pytest tests/fdwic/pa_hbg/spmd/test_spmd_contract.py \
  --platform a5sim --case mask3_n1 --pto-session-timeout 600 -q

# A5 真机，同一套入口；可用 --rounds 3 检查本轮来源及重复绑定。
python -m pytest tests/fdwic/pa_hbg/spmd/test_spmd_contract.py \
  --platform a5 --device 0 --case mask1_n1 --case mask2_n1 --pto-session-timeout 600 -q

# 生产实现完成后的全矩阵；当前运行不应全绿。
python -m pytest tests/fdwic/pa_hbg/spmd/test_spmd_contract.py \
  --platform a5 --device 0 --pto-session-timeout 600 -q

# standalone 走相同来源／golden 门禁。
python tests/fdwic/pa_hbg/spmd/test_spmd_contract.py -p a5sim --case mask1_n1

# 与用户原图逐项核对；原图缺失会明确失败。普通 UT 仅依赖小夹具。
python -m pytest tests/fdwic/pa_hbg/spmd/test_cla_source.py -q

# CLA 原生 TDD：当前返回非零，错误应是探针记录／哨兵通过后仍 mode=4。
python -m pytest tests/fdwic/pa_hbg/spmd/test_cla_hbg.py \
  --platform a5sim --case cla_full --pto-session-timeout 600 -q
# 用 --case 逐个选其他六个场景；实现完成后不传 --case 跑全部。
~~~

这里的 CASES 由 SceneTestCase 内部循环执行，不是 90 个独立 pytest node：
筛选参数场景要用 `--case`，不能用 `-k mask1_n1`。
一次 `1 passed` 可能执行了多个选定场景；失败会携带准确 case 名并停止当前类的后续场景。
禁止 `--skip-golden`；测试不加 xfail/manual/skip 来隐藏待实现项。
tests/fdwic 不自动继承 tests/st 下的 scheduler_mode fixture，
因此在类的公共运行／比较 hook 内检查，pytest 和 standalone 都生效。

## 四、本次验证记录及发现

- 已按当前源码重建本地扩展，消除 `85dafc4 → f5cc679` 的二进制／源码不匹配。
- 基础 CPU 包原为 423 项；CLA 增量后，加上原图来源门禁共 501 项通过，新增 78 项。
  其中原图门禁单独占 1 项，其余可只凭小夹具运行；这些不是设备执行测试。
- 七个 CLA 原生场景在 a5sim 逐个执行：均先通过探针记录与前驱哨兵，再以 mode=4 失败，
  因而保留七个真实 resident 红灯，而非通过返回码或编译失败凑红灯。
- Dummy 间接前驱的审计反例先实测为 `DID NOT RAISE`，修正测试侧验证器后转绿；
  没有借这一步修改生产 runtime。新 Host 编排和三个 kernel 的 A5 编译通过。
  本轮没有重跑真机，下面的 bootstrap 问题仍是上一轮未解决项。
- C++：5 个可执行测试目标全部通过，共 106 个 GoogleTest case；
  其中新增原语边界 4 个，既有 contracts 18、bootstrap 12、dispatch 41、ready 31。
- a5sim：纯 C/V 的 mask1_n1、mask2_n1 通过；mask1_n2 被 mode=4 门禁拒绝。
  standalone 的两个 N=1 场景各运行 3 轮也通过，逐轮检查来源发布计数与输出重置。
- A5 device 0：CANN 查询确认 Ascend950PR_958b、32 AIC/64 AIV；
  纯 C/V 的 mask1_n1、mask2_n1 曾在 resident 来源及完整输出检查下通过。
  mask1_n2 在同一设备被 mode=4 门禁拒绝，未被算成 resident 成功。
  最终版本追加 --rounds 3 复跑时，mask1_n1 在首轮也触发相同超时，后续轮次和 mask2 未执行；
  因此当前真机基线是**未稳定验收**，不能把先前一次通过当作最终全绿。
- 单 block Mix 的 mask3_n1 在 a5sim 暴露缺失 context；
  GDB 在 get_block_idx 的解引用处确认 LocalContext 后缀为零。
  加入保护后稳定以明确断言失败，不能将此红灯改成接受缺少上下文。
- A5 的 mask3_n1 在加入保护后仍失败：约 47 秒后报 ACL 507018、sched_error_code=100；
  runtime 的异常退出完成了恢复确认。尚未证明这一真机超时与模拟器空 context 是同一原因，
  也不把泛化错误码当作死锁证明；未继续重复运行这一故障场景。
  两次失败的对应 device log 均出现 timeout site=101（AICPU_INIT_BARRIER_TIMEOUT），
  Mix 记录同时有 wait_bootstrap_complete 超时，采样 task 状态未推进；
  目前定位到 resident 初始化／bootstrap 阶段，不是已经证明的用户 kernel 内同步死锁。
- 最后独立复跑仓库既有的 TestHbgSingleAicRoot::single_aic_root 对照，
  同样出现 ACL 507018、sched_error_code=100；问题并非只出现在新测试。
  当前环境的真机 resident 初始化稳定性应单独调查，不能通过放宽新测试断言处理。
- 没有执行并通过完整 90 场景，也没有宣称 native sync_start、GM Mix 或 CSA 已通过。
  本次不报告性能收益；没有采集新生产协议的完整泳道。

### 初始基线发现的真实代码缺口

在上述基线中，host/runtime_maker.cpp 只给普通 dispatch slots 设置两个 context 后缀地址及 block=0/N=1，
独立的 mix_payloads 区缺少对应初始化。scheduler_prepare_dispatch_slot 修改函数和
sub_block_id，却没有补齐这两个后缀；materialize 只写实际 tensor/scalar 参数。
因此即使 N=1，使用 block/context accessor 的 Mix kernel 也可能失败。
此前不读取这些 accessor 的数值 Mix 用例不能覆盖这个缺口。

编码顺序增加一个前置门槛：**own 和 Mix 两类 payload 的完整初始化及第二次绑定验证**。
修复后 mask3_n1 以及其他 mask 的 N=1 应转绿，然后再验多 block。
最初 TDD 提交未修改 runtime。此缺口现已由统一初始化修复，
C++ 重绑定和原生 N=1 仿真证据见实现过程；真机 bootstrap 问题仍单独追踪。

## 五、把测试接到后续生产实现

[trace_audit.py](../../../ut/py/hbg_spmd_contracts/trace_audit.py) 接受的是拟定的测试适配格式
`hbg-spmd-contract-v1`，**不是当前 swimlane JSON 格式**。
请求方提供独立的期望 graph/topology/epoch；不能让 producer 自报 N 和成员再自证正确。
头部必须包含 schema/epoch/producer/complete/dropped/events；
事件包含共同准入、带映射的 claim、publish、arrive、start、return、visible、
retire、block_done、task_done、cluster_close、local_done。
start 必须给 task/block/slot/worker/N/order/generation/source，完整性检查不能猜缺失字段。

当前样例 producer 明确是 reference；审计器默认要求 aicore，因此不能直接拿模型数据冒充设备结果。
tick 是适配格式内的因果事件序号，不是已经验证过的跨核统一 cycle。
接 native 时必须从实际采集到的发布／执行身份构造可验证因果序，
缺少事件或无法确定次序时失败，不能为了通过校验事后编造事件。
还要描述 own slot 与 Mix group 的不同 generation 域；不能将模型的单一 lease 序号
直接等同于现有两通道的原始计数器。

新增 runtime helper 后，把其可控交错接入 C++ fixture，与参考合同逐项对照；
模型的策略开关只用于测试错误协议，不是生产 feature flag。
必须另补普通旁路 entry permit、GM descriptor/ack/seal 可见性与复用、
inactive lane 资源释放、物理 Mix 配对和退出静止的实现测试。
最后使用原样 CSA 的七项 golden、完整执行集合与来源门禁收口。
