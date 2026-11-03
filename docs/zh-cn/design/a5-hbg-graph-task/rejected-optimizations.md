# 已否定的性能尝试

本文历史测量所引用的 JSON 仅保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`，不随仓库分发；下文保留文件名用于追溯。

独立 block 调度原型也已整体撤回：用户不接受 3.08 倍冷构建成本，当前代码恢复 `queue-ordinary-credit-v1`。
该思路没有被永久排除，保留为[后续待选方案](block-ready-scheduling.md)，含重启条件和历史证据。

本页记录 A5 HBG native Graph 实施中已经测量并撤回的候选，避免恢复上下文后重复采用。
当前验收状态以[实施记录](../../../../tests/docs/a5-hbg-graph-task/README.zh-CN.md)为准。

## 将 native 元数据检查移出内联

`scheduler_native_node_checked` 是 native 节点读取前的边界校验入口。
Graph 支持使普通任务也经过包含 native 分支的 helper；曾尝试仅给该函数加 `noinline`，
将较大的校验主体移出调用方，降低调度代码体积。该候选标识为 `p1-outline`。

功能验证没有发现行为变化：170 个 C++ 目标、A5 sim 的 37 个 L2 与 28 个 standalone 目标通过，
真机 Graph 六个 pytest 目标通过，独立协议审计覆盖 15 份 capture、2,881 次 execution。
这些结果不能替代性能验证。

设备 7 的每项测量先校验 golden，再采集三批各 100 个稳态轮。
三个用例共 12 个功能/计时进程均核验 fresh resident 模式。
以下为 Device 中位数相对此前相同设备 `p1-opt` 测量的变化；两次测量不是交错 A/B/A，保留此限制。

| 用例 | Device 中位数变化 | Device p95 变化 |
| ---- | ----------------- | --------------- |
| batch paged attention / Case1 | −0.35% | +5.79% |
| fanout 1024 | +11.90% | +11.72% |
| AIV chain 64 | +13.31% | +12.88% |

候选对最近主线的 Device 中位数仍分别增加约 2.91%、16.41%、43.90%。
因此撤回该 `noinline` 改动，保留 `p1-outline/compare-opt.json`、`compare-baseline.json`
及冻结运行库作为失败证据。局部代码变小不等于延迟改善，也不能据此推断硬件 I-cache 容量。

只有新的调用边界或代码生成证据能说明开销发生了变化，才值得重新考虑该方向；
重试仍须使用完整功能与无诊断性能对照，不能用开启 profiling 的结果代替普通运行。

## 三个校验函数外提与完整 Qwen 复测

后续 `p2-native-outline` 对 header、template、node 三个 checked helper 同时加 `noinline`，
在已完成热循环图类型特化和 canonical ordinary view 的版本上复测，并补充此前没有的完整 Qwen 对照。
此次恢复时漏读了上节的单函数否定记录；不能将这个方向记为首次尝试。

任务 `task_20261102_125537_261804917999` 在设备 3 独占完成，exit=0。
Qwen 及五项无 Graph 均按旧 native → 候选 → 旧 native 执行，每臂三批各 100 稳态轮，
先验证完整 golden，并逐进程核验 fresh resident 模式。五项无 Graph 共 60 个功能/计时进程。
完整数值见 Qwen 配对（本地 `qwen-outline-performance.json`）和
无 Graph 配对（本地 `native-outline-focus.json`）。
这里的参照是此前候选 `p1-ordinary-view`，不是主线，五项定位集也不替代完整 19 项验收。

Qwen Device 中位数为 65.237 → 63.221 → 65.155 ms，候选对后臂降低 2.97%；
Host 降低 2.70%，bind 增加 5.67%。完整 out/K/V 通过。
无 Graph 对照的结果如下；参照 Device 前后漂移为 +0.10%～+1.01%，不能解释主要回退。

| 用例 | 后臂旧版（ms） | 候选（ms） | Device 变化 |
| ---- | -------------- | ---------- | ----------- |
| batch paged attention / Case1 | 3.124 | 3.177 | +1.69% |
| fanout 1024 | 1.953 | 2.190 | +12.13% |
| AIV chain 64 | 0.473 | 0.530 | +12.14% |
| mixed chain 1024 | 6.039 | 6.613 | +9.50% |
| mixed fanin32 1024 | 6.937 | 7.636 | +10.07% |

native 主循环符号从 580464/591348 bytes 降至 373076/392900 bytes，
普通路径却从 272164/282972 bytes 增至 374752/393856 bytes。
该证据说明编译器的内联决策影响了其他路径，不证明具体的 I-cache miss 或硬件容量。
93 项 C++、28 项独立 A5sim 和 37 项场景仿真通过；Qwen L4 原始完整 golden 通过，
独立协议审计覆盖 237088 个事件、11165 个设备节点、19208 次执行和 641 个 cooperative task，
见 验证摘要（本地 `qwen-outline-validation.json`）。

结论仍为撤回，三个 `noinline` 标注均已移除；不能用 Qwen 约 3% 的收益交换短任务退化。
冻结候选、源码差异、符号表、原始日志及所有计时样本保留在 `outputs/a5-hbg-graph-task/p2-native-outline/`。
后续候选须先读取本页；以不同调用边界重新试验时，明确区分新假设与这个已否定组合。

## 普通任务让行与重复扫描 {#ordinary-service-and-pool-probe}

任务 `task_20261103_122234_11015103109` 在设备 0 顺序验证两个独立候选，均保留 T/H8 队列：

- `queue-ordinary-service-v1`：本核无 pending/current obligation 且持有普通 executable candidate 时，
  让未提交的 cooperative candidate 延后一轮 discovery；下一轮必须尝试 admission。
  不持有 commit lock/log position，不改变普通任务 entry gate 或已提交 obligation 的优先级。
- `queue-direct-pool-probe-v1`：在 ordered fill 与普通 slot fill 中，省去全局 pool 领取前只用于生成
  `allow_steal` 的目录扫描；pool claim 本身不使用该参数，仍独立执行轮转、检查与领取 CAS。
  此候选不包含上述 admission 改动。

两者沿用固定 37-kernel callable、40 层、B16、seq_len=3500、seed=1234；各先做三轮完整 out/K/V，
再独立采 5 轮预热与 100 轮稳态，108 次执行均验证 fresh native mode=1。

| 候选 | Device 中位数 / p95（ms） | Host 中位数（ms） | 对此前同设备 T/H8 的 Device 变化 |
| ---- | ------------------------- | ----------------- | -------------------------------- |
| 普通任务让行 | 41.094 / 41.375 | 46.105 | +0.355% |
| 省去重复扫描 | 40.800 / 41.067 | 45.945 | −0.362% |

参照为此前设备 0 的 T/H8 40.948 ms，未在本任务重测；历史主线设备 3 的 29.478 ms 也未重测。
两个候选均未建立有用的收益，不能把约 0.36% 的变化解释成确定加速或回退；撤回两者并保留冻结源、运行库与逐轮数据。
原来的重复 payload publication 现象仍值得研究，但这次结果不支持仅用本地一轮让行解决主要等待。

各 65 项 A5 仿真、6 个 Graph 真机目标、15 份协议 capture（各 2,881 次执行）通过。
让行候选的六项新增 CPU 测试在修改前失败、修改后通过；最终 ordered target 共 122 项通过。
97 个相关 C++ 目标是在最后一次排除 inline candidate 的小修正之前通过，不能记作最终源码的全套验证；
重复扫描候选未单独跑 C++ 全套。两者均未运行完整故障矩阵或完整 19 项性能集。
完整身份、数据和验证日志哈希见候选证据（本地 `qwen-scheduler-candidates.json`）。

## 控制节点提前发布 ready 前缀 {#control-ready-prefix}

`queue-control-prefix8-v1` 基于 `queue-ordinary-credit-v1` 的普通任务领取额度策略，保留 T/H8。
控制节点服务原本在本轮路由结束后发布 executable batch；候选在同类 ready 积累到 8 项时提前发布，
尝试让其他核心执行前缀，同时当前核心继续路由。8 个控制节点、64 个 waiter 的服务预算，
continuation、完成计数的发布顺序及 kernel entry permit 均不改变。
L/C 后端仍可进入私有待发布链，不能把调用 batch push 等同于其他核心已经可见。

任务 `task_20261103_151326_41791814623` 在设备 0 完成固定 37-kernel callable 的完整 Qwen：
40 层、B16、seq_len=3500、seed=1234；三轮完整 out/K/V 通过，另采 5 轮预热、100 轮稳态。
108 次执行均核验 fresh native mode=1。Device 中位数/p95 为 **38.408/38.644 ms**，
Host 中位数 **43.625 ms**，bind 中位数 **1.481 ms**。
相对历史普通任务领取额度版本的 38.454 ms 仅低 **0.120%**；历史参照在设备 6，未在本任务重测。
该幅度不足以建立有用收益，因此撤回提前发布改动，保持 `queue-ordinary-credit-v1` 的运行行为；主线没有重新采集。

提前前缀回归在旧代码失败，跨三次服务的 130 waiter 链测试在两版均通过；最终 97 个相关 C++ 目标、
65 项 A5 仿真、6 个 Graph 真机目标、15 份协议 capture（2,881 次执行）全部通过。
未运行完整故障矩阵或完整 19 项性能集。冻结源码、运行库、撤回补丁和逐轮样本保留，
见本轮证据（本地 `qwen-control-prefix.json`）。

## Work 协议推进函数的内联拆分 {#work-protocol-outline}

`queue-work-outline-v1` 保留空闲检查，将复杂 MIX lease/cohort 推进标为 `noinline`。
生产函数 17 项连续 2,000 轮、AIC/AIV 编译探针、15 项模拟场景通过。
设备 6 的任务 `task_20261103_192056_54555513936` 中，MIX 用例出现非法 GM 地址访问，
随后多个用例报告运行通道已失败；Qwen 对照未执行，不能记录性能收益。

任务 `task_20261103_192542_72923325067` 使用独立进程复验，
`burst256_mask6` 的 L3 捕获仍失败，纯 vector 的 L3/L0 均通过。
ELF 检查确认两个推进函数在 AIC/AIV 对象中为同名弱符号，最终 ELF 只保留了 AIC 的两份实现。
这是明确的链接缺陷：这类架构相关函数需要内部链接，不能跨 AIC/AIV 合并。

为检验归因，另行复制失败产物，仅按架构重命名、局部化两个函数并重复链接，
得到 `queue-work-outline-local-v1`，最终 ELF 保留四个 LOCAL 函数。
任务 `task_20261103_193450_10324693829` 的独立 MIX 复验仍失败，vector L3/L0 通过。
因此弱符号合并还不是经验证的完整解释；重链接也不等同于重新编译 `static` 源码。
`static` 源码通过了 CPU 压力与编译探针，但完整构建在本轮撤回时终止，未取得真机结论。

代码体积也需区分包装函数和真正的调度循环：DFX 版本的 native resident 包装函数
约 200/221 KB，拆分后约 79/82 KB；实际 `run_ready_dispatch_loop` 仅缩小约 5–11%，
整个 `.text` 从 12,570,392 B 增至 13,450,520 B。不能由包装函数变小推断取指收益。

本轮撤回全部内联拆分改动，保留失败源码、对象、转换步骤和日志，
只将有独立回归证据的跨逻辑任务发布预算修正进入下一候选。
该内联试验既未通过真机功能验收，也没有 Qwen 性能数据。
