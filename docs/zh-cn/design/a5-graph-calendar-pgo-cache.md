# A5 Graph Calendar PGO 分桶缓存设计

**状态** 已确认并进入实现。对应 [Issue #3244](https://github.com/hw-native-sys/simpler/issues/3244)。

## 问题与范围

当前 A5 online PGO 的 Graph Calendar 键包含 Definition 内容、worker 数、调用标量以及张量布局和别名关系。两次提交即使生成相同的调度结构，仅执行标量不同也可能各自生成初始计划、采样和保存计划。对于动态序列长度，Lib 可以根据工作量把调度表现相近的输入归到同一个桶，让同一 Worker 内的多个独立 run 复用一次训练的结果。 [当前 Graph Calendar 查找](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/a5/runtime/percounter/host/state.h#L284-L365)

分桶作用于 A5 Calendar 在线 PGO 的 Graph body 阶段。普通任务阶段继续使用当前键和训练流程。候选生成、评分与搜索仍由现有 PGO 负责。 [评分与搜索 Issue #3239](https://github.com/hw-native-sys/simpler/issues/3239)

## 提交接口

给 C++ `rt_submit_graph` 增加 `simpler::hbg::GraphSubmitOptions`，其中 `calendar_hash` 为 `std::optional<uint64_t>`。无 Config 的调用可直接传 options，带 Config 的调用将 options 放在 Config 之后。显式 `graph_id` 和函数推导身份的重载均支持该选项。现有调用省略选项时走当前精确参数键。`0` 和其他 `uint64_t` 值均可用作桶号。

```cpp
const simpler::hbg::GraphSubmitOptions options{bucket};
rt_submit_graph(&decode, args, options);
rt_submit_graph(&decode_with_config, args, config, options);
```

Lib 根据实参和工作量计算桶号，并选择运行耗时相近的桶成员。Runtime 判断计划能否用于当前调度结构。桶号表达复用意图，性能相似性由 Lib 的分桶规则及真机测量确定。

Graph Definition 的录制、复用与校验继续用 `graph_key` 和 Config 身份。`graph_key` 是单次 bind 内分配的序号，Calendar 用 `rt_submit_graph` 已接收或推导的 `graph_id` 跨 bind 标识 Graph。同一个 Definition 的不同调用可以带不同桶号，包括同一 bind 内的多个调用。 [Graph Definition 和 Config 合同](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/common/host_build_graph/docs/GRAPH_EXECUTION.md#L95-L152)

## 缓存与执行合同

### 查找和兼容校验

显式桶的查找范围为 `(Callable, Graph, calendar_hash)`。Worker 已经按 Callable 注册项持有 `CalendarCache`，桶内再按调度签名选择一个计划。显式桶和精确参数键使用不同的键类型标记。Graph 在每次 bind 中通过现有 Definition framing、边界、kernel 注册和任务形态校验后，才参与 Calendar 查找。

调度签名从当前 bind 的已解析 Graph 中构造，使用字段等值确定命中。签名至少覆盖以下输入。

- **拓扑**。按本地任务编号排列的 body 任务及其 fan-in 关系、sink 和 completion credits。Graph shell 与外部任务的关系继续由本次 bind 的阶段划分和计数器构建处理。
- **执行形态**。按角色排列的 kernel ID、active mask、逻辑 block 数、subtask 数、控制任务、`sync_start`、predicate 及调度器使用的其他执行或完成属性。
- **核资源**。当前 planner 使用的 worker 数与可用 AIC/AIV lane 和 cluster 布局。

当前标量值、张量地址、参数池偏移和通知地址由本次调用绑定。张量元数据变化仍需通过各自的 Definition 和参数校验，调度签名依据变化产生的实际 DAG 和任务形态判定。键的 hash 可用于加速查找，签名等值检查负责区分 hash 碰撞。

同一 Callable 注册期间，同一 `graph_id` 和相等 Config 必须录出相同的 Graph 调度结构。A5 Percounter 在 Calendar 查找或训练前核对本次录制的结构，发现变化时返回 bind 错误并指出第一个不同的任务或字段。换桶或省略桶号都适用。Config 按类型和 `operator==` 判等，稳定身份独立于本次 bind 分配的 `graph_key`。

4-block 和 8-block 结构由不同 Config 表达时，即使同桶，也各自训练并缓存对应的计划。后续提交分别命中各自的计划。

### 训练、重新绑定和释放

**跨 run 的复用保证。** 同一个 Worker 的 Callable 注册项持有 `CalendarCache`，一次 run 结束后缓存继续存活。run A 成功完成并用有效样本发布 plan 后，run B 重新执行 host bind。只要还是同一个 Callable 注册项，且 Graph、桶号和调度签名都相同，run B 就命中该 plan。两次 bind 各自分配 Definition 的 `graph_key`，Calendar 使用跨 run 稳定的 `graph_id`。注销 Callable 或结束 Worker 后，新注册项从冷缓存开始。 [Callable 缓存所有权](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/common/platform/onboard/host/device_runner_base.cpp#L2906-L2938) · [完成反馈时机](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/common/platform/onboard/host/c_api_shared.cpp#L2532-L2552)

冷路径继续使用现有初始 Calendar、设备采样和 Host PGO。待训练记录与已发布计划使用同一个完整键，兼容的并发准备可以共享已构造的初始计划，同一键由一个在途训练拥有采样。单次 bind 内有多个匹配调用时，沿用由最后一个匹配调用提供样本的规则。成功完成且样本有效后发布不可变的 `CalendarPlan`。bind 失败、执行失败或样本无效时，待训练记录随现有资源清理释放，后续提交可以重新训练。

每个调用使用本次 Definition 独立物化实参、kernel entry 和 predicate，并使用本次准备的计数器，按现有 Graph 边界规则执行。可跨调用共享不可变的计划和调度行布局。已 bind 的运行和直接 replay 继续执行其已准备的镜像。实际 CntNotify 租约继续在 bind 时申请并校验。 [Graph 镜像与逐次绑定](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/a5/runtime/percounter/host/calendar_image.h#L313-L375) · [现有训练流程](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/a5/runtime/percounter/host/calendar_cache.h#L130-L170)

## 实施落点

1. 扩展公共 Graph 提交重载和 Host orchestration 调用边界，让原始 `graph_id` 与可选桶号在公共的外层任务提交点进入 `GraphPendingUpload`。录制首次提交、在途 Definition 和已录制 Definition 路径走这个提交点。在 `bind_graph_definitions` 按 shell task ID 取出调用元数据并传给 A5 Calendar planner。元数据留在 Host 侧，设备 Graph Definition 和 Calendar wire layout 沿用当前协议。其他支持 Graph 提交的 runtime 接受同一 API，按各自已有的调度方式运行。 [现有外层任务提交点](https://github.com/hw-native-sys/simpler/blob/e5c0a697e90f100bb2d4445d1b8afda7f54ed3cb/src/common/host_build_graph/host/graph_submit.cpp#L432-L477)
2. 在 Calendar `state.h` 和 `calendar_cache.h` 接入显式桶查找。每个 bind 内对同一已解析 Definition 复用结构签名计算结果。复用现有训练中和训练完成的缓存容器，避免在 Worker 内额外保留每次调用的实参或 Definition 镜像。计划数量随实际访问的桶和结构变体数增长，因此节省量取决于 Lib 的桶划分。
3. 沿用现有 `MISS`、`REUSE`、`HIT`、`STORE` 诊断，补足区分 Graph 阶段和结果归属所需的信息。单测直接核对 plan 和训练项数量，场景测试通过正常提交入口核对事件和结果。接口落地时同步更新 Graph 执行文档和 A5 Calendar PGO 文档。

Calendar 调度器位于独立的 A5 `percounter` runtime。草稿 PR #3262 跟踪迁移到 A5 `host_build_graph`。 [A5 Calendar 迁移 PR #3262](https://github.com/hw-native-sys/simpler/pull/3262)

## 验收

| 提交场景 | 预期结果 |
| -------- | -------- |
| 同一 Worker 和 Callable 注册项执行两个独立 run。等待 run A 成功发布 plan 后再提交 run B，保持 Graph、桶号和调度签名相同，改变标量和输入输出缓冲区 | 目标 Graph 的事件为 run A `MISS`、`STORE`，run B `HIT`。两次输出分别符合各自实参，run B 的目标 Graph 无需再次采样。 |
| 在两个 run 中改变同一 Graph 的 Definition 分配顺序，使其取得不同的 `graph_key` | 目标 Graph 在第二个 run 仍命中已发布计划。 |
| 同一 bind 中提交多个兼容调用，覆盖首次录制、在途录制和已录制 Definition | 每个 shell 保留自己的桶号与实参。同键只产生一个采样调用。 |
| 同 Graph 同结构使用不同桶号，或不同 Graph、不同 Callable 使用相同桶号 | 各自冷启动。Graph 和 Callable 隔离用结构相同的 body 验证。 |
| 同 Graph 同桶使用不同 Config 交替提交 4-block 和 8-block，另外覆盖合法 DAG 或核拓扑变化 | 每个合法的新签名先 miss 并训练，后续分别命中。非法形态仍报原有错误。 |
| 同 Graph、相等 Config 跨 bind 录出不同 block 数，分别使用原桶、新桶和精确参数键 | bind 报错，指出首个不一致的任务及字段，不查找或启动 Graph Calendar 训练。 |
| 同一输入分别省略选项及显式传 `0` | 默认路径继续按精确参数查找，显式 `0` 独立成桶。 |
| 首次训练失败或样本无效，随后重试，另起 Worker 或重新注册 Callable | 失败样本未发布，重试可训练，新的缓存所有者重新冷启动。 |

先用 C++ 单测验证键等值、隔离和训练生命周期，再用 A5sim scene 验证 `rt_submit_graph` 到执行结果的正常路径。结构变化用有效的 Config 或独立 bind 生成，训练失败用确定性故障注入覆盖。正常提交的数值结果和任务依赖均需验证。

A5 真机使用同一实现、同一设备和输入序列，对照开启 `calendar_hash` 与省略选项。至少选择一个具有多个原本独立精确键的 Lib Graph，覆盖桶内低、中、高工作量和边界，轮换首个训练输入。分别记录 Graph 阶段的 plan 条目数、冷采样次数和命中率，并按输入报告预热后的 AICore wall P50 和 P95。对于固定结构、包含 `N` 个不同精确键的单桶，预期从 `N` 个成功训练和 plan 条目收敛到 `1` 个。AICore wall 的建议验收线为各代表输入 P50 回退不超过 5%、P95 不超过 10%。超线时由 Lib 细分该桶并复测。
