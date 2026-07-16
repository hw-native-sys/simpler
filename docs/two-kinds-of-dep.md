# Two Kinds of Dependency

## 背景与起因

a2a3 `tensormap_and_ringbuffer` runtime 在 `TensorCreateInfo` 改造后，submit 阶段只接受 `input_param`/`inout_param` 使用 existing `Tensor`，`output_param` 只接受 `TensorCreateInfo` 创建新 Tensor。这一改造把 Tensor 的两种不同依赖语义混在了一起：

1. **Resource / Owner 依赖**：某个 task 需要另一个 task 真正持有或产生 buffer 地址才能执行（如 creator 与第一个使用者之间）。
2. **Execution / Modifier 依赖**：两个 task 只是需要按顺序执行在同一 Tensor 上，不需要延长 producer 的 buffer 生命周期。

在改造前，所有依赖都被统一按 RESOURCE 处理。当用例出现跨 scope 的 `INOUT` modifier 链（例如 `batch_paged_attention`）时，EXECUTION 边在 submit 阶段没有 `fanout_count++` 去 pin 住 modifier 的 producer。于是 modifier 可能在 consumer 已经 submit 但尚未 wire 之间被 CONSUME 并复用 slot，导致 wire 阶段把 consumer 接到错误任务的 `fanout_head` 上，最终 hang 住或报 `507018`。

`batch_paged_attention` 的回归失败是这一问题的最直接表现：在 baseline 中它能通过，而在引入跨 scope EXECUTION 边设计后失败。

相关范围：仅涉及 `src/a2a3/runtime/tensormap_and_ringbuffer/`；a5 等其他 runtime 不在本次改动范围内。

## 目标

在 a2a3 `tensormap_and_ringbuffer` runtime 中显式区分两种依赖：

| 类型 | 语义 | 生命周期作用 | 代表场景 |
|------|------|--------------|----------|
| **RESOURCE** | creator / owner 依赖 | 必须延长 producer 生命周期，直到 consumer 完成并释放 | `owner_task_id` 指向的 creator 依赖；buffer 必须存在 |
| **EXECUTION** | modifier 间执行顺序 | 只保证 modifier 之间按顺序执行，不阻止 producer 被释放或复用 | tensormap 中的 modifier 依赖；纯执行顺序 |

需要达成的具体效果：

1. **自动推导保留**
   - Step A（`owner_task_id`）→ 标记为 RESOURCE。
   - Step B（tensormap 中的 modifier）→ 标记为 EXECUTION。

2. **手动 API 收口**
   - `add_dep(...)` 表示 RESOURCE 依赖。
   - `add_dep_exec(...)` 表示 EXECUTION 依赖。
   - 对于旧代码中不区分 kind 的 `set_dependencies`，内部默认按 RESOURCE 处理。

3. **消除 race**
   - EXECUTION 边在 submit 时也必须 `fanout_count++`（pin 住 producer，跨 submit→wire 窗口）。
   - consumer 完成 wire 后，EXECUTION 边的 pin 可以立即释放，不等待 consumer 完成。
   - `on_task_release` 中只处理 RESOURCE 边的 `release_producer`，避免 double-release。

4. **正确性与性能验证**
   - `batch_paged_attention` 必须恢复通过。
   - 非相关用例（如 `alternating_matmul_add`、`qwen3_14b_decode` 等）性能无实质回归。

5. **范围控制**
   - 先在 a2a3 验证；a5 暂不动，待 a2a3 合格后再镜像。

## 实现现状

### 已完成的代码改动

| 文件 | 改动内容 |
|------|----------|
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_types.h` | 新增 `DepKind` 枚举，区分 `RESOURCE` 与 `EXECUTION` |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h` | `PTO2FaninSpillEntry` 中压入 kind；`PTO2TaskPayload` 新增 `fanin_inline_dep_kind_mask` |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_ring_buffer.h` | `for_each_fanin_storage` 回调签名改为 `(slot, DepKind)` |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_dep_compute.h` | `compute_task_fanin` 改为两趟：先收集并发射 RESOURCE，再发射 EXECUTION |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` | `append_fanin_or_fail` 增加 `DepKind` 参数；STEP 1 拆分为 1a（RESOURCE）与 1b（EXECUTION）；runtime emit 写入 kind mask；`FaninBuilder` 跟踪 kind |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/pto_scheduler.h` | `wire_task` 中 EXECUTION 边在接线后立即 `release_producer`；`on_task_release` 跳过 EXECUTION 边 |
| `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_arg_with_deps.h` | `Arg` 增加 `explicit_dep_kinds` 与 `set_dependencies_with_kinds`；`L0TaskArgsWithDeps` 增加 `add_dep_exec` |
| `src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/dep_gen_replay.cpp` | 更新 oracle emit lambda 签名以接收 `DepKind` |
| `tests/ut/cpp/a2a3/tensormap_and_ringbuffer/test_wiring.cpp` | 适配新回调签名与 packed entry |
| `tests/ut/cpp/a2a3/tensormap_and_ringbuffer/test_fanin_pool.cpp` | 适配新回调签名与 packed entry |

### Race 修复要点

- **问题**：EXECUTION 边若在 submit 时不 pin producer，则 modifier 在 consumer submit 后、wire 前可能被 CONSUME 并复用。
- **修复**：EXECUTION 边在 submit 时同样 `fanout_count++`；`wire_task` 接线后立即 `release_producer` 释放 pin；`on_task_release` 不再处理 EXECUTION 边。
- **结果**：consumer 已安全挂入 `fanout_head` 或 `early_finished`，而 producer 的 pin 释放时机提前到 wire 阶段，不会阻塞后续生命周期。

### 验证结果

#### 模拟测试
通过：
- `dummy_task`（modifier 链）
- `dep_gen`（oracle/annot 差分）
- `spmd_basic`
- `spmd_multiblock_mix`
- `mixed_example`
- `vector_example`

#### 板载测试
- `batch_paged_attention`：baseline PASS → 第一版跨 scope EXEC 设计 FAIL（507018）→ **修复后 PASS**。
- 全量 benchmark（`tools/benchmark_rounds.sh -n 20 -r tensormap_and_ringbuffer`）：**8/10 通过**；剩余 2 个失败为 `spmd_paged_attention` 的 Case1/Case2，在 baseline 中同样失败，属 pre-existing。

#### 性能对比（Total 时间，baseline vs 修复后）

| 用例 | 基线 | 修复后 | 抖动 |
|------|------|--------|------|
| alternating_matmul_add C1 | 817.8 | 840.3 | +2.8% |
| benchmark_bgemm C0 | 701.5 | 707.4 | +0.8% |
| paged_attention_unroll C1 | 1163.7 | 1154.1 | -0.8% |
| paged_attention_unroll C2 | 596.4 | 591.0 | -0.9% |
| paged_attention_unroll_manual_scope C1 | 1153.5 | 1150.3 | -0.3% |
| paged_attention_unroll_manual_scope C2 | 581.2 | 576.0 | -0.9% |
| batch_paged_attention C1 | 3225.7 | 3313.6 | +2.7% |
| qwen3_14b_decode | 2117.4 | 2123.1 | +0.3% |

所有差异均在 ±2.8% 内，与 device 0 共享环境下的噪声底一致（`alternating_matmul_add` 几乎无 INOUT 也 +2.8%）。

### 未决事项

- a5 的镜像改造尚未开始，等待 a2a3 验证合格后再进行。
- `batch_paged_attention` 的 +2.7% 可能包含 wire 阶段额外 `release_producer` 开销；若后续需要进一步压低，可考虑把 EXEC 边 pin 释放从 `wire_task` 下沉到 `on_task_complete`（需要给 `PTO2DepListEntry` 增加 kind 标记），当前方案未引入此复杂度。
