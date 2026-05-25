# M5 设计说明

`PTO2_CLUSTER_MODE=5`：在硬件被迫跨 cluster 部署时（如 3+3 cluster），让 PTO2 runtime 的性能恢复到接近"全 4 线程同 cluster"基线。

## 背景

A3 / 未来同代芯片 AICPU 拓扑为 2 个 cluster，cluster 内 cache 一致性远比跨 cluster 廉价。生产配置 `aicpu_thread_num=4`（1 orch + 3 sched）。

**当前 A3 (CANN 9.0.0)** 自然分布是 4+2，gate 选 4 同 cluster ⇒ orch 和 sched 全在一边，没有跨 cluster 开销。这是 M0 baseline。

**未来受限拓扑**（如 cluster 内只 3 核）必然要跨 cluster。先前实验摸清了几种方案：

| Mode | 拓扑 | C1 Orch vs M0 | 备注 |
|---|---|---|---|
| M2 | 1 orch + 3 sched 跨 cluster | +19% | 最差 |
| M3 | 1 orch + 1 sched 同 cluster，2 sched 另 cluster；sched_idx 随机 | +11% | 中间 |
| M4 | 同 M3 拓扑；wiring 线程 (sched_idx 0) 强制跟 orch 同 cluster | +8% | 不够 |
| **M5** | M4 + 软件层约束 SM 写者 + spin cleanup | **+1.5%** | 目标解 |

M4 仅靠 placement 把 wiring queue 留在 orch cluster，但**剩 2 个 sched 仍跨 cluster 写 `last_task_alive`**，orch 每次 alloc 读 SM 都跨 cluster 拉。

## 设计目标

跨 cluster 拓扑下，让 SM 的关键 cache line `last_task_alive` **只被 orch cluster 内的线程写**，消除远端 sched 写带来的跨 cluster invalidate。

## 三个组件

M5 同时做 3 件事，缺一不可。

### 1. sync_to_sm 对非本地 sched 短路

`sync_to_sm()` 是 `advance_ring_pointers()` 末尾把本地 `last_task_alive` 写回 SM 的步骤。M5 加入短路：

```cpp
void sync_to_sm() {
    if (g_only_thread0_advances) return;   // M5: remote sched 不写 SM
    ring->fc.last_task_alive.store(last_task_alive, release);
}
```

`g_only_thread0_advances` 是 thread_local，在 `aicpu_executor::run()` 入口按 `runtime->cluster_mode == 5 && thread_idx != 0` 决定。

效果：

- sched 0 (跟 orch 同 cluster) 正常写 SM
- 远端 2 个 sched 完成 task 时仍然推进**本地 `RingSchedState::last_task_alive`**（这是 RingSchedState 内的 int32_t，所有 sched 共享），但**不把它写回 SM 的 `ring->fc.last_task_alive`**

副作用：SM 滞后于本地。需要 #2 让 sched 0 兜底，需要 #3 让 orch 容忍滞后。

### 2. sched 0 conditional publish

只把"非 sched-0 把 SM 写关掉"不够 —— SM 永远滞后会让 orch 误以为没有 task 完成。需要 sched 0 兜底 publish。

```cpp
// scheduler_dispatch.cpp 的 dispatch 主循环开头：
static thread_local int32_t last_published[PTO2_MAX_RING_DEPTH] = {0,0,0,0};
if (thread_idx == 0 && runtime->cluster_mode == 5) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto &rss = sched_->ring_sched_states[r];
        int32_t cur = rss.last_task_alive;
        if (cur != last_published[r]) {
            rss.ring->fc.last_task_alive.store(cur, release);
            last_published[r] = cur;
        }
    }
}
```

关键：**conditional**（只在变化时写），避免 unconditional 写带来的 sched 0 自身开销。如果远端 sched 推进了 local，sched 0 看到 `cur != last_published`，publish 一次；否则 0 store。

### 3. new_entry() spin + 主动 cleanup（关键）

只做 #1 #2 会触发原本被掩盖的 bug：**orch 的 tensormap entry pool 复用速度跟不上 entry 释放速度**。

#### 为什么会卡

orch 用 SM 的 `last_task_alive` 触发 tensormap cleanup：

```cpp
sync_tensormap(task_id, sm_last_task_alive) {
    cleanup_retired(ring, last_cleanup, sm_last_task_alive);
    last_cleanup = sm_last_task_alive;
}
```

在 M5 下 SM 滞后于真实进度 → `cleanup_retired` 清理范围保守 → free_entry_list 没及时补充 → orch 下一次 `new_entry()` 找不到空闲 entry → 走 `always_assert(next_entry_idx < pool_size)` 直接挂。

类似地，alloc / heap reclaim 路径都依赖 SM `last_task_alive`，但 alloc 本身**已经会 spin**（while loop），所以"看到旧值就 spin 等"天然安全。只有 tensormap 这条路是 fire-and-forget。

#### 解法 A：new_entry spin

```cpp
PTO2TensorMapEntry *new_entry() {
    while (true) {
        if (free_num > 0) return free_entry_list[--free_num];
        if (next_entry_idx < pool_size) return &entry_pool[next_entry_idx++];
        // 池耗尽：重读 SM 触发新一轮 cleanup
        int32_t before = free_num;
        try_cleanup_all_rings_from_sm();
        if (free_num == before) {
            __asm__ volatile("yield");   // 让 sched 0 跑
        }
    }
}
```

`try_cleanup_all_rings_from_sm()` 重读 SM `last_task_alive` 跑 cleanup，让 orch 主动消化最新进度。

#### 解法 B：cleanup_retired 范围 cap

仅靠 A 还不够。`cleanup_retired` 在范围 ≥ `task_window_size` 时**会 wrap**：

```cpp
for (int32_t local_id = old; local_id < new; ++local_id) {
    int32_t slot = local_id & (task_window_size - 1);
    // 当 (new - old) >= window，多个 local_id 哈希到同一 slot
    // task_entry_heads[slot] 在前一次循环已被清空 → 但被当成新 local 的入口
    // 同一组 entry 被 free 两次 → free_entry 的 always_assert 挂
}
```

M5 下 SM 一次可能跳很大（远端 sched 已经 advance 但 sched 0 还没 publish 时，sched 0 一觉醒来 publish 跨度可能 >> window）。Cap 后 cleanup 每次只清 `task_window_size - 1`，多次调用消化：

```cpp
int32_t max_step = task_window_sizes[r] > 0 ? task_window_sizes[r] - 1 : 0;
if (max_step > 0 && target - last_cleanup[r] > max_step) {
    target = last_cleanup[r] + max_step;
}
cleanup_retired(r, last_cleanup[r], target);
last_cleanup[r] = target;
```

cap 同时打到 `sync_tensormap` 和新的 `try_cleanup_all_rings_from_sm`，因为 M5 下 sync_tensormap 也可能拿到大跳的 SM 值。

## 不变量

- **本地 `RingSchedState::last_task_alive`** 仍然由所有 sched 在 `advance_ring_pointers` 末尾推进（thread-safe via `advance_lock` try-lock）
- **SM `ring->fc.last_task_alive`** 在 cluster_mode=5 时只由 sched 0 写
- SM 永远 ≤ 本地（因为 sched 0 是从本地拷贝过去的）
- orch 永远拿到 SM 的旧值，按"保守估计"行事；spin 等到 sched 0 推

orch 的三条 SM 消费路径（窗口检查 / heap reclaim / tensormap cleanup）现在统一**容忍旧值 + spin 等待**：

| 路径 | 旧值行为 | M5 改动 |
|---|---|---|
| alloc 窗口 | 觉得窗口紧 → spin | 无改动（本来就 spin） |
| alloc heap | 觉得 heap 不够 → spin | 无改动（本来就 spin） |
| tensormap entry | 之前：assert；现在：spin + 主动 cleanup | new_entry + try_cleanup_all_rings_from_sm |
| tensormap cleanup 范围 | 之前：可能 wrap → double-free；现在：cap 限 1 个 window | sync_tensormap + try_cleanup_all_rings 都加 cap |

## 测量结果

paged_attention_unroll on A3，100 rounds trimmed avg：

| Mode | C1 Orch | C1 Total | C2 Orch | C2 Total |
|---|---|---|---|---|
| M0 (4 同 cluster) | **854** | 1163 | 499 | 626 |
| M4 (1+1+2 placement only) | 919 (+8%) | 1185 (+2%) | 494 | 654 (+4.5%) |
| **M5** | **867 (+1.5%)** | 1180 (+1.5%) | 489 | 649 (+3.6%) |

M5 把 C1 Orch 的 8% regression 收到 1.5% —— 受限拓扑下基本追平 M0。

## 失败的早期版本（学习记录）

### M5 v1：non-sched-0 跳过 advance，置 `pending_advance` flag，sched 0 poll

死。non-sched-0 完全不动本地 last_task_alive，sched 0 自己 idle 时没人推 ring pointer，orch alloc 卡死。

### M5 v2：sync_to_sm 短路 + sched 0 unconditional publish

正确运行但 orch profile TOTAL 反而比 M4 略差。sched 0 每 iter 4 个 atomic store 的开销盖过了 cross-cluster line 留本地的收益。

### M5 v3：sync_to_sm 短路 + sched 0 conditional publish（首次）

assert 挂。SM 滞后时 cleanup 没跟上，pool 用尽 → always_assert。这次失败暴露了**v2 之所以"能跑"只是因为 SM 永远不滞后**。

### M5 v4：v3 + new_entry spin（但 cleanup 没 cap）

assert 挂在 `free_entry` 而不是 `new_entry`。原因：SM 跳跃太大，cleanup_retired 范围跨 `task_window_size` → 同一 entry 被 free 两次。

### M5 v6（终版）

v4 + cleanup_retired 范围 cap。所有路径都正确。

## 适用条件

启用 `PTO2_CLUSTER_MODE=5` 的前提：

1. AICPU 拓扑允许 2+2 分布（每 cluster ≥ 2 个 active 线程）
2. CANN 9.0+（其它版本 affinity gate 行为可能不同）
3. `aicpu_thread_num=4`（其它配置的 thread_idx/role 分配未验证）

在 A3 当前 CANN 上（4+2 ideal），M5 实际上**没必要** —— M0 default 已经把 4 个全放同 cluster，没跨 cluster 开销。M5 是为**未来受限拓扑**准备的解。

## 已知限制

- **sched 0 的 publish 是 lazy 的**：依赖 sched 0 iter 的频率。如果 sched 0 因为各种原因 stall（如长时间没 ready task），SM 会大幅滞后，orch 的 spin 会变长。实测在 paged_attention 上没观察到，但 workload 不同可能暴露。
- **C2 Total 仍比 M0 高 3.6%**：剩余开销主要在 sched 侧，远端 2 个 sched 仍然在 cluster B 上跑完成处理 + 跟 sched 0 同步。这部分是 M5 无法解的本质开销（必须有线程在远端）。
- **`try_cleanup_all_rings_from_sm` 跑全部 4 个 ring**：实际 paged_attention 只用 1 ring，3 次无效循环。可以加 ring 数缓存优化，但影响小。
- **`g_only_thread0_advances` 是 thread_local**：每个 sched 线程都要在 `run()` 入口设置。当前 `aicpu_executor::run()` 已经做了。

## 后续工作

1. 把 `try_cleanup_all_rings_from_sm` 改成 lazy（仅当 pool 真用尽才扫所有 ring）
2. sched 0 publish 可以再加 epoch counter，避免长时间没事时永远不 publish 的极端情况
3. 在真正 3+3 受限硬件上验证 —— A3 现在的 4+2 拓扑只能模拟

## 相关文件

```text
src/a2a3/runtime/tensormap_and_ringbuffer/
├── runtime/
│   ├── scheduler/
│   │   ├── pto_scheduler.h           # sync_to_sm 短路；g_only_thread0_advances 声明
│   │   ├── pto_scheduler.cpp         # g_only_thread0_advances 定义
│   │   └── scheduler_dispatch.cpp    # sched 0 conditional publish
│   ├── pto_tensormap.h               # new_entry spin
│   └── shared/
│       └── pto_tensormap.cpp         # try_cleanup_all_rings_from_sm + cap
├── aicpu/
│   └── aicpu_executor.cpp            # g_only_thread0_advances 赋值
├── runtime/
│   └── runtime.h                     # cluster_mode 字段
└── host/
    └── runtime_maker.cpp             # PTO2_CLUSTER_MODE env 读取
```

## 复现

```bash
source .venv/bin/activate
pip install --no-build-isolation .

# Baseline (M0)
task-submit --device auto --run \
    "cd $(pwd) && source .venv/bin/activate && \
     unset PTO2_CLUSTER_MODE && \
     bash tools/benchmark_rounds.sh -p a2a3 -d \$TASK_DEVICE -n 100"

# M5
task-submit --device auto --run \
    "cd $(pwd) && source .venv/bin/activate && \
     export PTO2_CLUSTER_MODE=5 && \
     bash tools/benchmark_rounds.sh -p a2a3 -d \$TASK_DEVICE -n 100"
```
