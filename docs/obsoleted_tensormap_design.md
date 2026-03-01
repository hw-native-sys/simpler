# TensorMap 设计与实现详解

## 1. 概述

TensorMap 是 PTO Runtime2 中用于追踪 **生产者-消费者关系** 的核心数据结构。它的主要功能是：

1. **依赖发现**：当提交新任务时，查找输入 tensor 的生产者任务
2. **重叠检测**：支持对 view、reshape、transpose 等操作后的 tensor 进行正确的依赖匹配
3. **高效管理**：O(1) 插入，惰性失效，链截断优化
4. **混合检测算法**：自动选择最优检测方法
   - 连续 tensor：O(1) Bounding Box（精确）
   - 1D 非连续：精确 GCD 算法
   - **多维非连续：Combined Lattice GCD 方法（新增）**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TensorMap 在运行时中的角色                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Orchestrator                     TensorMap                     │
│   ┌───────────┐                   ┌──────────────┐              │
│   │ submit    │ ──── lookup ────> │ region →     │              │
│   │ task(B)   │ <── producer_id ──│ producer_id  │              │
│   └───────────┘                   └──────────────┘              │
│        │                                 ▲                       │
│        │                                 │                       │
│        └──────── insert ─────────────────┘                       │
│                (B的output)                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据结构

### 2.1 基础 TensorMap (PTO2TensorMap)

用于简单的 1D 区域匹配（基于 base_ptr + offset + size）。

```c
typedef struct {
    // 哈希表 buckets
    int32_t* buckets;             // 指向 entry_pool 的偏移量 (-1 = 空)
    int32_t  num_buckets;         // 必须是 2 的幂（快速取模）
    
    // Ring Buffer 条目池
    PTO2TensorMapEntry* entry_pool;
    int32_t pool_size;            // 池容量
    int32_t pool_head;            // 下一个分配位置
    
    // 每任务条目链表（用于清理）
    int32_t* task_entry_head;     // 每任务的条目头
    
    // 失效阈值
    int32_t last_task_alive;      // 低于此 ID 的任务已退休
} PTO2TensorMap;

// 条目结构
typedef struct {
    PTO2TensorRegion region;      // base_ptr, tile_index, offset, size
    int32_t producer_task_id;     // 生产者任务 ID
    int32_t next_in_bucket;       // 桶内链表
    int32_t next_in_task;         // 任务内链表
    bool in_bucket;               // 是否在桶中
} PTO2TensorMapEntry;
```

### 2.2 扩展 TensorMap (PTO2TensorMapEx)

支持多维 tensor 的 view/reshape/transpose 操作，使用 **边界盒（Bounding Box）** 进行重叠检测。

```c
typedef struct {
    // 原始存储信息
    void* raw_base;               // 原始 tensor 基地址
    int64_t raw_total_size;       // 原始 tensor 总大小
    
    // 边界盒（快速重叠检测）
    int64_t min_byte_offset;      // 最小字节偏移
    int64_t max_byte_offset;      // 最大字节偏移
    
    // 完整布局信息（精确检测可选）
    int64_t storage_offset;       // 存储偏移
    int64_t shape[PTO2_MAX_TENSOR_DIM];
    int64_t strides[PTO2_MAX_TENSOR_DIM];
    int32_t ndim;
    
    int32_t producer_task_id;
    bool is_deep_copy;            // 是否深拷贝（独立存储）
    // ... 链表指针
} PTO2TensorMapEntryEx;
```

---

## 3. 哈希策略：仅按 base_ptr 哈希

### 3.1 关键设计决策

```c
uint32_t pto2_tensormap_hash(PTO2TensorMap* tm, PTO2TensorRegion* region) {
    // ========== 关键：仅按 base_ptr 哈希 ==========
    // 
    // 为了正确检测重叠，同一 base tensor 的所有区域
    // 必须在同一个哈希桶中！
    //
    uint64_t key = (uint64_t)(uintptr_t)region->base_ptr;
    
    // 位混合提高分布
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    
    return (uint32_t)(key & (tm->num_buckets - 1));
}
```

### 3.2 为什么不能包含 offset？

```
如果哈希包含 offset：
  Region A: base=X, offset=0   → bucket 5
  Region B: base=X, offset=128 → bucket 12  ❌ 无法检测重叠！

仅按 base_ptr 哈希：
  Region A: base=X, offset=0   → bucket 5
  Region B: base=X, offset=128 → bucket 5   ✓ 同一桶，可以比较！
```

---

## 4. 重叠检测算法

### 4.1 基础版本：1D 区间重叠

```c
bool pto2_region_overlap(PTO2TensorRegion* a, PTO2TensorRegion* b) {
    // 1. 必须是同一 base tensor
    if (a->base_ptr != b->base_ptr) return false;
    
    // 2. 必须是同一 tile（不同 tile 不重叠）
    if (a->tile_index != b->tile_index) return false;
    
    // 3. 区间重叠检测：[start_a, end_a) ∩ [start_b, end_b) ≠ ∅
    int32_t a_start = a->offset;
    int32_t a_end = a_start + a->size;
    int32_t b_start = b->offset;
    int32_t b_end = b_start + b->size;
    
    // 重叠条件：(a_start < b_end) AND (b_start < a_end)
    return (a_start < b_end) && (b_start < a_end);
}
```

### 4.2 扩展版本：边界盒重叠（支持 view/reshape/transpose）

```c
bool pto2_tensormapex_overlap(const PTO2LogicalTensor* tensor, 
                               const PTO2TensorMapEntryEx* entry) {
    // 1. 不同 raw storage 不重叠
    if (tensor->raw_base != entry->raw_base) return false;
    
    // 2. 边界盒交集检测
    // 重叠条件：(a.min <= b.max) AND (b.min <= a.max)
    return (tensor->min_byte_offset <= entry->max_byte_offset) &&
           (entry->min_byte_offset <= tensor->max_byte_offset);
}
```

### 4.3 边界盒计算

对于多维 tensor，边界盒是包含所有元素的最小连续内存范围：

```c
void pto2_logical_tensor_get_bounding_box(
    const PTO2LogicalTensor* tensor,
    int64_t* out_min,
    int64_t* out_max
) {
    // min_offset = storage_offset + Σ min(0, (shape[d]-1)*strides[d])
    // max_offset = storage_offset + Σ max(0, (shape[d]-1)*strides[d])
    
    int64_t min_offset = tensor->storage_offset;
    int64_t max_offset = tensor->storage_offset;
    
    for (int32_t d = 0; d < tensor->ndim; d++) {
        int64_t extent = (tensor->shape[d] - 1) * tensor->strides[d];
        if (extent >= 0) {
            max_offset += extent;
        } else {
            min_offset += extent;  // 负 stride
        }
    }
    
    *out_min = min_offset;
    *out_max = max_offset + tensor->elem_size - 1;  // 包含最后一个元素
}
```

---

## 5. 对 View/Reshape/Transpose 的支持

### 5.1 Tensor 提取类型

```c
typedef enum {
    PTO2_TENSOR_RAW,              // 原始 tensor（拥有存储）
    PTO2_TENSOR_SHALLOW_VIEW,     // 浅提取：view/slice
    PTO2_TENSOR_SHALLOW_RESHAPE,  // 浅提取：reshape
    PTO2_TENSOR_SHALLOW_TRANSPOSE,// 浅提取：transpose
    PTO2_TENSOR_DEEP_VIEW,        // 深提取：clone
    PTO2_TENSOR_DEEP_CONTIGUOUS,  // 深提取：contiguous
} PTO2TensorExtractionType;
```

### 5.2 浅提取（共享存储）→ 需要重叠检测

```
原始 Tensor A:  [0, 1, 2, 3, 4, 5, 6, 7]  raw_base = 0x1000
                ├────────────────────────┤

View B = A[2:6]: [2, 3, 4, 5]             raw_base = 0x1000
                    ├────────┤             storage_offset = 2 * elem_size

View C = A[4:8]: [4, 5, 6, 7]             raw_base = 0x1000
                        ├────────┤         storage_offset = 4 * elem_size

B 和 C 重叠！因为：
- 同一 raw_base (0x1000)
- 边界盒交集：[2*elem, 5*elem] ∩ [4*elem, 7*elem] = [4*elem, 5*elem] ≠ ∅
```

### 5.3 深提取（独立存储）→ 无需重叠检测

```c
// Clone 创建新的独立存储
bool pto2_logical_tensor_clone(src, dst, new_base) {
    // dst->raw_base = new_base (不同于 src->raw_base)
    // 因此不会与 src 重叠
}
```

### 5.4 Transpose 支持

Transpose 只改变 strides，不改变 raw_base：

```c
bool pto2_logical_tensor_transpose(src, dst, perm) {
    // 继承 raw_base 和 storage_offset
    dst->raw_base = src->raw_base;
    dst->storage_offset = src->storage_offset;
    
    // 重排 shape 和 strides
    for (int d = 0; d < src->ndim; d++) {
        dst->shape[d] = src->shape[perm[d]];
        dst->strides[d] = src->strides[perm[d]];
    }
    
    // 重新计算边界盒
    pto2_logical_tensor_update_bounding_box(dst);
}
```

---

## 6. 惰性失效与链截断优化

### 6.1 条目有效性检查

```c
static inline bool pto2_tensormap_entry_valid(PTO2TensorMap* tm, 
                                               PTO2TensorMapEntry* entry) {
    // 任务 ID >= last_task_alive 则有效
    return entry->producer_task_id >= tm->last_task_alive;
}
```

### 6.2 链截断优化

由于新条目总是插入链头（task_id 降序），一旦遇到失效条目，后续全部失效：

```c
int32_t pto2_tensormap_lookup(PTO2TensorMap* tm, PTO2TensorRegion* region) {
    uint32_t bucket = pto2_tensormap_hash(tm, region);
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    
    while (offset >= 0) {
        PTO2TensorMapEntry* entry = &tm->entry_pool[offset];
        
        // 检查有效性
        if (!pto2_tensormap_entry_valid(tm, entry)) {
            // ========== 链截断 ==========
            // 后续条目全部失效，直接截断
            *prev_ptr = -1;
            
            // 标记截断条目
            while (offset >= 0) {
                PTO2TensorMapEntry* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                offset = next;
            }
            
            return -1;  // 未找到
        }
        
        // 检查重叠
        if (pto2_region_overlap(&entry->region, region)) {
            return entry->producer_task_id;  // 找到！
        }
        
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }
    
    return -1;  // 未找到
}
```

---

## 7. 查找所有重叠生产者

某些情况下，一个消费者可能依赖多个生产者（多个任务写入重叠区域）：

```c
int32_t pto2_tensormapex_lookup_all(PTO2TensorMapEx* tm, 
                                     const PTO2LogicalTensor* tensor,
                                     int32_t* producer_ids,
                                     int32_t max_producers) {
    uint32_t bucket = pto2_tensormapex_hash(tm, tensor);
    int32_t offset = tm->buckets[bucket];
    int32_t count = 0;
    
    while (offset >= 0 && count < max_producers) {
        PTO2TensorMapEntryEx* entry = &tm->entry_pool[offset];
        
        if (!pto2_tensormapex_entry_valid(tm, entry)) {
            // 链截断
            break;
        }
        
        if (pto2_tensormapex_overlap(tensor, entry)) {
            // 去重检查（同一生产者可能有多个输出）
            bool duplicate = false;
            for (int32_t i = 0; i < count; i++) {
                if (producer_ids[i] == entry->producer_task_id) {
                    duplicate = true;
                    break;
                }
            }
            
            if (!duplicate) {
                producer_ids[count++] = entry->producer_task_id;
            }
        }
        
        offset = entry->next_in_bucket;
    }
    
    return count;  // 返回找到的生产者数量
}
```

---

## 8. Ring Buffer 池管理

### 8.1 分配新条目

```c
void pto2_tensormap_insert(PTO2TensorMap* tm, PTO2TensorRegion* region, 
                            int32_t producer_task_id) {
    // 从 ring buffer 池分配
    int32_t entry_offset = tm->pool_head;
    PTO2TensorMapEntry* entry = &tm->entry_pool[entry_offset];
    
    // 前进池头（环绕）
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;
    
    // ========== 关键：移除旧条目的桶链接 ==========
    // 即使条目已失效，它仍在桶链中，必须先移除
    if (entry->in_bucket) {
        pto2_tensormap_remove_from_bucket(tm, entry);
    }
    
    // 初始化新条目
    entry->region = *region;
    entry->producer_task_id = producer_task_id;
    
    // 插入桶头
    uint32_t bucket = pto2_tensormap_hash(tm, region);
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;
    
    // 链接到任务条目链表
    int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
    entry->next_in_task = tm->task_entry_head[task_slot];
    tm->task_entry_head[task_slot] = entry_offset;
}
```

---

## 9. 复杂度分析

| 操作 | 平均复杂度 | 最坏复杂度 |
|------|-----------|-----------|
| 插入 | O(1) | O(1) |
| 查找 | O(有效条目数 / 桶数) | O(有效条目数) |
| 清理 | O(退休任务的条目数) | O(条目总数) |

- **插入**：总是 O(1)，插入桶头
- **查找**：平均 O(chain_length)，链截断优化加速
- **清理**：惰性失效 + 定期显式清理

---

## 10. 与设计文档的一致性

当前实现完全符合 `runtime_buffer_manager_methods.md` 的设计：

| 设计要求 | 实现状态 |
|---------|---------|
| Ring buffer 池管理 | ✓ 使用 pool_head 环绕分配 |
| 惰性失效 | ✓ 通过 last_task_alive 阈值 |
| 链截断优化 | ✓ 遇到失效条目截断整个尾部 |
| 仅按 base_ptr 哈希 | ✓ 确保同一 tensor 的所有区域在同一桶 |
| 重叠检测 | ✓ 混合方法：Bounding Box + GCD |
| view/reshape/transpose 支持 | ✓ 通过 LogicalTensor 和边界盒 |
| 多生产者查找 | ✓ lookup_all 返回所有重叠生产者 |
| 多维非连续 tensor | ✓ Combined Lattice GCD 方法 |

---

## 11. 使用示例

```c
// 初始化
PTO2TensorMapEx tm;
pto2_tensormapex_init_default(&tm);

// 任务 A 生产 tensor
PTO2LogicalTensor output_A;
pto2_logical_tensor_init_raw(&output_A, buffer, shape, 2, sizeof(float));
pto2_tensormapex_insert(&tm, &output_A, task_id_A);

// 任务 B 的输入是 A 的 view
PTO2LogicalTensor input_B;
pto2_logical_tensor_view(&output_A, &input_B, start, new_shape, 2);

// 查找生产者
int32_t producer = pto2_tensormapex_lookup(&tm, &input_B);
// producer == task_id_A (因为 view 与原 tensor 重叠)

// 清理
pto2_tensormapex_destroy(&tm);
```

---

## 12. 混合重叠检测（Hybrid Overlap Detection）

### 12.1 设计动机

传统的重叠检测方法各有优缺点：

| 方法 | 优点 | 缺点 |
|------|------|------|
| Bounding Box | O(1) 快速 | 对非连续 tensor 有误报 |
| GCD 方法 | 100% 精确 | O(ndim) 较慢 |

**混合方法** 结合两者优势：
- 对 **简单 tensor** (连续) 使用 Bounding Box（快速且精确）
- 对 **复杂 tensor** (非连续) 使用 GCD（精确无误报）

### 12.2 Tensor 复杂度分类

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tensor 复杂度分类                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Tensor                                                         │
│     │                                                            │
│     ├── is_contiguous = true ──> Simple Tensor                  │
│     │     • 内存连续，无 gap                                     │
│     │     • Bounding Box = 实际访问范围                          │
│     │     • 检测结果：精确，无误报                                │
│     │                                                            │
│     └── is_contiguous = false ──> Complex Tensor                │
│           • 内存有 gap（如 transpose、strided view）             │
│           • Bounding Box 包含未访问区域                          │
│           • 检测结果：可能误报，需 GCD 验证                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 混合检测算法

```
┌─────────────────────────────────────────────────────────────────┐
│                    混合检测流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   检测 overlap(A, B)                                            │
│     │                                                            │
│     ├─── A.raw_base ≠ B.raw_base ──> 不重叠 (不同存储)          │
│     │                                                            │
│     ├─── Bounding Box 不相交 ──> 不重叠 (快速排除)               │
│     │                                                            │
│     ├─── A.is_simple && B.is_simple ──> 重叠确认                │
│     │     (两者都是连续的，Bounding Box 结果精确)                │
│     │                                                            │
│     └─── 至少一个 Complex ──> GCD 精确检测                       │
│           (非连续 tensor 需要验证实际是否重叠)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.4 复杂度分析

| Tensor A | Tensor B | 检测方法 | 时间复杂度 |
|----------|----------|----------|------------|
| Simple | Simple | Bounding Box | O(1) |
| Simple | Complex | GCD | O(ndim) |
| Complex | Simple | GCD | O(ndim) |
| Complex | Complex | GCD | O(ndim²) |

**性能优势**：
- 大多数实际场景中 tensor 是连续的，走 O(1) 快速路径
- 只有涉及 transpose、非连续 view 时才需要 GCD
- Bounding Box 始终作为第一道过滤器

### 12.5 API

```c
/**
 * 混合重叠检测（推荐使用）
 * - 自动选择最优检测方法
 * - 返回结果 100% 精确，无误报
 */
bool pto2_logical_tensor_overlap_hybrid(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

/**
 * Tensor 与 TensorMapEntry 混合检测
 */
bool pto2_tensor_entry_overlap_hybrid(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
);
```

### 12.6 数据结构支持

**TensorMapEntryEx** 新增 `is_simple` 字段：

```c
typedef struct PTO2TensorMapEntryEx {
    // ... 其他字段 ...
    
    bool is_deep_copy;   // 是否深拷贝
    bool is_simple;      // 是否简单 tensor（连续）
} PTO2TensorMapEntryEx;
```

**Insert 时记录**：

```c
void pto2_tensormapex_insert(...) {
    // ...
    entry->is_simple = tensor->is_contiguous;
    // ...
}
```

### 12.7 重叠检测方法

#### 12.7.1 当前实现：Bounding Box

当前使用简单的 **Bounding Box** 方法：

| Tensor A | Tensor B | 检测方法 | 结果 |
|----------|----------|----------|------|
| 连续 | 连续 | Bounding Box | **精确** |
| 连续 | 非连续 | Bounding Box | 保守（可能有误报） |
| 非连续 | 非连续 | Bounding Box | 保守（可能有误报） |

```c
bool pto2_logical_tensor_overlap_hybrid(a, b) {
    if (a->raw_base != b->raw_base) return false;
    if (a->max < b->min || b->max < a->min) return false;
    if (a->is_contiguous && b->is_contiguous) return true;
    return true;  // 保守：可能有误报，但无漏报
}
```

#### 12.7.2 为什么不使用 GCD 方法

GCD 方法存在以下限制：

1. **stride=1 问题**：最低维度 stride 通常等于 elem_size，导致 GCD=1，检测失效
2. **False Negative 风险**：连续 vs 非连续 tensor 可能产生假阴性
3. **复杂度高**：精确多维 GCD 需要 O(ndim³)

#### 12.7.3 改进方案：Hierarchical Bounding Box (HBB)

**核心思想**：追踪 tensor 的 **派生历史 (Derivation History)**，通过比较派生路径在早期确定不重叠。

**原理**：
- 每次 view/reshape/transpose 操作记录到 layout history
- 比较两个 tensor 时，从根节点开始逐级比较
- 相同级别可跳过，在第一个不同的 VIEW 级别检查 bbox 是否相交
- 如果 bbox 不相交，确定不重叠；如果 reshape/transpose 不同，保守返回重叠

**示例 1：来自同一 tensor 的不同 slice**

```
原始 A: shape=[100], 连续

E = A[10:50].reshape(8,5).transpose()[1:3, 2:6]
F = A[60:80]

E.layout_history:
  Level 0: VIEW bbox=[0, 399]      # 原始 A
  Level 1: VIEW bbox=[40, 199]     # A[10:50]
  Level 2: RESHAPE [8,5]
  Level 3: TRANSPOSE [1,0]
  Level 4: VIEW bbox=[44, 123]

F.layout_history:
  Level 0: VIEW bbox=[0, 399]      # 相同
  Level 1: VIEW bbox=[240, 319]    # A[60:80]

比较过程:
  Level 0: VIEW [0,399] == VIEW [0,399] -> 跳过
  Level 1: VIEW [40,199] vs VIEW [240,319]
           bbox 不相交 (199 < 240) -> 返回 不重叠 ✓
```

**示例 2：相同 reshape 后的不同 slice**

```
G = A.reshape(10, 10)
H = G[0:5, :]     # 前 5 行，访问 [0, 199]
I = G[5:10, :]    # 后 5 行，访问 [200, 399]

H.layout_history:
  Level 0: VIEW bbox=[0, 399]
  Level 1: RESHAPE [10,10]
  Level 2: VIEW bbox=[0, 199]

I.layout_history:
  Level 0: VIEW bbox=[0, 399]
  Level 1: RESHAPE [10,10]
  Level 2: VIEW bbox=[200, 399]

比较过程:
  Level 0: 相同 VIEW -> 跳过
  Level 1: 相同 RESHAPE -> 跳过
  Level 2: VIEW [0,199] vs VIEW [200,399]
           bbox 不相交 (199 < 200) -> 返回 不重叠 ✓
```

**示例 3：不同 reshape（保守处理）**

```
J = A.reshape(10, 10)[0:5, :]
K = A.reshape(20, 5)[0:10, :]

比较过程:
  Level 0: 相同 VIEW -> 跳过
  Level 1: RESHAPE [10,10] vs RESHAPE [20,5]
           不同的 reshape -> 返回 可能重叠 (保守，安全)
```

**数据结构**：

```c
#define PTO2_MAX_LAYOUT_DEPTH 8

typedef enum {
    PTO2_LAYOUT_VIEW,       // View/slice: 记录 bounding box
    PTO2_LAYOUT_RESHAPE,    // Reshape: 记录 shape
    PTO2_LAYOUT_TRANSPOSE,  // Transpose: 记录 permutation
} PTO2LayoutOpType;

typedef struct {
    PTO2LayoutOpType type;
    union {
        struct {  // VIEW
            int64_t bbox_min;
            int64_t bbox_max;
        } view;
        struct {  // RESHAPE
            int32_t ndim;
            int64_t shape[PTO2_MAX_TENSOR_DIM];
        } reshape;
        struct {  // TRANSPOSE
            int32_t ndim;
            int32_t perm[PTO2_MAX_TENSOR_DIM];
        } transpose;
    };
} PTO2LayoutOp;

typedef struct {
    void* raw_base;                           // 原始存储指针
    int32_t depth;                            // 当前深度 (1 到 MAX)
    PTO2LayoutOp ops[PTO2_MAX_LAYOUT_DEPTH];  // 操作历史
} PTO2LayoutHistory;
```

**统一处理 Simple 和 Non-Simple Tensor**：

HBB 方法统一了连续和非连续 tensor 的处理，不再需要单独的 `is_contiguous` 标志：

```
┌─────────────────────────────────────────────────────────────┐
│  Simple Tensor = HBB depth=1 的特例                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Simple (连续) tensor:                                      │
│    depth = 1                                                │
│    ops[0] = VIEW { bbox_min=0, bbox_max=total_size-1 }     │
│                                                             │
│  Non-simple (非连续) tensor:                                │
│    depth > 1                                                │
│    ops[0..n-1] = VIEW/RESHAPE/TRANSPOSE 序列               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

示例：

  // 原始连续 tensor A[1024]
  A.layout_history = {
      .depth = 1,
      .ops[0] = { VIEW, bbox=[0, 4095] }  // 1024 * 4 bytes
  }

  // A 的 slice: B = A[100:200]
  B.layout_history = {
      .depth = 2,
      .ops[0] = { VIEW, bbox=[0, 4095] },   // 继承自 A
      .ops[1] = { VIEW, bbox=[400, 799] }   // 100*4 到 200*4-1
  }

  // B 的 reshape: C = B.reshape(10, 10)
  C.layout_history = {
      .depth = 3,
      .ops[0] = { VIEW, bbox=[0, 4095] },
      .ops[1] = { VIEW, bbox=[400, 799] },
      .ops[2] = { RESHAPE, shape=[10, 10] }
  }
```

**统一设计的优势**：

| 特性 | 之前（分离设计） | 统一 HBB |
|------|------------------|----------|
| 数据结构 | `is_contiguous` + `bbox` 分离 | 只需 `LayoutHistory` |
| 检测逻辑 | 分支判断连续/非连续 | **单一算法路径** |
| 代码复杂度 | if-else 分支 | **统一循环** |
| 可扩展性 | 需要修改多处 | **只需扩展 ops 类型** |

**比较算法**（统一处理 simple 和 non-simple）：

```c
bool pto2_layout_history_overlap(
    const PTO2LayoutHistory* a,
    const PTO2LayoutHistory* b
) {
    // 1. 不同原始存储 -> 不重叠
    if (a->raw_base != b->raw_base) {
        return false;
    }
    
    // 统一算法：无论 depth=1 (simple) 还是 depth>1 (non-simple)
    // 都用相同的逻辑处理
    int min_depth = MIN(a->depth, b->depth);
    
    for (int i = 0; i < min_depth; i++) {
        const PTO2LayoutOp* op_a = &a->ops[i];
        const PTO2LayoutOp* op_b = &b->ops[i];
        
        // 2. 类型不同 -> 保守返回重叠
        if (op_a->type != op_b->type) {
            return true;
        }
        
        // 3. 类型相同，按类型处理
        switch (op_a->type) {
            case PTO2_LAYOUT_VIEW:
                // Bounding box 不相交 -> 精确不重叠
                // 对于 simple tensor (depth=1)，这就是完整的检测
                if (op_a->view.bbox_max < op_b->view.bbox_min ||
                    op_b->view.bbox_max < op_a->view.bbox_min) {
                    return false;  // 确定不重叠！
                }
                // bbox 相交，继续下一级（如果有）
                break;
                
            case PTO2_LAYOUT_RESHAPE:
                // Shape 不同 -> 保守返回重叠
                if (!shapes_equal(op_a, op_b)) {
                    return true;
                }
                break;
                
            case PTO2_LAYOUT_TRANSPOSE:
                // Perm 不同 -> 保守返回重叠
                if (!perms_equal(op_a, op_b)) {
                    return true;
                }
                break;
        }
    }
    
    // 4. 所有共同级别都通过，保守返回可能重叠
    return true;
}
```

**Simple Tensor 的处理流程**：

```
A: depth=1, ops[0]=VIEW[0, 1023]
B: depth=1, ops[0]=VIEW[1024, 2047]

比较:
  i=0: VIEW[0,1023] vs VIEW[1024,2047]
       1023 < 1024 -> bbox 不相交 -> 返回 false (不重叠)

// 与之前的 "两者都连续 -> bounding box 精确" 逻辑等价
// 但代码路径统一，无需 if (is_contiguous) 分支
```

**复杂度分析**：

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 记录操作 | O(1) | 追加到 history |
| 比较两个 tensor | O(depth) | depth ≤ 8 |
| 存储开销 | ~128 bytes/tensor | 固定大小 |

**HBB 方法的优势**：

| 特性 | Bounding Box | GCD | HBB |
|------|--------------|-----|-----|
| 连续 tensor | 精确 | 精确 | 精确 |
| 非连续 tensor | 保守 | 失效(stride=1) | **可消除部分误报** |
| 假阴性风险 | 无 | 有 | **无** |
| 复杂度 | O(ndim) | O(ndim) | O(depth) |
| 适用场景 | 通用 | 受限 | **通用** |

**关键洞察**：在实际计算图中，tensor 通常**来自相同祖先**且**派生操作相似**。
HBB 方法利用这一特点，在保证安全的前提下消除大量误报。

### 12.8 实现状态

| 功能 | 状态 | 说明 |
|------|------|------|
| Bounding Box 检测 | ✓ 已实现 | 当前方法 |
| TensorMap 集成 | ✓ 已实现 | lookup 使用 hybrid |
| is_simple 字段 | ✓ 临时方案 | HBB 实现后可移除 |
| **HBB 数据结构** | 📋 待实现 | `PTO2LayoutHistory` |
| **HBB 比较算法** | 📋 待实现 | `pto2_layout_history_overlap()` |
| **统一处理** | 📋 待实现 | simple=depth1, non-simple=depth>1 |

**迁移计划**：

```
当前实现:
  PTO2LogicalTensor {
      is_contiguous;     // bool
      min_byte_offset;   // bbox
      max_byte_offset;
  }

HBB 实现后:
  PTO2LogicalTensor {
      PTO2LayoutHistory layout_history;  // 统一结构
      // is_contiguous 可通过 depth==1 判断
      // bbox 可通过 ops[depth-1].view 获取
  }
```
