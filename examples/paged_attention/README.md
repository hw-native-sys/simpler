# Paged Attention AIC/AIV 子图切分实现

本样例展示了如何将 Paged Attention 算法按照计算特性手动切分为 AIC 和 AIV 子图，并在 a2a3sim 仿真平台上验证正确性。

## 算法原理

### Paged Attention

Paged Attention 是一种用于 LLM 推理的高效 KV cache 管理机制，将 KV cache 组织为固定大小的 block（page），每个 block 包含 `block_size` 个 token 的 K 和 V。

**输入：**

- `query`: (batch, num_heads, head_dim) - 当前 query
- `key_cache`: (total_blocks, block_size, kv_head_num, head_dim) - K cache
- `value_cache`: (total_blocks, block_size, kv_head_num, head_dim) - V cache
- `block_table`: (batch, block_num) - block 索引表
- `context_lens`: (batch,) - 每个序列的有效长度

**输出：**

- `out`: (batch, num_heads, head_dim) - attention 结果

### Online Softmax

为避免一次性加载所有 KV 到内存，Paged Attention 采用 **Online Softmax** 算法，逐 block 计算并累积结果：

```python
for each block bn:
    # Step 1: QK MatMul
    sij = qi @ kj.T  # (q_tile_size, valid_len)
    
    # Step 2: Softmax 预处理
    sij_scale = sij * scale_value
    mij = rowmax(sij_scale)           # (q_tile_size,)
    pij = exp(sij_scale - mij)        # (q_tile_size, valid_len)
    lij = rowsum(pij)                 # (q_tile_size,)
    
    # Step 3: PV MatMul
    oi_new = pij @ vj  # (q_tile_size, head_dim)
    
    # Step 4: Online 累积更新
    if bn == 0:
        mi, li, oi = mij, lij, oi_new
    else:
        mi_new = max(mi, mij)
        alpha = exp(mi - mi_new)
        beta = exp(mij - mi_new)
        li = alpha * li + beta * lij
        oi = alpha * oi + beta * oi_new
        mi = mi_new
    
    # Step 5: 最终归一化（融合在最后一个 block 中）
    if bn == last_block:
        out = oi / li
```

## AIC/AIV 子图切分方案

根据计算特性，将 Paged Attention 切分为 4 个 kernel（normalize 已融合进 online_update）：

| Kernel | Core Type | 功能 | 输入 | 输出 |
|--------|-----------|------|------|------|
| `aic_qk_matmul` | AIC (Cube) | Q @ K^T 矩阵乘法 | qi, kj | sij |
| `aiv_softmax_prepare` | AIV (Vector) | scale, rowmax, exp, rowsum | sij, scale | pij, mij, lij |
| `aic_pv_matmul` | AIC (Cube) | P @ V 矩阵乘法 | pij, vj | oi_new |
| `aiv_online_update` | AIV (Vector) | Online Softmax 累积 + 归一化 | mij, lij, oi_new, mi, li, oi, is_last | mi, li, oi, out |

### Task Graph 结构

对每个 (batch, head) 的计算流程：

```
┌─────────────────────────────────────────────────────────────┐
│                      Block Loop                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  [AIC] QK MatMul: sij = qi @ kj^T                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  [AIV] Softmax Prepare: pij, mij, lij = f(sij)       │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  [AIC] PV MatMul: oi_new = pij @ vj                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  [AIV] Online Update: mi, li, oi = f(mij, lij, ...)  │   │
│  │        (if is_last: out = oi / li)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Kernel 融合优化

本实现将 `aiv_normalize` 融合进了 `aiv_online_update`：

**原实现（5 个 kernel）：**
```
QK MatMul → Softmax Prepare → PV MatMul → Online Update → Normalize
```

**融合后（4 个 kernel）：**
```
QK MatMul → Softmax Prepare → PV MatMul → Online Update (+ Fused Normalize)
```

**优势：**
- 减少 task 数量：每个 (batch, head) 少 1 个 normalize task
- 避免额外的内存读写：`oi` 和 `li` 不需要再次从内存读取

## Task Graph 并行化优化

### 依赖分析

```
QK[bn] → SF[bn] → PV[bn] → UP[bn]
```

- QK → SF → PV：**不同 block 之间无依赖**，可以完全并行
- UP：依赖上一轮的 UP 结果（Online Softmax 累加器），**必须串行**
- 不同 (batch, head) 之间完全独立，可以并行

### 并行化策略

**串行版本（优化前）：**
```
Head 0: QK[0]→SF[0]→PV[0]→UP[0]→QK[1]→SF[1]→PV[1]→UP[1]→...
Head 1: (等待 Head 0 全部完成)→QK[0]→SF[0]→PV[0]→UP[0]→...

初始就绪任务：1
```

**全并行版本（优化后）：**
```
Head 0:  QK[0]→SF[0]→PV[0]─┬─→UP[0]
         QK[1]→SF[1]→PV[1]─┼──────→UP[1]
         QK[2]→SF[2]→PV[2]─┼─────────────→UP[2]
         QK[3]→SF[3]→PV[3]─┴─────────────────────→UP[3]

Head 1:  QK[0]→SF[0]→PV[0]─┬─→UP[0]                        (与 Head 0 并行!)
         QK[1]→SF[1]→PV[1]─┼──────→UP[1]
         ...

初始就绪任务：32（= batch × num_heads × block_num = 2 × 4 × 4）
```

### Buffer 分配策略

为实现 batch 间、head 间和 block 间的并行，中间 buffer 按以下方式分配：

| Buffer | 分配粒度 | 索引方式 | 原因 |
|--------|---------|---------|------|
| sij, pij, mij, lij, oi_new | per-batch × per-head × per-block | `[(b*H+h)*B+bn]` | QK/SF/PV 跨 batch/head/block 并行 |
| mi, li, oi | per-batch × per-head | `[b*H+h]` | UP 在 (batch,head) 内串行，跨 (batch,head) 并行 |

## 目录结构

```
paged_attention_example/
├── README.md                   # 说明文档
├── main.py                     # 主程序入口
├── golden.py                   # Python golden 参考实现
├── gen_mermaid_graph.py        # Mermaid 依赖图生成脚本
├── task_graph.md               # 完整 Task 依赖图
├── task_graph_b0h0.md          # Batch 0, Head 0 的依赖图
└── kernels/
    ├── kernel_config.py
    ├── aic/
    │   ├── aic_qk_matmul.cpp
    │   └── aic_pv_matmul.cpp
    ├── aiv/
    │   ├── aiv_softmax_prepare.cpp
    │   └── aiv_online_update.cpp
    └── orchestration/
        └── paged_attention_orch.cpp
```

## 编译和运行

```bash
cd /data/w00949583/simpler/examples/paged_attention_example
python3 main.py
```

### 生成依赖图

```bash
# 生成完整依赖图
python3 gen_mermaid_graph.py --output task_graph.md

# 生成单个 (batch, head) 的依赖图（更易读）
python3 gen_mermaid_graph.py --batch 0 --head 0 --output task_graph_b0h0.md
```

## 测试结果

```
batch=2, num_heads=4, head_dim=128
block_size=64, block_num=4

Allocated 32 per-batch-per-head-per-block buffers
Allocated 8 per-batch-per-head accumulators
Created 128 tasks (full parallel)

Initially ready tasks: AIC=32, AIV=0

SUCCESS: All 1024 elements match within tolerance
```

## 关键技术点

### 无外部库函数调用

a2a3sim 从 `.o` 文件中直接提取 `.text` section 作为可执行代码，跳过了链接步骤。
因此 kernel 代码不能调用任何标准库函数（如 `expf`, `memcpy`, `fmax`），否则会导致 bus error。

### Fused Kernel 设计

`aiv_online_update` 通过 `is_last` 参数判断是否执行归一化：

```cpp
if (is_last) {
    for (int i = 0; i < q_tile; i++) {
        float inv_li = 1.0f / li[i];
        for (int j = 0; j < head_dim; j++)
            dst[i * head_dim + j] = oi[i * head_dim + j] * inv_li;
    }
}
```

## License

本示例代码仅供学习和研究使用。
