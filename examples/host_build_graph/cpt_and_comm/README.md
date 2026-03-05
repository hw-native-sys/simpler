# cpt_and_comm

多卡「先计算，再通信」示例：GEMM → WindowMemCopyIn → TGATHER → WindowMemCopyOut。

## 流程

1. **GEMM**：每卡执行 C = A @ B（64x64）
2. **WindowMemCopyIn**：将 dev_C 前 64 元素拷贝到 HCCL window
3. **TGATHER**：root 从各 rank 收集到本地 window
4. **WindowMemCopyOut**：root 将 gathered 结果拷贝到 dev_out

## 依赖

- CANN（libhccl.so、libacl.so）
- pto-comm-isa（`pto::comm::TGATHER`、`ParallelGroup`）
- a2a3 真机（不支持 sim）

## 运行

```bash
# 设置 pto-comm-isa 路径
export PTO_COMM_ISA_ROOT=/path/to/pto-comm-isa

# 2 卡运行
python examples/scripts/multi_card_run_example.py \
  -k examples/host_build_graph/cpt_and_comm/kernels \
  -g examples/host_build_graph/cpt_and_comm/golden.py \
  --n-devices 2 --first-device 0
```

## 配置

- `RUNTIME_CONFIG.requires_comm`: True
- `RUNTIME_CONFIG.n_devices`: 2
- `RUNTIME_CONFIG.root`: 0
