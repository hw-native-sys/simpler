# Gather

纯 gather 通信算子：仅保留多卡之间的 TGATHER 通信，无计算。

流程：WindowMemCopyIn → CommBarrier → TGATHER → WindowMemCopyOut (root only)

每个 rank 有本地 src 数据（GATHER_COUNT=64 个 float），root 将各 rank 的前 GATHER_COUNT 个元素收集到 out 中。

## 运行

```bash
python examples/scripts/multi_card_run_example.py \
    -k examples/host_build_graph/gather/kernels \
    -g examples/host_build_graph/gather/golden.py
```

需要设置 `PTO_COMM_ISA_ROOT` 指向 pto-comm-isa 根目录，以及多卡 HCCL 环境。
