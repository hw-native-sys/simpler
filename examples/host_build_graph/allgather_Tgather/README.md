# AllGather (TGATHER 实现)

多卡 AllGather 通信算子：使用 **N 次顺序 TGATHER** 实现。每个 rank 获得所有 rank 数据的拼接结果。

**实现方式**：`for r in [0, n_ranks): Barrier -> Gather(root=r) -> [rank r: WindowMemCopyOut] -> Barrier(post)`

- 仅 root 调用 TGATHER，避免多 rank 同时调用导致的死锁
- 适用于性能对比测试

## 运行

```bash
python examples/scripts/multi_card_run_example.py \
    -k examples/host_build_graph/allgather_Tgather/kernels \
    -g examples/host_build_graph/allgather_Tgather/golden.py
```

需要设置 `PTO_COMM_ISA_ROOT` 指向 pto-comm-isa 根目录，以及多卡 HCCL 环境。
