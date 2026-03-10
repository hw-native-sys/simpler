# AllGather (Manual RDMA 实现)

多卡 AllGather 通信算子：使用 **直接 RDMA 读取**（HcclRemotePtr + TLOAD/TSTORE）实现。每个 rank 获得所有 rank 数据的拼接结果。

**实现方式**：`WindowMemCopyIn -> CommBarrier -> AllGatherManual -> WindowMemCopyOut -> CommBarrier(post)`

- 无 TGATHER 集体调用，所有 rank 并行执行
- 适用于性能对比测试

## 运行

```bash
python examples/scripts/multi_card_run_example.py \
    -k examples/host_build_graph/allgather_Manual/kernels \
    -g examples/host_build_graph/allgather_Manual/golden.py
```

需要设置 `PTO_COMM_ISA_ROOT` 指向 pto-comm-isa 根目录，以及多卡 HCCL 环境。
