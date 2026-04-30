# 调试信息说明

## 案例 1: End-to-End MoE Pipeline Scratch 缓冲区冲突问题

### 问题描述
在实现完整的 MoE pipeline（Dispatch + Compute + Combine）时，发现 Card 1 的 Expert 0 输出错误：
- **期望值**: 2.0 (1.0 input + 1.0 compute)
- **实际值**: 1.0 (只有 input，没有 compute)

### 调试过程

#### 步骤 1: 创建 Isolated Combine Test
**假设**: Combine 阶段本身有问题

**实现**: 在 test_end2end.py 中添加独立的 combine 测试
- 创建 `host_recv_test`: 填充正确的 2.0 值
- 创建 `host_output_test`: 用于存储 isolated test 的输出
- 创建 `host_scratch_print_test`: 独立的 debug 输出
- 创建 `scratch_test` buffer: 独立的 HCCL scratch 缓冲区
- 在 orchestrator 中添加 Part 2: Isolated Combine Test

**结果**: 
- ✅ Isolated Test: 所有 256 个值正确 (2.0)
- ❌ Full Pipeline: Card 1 的 Expert 0 仍然错误 (1.0)

**结论**: Combine 阶段本身是正确的，问题不在 combine kernel

#### 步骤 2: 分析数据流
重新分析数据流，确认问题所在：

**Dispatch 阶段**:
- Input: `send[card_i][expert_i][:][:]` = 1.0
- Output: `recv[card_i][card_j][:][:]` = `send[card_j][expert_i][:][:]`
- 对于 Card i: 从所有 Card j 接收 `send[j][i][:][:]`

**Compute 阶段**:
- Input: `recv[:][:4][:]`
- Output: `recv[:][:4][:] += 1.0`
- 所有 recv 的前 4 个 token 都加 1.0

**Combine 阶段**:
- Phase 1 (stage-in): 复制 `recv[:][:][:]` 到 `scratch[my_rank][card_j][:][:]`
- Phase 3 (direct-store): 从 `scratch[expert_i][my_rank][:][:]` 读取到 `output[expert_i][:][:]`

#### 步骤 3: 发现 Scratch 缓冲区冲突
**关键观察**:
- Full Pipeline 使用同一个 `scratch` buffer
- Isolated Test 使用独立的 `scratch_test` buffer → 成功！

**问题定位**:
当 Full Pipeline 复用同一个 scratch buffer 时：
1. Dispatch Phase 向 `scratch` 写入数据（布局: `scratch[card_j][expert_i][:][:]`）
2. Combine Phase 1 **应该**向 `scratch` 写入 `recv` 数据（布局: `scratch[my_rank][card_j][:][:]`）
3. Combine Phase 3 从 `scratch` 读取数据

**问题**:
- Combine Phase 1 只写入前 COUNT (4) 个 token
- Combine Phase 3 的 stride 使用 NUM_TOKENS (10) 计算 offset
- **Combine Phase 1 没有完全覆盖 Dispatch Phase 写入的数据**
- Combine Phase 3 读到了 Dispatch Phase 的残留数据

#### 步骤 4: 解决方案
**方案**: 为 Combine Phase 使用独立的 scratch 缓冲区

**实现**:
1. 在 `ChipBootstrapConfig` 中添加第二个 scratch buffer:
   ```python
   ChipBufferSpec(
       name="scratch_test",
       dtype="float32",
       count=scratch_count,
       nbytes=total_scratch_nbytes,
   )
   ```

2. 在 orchestrator 中:
   - Dispatch + Compute: 使用 `ext_scratch`
   - Combine: 使用 `ext_scratch_test`

3. 在 Python 中:
   - 添加 `contexts[i].buffer_ptrs["scratch_test"]`

**结果**: ✅ Full Pipeline 完全正确

### 关键经验

1. **隔离测试的重要性**:
   - 通过创建 isolated combine test，快速定位问题不在 combine kernel 本身
   - 这种方法可以推广到其他多阶段 pipeline 的调试

2. **缓冲区复用的陷阱**:
   - 当多个阶段使用同一个 scratch buffer 时：
     - **确保每个阶段完全覆盖**它写入的区域
     - **注意写入范围和读取范围的不匹配**
   - Phase 1 写入前 COUNT 个 token，但 Phase 3 的 stride 基于 NUM_TOKENS

3. **调试技巧**:
   - 使用唯一值初始化输入（而不是全 1.0）
   - 值编码: `(card_id * 1000000) + (expert_id * 10000) + (token * 100) + dim`
   - 这样可以清楚追踪每个数据点的流向

4. **独立的 HCCL 缓冲区**:
   - 如果不确定 buffer 是否被正确覆盖，使用独立 buffer
   - 内存成本: 2x scratch buffer (对于小 buffer 可以接受)
   - 避免了复杂的状态清理逻辑

### 相关文件
- `test_end2end.py`: 完整的 end-to-end 测试
- `moe_end2end_orch.cpp`: 使用独立 scratch_test 的 orchestrator
- `moe_combine_alltoall2.cpp`: Combine kernel

### 运行测试
```bash
source /data/miniconda3/etc/profile.d/conda.sh && \
conda activate simpler_issue && \
task-submit --device 10,11 --run \
  "export PTOAS_ROOT=/usr/local/bin/ptoas-bin && \
   ASCEND_PROCESS_LOG_PATH=device_log \
   ASCEND_GLOBAL_LOG_LEVEL=0 \
   python examples/workers/l3/moe_multi_chip_experts/test_end2end.py -p a2a3 -d 10,11"
```

---

## 添加的调试点

### Python 侧 (main.py)
1. **run() 函数入口**: 跟踪程序启动
2. **HCCL 配置**: 显示 scratch buffer 大小和 rootinfo 路径
3. **Tensor 分配**: 确认内存分配成功
4. **Worker 创建**: 跟踪 Worker 对象创建
5. **内核编译阶段**:
   - 编译 dispatch kernel
   - 编译 compute kernel
   - 编译 combine kernel
   - 提取 ELF text sections (硬件)
   - 编译 orchestration
6. **Worker 初始化**: 跟踪 init() 进度
7. **chip_contexts**: 显示每个 card 的 rank 和 device_ctx
8. **orch_fn**: 跟踪任务提交进度
9. **worker.run()**: 跟踪执行进度

### C++ Orchestration 侧 (moe_comm_orch.cpp)
1. **orchestration_entry 入口**: 显示 card_id, expert_id, num_cards, comm_ctx
2. **阶段 1 (Dispatch)**: 任务提交前后的状态
3. **阶段 2 (Compute)**: 任务提交前后的状态
4. **阶段 3 (Combine)**: 任务提交前后的状态
5. **完成**: 确认所有阶段完成

所有输出都使用 `flush=True` 或 `fflush(stdout)` 确保立即写入日志。

## 运行测试

```bash
# 重新运行测试，观察调试输出
source /data/miniconda3/etc/profile.d/conda.sh && \
conda activate simpler_issue && \
task-submit --device 4,5,6,7 --run "export PTOAS_ROOT=/usr/local/bin/ptoas-bin && python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3 -d 4,5,6,7 > moe_multi_chip_test_4chip_debug.log 2>&1"
```

## 可能的问题定位

### 情况 1: 卡在内核编译
**症状**: 看到 "[moe_multi_chip] [DEBUG] Starting kernel compilation..." 但没有后续输出
**原因**: 可能是 PTOAS_ROOT 路径不正确或编译器问题
**解决**: 检查 PTOAS_ROOT 环境变量和 ptoas-bin 目录

### 情况 2: 卡在 Worker.init()
**症状**: 看到 "Worker created" 但没有 "Worker initialized"
**原因**: 可能是 HCCL 初始化或设备通信问题
**解决**: 检查设备之间的 HCCL 通信配置

### 情况 3: 卡在 worker.run()
**症状**: 看到 "About to call worker.run()" 但没有看到 orchestration 输出
**原因**: 可能是任务提交或调度问题
**解决**: 检查 runtime 配置和任务队列

### 情况 4: 卡在某个阶段
**症状**: 看到 "Stage X: ..." 但没有 "Stage X+1"
**原因**: 可能是该阶段的 AIV 内核或 HCCL 通信问题
**解决**: 检查对应阶段的内核代码和通信逻辑

## 下一步

1. 运行带调试信息的测试
2. 观察最后一条成功的调试消息
3. 根据卡住的位置定位问题
4. 如果需要，在更具体的位置添加更详细的调试信息
