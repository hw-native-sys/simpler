# MoE Multi-Chip Testing Guide

This guide provides detailed commands for testing the distributed MoE implementation on Ascend hardware.

## Prerequisites

```bash
# Activate conda environment
conda activate simpler_issue

# Ensure environment variables are set
export PTOAS_ROOT=/usr/local/bin/ptoas-bin
export ASCEND_PROCESS_LOG_PATH=/data/fangjingzhi/simpler_distributed/device_log
export ASCEND_GLOBAL_LOG_LEVEL=0
```

## Test Files

| Test File | Purpose | Phase | Notes |
|-----------|---------|-------|-------|
| `test_dispatch_only.py` | Test dispatch phase only | Dispatch | Uses unique values for data tracing |
| `test_combine_only.py` | Test combine phase only | Combine | Uses unique values for data tracing |
| `test_dispatch_compute.py` | Test dispatch + compute | Dispatch + Compute | Verifies expert routing and compute |
| `test_end2end.py` | Test complete end-to-end pipeline | All phases | Uses independent scratch buffers to avoid conflicts |

## Test Commands



### Hardware Mode (a2a3)

Run on actual Ascend NPUs.

#### Quick Tests (2 chips)

```bash
# Dispatch phase test
python examples/workers/l3/moe_multi_chip_experts/test_dispatch_only.py \
  -p a2a3 \
  -d 10,11

# Combine phase test
python examples/workers/l3/moe_multi_chip_experts/test_combine_only.py \
  -p a2a3 \
  -d 10,11

# End-to-end pipeline test (recommended)
python examples/workers/l3/moe_multi_chip_experts/test_end2end.py \
  -p a2a3 \
  -d 10,11
```

#### Extended Tests (4 chips)

```bash
# 4-chip full pipeline
python examples/workers/l3/moe_multi_chip_experts/test_moe_multi_chip.py \
  -p a2a3 \
  -d 10,11,12,13
```

## Background Job Submission

For long-running tests, use `task-submit` to run in background.

```bash
# Submit combine-only test
task-submit --device 10,11 --run \
  "export PTOAS_ROOT=/usr/local/bin/ptoas-bin && \
   ASCEND_PROCESS_LOG_PATH=/data/fangjingzhi/simpler_distributed/device_log && \
   ASCEND_GLOBAL_LOG_LEVEL=0 && \
   python examples/workers/l3/moe_multi_chip_experts/test_combine_only.py \
   -p a2a3 -d 10,11 > moe_combine_only_$(date +%Y%m%d_%H%M%S).log 2>&1"

# Submit full pipeline test
task-submit --device 10,11 --run \
  "export PTOAS_ROOT=/usr/local/bin/ptoas-bin && \
   ASCEND_PROCESS_LOG_PATH=/data/fangjingzhi/simpler_distributed/device_log && \
   ASCEND_GLOBAL_LOG_LEVEL=0 && \
   python examples/workers/l3/moe_multi_chip_experts/test_moe_multi_chip.py \
   -p a2a3 -d 10,11 > moe_full_$(date +%Y%m%d_%H%M%S).log 2>&1"
```



## Test Verification

### Expected Output

Each test will print:
1. **Configuration**: Platform, device count, tensor shapes
2. **Input data**: Sample values for verification
3. **Scratch buffer**: Debug output from Phase 1 (stage-in)
4. **Output data**: Final results after combine
5. **Verification**: Match with golden output

### test_end2end.py 特殊说明

**关键特性**:
- 使用唯一值初始化输入: `(card * 1000000) + (expert * 10000) + (token * 100) + dim`
- 使用**独立的 scratch 缓冲区**避免阶段间冲突:
  - `scratch`: 用于 Dispatch + Compute 阶段
  - `scratch_test`: 用于 Combine 阶段
- 清晰的数据流追踪

**为什么需要独立的 scratch?**
- Dispatch 向 `scratch` 写入: `scratch[card_j][expert_i][:][:]`
- Combine 从 `scratch` 读取: `scratch[expert_i][my_rank][:][:]`
- Combine 的写入范围 (前 COUNT 个 token) 不能完全覆盖 Dispatch 的数据
- 使用独立 buffer 避免读到残留数据

### Success Criteria

```
✓ All values correct
✓ Output matches golden reference
✓ No device errors or timeouts
```

## Debugging Failed Tests

### Check Device Logs

```bash
# List latest device logs
ls -lt /data/fangjingzhi/simpler_distributed/device_log/debug/device-*/ | head -20

# Check specific device log for errors
grep -i "error\|fail\|stuck" \
  /data/fangjingzhi/simpler_distributed/device_log/debug/device-10/*.log
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Parameter mismatch | `kernel_id=-1`, STUCK-READY | Check tensor/scalar count matches kernel signature |
| Device fault | `Device fault, ret=0x7110011` | Check for illegal memory access or uninitialized tiles |
| Timeout | Task hangs, no progress | Check HCCL bootstrap and signal barrier logic |
| Wrong results | Output doesn't match golden | Verify data flow through dispatch→combine phases |

### Enable Verbose Logging

```bash
# Maximum verbosity for debugging
ASCEND_GLOBAL_LOG_LEVEL=0 \
ASCEND_PROCESS_LOG_PATH=/data/fangjingzhi/simpler_distributed/device_log \
python examples/workers/l3/moe_multi_chip_experts/test_combine_only.py \
  -p a2a3 -d 10,11
```


## Test Isolation

Each test creates unique temporary files:

```bash
# Rootinfo files for HCCL
/tmp/pto_*_PID*.bin

# Device logs
/data/fangjingzhi/simpler_distributed/device_log/debug/device-*/
```

