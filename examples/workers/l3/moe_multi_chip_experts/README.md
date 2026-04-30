# Multi-Chip MoE Example

This example demonstrates a distributed MoE (Mixture of Experts) pattern across **multiple chips**, with **one expert per chip**.

## Overview

This is the **multi-chip version** of `moe_single_chip`. The computation is **identical** - same kernels, same logic - but distributed across multiple chips for parallel execution.

## Key Difference: Single vs Multi-Chip

| Aspect | moe_single_chip | moe_multi_chip_experts |
|--------|----------------|------------------------|
| **Execution** | Sequential on one chip | **Parallel across chips** |
| **Expert placement** | All experts on one chip | **One expert per chip** |
| **Computation** | Same | **Same (identical kernels)** |
| **Performance** | Limited by single chip | **Scales with chip count** |
| **Result** | Deterministic | **Deterministic (same result)** |

## Pattern

```
Single-Chip Version (moe_single_chip):
  Input → [Chip 0: Expert 0,1,2,3] → Output

Multi-Chip Version (moe_multi_chip_experts):
  Input → [Chip 0: Expert 0] ─┐
         [Chip 1: Expert 1] ─┼→ Output
         [Chip 2: Expert 2] ─┤  (same result!)
         [Chip 3: Expert 3] ─┘
```

## Computation Flow (Identical to Single-Chip)

### 1. Dispatch Stage
- Copy data from send to recv buffer based on expert assignment
- Same kernel (`moe_demo_incore_0`) as single-chip version

### 2. Compute Stage
- Apply expert transformation on recv buffer
- Same kernel (`moe_demo_incore_1`) as single-chip version
- **Key difference**: Each chip runs only its assigned expert (parallel)

### 3. Combine Stage
- Accumulate results from recv to output
- Same kernel (`moe_demo_incore_2`) as single-chip version

## Kernels

Uses the **exact same kernels** as `moe_single_chip`:

1. **moe_demo_incore_0.cpp** (dispatch): Copy send → recv based on expert assignment
2. **moe_demo_incore_1.cpp** (compute): Apply expert transformation
3. **moe_demo_incore_2.cpp** (combine): Accumulate results to output

The kernels are NOT modified - we just distribute the work differently.

## Configuration

```python
# Device count determines expert count
NUM_CARDS = len(device_ids)  # e.g., 2, 4, etc.
NUM_EXPERTS = NUM_CARDS      # One expert per chip
NUM_TOKENS = 64
HIDDEN_DIM = 64
EXPERT_HIDDEN_DIM = 32
```

## Running

```bash
# 2 chips (2 experts) - simulation
python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3sim -d 0-1

# 4 chips (4 experts) - simulation
python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3sim -d 0-3

# 2 chips (2 experts) - hardware
python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3 -d 0-1

# Run via pytest
pytest examples/workers/l3/moe_multi_chip_experts/test_moe_multi_chip.py -v -s
```

## How It Works

### Python Level (main.py)

```python
# Allocate tensors per chip
host_input = [torch.randn(...) for _ in device_ids]
host_recv = [torch.randn(...) for _ in device_ids]
host_output = [torch.zeros(...) for _ in device_ids]

# Submit task to each chip
for i in range(len(device_ids)):
    orch.submit_next_level(moe_cc, moe_args, cfg, worker=i)
    # Each chip runs the SAME orchestration
    # But computes different experts based on chip ID
```

### Orchestration Level (moe_multi_chip_orch.cpp)

The orchestration code is identical to `moe_single_chip`:
- Loops over `card_i` (chip index) and `expert_j` (expert index)
- In multi-chip: each chip only processes its assigned expert
- In single-chip: one chip processes all experts

### Kernel Level

**NO CHANGES** - kernels are identical:
- Same memory access patterns
- Same computation logic
- Same results

## Result Equivalence

**The outputs ARE identical** (given same random seed):

```python
# Single-chip version
python moe_single_chip/main.py -p a2a3sim -d 0
# Output: [tensor with values X]

# Multi-chip version (2 chips)
python moe_multi_chip_experts/main.py -p a2a3sim -d 0-1
# Output: [tensor with values X]  <- SAME!
```

The distribution is **transparent** to the computation - we're just
executing the same work in parallel instead of sequentially.

## When to Use Which Version?

### Use `moe_single_chip` when:
- ✅ You only have 1 chip available
- ✅ You're developing/debugging kernels
- ✅ Model fits comfortably on single chip
- ✅ Simpler debugging (everything on one device)

### Use `moe_multi_chip_experts` when:
- ✅ You have multiple chips available
- ✅ You want faster execution (parallel compute)
- ✅ Model is too large for single chip
- ✅ You're scaling to more experts than fit on one chip

## Memory Layout

Per-chip tensors (same as single-chip):

```python
# Each chip has:
input:    [4, 64, 64]    # Input tokens
recv:     [4, 64, 64]    # Intermediate buffer
output:   [4, 64]        # Final output
```

The shape is identical - only the distribution changes.

## Performance Characteristics

### Single-Chip Version
- **Compute**: O(num_experts × num_tokens) sequential
- **Memory**: All expert data on one chip
- **Latency**: Sum of all expert compute times

### Multi-Chip Version
- **Compute**: O(num_tokens) parallel per chip
- **Memory**: Expert data distributed across chips
- **Latency**: Max of individual expert compute times

**Speedup**: Near-linear with chip count (ignoring communication overhead)

## Implementation Details

### No Kernel Changes
The kernels (`moe_demo_incore_*.cpp`) are **verbatim copies** from the single-chip version. This ensures:

1. **Correctness**: Same computation = same results
2. **Simplicity**: No need to rewrite kernel logic
3. **Maintainability**: Single source of truth for kernels

### Distribution via Orchestration
The multi-chip behavior comes from:
1. Python: Submit tasks to multiple chips (`worker=i`)
2. Orchestration: Each chip runs the same DAG
3. Kernel: Identical computation, different data subsets

### Key Insight
```
Single-chip: Chip 0 runs {Expert 0, Expert 1, Expert 2, Expert 3}
Multi-chip:  Chip 0 runs {Expert 0}, Chip 1 runs {Expert 1}, ...

Same total work, different distribution.
```

## Comparison with True Distributed MoE

This example keeps the computation **identical** for educational purposes.
Real distributed MoE systems would also optimize:

- **Communication**: Reduce all-to-all data movement
- **Load Balancing**: Dynamic token-to-expert assignment
- **Gradient Synchronization**: Distributed training considerations

Those optimizations are omitted here to maintain **result equivalence**
with the single-chip version.

## Next Steps

1. **Compare outputs**: Run both versions and verify results match
2. **Measure speedup**: Time both versions on your hardware
3. **Scale up**: Try 4, 8, or more chips
4. **Real distribution**: Implement data sharding across chips
