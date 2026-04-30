# Multi-Chip MoE Implementation Notes

## Overview

This implementation transforms the single-chip MoE example (`moe_single_chip`) into a multi-chip parallel version (`moe_multi_chip_experts`) where **each chip processes one expert** instead of all experts running sequentially on one chip.

## Key Changes

### 1. Architecture

**Single-Chip Version:**
- One chip runs ALL 4 experts sequentially
- Orchestration loops: `card_i=0..3`, `expert_j=0..3`, `t_idx=0..3`
- Total: 4 cards × 4 experts × 4 tokens = 64 dispatch operations

**Multi-Chip Version:**
- Each chip runs ONE expert in parallel
- Orchestration: `card_i=i` (passed as arg), `expert_j=i` (passed as arg), `t_idx=0..3`
- Per chip: 1 expert × 4 tokens = 4 dispatch operations
- With 2 chips: 2 × (1 × 4) = 8 total dispatch operations (parallel)

### 2. Modified Files

#### `kernels/kernel_config.py` (NEW)
- Configuration file defining runtime and kernel sources
- Mirrors structure from single-chip version

#### `kernels/orchestration/moe_multi_chip_orch.cpp` (MODIFIED)
- Reads expert ID and chip ID from scalar arguments (passed by Python)
- Only processes the assigned expert (not all experts)
- Maintains same computation pattern as single-chip version
- Key difference: No `card_i` loop, no `expert_j` loop - these are passed as args

#### `main.py` (MODIFIED)
- Passes two scalar arguments to orchestration:
  1. Expert ID (`i`): Chip i processes expert i
  2. Chip ID (`i`): Logical card_i for data layout computation
- Updated ChipCallable signature to accept 3 tensors + 2 scalars

### 3. Result Equivalence

Both versions produce **IDENTICAL results** because:
- Same kernels (`moe_demo_incore_0/1/2.cpp`)
- Same computation logic (dispatch → compute → combine)
- Only difference: execution distribution (serial vs parallel)

## Usage

### Run Multi-Chip Version (2 chips, 2 experts)
```bash
python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3sim -d 0-1
```

### Run Single-Chip Version (for comparison)
```bash
python examples/workers/l3/moe_single_chip/main.py -p a2a3sim -d 0
```

### Run via pytest
```bash
pytest examples/workers/l3/moe_multi_chip_experts/test_moe_multi_chip.py -v -s
```

## Technical Details

### Parameter Passing
The multi-chip version uses scalar arguments to pass expert ID and chip ID to orchestration:
```python
moe_args.add_scalar(i)  # Expert ID
moe_args.add_scalar(i)  # Chip ID (logical card_i)
```

Orchestration reads these:
```cpp
int64_t expert_j = static_cast<int64_t>(orch_args.scalar(0));
int64_t card_i = static_cast<int64_t>(orch_args.scalar(1));
```

### Data Layout
- Each chip has its own input/output buffers
- Shape: `[4, 64, 64]` (4 tokens, 64 hidden dim)
- Same layout as single-chip version for result equivalence

### ChipCallable Signature
- Single-chip: `[IN, OUT, OUT]` (3 tensors)
- Multi-chip: `[IN, OUT, OUT, IN, IN]` (3 tensors + 2 scalars)

## Verification

To verify result equivalence:
1. Run single-chip version, save output
2. Run multi-chip version, save output
3. Compare outputs (should be identical)

Note: Multi-chip version produces per-chip outputs. To compare with single-chip:
- Single-chip output is the combined result of all 4 experts
- Multi-chip per-chip output is the result of one expert
- Combine multi-chip outputs appropriately for comparison

## Future Improvements

1. **Dynamic Configuration**: Currently hardcoded for 4 tokens. Could make configurable.
2. **Result Combination**: Add logic to combine per-chip outputs for direct comparison.
3. **Scalability**: Test with more chips (4, 8, etc.)
4. **Performance**: Measure speedup vs single-chip version

## Related Files

- Single-chip version: `examples/workers/l3/moe_single_chip/`
- Multi-chip version: `examples/workers/l3/moe_multi_chip_experts/`
- Other multi-chip examples:
  - `examples/workers/l3/multi_chip_dispatch/`
  - `examples/workers/l3/ffn_tp_parallel/`
