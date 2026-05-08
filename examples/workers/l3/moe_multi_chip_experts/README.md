# `moe_multi_chip_experts/` — one expert per chip

Runs a small distributed Mixture-of-Experts pipeline across multiple chips.
Each rank owns one expert, exchanges token slices through HCCL window buffers,
applies a simple per-expert compute kernel, and gathers the processed expert
results back to the source ranks.

This example is intentionally tiny: `NUM_TOKENS = 10`, `HIDDEN_DIM = 16`, and
only the first `COUNT = 4` tokens are processed. The small shape makes the
data movement easy to inspect while still exercising cross-chip dispatch,
compute, and combine.

## What This Demonstrates

| Concept | Where it shows up |
| ------- | ----------------- |
| L3 multi-chip worker | `Worker(level=3, device_ids=[...])` in `main.py` |
| HCCL bootstrap buffers | `ChipBootstrapConfig` with `scratch1` and `scratch2` |
| Cross-rank dispatch | `kernels/aiv/moe_dispatch_alltoall.cpp` |
| Per-rank expert compute | `kernels/aiv/moe_simple_compute.cpp` |
| Cross-rank combine | `kernels/aiv/moe_combine_alltoall.cpp` |
| Device orchestration | `kernels/orchestration/moe_end2end_orch.cpp` |
| Pytest integration | `test_moe_multi_chip_experts.py` calls `main.run(...)` |

## Layout

```text
moe_multi_chip_experts/
  main.py                         # CLI demo and reusable run() entry
  test_moe_multi_chip_experts.py  # pytest wrapper, matching other L3 examples
  kernels/
    aiv/
      moe_dispatch_alltoall.cpp   # publish each rank's expert input
      moe_simple_compute.cpp      # add 1.0 to dispatched token slices
      moe_combine_alltoall.cpp    # gather processed expert outputs
    orchestration/
      moe_end2end_orch.cpp        # submit dispatch -> compute -> combine
  README.md
```

## Pipeline

For `N` chips, each chip owns one expert and starts with:

```text
send[expert_id][token][hidden]
recv[source_rank][token][hidden]
output[expert_id][token][hidden]
```

The orchestration submits three AIV kernels:

```text
┌──────────┐      ┌─────────┐      ┌─────────┐
│ Dispatch │ ───▶ │ Compute │ ───▶ │ Combine │
└──────────┘      └─────────┘      └─────────┘
```

1. Dispatch writes each rank's expert slice into the owner rank's `recv`.
2. Compute adds `1.0` to the first `COUNT` tokens in `recv`.
3. Combine copies each expert's processed slice into the source rank's
   `output[expert_id]` row.

`scratch1` is the HCCL window used by dispatch. `scratch2` is the HCCL window
used by combine. Compute only updates `recv`; it does not use either scratch
window.

The two communication phases use independent windows mainly because each
kernel places its barrier signal slots at the tail of its scratch buffer and
does not reset those slots before use. Dispatch leaves its signal slots
incremented after its cross-rank barrier. If combine reused the same window,
its `TWAIT` could observe the old dispatch signals and pass before combine has
staged its own data. A separate `scratch2` gives combine independent data
storage and independent signal slots.

## Data Pattern

Inputs are initialized with unique values:

```text
value = card_id * 1_000_000 + expert_id * 10_000 + token * 100 + dim
```

After compute, every checked output value should be the corresponding input
value plus `1.0`. `main.py` computes the golden reference in Python and checks
every `output[expert_id][token][hidden]` element for the processed token
range.

## Run

Hardware:

```bash
python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3 -d 0-1
```

Simulation:

```bash
python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3sim -d 0-1
```

The pytest wrapper follows the same style as the other L3 examples:

```bash
python -m pytest examples/workers/l3/moe_multi_chip_experts --platform a2a3 --device 0-1
```

For the CLI, device ids can be written as a range (`-d 0-1`) or a
comma-separated list (`-d 0,1`). For pytest, pass the same device spec to
`--device`. The examples use ranges because that matches the other L3 docs.

Expected successful output for the two-chip commands above includes:

```text
[End2End] End-to-end pipeline completed!
  Total: 256/256 correct
[End2End] All values correct! End-to-end pipeline works perfectly.
```

## Notes

- `test_moe_multi_chip_experts.py` is a thin pytest wrapper around
  `main.run(...)`.
- The pytest case runs on `a2a3` hardware and requires two available device
  ids.
- Each rank allocates independent `scratch1` and `scratch2` HCCL windows
  during worker bootstrap.
