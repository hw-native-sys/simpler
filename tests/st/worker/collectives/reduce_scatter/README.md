# Reduce-Scatter — scene tests

Each rank contributes a full input; after the collective each rank holds the reduced sum of its assigned chunk across all ranks.

## Algorithm (4-Phase Mesh)

1. **Stage-in**: all P input chunks → scratch slots in HCCL window
2. **Barrier**: mesh barrier (all-to-all notify/wait)
3. **Reduce**: `acc = my scratch[my_rank*C]; acc += peer.scratch[my_rank*C]` for all peers
4. **Stage-out**: `acc → output`

**Input**: Each rank owns `nranks * COUNT_PER_RANK` floats — P equal-sized chunks of `COUNT_PER_RANK=64`.
**Output**: Rank r receives `COUNT_PER_RANK=64` floats — the element-wise sum of chunk r from every rank.

## Golden Check

`output[j] = nranks*(my_rank*C + j) + 100*nranks*(nranks-1)//2` (integer arithmetic, `C = COUNT_PER_RANK`)

Each rank's input: `[i + rank*100 for i in range(nranks*64)]`. Rank r verifies the reduced sum for chunk `my_rank`.

## Test Classes

| Class | Ranks |
| ----- | ----- |
| `TestReduceScatterP2` | 2 |
| `TestReduceScatterP4` | 4 |

## Run

```bash
pytest tests/st/worker/collectives/reduce_scatter/ \
  --platform a2a3sim --device 0-3 -v
```

## File Structure

```text
reduce_scatter/
├── kernels/
│   ├── aiv/reduce_scatter_kernel.cpp
│   └── orchestration/reduce_scatter_orch.cpp
├── test_reduce_scatter.py
└── README.md
```
