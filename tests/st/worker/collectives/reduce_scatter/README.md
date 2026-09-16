# Reduce-Scatter — scene tests

Each rank contributes a full input; after the collective each rank holds the reduced chunk assigned to it across all ranks.

## Algorithm (4-Phase Mesh)

1. **Stage-in**: all P input chunks → scratch slots in HCCL window
2. **Barrier**: mesh barrier (all-to-all notify/wait)
3. **Reduce**: `acc = my scratch[my_rank*C]; acc = reduce_op(acc, peer.scratch[my_rank*C])` for all peers
4. **Stage-out**: `acc → output`

**Input**: Each rank owns `nranks * COUNT_PER_RANK` floats — P equal-sized chunks of `COUNT_PER_RANK=64`.
**Output**: Rank r receives `COUNT_PER_RANK=64` floats — the element-wise reduction of chunk r from every rank.

## Reduce Operations

The task args carry a `reduce_op` scalar (`CollectiveReduceOp` in
`simpler_setup/incore/collectives_reduce_op.hpp`): `0=Sum`, `1=Max`, `2=Min`,
`3=Prod`. The kernel dispatches to the matching tile op
(`TADD`/`TMAX`/`TMIN`/`TMUL`) for all four; `reduce_scatter_orch_fn` rejects an
unrecognized `reduce_op` before submission.

## Golden Check

`output[j] = nranks*(my_rank*C + j) + 100*nranks*(nranks-1)//2` for Sum
(integer arithmetic, `C = COUNT_PER_RANK`); see `reduce_scatter_expected_output`
in `_helpers.py` for the Max/Min/Prod goldens.

Each rank's input: `[i + rank*100 for i in range(nranks*64)]`. Rank r verifies the reduced chunk for `my_rank`.

## Test Classes

| Class | Ranks | Reduce Ops |
| ----- | ----- | ---------- |
| `TestReduceScatterP2` | 2 | Sum |
| `TestReduceScatterP4` | 4 | Sum |
| `TestReduceScatterP2MaxMinProd` | 2 | Max, Min, Prod |

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
