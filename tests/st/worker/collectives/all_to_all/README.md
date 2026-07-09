# All-to-All — scene tests

Personalized exchange: each rank sends a different slice to each peer through the HCCL window.

## Algorithm (3-Phase Mesh)

1. **Stage-in**: for d in 0..P-1: `input[d*C] → scratch[d*C]` (chunk d is for rank d)
2. **Barrier**: mesh barrier (all-to-all notify/wait)
3. **Exchange**: for s in 0..P-1: `TLOAD(peer s scratch[my_rank*C]) → output[s*C]`

**Input**: Each rank owns `nranks * COUNT_PER_RANK=64` floats. Chunk d is payload for rank d.
**Output**: Each rank receives `nranks * 64` floats — the data every peer sent to it.

## Golden Check

`output[src*C + j] = src*1000 + my_rank*100 + j`

Rank r receives in slot s the data that rank s originally prepared for rank r.

## Test Classes

| Class | Ranks |
| ----- | ----- |
| `TestAllToAllP2` | 2 |
| `TestAllToAllP4` | 4 |

## Run

```bash
pytest tests/st/worker/collectives/all_to_all/ \
  --platform a2a3sim --device 0-3 -v
```

## File Structure

```text
all_to_all/
├── kernels/
│   ├── aiv/all_to_all_kernel.cpp
│   └── orchestration/all_to_all_orch.cpp
├── test_all_to_all.py
└── README.md
```
