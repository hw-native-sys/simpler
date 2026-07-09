# Allgather — scene tests

Each rank contributes a slice; after the collective every rank holds the full concatenation of all ranks' inputs in rank order.

## Algorithm (3-Phase Mesh)

1. **Stage-in**: input → my scratch slot (HCCL window)
2. **Barrier**: mesh barrier (all-to-all notify/wait)
3. **Gather**: for r in 0..P-1: `TLOAD(rank r's scratch) → output[r * C]`

**Input**: Each rank owns `COUNT_PER_RANK=64` floats.
**Output**: All ranks receive `nranks * 64` floats — the concatenation of all inputs in rank order.

## Golden Check

`output[r*C + i] = r*100 + i`

Each rank's input: `[i + rank*100 for i in range(64)]`. After allgather every rank holds the concatenation in rank order.

## Test Classes

| Class | Ranks |
| ----- | ----- |
| `TestAllgatherP2` | 2 |
| `TestAllgatherP4` | 4 |

## Run

```bash
pytest tests/st/worker/collectives/allgather/ \
  --platform a2a3sim --device 0-3 -v
```

## File Structure

```text
allgather/
├── kernels/
│   ├── aiv/allgather_kernel.cpp
│   └── orchestration/allgather_orch.cpp
├── test_allgather.py
└── README.md
```
