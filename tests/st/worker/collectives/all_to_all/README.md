# All-to-All — scene tests

Personalized exchange: each rank sends a different slice to each peer through the HCCL window.

## Algorithm (push → barrier → local copy-out)

```text
push        for d in 0..P-1: TPUT input[d * C] → peer d's scratch[my_rank * C]
            Self-rank: CommRemotePtr to self returns the local pointer
barrier     mesh notify/wait (all-to-all)
copy-out    for s in 0..P-1: TLOAD(local scratch[s * C]) → output[s * C]
```

**Input**: Each rank owns `nranks * COUNT_PER_RANK=64` floats. Chunk d is payload for rank d.
**Output**: Each rank receives `nranks * 64` floats — the data every peer sent to it.

Scratch at offset `src * C` holds the chunk rank `src` pushed here; after the barrier, copy-out is local.

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
