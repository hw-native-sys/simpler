# Broadcast — scene tests

Root rank (0) sends its input to all other ranks via the HCCL window.

## Algorithm (3-Phase Mesh)

1. **Stage-in**: root → input → scratch (HCCL window)
2. **Barrier**: mesh barrier (all-to-all notify/wait)
3. **Broadcast**: `TLOAD(root's scratch) → output`

**Input**: Root rank (0) owns `COUNT_PER_RANK=64` floats. Non-root ranks pass a dummy input.
**Output**: All ranks receive the same 64 floats — the root's original data.

## Golden Check

`output[i] = ROOT_RANK*100 + i`  (ROOT_RANK=0)

Root's input: `[i for i in range(64)]`. Every rank verifies it received this exact data.

## Test Classes

| Class | Ranks |
| ----- | ----- |
| `TestBroadcastP2` | 2 |
| `TestBroadcastP4` | 4 |

## Run

```bash
pytest tests/st/worker/collectives/broadcast/ \
  --platform a2a3sim --device 0-3 -v
```

## File Structure

```text
broadcast/
├── kernels/
│   ├── aiv/broadcast_kernel.cpp
│   └── orchestration/broadcast_orch.cpp
├── test_broadcast.py
└── README.md
```
