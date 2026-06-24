# All-to-All Distributed

This example demonstrates distributed all-to-all (personalized exchange) — each rank sends a different slice to each peer.

## Algorithm (3-Phase Mesh)

```text
Phase 1: stage-in      for d in 0..P-1: input[d * C] → scratch[d * C] (chunk d is for rank d)
Phase 2: barrier       mesh barrier (all-to-all notify/wait)
Phase 3: exchange      for s in 0..P-1: TLOAD(peer s scratch[my_rank * C]) → output[s * C]
```

**Input**: Each rank owns `nranks * COUNT_PER_RANK` floats (chunk d is payload for rank d).
**Output**: Each rank receives `nranks * COUNT_PER_RANK` floats (chunk s is what rank s sent to me).

## Usage

```bash
# 2 devices (sim or hardware)
python main.py -p a2a3sim -d 0-1

# 4 devices on hardware
python main.py -p a2a3 -d 0-3
```

### CLI Arguments

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `-p`, `--platform` | `a2a3` | Platform backend (e.g., `a2a3`, `a2a3sim`, `a5`, `a5sim`) |
| `-d`, `--device` | `0-1` | Device range (e.g., `0-1`, `0-3`). 2-16 chips supported. |
| `--pto-isa-commit` | `None` | Optional PTO ISA commit/tag |

## File Structure

```text
all_to_all_distributed/
├── kernels/
│   ├── aiv/
│   │   └── all_to_all_kernel.cpp     # AIV kernel (stage-in, barrier, exchange)
│   └── orchestration/
│       └── all_to_all_orch.cpp       # C++ orchestration shim
├── main.py                            # CLI entry point
├── test_all_to_all.py                 # pytest test (sim + hardware)
└── __init__.py
```

## Golden Check

```text
output[src * C + j] = src * 1000 + my_rank * 100 + j
```

Rank r receives in slot s the data that rank s originally prepared for rank r.
