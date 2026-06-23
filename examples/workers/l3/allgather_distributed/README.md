# Allgather Distributed

This example demonstrates distributed allgather — each rank contributes a slice, and after the collective every rank holds the full concatenation of all ranks' inputs.

## Algorithm (3-Phase Mesh)

```text
Phase 1: stage-in      input → my scratch slot (HCCL window)
Phase 2: barrier       mesh barrier (all-to-all notify/wait)
Phase 3: gather        for r in 0..P-1: TLOAD(rank r's scratch) → output[r * C]
```

**Input**: Each rank owns `COUNT_PER_RANK` floats.
**Output**: All ranks receive `nranks * COUNT_PER_RANK` floats — the concatenation of all inputs in rank order.

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
allgather_distributed/
├── kernels/
│   ├── aiv/
│   │   └── allgather_kernel.cpp      # AIV kernel (stage-in, barrier, gather)
│   └── orchestration/
│       └── allgather_orch.cpp        # C++ orchestration shim
├── main.py                            # CLI entry point
├── test_allgather.py                  # pytest test (sim + hardware)
└── __init__.py
```

## Golden Check

```text
output[r * C + i] = r * 100 + i
```

Each rank's input is `[i + rank * 100 for i in range(C)]`; after allgather every rank holds the concatenation in rank order.
