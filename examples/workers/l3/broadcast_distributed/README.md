# Broadcast Distributed

This example demonstrates distributed broadcast — the root rank sends its input to all other ranks.

## Algorithm (3-Phase Mesh)

```text
Phase 1: stage-in      root: input → scratch (HCCL window)
Phase 2: barrier       mesh barrier (all-to-all notify/wait)
Phase 3: broadcast     TLOAD(root's scratch) → output
```

**Input**: Root rank owns `COUNT_PER_RANK` floats.
**Output**: All ranks receive the same `COUNT_PER_RANK` floats — the root's input.

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

## File Structure

```text
broadcast_distributed/
├── kernels/
│   ├── aiv/
│   │   └── broadcast_kernel.cpp      # AIV kernel (stage-in, barrier, broadcast)
│   └── orchestration/
│       └── broadcast_orch.cpp        # C++ orchestration shim
├── main.py                            # CLI entry point
├── test_broadcast.py                  # pytest test (sim + hardware)
└── __init__.py
```

## Golden Check

```text
output[i] = ROOT_RANK * 100 + i   # ROOT_RANK = 0
```

Root's input is `[i for i in range(C)]`; every rank verifies it received this exact data.
