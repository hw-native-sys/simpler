# Reduce-Scatter Distributed

This example demonstrates distributed reduce-scatter — each rank contributes a full input, and after the collective each rank holds the reduced sum of its assigned chunk across all ranks.

## Algorithm (4-Phase Mesh)

```text
Phase 1: stage-in      all P input chunks → scratch slots in HCCL window
Phase 2: barrier       mesh barrier (all-to-all notify/wait)
Phase 3: reduce        acc = my scratch[my_rank * C]; acc += peer.scratch[my_rank * C] for all peers
Phase 4: stage-out     acc → output
```

**Input**: Each rank owns `nranks * COUNT_PER_RANK` floats (P equal-sized chunks).
**Output**: Rank r receives `COUNT_PER_RANK` floats — the element-wise sum of chunk r from every rank.

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
reduce_scatter_distributed/
├── kernels/
│   ├── aiv/
│   │   └── reduce_scatter_kernel.cpp  # AIV kernel (stage-in, barrier, reduce, stage-out)
│   └── orchestration/
│       └── reduce_scatter_orch.cpp    # C++ orchestration shim
├── main.py                             # CLI entry point
├── test_reduce_scatter.py              # pytest test (sim + hardware)
└── __init__.py
```

## Golden Check

```text
output[j] = nranks * (my_rank * C + j) + 100 * nranks * (nranks - 1) / 2
```

Each rank's input is `[i + r*100 for i in range(nranks*C)]`; rank r verifies the reduced sum for chunk `my_rank`.
