# Allreduce Distributed — Multi-Mode Collective

This example demonstrates distributed allreduce with three selectable algorithm variants via the `--mode` parameter.

## Algorithm Variants

| Mode | Pattern | Remote Data/Rank | Barriers | Best For |
| ---- | ------- | ---------------- | -------- | -------- |
| `onephase` | Mesh direct: read full vector from all peers | O(P×N) | 1 mesh | Small P (2-4), simplicity |
| `twophase` | Mesh RS+AG: reduce-scatter then allgather | O(2N) | 2 mesh | Medium P, bandwidth |
| `ring` | Ring RS+AG: chunked reduce-scatter + allgather | O(2×(P-1)/P×N) | 2×(P-1) mesh (per round) | Large P (8+), bandwidth-optimal |

### One-Phase Mesh (Direct)

```text
Phase 1: stage-in     input → my scratch slot (HCCL window)
Phase 2: barrier      mesh barrier (all-to-all notify/wait)
Phase 3: compute      for peer in P: acc += TLOAD(peer.scratch)
Phase 4: stage-out    acc → output
```

Each rank reads the **full vector** from every peer and accumulates locally. Simple but O(P×N) remote data per rank.

### Two-Phase Mesh (RS+AG)

```text
Phase 1: stage-in     partition input → P chunk slots
Phase 2: RS barrier   mesh barrier
Phase 3: reduce       acc[my_rank] = sum over peers of peer.scratch[my_rank]
Phase 4: AG barrier   mesh barrier
Phase 5: gather       for r in P: output[r*C] = peer[r].scratch[r*C]
Phase 6: stage-out    (implicit in phase 5)
```

Each rank only reduces its **owned chunk** (N/P elements), then gathers all reduced chunks. Total remote data: 2×(P-1) chunks vs one-phase's (P-1) full vectors.

### Ring (RS+AG)

```text
Phase 1: stage-in     partition input → P chunk slots
Phase 2: RS rounds    (P-1) steps: publish → barrier → read left → accumulate
Phase 3: AG rounds    (P-1) steps: publish → barrier → read left → store
Phase 4: stage-out    chunks → output
```

Bandwidth-optimal for large P: each round moves one chunk along a logical ring (read from left neighbor). Total remote data: 2×(P-1)/P × N. Barriers are mesh-style notify/wait all peers each round (same as pre-consolidation ring example).

## Usage

```bash
# One-phase mesh (default) — 2 devices
python main.py -p a2a3sim -d 0-1 --mode onephase

# Two-phase mesh RS+AG — 4 devices
python main.py -p a2a3sim -d 0-3 --mode twophase

# Ring RS+AG — 4 devices (bandwidth-optimal)
python main.py -p a2a3 -d 0-3 --mode ring
```

### CLI Arguments

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `-p`, `--platform` | `a2a3` | Platform backend (e.g., `a2a3`, `a2a3sim`, `a5`, `a5sim`) |
| `-d`, `--device` | `0-1` | Device range (e.g., `0-1`, `0-3`). 2-16 chips supported. |
| `-m`, `--mode` | `onephase` | Algorithm variant: `onephase`, `twophase`, `ring` |

## File Structure

```text
allreduce_distributed/
├── kernels/
│   ├── aiv/
│   │   ├── allreduce_onephase_kernel.cpp   # Mesh direct
│   │   ├── allreduce_twophase_kernel.cpp   # Mesh RS+AG
│   │   └── allreduce_ring_kernel.cpp       # Ring RS+AG
│   └── orchestration/
│       ├── allreduce_onephase_orch.cpp
│       ├── allreduce_twophase_orch.cpp
│       └── allreduce_ring_orch.cpp
├── main.py                                  # CLI entry point with --mode
├── test_allreduce.py                        # Parametrized pytest over all modes
├── README.md                                # This file
└── __init__.py
```

## How It Works

1. **Host tensors**: Each rank owns a private input/output tensor via `torch.share_memory_()`.
2. **Communication domain**: `orch.allocate_domain()` creates an HCCL window for cross-rank communication.
3. **Kernel dispatch**: `main.py` compiles the selected kernel based on `--mode` and submits it to each chip.
4. **Barrier + remote read**: Kernels use PTO-ISA `TNOTIFY`/`TWAIT` for synchronization and `TLOAD` via `CommRemotePtr` for cross-rank reads.
5. **Golden check**: Output verified against `sum_r(i + r*100)` for all ranks.

## When to Use Which Mode

| Scenario | Recommended Mode |
| -------- | ---------------- |
| 2-4 ranks, small tensors | `onephase` (simplest) |
| 4-8 ranks, bandwidth-bound | `twophase` (mesh RS+AG) |
| 8+ ranks, large tensors | `ring` (bandwidth-optimal) |
| Latency-critical, small P | `onephase` (fewest barriers) |

## Testing

```bash
# Run all modes with 2 devices (sim — same as CI st-sim-a2a3)
pytest test_allreduce.py -v -k "test_allreduce_distributed" --platform a2a3sim -d 0-1

# Hardware
pytest test_allreduce.py -v -k "test_allreduce_distributed" --platform a2a3 -d 0-1

# Run specific mode
pytest test_allreduce.py -v -k "test_allreduce_distributed[onephase]" --platform a2a3sim -d 0-1
```
