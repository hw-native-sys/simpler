# Allreduce — scene tests

Five algorithm variants, each with its own `@scene_test(level=3)` class.

## Algorithm Variants

| Mode | Pattern | Remote Data/Rank | Barriers | Best For |
| ---- | ------- | ---------------- | -------- | -------- |
| `onephase` | Mesh direct: read full vector from all peers | O(P×N) | 1 mesh | Small P (2-4), simplicity |
| `twophase` | Mesh RS+AG: reduce-scatter then allgather | O(2N) | 2 mesh | Medium P, bandwidth |
| `ring` | Ring RS+AG: chunked reduce-scatter + allgather | O(2×(P-1)/P×N) | 2×(P-1) mesh (per round) | Large P (8+), bandwidth-optimal |
| `bidirectional_ring` | Two-ring bidirectional push RS+AG on disjoint data halves | O(2×(P-1)/P×N) | 2×(P-1)+1 mesh | Large P, double throughput |
| `ibing` | IBing interleaved RS+AG (Zong et al., ACM TACO 2025) | O((P+2)×N/P) | 2×(P-1)+1 mesh | P=2 only, shared-memory push-model |

### One-Phase Mesh (Direct)

Each rank reads the full vector from every peer and accumulates locally.

### Two-Phase Mesh (RS+AG)

Each rank only reduces its owned chunk (N/P elements), then gathers all reduced chunks.

### Ring (RS+AG)

Bandwidth-optimal for large P: (P-1) RS rounds + (P-1) AG rounds, each moving one chunk along a logical ring.

### Bidirectional Ring (Push RS+AG)

Two-ring push design: parallel push in both HCCS directions on disjoint data halves, doubling per-barrier throughput.

### IBing (Interleaved Bidirectional)

Paper-faithful interleaved RS+AG with AtomicAdd/AtomicNone phases. P=2 only — for P≥4 the AtomicNone forward phase overwrites peer chunks (shared-memory push-model race).

## Reduce Operations

Each mode's task args carry a `reduce_op` scalar (`CollectiveReduceOp` in
`simpler_setup/incore/collectives_reduce_op.hpp`): `0=Sum`, `1=Max`, `2=Min`,
`3=Prod`. `onephase`, `twophase`, and `ring` dispatch to the matching tile op
(`TADD`/`TMAX`/`TMIN`/`TMUL`) for all four. `bidirectional_ring` and `ibing`
implement only `Sum` — their orchestration rejects any other `reduce_op` via
`rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, ...)` before task submission (the
AIV kernel's `TPUT<AtomicAdd>` has no Max/Min variant).

## Golden Check

`output[i] = nranks*i + 100*nranks*(nranks-1)//2` for Sum; see
`allreduce_expected_output` in `_helpers.py` for the Max/Min/Prod goldens.

Each rank's input: `[i + rank*100 for i in range(256)]`.

## Test Classes

| Class | Ranks | Mode | Reduce Ops |
| ----- | ----- | ---- | ---------- |
| `TestAllreduceOnephaseP2` | 2 | onephase | Sum |
| `TestAllreduceTwophaseP2` | 2 | twophase | Sum |
| `TestAllreduceRingP2` | 2 | ring | Sum |
| `TestAllreduceBidirectionalRingP2` | 2 | bidirectional_ring | Sum |
| `TestAllreduceIbingP2` | 2 | ibing | Sum |
| `TestAllreduceOnephaseP4` | 4 | onephase | Sum |
| `TestAllreduceTwophaseP4` | 4 | twophase | Sum |
| `TestAllreduceRingP4` | 4 | ring | Sum |
| `TestAllreduceBidirectionalRingP4` | 4 | bidirectional_ring | Sum |
| `TestAllreduceIbingNranksError` | 4 | ibing (negative: expects `ValueError`) | Sum |
| `TestAllreduceOnephaseP2MaxMinProd` | 2 | onephase | Max, Min, Prod |
| `TestAllreduceTwophaseP2MaxMinProd` | 2 | twophase | Max, Min, Prod |
| `TestAllreduceRingP2MaxMinProd` | 2 | ring | Max, Min, Prod |
| `TestAllreduceBidirectionalRingRejectNonSum` | 2 | bidirectional_ring (negative: expects `RuntimeError`) | Max |
| `TestAllreduceIbingRejectNonSum` | 2 | ibing (negative: expects `RuntimeError`) | Max |

## Run

```bash
# P2 tests (2 devices)
pytest tests/st/worker/collectives/allreduce/ \
  --platform a2a3sim --device 0-1 -k "P2"

# P4 tests (4 devices)
pytest tests/st/worker/collectives/allreduce/ \
  --platform a2a3sim --device 0-3 -k "P4 or Ibing"
```

## File Structure

```text
allreduce/
├── kernels/
│   ├── aiv/
│   │   ├── allreduce_onephase_kernel.cpp
│   │   ├── allreduce_twophase_kernel.cpp
│   │   ├── allreduce_ring_kernel.cpp
│   │   ├── allreduce_bidirectional_ring_kernel.cpp
│   │   └── allreduce_ibing_kernel.cpp
│   └── orchestration/
│       ├── allreduce_onephase_orch.cpp
│       ├── allreduce_twophase_orch.cpp
│       ├── allreduce_ring_orch.cpp
│       ├── allreduce_bidirectional_ring_orch.cpp
│       └── allreduce_ibing_orch.cpp
├── test_allreduce.py
└── README.md
```

See the [minimal allreduce example](../../../../../../examples/workers/l3/allreduce/) for a simpler "how to do a collective" demo.
