# ep_dispatch_combine — MoE expert-parallel dispatch and combine

The largest L3 example: a two-card run of the real DeepSeek-V4 FLASH MoE
shapes, with three AIV kernels chained behind a single orchestration over a
shared communication window.

```text
dispatch.cpp      route tokens to the owning rank's experts
local_expert.cpp  recv_y[e, s, :] = recv_x[e, s, :] * recv_w[e, s]
combine.cpp       scatter recv_y back by route id, reduce over TOPK
```

`local_expert` stands in for the production expert MLP — the point of the
example is the **communication protocol around it**, not the arithmetic.

## Shapes

Production values, except for the rank count:

| Constant | Value | Meaning |
| -------- | ----- | ------- |
| `T` | 128 | tokens (decode batch × seq) |
| `TOPK` | 6 | experts per token |
| `D` | 4096 | hidden size |
| `L` | 16 | local experts per rank |
| `R` | 192 | per-expert receive cap |
| `N_RANKS` | 2 | **EP=2 here vs EP=16 in production** |

Each rank keeps the same 16-expert load, so only the global expert count
differs (32 vs 256). These must mirror the constants at the top of the
kernels.

## The dispatch protocol

One kernel walks five phases, and this is the part worth reading:

| Phase | What happens |
| ----- | ------------ |
| `histogram` | Scalar histogram of the routing indices, plus a `(dst, loc_e)`-sorted route table. |
| `publish` | Push the full `send_counts` table to every peer with `TNOTIFY(AtomicAdd)`, then a `count_done` barrier. |
| `prefix_sum` | Local prefix sums over the gathered `pub_counts` — no communication. This is what fixes each route's destination slot. |
| `payload_push` | Per route, `TPUT` three independent tiles (`x` BF16, `weight` FP32, `idx` INT32) into the peer's receive windows, then a `data_done` barrier. |
| `stage_out` | Copy the receive windows out to host-backed tensors. |

**Why the weight and index tiles are padded to 8 elements:** 8 FP32 = 32 B is
the minimum vector tile in the PTO ISA (one MTE burst). The host pre-packs each
row as `[value, 0, …, 0]`; the kernel `TROWSUM`-compacts the receive side down
to a dense `[L, R]` output. It wastes bandwidth and is still the cheapest legal
transfer.

`combine` relies on the communication window being **zero-initialised** — it
scatters into `routed_y_buf` without clearing it first.

## What this exercises

| Concept | How |
| ------- | --- |
| **Three children under one `ChipCallable`** | `children=[(0, dispatch), (1, local_expert), (2, combine)]` — the integers are `func_id`s, matching the `rt_submit_aiv_task(0/1/2, …)` calls in the orchestration. Each child declares only the args it consumes; the orchestration signature is the union. |
| **Collective group dispatch** | The complete per-rank callable is submitted as one NEXT_LEVEL group because its dispatch and combine phases wait on peer ranks. |
| **Ordering without dependencies** | The three tasks run back-to-back because `rt_submit_aiv_task` dispatches in submission order, not because any tensor edge forces it. |
| **Chaining through host-backed tensors** | `recv_x_out` / `recv_w_out` / `recv_count_out` are `OUTPUT_EXISTING` for dispatch and inputs to `local_expert`; `recv_y` likewise feeds `combine`. |
| **A hand-laid-out window** | `SCRATCH_NBYTES` sums every region — counts table, two signal areas, three receive windows, the combine push destination, a third signal — and must match the `kOff*` offsets in the kernels. |
| **Mixed dtypes across one transfer** | BF16 payload, FP32 weights, INT32 indices, in three separate tiles per route. |

## Run

```bash
# Simulation (2 ranks)
python examples/workers/l3/ep_dispatch_combine/main.py -p a2a3sim -d 0-1

# Hardware (2 ranks)
python examples/workers/l3/ep_dispatch_combine/main.py -p a2a3 -d 0-1
```

Or as a scene test — `a2a3sim`, `a2a3`, and `a5sim` are marked:

```bash
pytest examples/workers/l3/ep_dispatch_combine --platform a2a3sim --manual include
```

Exactly two devices are required.

## Verification

The host replays the whole dispatch protocol to build the golden, then:

- **`recv_x` is compared bit-exact.** The x channel is a pure copy, so BF16
  meets BF16 with no cast in between — no tolerance is needed, and none is
  granted, at any magnitude.
- **`recv_y` and `routed_y` allow 2 BF16 ULPs** (`rtol = atol = 2**-6`). A
  single BF16 cast can differ from torch's round-to-nearest-even by one ULP on
  an exact tie; the FP32 accumulation over `TOPK` terms is itself exact.
- **Failures print structure, not just a max-diff** — how many rows are bad,
  the span of `d` they cover, and the got/expected ratio. A dropped `TOPK` term
  shows up as a ratio of `(TOPK-1)/TOPK`; a lost tail shows as a bad-`d` span
  at high `d`.

**On the simulator, only `routed_y` is checked.** Sim keeps intermediate child
outputs device-local when they feed a later child task, so `recv_y` is not
host-visible there. Hardware runs both the dispatch-output and the reduce
checks.

## File structure

```text
ep_dispatch_combine/
├── kernels/
│   ├── aiv/
│   │   ├── combine.cpp
│   │   ├── dispatch.cpp
│   │   └── local_expert.cpp
│   └── orchestration/
│       └── ep_dispatch_combine_orch.cpp
├── main.py
├── test_ep_dispatch_combine.py
└── README.md
```
