# hbg: decomposing a whole-layer Graph body into per-block Definitions, and reusing them across layers

**Date**: 2026-08-19, revised 2026-08-20
**Verdict**: adopted at seven Definitions, once #1929 removed the single recording slot the first measurement failed against — the win is smaller than the orchestration window alone suggests, because seven Definitions are seven images to upload

> **Names in this entry are the ones in use at the time.** A task inside a Graph
> body was a "node" and the record phase was `record_node`; they are now a
> sub-task and `record_sub_task`. `strace_timing.py` still accepts the
> old phase name, so logs from this investigation remain readable.

## Question

`examples/a2a3/host_build_graph/deepseek_v4_flash_decode` casts its 43-layer
decode as **one** 743-node Graph Definition — a CSA+HCA layer pair — replayed 20
times, covering 40 of the 43 layers. Layers 0, 1 and 42 plus the head stay on
the ordinary submit path, so the host still submits 1131 tasks for a
15971-task network.

The intuition, and it is a natural one: the Definition is too coarse. A layer
*pair* occurs 20 times, but a **block** — one attention or one MoE FFN — occurs
far more often (the MoE block appears 43 times, once per layer, and is ~90% of
the network's tasks). Cut the body at block granularity and the same
Definitions should cover every layer, collapsing the 1111 remaining
submissions and shrinking host orchestration time.

## What was tried

Three forms of the same program, measured back to back on one machine
(`SIMPLER_SKIP_DEVICE_RUN=1`, so the host path is measured without device
execution behind it; `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1` for the per-kind
`host-orch phase=` breakdown; 3 rounds × 2 ranks per form):

- **pair** — as merged: one 743-node Definition, 20 replays.
- **split/l42** — the body's four top-level scopes become four Definitions
  (`csa_attn` / `csa_moe` / `hca_attn` / `hca_moe`), each carrying only the
  boundary args its own scope reads; the three tensors that crossed scope
  boundaries move to the caller and travel as INOUT so the ordinary tensormap
  path chains the four outer GRAPH tasks; and layer 42 replays the loop's
  `csa_attn` + `hca_moe` at layer index 42 instead of being submitted node by
  node.
- **full** — additionally covers layers 0 and 1 with a Definition per peeled
  scope (8 Definitions in total).

Before measuring, each reuse candidate was checked structurally, because a
kernel-name match is not evidence that one site's body can stand in for
another's:

| check | how |
| ----- | --- |
| same operation sequence | split each orchestration into blocks at every `hc_pre_rms` and compare kernel-name sequences |
| interchangeable kernels | normalize every kernel source by renaming *generated* identifiers (`_inlineNNN`, `__ssa_vN`, `vNN`/`iNN`, `_lN`, the kernel's own name) by first-occurrence order; equal token streams prove alpha-equivalence, and every other token — including every numeric literal — must match verbatim |
| same arguments | resolve every tensor argument back through `reshape`/`view` to the entry-level tensor it derives from, and diff the two blocks position by position on (kernel, [(tag, root)], scalar count) |

## Result

### The structural map (the reusable part of this investigation)

367 kernels fall into **169** classes keyed on (core type, code, declared
signature) and **132** keyed on code alone: 37 classes are the same code
declared OUT at the peeled sites and INOUT at the loop's, because the loop
pre-allocates its intermediates and the peeled layers let the runtime allocate.
By block:

| block | occurrences in the network | static sites | can share one Definition? |
| ----- | -------------------------- | ------------ | ------------------------- |
| SWA attn | 2 (layers 0, 1) | 2 | yes — 26/26 positions are the same kernel |
| CSA attn | 21 (loop ×20, layer 42) | 2 | yes — 47/47 same sequence, 24 differ only in declared direction, **0 in code** |
| HCA attn | 20 (loop) | 1 | n/a |
| MoE, hash-routed | 2 (layers 0, 1) | 2 | **no** — 3 kernels differ in code |
| MoE, hash+sort | 20 (loop CSA) | 1 | n/a |
| MoE, sort-routed | 21 (loop HCA, layer 42) | 2 | yes — 29/29 same sequence, 14 direction-only, 0 in code |

Three things a name comparison cannot see, all of which the checks above caught:

- **Two counters index the weights.** Most views take the absolute layer number,
  but the per-attention-type tables (CSA/HCA compress weights, index caches)
  take the *loop iteration*: `csa_cmp_wkv` is offset by `loop_i * 1024`, and the
  peeled layer 42 uses the literal `20480` = 20 × 1024. An emitter reused at
  layer 42 needs both indices (42 and 20).
- **The last layer's MoE writes somewhere else.** The loop's final `hc_post`
  writes the internal running hidden state; layer 42's writes
  `ext_pre_hc_hidden_out`, the tensor the head reads. Same kernel, different
  destination — and the *only* argument-level difference across the block's 29
  aligned nodes.
- **The hash-routed MoE bakes its epoch.** `dispatch_wait` holds `32` where
  `dispatch_wait_0` holds `64`; the loop's variant takes the value as a scalar
  parameter instead. So layers 0 and 1 cannot share one Definition even though
  their operation sequences are identical.

### Why it regressed between #1897 and #1929

Over that range `graph_begin` held **one recording slot for the whole
orchestration**. A Graph whose `full_key` differed from the in-flight recording's
got the default `GraphScopeResult` back — `execute_block=true, recording=false` —
and `rt_submit_graph_impl` ran its body on the ordinary path, with no warning and
no retry, even though the `rt_graph_commit()` on that same branch then waited for
the recorder to go idle. So k distinct Definitions submitted back to back got
roughly every other one recorded.

That is exactly what block decomposition produces: where the pair form submits
one key per iteration and never contends, the four-block form submits four
distinct keys back to back. Measured on `main` at 6f56ce64 + this PR, median of
12 samples per form (2 ranks × 3 rounds × 2 alternating passes):

| form | Definitions | graph submissions | host-submitted tasks | orch window |
| ---- | ----------- | ----------------- | -------------------- | ----------- |
| pair | 1 | 20 (all hit) | 1131 | 3.96 ms |
| four blocks + layer 42 | 4 | **79** of 82 intended | **1486** | 3.78 ms |
| + layers 0/1 covered | **6** of 8 intended | **81** of 86 intended | **1360** | 4.82 ms |

Every demoted submission pays its block's nodes at full ordinary cost, which is
why the task count goes *up*: the three lost submissions are two ~336-node MoE
blocks and one attention block, ~650 tasks. Machine load moved the orchestration
window by 35% between two runs of the identical configuration (3.57 ms and
4.82 ms for `pair`), so the window column resolves nothing on its own — the task
and submission counts are the reliable signal and they all move the wrong way.

### What the decomposition did buy, before #1897

Measured on `f74ad5e1`, where recording was synchronous and every submission
hit its Definition, the same four-block + layer-42 form gave host-submitted
tasks 1131 → **835** and the orchestration window 4.31 → 3.15 ms, with the
device-side task count invariant at **15971** in all three forms (the 378 nodes
the two layer-42 replays contribute are exactly the 378 tasks the deleted peeled
region used to submit). Two per-operation effects drove it, both from the
narrower boundary — `graph_tensor_from_boundary` makes two full linear passes
over the boundary per classified tensor, and `graph_boundary_matches` is
quadratic in boundary width — so cutting 118 boundary tensors to 27–47 made
recording ~32% cheaper per node (1.58 → 1.07 µs) and a replay ~69% cheaper
(7.5 → 2.3 µs).

Even there the ceiling was low, for a reason worth keeping: **a recorded node
costs about what a submitted task costs** (1.07 µs against 0.75 µs), so coverage
moves per-node host work from `submit_task` to `record_node` rather than
deleting it. The saving comes only from *replaying* a Definition another
occurrence already recorded — which is why layer 42 paid for itself (2 replays
of Definitions the loop had already recorded) and why a Definition with one or
two occurrences did not: a ~336-node block costs ~0.30 ms to record plus
~0.34 ms to build against ~0.50 ms to submit twice, so break-even sits near
three occurrences.

## Why not (now)

Nothing about the decomposition itself was wrong even when it measured as a
regression — the structural map above holds, the device task count is preserved
exactly, and every submission that *did* record hit its Definition with no
boundary mismatch in any run. What was wrong was the single recording slot, which
was replaced in #1929 by a keyed in-flight map plus a prewarmed recorder pool, so
distinct keys submitted back to back all record.

## What the decomposition is worth on top of #1929

Seven Definitions cover all 43 layers — the four loop blocks, the peeled
`swa_attn` shared by layers 0 and 1, and one per peeled MoE scope, which the
folded epoch keeps distinct. Host submissions drop 1131 → **129**; the loop's
`csa_attn`/`hca_moe` replays cover layer 42 as before. Measured on a2a3,
`--rounds 5` with the device run skipped, first pass per rank dropped, per-phase
minimum over the 8 warm passes:

| phase | main | seven Definitions | delta |
| ----- | ---- | ----------------- | ----- |
| `host_orch` | 2.314 ms | 1.291 ms | -44% |
| `graph_upload` | 0.451 ms | 1.384 ms | +207% |
| `sm_h2d` | 0.741 ms | 0.109 ms | -85% |
| control-plane total | 3.594 ms | 2.998 ms | -17% |

The shape of that is the result, not the `host_orch` line. Recording work leaves
the submitting thread, so `host_orch` drops; but seven Definitions are seven
images to ship, so `graph_upload` triples and takes back most of the win.
`sm_h2d` falls because 129 task descriptors are shipped instead of 1131. Netted
within a pass, the control plane improves 17% at best, and **at the median it
does not improve**: 3.883 ms against main's 3.715 ms. The single-recorder form is
nearly deterministic (`host_orch` spread 2.314-2.469, 6%); this form spans
1.291-3.010 (133%), because its wall now depends on seven recording threads
getting CPU on a shared box. So the trade is a predictable cost for a lower floor
and a higher ceiling, and on a loaded machine the ceiling is what a caller sees.

Two costs on the recorder side were large enough to matter at this Definition
count, both fixed alongside the decomposition rather than tuned around:
`graph_prepare` re-derived the boundary match it was already handed (32-46 µs per
recording, before the recording's first node), and growing the recorder pool past
four prewarmed workers put `pthread_create` on the submitting thread mid-burst
(170 µs and 98 µs gaps between Graph submissions, one of them three thread
creations at once).

The stage this was aimed at is still not where the case's host time lives.
`record_node` + `build_definition` + `graph_upload` are rebuilt from scratch on
every run because the Definition cache lives in the per-run `GraphHostState`:
roughly half the host prepare path re-deriving artifacts identical to the
previous run's. And at the measured commit the `bind` stage as a whole is
dominated by neither, at **2.65 s** for `args` (staging this network's weights
and registering their device buffers) and 312 ms for `host_view_close` against
~4 ms of orchestration. The staging-view change later removed those per-tensor
registrations, so current runs report `host_view_close count=0 bytes=0`; the
remaining weight-staging cost is still a fixture artifact a serving loop would
pay once.

## What is still open

- **`graph_upload` is now the dominant control-plane phase for this case**, at
  1.384 ms for seven images against 0.451 ms for one. Uploading a Definition once
  as a shared device object is the lever, not a lower Definition count.
- **After cross-run Definition retention lands** coverage is what matters: on
  rounds 2..N a covered layer costs one ~2.3 µs replay and an uncovered one
  ~250 µs of submits, so the seven Definitions stop being re-recorded per run and
  the floor becomes the median.
- **A network whose layers repeat more than this one's does better.** The
  arithmetic here is driven by DeepSeek-V4 having 2 SWA + 21 CSA + 20 HCA layers
  and three MoE routing variants; a uniform stack would have one attention and one
  MoE block covering every layer from a single recording.

## References

- PR #1900 (the case and its Graph form), #1860 (its predecessor), #1897 (the
  overlap-recording change the first verdict turned on), #1929 (the keyed
  in-flight map and prewarmed recorder pool that removed the single slot).
- [2026-08 — hbg: uploading Graph Definitions once as shared device objects](2026-08-hbg-graph-definition-single-upload.md)
  — the same `graph_upload` stage, and the source of the ~17 µs per
  alloc-and-copy pair figure the submission count multiplies.
- `KNOWN_ISSUES.md`: the per-run Definition cache discard, and the boundary-tag
  defect this work surfaced (ten boundary tensors the body writes are tagged
  `add_input`, so the outer GRAPH task never registers as their producer).
