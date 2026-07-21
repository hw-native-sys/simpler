---
name: l0-swimlane
description: Produce an L0 (intra-core AICore pipeline) swimlane for one task — a single kernel or a mix — via the dump-driven `simpler_setup.tools.l0_swimlane` tool. Use when the user asks to "run/produce an l0 swimlane", "trace a task's intra-core pipeline", profile why one AICore task is slow inside the core(s), or needs help choosing the tool's manual flags (`--func-id`, `--set-arg`, `--spmd-block-num`, `--case`). The tool captures real per-task args from an args dump and auto-generates the `msprof op simulator` replay — no hand-authored workspace. For a hand-authored single-`kernel_entry` replay use [insight-trace](../insight-trace/SKILL.md); for cross-task / scheduler / dependency timing use the L2 swimlane.
---

# L0 Swimlane — Intra-core Pipeline Trace for a Task

`python -m simpler_setup.tools.l0_swimlane` dumps a task's real `args[]`,
reconstructs them, generates a combined `msprof op simulator` replay of the
**whole task** (a mix runs AIC + AIV0 + AIV1 in one op), and exports an
Insight `trace.json` whose lanes are the cluster's pipes. Full reference:
[docs/dfx/l0-swimlane-profiling.md](../../../docs/dfx/l0-swimlane-profiling.md).
This skill is the **operating procedure** — above all the one genuinely
manual decision: the slot/value for `--set-arg`.

## When to use

- **Use** when one task (single kernel or mix) is slow and you need the
  per-pipe (`MTE2` / `MTE1` / `CUBE` / `FIXP` / `SCALAR` / `VECTOR`)
  intra-core picture, or to confirm AIC↔AIV overlap inside a mix.
- **Not** for cross-task dependencies / scheduler / dispatch / finish
  timing — that is the **L2 swimlane**. L0 traces ONE task in isolation
  with no AICPU, so inter-task ordering is out of scope (doc §9, tier C).
- **vs `insight-trace`**: that skill hand-authors a wrapper around one
  `kernel_entry`; `l0_swimlane` automates the whole thing from a real dump
  (real args, mix-together, SPMD context synthesised). Reach for
  `insight-trace` only when there is no test/dump to drive the capture.

## Run

```bash
source .venv/bin/activate
source "$ASCEND_HOME_PATH/set_env.sh"          # CANN env (msprof on PATH)
# Sim dump (no NPU); task-submit locks a device for the step-5 collect.
task-submit --device auto --max-time 1800 --run \
  "python -m simpler_setup.tools.l0_swimlane --platform a2a3sim \
     --func-id <set> --test <test_file.py>"
```

Onboard `a2a3` instead of `a2a3sim`: run
`.claude/skills/onboard-arch-precheck/check.sh a2a3` first (the dump then
runs on the locked device). The five internal steps and all flags are in
doc §3.2 / §3.3.

## Choosing the manual flags (the hard part)

### `--func-id` — the task's member set

You wrote the orchestration, so the members are known. `--func-id 0` traces
the single-kernel task `{0}`; `--func-id 0,1,2` traces that 3-way mix.
It must equal a dispatched task's func_id **set** — for a same-AIV-on-both-
lanes SPMD mix the dump records a duplicate (`[0,1,1]`), so pass
`--func-id 0,1` (`set([0,1,1]) == {0,1}`). Wrong set → the tool lists the
func_id shapes present in the dump; pick one of those.

### `--set-arg SLOT=VALUE` — only when a loop count must shrink

First classify where the kernel's loop trip count comes from:

| Trip count from | `--set-arg`? | Rule |
| --------------- | ------------ | ---- |
| **Tensor shape** (e.g. `shapes[0] / TILE_ELEMS`) | **No** | shape is the real dump value; changing it distorts. (mixed_example / single-kernel rows need no `--set-arg`.) |
| **A scalar arg** (e.g. `n_blocks`) | **Yes** — set the count directly | camodel would run the full loop; shrink to ≥ 3–4 (doc §7.2: floor 3, prefer 4). |
| **A control-tensor's content** (e.g. `context_lens`) | **Yes** — fill the buffer | the kernel *derives* the count from the data; fill so the derived count ≈ 4 (need `block_size` to back out the value). Integer dtypes only. |
| **The SPMD `block_num`** | **No** — use `--spmd-block-num` | block_num lives in the synthesised slot-48 context, which `--set-arg` cannot reach. |

Then find the slot — it is **per-kernel, never fixed**. Discover it:

1. Run once with `--no-collect`; step 3 prints the **arg-slot table**
   (every slot: index / kind / shape / scalar value).
2. Identify which slot is the loop bound by cross-referencing **any** of:
   the kernel's `args[N]` reads, the kernel-top **args-layout comment**
   (paged-attention kernels have one, e.g.
   `args[15] = total_logical_blocks scalar`), or the orchestration's
   `add_input` / `add_scalar` **order** (the i-th `add_*` is slot `i`).
3. Set the value per the table above.
4. Re-run, then **self-check** (below).

Verified examples (slots read from source):

| Test | Loop bound | Flag |
| ---- | ---------- | ---- |
| `paged_attention_unroll` | `aic_qk_matmul.cpp` `args[4] = n_blocks` (scalar) | `--set-arg 4=4` |
| `qwen3_14b_decode` (fa_fused) | `fa_fused_aic/aiv.cpp` outer loop `for(i=block_idx; i<v1[0]; i+=24)`; `v1[0]` = slot 0 `fa_total` (a 1-elem **INT32 tensor** read as the work-item count) | `--set-arg 0=96` → `ceil(96/24)=4` blocks. Slot 0 is 0 in replay → empty trace without this |
| `batch_paged_attention` | `context_lens` **tensor** (slot 1; 2nd `params_sf` `add_input`); the SF kernel (func 1) derives per-batch blocks from its content | `--set-arg 1=512` |

### `--spmd-block-num N` — SPMD grid width

`block_num` is written into the synthesised slot-48 `LocalContext`. Default
is the case's `block_dim`; override only for a kernel that branches or
grid-strides on `block_num` (e.g. set the real hw width `24`). `block_idx`
is always synthesised to `0` (a representative block) and is **not** a flag —
it has no instruction-stream branches (doc §8).

### `--case NAME` — pin a small case on a multi-case test

When the test declares several `CASES[*]`, omitting `--case` auto-pins the
**first** case that lists your `--platform` (a deterministic single-case
dump). That first case is **not** guaranteed to be the smallest, and the
replay rebuilds every tensor at its **real dumped shape**: a production-size
case (long sequence, big batch, large KV cache) makes the camodel — a
cycle-accurate, whole-chip, serial simulator — **crawl or look hung** on the
oversized buffers. So **pin the small one yourself** with `--case <name>`
(accepts `ClassName::Case`) whenever the first-platform case is not the
smallest. Pick the case with the smallest shapes. `--set-arg` shrinks a
*loop count*; `--case` shrinks the *tensor shapes* — reach for `--case`
first when a replay stalls. Single-case tests need no `--case`.

Pick a case that is **scaled down, not reshaped** — same tile geometry
(M/K/N, head_dim, tile size), just fewer blocks / shorter sequence — so the
per-block pipeline stays identical to production (you lose only iteration
*count*, which does not change the pipeline shape). A case with different
tile shapes traces only itself, not production.

## Self-check after every run

A known msprof/camodel export bug can truncate the last loop iteration(s).
Verify `MMAD == FIX_L0C_TO_DST == n_blocks` in the trace; if they disagree,
the tail was cut — do not draw timing conclusions, re-run or change the loop
count (doc §7.4). Read the auto-generated `*_trace_perfetto.json`, not the
raw Insight `trace.json`, for sub-laned per-instruction overlap (doc §3.4).

## Coverage

Representative command per task shape (single AIC / single AIV / 1+1 mix /
2-AIV mix / 3-way mix / SPMD single-source / SPMD coop mix / same-AIV-both-
lanes / paged-attn scalar & control-tensor loops / qwen3) is in doc §3.7.
