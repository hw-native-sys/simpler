# Core Swimlane Profiling — Intra-core Pipeline Trace for a Task

## 1. Background & Motivation

[chip swimlane](chip-swimlane-profiling.md) answers *where each task ran on
the wall clock and how the scheduler spent its loop*. It stops at the
AICore task boundary — one task is one opaque `[start, end]` block. When
a single task is slow, the next question is **why inside the core(s)**:
which pipe (`MTE2` GM→L1, `MTE1` L1→L0, `CUBE` matmul, `FIXP` write-back,
`SCALAR`, `VECTOR`) is the bottleneck, and how the per-instruction issue
overlaps across the cluster's sub-cores.

core swimlane captures exactly that — the **intra-core pipeline** of a
task. It runs the task in isolation under `msprof op simulator` (the
AICore camodel) and exports a MindStudio Insight `trace.json` whose lanes
are the cluster's pipes, not the chip's cores. It deliberately
**bypasses AICPU orchestration**: scheduler / tensormap / ringbuffer
state is out of scope (that is L2's job, and needs real silicon). Core swimlane is
the per-pipe, per-instruction zoom that sits one level below an L2 task
block.

A task may be a single kernel or a **mix** — multiple sub-task kernels
sharing one `args[]` on the 1C2V cluster (1 AIC + up to 2 AIV). Core swimlane
replays the **whole task together**: a mix runs its AIC + AIV0 + AIV1
kernels in one combined op, so the trace shows all the cluster's
sub-cores side by side, not one kernel in isolation.

The hard part of an isolated replay is rebuilding the task's exact
`args[]` — Tensor descriptors (shape / dtype / strides / start_offset)
plus scalar values — which orchestration normally computes on the fly.
Hand-authoring them is error-prone. core swimlane removes the guesswork:
it captures the **real** per-task args from an [args
dump](args-dump.md), uses the dump's `func_id` array to identify the
task's mix members, and generates the whole replay workspace from those
captured args — zero hand-written shapes or scalars.

## 2. Overview

- **Per-pipe instruction timeline** — one Insight lane per sub-core pipe
  (`MTE2` / `MTE1` / `CUBE` / `FIXP` / `SCALAR` / `VECTOR`), each carrying
  the kernel's individual instructions with simulated `ts` / `dur`.
- **Mix-together replay** — an entire mix task (any mix: same- or
  different-source members, 2-way or 3-way) replays as **one** combined
  `msprof op simulator` op. The cube sub-core runs the AIC member, the
  vec sub-cores run the AIV member(s) → a combined AIC+AIV swimlane.
- **Zero-guess args** — the task's real Tensor descriptors and scalars
  come from a `--dump-args hybrid` capture. When a structured
  control tensor also needs its real contents, repeatable
  `--restore-arg SLOT` restores payload already selected in orchestration by
  `CoreTaskArgs::dump(...)` into the replay on `a2a3` / `a2a3sim`. The dump's
  `func_id` array gives the task's mix membership directly.
- **Loop-count control (`--set-arg SLOT=VALUE`)** — when a kernel's loop
  trip count comes from a scalar or a control tensor, override it to
  shrink a runaway loop (so the camodel doesn't hang) or to fix a
  "fake-fast" zero-filled control tensor — without distorting the
  per-iteration pipeline structure. Repeatable; scalars keep their real
  dump values, while unselected tensor payloads default to zero. See
  [§7.2](#72-arg-floor-for-a-loop-count-without-distortion).
- **Source-line attribution (`--debug-line` / `-g`)** — compile the
  kernel with `-g` (skipping the link strip) so the trace carries
  `debug_line` and Insight maps each instruction back to its kernel
  source line. Off by default.
- **Sim or onboard capture** — with a sim `--platform` (`a2a3sim` /
  `a5sim`) the dump runs with no NPU; with an onboard `--platform`
  (`a2a3` / `a5`) it runs on a real device. The dump only needs arg
  **geometry**, which sim captures identically to onboard, and the
  replay is camodel either way — so `a2a3sim` is the default and needs
  no NPU and no arch-precheck. Use onboard only for a kernel whose sync
  idiom (e.g. a manual `prod.record()`) compiles only for the device.
  Tensor restoration is currently an a2a3-only capability; a5 metadata-only
  and zero-filled replays remain available.
- **Two trace outputs** — a native Insight `trace.json` and an
  auto-generated Perfetto-friendly variant (sub-laned + atomic flags;
  see [§3.4](#34-viewing-insight-vs-perfetto)).

Drive it in one line (`--func-id` is the task's member set):

```bash
python -m simpler_setup.tools.core_swimlane \
    --test tests/st/<case>/test_<name>.py --func-id 0,1,2 --platform a2a3sim
```

## 3. How to Use

### 3.1 Prerequisites (one-time per test case)

core swimlane reuses the args-dump pipeline to recover args, so the target
case must satisfy what the dump needs (see
[args-dump.md](args-dump.md)):

1. **Args dump is compiled in.** Built into the platform code; needs a
   `pip install --no-build-isolation -e .` so it is compiled in.
2. **Incores declare complete signatures.** Under the #1181 positional
   model, each incore declares its full tensor `signature` (covering the
   task payload in slot order); the dump maps signature entry `i` to
   payload slot `i` and stamps every record with the task's active
   sub-task `func_id` **array** (its mix membership). This is the repo
   norm — no l0-specific marker.
3. **The case declares the `--platform` you pass.** `CASES[*].platforms`
   must include it. Pick a case with shapes small enough for the camodel
   replay buffers.
4. **`name` is optional.** When `incores[*].name` is absent the tool
   falls back to the kernel source filename for labels / paths.

### 3.2 Run

```bash
# Environment (once per shell): activate the venv and source CANN.
source .venv/bin/activate
export ASCEND_HOME_PATH=<your CANN install>     # e.g. .../cann-9.0.0
source "$ASCEND_HOME_PATH/set_env.sh"

# Sim capture (no NPU dump) — the default.
python -m simpler_setup.tools.core_swimlane \
    --test tests/st/a2a3/tensormap_and_ringbuffer/mixed_example/test_mixed_example.py \
    --func-id 0,1,2 --platform a2a3sim

# Onboard capture — wrap the WHOLE tool in one task-submit so the dump and
# the collect share the locked $TASK_DEVICE (no nested lock). Only needed for
# a kernel whose sync idiom compiles only for the device.
task-submit --device auto --device-num 1 --run \
    "python -m simpler_setup.tools.core_swimlane \
        --test tests/st/<case>/test_<name>.py --func-id 0 --platform a2a3"
```

The tool runs five steps internally (the "Uses NPU" column is for an
onboard `--platform`; a sim `--platform` uses no NPU until step 5):

| Step | Action | Uses NPU |
| ---- | ------ | -------- |
| 1 | Read the test's `CALLABLE`; build a `func_id → (source, core_type)` table | No |
| 2 | Run `--dump-args hybrid` → full `args_dump.json` plus `args.bin` for tensors marked with `CoreTaskArgs::dump(...)` (or reuse via `--dump-json`) | Onboard only |
| 3 | Select the task whose member set == `--func-id`, reconstruct its full positional args, **print the arg-slot table** (slot / kind / shape / value) | No |
| 4 | Emit the replay workspace and smoke-build it locally | No |
| 5 | `msprof op simulator` collect + export → `trace.json`, then auto-converts a Perfetto variant | **Yes** |

Step 3 prints every arg slot so you can pick a `--set-arg` target without
reading the kernel source — names are not in the dump (only kind / shape
/ value), so cross-reference the kernel's `args:` header for those:

```text
[core_swimlane] func_id=0 task=0x... mix=[0, 1, 2] mode=mix block_num=3
              members=[MATMUL(aic,func 0), ADD(aiv,func 1), MUL(aiv,func 2)]
[core_swimlane] arg slots (override with --set-arg SLOT=VALUE or --restore-arg SLOT):
    slot 0  tensor  FLOAT32  [16384]
    ...
```

A scalar slot holds the value directly (`--set-arg 4=4`); a tensor slot
holds a pointer, so `--set-arg` fills its buffer (`--set-arg 4=512`). See
[§7.2](#72-arg-floor-for-a-loop-count-without-distortion).
Use `--restore-arg 6` instead when a tensor contains a non-uniform
structure or lookup table that a single fill value cannot represent.

### 3.3 Key flags

| Flag | Meaning |
| ---- | ------- |
| `--test <file.py>` | SceneTest test file (required) |
| `--func-id A[,B,C]` | The task's **member set** (comma-separated func_ids), required. `--func-id 0` traces the single-kernel task `{0}`; `--func-id 0,1,2` traces that 3-way mix. The set must exactly match a dispatched task's `func_id` array (you wrote the orchestration, so you know the members) |
| `--task-id <hex>` | Which task instance to replay (default: lowest). Instances of the same mix shape are structurally identical |
| `--platform <p>` | Dump platform → arch / compile / SoC params (default `a2a3sim`). Sim (`a2a3sim` / `a5sim`) dumps with no NPU; onboard (`a2a3` / `a5`) dumps on `$TASK_DEVICE` (wrap the tool in `task-submit`). The replay is camodel regardless; geometry is identical, so prefer sim. Tensor restoration is supported only on `a2a3` / `a2a3sim` |
| `--device <ID>` | NPU device for an onboard dump + collect. **Auto-supplied** — `task-submit` appends `--device <id>` (also `$TASK_DEVICE`). Sim platforms ignore it |
| `--case <NAME>` | Pin the dump to one `CASES[*].name`. Omitting it auto-pins the first case that lists `--platform`; pass it to target a smaller case when that first one overflows the camodel. Accepts `ClassName::Case` |
| `--dump-json <path>` | Reuse an existing `args_dump.json`, skipping the dump re-run. `--restore-arg` requires its selected record payload and sibling `args.bin` |
| `--restore-arg SLOT` | Restore one tensor's captured `before_dispatch` payload on `a2a3` / `a2a3sim`. Repeatable. The tensor must already be marked with `CoreTaskArgs::dump(...)` in orchestration; scalar/output-only/unmarked/truncated slots are rejected. Cannot target the same slot as `--set-arg`. Do not use with a5 payloads while #1560 is open |
| `--set-arg SLOT=VALUE` | Override an arg by `args[]` slot. Scalar slot → rewrite value; tensor slot → fill its buffer (integer dtypes). Shrinks a loop count without distortion. Repeatable. By default scalars retain dump values and tensor payloads are zero-filled |
| `--spmd-block-num N` | `block_num` written into the synthesized SPMD context (slot 48). Default: 1 |
| `--debug-line` / `-g` | Compile with `-g` (skip strip) so the trace carries `debug_line` → Insight maps instructions to source lines |
| `--no-collect` | Stop after workspace generation and smoke build; an onboard dump still uses its locked NPU |
| `--max-time <sec>` | `task-submit` budget (default 1800) |
| `--msprof-timeout <minutes>` | `msprof op simulator --timeout` application-run limit in **minutes**. The Core swimlane tool always passes it; default 120, valid range 1–2880. Use a shorter value deliberately for large, repetitive workloads that only need a partial pipeline |

Payload restoration currently targets the a2a3 platform family. The target
tensor must be marked in the task's orchestration before capture. a5/a5sim
tensor payload is not trustworthy because of
[#1560](https://github.com/hw-native-sys/simpler/issues/1560), so it is outside
the current restore contract. An a5/a5sim Level-3 run with marked tensors may
produce `args.bin`, but a Core swimlane that restores its tensor payload is
untrustworthy; this does not prevent a5 metadata-only or zero-filled Core
replays.

Per-arch build parameters are fixed in the tool's `ARCH_CONFIG`:

| arch | SoC (camodel) | aicore-arch (compile) | prologue macros |
| ---- | ------------- | --------------------- | --------------- |
| a2a3 | `dav_2201` | `dav-c220` | `__CCE_AICORE__ 220` / `PTO_NPU_ARCH_A2A3` |
| a5 | `dav_3510` | `dav-c310` | `__CCE_AICORE__ 310` / `PTO_NPU_ARCH_A5` |

### 3.4 Viewing: Insight vs Perfetto

The workspace lands at
`outputs/core_swimlane_<label>_<ts>/`, where
`<label>` = `<TestClass>_<Case>_<platform>_<kernel>_mix<members>` (the
`mix<members>` segment is the task's func_id set, e.g. `mix0_1_2` for a
3-way mix or `mix0` for a single-kernel task). Two final
traces are written, both using that same `<label>` (pretty-printed):

| File | Open in |
| ---- | ------- |
| `<label>_trace.json` | **MindStudio Insight** (a copy of the export) |
| `<label>_trace_perfetto.json` | **Perfetto** (auto-converted, see below) |

The raw export is under `<ws>/insight_export/OPPROF_*/simulator/`.

- **Insight** — drag the `simulator/` directory in (native, correct), or
  open `<label>_trace.json`.
- **Perfetto** — opening the raw Insight `trace.json` directly **drops
  task records and mis-pairs flags** (Insight packs concurrent,
  pipelined instructions onto one track; overlapping `ph:X` events break
  stack nesting and `B`/`E` pairing). The tool therefore emits
  `<label>_trace_perfetto.json` with two **lossless** transforms:
  concurrent instructions on a pipe are split into sub-lanes
  (`MTE1#0..#k`, none overlapping within a lane), and each `B`/`E` flag
  pair is merged into one `ph:X` slice. Open that file in Perfetto. The
  same transform is documented in
  [`.claude/skills/insight-trace/SKILL.md`](../../.claude/skills/insight-trace/SKILL.md);
  here it is built into the tool.

### 3.5 Selecting a task / mix, and what to initialize

`--func-id` **is** the task's member set — you name the exact func_ids the
task is made of, and the tool picks the task whose `func_id` array matches.
There is no shape-guessing: you wrote the orchestration, so you know which
func_ids bind into a task. Name the task's **full** member set — for a mix,
that means all of its members, so the trace shows the whole cluster's
sub-cores cooperating as they do in production.

- **Single-kernel task** — `--func-id 0` selects the task whose set is
  exactly `{0}` — a kernel the orchestration dispatches on its own (e.g.
  `vector_example`'s `kernel_add`, or a standalone AIC matmul).
- **A mix** — name every member: `--func-id 0,1,2` selects the 3-way mix
  `{0,1,2}`, `--func-id 3,4` the 2-AIV mix `{3,4}`.
- If the set matches no dispatched task, the tool errors and lists the
  `func_id` shapes present in the dump.

**What loop count to shrink** for a fast camodel is a scalar `n_blocks`,
a control *tensor* (`context_lens`), or nothing — the slot is shown in
the step-3 table and `4` is the prefetch floor (see
[§7.2](#72-arg-floor-for-a-loop-count-without-distortion)). For
the `mixed_example` matmul/add/mul kernels the loop count derives from
the tensor **shape** (`shapes[0]`), which the dump captures truthfully,
so **no `--set-arg` is needed** — the real count (one 128×128 tile) is
already small.

Use `--restore-arg` only when the kernel reads non-uniform tensor contents
that affect control flow or addressing — for example a runtime-built tiling
structure. Mark the tensor in the task's `CoreTaskArgs::dump(...)`; a fresh run
then uses the hybrid dump mode for full metadata and reuses `partial`'s mask for
payload. Restoration is currently supported only with `--platform a2a3` or
`a2a3sim`; do not consume a5 tensor payload while #1560 is open. Do not mark
weight, activation, or KV-pool tensors merely for replay.
Slots restored from their `before_dispatch` snapshot cannot also use
`--set-arg`.

Omitting `--case` auto-pins the **first** `CASES[*]` that lists your
`--platform`, so the dump always targets exactly one case (deterministic —
no "run every case, reconstruct from the newest dump dir" ambiguity). Pass
`--case` explicitly when that first case is not the smallest — a full-size
production case's shapes can make the camodel impractical (§3.6). The
synthesized slot-48 `block_num` comes from `--spmd-block-num` (default 1);
the args dump does not carry that context, so consult the selected task's
orchestration for its real logical width.

### 3.6 Oversized CAModel cases

CAModel is a cycle-accurate, whole-chip simulator. The replay allocates every
tensor at the shape recorded in `args_dump.json`, including tensors whose
payload is left zero. Marking only the required tensors with
`CoreTaskArgs::dump(...)` reduces the level-3 `args.bin` size and onboard dump
pressure; it does **not** reduce replay allocation, zero-initialization, or
kernel loop work. A large KV pool or weight tensor can therefore remain
expensive even when only a small metadata slot is restored.

Use this order:

1. **Smoke-build before collecting.** Reuse the dump when available and pass
   `--no-collect`. Step 3 prints every slot and shape, and the generated
   workspace proves the sources and argument layout compile without spending
   time in CAModel.
2. **Shrink shapes with `--case`.** Choose a bounded case that preserves the
   target kernel's compile-time batch/head/tile geometry and mix membership,
   while reducing shape-driving quantities such as layer count, page count,
   cache rows, or sequence blocks. Changing only a runtime `seq_len` does not
   help when allocation still uses a fixed `MAX_SEQ`, layer count, or cache
   extent.
3. **Shrink only the loop when shapes are already acceptable.** Use
   `--set-arg SLOT=VALUE` for scalar or uniform integer control inputs. It does
   not resize tensor descriptors. Keep at least 3–4 iterations when the goal
   is steady-state pipeline analysis.
4. **Restore only structured control inputs on a2a3/a2a3sim.** Use
   `--restore-arg` for tiling structs or lookup tables; do not restore weights,
   activations, or KV pools. Restoration preserves control truth but is not a
   scaling mechanism. Do not consume A5 tensor payload while
   [#1560](https://github.com/hw-native-sys/simpler/issues/1560) is open.
5. **Use a shorter timeout only when partial simulation is useful.** The
   default is 120 minutes, so ordinary cases normally finish in full without
   changing this option. For a large, repetitive case,
   `--msprof-timeout 4` asks msProf to stop after four minutes and parse the
   simulation data produced so far. This follows the
   [Ascend CANN official QA/parameter guidance for msProf `--timeout`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/devaids/optool/atlasopdev_16_0082.html):
   the option is intended for data-heavy, repeatedly computed operators whose
   full simulation takes a long time, when a partial pipeline is enough to
   obtain the necessary information. The official range is 1–2880 minutes.

A timeout-generated trace is intentionally partial. It can answer questions
about an observed steady-state pipeline segment, but it cannot establish total
kernel duration or complete tail behavior, and the normal MMAD/FIX tail-count
self-check does not apply to a deliberately truncated run. If the timeout
fires before the target kernel produces any profiler data, msProf may have
nothing to parse and no `trace.json` will be available.

Typical two-stage invocation:

```bash
# Validate task selection, restored slots, shapes, and compilation.
python -m simpler_setup.tools.core_swimlane ... \
    --case <bounded-case> --dump-json <args_dump.json> --no-collect

# Request a four-minute partial simulation for a large, repetitive case.
python -m simpler_setup.tools.core_swimlane ... \
    --case <bounded-case> --dump-json <args_dump.json> --msprof-timeout 4
```

If the workload has no bounded case, create a focused workload-local case that
submits the same target task with the same func-id set, argument order, tiling
path, and tile geometry, but smaller shape-driving extents. A one-off
diagnostic case should not be committed unless it provides durable regression
coverage.

### 3.7 Reusing a dump across kernels

The dump in step 2 is the slow part. When tracing several tasks from the
same case, capture once and reuse:

```bash
# First: runs the dump, traces one task (the 3-way mix).
python -m simpler_setup.tools.core_swimlane --test <file> --func-id 0,1,2 --platform a2a3sim

# Subsequent: another task from the same case, reusing the manifest.
python -m simpler_setup.tools.core_swimlane --test <file> --func-id 3,4 --platform a2a3sim \
    --dump-json outputs/<ClassName>_<Case>_<ts>/args_dump/args_dump.json
```

The manifest holds every task's args for the whole case. The default level-3
manifest is sufficient for descriptor/scalar reconstruction. Reusing it with
`--restore-arg` is rejected if the requested tensor was not marked with
`CoreTaskArgs::dump(...)`. Add the marker and capture a fresh Level-3 run, or
reuse a payload-carrying manifest together with its sibling `args.bin`.

### 3.8 Coverage across the #1181 test suite

Commit `b1e4bd23` (#1181) touched ~70 test files. The
`tensormap_and_ringbuffer` kernels among them fall into these Core swimlane
categories — one representative each, with its verified `--func-id`. The
runnable commands follow the table, wrapped in `task-submit` (the step-5
`msprof` collect takes a device on the shared box). Most use
`--platform a2a3sim` (the dump runs off-NPU); `alternating_matmul_add`,
and `paged_attention_unroll`, whose `CASES` declare
**no `a2a3sim`**, are grouped separately and use `--platform a2a3` after an
arch-precheck (the case must declare the `--platform` you pass — §3.1).

| Category | Representative `<TEST>` | `--func-id` + flags | What it exercises |
| -------- | ----------------------- | ------------------- | ----------------- |
| Single AIC | `alternating_matmul_add` | `--func-id 0` | standalone `rt_submit_aic_task(MATMUL)` — a genuine single-AIC task, not a mix member (a2a3-only) |
| Single AIV | `vector_example` | `--func-id 0` | `kernel_add`, dispatched `rt_submit_aiv_task(0)` (vec only) |
| Mix 2 AIV (per-lane) | `mixed_example` | `--func-id 3,4` | ADD_STD@AIV0 + MUL_STD@AIV1 (`get_subblockid` routing) |
| Mix 3-way 1C2V | `mixed_example` | `--func-id 0,1,2` | MATMUL@AIC + ADD@AIV0 + MUL@AIV1 |
| SPMD single-source | `spmd_multiblock_aiv` | `--func-id 0 --spmd-block-num 4` | single AIV reading `get_block_idx`; the default lowest task-id is orchestration task T0 with logical `block_num=4` |
| SPMD mix, 2 AIV share a source | `spmd_multiblock_mix` | `--func-id 0,1,2 --spmd-block-num 2` | the default lowest task-id is orchestration task T0 with logical `block_num=2`. Func 1 & 2 are distinct ids but **both `kernel_spmd_mix.cpp`** → the 2 AIV collapse to one (both lanes run it). Routes by `get_sub_block_id` (slot 49) → in replay both lanes read `sub_block_id=0`; AIV0/AIV1 differ only by write offset, so the pipeline stays representative. (The same-source collapse also covers the duplicate-func_id `[0,1,1]` shape an SPMD mix produces when `aiv0 = aiv1`.) |
| Paged-attn, loop = scalar | `paged_attention_unroll` | `--func-id 0 --set-arg 4=4` | QK stage; `n_blocks` scalar (slot 4) → shrink to 4 ([§7.2](#72-arg-floor-for-a-loop-count-without-distortion)) |
| Paged-attn, loop = control tensor | `batch_paged_attention` | `--func-id 1 --set-arg 1=512 --case CaseSmall1` | SF reads `context_lens` (**slot 1**) content (`aiv_softmax_prepare.cpp`); `--set-arg 1=512` fills it uniformly → shrinks the derived per-batch block count |

Runnable commands (one per category):

```bash
T=tests/st/a2a3/tensormap_and_ringbuffer        # most representatives
E=examples/a2a3/tensormap_and_ringbuffer        # vector_example

# --- a2a3sim cases (case declares a2a3sim; dump takes no NPU) ---
CORE="python -m simpler_setup.tools.core_swimlane --platform a2a3sim -g"  # -g: source-line attribution
# Single AIV — vector_example kernel_add
task-submit --device auto --max-time 1800 --run "$CORE --func-id 0     --test $E/vector_example/test_vector_example.py"
# Mix 2 AIV (per-lane) — ADD_STD + MUL_STD
task-submit --device auto --max-time 1800 --run "$CORE --func-id 3,4   --test $T/mixed_example/test_mixed_example.py"
# Mix 3-way 1C2V — MATMUL + ADD + MUL
task-submit --device auto --max-time 1800 --run "$CORE --func-id 0,1,2 --test $T/mixed_example/test_mixed_example.py"
# SPMD single-source
task-submit --device auto --max-time 1800 --run "$CORE --func-id 0 --spmd-block-num 4 --test $T/spmd_multiblock_aiv/test_spmd_multiblock_aiv.py"
# SPMD mix, 2 AIV share a source
task-submit --device auto --max-time 1800 --run "$CORE --func-id 0,1,2 --spmd-block-num 2 --test $T/spmd_multiblock_mix/test_spmd_multiblock_mix.py"
# Paged-attn, loop = control tensor (context_lens = slot 1; fill it to shrink the per-batch block count)
task-submit --device auto --max-time 1800 --run "$CORE --func-id 1 --set-arg 1=512 --case CaseSmall1 --test $T/batch_paged_attention/test_batch_paged_attention.py"

# --- a2a3-ONLY cases (CASES declare no a2a3sim) ---
# Onboard: run arch-precheck once, then --platform a2a3 (the dump runs on the locked device).
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
CORE_HW="python -m simpler_setup.tools.core_swimlane --platform a2a3 -g"
# Single AIC — standalone matmul (genuine single-AIC task)
task-submit --device auto --max-time 1800 --run "$CORE_HW --func-id 0 --test $T/alternating_matmul_add/test_alternating_matmul_add.py"
# Paged-attn, loop = scalar (shrink n_blocks to 4)
task-submit --device auto --max-time 1800 --run "$CORE_HW --func-id 0 --set-arg 4=4 --test $T/paged_attention_unroll/test_paged_attention_unroll.py"
```

`qwen3_14b_decode` used to serve as the "real SPMD workload" row here, driving
its generated `fa_fused` mix with `--set-arg 0=96` on the `fa_total` work-item
count. That kernel is gone: its attention is now a CANN
`FusedInferAttentionScore` extern. Being an extern is not a blocker — the replay
still feeds `kernel_entry`. Its work distribution now comes from a
`FAInferTilingData` struct in the slot-6 `UINT8 metadata` buffer that
`paged_attention_tiling_cce` fills at runtime. After that slot is marked with
`CoreTaskArgs::dump(...)` in orchestration, `--restore-arg 6` can restore its real
bytes without trying to synthesize a struct through uniform `--set-arg` filling.
The committed production case is too large for a practical
cycle-accurate replay, so it is not listed as a runnable coverage case here.

**Not Core swimlane targets (excluded).** Runtime-mechanics tests (`orch_so_cache`,
`prepared_callable`, `dynamic_register`, `l3_group`, `l3_dependency`,
`worker_chip_orch_comm`, `aicore_op_timeout`, `scope_stats`); comm / notify tests
(`tests/st/worker/comm_domain/async_notify`,
`tests/st/worker/comm_domain/deferred_notify`, and
`sdma_async_completion_demo`); DFX wrappers that reuse other kernels
(`dep_gen`, `pmu`, `args_dump`, `chip_swimlane` — they trace `vector_example`
/ `mixed_example`); `host_build_graph/*` (a different runtime whose dump
stamps `func_id=[-1]`); `spmd_paged_attention` (`pytest.mark.skip` — a known a2a3 507018 flake, #1156; its `[0,1,1]` same-source collapse is covered by `spmd_multiblock_mix`); and the `ut/py/test_task_interface.py` unit test.

## 4. Capabilities

What the core swimlane shows:

- **Per-pipe occupancy** per sub-core for one task, so a memory-bound vs
  compute-bound diagnosis is direct.
- **Cluster overlap** — for a mix, the AIC and AIV sub-cores appear as
  separate lanes in one trace, so you see how the cooperating kernels'
  pipelines overlap intra-cluster.
- **Per-instruction issue overlap** — each instruction is a slice on its
  pipe lane; the Perfetto sub-lane split makes concurrent issue legible.
- **Source-line attribution** (with `--debug-line`).
- **Cross-arch comparison** (`a2a3sim` vs `a5sim`) surfaces real ISA
  differences (see [§7](#7-findings)).

What it does **not** show (use [chip swimlane](chip-swimlane-profiling.md)):

- AICPU dispatch / finish latency, scheduler phases, dependency arrows.
- **Cross-core synchronization timing.** The isolated replay has no
  AICPU, so orchestration-driven inter-core waits are absent — sub-cores
  appear freely parallel (see [§9](#9-limitations), tier C).
- Multi-task placement across clusters. Core swimlane is one task, one cluster.

## 5. How It Works

core swimlane is **tooling-only** — there is no dedicated device-side data
path. It composes three existing pieces: args dump (for args), `msprof
op simulator` (for the pipeline trace), and a generated replay workspace
(for the isolated build).

### 5.1 The generated workspace

A single mix-arch translation unit — no per-member files:

| File | Role |
| ---- | ---- |
| `replay_kernel.cpp` | The combined `replay_entry`. The AIC member is `#include`d under `#if defined(__DAV_CUBE__)`, the AIV member(s) under `#if defined(__DAV_VEC__)`; `replay_entry` routes each sub-core to its kernel (see [§5.4](#54-mix-together-codegen)) |
| `replay_launch.cpp` | `replay_entry<<<1, ...>>>` launcher — one block = 1 AIC + 2 AIV sub-cores |
| `replay_host.cpp` | Builds the 128-byte Tensor descriptors from the dump's real args + fills scalars, then launches. **Auto-generated; never hand-edited** |
| `CMakeLists.txt` | Single mix-arch `.so` (`--cce-aicore-arch=dav-cXXX`) |
| `run_collect.sh` | `msprof op simulator` collect (`--kernel-name=replay_entry`) + export |

### 5.2 Args reconstruction (the zero-guess part)

`reconstruct_task_args` reads `args_dump.json` (`data["args"]`), selects
the task whose `func_id` SET equals `--func-id`, groups by `task_id`
(default: lowest), and
**unions both dump stages** — inputs + scalars from `before_dispatch`,
outputs from `after_completion` — keyed by `arg_index`. It returns the
task's **full positional payload** (every slot, sorted by `arg_index`)
plus the mix membership (`func_id` array, slot order AIC, AIV0, AIV1).
Each member kernel reads its own slice of the shared `args[]` (the
replay places each tensor at its real slot), so feeding the whole
payload to every member is correct. For each tensor it emits the literal
shape / strides / dtype / start_offset into `make_desc`, with these
correctness-critical details:

- **Descriptor field offsets** are pinned by the `static_assert`s in
  `src/{arch}/runtime/tensormap_and_ringbuffer/runtime/tensor.h`
  (`sizeof(Tensor) == 128`).
- **dtype** comes from a string→enum table mirroring
  `src/common/task_interface/data_type.h` (note `BFLOAT16 = 6`).
- **Buffer size uses the extent formula**
  `(start_offset + 1 + Σ(shape[i]-1)*stride[i]) * elem_size`, not
  `numel` — strided / offset views read past `numel`.
- Tensor payloads default to **memset 0**. `--restore-arg` reads the selected
  logical-contiguous payload from `args.bin`, scatters it through the original
  shape/strides/start-offset view, embeds the physical bytes in
  `replay_host.cpp`, and initializes that device buffer with `aclrtMemcpy`.
  `--set-arg` remains the uniform-fill alternative.

### 5.3 Build & collect

The smoke build (no NPU) runs cmake + builds `replay_host`, then asserts
`replay_entry` and `launch_replay` are present in `libreplay_kernel.so`.
With `--no-collect` it stops here. Otherwise the collect step runs
`run_collect.sh` (the camodel needs a device context), locates the
exported `trace.json`, and writes the two viewer copies. Device
selection follows the lock already held:

- **Under an outer `task-submit`** (`$TASK_DEVICE` set): reuse it, no
  nested `task-submit`.
- **Standalone** with `task-submit` on `PATH`: self-lock via
  `task-submit --device auto`.
- **No `task-submit`**: unlocked run with a warning (per
  [running-onboard.md](../../.claude/rules/running-onboard.md)).

### 5.4 Mix-together codegen

`emit_replay_kernel_combined` builds one `replay_entry` that runs every
member of the mix on its sub-core, in a single translation unit:

- **AIC member** — `#include`d under `#if defined(__DAV_CUBE__)`, so it
  compiles in the cube ISA variant.
- **AIV member(s)** — `#include`d under `#if defined(__DAV_VEC__)`, so
  the vector ISA target feature is in scope (compiling an AIV kernel
  outside the vec variant fails on `vadd` / `set_vector_mask`).
- **2 AIV members** — both kernels live in the **same** vec section. To
  avoid same-TU clashes (both define `extern "C" kernel_entry` and a
  `static get_num_tiles`), each `#include` is wrapped in
  `#define kernel_entry l0_f<id>_entry` + `#define get_num_tiles
  l0_f<id>_get_num_tiles` … `#undef`. Keeping it one TU avoids the
  cross-object device-link problem (bisheng device-links per `.o`, so a
  call into a separately-compiled member object does not resolve).
- **`replay_entry`** (`__global__`) routes: the cube section calls the
  AIC member; the vec section calls
  `get_subblockid() == 0 ? <AIV0> : <AIV1>`. A sub-core with no member in
  the set gets an empty body.

**Per-AIV-lane routing primitive — `get_subblockid()`.** simpler's
*runtime* treats CCE `get_subblockid()` as unreliable (issue #900: it
returns 0 for both AIV lanes because the runtime does not program that
register) and reads `get_sub_block_id(args)` from the slot-49
`GlobalContext` instead. That variant is **not** usable here: the
isolated replay synthesizes one shared `args[]`, so slot-49 is a single
value both lanes read identically. The bare camodel op, however, **does**
model the physical sub-block id per AIV lane, so `get_subblockid()` is
the correct primitive in this context — and it is validated to route
correctly (see [§6](#6-validation)).

### 5.5 SPMD context synthesis

SPMD kernels read an execution context the orchestration builds per
dispatch — `LocalContext{block_idx, block_num}` at args slot 48 and
`GlobalContext{sub_block_id}` at slot 49. The isolated replay has no
orchestration, so `replay_host.cpp` **synthesizes** it: one
`LocalContext{block_idx=0, block_num=--spmd-block-num}` + `GlobalContext`
pointed at slots 48/49. This is harmless for positional kernels (they
ignore 48/49) and required for SPMD kernels that read `get_block_idx` /
`get_block_num` (which would otherwise dereference null). `block_idx=0`
traces a representative block. The default `block_num=1` models a single-block
replay and does **not** take multi-block branches such as
`block_idx+1 < block_num`. That default is sufficient only when the kernel
ignores `block_num`. The args dump does not record the synthesized slot-48
context, so read the selected task's logical grid width from its orchestration
and supply it with `--spmd-block-num`. Some workloads derive that value from
`rt_available_cluster_count()`; others set an explicit task-specific value.
Pass the selected task's real value for faithful branch and grid-stride
behavior. A value of at least 2 can exercise the next-block branch as a bounded
approximation when the real value is unavailable, but it is not a substitute
for that value.
Note the per-AIV-lane routing for a mix uses the hardware
`get_subblockid()` (§5.4), not the synthesized slot-49 value.

## 6. Validation

Confirmed on the `a2a3sim` camodel (`mixed_example`):

| Mix | func_id | Result |
| --- | ------- | ------ |
| MATMUL + ADD | `[0,1]` | `cubecore0` MMAD (MATMUL) + `veccore` VADD (ADD) |
| ADD_STD + MUL_STD | `[3,4]` | `veccore0` VADD (ADD), `veccore1` VMUL (MUL) |
| MATMUL+ADD+MUL | `[0,1,2]` | `cubecore0` MMAD, `veccore0` VADD, `veccore1` VMUL |

The 2-AIV cases (`[3,4]`, `[0,1,2]`) confirm `get_subblockid()` routes
the two physical AIV lanes to distinct kernels in the bare camodel op —
i.e. the issue-#900 "0-for-both" behavior is a *runtime* artifact and
does not apply to an isolated replay.

## 7. Findings

Measured behaviors worth knowing before you read a trace.

### 7.1 The a5 camodel is much slower than a2a3 (wall-clock)

The camodel is a cycle-by-cycle, whole-chip (32-core), serial software
model. "Total tick" is **not** comparable across platforms (tick
granularity differs); wall-clock and the simulated µs are. a5 pays
roughly ~25× per tick vs a2a3, so it is slower end-to-end even with
fewer ticks. Much of the cost is fixed setup, so shrinking a loop count
helps only modestly. **Prefer a2a3 for logic validation; run a5 only
when you specifically need the a5 pipeline.**

### 7.2 Arg floor for a loop count (without distortion)

Double-buffered prefetch kernels guard the prefetch + `pipe_barrier`
with `if (i+1 < n_blocks)`:

| `n_blocks` | Captures | Distortion |
| ---------- | -------- | ---------- |
| 1 | No prefetch (`if` never runs) | **Distorted** — double-buffering lost |
| 2 | Prefetch, single buffer phase | Slightly incomplete |
| 3 | Ping-pong both phases + tail block | Faithful (minimum) |
| 4 | Plus one steady-state block | Faithful, most stable |

→ **Floor 3, recommend 4; `n_blocks = 1` always distorts.** Shrinking
the loop count cuts iterations without changing per-block pipeline
structure; it does **not** change template branches (those are decided
by tile shape `shapes[0]`, which must stay real).

**Where the loop count lives — scalar vs tensor.** A single-task
`n_blocks` is a **scalar** (`--set-arg 4=4`); a mix paged-attention
`n_blocks` is **derived from a `context_lens` tensor**, so `--set-arg`
fills that buffer (`--set-arg 4=512` → every element 512 →
`n_blocks = ceil(512 / block_size)`). `--set-arg` accepts a tensor slot
only for **integer** dtypes. A kernel whose loop count is purely a
function of tensor **shape** needs no `--set-arg` (the dump shape is
real).

### 7.3 a2a3 (`dav-c220`) vs a5 (`dav-c310`) swimlanes differ

Each platform runs the kernel under a different msprof SoC config (a2a3 =
`dav_2201` / `dav-c220`, a5 = `dav_3510` / `dav-c310`), so the same kernel
produces a different core swimlane — both in **lane export** (a2a3 shows
`cubecore0 + veccore0/1`; a5 exports only the sub-cores that ran real
code, so an AIC-only kernel shows just `cubecore0`) and in **instructions
/ per-pipe timing** (real ISA). Both are expected, not tool bugs. Read
`cubecore0` for AIC and `veccore` for AIV, and compare structure *within*
one platform, not absolute numbers across the two.

### 7.4 Trace can truncate the last loop iteration(s) — self-check

A known **collection-side bug** in CANN's msprof/camodel: the exported
instruction stream sometimes **ends early**, dropping the last loop
iteration's compute / write-back. Symptom: `MMAD` / `FIX_L0C_TO_DST`
counts come out **less than `n_blocks`** while the loads are complete.
It is **not** a fixed `n-1` rule; the sim runs all blocks, only the
exported stream is cut.

**Self-check:** after each run, verify `MMAD == n_blocks` **and**
`FIX_L0C_TO_DST == n_blocks`. If they disagree the tail was truncated —
do not draw timing conclusions; retry with a different `n_blocks`.

## 8. Fidelity Rules

| Knob | Change it? | Distorts? |
| ---- | ---------- | --------- |
| Tile M/K/N (shape) | **No** | Alters cycle counts and switches template branches |
| Case selection (`--case`) | Pick a *scaled-down* case | Faithful if it keeps the tile geometry (just fewer blocks / shorter sequence); a case that changes tile M/K/N / head_dim traces only itself, not production |
| Scalar values (`scale` / offsets …) | Use real dump values | Wrong value → wrong branch → distorted |
| Loop count (`n_blocks`, via `--set-arg`) | Shrinkable to ≥ 3–4 | Faithful at ≥ 3–4; `= 1` distorts (§7.2) |
| Tensor contents | Zero by default; on a2a3/a2a3sim, `--restore-arg` selected control tensors | Unrestored data-dependent branches/addresses can distort; supported restored slots use their real `before_dispatch` bytes. A5 restoration is excluded by #1560 |
| SPMD `block_idx` (slot 48) | Fixed 0 | Traces a real block 0 — representative for uniform SPMD |
| SPMD `block_num` (slot 48) | Default 1; pass the selected task's real logical width with `--spmd-block-num` when the kernel reads it | `1` is a single-block path; ≥ 2 only approximates a multi-block branch, while the selected task's real value preserves branch and grid-stride behavior |
| Per-AIV-lane routing (`get_subblockid`) | Automatic | Faithful — lanes run their real kernels (§6) |
| Cross-core sync timing | Not modeled | **Optimistic** — sub-cores appear freely parallel (§9 tier C) |
| Cross-platform (a2a3 vs a5) | Set by target | Instructions / timing genuinely differ (real silicon — §7.3) |

## 9. Limitations

- **AICPU orchestration is out of scope.** Core swimlane sees only the AICore
  pipeline of one task. For dispatch / finish / scheduler / dependency
  data use [chip swimlane](chip-swimlane-profiling.md).
- **Orchestration-driven sync is not modeled (tier C).** Two kinds of
  cross-core sync: **(a) in-kernel** — cross-core flags / L2 FIFOs written
  in the kernel (the AIC↔AIV producer/consumer handshake of a cooperative
  mix) — these **are reproduced**, the camodel runs the combined binary
  instruction-by-instruction; **(b) orchestration-driven** — task
  dependencies / barriers / scheduling the AICPU enforces — these are
  **absent**, the isolated replay has no AICPU. So a mix's own in-kernel
  AIC↔AIV coordination is faithful; what's lost is mainly **inter-task**
  ordering (task A → task B), which is out of Core swimlane's single-task scope
  anyway — that is [chip swimlane](chip-swimlane-profiling.md)'s view. Edge
  case: if a mix's sub-core ordering relied on the AICPU rather than
  in-kernel flags, the replay shows those cores more parallel than
  reality.
- **Simulation clock, not silicon.** Use it for *relative* per-pipe /
  per-arch structure, not absolute-latency claims.
- **Unselected tensor payloads are zero.** On a2a3/a2a3sim, restore structured
  control inputs with `--restore-arg`; other data-driven control flow can
  still diverge (§8).
- **A5 restored-payload Core swimlane traces are untrustworthy.** While
  [#1560](https://github.com/hw-native-sys/simpler/issues/1560) is open, do
  not treat A5 args-dump payload as tensor truth or use it with
  `--restore-arg`. A5 may still write level-3 `args.bin` for marked tensors, but
  a Core swimlane restored from it is untrustworthy; metadata-only and
  zero-filled Core swimlane replays remain usable.
- **Tail-truncation collection bug.** Validate `MMAD`/`FIX` counts every
  run (§7.4).
- **1C2V only.** The mix path assumes 1 AIC + up to 2 AIV (the only
  cluster shape both current chips support). A mix with > 2 AIV members,
  or > 1 AIC, is rejected.

## 10. FAQ / Debug Guide

**`func_id=N not found`.** The first `--func-id` member is not an incore;
the tool prints the available `(func_id, name, core_type)` from the test's
`CALLABLE.incores`.

**`--func-id [...] matches no task`.** No dispatched task has exactly that
member set. The tool lists the `func_id` shapes present in the dump — pick
one of those (a shape it printed, not an arbitrary combination of func_ids).

**No dump records for the task.** The incore `signature` likely
disagrees with the dispatched payload, so the dump skipped it — see
[§3.1](#31-prerequisites-one-time-per-test-case) and
[args-dump.md](args-dump.md).

**`--restore-arg` says the manifest has no `bin_file`.** The reused dump
was captured without a matching `CoreTaskArgs::dump(...)` marker. Mark the target
tensor in orchestration and capture a fresh Level-3 run, or reuse a
payload-carrying manifest together with its sibling `args.bin`.

**A5 `--restore-arg` is rejected.** A5 tensor payload restoration is outside
the supported contract while
[#1560](https://github.com/hw-native-sys/simpler/issues/1560) is open. Use
`a2a3` / `a2a3sim` for tensor truth; A5 metadata-only or zero-filled replay
remains available.

**Smoke build fails on a missing symbol.** `replay_entry` /
`launch_replay` must appear in `libreplay_kernel.so`. A wrong
`--platform` picks the wrong `ARCH_CONFIG`.

**`ASCEND_HOME_PATH is not set`.** Source CANN's `set_env.sh` first.

**Both AIV lanes show the same kernel.** `get_subblockid()` did not
distinguish the lanes in your camodel build (the issue-#900 behavior).
Trace each AIV kernel as its own single-kernel task instead (e.g.
`--func-id 3` then `--func-id 4`).

**Perfetto shows overlapping / missing slices.** Open
`<label>_trace_perfetto.json`, not the raw Insight `trace.json`
(see [§3.4](#34-viewing-insight-vs-perfetto)).

**`MMAD` / `FIX` count < `n_blocks`.** The export truncated the tail
(§7.4). Re-run or change `n_blocks`.

## 11. Related docs

- [`.claude/skills/core-swimlane/SKILL.md`](../../.claude/skills/core-swimlane/SKILL.md)
  — the operating procedure for this tool (picking `--func-id` /
  `--restore-arg` / `--set-arg` / `--spmd-block-num`).
- [chip-swimlane-profiling.md](chip-swimlane-profiling.md) — the
  per-task / scheduler swimlane one level up.
- [args-dump.md](args-dump.md) — the `func_id`-array-tagged per-task arg
  capture Core swimlane reconstructs from.
- [`.claude/skills/insight-trace/SKILL.md`](../../.claude/skills/insight-trace/SKILL.md)
  — the manual `msprof op simulator` replay recipe + Perfetto notes.
- [chip-level-arch.md](../chip-level-arch.md) — the AICore pipe model the
  lanes represent.
