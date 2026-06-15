# L0 Swimlane Profiling — Intra-core Pipeline Trace for One Kernel

## 1. Background & Motivation

[L2 swimlane](l2-swimlane-profiling.md) answers *where each task ran on
the wall clock and how the scheduler spent its loop*. It stops at the
AICore task boundary — one task is one opaque `[start, end]` block. When
a single kernel is slow, the next question is **why inside the core**:
which pipe (`MTE2` GM→L1, `MTE1` L1→L0, `CUBE` matmul, `FIXP` write-back,
`SCALAR`) is the bottleneck, and how the per-instruction issue overlaps.

L0 swimlane captures exactly that — the **intra-core pipeline** of one
`kernel_entry(args)`. It runs the kernel in isolation under `msprof op
simulator` (the AICore camodel) and exports a MindStudio Insight
`trace.json` whose lanes are the core's pipes, not the chip's cores. It
deliberately **bypasses AICPU orchestration**: scheduler / tensormap /
ringbuffer state is out of scope (that is L2's job, and needs real
silicon). L0 is the per-pipe, per-instruction zoom that sits one level
below an L2 task block.

The hard part of an isolated replay is rebuilding the kernel's exact
`args[]` — Tensor descriptors (shape / dtype / strides / start_offset)
plus scalar values — which orchestration normally computes on the fly.
Hand-authoring them is error-prone. L0 swimlane removes the guesswork:
it captures the **real** per-task args from a [tensor
dump](tensor-dump.md), filters them by `func_id` to pin one kernel
unambiguously, and generates the whole replay workspace from those
captured args — zero hand-written shapes or scalars.

## 2. Overview

- **Per-pipe instruction timeline** — one Insight lane per AICore pipe
  (`MTE2` / `MTE1` / `CUBE` / `FIXP` / `SCALAR`), each carrying the
  kernel's individual instructions with simulated `ts` / `dur`.
- **Zero-guess args** — the kernel's real Tensor descriptors and scalars
  come from a JSON-only `--dump-tensor 3` capture (metadata + scalar
  values, no `.bin` payload — all reconstruction needs), joined on
  `func_id`. No manual shape / dtype / stride / scalar authoring.
- **Sim or onboard capture** — with a sim `--platform` (`a2a3sim` /
  `a5sim`) the dump runs with no NPU; with an onboard `--platform`
  (`a2a3` / `a5`) it runs on a real device. Onboard is required for
  kernels whose sync idiom (e.g. a manual `prod.record()`) only compiles
  for the device, not the cpu sim. Run the whole tool under one
  `task-submit` so the onboard dump and the `msprof op simulator` collect
  share the locked `$TASK_DEVICE` (see [§3.2](#32-run)).
- **Two trace outputs** — a native Insight `trace.json` and an
  auto-generated Perfetto-friendly variant (sub-laned + atomic flags;
  see [§3.4](#34-viewing--insight-vs-perfetto)).

Drive it in one line (per kernel, selected by `func_id`):

```bash
python -m simpler_setup.tools.l0_swimlane \
    --test tests/st/<case>/test_<name>.py --func-id <N> --platform a2a3sim
```

## 3. How to Use

### 3.1 Prerequisites (one-time per test case)

L0 swimlane reuses the tensor-dump pipeline to recover args, so the
target case must satisfy what the dump needs:

1. **Dump records carry `func_id`.** Built into the platform code; needs
   a `pip install --no-build-isolation -e .` so it is compiled in. See
   [tensor-dump.md §3.3](tensor-dump.md#33-output) for the field.
2. **The incore signatures sum to the real payload.** The dump sums the
   `signature` tensor count of every *active* subtask in a task and requires
   that sum to equal the payload's actual `add_input` / `add_output` count;
   otherwise it skips that task and the kernel becomes un-replayable.
   - **Single-task kernel:** that incore's `signature` tensor count must
     equal its payload tensor count.
   - **Cooperative mix pair** (the same source compiled for both `aic` and
     `aiv`, sharing one `args[]`): declare the full tensor `signature` on
     **exactly one** of the two incores and leave the other's **empty /
     absent**. Both subtasks are active and address the *same* shared
     tensors, so declaring the signature on both doubles the sum (e.g.
     `9 + 9 = 18 != 9`) and the dump is skipped. l0_swimlane merges the pair
     by `func_id`, so the half that carries the signature can be either one
     (convention: the `aic` entry); every dumped tensor is then stamped with
     that subtask's `func_id`. Example: `spmd_paged_attention` puts all 9
     tensors on `PA_AIC` (func_id 0) and gives `PA_AIV` (func_id 1) **no**
     `signature` — so its dump records all carry `func_id 0`, which is
     expected, not a bug.
3. **The case declares the `--platform` you pass.** `CASES[*].platforms`
   must include it — a sim platform (`a2a3sim` / `a5sim`, dump runs with
   no NPU) or an onboard one (`a2a3` / `a5`, dump runs on a real device).
   Pick a case with shapes small enough for the camodel replay buffers;
   device-only-sync kernels (manual `prod.record()`) compile only onboard,
   so they need a small **onboard** case.
4. **`name` is optional.** When `incores[*].name` is absent the tool
   falls back to the kernel source filename for labels / paths.

### 3.2 Run

```bash
# Environment (once per shell): activate the venv and source CANN.
source .venv/bin/activate
export ASCEND_HOME_PATH=<your CANN install>     # e.g. .../cann-9.0.0
source "$ASCEND_HOME_PATH/set_env.sh"

# Sim capture (no NPU dump) — for a kernel whose case declares a sim platform.
python -m simpler_setup.tools.l0_swimlane \
    --test tests/st/a2a3/tensormap_and_ringbuffer/spmd_multiblock_aiv/test_spmd_multiblock_aiv.py \
    --func-id 0 --platform a2a3sim

# Onboard capture — wrap the WHOLE tool in one task-submit so the dump and the
# collect share the locked $TASK_DEVICE (no nested lock). Required for
# device-only-sync kernels (manual prod.record()). --case pins the dump to the
# small case when the test also has a full-size production case.
task-submit --device auto --device-num 1 --run \
    "python -m simpler_setup.tools.l0_swimlane \
        --test tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/test_spmd_paged_attention.py \
        --func-id 0 --platform a2a3 --case SmallCase1"
```

The tool runs five steps internally (the "Uses NPU" column is for an onboard
`--platform`; a sim `--platform` uses no NPU until step 5):

| Step | Action | Uses NPU |
| ---- | ------ | -------- |
| 1 | Read the test's `CALLABLE`; resolve the kernel `source` + `core_type` by `func_id` | No |
| 2 | Run `--dump-tensor 3` (JSON-only) → `tensor_dump.json` (or reuse one via `--dump-json`); onboard, on `$TASK_DEVICE` | Onboard only |
| 3 | Filter the dump by `func_id`, reconstruct that task's real args, and **print the arg-slot table** (slot / kind / shape / value) for `--set-arg` | No |
| 4 | Emit the 5-file replay workspace and smoke-build it locally | No |
| 5 | `msprof op simulator` collect + export → `trace.json`, then auto-converts a Perfetto variant. Reuses the outer `$TASK_DEVICE`; if none, self-locks via `task-submit --device auto` | **Yes** |

Step 3 prints every arg slot so you can pick a `--set-arg` target without
reading the kernel source — names are not in the dump (only kind / shape /
value), so cross-reference the kernel's `args:` header for those:

```text
[l0_swimlane] arg slots (override with --set-arg SLOT=VALUE):
    slot 4  tensor  INT32    [24]          # context_lens (mix) -> --set-arg 4=512
    slot 15 scalar  = 24                   # total_logical_blocks
    ...
```

A scalar slot holds the value directly (`--set-arg 4=4` for a single-task
`n_blocks`); a tensor slot holds a pointer, so `--set-arg` fills its buffer
(`--set-arg 4=512` makes every `context_lens` element 512 →
`n_blocks = ceil(512 / block_size)`). See [§6.2](#62---set-arg-floor-for-a-loop-count-without-distortion).

### 3.3 Key flags

| Flag | Meaning |
| ---- | ------- |
| `--test <file.py>` | SceneTest test file (required) |
| `--func-id <N>` | Which incore kernel to replay (required). For a cooperative **mix pair** (same source, `aic`+`aiv`) either func_id is equivalent — the tool auto-detects the pair and reconstructs/replays the whole `{aic, aiv}` set regardless of which one you name (the dump records all sit under the signature-carrying half, so the merge is what makes the other id resolve). The label follows the id you pass (`PA_AIC` vs `PA_AIV`) |
| `--platform <p>` | Dump platform → arch / compile / SoC params (default `a2a3sim`). Sim (`a2a3sim` / `a5sim`) dumps with no NPU; onboard (`a2a3` / `a5`) dumps on `$TASK_DEVICE` (wrap the tool in `task-submit`) |
| `--device <ID>` | NPU device for an onboard dump + collect. **Auto-supplied** — `task-submit` appends `--device <id>` to the wrapped command (also `$TASK_DEVICE`); that one device threads through both steps. Sim platforms ignore it. Rarely passed by hand |
| `--case <NAME>` | Pin the dump to one `CASES[*].name` (e.g. `SmallCase1`). Use when the test has several cases on `--platform`, so the dump targets the small one — not the full-size production case whose shapes overflow the camodel replay. Accepts `ClassName::Case` too |
| `--task-id <hex>` | Which task instance of that `func_id` to replay (default: lowest) |
| `--dump-json <path>` | Reuse an existing `tensor_dump.json`, skipping the dump re-run |
| `--set-arg SLOT=VALUE` | Override an arg by `args[]` slot for the replay. Scalar slot → rewrite value; tensor slot → fill its buffer with VALUE (integer dtypes). Shrinks a loop count without distortion — scalar `n_blocks` (`--set-arg 4=4`) or the mix `context_lens` tensor (`--set-arg 4=512`). Repeatable. Default: real dump values |
| `--spmd-block-num N` | `block_num` written into the synthesized SPMD context (slot 48). Default: the case's `block_dim`. Only matters for kernels that branch/stride on `block_num` |
| `--debug-line` / `-g` | Compile the kernel with `-g` (and skip link strip) so the trace carries `debug_line` → Insight maps instructions to source lines. Default off |
| `--no-collect` | Generate + smoke-build only; do not take an NPU |
| `--max-tensor-bytes <N>` | Warn threshold for buffer size (bytes); **never truncates** — full size is always allocated for correctness |
| `--max-time <sec>` | `task-submit` budget (default 1800) |

Per-arch build parameters are fixed in the tool's `ARCH_CONFIG`:

| arch | SoC (camodel) | aicore-arch (compile) | prologue macros |
| ---- | ------------- | --------------------- | --------------- |
| a2a3 | `dav_2201` | `dav-c220` | `__CCE_AICORE__ 220` / `PTO_NPU_ARCH_A2A3` |
| a5 | `dav_3510` | `dav-c310` | `__CCE_AICORE__ 310` / `PTO_NPU_ARCH_A5` |

### 3.4 Viewing — Insight vs Perfetto

The workspace lands at
`outputs/l0_swimlane_<TestClass>_<Case>_<platform>_<kernel>_<ts>/`, with
**two** final traces (both self-describing names, pretty-printed):

| File | Open in |
| ---- | ------- |
| `<label>_trace.json` | **MindStudio Insight** (a copy of the export) |
| `<label>_trace_perfetto.json` | **Perfetto** (auto-converted, see below) |

The raw export is under
`<ws>/insight_export/OPPROF_*/simulator/` (`trace.json` +
`visualize_data.bin` + per-core subdirs).

- **Insight** — drag the `simulator/` directory in (native, correct), or
  open `<label>_trace.json`.
- **Perfetto** — opening the raw Insight `trace.json` directly **drops
  task records and mis-pairs flags** (Insight packs concurrent, pipelined
  instructions onto one track; overlapping `ph:X` events break stack
  nesting and `B`/`E` pairing). The tool therefore emits
  `<label>_trace_perfetto.json` after export with two **lossless**
  transforms: concurrent instructions on a pipe are split into sub-lanes
  (`MTE1#0..#k`, none overlapping within a lane), and each `B`/`E` flag
  pair is merged into one `ph:X` slice. Open that file in Perfetto for a
  correct preview. The same transform is documented in
  [`.claude/skills/insight-trace/SKILL.md`](../../.claude/skills/insight-trace/SKILL.md);
  here it is built into the tool.

### 3.5 Kernel shapes & what to `--set-arg`

l0_swimlane handles five kernel shapes. Two things vary by shape. The
**dump platform** is whichever the test case declares: a sim platform
(`a2a3sim`) dumps with no NPU and the collect self-locks; an onboard one
(`a2a3`) runs the dump on the task-submit-locked device (wrap the tool in
`task-submit`); a cooperative mix kernel whose sync only compiles for the
device must be onboard. **Which loop count to shrink** for a fast camodel
is a scalar `n_blocks`, a control *tensor* (`context_lens`), or nothing —
the slot is shown in the step-3 table and `4` is the prefetch floor (see
[§6.2](#62---set-arg-floor-for-a-loop-count-without-distortion)).

| Kernel shape | Representative test · func_id | Dump platform | `--set-arg` (loop count) |
| ------------ | ----------------------------- | ------------- | ------------------------ |
| AIC-only single-task | `paged_attention_unroll` · 0 (QK) | `a2a3` | `4=4` — `n_blocks` **scalar** @slot 4 |
| AIV-only single-task | `paged_attention_unroll` · 1 (SF) | `a2a3` | `5=4` — `n_blocks` **scalar** @slot 5 (slot differs from QK) |
| SPMD AIV (non-mix) | `spmd_multiblock_aiv` · 0 (SPMD_WRITE_AIV) | `a2a3sim` | **none** — one write per block, no inner loop |
| SPMD mix (cooperative) | `spmd_paged_attention` · 0 (PA_AIC) | `a2a3` (must) | `4=512` — `context_lens` **tensor** @slot 4 → `n_blocks=ceil(512/block_size)` |
| Offset subtask (independent kernel packed in a mix dispatch) | `mixed_example` · 1 (ADD) | `a2a3sim` | **none** — `case1` is already small; args sit at slots **3/4/5**, not 0 |

```bash
# AIC-only (QK): n_blocks scalar @slot 4 — onboard, wrap in task-submit
task-submit --device auto --run \
  "python -m simpler_setup.tools.l0_swimlane --func-id 0 --platform a2a3 --case Case1 --set-arg 4=4 \
     --test tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py"

# AIV-only (SF): n_blocks scalar @slot 5
task-submit --device auto --run \
  "python -m simpler_setup.tools.l0_swimlane --func-id 1 --platform a2a3 --case Case1 --set-arg 5=4 \
     --test tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py"

# SPMD AIV: nothing to shrink — sim dump (no NPU), collect self-locks
python -m simpler_setup.tools.l0_swimlane --func-id 0 --platform a2a3sim --case Case1 \
    --test tests/st/a2a3/tensormap_and_ringbuffer/spmd_multiblock_aiv/test_spmd_multiblock_aiv.py

# SPMD mix (cooperative): context_lens tensor @slot 4
task-submit --device auto --run \
  "python -m simpler_setup.tools.l0_swimlane --func-id 0 --platform a2a3 --case SmallCase1 --set-arg 4=512 \
     --test tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/test_spmd_paged_attention.py"

# Offset subtask (mixed_example ADD): an independent AIV kernel packed as a
# non-first mix subtask, so its args start at slot 3, not 0 (confirmed by the
# step-3 slot table). Plain AIV — not cooperative — so it dumps on a2a3sim and
# the collect self-locks; nothing to shrink.
python -m simpler_setup.tools.l0_swimlane --func-id 1 --platform a2a3sim --case case1 \
    --test tests/st/a2a3/tensormap_and_ringbuffer/mixed_example/test_mixed_example.py
```

Always pass `--case` to pin the dump to exactly one case — the dump runs the
whole test otherwise, which wastes time and lets an unrelated case's golden
failure abort the run. Without `--case`, only the default (non-manual) cases
run.

A **scalar** slot holds the value directly; a **tensor** slot holds a
pointer, so `--set-arg` fills its buffer instead. Pick the slot from the
printed slot table — names are in the kernel's `args:` header, not the
dump.

Slots need not start at 0. An independent kernel packed as a non-first mix
subtask reads its args at an offset — `mixed_example`'s ADD (`func_id 1`) sits
at slots **3/4/5** because MATMUL (`func_id 0`) occupies 0/1/2 ahead of it. The
slot table shows where; l0_swimlane places each tensor at its real slot, so
`--set-arg` and the replay both address the offset slot, not a 0-based index.

### 3.6 Reusing a dump across kernels

The dump in step 2 is the slow part (a full sim run). When tracing
several kernels from the same case, capture once and reuse:

```bash
# First kernel: runs the sim dump.
python -m simpler_setup.tools.l0_swimlane --test <file> --func-id 0 --platform a2a3sim

# Subsequent kernels: point at the manifest the first run produced.
python -m simpler_setup.tools.l0_swimlane --test <file> --func-id 2 --platform a2a3sim \
    --dump-json outputs/<ClassName>_<Case>_<ts>/tensor_dump/tensor_dump.json
```

The dump manifest is graph/instance-shaped for the whole case; every
`func_id`'s args live in the one file.

## 4. Capabilities

What the L0 swimlane shows:

- **Per-pipe occupancy.** Busy time per pipe (`MTE2` / `MTE1` / `CUBE` /
  `FIXP` / `SCALAR`) for one kernel, so a memory-bound vs compute-bound
  diagnosis is direct (e.g. "MTE2 spans the whole window → GM-load
  bound").
- **Per-instruction issue overlap.** Each instruction is a slice on its
  pipe lane; the Perfetto sub-lane split makes concurrent issue on the
  same pipe legible.
- **Source-line attribution** (with `--debug-line`). Insight maps each
  instruction back to the kernel source line.
- **Cross-arch comparison.** The same kernel under `a2a3sim` vs `a5sim`
  surfaces real ISA differences (see [§6](#6-findings)).

What it does **not** show (use [L2 swimlane](l2-swimlane-profiling.md)):

- AICPU dispatch / finish latency, scheduler phases, dependency arrows.
- Multi-task placement across cores. L0 is one kernel, one core.

## 5. How It Works

L0 swimlane is **tooling-only** — there is no dedicated device-side data
path. It composes three existing pieces: tensor dump (for args),
`msprof op simulator` (for the pipeline trace), and the 5-file replay
recipe from the insight-trace skill (for the isolated kernel build).

### 5.1 The generated workspace (5 files)

| File | Role |
| ---- | ---- |
| `replay_kernel.cpp` | Mix-arch wrapper that `#include`s the target kernel. AIC-only kernels include the body only under `__DAV_CUBE__` (empty stub under `__DAV_VEC__`); AIV-only is the mirror image |
| `replay_launch.cpp` | `replay_entry<<<1, ...>>>` launcher (`HW_BLOCK_NUM = 1`, single task in isolation) |
| `replay_host.cpp` | Builds the 128-byte Tensor descriptors from the dump's real args + fills scalars, then launches. **Auto-generated; never hand-edited** |
| `CMakeLists.txt` | Single mix-arch `.so` (`--cce-aicore-arch=dav-cXXX`) |
| `run_collect.sh` | `msprof op simulator` collect (`--kernel-name=replay_entry`) + export |

### 5.2 Args reconstruction (the zero-guess part)

`reconstruct_task_args` filters the dump records on `func_id`, groups by
`task_id` (default: lowest), and **unions both dump stages** — inputs +
scalars come from `before_dispatch`, outputs from `after_completion` —
keyed by `arg_index`. Args need **not** start at slot 0 or be
contiguous: a kernel dispatched as a non-first MIX subtask reads its
tensors at an offset (e.g. `mixed_example`'s ADD reads `args[3..5]`, MUL
`args[6..8]`). The replay places each tensor at its **real** `args[]`
slot, while the descriptor array `d_tensors` is indexed 0-based — the two
are decoupled, so an offset kernel traces correctly. For each tensor it
emits the literal shape / strides / dtype / start_offset into
`make_desc`, with these correctness-critical details:

- **Descriptor field offsets** are pinned by the `static_assert`s in
  `src/{arch}/runtime/tensormap_and_ringbuffer/runtime/tensor.h`
  (`buffer.addr@0`, `buffer.size@8`, `start_offset@24`, `ndims@36`,
  `dtype@40`, `is_contiguous@42`, `shapes@44`, `strides@72`;
  `sizeof(Tensor) == 128`).
- **dtype** comes from a string→enum table mirroring
  `src/common/task_interface/data_type.h` (note `BFLOAT16 = 6`, not 5 —
  a wrong table corrupts every bf16 descriptor).
- **Buffer size uses the extent formula**
  `(start_offset + 1 + Σ(shape[i]-1)*stride[i]) * elem_size`, not
  `numel` — strided / offset views read past `numel` and would go
  out of bounds with a `numel` allocation.
- Replay data is **memset to 0**; only the descriptor metadata is real.
  This is why data-dependent branches / addresses can distort while pure
  pipeline structure stays faithful (see [§7](#7-fidelity-rules)).

### 5.3 Build & collect

The smoke build (no NPU) runs cmake + builds `replay_host`, then asserts
`replay_entry` and `launch_replay` are present in
`libreplay_kernel.so`. With `--no-collect` it stops here. Otherwise the
collect step runs `run_collect.sh` (the camodel needs a device context),
locates `insight_export/OPPROF_*/simulator/trace.json`, and writes the
two viewer copies. Device selection follows the lock already held:

- **Under an outer `task-submit`** (`$TASK_DEVICE` set — the onboard
  workflow where the whole tool is wrapped so the dump and collect share
  one lock): reuse `$TASK_DEVICE`, **no nested `task-submit`**.
- **Standalone** (no outer lock) with `task-submit` on `PATH`: self-lock
  via `task-submit --device auto`.
- **No `task-submit`** at all: unlocked run with a warning (per
  [running-onboard.md](../../.claude/rules/running-onboard.md)).

### 5.4 SPMD context synthesis (single-core and mix)

SPMD kernels read an execution context the orchestration builds per
dispatch — `LocalContext{block_idx, block_num}` at args slot 48 and
`GlobalContext{sub_block_id}` at slot 49 (`get_block_idx` /
`get_block_num` / `get_sub_block_id`). The isolated replay has no
orchestration, so it **synthesizes** that context host-side. None of its
inputs need a per-test marker; they are all derived:

- **is-mix** — a cooperative mix kernel is the *same* source compiled for
  both sub-cores, so it appears as an `(aic, aiv)` incore **pair sharing
  one source** (e.g. `paged_attention_parallel`). `load_kernel_meta`
  detects this. Everything else — including *independent* kernels packed
  into a mix dispatch (different sources, e.g. `mixed_example`) — takes
  the AIC/AIV-only path.
- **block_num / hw_block_dim** — the case's `block_dim` (the SPMD grid
  width), read from `CASES`. Override with `--spmd-block-num`.
- **aiv_lanes_per_block** — the arch's hardware subblockdim (2 for the
  1C2V a2a3/a5 clusters), from `ARCH_CONFIG`.

**AIC/AIV-only path** — `replay_host.cpp` *always* builds one
`LocalContext{block_idx=0, block_num}` + `GlobalContext{sub_block_id=0}`
and points slots 48/49 at them, then launches `<<<1>>>`. This is
harmless for positional kernels (they ignore 48/49) and required for
single-core SPMD kernels (e.g. `spmd_basic`, `spmd_multiblock_aiv`),
which would otherwise dereference a null context. `block_idx=0` traces a
representative block; `block_num = block_dim` keeps steady-state branches
(`block_idx+1 < block_num`) on their normal path — see
[§7](#7-fidelity-rules) for what `block_idx=0` / `block_num` do and don't
represent.

**Mix path** (auto-detected) differs in three places:

- **Wrapper** — `replay_kernel.cpp` includes the kernel under *both*
  `__DAV_CUBE__` and `__DAV_VEC__` (no empty stub), and `replay_entry`
  takes two args rows (`aic_args`, `aiv_args`), indexing `aic_args` by
  hardware block and `aiv_args` by AIV lane
  (`block * subblockdim + subblockid`).
- **Launch** — `<<<hw_block_dim>>>` (the case `block_dim`), not `<<<1>>>`.
- **Args** — the real tensors *and* the 3 FIFO rings come from the dump
  unchanged (the orchestration registers the rings as full-size
  `add_output` tensors). Only the per-row slot 48/49 context is
  synthesized, with `block_num = hw_block_dim` (the kernel uses it as the
  stride step) and `block_idx` / `sub_block_id` derived from the row.

Two dump-side requirements specific to **mix** (single-task kernels do
not need them):

- **One incore must declare the full tensor signature.** A mix task
  shares one `args[]` across cube + vec, but the tensor dump concatenates
  each active subtask's signature tensors and requires their sum to equal
  the payload tensor count — so the mix kernel must declare all its
  tensors on **one** incore's `signature` (and leave the other empty), or
  the dump skips the whole task. (The signature is consumed only by the
  dump; dispatch ignores it.) This is a standard `CALLABLE` field, not a
  tool-specific marker.
- **Reconstruction merges the whole func_id pair.** The dump assigns a
  per-subtask func_id, so `reconstruct_task_args` gathers records across
  *all* func_ids sharing the mix source and merges by `arg_index` — a
  single-func_id filter would capture only one subtask's slice.

Because the contexts cannot come from the dump (they are runtime
scaffolding, not payload), the SPMD paths are **not zero-guess** for the
context — but the constants are all derived (grid width from `block_dim`,
lanes from the arch), so no test markers are required. The mix path
currently assumes **1C2V** (the only binding both current chips support).

## 6. Findings

Measured behaviors worth knowing before you read a trace.

### 6.1 The a5 camodel is much slower than a2a3 (wall-clock)

Same QK kernel, `n_blocks = 4`:

| Metric | a2a3 (`dav_2201`) | a5 (`dav_3510`) |
| ------ | ----------------- | --------------- |
| camodel wall-clock | ~3.9 min | ~7 min |
| Total tick | 2,059,859 | 150,217 |
| Host cost per tick | ~0.11 ms/tick | **~2.8 ms/tick (~25×)** |
| End-to-end | ~4.4 min | ~8.3 min |

The camodel is a cycle-by-cycle, whole-chip (32-core), serial software
model. "Total tick" is **not** comparable across platforms (tick
granularity differs); wall-clock and the simulated µs are. a5 pays ~25×
per tick, so it is slower end-to-end even with fewer ticks. Much of the
cost is fixed setup independent of `n_blocks`, so shrinking `n_blocks`
helps only modestly. **Prefer a2a3 for logic validation; run a5 only
when you specifically need the a5 pipeline.**

### 6.2 `--set-arg` floor for a loop count (without distortion)

QK/PV are **double-buffered prefetch** kernels: an `if (i+1 < n_blocks)`
guards the prefetch + `pipe_barrier`.

| `n_blocks` | Captures | Distortion |
| ---------- | -------- | ---------- |
| 1 | No prefetch (`if` never runs) | **Distorted** — double-buffering entirely lost |
| 2 | Prefetch, single buffer phase | Slightly incomplete |
| 3 | Ping-pong both phases + tail block | Faithful (minimum) |
| 4 | Plus one steady-state block | Faithful, most stable |

→ **Floor 3, recommend 4; `n_blocks = 1` always distorts.** Shrinking
the loop count cuts iterations without changing per-block pipeline
structure; it does **not** change template branches (those are decided
by tile shape `shapes[0]`, which must stay real).

**Where the loop count lives — scalar vs tensor.** How you set
`n_blocks` depends on the kernel:

- **Single-task** (`aic_qk_matmul`, `aic_pv_matmul`): `n_blocks` is a
  **scalar** at `args[4]` → `--set-arg 4=4`.
- **Mix paged-attention**: `n_blocks` is **derived from the
  `context_lens` tensor** (`args[4]`), so `--set-arg` fills that buffer:
  `--set-arg 4=512` → every `context_lens` element = 512 →
  `n_blocks = ceil(512 / block_size)`. Dump on a golden-passing
  `context_len` (validates wiring), then shrink the camodel loop with the
  fill — the wiring is already proven and only timing structure is traced.
  `--set-arg` accepts a tensor slot only for **integer** dtypes.

### 6.3 a2a3 exports veccore lanes, a5 does not (camodel policy)

An AIC-only kernel on a2a3 exports `cubecore0 + veccore0/1`; on a5 it
exports **only `cubecore0`**. This is a camodel export policy difference
confirmed by: symmetric workspaces (only forced arch/SoC/macro/include
differ), a5 sim logs showing `block_end AIV` (the vec lanes *do*
execute), and disassembly showing the a5 stub compiles to real code
(200 B FUNC), not zero instructions. The a5 (`dav_3510`) camodel does
not export cores that ran only an empty stub with no real operator work;
a2a3 (`dav_2201`) exports even the stub. **Read `cubecore0` for AIC-only
and `veccore` for AIV-only; a2a3's veccore on an AIC kernel is empty-stub
noise.**

### 6.4 a2a3 vs a5 instructions / timing genuinely differ (real ISA)

Same QK, `n_blocks = 4`, `cubecore0`: a5 (`dav-c310`) uses newer, more
compact instructions — `LOAD_2Dv2`, explicit `DC_PRELOAD`, `CALL` — for
189 events vs a2a3's 326 (cube binary 228 B vs 1312 B). Per-pipe busy
time differs accordingly: this QK is **MTE2 (GM-load) bound on a5** (load
fills the span) while a2a3 spreads loads across MTE1 + MTE2; cube compute
itself is tiny on both. These are **real silicon ISA differences, not
tool bugs**. (Some markers — `CACHEMISS`, `BAR` counts — may mix real
behavior with the §6.3 export-policy effect and are not separated.)

### 6.5 Trace can truncate the last loop iteration(s) — self-check every run

A known **collection-side bug** in CANN's msprof/camodel (closed-source,
not locatable from outside): the exported instruction stream sometimes
**ends early**, dropping the last one or few loop iterations' compute /
write-back. Symptom: `MMAD` / `FIX_L0C_TO_DST` counts come out **less
than `n_blocks`** while the loads (`LOAD_2Dv2` / `MOV_OUT_TO_L1`) are
complete — the trace's last work event is a final-iteration load with no
trailing `MMAD` / `FIX`.

Observed (a5sim, QK): `n_blocks = 4` reproduced 3 `MMAD` / 3 `FIX`
(missing one block); `n_blocks = 6` came out complete (6 / 6). It is
**not** a fixed `n-1` rule. The sim itself runs all blocks
(`block_end AIC`, `All task success`, no truncate messages); only the
exported event stream is cut at the tail.

**Self-check / workaround:** after each run, verify
`MMAD == n_blocks` **and** `FIX_L0C_TO_DST == n_blocks`. If they
disagree the tail was truncated — do **not** draw timing conclusions from
that trace. Retry with a different `n_blocks` (6 was complete), or re-run
the same config until one comes out whole. This is distinct from the
§6.3 "a5 never exports certain sync events" — that is a stable
*instruction class* omission; this is a *tail-iteration* cut.

## 7. Fidelity Rules

What distorts the trace, and what is safe:

| Knob | Change it? | Distorts? |
| ---- | ---------- | --------- |
| Tile M/K/N (`q_tile` / `head_dim` / `block_size`) | **No** | Changing it distorts — alters cycle counts and switches template branches |
| Scalar values (`scale` / offsets / `is_first` …) | Use real dump values | Wrong value → wrong branch → distorted |
| Loop count (`n_blocks`, via `--set-arg` — scalar slot, or fill the `context_lens` tensor) | Shrinkable to ≥ 3–4 | Faithful at ≥ 3–4; `= 1` distorts (see §6.2) |
| Data filled to 0 / `block_table = 0` | Default (memset 0) | Cache / address / data-dependent branches distort; pure pipeline structure is fine |
| SPMD `block_idx` (slot 48) | Fixed 0 | Traces a real block 0 — representative for uniform SPMD; an edge/special-cased block 0 would show *its* path, not steady state |
| SPMD `block_num` (slot 48) | Default `block_dim`; `--spmd-block-num` | Any value ≥ 2 keeps steady-state branches (`block_idx+1 < block_num`); only `block_num`-proportional work needs the exact value |
| SPMD `sub_block_id` (slot 49) | Fixed 0 | Traces AIV0 (left lane); faithful when the two vec lanes are symmetric |
| Cross-platform (a2a3 vs a5) | Set by target | Instructions / timing genuinely differ (real silicon, not a bug — see §6.4) |

## 8. Limitations

- **AICPU orchestration is out of scope.** L0 sees only the AICore
  pipeline of one kernel. For dispatch / finish / scheduler / dependency
  data use [L2 swimlane](l2-swimlane-profiling.md).
- **Simulation clock, not silicon.** The camodel's absolute timing is a
  model, not measured hardware. Use it for *relative* per-pipe / per-arch
  structure, not for absolute-latency claims.
- **Replay data is zero.** Only descriptor metadata is real; data-driven
  control flow can diverge (see §7).
- **Tail-truncation collection bug.** Validate `MMAD`/`FIX` counts every
  run (see §6.5).
- **Scope.** AIC-only / AIV-only single-task kernels — including
  SPMD single-core kernels (slot 48/49 context synthesized) and kernels
  reading args at an offset / non-contiguously (e.g. `mixed_example`) —
  are automatic. **SPMD mix** kernels are auto-detected (an `(aic, aiv)`
  incore pair sharing one source) and need only a full-signature incore
  for the dump; the grid width and lane count are derived. The mix path
  assumes **1C2V** (the only binding both chips support). See
  [§5.4](#54-spmd-context-synthesis-single-core-and-mix). What
  `block_idx=0` / `block_num` faithfully represent (and don't) is in
  [§7](#7-fidelity-rules).

## 9. FAQ / Debug Guide

**`func_id=N not found`.** The tool prints the available
`(func_id, name, core_type)` from the test's `CALLABLE.incores`. Pick one
of those.

**No dump records for the kernel.** The incore `signature` tensor count
likely disagrees with orchestration's real `add_input`/`add_output`
count, so the dump skipped it — see [§3.1](#31-prerequisites-one-time-per-test-case)
and [tensor-dump.md](tensor-dump.md).

**Smoke build fails on a missing symbol.** `replay_entry` /
`launch_replay` must appear in `libreplay_kernel.so`. Check the kernel
source compiles standalone under the arch's prologue; a wrong
`--platform` picks the wrong `ARCH_CONFIG`.

**`ASCEND_HOME_PATH is not set`.** Source CANN's `set_env.sh` first; the
tool requires it for both the smoke build and the collect step.

**Perfetto shows overlapping / missing slices.** You opened the raw
Insight `trace.json`. Open `<label>_trace_perfetto.json` instead
(see [§3.4](#34-viewing--insight-vs-perfetto)).

**Instructions don't map to source lines in Insight.** Re-run with
`--debug-line` / `-g` so the kernel carries `debug_line`.

**`MMAD` / `FIX` count < `n_blocks`.** The export truncated the tail —
see [§6.5](#65-trace-can-truncate-the-last-loop-iterations--self-check-every-run).
Re-run or change `n_blocks`; do not trust the trace's timing.

**The a5 run is taking forever.** Expected — the a5 camodel is ~25× per
tick vs a2a3 (see [§6.1](#61-the-a5-camodel-is-much-slower-than-a2a3-wall-clock)).
Prefer a2a3 unless you specifically need the a5 pipeline; shrink
`n_blocks` with `--set-arg` for a modest speedup.

## 10. Related docs

- [l2-swimlane-profiling.md](l2-swimlane-profiling.md) — the
  per-task / scheduler swimlane one level up; L0 zooms into a single L2
  task block.
- [tensor-dump.md](tensor-dump.md) — the `func_id`-tagged per-task arg
  capture L0 reconstructs from.
- [`.claude/skills/insight-trace/SKILL.md`](../../.claude/skills/insight-trace/SKILL.md)
  — the manual 5-file `msprof op simulator` replay recipe this tool
  automates, plus the Perfetto conversion notes.
- [chip-level-arch.md](../chip-level-arch.md) — the AICore pipe model
  (MTE2 / MTE1 / CUBE / FIXP / SCALAR) the lanes represent.
