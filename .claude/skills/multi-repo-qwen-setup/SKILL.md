---
name: multi-repo-qwen-setup
description: Concrete qwen3-14B guide on NPU against the current worktree's simpler. Points at simpler's own in-repo qwen3_14b_decode example (zero cross-repo), then covers the two cross-repo paths — the pypto-serving runner (prefill/decode TPOT) and the pypto-lib decode_layer kernel (scheduler-overhead DFX). Defers cross-repo clone/install to multi-repo-setup, then gives the exact run commands (batch-1 and batch-16), the prefill ring/timeout gotchas that otherwise surface as 507018, how to read the per-token timing layers, and the kernel-level overhead-analysis flow. Invoke when running qwen3 on simpler, measuring decode/prefill TPOT, reproducing a qwen3 perf number, running qwen3 decode_layer overhead analysis, or chasing a 507018 on the qwen3 path.
---

# Qwen3-14B on NPU: setup + TPOT / overhead measurement

This skill is the **qwen3-specific concrete guide**. It first points at
simpler's own in-repo qwen3 example (the zero-cross-repo path), then covers
the two cross-repo paths against the worktree's simpler: the **pypto-serving
runner** (prefill/decode TPOT) and the **pypto-lib `decode_layer` kernel**
(scheduler-overhead DFX). For the cross-repo paths it gives the exact run
commands, the prefill ring/timeout gotchas, and how to read per-token timing
— distilled from real runs plus each upstream repo's own docs — sitting on
top of [`multi-repo-setup`](../multi-repo-setup/SKILL.md), which owns the
generic repo-graph + clone + install steps this skill does not repeat.

## Three ways to run qwen3 — pick by what you're measuring

There are **three distinct qwen3 entry points with different invocation
styles**. They measure different things; don't confuse them:

| aspect | Path 0 (in-repo) | Path A (serving) | Path B (decode_layer) |
| ------ | ---------------- | ---------------- | --------------------- |
| repo | simpler (this) | pypto-serving | pypto-lib |
| entry | qwen3_14b_decode | npu_generate.py | decode_layer.py |
| cross-repo? | no | yes | yes |
| measures | 2-layer decode correctness/timing | end-to-end TPOT | layer sched overhead |
| see | this section | sections 2-5 | section 6 |

- **Path 0 — in-repo example (simpler itself, NO cross-repo).** Start here
  if you just want qwen3 decode running on simpler. simpler ships a
  self-contained SceneTestCase at
  `examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/` — a fused chunk
  of **two** Qwen3-14B decode layers (harvested pypto codegen: 8 AIC + 27
  AIV + orchestration, golden in `simpler_setup/goldens/qwen3_14b_decode.py`).
  No pypto / pypto-lib / JIT descent — it builds and runs like any simpler
  example. Onboard rules still apply (per-die lock + `onboard-arch-precheck`):

  ```bash
  .claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
  task-submit --device auto --device-num 1 --run "\
    python -m pytest examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode \
      --platform a2a3 --device \$TASK_DEVICE"
  # standalone: python .../qwen3_14b_decode/test_qwen3_14b_decode.py -p a2a3 -d \$TASK_DEVICE
  ```

  This is **not** a cross-repo path, so the rest of this skill (§1 setup,
  §2–§6) does not apply to it — run it like any other example/scene test
  (see the example's own `README.md`). It's listed here only so you know it
  exists before reaching for the heavier cross-repo paths below.
- **Path A — serving runner (pypto-serving).** Full engine: builds an
  `Engine` and runs `generate` over a prompt. Entry
  `examples/model/qwen3_14b/npu_generate.py`, flags `--model-dir --prompt
  --max-seq-len --max-new-tokens --num-layers-override --profile[-verbose]`.
  Measures **end-to-end prefill/decode TPOT** for the whole model; read via
  the `--profile` report + `device_log_timing` (§2–§5). Use it to reproduce
  a serving perf number or chase a serving `507018`.
- **Path B — `decode_layer` kernel (pypto-lib).** A single JIT case for one
  decode layer, not the engine. Entry
  `models/qwen3/14b/decode_layer.py`, flags `-p <platform> -d <device>`,
  run in two rounds (`--no-dep-gen --enable-l2-swimlane`). Measures **one
  layer's scheduler overhead** (Orch/Sched breakdown); read via
  `sched_overhead_analysis` / `swimlane_converter` (§6). Use it to dig into
  per-layer scheduling cost / the overhead model.

Path 0 needs no setup beyond simpler itself. **The cross-repo paths A and B**
(everything from §1 on) share only the §1 setup and the onboard rules — their
run commands, flags, and analysis tools are otherwise entirely different.

For Path A, `pypto-serving`'s `cpu_generate.py` is the torch reference, and
`npu_generate.py` is single-prompt (**batch-1**); there is no `npu_stress.py`
upstream — the batch-16 + `--prefill-on-cpu` harness lives in a private
`opt-qwen3` fork (§3 shows the batch-N driver).

## 1. Setup (cross-repo paths A & B only) — defer to multi-repo-setup

Run [`multi-repo-setup`](../multi-repo-setup/SKILL.md) first — it clones
pto-isa / pypto / pypto-lib / **pypto-serving** under `build/`, exports the
toolchain env, and installs the simpler you want (worktree or main). Its
Step 5 hands off to each external repo's own skills/docs; read those for
anything this skill doesn't cover:

- **Path A** — `build/pypto-serving/.claude/skills/` (if present) and its
  `README.md` (engine config, model-dir layout, weights conversion).
- **Path B** — `build/pypto-lib/.claude/skills/` / `README.md` and the
  `decode_layer.py` docstring / `--help` (case flags, golden data).

Then, from the worktree, the qwen runs assume this env:

```bash
source .venv/bin/activate
eval "$(pypto-setup --export)"            # PTOAS_ROOT, ASCEND_HOME_PATH, gcc-15, PATH
export PTO_ISA_ROOT="$PWD/build/pto-isa"  # needed to build the simpler runtime
PY="$PWD/.venv/bin/python"
```

Verify the loaded simpler is the worktree's (not a user-site `.pth` shadow):
`python -c "import simpler; print(simpler.__file__)"`.

## 2. Path A — serving runner: decode (batch-1)

*Path A (§2–§5) drives the **pypto-serving** `npu_generate.py` runner.*

Onboard work MUST hold an exclusive die — wrap in `task-submit` and pass the
[`onboard-arch-precheck`](../onboard-arch-precheck/SKILL.md) gate. Contention
inflates every timing number, so a clean exclusive run is mandatory for perf.

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
task-submit --timeout 2400 --max-time 2400 --device auto --device-num 1 --run "\
  SIMPLER_CHIP_TIMING=1 $PY build/pypto-serving/examples/model/qwen3_14b/npu_generate.py \
    --model-dir <QWEN3_14B_DIR> --prompt 'Huawei is' \
    --platform a2a3 --device-id \$TASK_DEVICE \
    --max-seq-len 4096 --max-new-tokens 32 \
    --num-layers-override 40 --profile-verbose"
```

- `--num-layers-override 40` = full model; `--profile[-verbose]` prints the
  timing report (`api.run_decode` per-step = end-to-end TPOT, `kernel.decode_layer`
  host wall, per-step layer breakdown).
- Decode dispatches through the **L3 `DistributedWorker`**, which forks a
  **chip-child (L2)** that runs `run_prepared`. The per-step timing lives at
  **L2**: `run_prepared` computes both host (`host_wall`) and device
  (`device_wall` = the fused-decode orchestrator wall, ~33 ms — nonzero) plus
  the device orch/sched markers. The L3 parent's `Worker.run` returns
  `RunTiming(python_wall, 0)` — its `device_wall=0` is **expected** (L3 is a
  dispatcher with no device wall of its own) and is **not** the source you
  read. Read L2 instead (next section).

## 3. Run decode (batch-16)

`npu_generate.py` is single-prompt. For batch-N, build the same engine but call
`engine.generate_batch(model_id, [prompt]*N, config)` — a ~30-line driver that
reuses `npu_generate`'s `_TimingCollector` / `InstallProfiling` /
`InstallThroughputMeter` and swaps the final `generate_result` for
`generate_batch`. The engine is already configured `max_batch_size=16`.

## 4. Prefill on NPU — the two gotchas (else 507018)

Batch-16 256-token × 40-layer prefill on NPU is heavy. Two independent
failures both surface as the generic `507018`/`507046` — **read the device
log to tell them apart**:

1. **Heap-exhausted deadlock** (default 256 MB ring): the open prefill scope's
   live-set exceeds one ring → device log prints
   `FATAL: Task Allocator Deadlock - Heap Exhausted! ... Provable head-of-line`.
   Fix: raise the ring. batch-1 needs just over 256 MB; batch-16 needs the
   prefill config `PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072
   PTO2_RING_DEP_POOL=131072` (heap, task-window, and dep-pool all scale with
   batch; default `task_window=16384` also overflows at batch-16).
2. **Op-execute timeout** (slow/contended prefill): device log prints
   `HandleTaskTimeout ... timeOut:<N>` and kills `aicpu-sd`. The compiled
   defaults (`PLATFORM_OP_EXECUTE_TIMEOUT_US`=45 s,
   `PLATFORM_STREAM_SYNC_TIMEOUT_MS`=50 s — raised in #1175) clear a ~16.7 s
   clean batch-16 prefill with margin, so a clean exclusive run does **not**
   need an override. You only hit this on a **contended** box, where prefill
   stretches past 45 s. In that case **raise the timeouts at runtime via the
   env overrides — no rebuild, no header edit:**

   ```bash
   # add to the task-submit --run, before the python call:
   export PTO2_OP_EXECUTE_TIMEOUT_US=120000000  # 120 s
   export PTO2_STREAM_SYNC_TIMEOUT_MS=130000    # 130 s (must exceed op-execute)
   ```

   The runtime reads these and overrides the compiled defaults
   (`resolve_runtime_timeout_config`, wrapped by `resolve_onboard_timeout_config`).
   The override is **only accepted if the ordering holds**:
   `scheduler (10 s) < op_execute < stream_sync`, and `stream_sync` must clear
   the scheduler-arming guard (scheduler + 1.5 s) — otherwise the **whole
   override set is dropped** and the 45 s / 50 s defaults stand. `120 s / 130 s`
   satisfies all three. Pair with a long `task-submit --timeout`. A clean
   exclusive die runs batch-16 256-prefill in ~16.7 s; a contended box can
   blow past any timeout — prefer holding the die exclusively over chasing
   ever-larger timeouts.

(The private fork sidesteps both with `--prefill-on-cpu` — CPU prefill, NPU
decode only — which `npu_generate.py` does not have.)

## 5. Read the per-token timing layers

Redirect the CANN device log out of the shared default, then parse it:

```bash
# add to the task-submit --run, before the python call:
export ASCEND_PROCESS_LOG_PATH="$PWD/build/pypto-serving/build_output/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
# after the run (local, no device):
$PY -m simpler_setup.tools.device_log_timing \
    --device-log "$ASCEND_PROCESS_LOG_PATH/device-*/device-*.log"
```

`device_log_timing` reports per-round **Total / Orch / Sched** from the
`PTO2_PROFILING` markers (on by default, no swimlane needed). Round 0 is the
prefill; the rest are decode steps. **Total ≈ on-device kernel makespan** =
the "kernel run time" layer.

For layers ②③④ — the host↔device and bind/validate spans — parse the
`[STRACE]` host-trace markers with `strace_timing.py` (landed in
[simpler #1177](https://github.com/hw-native-sys/simpler/pull/1177)). The
markers are emitted at `LOG_INFO_V9` under `PTO2_PROFILING` (no new flag),
so the same log captured above carries them:

```bash
# TPOT table (per-callable; decode = most-invoked hid bucket):
$PY -m simpler_setup.tools.strace_timing \
    "$ASCEND_PROCESS_LOG_PATH/device-*/device-*.log"
# also emit a Perfetto/Chrome-trace JSON (L3 parent + each L2 child = a lane):
$PY -m simpler_setup.tools.strace_timing \
    "$ASCEND_PROCESS_LOG_PATH/device-*/device-*.log" --trace-out strace.json
```

The full per-token decomposition (what each layer means / how to get it):

| layer | = | source |
| ----- | - | ------ |
| ① kernel run time | makespan | `device_log_timing` Total |
| ② device init/finalize | `device_wall − makespan` | `strace_timing` (`[STRACE]` markers) |
| ③ host↔device handshake | `runner_run − device_wall` | `strace_timing` (`runner_run` span) |
| ④ attach+bind+validate | `host_wall − runner_run` | `strace_timing` (`bind`/`validate` spans) |
| ⑤ python/executor wrap | `end-to-end − host_wall` | `npu_generate --profile` |

②③④ come from the **L2 chip-child** `run_prepared`'s `host_wall` /
`runner_run` / `device_wall`. `host_wall` and `device_wall` are already
computed there (and `device_wall` is nonzero, ~33 ms — it is the L2 orchestrator
wall, *not* the L3 parent's `RunTiming.device_wall=0`); the L3 parent drops
the child's `RunTiming`, so they never reach a return value.

Simpler surfaces them instead as **`[STRACE]` host-trace markers** — one
line per `run_prepared` stage
(`bind`/`bind.args`/`bind.prebuilt`/`runner_run`/`validate`) plus the AICPU
device-phase subdivision (`preamble`/`so_load`/`graph_build`/`post_orch`).
`strace_timing.py` groups by `(pid, inv)`, rebuilds each invocation's span
tree from `depth`, and prints per-callable means; see
[docs/dfx/host-trace.md](../../../docs/dfx/host-trace.md) for the marker
grammar and span tree. The L3-side tensor-management ask is
[pypto-serving #44](https://github.com/hw-native-sys/pypto-serving/issues/44).

## 6. Path B — `decode_layer` kernel: scheduler-overhead DFX

*Path B is a different repo (**pypto-lib**) and a different entry point
than §2–§5 — a single JIT kernel case, not the serving engine.*

Path A above measures end-to-end TPOT through the serving runner. Path B
instead digs into the **scheduler overhead** of the qwen3 decode kernel
itself: run `build/pypto-lib/models/qwen3/14b/decode_layer.py` in two
rounds and feed both into simpler's overhead tools. Onboard rules apply
(per-die lock + `onboard-arch-precheck`).

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
cd "$PWD/build/pypto-lib/models/qwen3/14b"
# Round 1 — dep_gen (topology); Round 2 — swimlane (clean timing). NEVER co-run:
# dep_gen perturbs the timing the overhead analysis reads.
task-submit --device auto --device-num 1 --run "python decode_layer.py -p a2a3 -d \$TASK_DEVICE"
task-submit --device auto --device-num 1 --run "python decode_layer.py -p a2a3 -d \$TASK_DEVICE --no-dep-gen --enable-l2-swimlane"
```

Both write to `build_output/_jit_*/dfx_outputs/` (`deps.json` from round 1,
`l2_swimlane_records.json` from round 2). Analyze from the simpler worktree
(its `simpler_setup/tools` wins by cwd precedence):

```bash
# ROUND1_DIR / ROUND2_DIR are the two build_output/_jit_*/dfx_outputs/ dirs above.
$PY -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json "ROUND2_DIR/l2_swimlane_records.json" \
    --deps-json "ROUND1_DIR/deps.json"
# Visual: add the Overhead Analysis track to the Perfetto trace
$PY -m simpler_setup.tools.swimlane_converter "ROUND2_DIR/l2_swimlane_records.json" \
    --deps-json "ROUND1_DIR/deps.json" --overhead -o swimlane.json
```

See [docs/dfx/sched-overhead-model.md](../../../docs/dfx/sched-overhead-model.md)
for what the report and the overhead tracks mean.

## 7. When you change simpler C++ (instrumentation)

Timeouts no longer need this — use the §4 env overrides. But for *code*
changes (new instrumentation, log lines), `pip install` may skip the runtime
`.so` rebuild. Rebuild + sync explicitly (same `find`-based copy as
[`multi-repo-setup`](../multi-repo-setup/SKILL.md) Step 3 — arch-parametrized
and python-version-agnostic, so it works on a2a3 / a5 and any venv layout):

```bash
ARCH=a2a3   # or a5
BD="build/cache/$ARCH/onboard/tensormap_and_ringbuffer/host"
cmake --build "$BD" -j"$(nproc)"
SO="$BD/libhost_runtime.so"
# sync to every load location (build/lib AND the installed venv), no hardcoded
# python version — the glob matches whatever site-packages path exists:
for d in $(find .venv build/lib -path "*onboard*tensormap_and_ringbuffer*$(basename "$SO")"); do
  cp -f "$SO" "$d"
done
strings "$SO" | grep -m1 '<your-new-log-string>'   # confirm it baked in
```

(`.venv/lib64` is a symlink to `lib`, so the `find` covers both.) Host-side
instrumentation lives in the **host** lib; only that target needs rebuilding.

## Anti-patterns

- ❌ Comparing TPOT across runs on a contended box — contention inflates
  prefill (50 s+ vs 16.7 s) and decode spikes. Hold the die exclusively.
- ❌ Reading `507018` as "simpler bug" without the device log — it masks a
  heap deadlock, an op-timeout, and a forward-progress stall.
- ❌ Editing `platform_config.h` to raise timeouts — use the `PTO2_OP_EXECUTE_TIMEOUT_US`
  / `PTO2_STREAM_SYNC_TIMEOUT_MS` env overrides (§4) instead; no rebuild, and it
  doesn't change the default for everyone else.
- ❌ Setting the timeout env but breaking the ordering (`stream_sync` ≤ `op_execute`,
  or `op_execute` ≤ scheduler 10 s, or `stream_sync` not clearing scheduler + 1.5 s
  guard) — the **whole override set is dropped** and you debug the unchanged 45 s /
  50 s defaults.
