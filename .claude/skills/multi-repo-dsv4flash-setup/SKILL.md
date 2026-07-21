---
name: multi-repo-dsv4flash-setup
description: Concrete guide to benchmark DeepSeek-V4 (dsv4) FLASH on NPU against the current worktree's simpler — the three decode attentions (swa/csa/hca) and the distributed MoE, all in pypto-lib's models/deepseek/v4/. Defers cross-repo clone/install to multi-repo-setup, then gives the exact PYPTO_BENCH run commands and the hard-won config that makes the numbers reproducible: pin ptoas 0.48 (0.50/ptoas-bin cannot compile csa), pin --start-pos 8192, edit moe.py for 16 experts/card, read fast_effective_us not effective_us, pin an even die, and apply the eager-init idempotency guard so distributed MoE runs on newest simpler. Invoke when benchmarking dsv4 flash attention or MoE, reproducing a dsv4 effective-time number, or hitting "add_worker after init" / a csa compile timeout / wandering attention numbers.
---

# DeepSeek-V4 FLASH on NPU: attention + MoE effective-time benchmark

This skill is the **dsv4-flash-specific concrete guide**. It benchmarks the
four kernels that live in **pypto-lib** at `models/deepseek/v4/` against the
current worktree's simpler:

| kernel | file | measures | dies |
| ------ | ---- | -------- | ---- |
| SWA attention | `decode_attention_swa.py` | decode attn effective_us | 1 |
| CSA attention | `decode_attention_csa.py` | decode attn effective_us | 1 |
| HCA attention | `decode_attention_hca.py` | decode attn effective_us | 1 |
| MoE (distributed) | `moe.py` | decode MoE fast_effective_us | ≥2 (EP) |

It sits on top of [`multi-repo-setup`](../multi-repo-setup/SKILL.md), which
owns the generic repo-graph + clone + install steps this skill does not
repeat. Everything below is the dsv4-specific run recipe and the five
configuration facts that make the effective-time numbers **reproducible to
~1%** instead of wandering ±18% run-to-run.

## 1. Setup — defer to multi-repo-setup

Run [`multi-repo-setup`](../multi-repo-setup/SKILL.md) first — it clones
pto-isa / pypto / pypto-lib under `build/`, exports the toolchain env, and
installs the simpler you want (worktree or main). Then, from the worktree:

```bash
source .venv/bin/activate
eval "$(pypto-setup --export)"            # ASCEND_HOME_PATH, gcc, PATH
```

Verify the loaded simpler is the worktree's, not a user-site shadow:
`python -c "import simpler; print(simpler.__file__)"`.

## 2. The benchmark env: PYPTO_BENCH and its two metrics

`PYPTO_BENCH=1` (a pypto-lib golden/runner gate) times each kernel over 100
rounds (5 warmup) and prints:

- **`effective_us`** — the on-device effective time. For the **single-card
  attentions** this is *the* number to read (min/median/mean/max over 100
  rounds).
- **`fast_effective_us`** — for **distributed MoE only**: each round
  contributes the fastest valid rank's Effective time. This is the **clean,
  stable** MoE metric (valid_rounds should be 100/100). The MoE
  `effective_us` (all-rank) carries a long contention tail — median wobbles
  and max can spike 20× — so **quote `fast_effective_us` for MoE**, never the
  all-rank `effective_us`.

## 3. Five configuration facts — get these right or the numbers lie

These were each established empirically; skipping any one makes a run
non-comparable.

### 3.1 ptoas 0.48 — csa cannot compile on 0.50 / default ptoas-bin

`decode_attention_csa`'s `csa_slots_build_valid_qk_plan` does **not** compile
within the 60 s ptoas timeout on **ptoas 0.50** or the default
`/usr/local/bin/ptoas-bin`. Versions **0.45–0.48 compile it in ~4 s**.
Pin 0.48 explicitly — and note `pypto-setup --export` sets PATH to the slow
default, so pinning must come **after** the export:

```bash
eval "$(pypto-setup --export)"
export PTOAS_ROOT=/usr/local/ptoas/0.48
export PATH="/usr/local/ptoas/0.48/bin:$PATH"   # must win over pypto-setup's default
```

(`task-submit --ptoas 0.48` also selects it, but `pypto-setup --export` inside
the `--run` body overrides that — so set `PTOAS_ROOT`/`PATH` in the driver.)
swa and hca compile on any version; only csa is version-sensitive.

### 3.2 pto-isa pinned commit

Keep pto-isa at the commit the worktree expects
(`SIMPLER_PTO_ISA_COMMIT`, e.g. `83d01313`) with `PTO_ISA_ROOT` pointing at
the clone. A different pto-isa changes tile codegen and the numbers.

### 3.3 Attention: pin `--start-pos 8192`

The attention scripts take `--start-pos` (default `None` → a *mixed*
canonical fixture, `swa_decode_start_set`: sliding-window regimes + 8k). The
default is **not** a fixed workload — HCA in particular swings (~353 µs on the
mixed default vs ~270 µs at uniform 8192) because the KV length changes. For a
comparable long-context number, force **`--start-pos 8192`** on all three.
SWA/CSA move less but pin it anyway for a single canonical config.

### 3.4 MoE: experts-per-card via the `moe.py` divisor

`moe.py:36` computes global experts from a divisor; **per-card experts
(`N_LOCAL`) = 256 // divisor**, independent of `--ep`:

```python
# base config.FLASH.n_routed_experts == 256
config.FLASH = dataclasses.replace(config.FLASH,
    n_routed_experts=config.FLASH.n_routed_experts // 16 * EP)   # // 8 → 32/card, // 16 → 16/card
```

| divisor | per-card experts (`N_LOCAL`) |
| ------- | ---------------------------- |
| `// 8` (repo default = EP8/32-per-card) | 32 |
| `// 16` | **16** |

For **16 experts/card**, edit `// 8` → `// 16`, then run `--ep 2` (EP2, 2
dies). `--ep {2,4,8}` picks the EP world size; `N_LOCAL` stays 16 regardless
(EP8 → 128 global / 8 cards). This is a benchmark-only edit — do not commit it.

### 3.5 Eager-init adaptation — else MoE dies with `add_worker after init`

Newest simpler (eager-init, upstream #1397) makes `Worker.init()` eagerly run
the hierarchical start; the C++ `add_worker` then throws
`Worker: add_worker after init` when pypto's `DistributedWorker.__init__`
calls `_start_hierarchical()` a second time to force an eager pre-fork
(pypto's own comment calls that call "idempotent"). Newest simpler broke that
idempotency. **Restore it** with a guard at the top of
`_start_hierarchical` in `python/simpler/worker.py`:

```python
# Idempotent once init() has driven the one hierarchical start
# (state == "started"): a repeat call is a no-op — the C++ Worker is
# already inited and re-running add_worker would throw
# "add_worker after init". A distributed runner may call this again to
# force an eager pre-fork; that fork already happened inside init().
if self._hierarchical_start_state == "started" or getattr(self, "_hierarchical_started", False):
    return
```

After editing simpler python, reinstall editable
(`pip install --no-build-isolation -e .`) so the change is picked up. This is
a genuine eager-init ↔ distributed compatibility fix, not a downgrade — it
belongs upstream; land it rather than carrying it locally forever.

## 4. Run it — onboard, pinned, reproducible

Onboard rules apply: hold an exclusive die via `task-submit` and gate on
[`onboard-arch-precheck`](../onboard-arch-precheck/SKILL.md). **Pin an even
die** — `--device auto` picks a different die each run and neighbor HBM
contention on a shared package swings memory-bound decode kernels ±18%.
Pinning die + start_pos + ptoas reproduces to ~1%.

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
SKILL=.claude/skills/multi-repo-dsv4flash-setup

# Three attentions on one pinned even die (start_pos=8192, ptoas 0.48):
task-submit --timeout 1200 --device 0 --run "$SKILL/bench_attn.sh $PWD 0"

# MoE 16-experts/card, EP2, on two pinned even dies (edit moe.py //16 first):
task-submit --timeout 900 --device 2,4 --run "$SKILL/bench_moe.sh $PWD 2,4"
```

Both drivers set `PTOAS_ROOT=0.48`, `PYPTO_BENCH=1`, and the pto-isa pin, then
run the kernels and grep the effective lines. Read `bench_attn.sh` /
`bench_moe.sh` for the exact env; pass the simpler-worktree root as `$1` and
the device(s) as `$2`.

## 5. Canonical baseline (a2a3, ptoas 0.48, start_pos=8192, PYPTO_BENCH)

Quiet single pinned die; expect ~1% run-to-run at this config:

| kernel | metric | µs (median) |
| ------ | ------ | ----------: |
| decode_attention_swa | effective_us | ~246 |
| decode_attention_csa | effective_us | ~383 |
| decode_attention_hca | effective_us | ~270 |
| MoE (16/card, EP2) | fast_effective_us | ~465–490 |

If a run lands well outside these, check the five facts in §3 **before**
suspecting a regression — a mismatched die/ptoas/start_pos explains almost
every discrepancy.

## Anti-patterns

- ❌ Reading MoE `effective_us` (all-rank) as the headline — it has a
  contention tail (median wanders, max spikes 20×). Use `fast_effective_us`.
- ❌ Comparing attention numbers across runs on different dies via
  `--device auto` — ±18% die/contention noise. Pin an even die.
- ❌ Comparing attention numbers without fixing `--start-pos` — the default
  fixture is a mixed set; HCA alone moves ~30% between the default and 8192.
- ❌ Letting `pypto-setup --export` leave ptoas at the default `ptoas-bin` —
  csa then times out at 60 s and looks like a codegen bug. Pin 0.48 after the
  export.
- ❌ Reading `add_worker after init` on MoE as a simpler bug — it's the
  eager-init ↔ distributed idempotency gap (§3.5). Apply the guard.
- ❌ Committing the `moe.py` `// 16` edit — it's a benchmark-only knob.
