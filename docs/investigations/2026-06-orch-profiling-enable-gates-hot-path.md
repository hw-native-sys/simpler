# Gating the two residual profiling `enable()` calls left on the orch/scheduler hot path

**Date**: 2026-06-25
**Verdict**: applied (gated under the **existing** `SIMPLER_DFX`); magnitude
deliberately **not** measured; a new `PTO_PROFILING` macro was rejected.

## Question

Profiling must not slow the orchestration/scheduler flow it instruments. The
runtime off switch (`is_*_enabled()` returning `false`) is not free: even when
off, each gate is a live `if (xxx_enable())` on the hot path. Proposal: let
production inference and perf-measurement runs shed that residual cost.

Why the residual cost is non-zero: each gate is an O(1) global-bool read
(`return g_enable_pmu;`), but `is_pmu_enabled()`, `is_dep_gen_enabled()`, … are
`extern "C"` / weak symbols defined in a **different translation unit** from the
scheduler loop. The compiler **cannot inline or hoist them**, so each "off"
check is still a real `bl`/`ret` — once per scheduler iteration, once per task
submit.

This is already acknowledged in-tree: a2a3 `scheduler_dispatch.cpp` caches the
PMU gate at function scope with the comment *"is_pmu_enabled() is extern "C" and
the compiler cannot hoist it across the dispatch loop on its own"* — the team
already treats the call cost as real enough to hand-hoist once.

The answer is to move the toggle from run time to **compile time** via the
existing `SIMPLER_DFX` macro, giving two regimes from one switch:

- **Default build (`SIMPLER_DFX=1`)** keeps the full runtime toggle —
  profiling stays available and can be turned on/off dynamically, behavior
  unchanged.
- **Perf-measurement and production inference builds (`SIMPLER_DFX=0`)**
  compile the gates *out entirely* — no branch, no call, no residual `if` — so
  the hot path runs at its best, paying nothing for an observation feature it
  isn't using.

The original report was that these gates make orchestration "much slower" and
proposed wrapping them in a compile macro so production inference can drop them.

## What was tried

Swept every `is_*_enabled()` call site across both arches
(`is_pmu_enabled`, `is_dep_gen_enabled`, `is_dump_args_enabled`,
`is_scope_stats_enabled`, `is_l2_swimlane_enabled`) and classified each as
gated/ungated by `SIMPLER_DFX`. Findings:

- The existing `SIMPLER_DFX` compile macro
  (`src/common/task_interface/profiling_config.h`, default `1`) already wraps
  the **vast majority** of gates — `is_dump_args_enabled`,
  `is_scope_stats_enabled`, and most `is_pmu_enabled` sites compile out at
  `SIMPLER_DFX=0`.
- Exactly **two** hot-path sites per arch were ungated:
  1. `is_pmu_enabled()` in `scheduler_dispatch.cpp` — a2a3 already cached it
     once at function scope; a5 re-called it **per scheduler main-loop
     iteration**. The value is loop-invariant (PMU is latched once at kernel
     entry via `kernel.cpp`'s `set_pmu_enabled`), so a5 was also hoisted to
     function scope for parity. This gate is **load-bearing**: `pmu_active`
     feeds `dispatch_ready_tasks(...)` and forces single-issue dispatch.
  2. `is_dep_gen_enabled()` in `pto_orchestrator.cpp::submit_task_common` —
     once per task submission.

Fix applied (issue #1146): gate both, both arches, under the **existing**
`SIMPLER_DFX`. A new `PTO_PROFILING` macro was rejected — it would only
duplicate `SIMPLER_DFX`. For the load-bearing PMU site the call is replaced,
not deleted, so the value stays well-defined when profiling is compiled out:

```cpp
#if SIMPLER_DFX
    const bool pmu_active = is_pmu_enabled();
#else
    // PMU is definitionally off when profiling is compiled out; hard-set false
    // so dispatch keeps its overlapping (non-single-issue) fast path.
    constexpr bool pmu_active = false;
#endif
```

The `dep_gen` per-task capture wraps the whole `if (is_dep_gen_enabled()) { ... }`
block in `#if SIMPLER_DFX / #endif`. To keep the subsystem internally
consistent, the **three remaining cold `dep_gen` sites** were folded under the
same macro: `dep_gen_aicpu_init` (`scheduler_cold_path.cpp`, one-time boot) and
`dep_gen_aicpu_set_orch_thread_idx` / `dep_gen_aicpu_flush` (`aicpu_executor.cpp`,
once per orch thread). Gating only the per-task capture would leave a half-on
state at `SIMPLER_DFX=0` — buffer init + an empty flush, but no records —
which is worse than a clean compile-out. This supersedes the previous in-tree
comment that dep_gen was "gated independently of `SIMPLER_DFX`"; that comment
was updated in the same change.

## Result

The two hot-path gates plus the whole dep_gen subsystem (init / set_idx / flush)
gated under `SIMPLER_DFX`, both arches, plus the a5 PMU gate hoisted to
function scope to match a2a3 — no behavioral change at the default
`SIMPLER_DFX=1`. All touched translation units (`scheduler_dispatch.cpp`,
`pto_orchestrator.cpp`, `scheduler_cold_path.cpp`, `aicpu_executor.cpp`)
`-fsyntax-only` clean under `SIMPLER_DFX=1` **and** `SIMPLER_DFX=0`.

**Magnitude is unmeasured.** No AICPU scheduler profile was taken. "Much
slower" is mechanism-true (one non-inlinable call per iteration / per submit)
but quantitatively unverified. The strongest existing evidence the cost is real
is the in-tree hand-hoist of the PMU gate; that is an argument, not a number.

## Why not (now)

- **No new macro.** Reusing `SIMPLER_DFX` avoids a duplicate config
  permutation (see `.claude/rules/env-macro-gating.md`).
- **No benchmark.** The change is a zero-risk compile-out at the default build,
  so it shipped without first quantifying the win. If anyone needs to justify
  it harder, profile before claiming a number.
- **The `SIMPLER_DFX=0` + dep_gen-enabled combo is given up on purpose.**
  Gating the whole dep_gen subsystem means a `=0` build can no longer capture
  dep graphs. That combo is exercised by nothing today — the only `=0` CI leg
  (`profiling-flags-smoke / pto2-off`) runs `vector_example`, which never
  enables dep_gen — and `=0` is the production-inference build where dep_gen (a
  diagnostic) is unwanted anyway. A `=1` build still toggles dep_gen at runtime
  via the host SubmitTrace flag exactly as before.

## When to reconsider

- If an AICPU scheduler profile shows either gate as a non-trivial slice of the
  dispatch loop / submit path, that turns the "unmeasured" caveat into a real
  number — record it here.
- If a real need appears for capturing dep graphs in a `SIMPLER_DFX=0`
  build, the dep_gen subsystem would have to be split back out from
  `SIMPLER_DFX` onto its own gate (host-driven, as it was before this
  change).

## References

- Issue [#1146](https://github.com/hw-native-sys/simpler/issues/1146) — the
  code-health issue this implements. Related: #1103.
- `docs/.../tensormap_and_ringbuffer/docs/profiling_levels.md` — the
  `SIMPLER_DFX` macro hierarchy and defaults.
- `.claude/rules/env-macro-gating.md` — why no new macro was added.
