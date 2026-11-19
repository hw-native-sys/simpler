# percounter faulting the device after an upstream instrumentation change

**Date:** 2026-11-12 · **Silicon:** a5 · **Outcome:** percounter now owns the
accesses its lane runs on, in `src/a5/runtime/percounter/runtime/scheduler/`.

## What happened

Merging `upstream/main` into the percounter branch built cleanly and then
faulted the device on the qwen3-14B decode scene, at a fixed 21.000 s, with
`507018` and `SCHEDULER_TIMEOUT`. Raising `SIMPLER_SCHEDULER_TIMEOUT_MS` to
ten minutes changed nothing, so it was a fault and not a watchdog.

## Finding it

A bisect over the sixteen upstream commits, one hardware run each:

| merged up to | result |
| ------------ | ------ |
| nothing (pre-merge control, same machine, same environment) | PASSED |
| `143050346` "simplify A5 HBG scheduling and compact shared state" | PASSED |
| `737b73057` | PASSED |
| `100dece2e` | PASSED |
| **`0e8ddaca1` "improve A5 L3 scalar profiling and worker timelines"** | **faulted** |
| `0f6fd790f` | faulted |
| `upstream/main`, including `868fa33ad` which deletes `scheduler_ssbuf.h` | faulted |

Two commits were guessed at before the bisect ran and both were wrong:
`868fa33ad`, because deleting a header percounter used looked like the
obvious cause, and `143050346`, because "compact shared state" reads like a
layout change. Each tested clean on its own. **The control matters as much as
the bisect** -- it was run after the fact here, and it is what makes the four
failures mean anything.

a5sim does not reproduce it: eight scene tests under percounter, including
graph, MIX, cohort and SPMD cases, all pass on the merged tree. Whatever this
is, the simulator does not model it.

## What `0e8ddaca1` did

It routed every primitive in `scheduler_memory.h` through
`scheduler_scalar_memory_issue`, an `__attribute__((noinline))` tracing
wrapper, added a `[[block_local]] static` trace pointer to the AICore image,
and -- the part that reaches this runtime -- pointed `scheduler_ssbuf.h` at
`scheduler_memory.h` and replaced the `dsb((mem_dsb_t)0)` inside
`scheduler_ssbuf_load_acquire` and `scheduler_ssbuf_store_release` with a
call to the wrapped `scheduler_cache_barrier()`.

Those two accessors are the MIX rendezvous's only ordering, and
`percounter_mix_wait` runs one per peer per spin. host_build_graph's own
AICore side mostly waits to be told; this one schedules itself, so the same
edit lands on the two runtimes very differently.

**Which of those three changes a lane cannot absorb was not isolated.** The
bisect's granularity is the commit, and the fix covers all three. An earlier
attempt to isolate the wrapper by removing `scheduler_memory.h` from the
AICore image failed to isolate anything, because `scheduler_ssbuf.h` pulls it
back in -- which is stated in the first hunk of that commit's own diff. The
conclusion drawn from that attempt ("the wrapper is not the cause") was
wrong and is retracted here.

## The fix

Three headers under `percounter/runtime/scheduler/`, each a copy of the
pre-`0e8ddaca1` version with percounter's prefix:

- `ssbuf.h` -- the window constants and the two accessors, barrier as an
  instruction
- `gm.h` -- the GM reads, stores, atomics and payload writeback, inlined
- `image_read.h` -- the predicate evaluator and the callable lookup, all the
  lane needed from a ready-queue header it has no other use for

What the two runtimes genuinely share stays shared: the graph image's types,
the layout, the topology, the watchdog. What moved is the half whose *cost*
differs between a scheduler the AICPU drives and one the lanes drive
themselves -- which is the half a sibling's instrumentation can change
underneath this one.

Merged to `upstream/main` afterwards: 30 rounds of the qwen decode scene
against golden, 22.570 ms mean, against a 22.573-22.737 ms spread for the
same code across load windows.

## What this says about the arrangement

percounter compiles host_build_graph's source directories rather than copying
them, and the PR argues that a fix landing there reaches this runtime
unchanged. That is true, and so is the converse: this is the first time an
upstream change reached it as a device fault. The review raised the same
coupling under "source ownership" before it happened. Three headers is the
narrowest answer; the AICPU and host images still compile the sibling's
sources whole, so the next instance of this will land there.
