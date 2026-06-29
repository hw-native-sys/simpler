# Post-fork zero-copy host buffers are per-run performance-neutral

**Date**: 2026-07-02
**Verdict**: no-signal — `create_host_buffer` does not change steady-state
`run()` latency vs a fork-inherited tensor. Do not chase or claim a per-run win.

## The problem it solves (and the one it does NOT)

L3 chip children are forked lazily on the first `run()`. A host tensor created
*after* that fork is not in their address space, so its raw pointer is
stale/unmapped in the child. `Worker.create_host_buffer(nbytes)` hands back
born-shared memory already attached into every child (POSIX shm +
`_CTRL_MAP_HOST` broadcast), so a `torch.frombuffer` view over it round-trips
with **no host-side copy** to make it child-visible. The user contract lives in
[docs/comm-domain.md](../comm-domain.md) (host-buffer section) and the
`create_host_buffer` docstring.

What "zero-copy" means here is **host parent↔child visibility only**. It does
**not** mean the device transfer is avoided.

## Why it is per-run neutral (the fact to remember)

The per-run cost that scales with buffer size is a full-buffer H2D in the chip
child, and it is **registration-agnostic**:
`src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp:410-453` — for
every host INPUT tensor, per run: `device_malloc(size)` →
`copy_to_device(host_ptr, dev_ptr, size)` (line 439) → D2H copy-back for non-IN
tensors. A registered `create_host_buffer` buffer and an unregistered
fork-inherited tensor arrive here as the same `host_ptr` and take the identical
path — there is no `if registered` branch that skips or changes the H2D. The
Python submit-time `_stage_host_buffers_for_chip_submit`
(`python/simpler/worker.py:4169-4198`) only *validates* a registered buffer's
range (a bisect lookup, O(tensor count)); it copies nothing for either kind.

So both paths pay the same per-run device H2D; `create_host_buffer` only removes
a *host-side* visibility copy. Onboard a2a3 up to 64 MiB, the two are within
±1% per run — neutral.

**Gotcha for a future reader:** "zero-copy ⇒ faster runs" is a wrong intuition
here. If a measurement seems to show a per-run win, suspect the harness (e.g. a
fixed-order A-then-B paired interleave loads any first-vs-second position effect
onto the always-first arm) before believing it — the code says there is no
per-run difference to find.

## The real cost of `create_host_buffer`

Its non-zero cost is **one-time setup**: `SharedMemory(create)` + the
`_CTRL_MAP_HOST` broadcast attach into every chip child (scales with N_card),
plus the symmetric `free_host_buffer`. This is outside the `run()` window. If a
single-shot allocate→fill→run→free cost matters, measure *that* — not steady
per-run latency.

## When this verdict would change

If a future change makes the per-run device path registration-aware — e.g. a
registered host buffer gets a persistent device mapping so the per-run H2D is
skipped or replaced by a device-direct read — re-open this. The lever is a
branch in `runtime_maker.cpp`'s tensor loop keyed on `_find_host_buf_entry` /
the host-buffer registry. As of #1027 no such branch exists.

## References

- #1027 — L3 post-fork zero-copy host buffers (commit `eeff3fd5`).
- `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp:410-453`
  (per-run H2D, registration-agnostic — line 439 is the copy).
- `python/simpler/worker.py:3986-4076` (`create_host_buffer`),
  `:4169-4198` (`_stage_host_buffers_for_chip_submit`, validate-only).
- [docs/comm-domain.md](../comm-domain.md) — user-facing host-buffer contract.
