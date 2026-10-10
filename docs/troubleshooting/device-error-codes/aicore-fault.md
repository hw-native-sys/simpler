# Chasing down an AICore fault (UB out of bounds, VEC/CUBE instruction error)

← [Device Error Codes](../device-error-codes.md)

An AICore addressing fault reaches the host as the same generic `507018` a stall
does, usually with a `SCHEDULER_TIMEOUT` tail, because the faulting core stops
answering and a watchdog reaps the op afterwards. The device log is the only
place the fault itself appears:

```text
VEC instruction error: the ub address out of bounds
  core id 15, blk:1, errcode 0x4000000000000000
→ aicpu exception → 507018 → sched_error_code=100 SCHEDULER_TIMEOUT
```

**Redirect the device log before the run that you expect to fail.** The harness
does not do this for you: `outputs/<case>_<ts>/` is only created when a DFX flag
is on, so a plain onboard run leaves its device log in the shared
`~/ascend/log/debug/device-<id>/` among every other user's. A fault you cannot
attribute to a file is a fault you cannot diagnose, and the evidence is gone
once the run ends:

```bash
LOGDIR="$PWD/outputs/<label>/ascend"; mkdir -p "$LOGDIR"   # driver fopens; it will not mkdir
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"
```

See the "Device logs" section of
[`running-onboard.md`](../../../.claude/rules/running-onboard.md).

## F1: is it the kernel's own addressing, or did the core jump somewhere else?

These two look identical from the host and have nothing in common as bugs, so
settle it before reading any kernel arithmetic:

- **`PrintErrorInfoForDavinciTask`** — `fault kernel_name`, `hash`, and
  `GetBinAndKernelNameExceptionArgs: binSize`. If `binSize` matches the
  runtime's own `aicore_kernel.o` rather than your kernel, the report is naming
  the **polling-dispatch executor**, and the faulting code is whatever it jumped
  to — a dispatch-payload / `function_bin_addr` problem, not kernel arithmetic.
- **`PrintCoreInfo`** — `pc current`. Whether that offset lands inside the
  executor's binary size or far outside it tells you the same thing
  independently.

[#1036](https://github.com/hw-native-sys/simpler/issues/1036) has the worked
example of making this distinction.

## Match the PC to the uploaded callable

Onboard runtimes emit `Callable image`, `Callable code`, and `Callable registration`
records at the default **TIMING** threshold. Keep the complete per-process host
log, including registration, alongside the device log. An explicit WARN/ERROR
threshold suppresses the publication records.

Each `Callable code` record names `device`, `runner`, `chip_hash`, `chip_dev`,
`func_id`, `code_begin`, `code_end_exclusive`, `code_bytes`, and `code_fnv1a64`.
The addresses and fingerprint describe the patched **host source of a successful
H2D**, not a device readback. The fingerprint is FNV-1a over that child's code
bytes; it is neither a hash of the whole ELF nor proof that device memory stayed
unchanged afterwards.

For a failed run boundary, `Run device failure` is emitted before recovery. It
links `run_epoch`, `generation`, `dispatch_id`, `pipeline_slot`, and `cid` to
`chip_hash`, `chip_dev`, and the device `runtime_args` address. This identifies
the run whose completion failed; an error propagated from a predecessor does
not prove this run caused the fault. Match the device log's PID/device and time
as well. The record reads host bookkeeping only and issues no D2H on a faulted
card.

Find the publication for that PID/device/runner and image, then compare the PC
with each **half-open** range: `code_begin <= PC < code_end_exclusive`. A PC equal
to `code_end_exclusive` is outside that range; it must not silently be assigned
to the preceding function. The hardware report's PC semantics still need to be
checked before concluding that execution fell through. `Callable image freed`
and `Callable images abandoned` delimit allocation reuse; abandonment records
host ownership being dropped, not successful per-image device frees.

These records do not contain tensor data, the device's observed dispatch
arguments, or complete code binaries. Preserve the matching compiled artifacts
before rebuilding or clearing caches. A matching address in a later passing
run is not evidence of the upload address in an earlier failing run.

## Match HBG's pending and running dispatches

HBG's scheduler-timeout shutdown snapshot also emits `DISPATCH` at WARN for
occupied running and pending tokens. Each record includes the logical `core`,
`physical_core`, slot, token, token-selected payload bank, payload and argument
addresses, `function_bin_addr`, `src_payload`, and register base. Idle tokens
produce no record; periodic stall reports do not dump payloads.

Join the logical core to the existing `CLUSTER` and `TASK` records, then compare
its physical identity to the CANN report using that report's core-numbering
convention. Compare `function_bin_addr` with the same run's host `Callable code`
entry. A pending dispatch is distinct from the running dispatch; do not assign
its function to the fault merely because it was published most recently.

These are AICPU-side publication observations, not proof of the bytes the
AICore fetched. Other scheduler threads may still be retiring work, so the
records are not an atomic snapshot. The dump neither dereferences tensor
addresses nor reads the gated argument contents written by the AICore. A run
killed before the scheduler's shutdown snapshot may have no such records.

## F2: rule the kernel's own addressing in or out statically

Before instrumenting, check whether the kernel *can* compute an out-of-range UB
address at all. Compile-time `TASSIGN` offsets and template-derived tile extents
do not make that impossible — the constants themselves can overrun — but they
make it **decidable on paper**, because runtime values that only shrink a tile
(a per-rank chunk count, a shorter row) move no base and grow no extent.

So compute it. Sum the highest byte each `TASSIGN` base plus its tile extent
reaches and compare against the AIV UB size:

- **Fits for every input the guards admit** → the address did not come from
  here, and you are in F1's second case.
- **Does not fit** → you have found the bug without touching the device.

The allreduce collectives were cleared this way in
[#1489](https://github.com/hw-native-sys/simpler/issues/1489): all five modes
bind every tile to a constant base, `nranks` only divides the chunk count, and
the whole footprint stays under 132 KiB of the 192 KiB UB at every rank count.

That case is worth following to its end, because F2's verdict held. The fault
was not in the kernels; it was almost certainly
[#1477](https://github.com/hw-native-sys/simpler/pull/1477), a `CoreTracker`
whose `uint64_t` state overran past 21 clusters while those cases ran at the
device's full 24. Corrupted dispatch state, F1's second case — reached by
believing F2 rather than re-reading the kernel arithmetic a second time.

## F3: separate a real fault from a post-mortem register dump

A core killed while spinning can produce a fault-looking line that describes the
state it was reaped in rather than an instruction it executed. Count the
detectors, per [`running-onboard.md`](../../../.claude/rules/running-onboard.md):
if `FATAL: Task Allocator Deadlock` and `Timeout (N cycles):
producer/consumers` are both **zero** and only `HandleTaskTimeout` fired, then
nothing on the device proved a fault or a deadlock — treat the wait that never
completed as the primary object of the investigation, not the addressing line.
