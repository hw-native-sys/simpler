# Host runtime trace markers — `[STRACE]`

`simpler_run()` spans several host-side stages (`bind`, `runner_run`,
`validate`) plus, inside `runner_run`'s enqueue-through-drain lifetime, an
on-NPU AICPU window that itself subdivides into preamble / SO-load /
graph-build / post-orch. The
two headline walls (`host_wall` / `device_wall`, see
[l2-timing.md](l2-timing.md)) cannot show *where* the time goes.

`[STRACE]` markers are simpler's answer — host-side trace spans emitted to the
log, analogous to Android atrace/systrace. A consumer (e.g. pypto-serving)
reads the per-stage breakdown **from the log**, with **no code change** on its
side and no API contract: `run()` returns `None`, so markers (not a return
value) are the channel, and the log is the one sink the L3 parent and its L2
children share.

`[STRACE]` rides on the compile-time `SIMPLER_HOST_STRACE` macro (default on, in
`src/common/task_interface/profiling_config.h` — separate from the
`SIMPLER_DFX` gate on the device Orch/Sched markers) and is emitted at
`LOG_TIMING` (the default threshold) — **no new env var or flag**. In a
`SIMPLER_HOST_STRACE`-off build the RAII macros compile to nothing.

## Marker grammar

One line per span, emitted on scope exit
(`src/common/log/include/common/strace.h`):

```text
[STRACE] v=1 pid=<n> tid=<n> inv=<n> hid=<hex> depth=<n> name=<dotted> ts=<ns> dur=<ns> [k=v ...]
```

| Field | Meaning |
| ----- | ------- |
| `v` | format version; the parser branches on it. Lets device-side markers align later by reusing the prefix + adding fields. |
| `pid` `tid` | process / thread id — L3 parent and each L2 child are distinct pids, so they land on separate lanes. |
| `inv` | process-wide `simpler_run` invocation id (allocated from an atomic, so `(pid, inv)` is unique even across concurrent calls) — **a grouping key only** (gathers one call's spans), NOT a token index. Set once per call. |
| `hid` | callable content hash (ELF Build-ID 64), stable across slot reuse / processes / runs. The parser buckets by `hid`; the most-frequent bucket is decode (one invocation per token), a once-seen bucket is prefill. |
| `depth` | thread-local nesting depth (`++` on enter, `--` on exit). The parser rebuilds the call tree from `depth` — **not** from timestamp containment. |
| `name` | dotted span name (self-locating even without the tree). |
| `ts` `dur` | start + duration in ns. Maps 1:1 onto a Chrome-trace `"X"` event. For host spans `ts` is `CLOCK_MONOTONIC` (`steady_clock`), same-host cross-process comparable. For `clk=dev` device spans (see below) `ts` is instead a **device-clock** start offset on a per-invocation origin — comparable to the other device spans (so the orch∪sched window is recoverable), not the host clock. |
| `k=v ...` | optional per-span attributes (e.g. `ntensor=4`); a parser that doesn't recognize one ignores it. |

## Span tree

```text
simpler_run                                   (= host_wall)
├─ simpler_run.bind
│  ├─ simpler_run.bind.args        (ntensor=N: per-tensor device_malloc + H2D)
│  └─ simpler_run.bind.prebuilt    (prebuilt runtime-arena cache hit or build + upload)
├─ simpler_run.runner_run          (device enqueue + completion drain)
│  └─ simpler_run.runner_run.device_wall      (whole on-NPU AICPU wall)
│     └─ .{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}
│           device-domain (clk=dev): AICPU subdivision of the on-NPU wall
└─ simpler_run.validate
```

The `device_wall` + its `.{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}`
spans are **device-domain**, tagged `clk=dev`. They are not host `steady_clock`
spans: the AICPU stamps raw sys-counter cycles into a host-allocated buffer
(whose address rides on `KernelArgs::device_wall_data_base`), the host reads it
back after stream-sync, converts cycles → ns, and emits the marker. `orch`/
`sched` are the orchestrator/scheduler windows that formerly only appeared as
device-log lines. A phase that was never stamped
(0 ns) is skipped — e.g. `so_load` is ~0 on a cached-callable run. See
[device-phases.md](device-phases.md) for the device-side mechanism.

The phased native-run interface preserves this same marker contract. Prepare
allocates one `inv` and records the host-wall start; prepare, the child progress
path's launch/drain lifecycle, and finalize bind that `(inv, hid)` while
emitting their spans. Finalize releases the runner claim, destroys the per-run
state, and then emits the stored `simpler_run` wall, so the root includes that
cleanup tail.
No trace scope or synthetic nesting remains active between C API calls. For
direct phased use the host wall is the full prepare-to-finalize lifetime,
including time the caller spends polling or doing other host work; blocking
`simpler_run` is the same phases composed back-to-back.

| Depth | Span names |
| ----- | ---------- |
| 0 | `simpler_run` |
| 1 | `simpler_run.bind`, `simpler_run.runner_run`, `simpler_run.validate` |
| 2 | `simpler_run.bind.args`, `simpler_run.bind.prebuilt`, `simpler_run.runner_run.device_wall` |
| 3 | `simpler_run.runner_run.device_wall.{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched,task_slot_*}` |

## L3/L4 host scheduler spans

A hierarchical worker with direct local chip children also emits these spans
through the same process-global `libsimpler_log.so` sink:

| Span | Host decision point |
| ---- | ------------------- |
| `l3.graph_build` | serialized Python graph callback |
| `l3.submit` | next-level task publication after slot allocation |
| `l3.dispatch` | scheduler handoff to a worker thread |
| `l3.frame_submit` | local child mailbox-frame publication |
| `l3.activate` | prepared-frame activation |
| `l3.complete` | terminal child progress handling |

Their attributes carry the available `run_id`, `task_slot`, `group_index`,
`worker_id`, `dispatch_id`, and endpoint kind. The logger is loaded
and the fixed host-span ABI is resolved before local children are forked, so
parent and child markers reach one sink. Topologies without a local chip binary
path do not initialize this bridge and continue without scheduler markers.

## Reading the markers — `strace_timing.py`

```bash
# TPOT table (per-callable, decode = most-invoked hid bucket)
python -m simpler_setup.tools.strace_timing path/to/host_or_device.log

# also emit the established per-invocation call-tree JSON
python -m simpler_setup.tools.strace_timing path/to/log --trace-out strace.json

# emit the L3/L4 host scheduler timeline on real OS pid/tid lanes
python -m simpler_setup.tools.strace_timing path/to/log --swimlane host_swimlane.json
```

The tool groups by `(pid, inv)`, rebuilds each invocation's tree from `depth`,
buckets by `hid`, and prints each callable's mean `simpler_run` plus per-stage
means. With `--trace-out` it writes one `ph:"X"` event per span on a synthetic
per-invocation lane, so each call renders as an isolated nested tree in
[Perfetto](https://ui.perfetto.dev) / `chrome://tracing`.

`--swimlane` is a separate view. Host slices keep their real OS pid/tid, and
task submission-to-dispatch handoffs render as flow arrows. Chrome Trace JSON
has only one visible timestamp axis, so putting the raw per-invocation device
clock beside `CLOCK_MONOTONIC` would create a multi-day empty interval. The
converter therefore keeps `clk=dev` records, with their original ns timestamps,
in the top-level `unalignedDeviceSpans` array instead of `traceEvents`; it does
not guess a clock offset. Perfetto opens directly on the host activity, while
the existing tables, tree, and `--trace-out` still provide the device-phase
timing views.

## Why markers, not a return value

Android's atrace writes to the ftrace `trace_marker` sink and systrace renders
it; nobody changes their code to be observed. `[STRACE]` mirrors that: the
runtime emits, tooling renders, the caller is untouched. Concretely, `run()`
returns `None`: an L3 `DistributedWorker.run` has no single device wall, and a
return-value channel could not carry each L2 child's host/device breakdown up
anyway. The log can. This is also why device phases are emitted as markers from
the host C++ rather than threaded back through any return struct to Python.
