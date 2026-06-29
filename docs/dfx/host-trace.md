# Host runtime trace markers — `[STRACE]`

`run_prepared()` spans several host-side stages (`bind`, `runner_run`,
`validate`) plus, inside `runner_run`'s blocking wait, an on-NPU AICPU window
that itself subdivides into preamble / SO-load / graph-build / post-orch. The
two headline walls (`host_wall` / `device_wall`, see
[l2-timing.md](l2-timing.md)) cannot show *where* the time goes.

`[STRACE]` markers are simpler's answer — host-side trace spans emitted to the
log, analogous to Android atrace/systrace. A consumer (e.g. pypto-serving)
reads the per-stage breakdown **from the log**, with **no code change** on its
side and no API contract: `run()` returns `None`, so markers (not a return
value) are the channel, and the log is the one sink the L3 parent and its L2
children share.

`[STRACE]` rides on the compile-time `SIMPLER_PROFILING` macro (default on, in
`src/common/task_interface/profiling_config.h` — separate from the
`PTO2_PROFILING` gate on the device Orch/Sched markers) and is emitted at
`LOG_INFO_V9` (the must-see INFO tier) — **no new env var or flag**. In a
`SIMPLER_PROFILING`-off build the RAII macros compile to nothing.

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
| `inv` | process-wide `run_prepared` invocation id (allocated from an atomic, so `(pid, inv)` is unique even across concurrent calls) — **a grouping key only** (gathers one call's spans), NOT a token index. Set once per call. |
| `hid` | callable content hash (ELF Build-ID 64), stable across slot reuse / processes / runs. The parser buckets by `hid`; the most-frequent bucket is decode (one invocation per token), a once-seen bucket is prefill. |
| `depth` | thread-local nesting depth (`++` on enter, `--` on exit). The parser rebuilds the call tree from `depth` — **not** from timestamp containment. |
| `name` | dotted span name (self-locating even without the tree). |
| `ts` `dur` | start + duration in ns. Maps 1:1 onto a Chrome-trace `"X"` event. For host spans `ts` is `CLOCK_MONOTONIC` (`steady_clock`), same-host cross-process comparable. For `clk=dev` device spans (see below) `ts` is instead a **device-clock** start offset on a per-invocation origin — comparable to the other device spans (so the orch∪sched window is recoverable), not the host clock. |
| `k=v ...` | optional per-span attributes (e.g. `ntensor=4`); a parser that doesn't recognize one ignores it. |

## Span tree

```text
run_prepared                                   (= host_wall)
├─ run_prepared.bind
│  ├─ run_prepared.bind.args        (ntensor=N: per-tensor device_malloc + H2D)
│  └─ run_prepared.bind.prebuilt    (prebuilt runtime-arena image build + upload)
├─ run_prepared.runner_run          (launch + blocking sync on the AICPU)
│  └─ run_prepared.runner_run.device_wall      (whole on-NPU AICPU wall)
│     └─ .{preamble,so_load,graph_build,post_orch,orch,sched}
│           device-domain (clk=dev): AICPU subdivision of the on-NPU wall
└─ run_prepared.validate
```

The `device_wall` + its `.{preamble,so_load,graph_build,post_orch,orch,sched}`
spans are **device-domain**, tagged `clk=dev`. They are not host `steady_clock`
spans: the AICPU stamps raw sys-counter cycles into a host-allocated buffer
(whose address rides on `KernelArgs::device_wall_data_base`), the host reads it
back after stream-sync, converts cycles → ns, and emits the marker. `orch`/
`sched` are the orchestrator/scheduler windows that formerly only appeared as
device-log lines. A phase that was never stamped
(0 ns) is skipped — e.g. `so_load` is ~0 on a cached-callable run. See
[device-phases.md](device-phases.md) for the device-side mechanism.

## Reading the markers — `strace_timing.py`

```bash
# TPOT table (per-callable, decode = most-invoked hid bucket)
python -m simpler_setup.tools.strace_timing path/to/host_or_device.log

# also emit a Chrome-trace / Perfetto JSON (lane = pid → host call tree)
python -m simpler_setup.tools.strace_timing path/to/log --trace-out strace.json
```

The tool groups by `(pid, inv)`, rebuilds each invocation's tree from `depth`,
buckets by `hid`, and prints each callable's mean `run_prepared` plus per-stage
means. With `--trace-out` it writes one `ph:"X"` event per span keyed by pid, so
the L3 parent and each L2 child render as separate lanes in
[Perfetto](https://ui.perfetto.dev) / `chrome://tracing`.

## Why markers, not a return value

Android's atrace writes to the ftrace `trace_marker` sink and systrace renders
it; nobody changes their code to be observed. `[STRACE]` mirrors that: the
runtime emits, tooling renders, the caller is untouched. Concretely, `run()`
returns `None`: an L3 `DistributedWorker.run` has no single device wall, and a
return-value channel could not carry each L2 child's host/device breakdown up
anyway. The log can. This is also why device phases are emitted as markers from
the host C++ rather than threaded back through any return struct to Python.
