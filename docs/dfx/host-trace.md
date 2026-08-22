# Host runtime trace markers — `[STRACE]`

`simpler_run()` spans several host-side stages (`bind`, `runner_run`,
`validate`) plus, inside `runner_run`'s enqueue-through-drain lifetime, an
on-NPU AICPU window. TMR subdivides that window into preamble / SO-load /
graph-build / post-orch; HBG emits the whole device wall without those
device-orchestrator phases. The two headline walls (`host_wall` / `device_wall`, see
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
`SIMPLER_HOST_STRACE`-off build the RAII macros compile to nothing. In an
enabled build, Host call sites also read the live process threshold before
constructing span attributes; `WARN`, `ERROR`, and `NUL` therefore disable the
instrumentation work as well as the output. Device logging still uses the
initialization-time policy described in [logging.md](../logging.md).

Set `SIMPLER_HOST_STRACE_DIR` to an existing directory to write host span
records to `host-strace.<pid>.log` instead of stderr. Each process uses a
fully-buffered private file and flushes it when a depth-zero invocation record
is emitted. Ordinary log records and the clock anchor remain on stderr. If the
file cannot be opened, host spans fall back to stderr. This diagnostic sink is
useful for multi-process profiling where synchronous writes to one shared pipe
would perturb the short host phases being measured.

## Marker grammar

Every host log record starts with a `CLOCK_MONOTONIC` nanosecond timestamp:

```text
[mono_ns=<ns>][T0x<thread>][<level>] <func>: ...
```

Each process emits one TIMING-level mapping from that clock to wall time when
its logger starts:

```text
[CLOCK_ANCHOR] v=1 pid=<pid> mono_ns=<ns> wall_ns=<ns>
```

For host-clock records, consumers recover an approximate absolute timestamp
with `wall_ns + record_mono_ns - mono_ns`. Their event ordering and duration
calculations remain entirely on the monotonic clock and are unaffected by
wall-clock corrections. Records tagged `clk=dev` use the separate device-clock
domain described below and do not use this anchor.

`strace_timing.py` applies that mapping to both `--trace-out` and `--swimlane`.
The visible Perfetto axis remains monotonic; each mapped host event exposes the
exact decimal `wall_ts_ns` and a UTC `wall_time` in its arguments, while the JSON
top level retains the source mappings in `clockAnchors`. Nanosecond epoch values
are strings because JSON consumers commonly use IEEE-754 numbers, which cannot
represent current epoch nanoseconds exactly. A log with no anchor for a pid gets
no wall time for that pid's events and is otherwise unaffected, and `clk=dev`
records never receive host wall time.

One line per span, emitted on scope exit
(`src/common/log/include/common/strace.h`):

```text
[STRACE] v=1 pid=<n> tid=<n> inv=<n> hid=<hex> depth=<n> name=<dotted> ts=<ns> dur=<ns> [k=v ...]
```

| Field | Meaning |
| ----- | ------- |
| `v` | format version; the parser branches on it. Lets device-side markers align later by reusing the prefix + adding fields. |
| `pid` `tid` | process / thread id — L3 parent and each L2 child are distinct pids, so they land on separate lanes. |
| `inv` | 64-bit process-wide `simpler_run` invocation id (allocated from an atomic, so `(pid, inv)` is unique even across concurrent calls) — **a grouping key only** (gathers one call's spans), NOT a token index. Set once per call. |
| `hid` | callable content hash (ELF Build-ID 64), stable across slot reuse / processes / runs. The parser buckets by `hid`; the most-frequent bucket is decode (one invocation per token), a once-seen bucket is prefill. |
| `depth` | thread-local nesting depth (`++` on enter, `--` on exit). The parser rebuilds the call tree from `depth` — **not** from timestamp containment. |
| `name` | dotted span name (self-locating even without the tree). |
| `ts` `dur` | start + duration in ns. Maps 1:1 onto a Chrome-trace `"X"` event. For host spans `ts` is `CLOCK_MONOTONIC` (`steady_clock`), same-host cross-process comparable. For `clk=dev` device spans (see below) `ts` is instead a **device-clock** start offset on a per-invocation origin — comparable to the other device spans (so the orch∪sched window is recoverable), not the host clock. |
| `k=v ...` | optional per-span attributes (e.g. `ntensor=4`); a parser that doesn't recognize one ignores it. |

Span names and attributes percent-encode control bytes and record delimiters.
They are length-capped (with `~` marking truncation) so each marker remains a
single atomic pipe write even when forked workers share captured stderr.
`strace_timing.py` decodes both on the way back in, so a consumer reading its
output sees the original text; a consumer reading the raw log does not.

## Span tree

```text
chip.run                                      (= host_wall)
├─ chip.run.bind
│  ├─ chip.run.bind.args        (ntensor=N: per-tensor device_malloc + H2D)
│  └─ chip.run.bind.prebuilt    (prebuilt runtime-arena cache hit or build + upload)
├─ chip.run.runner_run          (device enqueue + completion drain)
│  └─ chip.run.runner_run.device_wall      (whole on-NPU AICPU wall)
│     └─ .{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}
│           TMR device-domain (clk=dev): AICPU subdivision of the on-NPU wall
└─ chip.run.validate
```

The `device_wall` span exists for both runtimes. Its
`.{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}`
children are TMR-only; HBG orchestration runs on the host and stamps none of
those phases. All emitted device spans are tagged `clk=dev`. They are not host
`steady_clock` spans: the AICPU stamps raw sys-counter cycles into a host-allocated buffer
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
state, and then emits the stored `chip.run` wall, so the root includes that
cleanup tail.
No trace scope or synthetic nesting remains active between C API calls. For
direct phased use the host wall is the full prepare-to-finalize lifetime,
including time the caller spends polling or doing other host work; blocking
`simpler_run` is the same phases composed back-to-back.

| Depth | Span names |
| ----- | ---------- |
| 0 | `chip.run` |
| 1 | `chip.run.bind`, `chip.run.runner_run`, `chip.run.claim_release`, `chip.run.validate` |
| 2 | `chip.run.bind.args`, `chip.run.bind.prebuilt`, `chip.run.runner_run.device_wall` |
| 3 | TMR phase spans `chip.run.runner_run.device_wall.{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}` and optional `task_slot_*` spans |

## Host scheduler spans

Every hierarchical worker that drives next-level children emits these spans
through the logger compiled into `_task_interface`, since the orchestrator and
scheduler code they run is the same at every level above the chip. **The leading
word names the level that emitted them**, so `<level>` below is one of `node`
(L3), `network1` (L4), `network2` (L5) or `network3` (L6) — see
[`python/simpler/worker_level.py`](../../python/simpler/worker_level.py) for the
ladder. It is never `host`: that names the *processor* this span ABI belongs to,
which every one of those levels runs on:

| Span | Decision point |
| ---- | -------------- |
| `<level>.graph_build` | serialized Python graph callback |
| `<level>.submit` | next-level task publication after slot allocation |
| `<level>.dispatch` | scheduler handoff to a worker thread |
| `<level>.frame_submit` | local child mailbox-frame publication |
| `<level>.activate` | prepared-frame activation |
| `<level>.complete` | terminal child progress handling |
| `<level>.post_fence_retirement` | run erase + quiescent compaction, after the completion fence |

Their attributes carry the available `run_id`, `task_slot`, `group_index`,
`worker_id`, `dispatch_id`, endpoint kind, and the dispatch's pipeline lease
(`slot_id` / `generation`).

The word is resolved once per process, from its Worker's level, and the **first
binding wins**: a `SpanScope` keeps the name pointer it was handed, so a process
that inits Workers at two levels keeps the first one's word and logs a warning
rather than relabelling spans that are still open. One process therefore carries
one vocabulary — which is what makes an L4 run readable, since its own spans say
`network1.` while each of its L3 children says `node.`.

One process contributes at most two host lanes, because the scheduler runs on
one thread: the facade thread emits `<level>.graph_build` and `<level>.submit`,
and the scheduler thread emits the other four. `role=worker` on
`<level>.frame_submit`, `<level>.activate` and `<level>.complete` names the
worker a dispatch targets, not the thread that ran it.

The spans reach the logger over the fixed POD `SimplerHostSpan` ABI in
`common/host_span.h`. `_task_interface` compiles the host logger directly, so
there is no nullable sink and host-span support cannot disappear because a
separate logger DSO was absent. Every other host consumer also compiles a
private logger implementation, then binds it to the
`SimplerHostLogState` owned by `_task_interface`. The hierarchical parent seeds
that state before `fork()`; a chip child re-seeds its inherited copy and passes
the same pointer to every runtime module it loads. The threshold and one-anchor
coordination are therefore shared within each process without relying on
`RTLD_GLOBAL` logger symbols. A private logger is silent before that binding;
afterward its enabled query follows the shared Host threshold on every span.

## `ext.` — spans from outside simpler

Every level word above belongs to simpler, so a span from any other producer
leads with the reserved word `ext.` and names itself:

```text
ext.<producer>.<span>          e.g. ext.pypto.decode_layer
```

All three segments are required. `ext.foo` names no span of its own and
`ext..foo` names no producer, so neither attributes to anyone — such a span is
still recognized as external (it can never be mistaken for ours) but renders in
an unattributed lane. A producer segment may itself be one of our level words:
`ext.host.foo` attributes to a producer called `host` and remains external.

What the namespace guarantees, in both directions:

| Guarantee | What holds |
| --------- | ---------- |
| **Visible in** | `--swimlane`, on the emitting pid/tid. This is the only view that renders external spans. |
| **Excluded from** | the TPOT, rounds and `--tree` tables and `--trace-out`, all of which key on `(pid, inv)`. `inv` is our native run epoch and no public surface exposes it, so admitting external spans would collapse every one of them into a single forged invocation. |
| **Cannot affect** | lane naming, lane splitting, or dispatch-flow pairing. Our views infer those only from our own spans, so `role`, `slot_id`, `run_id` and `depth` on an external span carry whatever meaning its producer wants. |
| **Process label** | a process that emitted only external spans is labelled `external producer <name> (pid=N)`, with no `simpler` prefix. A producer emitting into one of our processes — the common case for a caller of the public API — leaves that process labelled as ours. |

The contract is executable: `tests/ut/py/test_strace_timing.py` asserts each row
above under the `ext.` heading near the end of the file. A repository adapting to
this namespace can read those tests as the specification and mirror them against
its own emitter.

## Reading the markers — `strace_timing.py`

```bash
# TPOT table (per-callable, decode = most-invoked hid bucket)
python -m simpler_setup.tools.strace_timing path/to/host_or_device.log

# also emit the established per-invocation call-tree JSON
python -m simpler_setup.tools.strace_timing path/to/log --trace-out strace.json

# emit the L3/L4 host scheduler timeline on real OS pid/tid lanes
python -m simpler_setup.tools.strace_timing path/to/log --swimlane host_swimlane.json

# ... and subdivide the bind stage with a runtime's own per-event records
python -m simpler_setup.tools.strace_timing path/to/log --swimlane host_swimlane.json \
    --host-phase-records outputs/<case>/host_phase_records.jsonl
```

The tool groups by `(pid, inv)`, rebuilds each invocation's tree from `depth`,
buckets by `hid`, and prints each callable's mean `chip.run` plus per-stage
means. With `--trace-out` it writes one `ph:"X"` event per span on a synthetic
per-invocation lane, so each call renders as an isolated nested tree in
[Perfetto](https://ui.perfetto.dev) / `chrome://tracing`.

`--swimlane` is a separate view. Host slices keep their real OS pid/tid, and
task submission-to-dispatch handoffs render as flow arrows.

A runtime may subdivide a stage it owns, which the markers deliberately do not
describe — the marker grammar is a fixed per-run-stage contract, and a runtime's
internal breakdown of one stage does not belong in it. `--host-phase-records`
takes such a breakdown from the artifact the runtime wrote and draws each record
inside its `chip.run.bind`, matched on `(pid, inv)`. Both sides are the same
`CLOCK_MONOTONIC` axis, so nothing is converted. Without the artifact the tool
still recovers the stage's own segments from the runtime's timing log lines; where
both are present the artifact wins, so a segment is not drawn twice. See
[host_build_graph's profiling levels](../../src/a2a3/runtime/host_build_graph/docs/profiling_levels.md)
for what that runtime records, and
[hbg-bind-phases.md](hbg-bind-phases.md) for what those segments are and
what produces both the log and the artifact on a real decode network.

**One exception, because a K-deep pipeline is not K threads.** The direct-chip
lane drives prepare(N+1) and finalize(N) from the *same* OS thread, so a 40-run
K=2 stress produces 40 overlapping run lifetimes on one tid. Perfetto nests
slices by timestamp containment within a track, so flattening them there puts
run N+1's spans *inside* run N's root — false nesting that hides the very
overlap the view exists to show. A thread whose depth-0 spans overlap is
therefore split into one lane per pipeline slot (`pipeline slot 0 (tid …)`),
which reads as a plain sequence because a run holds its slot exclusively. The
overlap then shows where it belongs: across lanes. Each slice keeps `os_tid` in
its args, and a thread that ran sequentially — the L3 scheduler, which carries a
slot but never interleaves — keeps its real tid.

Chrome Trace JSON has only one visible timestamp axis, so putting the raw
per-invocation device clock beside `CLOCK_MONOTONIC` would create a multi-day
empty interval. The converter therefore keeps `clk=dev` records, with their
original ns timestamps, in the top-level `unalignedDeviceSpans` array instead of
`traceEvents`; it does not guess a clock offset. Perfetto opens directly on the
host activity, while the existing tables, tree, and `--trace-out` still provide
the device-phase timing views.

## Async pipeline proof

The phased native lane claims that run N+1's preparation runs *concurrently*
with run N's device execution. That claim is checkable from a captured log
without any new marker family: the windows are already in the `chip.run`
tree, and the root span already carries the identity that tells two runs apart.

| Property | Read from |
| -------- | --------- |
| successor's preparation | `chip.run.bind` — its arena build + host orchestration |
| predecessor's device work | `chip.run.runner_run` |
| when a successor may launch | `chip.run.claim_release` |
| which run each belongs to | root `chip.run` attrs, joined by `(pid, inv)` |

Only `claim_release` was added for this: it wraps `release_native_run` inside
finalize, the point a successor's launch becomes admissible, and no other span
marks that boundary. `node.post_fence_retirement` covers the L3 orchestrator's
`release_run` tail for the same reason.

The identity is `run_id / dispatch_id / run_epoch / slot_id / generation`. Each
field means one thing: `run_id` and `dispatch_id` are zero on the direct-chip
lane, which allocates neither, and `run_epoch` is a per-process monotonic counter
that is always set — so it is what orders runs when the other two are absent.
`NativeDispatchIdentity.sequence` makes that choice in the parser, where it is
visible, rather than in the record.

```bash
python -m simpler_setup.tools.strace_timing path/to/log --assert-native-overlap
```

Per adjacent run on one child process, the command requires `bind(N+1)` to
**overlap** `runner_run(N)` — the intervals intersect — and `runner_run(N+1)` not
to start before `claim_release(N)`. It exits nonzero on a missing identity, a
missing span, or an ordering violation.

Reading `bind` rather than the whole prepare is deliberate: `bind` sits inside
prepare, so an overlap it reports is one the prepare certainly had.

`--require-hidden-prepare` adds the stronger claim that the preparation also
*finishes* inside the predecessor's device window — fully hidden rather than
merely concurrent. That is a statement about pipeline depth and it is sensitive
to host scheduling, so it is opt-in and the scene test asserts only the overlap
property.

### The scene test carries its own negative controls

`tests/st/a2a3/host_build_graph/concurrent_prepare_stress` drives three arms
through one pipeline driver and reads this same verdict, so each control differs
from the positive arm in exactly one variable:

| Arm | Variable moved | Required verdict |
| --- | -------------- | ---------------- |
| overlap stress | — | accepted, one check per adjacent pair |
| serial submission | one run in flight instead of two | rejected, `did not overlap` |
| diagnostics config | `enable_scope_stats` set | rejected, `did not overlap` |

The second arm is what makes the first a detector rather than a formality.
Between the pipeline and the verdict sits a chain — which spans are emitted,
where their endpoints land, `bind` standing in for preparation, `runner_run`
being a host wall span that includes caller polling — and if any link reported an
intersection independent of real concurrency, the positive arm would still be
green. Matching the message matters: it separates a real rejection from the
vacuous "need at least two complete native runs" one.

The third arm covers a fallback that is otherwise silent. `allow_prepared_successor`
folds in `CallConfig::diagnostics_any()` — the OR of all five diagnostic flags —
because a collector's setup mutates runner-global state that is not yet
per-epoch, so *any* one of them keeps a run and its successor on separate device
windows even at depth 2. The lane's own check declines to stage rather than
raising, so the submissions still succeed and the goldens still pass; nothing
else would notice. Which flag is set does not matter, only that
`diagnostics_any()` becomes true, so the arm picks the lightest.

Staging has three inputs and only that one is reachable from a submission. The
other two — the runtime PipelineContract's `pipeline_depth` and the runtime's
concurrent-prepare capability symbol — are compile-time properties: every onboard
runtime declares depth 2 (`PTO_PIPELINE_MAX_DEPTH` is 2) and returns 1 from the
capability impl, while the sim platform hardcodes 0, so overlap never happens
under simulation. Because neither is configurable, being unable to stage a
successor is a **failure** in the scene test rather than a skip — a skip would
report green for the one state in which the property cannot hold. The platform
gate runs first, so the sim path never reaches that assert.

The two negative arms differ in how long they are meant to last. The serial one
names no mechanism — one run in flight cannot overlap under any admission policy
— so it is permanent. The diagnostics one is deliberately perishable:
`concurrent_native_prepare_supported_impl` keeps collector-bearing configurations
sequential only *until their state is per-epoch*, and once that lands and
`diagnostics_any()` leaves `allow_prepared_successor`, this arm fails with the
very `did not overlap` it now requires. **Delete it then; do not restore the
serialization** — its value and its lifetime both come from the fallback being
silent.

## Why markers, not a return value

Android's atrace writes to the ftrace `trace_marker` sink and systrace renders
it; nobody changes their code to be observed. `[STRACE]` mirrors that: the
runtime emits, tooling renders, the caller is untouched. Concretely, `run()`
returns `None`: an L3 `DistributedWorker.run` has no single device wall, and a
return-value channel could not carry each L2 child's host/device breakdown up
anyway. The log can. This is also why device phases are emitted as markers from
the host C++ rather than threaded back through any return struct to Python.
