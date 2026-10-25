# Framework queue → public Worker eager execution

This bounded A3 reference consumer puts `Worker.submit` in torch_npu's actual
host emission queue. Torch operations produce two DEVICE inputs; the ordinary
public Worker launches one add kernel; a Torch tensor addition consumes its
output on the same framework stream. TMR and HBG use the same consumer.
Additional consumers exercise caller-stream switching, aliased INOUT views,
accepted callback failures and per-call values through that same public Worker
and host queue bridge. A persistent-state consumer reuses that path across
DEVICE-controlled updates to caller-owned storage.

| Consumer | Executable assertions |
| -------- | --------------------- |
| `main.py` | Same-stream producers → Worker → successor; retained caller frames, observable callback error and close ordering |
| `cross_stream.py` | Separate producer and consumer streams, caller A → B → A on one Worker, explicit framework event dependencies |
| `alias_inout.py` | One Buffer backing full and partially overlapping strided views, INPUT plus INOUT, repeated updates and unchanged guards |
| `scope_statistics.py` | Per-call ScopeStats selection with two nested task scopes; TMR records and allocation frontiers, HBG metadata only |
| `failure_recovery.py` | A Python callback fails after submit acceptance; later requests complete and init-thread Worker close retires all accepted runs |
| `value_arguments.py` | Per-call FLOAT32, INT8, UINT64 and FLOAT64 values; HBG HOST/NONE controls and TMR clean HOST refusal followed by valid DEVICE calls |
| `persistent_state.py` | Twelve DEVICE-controlled weighted cache updates, unchanged guards and weights, zero acceptance and revisited rows |

Public `Worker.close()` drains accepted direct-chip runs through their normal
RunHandle completion and retirement path before checking workspace ownership.
Callers need not individually wait successful runs before closing the Worker.
Unwaited L2 handles and their argument snapshots stay retained until
`RunHandle.wait()` finalizes them or `Worker.close()` drains them. Long-lived
callers should regularly retire completed handles to bound this host bookkeeping.
A context refusal or timeout leaves the tree and ownership available for a later
close retry; error delivery and completion proof remain separate. The framework
must still publish and finish its queued callbacks before closing the Worker.

The C++ extension uses torch_npu's existing `OpCommand::RunOpApi`. It provides
no separate executor or public Torch integration API. PyPTO's process adapter
and capture/replay are outside this eager example.

## Run

Use an installation with matching Torch, torch_npu and CANN, plus the normal
Simpler SDK and PTO-ISA prerequisites. Build the optional host extension without
initializing a device:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=2 python examples/workers/l2/framework_eager/main.py --compile-only
```

Run hardware through the queue, after architecture detection:

```sh
.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --timeout 600 --max-time 180 --device auto --device-num 1 --run   'TASK_QUEUE_ENABLE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=2 python examples/workers/l2/framework_eager/main.py --runtime tensormap_and_ringbuffer --device "$TASK_DEVICE"'
```

Select `host_build_graph` for HBG, and run `TASK_QUEUE_ENABLE=0` separately for
the synchronous framework emission policy. A PASS record contains the actual
caller/callback native thread IDs and the number of verified invocations.
TQ=1 requires callback execution on a different thread; TQ=0 requires the caller
thread. Golden values are checked only after every framework successor has
been enqueued and the final framework stream completion boundary returns.

The system tests cover both runtimes and both policies. A session-scoped
fixture builds the optional bridge artifact once. The additional consumer
CLIs load the unique existing artifact, building only when it is absent; the
original same-stream CLI still checks its incremental build cache. All use
the original `build_bridge` / `_load_bridge` helpers. The
`test_environment("torch_npu")` marker selects an isolated framework venv in the
dedicated `st-framework-eager-a2a3` PR and Daily job. The default environment
automatically deselects these tests; no directory exclusion is needed.
Selecting the suite without its dependencies fails. See [CI qualification](../../../../docs/ci.md#framework-eager-qualification-a2a3) for the pinned compatibility contract. Run this suite
serially inside one exclusive `task-submit` allocation of one device, as below;
resource-parallel dispatch and xdist are not qualified by this example:

```sh
.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 --run   'python -m pytest examples tests/st --test-environment torch_npu --platform a2a3 --device "$TASK_DEVICE" --junitxml outputs/framework-eager-ci/results.xml -v'
```

## Ordering and lifetime contract

- Resolve the raw stream integer on the caller before enqueuing this consumer's
  work, then pass that value to the callback. In torch_npu 2.7.1 the Python
  `stream.npu_stream` getter drains the emission queue; calling it from that
  queue's own callback waits on itself.
- Allocate and describe Tensor views before submission. Retain each TaskArgs,
  callable, Worker, Buffer, Torch tensor and stream through queued callback
  execution and runtime retirement. Tensor `record_stream` additionally names
  its allocator use; this example also retains the backing tensors explicitly.
- The callback uses the existing `Worker.submit(..., caller_stream=raw_stream)`.
  There is no intermediate `stream.synchronize`, `RunHandle.wait` or device
  synchronization inserted between producers, the Worker and successors.
- Native callback entry and final framework draining release the caller's GIL;
  queued Python work acquires it while invoking the public Worker. Exceptions
  remain in the queued-call result and `check()` reports them before success.
- The reference consumer inserts no explicit completion wait between producers,
  submission and successors. Its explicit wait is the final framework boundary;
  native admission, predecessor retirement and capacity backpressure can still
  wait. These checks establish ordering, not overlapping execution or throughput.
  Result checking precedes
  `RunHandle.wait`, Worker close and Buffer release.
  Unknown completion retains the failing process's owners rather than treating
  a callback exception as permission to free them.
  `FRAMEWORK_UNPROVEN_RETAINED` explains this deliberate retention on failure;
  exit-time nanobind leak warnings can then be diagnostic noise from retained
  owners, rather than evidence of a binding reference-count defect.

The executable assertions cover three successive calls and construction-scope
exit plus `gc.collect()` with caller-owned frames retaining the Torch tensors.
`caller_tensors_retained` measures that caller responsibility, not automatic
Torch ownership by the runtime. `zero_stream_preflight_refusals` covers only the
public `_nonzero_uint64` zero-value check followed by a valid submission; it does
not reach or qualify the native context/capture validators.
The assertions also cover an observable queued Python exception,
and use of the caller's framework stream after Worker close. The framework
consumer must keep unsubmitted callbacks alive and finish them before close;
Worker cannot see a callback that has not called it yet.

`cross_stream.py` creates producer, caller A, caller B and consumer streams.
Each producer records readiness; the selected caller waits before submission,
and the consumer waits for that caller's completion event before reading.
Three calls use A → B → A while retaining every event, frame and stream through
retirement. All raw stream handles are resolved before this chain. Only the
final consumer boundary is explicitly drained; native backpressure may wait.
The assertions check actual submission handles and golden values, not overlap
or a deterministic numerical failure for every omitted dependency edge.

`scope_statistics.py` queues four calls with ScopeStats off/on/on/off. Each call
submits two vector-add tasks in nested scopes; the final Torch successor checks
10, 16, 22 and 28. After each accepted submit, the callback changes the caller
config and output prefix. Only the two enabled calls may export records to their
original prefixes.

TMR checks six ordered root/outer/inner boundaries, distinct run epochs, and
cumulative task/heap allocation frontiers. HBG exports metadata only because
it has no device scope producer. These checks establish diagnostic attribution;
they do not measure live allocations, overlap or performance. Retirement and
export occur after the final framework completion boundary. Unknown outcomes
retain the caller and runtime owners.

`alias_inout.py` uses one Torch allocation wrapped by one Buffer. Its distinct
64 × 64 Tensor views have row stride 96 and nonzero byte offsets. Each mode
performs successive INPUT + INOUT updates on that same backing: either full
alias or partial overlap by eight floats. The single-tile kernel loads both
views before any store, so the reference uses each step's pre-update snapshot.
A framework successor checks the entire backing, including unchanged guard
regions, after each update. Initial host-to-device preparation precedes all
update chains; no host decision or device copy-back occurs between updates.

`failure_recovery.py` publishes each accepted RunHandle to a caller-retained
list before the first callback raises a Python exception. The queued-call
result reports that exception; it does not cancel the accepted run. Two later
requests and all framework successors complete with correct results. After the
final framework boundary, the init thread calls `Worker.close()` without
individual `RunHandle.wait()` calls, then closes Buffer wrappers and reuses the
caller stream. This covers ordinary Python callback failure and successful
close after completion, not failed native drain, device faults or close while
an unpublished framework callback is pending. An unproved failure retains the
caller and runtime owners.

`value_arguments.py` submits fresh FLOAT32, INT8, UINT64 and FLOAT64 values for
each call, then clears the TaskArgs and changes the scalar wrappers. High bits,
signed values, FLOAT64-only precision and negative zero affect independent
expected results. HBG additionally reads an explicit HOST tensor with
`TensorTransfer.NONE` during host orchestration; changing its contents after
submit must not alter the already prepared result. Its backing remains
caller-owned until retirement: changing contents is not an early-free test.
TMR's unsupported HOST/NONE operand is refused before a subsequent valid DEVICE
request on the same Worker. No tensor H2D transfer is implicit, and this does
not add HOST tensor support to TMR.

`persistent_state.py` keeps one DEVICE weights vector and one caller-owned
8 × 32 strided cache alive across twelve requests. DEVICE cursor/count controls
select a prefix of four proposal rows inside a fixed kernel; host orchestration
does not read the controls or choose subsequent work. Each accepted row adds its
weighted proposal to the existing cache. Zero counts and revisited rows test
retention across calls. Framework successors snapshot the entire backing after
each update, then compare against an independent multi-step reference and
unchanged padding/guards and read-only weights. H2D initialization precedes the
first callback; controls/proposals change through framework DEVICE operations.
This is a bounded persistent-state/accepted-prefix reference, not a complete
MTP model, dynamic host task generation or throughput result.

These are bounded reference cases. Full framework process shutdown, arbitrary
strides or cross-tile overlap, A5 and capture remain separate qualification.

This is a reference consumer of the existing eager contract, not acceptance of
PyPTO production dispatch, vLLM/model numerics, A5 or capture/replay.
