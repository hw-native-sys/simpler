# Framework queue → public Worker eager execution

This bounded A3 reference consumer puts `Worker.submit` in torch_npu's actual
host emission queue. Torch operations produce two DEVICE inputs; the ordinary
public Worker launches one add kernel; a Torch tensor addition consumes its
output on the same framework stream. TMR and HBG use the same consumer.

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

The optional system test covers both runtimes and both policies. It is manual
because generic onboard environments need not include torch_npu. Run this suite
serially inside one exclusive `task-submit` allocation of one device, as below;
resource-parallel dispatch and xdist are not qualified by this example:

```sh
.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 --run   'python -m pytest tests/st/a2a3/framework_eager --platform a2a3 --device "$TASK_DEVICE" --manual include -v'
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
Worker cannot see a callback that has not called it yet. Cross-stream switching,
alias/INOUT, full framework process shutdown and capture are remaining gates.

This is a reference consumer of the existing eager contract, not acceptance of
PyPTO production dispatch, vLLM/model numerics, A5 or capture/replay.
