# Framework host-queue GIL probe

Diagnostic for [Simpler #2774](https://github.com/hw-native-sys/simpler/issues/2774).
It uses the real Torch NPU emission queue and Python `stream.query()` /
`torch.npu.empty_cache()` entry points. It does not load Simpler, dispatch a
kernel, or qualify a production native submission interface.

The callback waits on a native condition variable before attempting to acquire
the GIL. The caller waits for that gate to become ready with the GIL released,
then opens it and enters the selected drain operation while holding the GIL.
An extended Python switch interval prevents an incidental interpreter timeslice
between opening the gate and entering the operation. No sleep controls callback
ordering. Each case runs in a fresh subprocess with `TASK_QUEUE_ENABLE=1`.

The queue is warmed with a native no-op before the gated callback. Torch NPU
2.10.0 drains the first enqueue while its repository is in `INIT`; gating that
first callback would deadlock the probe before reaching the operation under test.
Initialization and stream synchronization alone do not perform this transition.

## Run

Use a project-local venv containing matching Torch and Torch NPU packages,
setuptools, and the CANN runtime environment. No Simpler runtime build is required.

```bash
source .venv/bin/activate
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tools/framework_queue_gil/probe.py build
```

The build command prints the extension path. Substitute that path below. On a
shared NPU host, run the complete probe under one device allocation:

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --device auto --device-num 1 --timeout 300 --max-time 280 \
  --run "python tools/framework_queue_gil/probe.py run \
    --bridge <extension-path> --device \$TASK_DEVICE \
    --output build/framework-queue-gil/results"
```

Use the actual hardware architecture for the precheck. The run inherits the
active Python/CANN environment. Set `ASCEND_PROCESS_LOG_PATH` to an existing,
per-run directory before submitting to isolate device logs.

## Interpret results

Four positive controls must complete:

1. Native callback followed by Python `query()`.
2. Native callback followed by Python `empty_cache()`.
3. Python callback followed by a C++ query wrapper that releases the GIL.
4. Python callback followed by a C++ empty-cache wrapper that releases the GIL.

The target cases use a Python callback followed by the ordinary Python query
and empty-cache entry points. The diagnostic records `gil_wait_timeout` only
when the callback reached its gate, the caller entered drain, and the callback
attempted but did not finish acquiring the GIL. Other timeouts are unclassified.
On timeout the parent attempts a native all-thread stack capture with `gdb`,
then kills and reaps the child. Missing `gdb` does not invent stack evidence.

`results.json` and one log per case are retained. A failed positive control
stops the probe immediately. Exit zero means the diagnostic completed with
passing controls and classified target outcomes; **it does not mean the product
is fixed**. Target outcomes may be either completed or GIL-wait timeout, depending
on the framework version. Inspect the JSON and stacks rather than the exit code
alone. Do not use this probe's intentional timeout as an ordinary passing ST.

### Observed baseline

On 2026-10-29, the probe built against Torch `2.10.0+cpu` and Torch NPU `2.10.0`
ran on A3 with CANN `9.0.0`, CPython `3.10.9`, Linux aarch64, and an exclusive
device allocation. The Simpler checkout base was
`fb63f4c02220ed99a70150409c676711dcdbb418`; no Simpler runtime was loaded.

| Callback | Drain entry | Result |
| -------- | ----------- | ------ |
| Native no-op | Python query | Completed |
| Native no-op | Python empty_cache | Completed |
| Python no-op | C++ query with GIL released | Completed |
| Python no-op | C++ emptyCache with GIL released | Completed |
| Python no-op | Python query | GIL-wait timeout |
| Python no-op | Python empty_cache | GIL-wait timeout |

Both target children reached `DRAIN_ENTER_GIL_HELD` and `CALLBACK_WAIT_GIL`,
never reached `CALLBACK_ACQUIRED_GIL`, and were killed after the 30-second
observation window and bounded stack-capture attempt. Query's native stack
showed the caller in `NPUStream::query` → `NPUStream::stream` → `eventfd_read`,
and `acl_thread` in `pybind11::gil_scoped_acquire` → `PyEval_AcquireThread`.
Empty-cache's Python stack stopped in `torch_npu/npu/memory.py::empty_cache`;
its native stack capture timed out after ten seconds, so no complete native
stack is claimed for that case.

The successful run's local evidence is under `build/framework-queue-gil/results/`:
`results.json`, `source.json`, executed-source copies, per-case logs, and native
stack-capture outputs. The executed Python source differs from the checked-in
candidate only by the subsequently added copyright header. Earlier probe attempts
without the first-enqueue warmup failed before the target boundary and are not
evidence for the GIL defect. This is one controlled run per case, not a measured
production incidence rate or a claim about other framework releases.

The Python callback is deliberately a no-op: this isolates the need to enter
Python before `Worker.submit`, not Simpler's device execution. The native control
also does no work and establishes only that this queue/drain combination can
progress without acquiring the GIL. It does not validate native resource release,
Worker closure, numerical results, or TMR/HBG behavior. The released-GIL controls
are diagnostic comparisons, not a proposal to patch framework locks blindly.

## Native submission design constraints

The existing `_submit_chip_run_direct` binding already releases the GIL during
native execution. Exposing it alone loses the Python Worker's operation lease,
source-buffer reservations, materialization, accepted-run registration, and
finalization bookkeeping. A supported interface must preserve those contracts.

The proposed boundary is preparation on the Python thread, a native-only token
consumed by the framework callback, and completion observed through the existing
run lifecycle. The following decisions must be implemented together:

- Preparation reserves the callable and buffers before any asynchronous gap;
  descriptor copying alone is not storage retention. Callable unregister must
  observe prepared uses as well as already submitted native runs.
- Native submission reuses `ChipRunLane` admission and caller-stream enqueue
  semantics. Preparation must not consume launch capacity in a way that makes
  the caller wait for a callback it has not yet enqueued.
- A token is one-shot. Submit versus cancel/close needs one native linearization
  point; losing either race must not dereference a destroyed Worker. A raw worker
  pointer plus a Python-side keepalive does not define this protocol.
- Close distinguishes pending preparation from native submission in progress.
  The treatment of queued-but-not-submitted tokens must be explicit, including
  how cancellation reaches the framework error channel and prevents a successor
  from silently consuming an unwritten output.
- Native callback destruction, error storage, and capacity recovery must not
  depend on Python finalization. Python owners can be released at a later GIL-safe
  boundary, but queue progress cannot wait for that boundary.
- Device completion and Python-owner retirement remain separate. Unknown native
  completion must retain owners and preserve the existing poisoned/quarantine
  behavior; a callback returning is not device completion.

The candidate implementation now exposes `Worker.prepare_submission` and the
`src/common/worker/native_submission.h` capsule SDK. See the framework eager
`native_submission.py` example and its system test. The baseline diagnostic
results above describe the original Python callback, not this candidate.
Production acceptance additionally requires native submit/error/destruction tests,
close/unregister races, and real producer-kernel-consumer numerical cases for each
supported runtime and queue configuration. PyPTO adapter migration remains tracked
in [PyPTO #3209](https://github.com/hw-native-sys/pypto/issues/3209).
