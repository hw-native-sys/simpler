# Containing A2/A3 SDMA stream teardown after an AICore fault

**Date**: 2026-07-21 (containment revised 2026-07-22)
**Verdict**: contained by making SDMA an explicit per-Worker opt-in
(`enable_sdma`), so only a Worker that asks for SDMA carries the slow-teardown
risk and ordinary Workers are unaffected; full recovery inside an SDMA-enabled
Worker after a fault is deferred pending a CANN runtime-and-driver fix

## Question

Can simpler keep PTO-ISA async SDMA available while avoiding the roughly
five-minute cleanup stall caused by its 48 device-only STARS streams after an
AICore fault?

The regression was universal while the runtime provisioned the workspace at
every Worker initialization (and, in an earlier iteration, on the first run of
any callable declaring the workspace). The same risk was already latent in the
communication path: its first window/domain allocation provisioned SDMA on each
communication handle, then cached that manager for the handle's lifetime.

## What was tried

The `aicore_op_timeout` hardware test was made deterministic by observing the
AICore failure before AICPU cleanup, then timed through `Worker.close()`.
The following alternatives were tested or traced:

- no SDMA workspace versus the pinned PTO-ISA manager at `83d01313`;
- destroying the manager's streams before device reset;
- destroying every stream while healthy, then immediately faulting an ordinary
  callable in the same Worker;
- `aclrtSynchronizeDevice()` after healthy stream destruction;
- aborting the device-only streams before destroying them;
- ordinary streams (`flags=0`) and vector-core streams (`0x1000`);
- force stream destruction, stream clear, force/non-force device reset,
  separate ACL contexts, and a helper-process ownership model by following the
  CANN 9.0 runtime and driver source paths.

## Result

| Setup | Fault surfaced | Cleanup/reset | End-to-end |
| ----- | -------------- | ------------: | ---------: |
| No SDMA workspace | `507046` | ~0.3 s | ~9 s |
| 48 device-only SDMA streams | `507015` | ~306 s | ~318 s |

The delay is one 300,000 ms remote TRS event timeout, not 48 cumulative
timeouts. Once the device is `DEV_RUNNING_DOWN`, CANN's `Stream::~Stream()`
still calls `FreeLogicCq()`. That reaches
`NpuDriver::StreamUnBindLogicCq()` → `halResourceConfig()` → the remote TRS
synchronous event. The following 47 stream releases fail immediately after the
first event times out.

A second controlled test exposed a distinct transition race. A real SDMA run
completed, all 48 streams were destroyed while the device was healthy, and the
CANN debug log showed successful device replies for every
`StreamUnBindLogicCq`, `LogicCqFree`, normal SQ/CQ free, and remote stream-ID
free. Nevertheless, an ordinary AICore fault launched immediately afterwards
made `Worker.close()` take 163.5 seconds, versus 13.8 seconds without the SDMA
generation. `aclrtSynchronizeDevice()` returned success between destruction
and the fault but did not change the result.

The debug timeline leaves only about 1.38 ms between the last remote free reply
and the next ordinary dispatch. Driver source explains why those replies are
not a retirement contract: `_trs_hw_sqcq_free_pair()` reaches
`hal_kernel_trs_chan_destroy_ex()`, which removes the channel ID and drops a
kref; final TS mailbox/HW release occurs in `_trs_chan_inst_release()`. The
ordinary/remote path does not expose a completion fence for that final release.
Resource queries also consult ownership maps from which the ID has already
been removed, so “not found” cannot prove retirement. The subsequent force
reset matches `trs_proc_release_check_ts()`'s fixed 160-round retry schedule
(10 × 100 ms followed by 150 × 1 s), plus close/reset overhead.

The logs do not prove which internal reference or firmware-generation state
was still live in that exact 1.38 ms window. They do prove the
application-level contract that matters: even device-confirmed free replies
and device sync are insufficient to admit an immediately faulting ordinary
callable safely.

CANN runtime source also explains why a successful ordinary reset was not a
generation boundary in the later Worker-isolation experiment.
`ApiImpl::SetDevice()` calls `Runtime::PrimaryContextRetain()`, whose
existing-context path increments the reference on every `rtSetDevice`. Normal
`Runtime::PrimaryContextRelease()` calls `TryDecRef()` once and returns success
when that decrement succeeds, even if references remain and teardown is
skipped. The force branch loops `TryDecRef()` until reset becomes true, then
calls `Context::TearDown()`, deletes the context, and clears its stored value.

Moving destruction before reset only moves the wait. Stream abort rejects the
CP-process streams. Force destruction still reaches the same C++ destructor.
A force reset issued after an SDMA-exposed fault still encounters that slow
resource release, so it does not repair the already-faulted generation. The
useful force-reset boundary is while the SDMA generation is healthy. In that
case it drains all primary-context references before any ordinary fault;
ordinary `rtDeviceReset` may decrement only one reference and return success
without reaching context teardown.

Ordinary and vector-core streams create successfully but the AICPU STARS query
fails with `507018`. This is required by the hardware contract: ACL translates
`ACL_STREAM_DEVICE_USE_ONLY` (`0x20`) to runtime
`RT_STREAM_CP_PROCESS_USE` (`0x800`), which allocates CP-local SQ/CQ/register
resources. PTO's AICore code writes SQEs and rings those registers directly;
host-local stream mappings cannot replace them.

## Why not (now)

There is no supported application-level API that both provides the CP-local
SDMA queues and proves their final retirement. A fixed sleep is not a correct
substitute because CANN publishes neither a completion condition nor an upper
bound. A complete CANN fix needs:

1. Make the remote free reply wait for final channel release, or expose a
   generation-based retirement fence that applications can wait on.
2. Skip or wake failed remote logic-CQ operations when the device is already
   down, matching the existing guard around stream-ID release.
3. Make force reset terminate `trs_proc_release_check_ts()` early when the
   TS/CP process is faulted or being rebooted instead of paying all 160 rounds.

The simpler-side containment makes SDMA an explicit per-Worker opt-in rather
than trying to make an SDMA-provisioned device safe for ordinary faulting work.
A Worker is constructed with `enable_sdma=True`; the runtime then provisions the
SDMA workspace once at init, latches its address into the resident `KernelArgs`,
and the AICPU scheduler snapshots it into `GlobalContext::dma_workspace` on every
run. A Worker without the flag creates no SDMA streams, so it keeps main's fast
(~0.3 s) teardown and ordinary `507046` recovery. The workspace is released at
Worker finalize by ordinary stream/manager teardown.

Because the slow-teardown risk is confined to Workers that explicitly opt in,
the containment is organizational: keep SDMA work on its own Worker, and in CI
run it as its own `task-submit` task on a dedicated device (see the "SDMA
workspace smoke" step in `.github/workflows/ci.yml`) so a fault there cannot
slow an unrelated workload sharing the device. `enable_sdma` is honored only by
the a2a3 onboard `tensormap_and_ringbuffer` path that implements both the
provider and `GlobalContext` injection; host-build-graph, simulation, a5, and
provider-disabled builds fail Worker init fast when it is set. Every SDMA event
must still be waited in the kernel or registered with runtime deferred
completion before the callable can return successfully. Communication domains do
not create SDMA streams or transport the address through `CommContext`.

A fault inside an SDMA-enabled Worker still occurs on a device that provisioned
the 48 STARS streams and therefore still pays the slow teardown; that residual
case requires the CANN fix above. Isolating it to opt-in Workers is what keeps
the rest of the system on the fast path.

## Containment validation

- The onboard `prefetch_async_demo` runs the full opt-in path end to end:
  `enable_sdma=True` provisions the 48 STARS streams at init
  (`[SDMA] Created 48 STARS streams OK`), the injected address reaches the
  kernel via `get_dma_workspace`, the prefetch+copy output is bit-exact, and the
  Worker closes cleanly on a healthy card.
- A Worker constructed without `enable_sdma` creates no SDMA streams, so an
  AICore fault on it recovers on main's fast path (`507046`, ~0.3 s reset)
  rather than the ~306 s SDMA-generation stall documented above.
- The two-device `sdma_async_completion_demo` (a separate, pre-existing
  comm-domain SDMA overlay, not the injection path) is unaffected by this change.

## When to reconsider

Retest full fault recovery when a CANN package includes the runtime guard and
driver retirement/reset changes. The acceptance pair is: the real
`prefetch_async_demo` still uses SDMA successfully, and an AICore fault on an
`enable_sdma` Worker after provisioning cleans up in seconds rather than
160--306 seconds. Only then does the per-Worker opt-in stop being load-bearing
for teardown latency (it may still be useful as an explicit capability switch).

## References

- [simpler issue #1425](https://github.com/hw-native-sys/simpler/issues/1425)
- [CANN runtime v9.0.0](https://gitcode.com/cann/runtime/tree/v9.0.0):
  `api_c.cc`, `api_c_device.cc`, `api_impl.cc`, `runtime.cc`, `stream.cc`,
  `stream_sqcq_manage.cc`, `npu_driver_res.cc`, `coprocessor_stream.cc`,
  `context_manage.cc`
- [CANN driver v9.0.0-rc.1](https://gitcode.com/cann/driver/tree/v9.0.0-rc.1):
  `trs_sqcq.c`, `trs_interface.c`, `trs_master_event.c`, `trs_hw_sqcq.c`,
  `trs_proc.c`, `chan_init.c`
- [Ascend memfabric's device-only SDMA stream setup](https://github.com/Ascend/memfabric_hybrid/blob/004d9317289fe99bd6bf13def0500b3fa3795ccc/src/hybm/csrc/transport/device/aiv_sdma_transport_manager.cpp)
- PTO-ISA `83d01313`: `sdma_workspace_manager.hpp`,
  `sdma_async_intrin.hpp`, `sdma_types.hpp`
