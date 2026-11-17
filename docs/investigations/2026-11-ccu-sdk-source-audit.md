# CCU source audit

[Reproduction, terminal protocol and acceptance criteria](2026-11-ccu-terminal-cleanup-evidence.md)

## Audited versions and evidence limits

The source audit refreshed official repositories, read API documentation,
traced implementations, and inspected examples and tests. Its findings are
source evidence, separate from the fix's hardware validation. These are the
snapshots inspected, not a claim that all are one release:

| Repository | Official GitCode master snapshot | Release or dependency comparison |
| ---------- | -------------------------------- | -------------------------------- |
| CANN Runtime | `6a40e4d07c011ceaf14116b7ac18a6a109b71edc` | v9.2.0: `09c468b359b0e393f15436d7e1fc8eb9a77bcab0` |
| HCOMM | `8ca847d6d6444e4daec098754b037a9961846292` | v9.2.0: `95dd5a3f4dd78d05b32fa7a679b8d16f5b3f2cea` |
| HCCL | `348cb971a64bd1f8ce09fad9b244f9aa68c00802` | v9.2.0: `f9a2fd998475dcc78d0066660c3be5d8bd2f05bc` |
| PTO-ISA | `79403e46b48913c687ebe80464a170b150b1ffca` | simpler pin in hw-native-sys fork: `83b1b2b52ded9ec5d91e02532a57cd95e497ba86` |

The hw-native-sys PTO-ISA fork was also checked at
`85969aa8a26e81da253c959541b31d64ae9a47c3`. Its CCU implementation matches
the pin and official snapshot; the official ST host setup has newer peer-address
exchange. GitHub mirrors lagged the official Runtime and HCOMM masters, so
the evidence below uses fixed official revisions.

**Correction to the initial investigation:** the directory named `cann-9.3.0`
does not establish the installed component versions. Runtime, HCOMM and HCCL
version headers/manifests report **9.2.0** on both inspected A5 hosts:

| Host type | Component manifest timestamp | Driver package manifest |
| --------- | ---------------------------- | ----------------------- |
| A5DT, arm64 | `20260923_000324205` | `25.6.rc2.b023` |
| A5PR, x86_64 | `20261009_000323778` | `25.7.rc1.12` |

The release tags are useful comparisons, but these manifests do not identify
the precise source commits or prove equivalence with the installed binaries.
Consequently, a source-level finding below is not a reproduced failure of those
binaries. The HCCL [version guidance][hccl-version] also requires matching the
source release to the installed CANN package.
Driver package metadata does not establish the actual loaded firmware version.

## What the repositories actually establish

### Normal completion: positive evidence

HCOMM's [CCU Write documentation][hcomm-write] requires waiting on the
corresponding event before treating the data as visible to the peer. Its Read
documentation similarly requires completion before accessing the local result.
These describe successful transfer completion, not cancellation.

The official HCCL [CCU AllGather kernel][hccl-kernel] implements
`Write -> EventWait -> peer NotifyRecord/NotifyWait`. Its
[host example][hccl-example] synchronizes the stream and verifies the result
before destroying the communicator and freeing buffers. The v9.2.0 kernel is
identical at these points. The error-check macros return immediately on failure;
they do not implement a fault-stop protocol.

PTO-ISA's [CCU collective implementation][pto-common] and
[broadcast data path][pto-write] implement:

```text
AIV trigger
    -> CCU ReadNb/WriteNb
    -> WaitEvent for transfer completion
    -> all-rank PostSync
    -> RecordEvent(done)
    -> host waits for AIV and CCU streams
    -> verify + MPI barrier
    -> teardown
```

The older `ReadNb`/`WriteNb`/`WaitEvent` and newer HCOMM primitive APIs reach
the same underlying CCU operations in the inspected HCOMM source. Applying the
documented normal-completion semantics across those APIs is a source-based
inference, not a separate abort guarantee.

PTO's `TBROADCAST<CCU>` return is a trigger-stage result; `pto::RecordEvent`
is an empty type and the backend does not wait for CCU completion. The
[ST explicitly waits for both streams][pto-test]. `AsyncEvent::Wait` dispatches
SDMA/URMA/RDMA, so it cannot be repurposed as an existing CCU drain API.

The four inspected A5 CCU ST files contain 14 normal correctness cases, with
no missing-trigger/done or in-flight-abort injection. The official
[runner isolates each CCU GTest in a separate mpirun][pto-runner]. Cleanup
frees buffers before communicator destruction and ignores cleanup return codes;
GTest teardown can also reach it after an early error return. That code is not
evidence that abnormal teardown is safe. No claim of an observed late access
follows from reading it either.

### Stream abort: a stop contract with an unresolved CCU boundary

The Runtime [C++ stream-abort contract][rt-stream-abort] supports 950, stops
executing tasks on the specified stream, discards queued tasks, and requires
dependent streams to be aborted separately. The
[v9.2.0 implementation][rt-stream-code] sends the TS abort command and waits
for `DAVID_ABORT_TERMINATE_SUCC`, then recycles task bookkeeping and resets
the queue. Single-stream abort does not run HCOMM's device-abort callbacks.
If the device is already marked as undergoing device abort, `StreamAbort`
short-circuits to success; concurrent stop operations require serialization.

That direct call graph is not the whole communication cleanup path. HCOMM's
[HCCP event receiver][hcomm-esched] separately accepts TS kill messages when
the custom UDMA path is enabled. It handles UB force-kill by deleting listed
jettys, handles application exit by cleaning the resource lists, and sends
`TOPIC_KILL_DONE_MSG`. Processing errors are logged before that reply; the
reply itself does not carry the local error. Its message header also declares
a CCU force-kill command, but the inspected receiver has no matching case.
Whether a particular stream abort invokes this event path, and which hardware
conditions TS checks before/after the reply, requires the sending implementation.

The missing fact is whether that TS terminal state also covers **every CCU
memory request already issued before abort**, including remote reads/writes
and requests arriving from peers. The inspected API text does not specify that
boundary. Stream completion alone must not silently substitute for the
normal-path transfer events and peer synchronization shown above.
The public [driver RPC path][driver-rpc], inspected at
`d04bdcb1fb1c5d5b2c3188a1fa8da8372de8771e`, transports the request and TS reply.
The algorithm producing the terminal state is across the TS firmware boundary.

The [StreamQuery restrictions][rt-query] also exclude meaningful business
status for persistent/device-only/CPU-schedule streams. Simpler's mission pool
uses ordinary `aclrtCreateStream`; its SCHED expansion-mode value 19 is not a
stream flag. Those restrictions therefore do not by themselves invalidate
simpler's query, nor do they supply the missing CCU scope guarantee.

### Device abort: a real CCU recovery candidate

There is more than an absent API: [DavidDeviceTaskAbort][rt-device-code]
and the [HCOMM callback][hcomm-abort] implement a broader recovery path:

```text
block submission for the process/device
    -> PRE callback: suspend communication submission
    -> OP_ABORT_APP + TS termination/status wait
    -> POST callback: CCU task-kill + communication Clean + kill-state clear
    -> Runtime recovery + reopen submission
```

Both inspected HCOMM versions also clear residual CKE state. Public
`HcclCommSuspend` remains unsupported on 950; the internal callback directly
uses `CollComm::Suspend/Clean`, so these facts are compatible. Public
`HcommChannelDrainOnThread` does not support the CCU engine.

The Runtime [C++ DeviceTaskAbort documentation][rt-device-doc] lists 950 but
marks the API experimental, unsuitable for production, and requiring
stop-on-error mode. These constraints are present in both v9.2.0 and the
inspected master. The Python documentation is inconsistent (reserved/unsupported);
the C++ page is the relevant one for simpler. The operation also reopens
Runtime submission, so simpler would still need its own permanent admission
closure.

HCOMM's [SetTaskKill implementation][hcomm-kill] sends an opcode through
HCCP to the CCU driver. `SetTaskKillDone` **clears** the task-kill state;
its name is not evidence that it queries completion. The inspected HCOMM
code crosses into `libccu-user-drv.so`; the return contract for outstanding
remote transactions is not stated at that boundary. The device-abort path
is the strongest candidate to confirm with the Runtime/HCOMM/driver owners,
not yet a validated replacement for the current lifecycle.

### Destruction and tests: what success does not prove

Instance destruction itself has a concrete error-propagation gap in both
inspected HCOMM versions. In v9.2.0, [CcuInstanceMgr::Destroy][hcomm-destroy]
erases the instance and returns `CCU_SUCCESS`. Its destructor ignores kernel
unregistration and variable/event release results. The
[resource-pack destructor][hcomm-res-pack] calls `CcuReleaseResHandle`, logs
any failure, and clears its handle regardless. Thus a lower release failure
can coexist with public destroy success and an invalidated instance handle.
This is a source-level finding, not an observed hardware failure.

Deleting the Host instance registry is also distinct from revoking an issued
memory request. PTO's [task arguments][pto-addresses] contain peer addresses
and tokens, which its CCU program loads before passing them to `WriteNb`.
An already issued operation does not need to look up the Host instance again.
This explains why handle invalidation alone cannot prove access completion;
it does not establish that late accesses actually occur after destruction.

Failure handling must preserve the distinction between a known intact handle
and partial destruction. Retrying is appropriate only when the failure leaves
the handle valid and the operation is retryable. `CCU_E_NOT_FOUND` after an
ambiguous attempt proves neither physical release nor completion. A reliable
SDK shutdown path would return cleanup errors before erasing the instance,
retain per-resource progress for retry, and define whether it drains accesses
or requires a prior stop. For the agreed simpler implementation, public SDK
results remain authoritative: successful stop/query permits progress to
ordinary owned-handle destruction; a visible failure keeps the affected
owners. Simpler does not compensate for hidden errors by directly releasing
SDK-internal resources.

The inspected HCOMM [extended-jetty deletion][hcomm-jetty] logs deactivate/free
failures without returning them. [RsUbCtxJettyDestroy][hcomm-jetty-destroy]
can then remove its bookkeeping and return success. The v9.2.0 source has the
same error-propagation limitation. Explicit per-channel destruction reaches
this path too. Internal CCU connection cleanup also ignores individual jetty
cleanup results; driver deinitialization has similar unchecked results.

Therefore, a successful high-level destroy/clean return is not independent
evidence that every lower-level release succeeded. If an earlier abort already
guarantees memory quiescence, these failures concern resource reclamation
instead of the earlier access-stop proof; that distinction must be preserved.
Task-kill has another unresolved return convention: its wrapper checks the
`RaTlvRequest` return but not `CustomChannelInfoOut.opRet`, while the ST helper
checks both. Whether the loaded driver folds opcode errors into its function
return needs confirmation; an actual swallowed task-kill error is not established.

HCOMM abort-handler UTs mock task-kill/clean operations. Its inspected CCU
task-kill ST uses a simulated shadow. HCCL's
[resume/rebuild UT][hccl-resume-test] uses stubbed communicator operations;
its algorithm ST is an offline simulator. These check control flow, not actual
in-flight CCU retirement. The inspected examples/tests do not establish
missing-trigger/done recovery followed by new-process real-data transfer on
the same cards.
Runtime's [abort example][rt-example] uses a local LongOP, not CCU or a peer
window; its TS/driver UTs use mocks. It does not close this coverage gap.

[hccl-version]: https://gitcode.com/cann/hccl/blob/348cb971a64bd1f8ce09fad9b244f9aa68c00802/README.md#L69
[hcomm-write]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/docs/zh/api_ref/comm_opdev/data_plane_api/ccu/data_movement/Write.md#L79
[hccl-kernel]: https://gitcode.com/cann/hccl/blob/348cb971a64bd1f8ce09fad9b244f9aa68c00802/examples/05_custom_ops_allgather/ccu/op_kernel_ccu/ccu_kernel.cc#L155
[hccl-example]: https://gitcode.com/cann/hccl/blob/348cb971a64bd1f8ce09fad9b244f9aa68c00802/examples/05_custom_ops_allgather/testcase/main.cc#L143
[pto-common]: https://gitcode.com/cann/pto-isa/blob/79403e46b48913c687ebe80464a170b150b1ffca/include/pto/comm/async/ccu/ccu_mesh_common.hpp#L196
[pto-write]: https://gitcode.com/cann/pto-isa/blob/79403e46b48913c687ebe80464a170b150b1ffca/include/pto/comm/async/ccu/ccu_broadcast_kernel.hpp#L86
[pto-test]: https://gitcode.com/cann/pto-isa/blob/79403e46b48913c687ebe80464a170b150b1ffca/tests/npu/a5/comm/st/testcase/tbroadcast_ccu/main.cc#L358
[pto-runner]: https://gitcode.com/cann/pto-isa/blob/79403e46b48913c687ebe80464a170b150b1ffca/tests/script/run_st.py#L219
[rt-stream-abort]: https://gitcode.com/cann/runtime/blob/09c468b359b0e393f15436d7e1fc8eb9a77bcab0/docs/zh/api_ref/06_stream_management.md#L786
[rt-query]: https://gitcode.com/cann/runtime/blob/09c468b359b0e393f15436d7e1fc8eb9a77bcab0/docs/zh/api_ref/06_stream_management.md#L527
[rt-stream-code]: https://gitcode.com/cann/runtime/blob/09c468b359b0e393f15436d7e1fc8eb9a77bcab0/src/runtime/core/src/stream/stream.cc#L1156
[rt-device-code]: https://gitcode.com/cann/runtime/blob/09c468b359b0e393f15436d7e1fc8eb9a77bcab0/src/runtime/core/src/dfx/fast_recover.cc#L233
[rt-device-doc]: https://gitcode.com/cann/runtime/blob/09c468b359b0e393f15436d7e1fc8eb9a77bcab0/docs/zh/api_ref/13_exception_handling.md#L878
[hcomm-abort]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/coll_communicator_mgr/dfx/ns_recovery/task_abort_handler.cc#L51
[hcomm-kill]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_comp.cc#L1394
[hcomm-jetty]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/base_comm/resources/hccp/rdma_service/ctx/rs_ub_jetty.c#L133
[hcomm-jetty-destroy]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/base_comm/resources/hccp/rdma_service/ctx/rs_ub.c#L1889
[hccl-resume-test]: https://gitcode.com/cann/hccl/blob/348cb971a64bd1f8ce09fad9b244f9aa68c00802/test/ut/comm_negotiation/comm_negotiation_stub.cc#L225
[driver-rpc]: https://gitcode.com/cann/driver/blob/d04bdcb1fb1c5d5b2c3188a1fa8da8372de8771e/src/sdk_driver/trsdrv/trs/trs_core/trs_fops.c#L673
[rt-example]: https://gitcode.com/cann/runtime/blob/6a40e4d07c011ceaf14116b7ac18a6a109b71edc/example/1_basic_features/stream/1_stream_with_failure_mode/main.cpp#L102
[hcomm-esched]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/base_comm/resources/hccp/rdma_service/ctx/rs_esched.c#L24
[hcomm-destroy]: https://gitcode.com/cann/hcomm/blob/95dd5a3f4dd78d05b32fa7a679b8d16f5b3f2cea/src/base_comm/resources/ccu/ccu_instance/ccu_instance_mgr.cc#L93
[hcomm-res-pack]: https://gitcode.com/cann/hcomm/blob/95dd5a3f4dd78d05b32fa7a679b8d16f5b3f2cea/src/base_comm/resources/ccu/ccu_instance/ccu_res_pack.cc#L18
[pto-addresses]: https://gitcode.com/cann/pto-isa/blob/79403e46b48913c687ebe80464a170b150b1ffca/include/pto/comm/async/ccu/ccu_mesh_common.hpp#L102
