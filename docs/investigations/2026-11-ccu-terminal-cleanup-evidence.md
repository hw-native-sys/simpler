# CCU terminal cleanup: ownership and SDK boundaries

**Date**: 2026-11-16
**Updated**: 2026-11-17
**Decision**: simpler treats public SDK success as its cleanup boundary and
manages only its own allocations and handles. SDK-owned physical resources
remain under SDK lifetime management. Observable failures retain the affected
owners; hidden driver reclamation is not a prerequisite for simpler cleanup.

This investigation supports [issue #3143](https://github.com/hw-native-sys/simpler/issues/3143).
The source audit does not establish who occupied the driver in the reported
subsequent process. Hardware validation of the lifecycle fix is separate from
the source evidence below.

## What the local reproduction establishes

At simpler `bd5cd0f7c`, a CPU probe drives a `DomainCcuOwner` with a backend
reporting drained and an SDK reporting successful launch/abort and stream
completion. It launches one mission, calls `abort()`, calls `stop()`, and then
calls `owner.close()`:

```text
stream_complete=1 confirmed_stopped=1 admission_open=0 retained=1 owner_closed=0
```

The probe exits nonzero because close is refused. `finish_confirmed()` clears
the active-run state, but it does not clear the retained-resource state;
`DomainCcuOwner::drained()` also requires admission to remain open. A preceding
SDK error makes the path stricter still: `poll_terminal()` returns before
querying streams when the run has recorded a failure.

This proves the missing lifecycle transition. The fake SDK tests simpler's
response to successful and failed SDK operations; it does not test physical
driver reclamation. The agreed fix must separate permanent admission closure
from terminal cleanup and check each relevant public SDK result.

## Why close needs a separate protocol

The failure propagates through several owners:

```text
CCU failure -> retained run -> ChipWorker communication cleanup unconfirmed
                                |
Worker.close() -> Host mapping RECOVER -> quiesce_host_uses()
                                |
                     rejected before native drain
                                |
                domains, buffers and children remain owned
```

`ChipWorker::stop_and_cleanup_native_runs()` rejects an unconfirmed CCU cleanup
before its native drain. The A5 communication backend separately latches
`cleanup_failed`; its normal release/destroy paths and failure markers cannot
serve as a retryable terminal protocol. Python also caches domain release
outcomes to prevent replaying an ambiguous free. None of these states can be
reset merely because one mission subsequently reports complete.

A terminal protocol needs distinct, generation-bound facts:

1. Permanently close admission and confirm every native trigger producer and
   CCU mission has stopped, across all participating children and domains.
   Keep all backing alive until every participant acknowledges this state.
2. Finish SDK resource shutdown in ownership order, preserving partial progress
   for retry. Keep the backing until all participants confirm the necessary
   SDK shutdown facts.
3. Release storage and Host mappings, then finalize contexts and reap children.
   Preserve the original run failure separately from terminal cleanup results.

Assigned CCU instances belong to the communicator; the protocol must not
destroy them a second time. Unassigned staging instances retain their separate
owner.

The user clarified the ownership contract: default C++ programming wrappers
allocate virtual resources, and their destructors do not release physical CCU
resources. Physical resources are managed with the CCU instance lifetime.
The inspected [registration implementation][hcomm-register] first executes
`BuildKernel` (the Host callback records operations/resource requirements),
then calls `AllocRes` and saves the kernel under the instance. This Host
callback returning is not completion of a subsequently launched device task.
The [instance destructor][hcomm-instance] unregisters kernels and releases its
resource pack. This establishes the reclamation owner and excludes wrapper
scope exit as a physical-resource release event. It does not establish that
instance destruction cancels and drains an executing mission, nor does it
transfer ownership of simpler's separately allocated data/argument buffers.

An error delivered to L3, an empty lane FIFO, and `normal_done=1` in a task log
are not whole-run physical exit proofs. The existing runtime poll/wait C APIs
also perform retirement and can release storage. PREPARE uses `simpler_query_run_completion` to
observe physical completion through `DeviceRunner::poll_execution()` while
retaining the carrier and its backing. `NativeRunEntry::REAPED` is positive physical completion
evidence; `PREPARED` is not proof that a partial launch submitted nothing.
Never re-enter a carrier whose destruction is unknown.

## SDK source evidence

The [source audit](2026-11-ccu-sdk-source-audit.md) records the repository revisions, public API contracts, driver paths and evidence limits.

## Remaining implementation boundary

The user requested further source-level verification before escalating semantic
questions to SDK teams. The five primary repository locations are known:
[PTO-ISA](https://gitcode.com/cann/pto-isa),
[HCOMM](https://gitcode.com/cann/hcomm),
[HCCL](https://gitcode.com/cann/hccl),
[driver](https://gitcode.com/cann/driver), and
[Runtime](https://gitcode.com/cann/runtime).

The user subsequently confirmed that the following two implementations are
not open source. The source-location search is therefore closed:

1. `libccu-user-drv.so`: implementations of `ccu_tlv_request` and
   `ccu_custom_channel`, especially `CCU_U_OP_SET_TASKKILL`. HCOMM contains the
   declarations, dynamic loader and test substitutes, which are distinct from
   the production implementation.
2. The TS handlers for `OP_ABORT_STREAM`/`OP_ABORT_APP`, including creation of
   `DAVID_ABORT_TERMINATE_SUCC` and the TS side of the HCCP kill-message exchange.
   The inspected driver forwards requests and contains the host `ts_agent`;
   finding those components does not identify the terminal-state producer.

The user explicitly selected the public SDK boundary after reviewing the
error-propagation findings: successful SDK calls permit simpler cleanup;
physical-resource reclamation belongs to the SDK. This closes the design
decision. The source limitations remain documented facts, not a requirement to
inspect driver state, double-destroy assigned instances, or release physical
resources outside the public ownership APIs.

Implement the terminal protocol with tests
for partial launch, all-rank acknowledgement, original error preservation,
lost replies, partial teardown, and abort/query/destroy failures. A5 acceptance
must prove injection occurred, old L2 processes exited, and new L2 processes
transferred real data on the same pair in both fault orders. Normal multi-run
reuse and overlapping domains remain required regressions. A visible failed
stop/query/destroy or missing participant acknowledgement must still be
reported as unconfirmed cleanup.

[hcomm-register]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/base_comm/resources/ccu/ccu_kernel/ccu_kernel_mgr.cc#L186

[hcomm-instance]: https://gitcode.com/cann/hcomm/blob/8ca847d6d6444e4daec098754b037a9961846292/src/base_comm/resources/ccu/ccu_instance/ccu_instance.cc#L35
