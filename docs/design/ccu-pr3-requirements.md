# Embedded CCU callable and domain integration

The requirements and implementation status are also available in [Chinese](../zh-cn/design/ccu-pr3-requirements.md).
This page describes the current integration candidate. Runtime and hardware validation are reported separately;
passing the earlier component tests does not validate the full path.

## Public contract

`CcuFunction.build(binary, entry)` embeds an ELF64 host builder in a normal
`ChipCallable`. Its list position is the `ccu_id`. The module exports
`uint32_t simpler_ccu_module_abi_version()` returning 2, and an entry with the
signature `int(CcuBuildContext*)`. Tensors and scalars use positional indices.
Legacy module ABI 1 must be rebuilt. The independent internal `Builder/Program`
test component retains ABI 1 and does not load production modules.

A domain must select `workspace_engines=("ccu",)` and have exactly two members
on A5 using `tensormap_and_ringbuffer`. Task arguments carry a live domain context
and domain-owned tensors. Simulators, other architectures/runtimes and direct L2
CCU submissions remain unsupported. Caller-provided program keys, platform
handles and callable IDs are not part of the public interface.

## Preparation and execution

Domain creation or reuse accepts `ccu_callables=(registered_handle, ...)` and
prepares only these callables in stable digest order before the callback's first
task submission. The default empty list prepares none; an unused registered
builder cannot fail another callable's domain preparation. Reuse can add selected
callables without changing the domain's identity. Cached instances need no new
control operation; submitting an unprepared callable is rejected. Each callable/domain pair
has its own Session, communicator root information, module ownership and memory
registration. Preparation has one initialization cost per selected pair. Both
ranks verify the domain generation and hidden native callable ID.

The domain owner retains its VMM window, Session and uploaded publication images.
The PR2 Workspace points to the PR1 CcuDeviceContext; the real request record is
owned by PR1. Workspace/context ABI is 2. ChipCallable wire format remains 1,
with its embedded module ABI field independently set to 2. Rebuild the Python
extension, host runtime, modules and device kernels together.

Native preparation acquires a run lease. After ordinary producers stop, PR2
polls or drains the CCU request before releasing its dependencies and owners.
A preparation proven never launched cancels only its own run reference.
Unconfirmed readers and partially created objects stay owned until SDK cleanup
is confirmed. Domain close stops CCU before freeing windows and parent
communication resources. A closed Session alone does not authorize device reset:
the parent CommHandle must also be fully released. Recovery skips device-wide
synchronization while CCU communication owners remain. Fatal finalization stops
Host collectors first, then retains the context and runtime if readers, service
shutdown or parent communication cleanup remain unconfirmed. An error code is
not proof that a producer stopped. The ordinary non-CCU path is unchanged.
Release a domain before unregistering its callable. Preparation in progress,
partially created instances and published instances all prevent hidden
callable-slot reuse until domain reclamation succeeds. Parameter or late-control
admission refusals leave an existing domain and its preparation set unchanged.

Preparation fanout joins all members before rollback. Failed creation and
failed preparation of a newly registered callable in a reused domain both
retire through the existing fence-owned domain ledger. Failed cleanup remains
visible and prevents subsequent work.

## Validation

The [two-rank end-to-end scene](../../tests/st/a5/tensormap_and_ringbuffer/ccu_deferred_completion/README.md)
loads two positional entries, returns from AIV after asynchronous submission,
and validates dependent consumers over four or 1000 changing-address rounds.
It checks data, destination guards, unchanged source/scratch and domain cleanup.
This test requires the combined three-PR build and real A5 hardware.

Focused component tests remain under `tests/ut/cpp/common/platform_comm/ccu/`
and `tests/ut/py/test_ccu_runtime.py`. The PR1 standalone onboard fixture lives
under `tests/ut/cpp/a5/platform/support/ccu_session_onboard.cpp`; it is useful
for platform diagnosis but does not replace Worker/TMR end-to-end validation.

### Extended validation and failure handling (2026-11-08)

The scene now contains eight hardware cases: four-round and 1000-round execution,
invalid/cross-domain arguments, retained-domain reuse and recreation, two
callables both using ID 0, one-rank builder rejection and recovery, a forced
post-AIV-FIN pending request, and a confirmed failed request with domain cleanup.
The [scene README](../../tests/st/a5/tensormap_and_ringbuffer/ccu_deferred_completion/README.md)
lists each predicate and the queued test command.

The pending test uses a full-width sync-start cohort to establish that the
sender AIV has returned before observing the real request still pending.
Witness and consumer result slices do not overlap. The failed-request test
captures the exact failed task and successor identities, requires a complete
scan with a published successor, and checks that no successor block was
dispatched. It does not infer device behavior from an unchanged Host buffer.

Fault testing exposed two distinct cleanup problems. Returning confirmed CCU
error 105 as an SDK kernel failure prevented independent completion proof.
The native completion owner now preserves the Error/-105 terminal and payload,
but returns SDK success only with explicit AICore retirement proof, no other
callback error and successful publication. Host reports the business failure
and withholds outputs after actual stream completion and CCU drain. An early
Host observation wait could then skip result publication and stop the child
control loop. When native destruction and caller last use are proven, that
path now publishes the original exception and keeps domain cleanup available.
Neither fix treats an error record as proof of quiescence; error 106, unknown
readers, failed retirement/publication and unresolved parent communication
ownership retain their existing refusal rules.

| Validation | Evidence |
| ---------- | -------- |
| CPU and Python | Eight CTest groups and 38 focused Python tests passed; the final candidate's C++ code matches `9fd5b9cc` |
| Final hardware run | `e8d6081b5264b87a879a7f6faa883d6468950e11`, task `task_20261108_173610_121532929008`, devices 2 and 3: all eight cases PASS, exit 0 |
| Logs | `extended-e8d6081b-full-20261108-173610/` |

Previously, `68e00c466` passed four and 1000 rounds through Worker/TMR, including
data, guards and normal shutdown. Tensor descriptors are read field by field
from GM, and device loops do not prepare resources on the Host each round.
The seven non-FAILED cases passed on `9fd5b9cc` in 87.36 seconds; its remaining
cleanup failure motivated the early-wait fix and is not full-suite acceptance.
PR1 standalone results remain separate component evidence.

The final hardware run accepts the eight tested predicates, including retained
domain reuse/recreation, cross-domain and callable isolation, rank-local
preparation rollback, post-FIN pending completion, and confirmed error cleanup.
Both FAILED-case ranks reported a complete successor scan with no dispatched
blocks; domain reclamation and normal Worker close also passed. Case durations
are listed in the scene README. Forced SDK close failures, arbitrary hardware
loss, more than two ranks and concurrent requests within one Session remain
outside this evidence.
Unconfirmed cleanup must continue retaining the affected owners.
