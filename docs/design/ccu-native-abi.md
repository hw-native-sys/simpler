# Native CCU module and three-Event ABI

ABI V2 packages native HCOMM CCU definitions with a `ChipCallable`. The module
owns the algorithm and its control flow. Simpler validates exact module bytes
and provides domain resources and asynchronous completion through later runtime
layers. The public C/POD layouts are in
`src/common/platform_comm/ccu/ccu_native_abi.h`.

## Version and packaging

V2 is incompatible with V1. A module exports:

```c
const SimplerCcuModuleManifestV2 *simpler_ccu_module_manifest_v2(void);
```

The entry must not throw and returns an immutable contiguous manifest. Kernel,
logical instance and string tables use byte offsets from its base. All reserved
fields and flags must be zero unless an explicit capability is required.
Kernel records require `SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS` and V2 registration
and task layouts. Group selection is deferred; all declared kernels are prepared.

The package envelope remains version 1 because its byte layout is unchanged;
each module record now requires ABI version 2. An old package record, V1-only
export, old manifest version, invalid layout, unknown capability or mismatched
SDK/PTO-ISA build identity is rejected before resource preparation. Packaging
preserves opaque bytes; it does not inspect ELF exports. The native loader
performs manifest validation when opening the module.

The loader opens exact ELF bytes through a private unlinked file with
`RTLD_NOW | RTLD_LOCAL`. Cache identity includes those bytes, the complete
manifest, native entry names and both build identities.
The backing file descriptor stays alive until after `dlclose`, preventing its
`/proc/self/fd` path from being reused by another live module. Cache lookup
currently loads and validates a candidate first, including its returned manifest;
it shares validated module ownership rather than avoiding all loader work.

## Kernel definitions and logical instances

`SimplerCcuKernelRecordV2` describes a kernel's native entry, parameter size and
alignment, and supported layouts. Kernel IDs must be unique within a module.
The entry name must be a valid C symbol exported by that module.

`SimplerCcuMissionInstanceV2` separately binds each logical `mission_index` to
a kernel ID. Its index must equal its table position, starting at zero without
gaps. Multiple instances may reference one kernel; every kernel requires at
least one instance. There is no implicit default instance: a single-instance
module explicitly declares index 0. Instances require independent mutable
parameters, Events and execution resources.

For a callable with multiple modules, the domain concatenates instance tables
in package order. Its global mission index is the module's prefix instance
count plus the module-local index. This mapping is fixed for that callable and
does not depend on SDK registration order or hardware mission IDs. The runtime
checks aggregate limits before allocating resources. Module-local kernel IDs
are interpreted together with the descriptor's `module_index`.

## Registration and launch

The native definition signature is:

```c
int32_t entry(const SimplerCcuRegistrationBindingV2 *binding);
```

It must not throw. Query and registration receive the same one-element argument
array `{&binding}`. The Host-only binding identifies the kernel, domain-global
mission index and rank, with a channel and three distinct opaque Event handles:
`ready_event_handle`, `trigger_event_handle`, `done_event_handle`. These handles
are SDK values, not device addresses. Each Event uses bit 0.

`SimplerCcuLaunchArgumentsV2` contains four 64-bit words, passed directly to
`HcommCcuKernelLaunch`:

| Word | Value |
| ---- | ----- |
| 0 | `layout_version` = 2 |
| 1 | `parameter_addr` |
| 2 | `parameter_token` |
| 3 | `reserved` = 0 |

The native definition loads all four words; `HcommCcuGetTaskArgsNum` must report
4. The address and access token are per-run values for the kernel's declared
parameter region, and must not be captured in cached registration. Ready,
trigger and done are bound through registration. There are no run-control or
completion-slot launch words.

## Domain workspace

The domain runtime context publishes `SimplerCcuWorkspaceV2`: ABI/layout version,
domain identity, run generation, mission count, descriptor stride and device
address of a dense `SimplerCcuMissionDescriptorV2` array. It is a read-only view
for the duration of the run. Each descriptor supplies its global mission index,
module/kernel identity, matching run generation, parameter address/size/alignment
and token, and three `{address, mask}` Event descriptors. Masks equal
`SIMPLER_CCU_EVENT_MASK` (1). Addresses are opaque platform mappings; register
encoding, cache maintenance and clear confirmation belong in platform helpers.

The runtime validates versions, sizes, domain/run identity, indexes, parameter
ranges/alignment and distinct Event mappings before admitting submissions.
The AIV looks up `missions[mission_index]` through this workspace. Orchestration
does not pass individual resource addresses in business arguments.

## Submission and completion

1. Host safely resets Events and launches the missions. After required
   initialization, CCU records ready. AICPU observes, clears and confirms ready
   before admitting ordinary tasks.
2. AIV resolves its mission and registers its task's done completion condition
   before writing parameters. It publishes parameter/input visibility, records
   trigger with the required synchronization, and returns immediately.
3. CCU waits for and consumes trigger, reads parameters, executes the algorithm
   and completes data movement and required cross-rank synchronization before
   publishing done with output visibility.
4. The existing asynchronous task mechanism waits for both ordinary execution
   completion and done. The completion adapter clears done and confirms the
   clear before retiring the condition and releasing dependent tasks.

There is no consumed Event, acknowledgement wait, GM completion sequence,
mandatory stop/final-wake protocol, or CCU-published EXITED state. Trigger means
submission has been issued; it does not prove CCU has consumed the parameters.
Task IDs, owner generations and error reporting belong to the runtime, not to
a CCU-written completion slot. Event bits do not encode native error codes.

Users serialize operations on one mission through the entire asynchronous
task's completion, including before the next parameter write. Parameters and
referenced data stay alive until then. Waiting only for AIV return is
insufficient. Repeated submission to the same mission within one AIV is outside
this contract. Independent missions may execute concurrently. The initial
implementation provides no owner/busy detection or queue; an atomic owner claim
before parameter writes is a separate extension.

## Exit and failure

The user's finite kernel control flow returns normally. Host confirms termination
using the mission's dedicated stream query or bounded synchronization. Final done
does not prove stream termination. The runtime rejects overlapping runs sharing
mutable resources and reuses them only after confirmed termination.

Startup, completion and final stream waits have bounded failure handling through
runtime/SDK error channels. A failure may terminate the whole run. If stopping
cannot be confirmed, submissions remain disabled and module, Event, parameter,
window, token and stream resources remain alive. Neither a timeout nor a clear
attempt authorizes resource reuse. These runtime obligations and hardware
acceptance are implemented in the dependent runtime changes, not in this ABI.
