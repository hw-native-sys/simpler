# A5 CCU Broadcast triggered by a simpler AIV task

Two A5 ranks run Broadcast over a UB channel. The Host registers and launches
a gated HCOMM CCU program. A normal simpler `ChipWorker` submits an AICPU
orchestration, which dispatches an AIV task calling
`TBROADCAST<CollEngine::CCU>`. The Host waits for the CCU communication stream
before reading results or releasing resources.

## Requirements

- Linux and two Ascend 950 devices with a `COMM_PROTOCOL_UBC_CTP` rank-graph link.
- The worktree's own `.venv`, an installed simpler, and prebuilt
  `a5/tensormap_and_ringbuffer` runtime binaries.
- A configured CANN 9.3 SDK exposing the HCOMM CCU C/DSL interface:
  `HcommCcuKernelRegister`, `HcommCcuEventAlloc`, `HcommCcuEventGetAddr`,
  `HcommCcuKernelLaunch`, and `AscendC::ccu` primitives.

The Host records a small Broadcast program with HCOMM's public DSL. It uses
HCCL for communicator, topology, channel and stream resources. The AIV trigger
uses PTO at `pto_isa.pin`. The Host deliberately does not use the older PTO
C++ Host registration classes, which are absent from the CANN 9.3 public headers.
The reserved HCOMM event provides the same CKE both to the CCU program and the
AIV scalar arguments; no private Runtime address-query ABI is needed.

A device has two IO dies. The CCU program follows its UB channel's IO die, so
its external gate must be reserved on that same die. `HcommCcuKernelQueryResReq`
validates a small channel-only program against each die without registering or
launching it. The helper reserves the gate on the matching die and reports
`CCU_SETUP rank=... io_die=... protocol=UBC_CTP`. Reserving an arbitrary die's
CKE can leave CCU waiting even though the AIV task returned successfully.

## Run

Run the architecture precheck before occupying devices. From the repository root:

```bash
source /path/to/cann/set_env.sh
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --timeout 900 --max-time 900 --run \
  '.venv/bin/python examples/a5/tensormap_and_ringbuffer/ccu_broadcast_demo/test_ccu_broadcast_demo.py -p a5 -d "$TASK_DEVICE"'
```

`-d` expects two comma-separated device IDs. Adapt the allocated-device placeholder
to that machine's `task-submit --help` when necessary. Use a separate
`ASCEND_PROCESS_LOG_PATH` directory for the run's device logs.

The equivalent pytest invocation, also executed inside the device queue, is:

```bash
.venv/bin/python -m pytest \
  examples/a5/tensormap_and_ringbuffer/ccu_broadcast_demo \
  --platform a5 --device 0-1 -v
```

## Implementation and checks

| File | Responsibility |
| ---- | -------------- |
| `ccu_driver.py` | Isolate ranks in processes; exchange root info and memory metadata; coordinate phases |
| `host/ccu_session.cpp` | Acquire CCU resources; register HCOMM Broadcast; populate peer addresses/tokens; reserve a mapped CKE; wait and verify |
| `kernels/orchestration/ccu_broadcast_orch.cpp` | Pass five scalar arguments to the AIV task |
| `kernels/aiv/ccu_broadcast_trigger.cpp` | Trigger PTO CCU Broadcast and publish an independent AIV execution marker |

Defaults cover roots 0/1, 64/1024 float elements, and two repetitions: eight cases.
Each case uses fresh rank processes and a fresh CCU session. This exercises repeated
creation and release. Each CCU program waits for its own AIV gate and exchanges
ready/done notifications over the UB channel.

Each rank verifies its input is unchanged, its output matches the root's input,
the full window's guards and unused region are unchanged, and the AIV trigger marker
is present. Success prints `CCU_BROADCAST_PASS`. Unsupported SDKs, missing UB links,
errors and timeouts fail explicitly.

All data addresses and tokens are populated in the Host `uint64_t` task argument
array before `HcommCcuKernelLaunch`. Its argument count is the number of 64-bit
elements, not the number of bytes. AIV receives only CKE address, mask, marker address, root and rank.
The example does not create RDMA SQ/CQ completion conditions or alter the common
communication backend.

## Completion and lifetime

Completion uses Host-side CCU stream synchronization. An AIV return or consumed CKE
does not prove communication completion. No CCU deferred-completion backend is
registered, so a consumer of CCU output must not be added to the same task graph.
The helper refuses resource release without proven CCU completion. A control-phase
timeout reaps only this example's rank processes.

The AIV kernel reads its logical block index with `get_block_idx(args)` from
`intrinsic.h`. Native CANN `get_block_idx()` reads hardware topology state and
does not identify simpler's logical task block. The AIV marker is checked after
`ChipWorker.run` returns and before waiting on CCU, so a skipped trigger fails
directly rather than being reported only as a communication timeout.
