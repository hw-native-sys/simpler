# sdma_async_completion_demo — deferred completion over SDMA

Two ranks, one transfer, one dependency:

```text
producer:  send_request_entry(async_ctx, SdmaTget(...)) reads the peer rank's
           input from the HCCL window into local `out` asynchronously;
           the adapter registers completion internally
consumer:  depends on the producer's output, writes result = out + 1
```

Checking `out` and `result` tests two separate things: `out` proves SDMA
completion polling saw the transfer land, `result` proves the deferred-release
dependency held the consumer until it had. A consumer that ran early would
still produce a plausible `result` from a partially written `out`, which is why
both are checked.

The remote address is plain symmetric-window arithmetic — take the local
pointer's offset from `windowsIn[rankId]` and add it to `windowsIn[peer_rank]`.
Every rank's window is laid out identically, so an offset is rank-independent.

## GET and PUT completion semantics

The [producer kernel](kernels/aiv/kernel_sdma_tget_async.cpp) uses
`send_request_entry(async_ctx, SdmaTget(...))`. The
[SDMA adapter](../../../../src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_kernel.h)
submits `TGET_ASYNC` and registers its event through
`pto2::detail::register_pto_async_event`, an internal adapter helper.

For successful, nonempty SDMA transfers in the repository's
[pinned PTO-ISA version](../../../../pto_isa.pin):

| Platform / operation | Transfer and completion | Overlap with later tasks |
| -------------------- | ----------------------- | ------------------------ |
| A5 GET (`SdmaTget` / `TGET_ASYNC<DmaEngine::SDMA>`) | Asynchronous SDMA transfer; the adapter registers post-done records for deferred completion. | The AIV can return while the transfer is pending; dependent consumers wait for completion. This is the path demonstrated here. |
| A5 PUT (`SdmaTput` / `TPUT_ASYNC<DmaEngine::SDMA>`) | Synchronous MTE2/MTE3 fallback: payload moves in chunks through the AIV's UBuffer. The transfer is complete when the call returns, with `AsyncEvent(0, DmaEngine::SDMA)`. | This transfer adds no deferred wait and cannot release the AIV while the copy is pending to overlap later tasks. |
| A2/A3 PUT (`SdmaTput` / `TPUT_ASYNC<DmaEngine::SDMA>`) | Real asynchronous SDMA PUT with an event registered for deferred completion. | Can release the AIV while the transfer is pending; the A5 synchronous fallback does not apply. |

In this `tensormap_and_ringbuffer` (TMR) demo, the adapter checks
`event.handle == 0` **inside the AIV, before the kernel returns**, and skips
condition registration for the completed PUT event. After normal kernel
completion, the [AICPU scheduler](../../../../src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp)
checks the published error and condition count. If the task has no errors and
no other registered conditions across its subtasks, it takes the ordinary
completion path once all subtasks finish. The AICPU consumes registered
conditions; it does not interpret the original PTO event's zero handle.

The completed zero-handle event described here is specific to a successful A5
synchronous PUT. It is not a general success rule for other operations or failed
events. The separate
[A5 HBG backend and lifetime contract](../../../../docs/zh-cn/design/a5-hbg-async-completion/backends-and-lifetime.md)
describes `host_build_graph`'s resident completion mechanism and requires an
unexpected zero handle from a nonempty GET to fail; the TMR AICPU sequence above
does not describe HBG.

See the pinned [PTO-ISA `TPUT_ASYNC` instruction documentation](https://github.com/hw-native-sys/pto-isa/blob/83b1b2b52ded9ec5d91e02532a57cd95e497ba86/docs/isa/comm/TPUT_ASYNC.md#sdma-path-on-a5)
for the A5 fallback and GET/PUT boundary.

## Requirements

The a5 host runtime includes the async-SDMA workspace by default:

| Gate | Effect |
| ---- | ------ |
| `CASES[*]["platforms"] = ["a5"]` | deselected on any other `--platform` |
| `CASES[*]["config"]["device_count"] = 2` | needs two dies |

The demo explicitly requests `workspace_engines=("sdma",)` and reads
`comm_workspace<CommWorkspaceKind::SDMA>(*ctx).addr`. It also runs in builds containing URMA or
RDMA; those providers do not replace its workspace.

```bash
pytest examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo \
  --platform a5 --device 0-1
```

Wrap the hardware run in `task-submit` on a shared box.

## Compare with

- [`../urma_deferred_completion_demo/`](../urma_deferred_completion_demo/) — the same protocol over URMA. `kernel_consumer.cpp` is byte-identical; only the transfer kernel, its completion header, and the build flag differ. The URMA example also includes a mixed SDMA/URMA DAG that runs both in one build and one domain.
- [`examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/`](../../../a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/) — the a2a3 port of this demo, which needs no overlay flag.
