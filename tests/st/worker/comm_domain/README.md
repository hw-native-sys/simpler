# Communication-Domain Scene Tests

Cross-architecture regression tests for communication-domain orchestration and
deferred completion under the `tensormap_and_ringbuffer` runtime.

| Test | Platforms | Contract under test |
| ---- | --------- | ------------------- |
| [`async_notify/`](async_notify/) | a2a3, a5 | A notification-counter producer releases its dependent consumers only after deferred completion. |
| [`deferred_notify/`](deferred_notify/) | a2a3, a2a3sim, a5sim | A peer notification publishes the remote rank's mailbox contents before the consumer reads them. |

These are runtime-mechanism tests rather than public communication-API
tutorials. For a direct `Worker` communication-domain walkthrough, see
[`examples/workers/l3/allreduce/`](../../../../examples/workers/l3/allreduce/).

SDMA and URMA completion tests remain architecture-local because their
workspace provisioning and availability gates differ.
