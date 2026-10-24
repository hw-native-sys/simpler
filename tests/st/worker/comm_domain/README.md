# Communication-Domain Scene Tests

Cross-architecture regression tests for communication-domain orchestration and
deferred completion under the `tensormap_and_ringbuffer` runtime.

| Test | Platforms | Contract under test |
| ---- | --------- | ------------------- |
| [`test_unified_domain.py`](test_unified_domain.py) | L3: a2a3sim, a5sim, a2a3, a5; L4: a2a3sim, a5sim | L3 cross-run use; shared L4 acceptance for cached Tensors, independent outputs and deferred release. |
| [`test_domain_tensor_remote.py`](test_domain_tensor_remote.py) | a2a3sim, a5sim | Shared checks for rank data, identity, nonzero views, exact chip, borrowed restrictions and stale references after rebuild; TCP uses independent Remote output Buffers. |
| [`test_domain_tensor_network1.py`](../../a2a3/tensormap_and_ringbuffer/l4_network1/test_domain_tensor_network1.py) | a2a3 / a3-fabric-v1 | The same flow on local L4, two-machine TCP and MPI group in existing Network1 CI; the second cached run's root callback does not look up the domain. |
| [`async_notify/`](async_notify/) | a2a3, a5 | A notification-counter producer releases its dependent consumers only after deferred completion. |
| [`deferred_notify/`](deferred_notify/) | a2a3, a2a3sim, a5sim | A peer notification publishes the remote rank's mailbox contents before the consumer reads them. |

These are runtime-mechanism tests rather than public communication-API
tutorials. For a direct `Worker` communication-domain walkthrough, see
[`examples/workers/l3/allreduce/`](../../../../examples/workers/l3/allreduce/).

SDMA and URMA completion tests remain architecture-local because their
workspace provisioning and availability gates differ.
