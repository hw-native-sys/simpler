# Communication-Domain Scene Tests

Cross-architecture regression tests for communication-domain orchestration and
deferred completion under the `tensormap_and_ringbuffer` runtime.

| Test | Platforms | Contract under test |
| ---- | --------- | ------------------- |
| [`test_unified_domain.py`](test_unified_domain.py) | L3: a2a3sim, a5sim, a2a3, a5; L4: a2a3sim, a5sim | L3 cross-run use; shared L4 acceptance for cached Tensors, independent outputs and deferred release. |
| [`test_domain_tensor_remote.py`](test_domain_tensor_remote.py) | a2a3sim, a5sim | Rank data, canonical identity, nonzero views, exact chip, borrowed restrictions and stale references after same-name rebuild; domain inputs mix with independent remote outputs and release precedes wait. After rebuild, explicit ref/rank and local context drive the existing peer-TLOAD kernel, writing and reading a domain output. |
| [`test_domain_tensor_network1.py`](../../a2a3/tensormap_and_ringbuffer/l4_network1/test_domain_tensor_network1.py) | a2a3 / a3-fabric-v1 | Three thin local L4, cross-machine TCP and MPI group entries share the flow; the cached-Tensor run's root callback performs no lookup, and unified domain Tensors are also tested with peer context. |
| [`async_notify/`](async_notify/) | a2a3, a5 | A notification-counter producer releases its dependent consumers only after deferred completion. |
| [`deferred_notify/`](deferred_notify/) | a2a3, a2a3sim, a5sim | A peer notification publishes the remote rank's mailbox contents before the consumer reads them. |

These are runtime-mechanism tests rather than public communication-API
tutorials. For a direct `Worker` communication-domain walkthrough, see
[`examples/workers/l3/allreduce/`](../../../../examples/workers/l3/allreduce/).

SDMA and URMA completion tests remain architecture-local because their
workspace provisioning and availability gates differ.
