# Communication-Domain Scene Tests

Cross-architecture regression tests for communication-domain orchestration and
deferred completion under the `tensormap_and_ringbuffer` runtime. Host Fabric
mapping tests also cover `host_build_graph`; their SDMA cases remain TMR-only.

| Test | Platforms | Contract under test |
| ---- | --------- | ------------------- |
| [`test_unified_domain.py`](test_unified_domain.py) | L3: a2a3sim, a5sim, a2a3, a5; L4: a2a3sim, a5sim | Cached/explicit L3 lookup and exact L4 reuse across runs, with deferred release until the submitted chip kernel completes. |
| [`test_domain_tensor_remote.py`](test_domain_tensor_remote.py) | a2a3sim, a5sim | Local/TCP logical views, cached reuse, revocation and explicit peer context. |
| [`Network1 domain tensors`](../../a2a3/tensormap_and_ringbuffer/l4_network1/test_domain_tensor_network1.py) | a2a3 | The same kernel contract on Fabric-local, TCP and MPI topologies. |
| [`test_host_domain.py`](test_host_domain.py) | a2a3sim, a2a3 | HBG/TMR L2/L3 native and mapped access, restricted grants, unmap/remap, accepted-reader retirement and BF16 sparse rows; TMR-only onboard SDMA completion. |
| [`async_notify/`](async_notify/) | a2a3, a5 | A notification-counter producer releases its dependent consumers only after deferred completion. |
| [`deferred_notify/`](deferred_notify/) | a2a3, a2a3sim, a5sim | A peer notification publishes the remote rank's mailbox contents before the consumer reads them. |

These are runtime-mechanism tests rather than public communication-API
tutorials. For a direct `Worker` communication-domain walkthrough, see
[`examples/workers/l3/allreduce/`](../../../../examples/workers/l3/allreduce/).

SDMA and URMA completion tests remain architecture-local because their
workspace provisioning and availability gates differ.

Host per-PR CI keeps four mechanism scenes (two L2 and two L3), plus two
isolated onboard SDMA directions. Use `--manual only` for the five extended
write/BF16/runtime qualification cases, or `--manual include` for full acceptance.
Grant, context, stale-reference and failure-retention unit tests remain in CI.
