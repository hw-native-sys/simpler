# SPSC queue Region Template scene tests

Scene tests for `Orchestrator.create_spsc_queue`.

| Test | Topologies | Platforms |
| ---- | ---------- | --------- |
| [`test_l3_single_hop.py`](test_l3_single_hop.py) | L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_l4_two_hop.py`](test_l4_two_hop.py) | L4 → L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_child_l3_own_queue.py`](test_child_l3_own_queue.py) | L4 tree, child L3 creates | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_failure.py`](test_failure.py) | L3 construction and poison | `a2a3sim`, `a5sim` |
| [`test_native_identity.py`](test_native_identity.py) | L4 and child L3 native identity | `a2a3sim`, `a5sim` |

Creation uses `create_spsc_queue(initiator, peer, ...)`. The returned object
is an `SpscQueue`. `peer_task_arg_scalars()` is the ten-field `SPSQ` binding.
The child L3 case is driven by a real L4 tree and creates the queue in that
L3 worker's own registry.

## Contract under test

- Exact ten-scalar `SPSQ` binding, nonzero `transaction_id`, asymmetric arenas.
- Descriptor-ring exact-depth full before the peer starts, then empty after drain.
- Bidirectional variable-length DATA, zero-byte DATA, arena wrap, and padding
  retirement onto the output arena base.
- `ERROR` is a normal message: ERROR, DATA, ERROR stay live.
- One AIV-produced DATA output flushes before `publish`.
- After STOP, further input is rejected locally; one post-STOP DATA still drains.
- `queue.free()` is logical and rejects later enqueue. One L3 run frees
  explicitly; the next run does not, and both are reclaimed after drain.
  Each run gets a new transaction id. Mapping close is recorded before the
  payload and counter releases, once each.
- A positive-timeout empty peek does not poison the queue.
- Zero-transaction and wrong-magic peer construction fail without a trusted
  `SPSC queue endpoint error session=` marker. The kernel has no allocation
  identity to publish, so the host failure stays generic.
- A smashed descriptor is reported by the native helper. The host exception
  carries `SPSC queue endpoint error session=0x… transaction=…` for that
  queue only. L4 forwarding and a child L3's own queue are covered separately.
  The native log keeps the longer SPSQ detail. A stale output release is
  local poison. These fault cases do not stand in for hardware cache or
  HostVmmCopyAccess.

Success cases run on simulation and onboard platforms. Fault cases stay on
simulation.

## Run

```bash
source .venv/bin/activate
python -m pytest tests/st/worker/comm_region/templates/spsc_queue --platform a2a3sim
```
