# SPSC queue Region Template scene tests

Private L3→L2 ST for the bound SPSC queue. This directory is the formal Queue
Acceptance record. The example under
`examples/workers/l3/worker_chip_message_queue/` remains a smoke path.

| Test | Topologies | Platforms |
| ---- | ---------- | --------- |
| [`test_l3_single_hop.py`](test_l3_single_hop.py) | L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_failure.py`](test_failure.py) | L3 → L2 construction and poison | `a2a3sim`, `a5sim` |

Each success case uses `Orchestrator.create_worker_chip_queue`. The factory
binds the SPSC template; it does not call `create_worker_chip_region()`.

## Contract under test

- Exact ten-scalar `SPSQ` binding, nonzero `transaction_id`, asymmetric arenas.
- Descriptor-ring exact-depth full before the peer starts, then empty after drain.
- Bidirectional variable-length DATA, zero-byte DATA, arena wrap, and padding
  retirement onto the output arena base.
- `ERROR` is a normal message: ERROR, DATA, ERROR stay live.
- One AIV-produced DATA output flushes before `publish`.
- After STOP, further input is rejected locally; one post-STOP DATA still drains.
- `queue.free()` is logical and rejects later enqueue.
- Sim-only faults: positive-timeout empty peek, zero-transaction and wrong-magic
  peer construction (run fails without a trusted `transaction=` marker on the
  host exception), smashed descriptor (host observes remote abort), and stale
  output release (local poison). Peer fatal strings live on the AICPU log and
  are not Python exception text. Those cases do not stand in for hardware cache
  or HostVmmCopyAccess.

A2/A3 onboard covers HostVmmCopyAccess through queue payload copies. A5 is
CI-only. Fault cases stay on sim.

## Run

```bash
source .venv/bin/activate
python -m pytest tests/st/worker/comm_region/templates/spsc_queue --platform a2a3sim
```
