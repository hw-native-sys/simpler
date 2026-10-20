# Recursive single-owner scene tests

Cross-architecture scene tests for a delegated region created with
`Orchestrator.create_region`. This directory is not an example.

| Test | Topologies | Platforms |
| ---- | ---------- | --------- |
| [`test_l3_single_hop.py`](test_l3_single_hop.py) | L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_l4_two_hop.py`](test_l4_two_hop.py) | L4 → L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_child_l3_own_region.py`](test_child_l3_own_region.py) | L4 tree, child L3 creates | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_fault_injection.py`](test_fault_injection.py) | L3/L4 control-plane faults | `a2a3sim`, `a5sim` |
| [`test_native_identity.py`](test_native_identity.py) | native helper identity | `a2a3sim`, `a5sim` |

L3, L4, and the child L3 all call `create_region` inside the owning orch
callback. The child L3 case is driven by a real L4 tree and creates the region
in that L3 worker's own registry. L4 forwarding keeps scalar order, including
the sentinel values around the eight-field descriptor and the application
arguments.

## Contract under test

- PAYLOAD and COUNTER are independent allocations; host import/close ordering is
  `import/import/close/close/release`.
- Host payload write/read and counter notify/wait.
- Provider-local views reach a real AICPU task through existing TaskArgs
  descriptor scalars.
- Release/backend cleanup is one-shot.
- The same Worker session runs two complete lifecycles; transaction IDs are
  monotonic and each run allocates a fresh provider resource.
- The first lifecycle does not call `free()` and is still reclaimed when the
  run drains. The second calls `free()` while the chip may still be finishing;
  later host access is rejected, and physical close still happens after drain.
- Mapping close is recorded before each backing's provider release. Payload
  and counter are each released once.
- A native view failure publishes `Region endpoint error session=0x… transaction=…`
  and poisons only the matching allocation. A decode that has no trusted
  transaction publishes `identity=untrusted` and does not guess an owner.
  An application sequence mismatch stays a generic fatal with no short marker.
  The detailed `op` / `kind` / counter fields stay in the native log.

A2/A3 onboard covers the cache-maintenance call path. A5 is CI-only. Fault
cases stay on sim. They cover reply loss, malformed replies, timeout, child
death, and the native identity path. They do not inject those faults on
onboard hardware.

## Run

```bash
source .venv/bin/activate
python -m pytest tests/st/worker/comm_region/recursive_single_owner --platform a2a3sim
```
