# Global TLOAD mpirun L3 example

This example runs the same Global CommDomain peer-`TLOAD` workload as
[`global_tload_mixed_l3`](../global_tload_mixed_l3/), but brings the two L3
workers up through a single parent-owned `mpirun` instead of a TCP daemon:

```text
L4 parent on machine A ─ mpirun -np 2 ─┬─ L3 rank 0 on machine A → its NPUs
                                       └─ L3 rank 1 on machine B → its NPUs
```

The kernels, the chip callable, and the rank-side orchestration are imported
from `global_tload_mixed_l3` — this example ships no kernels of its own. What
it exercises that the sibling cannot:

- `add_mpirun_worker_group` / `MpiL3GroupSpec`: rank 0 must land on the parent
  machine (only it can read the parent-written group manifest, which it then
  broadcasts over MPI), and rank 0 marks the group's named shared-memory
  mailbox READY once every rank reports in over MPI — task and control
  traffic then flows through that mailbox instead of per-rank TCP sessions.
- The full-group MPI descriptor exchange: the Global CommDomain members cover
  every device of both ranks, so descriptors are exchanged rank-side over an
  MPI collective and the L4 import fanout is skipped.
- One TLOAD task per NPU device on each machine (the sibling drives one device
  per side), so the domain spans four windows on the network1 pair.
- One `submit_next_level_group` round over the full MPI group, so the batched
  `PER_RANK` mailbox envelope path runs alongside the directed dispatches.

## Prerequisites beyond the sibling example

- `mpirun` and `mpi4py` on **both** machines, built against the **same** MPI
  implementation (MPICH's Hydra and Open MPI both work; a `mpi4py` linked
  against a different MPI than the `mpirun` that launches it degrades every
  rank into an isolated single-process world).
- `mpirun` executes one identical command line on both machines, so `--python`
  must name an interpreter path valid on BOTH. Point it at a per-machine
  launcher script installed at one shared absolute path; each machine's copy
  sources the CANN environment, enters that machine's copy of this source
  tree (the rank imports this package's orchestration module by name), and
  execs that machine's `.venv` python:

```bash
cat > /tmp/simpler-mpi-python <<'EOF'
#!/usr/bin/env bash
source /usr/local/Ascend/cann/set_env.sh
cd /path/to/this/machines/simpler
exec /path/to/this/machines/simpler/.venv/bin/python "$@"
EOF
chmod +x /tmp/simpler-mpi-python
```

## Run on the L4 parent

`192.0.2.10` / `192.0.2.20` are documentation placeholders. Hosts must be
numeric IPs; the local host is where rank 0 lands:

```bash
source .venv/bin/activate
python examples/workers/l4/global_tload_mpirun_l3/main.py \
  --local-host 192.0.2.10 --remote-host 192.0.2.20 \
  --python /tmp/simpler-mpi-python \
  --local-devices 0,1 --remote-devices 0,1
```

Expected output ends with:

```text
[global-tload-mpirun-l3] domain_rank=3 node=1 device_index=1 max_diff=0.000e+00
global_tload_mpirun_l3 passed
global_tload_mpirun_l3 cleanup ledger: worker 0 command=not_observed process=exited(0) cleanup=succeeded@rank0
```

The ledger line comes after `passed` because `close()` is what produces it. It
is this example's second assertion, and the only one on hardware that carries a
release record end to end: rank 0 writes it beside the group manifest and L4
reads it back after reaping `mpirun`. The check requires *an* answer on both
counts — `unknown` fails it — but neither a `succeeded` release nor an
`observed` command. A release that genuinely failed is a defect in whatever
failed rather than in the ledger that reported it, and `command` reads
`not_observed` on this route because no shutdown command is delivered to an
mpirun group: the mailbox SHUTDOWN opcode is sent only from
`RemoteL3Endpoint::request_progress_stop()`, which a healthy close never calls,
so the group is reaped by SIGTERM instead.

Any `max_diff` above the tolerance exits non-zero.

## Running it in CI

The network1 job's `network1-stage` action writes the per-machine launcher on both
machines at one shared path and exports it as `NETWORK1_MPI_PYTHON`; the
`test_global_tload_mpirun_l3.py` wrapper reads it together with
`NETWORK1_LOCAL_IP` and the standard network1 fixtures, and skips when `mpirun`,
`mpi4py`, or either variable is absent.
