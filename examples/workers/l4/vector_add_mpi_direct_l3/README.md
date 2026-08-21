# L4 direct MPI L3 vector add

This example launches one static MPI world across two machines: rank 0 owns
L4, rank 1 owns a real L3 on the L4 machine, and rank 2 owns a real L3 on the
peer. L4 sends SLR3 task and control frames directly to both L3 ranks, and
each L3 dispatches to its local L2 workers.

```text
machine A                                      machine B

MPI rank 0 / L4 -------- direct MPI --------> MPI rank 2 / real L3
       |
       +------------- direct MPI -----------> MPI rank 1 / real L3
```

The example validates direct MPI startup and teardown, remote memory copy,
L3-to-L2 vector-add execution, result checking, and Global CommDomain
lifecycle. The vector-add kernel does not perform a peer `TLOAD`.

## Prerequisites beyond the sibling example

- `mpirun`/`mpiexec` and `mpi4py` must be installed on **both** machines and
  built against the **same** MPI implementation.
- `mpirun` executes one identical command line on both machines, so `--python`
  must name an interpreter path valid on BOTH. Point it at a per-machine
  launcher script installed at one shared absolute path; each machine's copy
  sources CANN, enters that machine's checkout, and execs its `.venv` Python:

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
numeric IPs; the local host is where ranks 0 and 1 run:

```bash
source .venv/bin/activate
python -m examples.workers.l4.vector_add_mpi_direct_l3.main \
  --local-host 192.0.2.10 --remote-host 192.0.2.20 \
  --python /tmp/simpler-mpi-python \
  --local-devices 0,1 --remote-devices 0,1 \
  --mpirun-path "$(command -v mpiexec)" \
  --launcher-family mpich
```

Success requires the process to return status 0 after printing:

```text
vector_add_mpi_direct_l3 passed
```

Any failed vector validation exits non-zero.

## Running it in CI

The network1 job's `network1-stage` action writes the per-machine launcher on both
machines at one shared path and exports it as `NETWORK1_MPI_PYTHON`; the
`test_vector_add_mpi_direct_l3.py` wrapper reads it together with `NETWORK1_LOCAL_IP`
and the standard network1 fixtures, and skips when `mpirun`, `mpi4py`, or either
variable is absent.
