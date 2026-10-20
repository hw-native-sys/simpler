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

## Transport and failure handling

The MPI path carries the current SLR3 wire protocol through the existing
`RemoteL3Transport` interface. MPI tags separate requests, replies, health,
and lifecycle frames. Python checks tags and frame limits against the native
extension before ranks enter MPI. The supervisor passes one inline manifest
to every rank and releases the startup gate after all L3 workers have forked;
launch through the example CLI above or `simpler.mpi_direct_supervisor`.
The L4 controller must have no local SubWorker, chip, or Worker children;
configure those children on the L3 executor ranks, which fork before MPI
initialization. Attachment and startup reject local controller children.

MPI calls use `MPI_THREAD_SERIALIZED`. Controller progress and executor
receive loops poll without sleeping on the task path; an idle rank can
therefore occupy a CPU core. Startup retries and periodic heartbeats may
block because they do not delay task dispatch. The controller detects a
missing heartbeat after `session_timeout_s`, including while no task is active;
`heartbeat_interval_s` must be shorter than that timeout. A progress failure
calls `MPI_Abort` from the progress thread even when the controller callback is
waiting for an external event. If progress cannot drain, rank 0 exits without
making another concurrent MPI call and the launcher reaps the world.

Normal close stops task admission, sends lifecycle `SHUTDOWN` to each idle
executor, then drains MPI sends. Native `Worker.close()` releases the Python
GIL so progress can release outbound byte credit during teardown. Python
still enforces init-owner thread affinity and serializes close attempts.

The supervisor owns the launcher process group; the launcher owns the MPI
world. A rank failure, invalid frame, or shared credit/health timeout fails
the whole world. Runtime errors call `MPI_Abort`. A controller crash relies
on the launcher's rank-failure cleanup. A network partition is detected through
MPI errors or the finite session/heartbeat deadline, rather than a daemon lease.
The supervisor runs on Python's main thread. On `SIGTERM` or `SIGHUP` it
terminates the launcher's process group and restores the caller's signal
handlers after cleanup. Each startup peer has one absolute read deadline,
capped at five seconds and the remaining startup budget; fragmented messages
cannot extend it. Unauthenticated connections are ignored, while authenticated
rank or state inconsistencies fail startup immediately. MPI launcher's rank variables
take priority over an inherited `SLURM_PROCID`, which is only a fallback.
MPICH and Open MPI command construction is covered by unit tests. The CPU
integration test exercises a real three-rank roundtrip, kills rank 0, and stops
an executor while the controller is idle to check that no rank survives;
launcher cleanup is verified with MPICH 4.1.3.
Open MPI cleanup and cross-host partitions still require validation in the
deployment environment.
