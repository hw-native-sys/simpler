# Global TLOAD mixed L3 example

This example validates the L4-brokered Global CommDomain path across one local
and one remote L3 node, ending in a device-side peer `TLOAD`:

```text
L4 parent on machine A -> local L3 on machine A  -> local NPU, rank 0
                       -> remote L3 on machine B -> remote NPU, rank 1
```

Both ranks join one Global CommDomain without `mpirun`. The domain build runs
the complete control flow: every L2 exports its window descriptor (PREPARE),
L4 collects the rank-ordered table, sends it back to every L3 (IMPORT), and
commits only after all imports succeed. Each rank's AIV kernel then reads the
peer rank's `input` window with `TLOAD`, sums all rank inputs, and stores the
sum to its own `result` window. The parent reads both `result` windows back and
checks them against the golden sum, so a successful descriptor exchange
without working peer `TLOAD` is not reported as success.

`vector_add_mixed_l3` covers the same mixed local/remote topology for plain
task dispatch and remote buffers; this example exists for what that one leaves
out — the Global CommDomain build and the cross-machine peer read.

## Prepare both machines

Use the same source revision on machine A and machine B:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .
```

## Start the remote L3 daemon

On shared hosts, start the daemon and the L4 parent in separate
`task-submit` allocations, one device on each host. Use the device IDs
assigned by the respective queues.

Start this on machine B. The daemon is the generic session server — nothing
about it is specific to this example. `192.0.2.20` is a documentation
placeholder, not a usable address:

```bash
source .venv/bin/activate
python -m simpler.remote_l3_worker --host 192.0.2.20 --port 19073
```

## Run on the L4 parent

Run this on machine A. The local device id is owned by the forked local L3 on
machine A; the remote device id is owned by the daemon-started L3 on machine B.
Only the first id of each device list is used — each L3 owns exactly one L2:

```bash
source .venv/bin/activate
export SIMPLER_GLOBAL_TLOAD_MIXED_L3_REMOTE=192.0.2.20:19073
export SIMPLER_GLOBAL_TLOAD_MIXED_L3_LOCAL_DEVICES=0
export SIMPLER_GLOBAL_TLOAD_MIXED_L3_REMOTE_DEVICES=0
bash examples/workers/l4/global_tload_mixed_l3/run_parent.sh
```

Expected output ends with:

```text
[global-tload-mixed-l3] local rank=0 max_diff=0.000e+00
[global-tload-mixed-l3] remote rank=1 max_diff=0.000e+00
global_tload_mixed_l3 passed
```

Any `max_diff` above the tolerance exits non-zero.

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_REMOTE` | (required) | Remote daemon endpoint, `HOST:PORT` |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_LOCAL_DEVICES` | `0` | Device list; the first id is the local L3's L2 |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_REMOTE_DEVICES` | `0` | Device list; the first id is the remote L3's L2 |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_PLATFORM` | `a2a3` | Target platform |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_RUNTIME` | `tensormap_and_ringbuffer` | Runtime |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_COMM_PROFILE` | `a3-fabric-v1` | Global CommDomain backend profile. Set `a5-fabric-v1` for real A5. |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_SESSION_LISTEN_HOST` | `0.0.0.0` | Bind address for the remote L3 command and health listeners |
| `SIMPLER_GLOBAL_TLOAD_MIXED_L3_SESSION_TIMEOUT` | `120` | Seconds to wait on the remote session |

The default profile is `a3-fabric-v1` for real A3 devices on both machines.
For two A5 machines in the same superpod, select platform `a5` and profile
`a5-fabric-v1`. When setting an explicit session listen address, use the
peer's address. The parent connects to the command and health listeners on
that address.

In CI the network1 job runs the `test_global_tload_mixed_l3.py` wrapper
through `network1-run-pytest`.
CI supplies `NETWORK1_REMOTE_ENDPOINT`, `NETWORK1_REMOTE_DEVICES`,
`NETWORK1_L3_SESSION_TIMEOUT_S`, `NETWORK1_L3_SESSION_LISTEN_HOST`, and pytest's
`--platform` / `--device` options instead of the `SIMPLER_*` variables above;
`run_parent.sh` remains the manual entry point.
