# A3 L4 TCP Global CommDomain smoke

This smoke validates the complete no-`mpirun` path:

1. L4 connects to the pre-started TCP daemon on each server.
2. Each daemon starts its local L3, and L3 starts the selected L2 worker.
3. L2 exports one Fabric V2 key; L3 returns it to L4.
4. L4 builds the dense rank-ordered key table and sends it to every L3.
5. Every L3 asks its L2 to import the complete table and create `CommContext`.
6. The AIV kernel on every rank executes peer `TLOAD`, sums all rank inputs,
   and writes the result to its local Global CommDomain window.

Start one daemon on each server from the same source revision:

```bash
python -m simpler.remote_l3_worker --host 10.0.0.1 --port 19073
python -m simpler.remote_l3_worker --host 10.0.0.2 --port 19073
```

Run the L4 driver on the control host:

```bash
python tools/a3_l4_tcp_smoke/global_tload_smoke.py \
  --endpoint 10.0.0.1:19073 --device-id 0 \
  --endpoint 10.0.0.2:19073 --device-id 0
```

The test passes only when every result equals the sum of all rank inputs. A
successful descriptor import without working peer `TLOAD` is therefore not
reported as success.

## Compute then communicate case

`compute_then_tload_smoke.py` keeps the same L4/L3/L2 processes alive for two
ordered task rounds:

1. Every L3 submits one local vector-add task to its L2.
2. After all compute tasks finish, every L3 submits one peer `TLOAD` sum task
   to its L2.

Run it against the same two daemons:

```bash
python tools/a3_l4_tcp_smoke/compute_then_tload_smoke.py \
  --endpoint 10.0.0.1:19073 --device-id 0 \
  --endpoint 10.0.0.2:19073 --device-id 0
```

The case checks each rank's local addition result before communication, then
checks that every rank receives the sum of all computed rank results.

For a simpler real-device baseline that first verifies L4 remote dispatch,
per-machine two-NPU L3 group execution, remote-buffer upload/download, and
golden checking without peer-machine memory access, run
[`remote_l4_npu`](../remote_l4_npu/README.md).
