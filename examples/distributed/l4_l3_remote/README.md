# L4 to L3 Remote Dispatch

Terminal 1:

```bash
python examples/distributed/l4_l3_remote/l3_worker.py --port 5050
```

Terminal 2:

```bash
python examples/distributed/l4_l3_remote/l4_master.py --remotes 127.0.0.1:5050
```

Expected output:

```text
remote result=7
```

The callable registered on L4 is serialized and executed inside the L3 daemon,
so it must not rely on mutating Python objects captured from the L4 process.
This example returns the distributed result through an `OUTPUT_EXISTING` tensor,
which is copied back into the L4-local buffer after dispatch completes.
