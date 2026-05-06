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
remote counter=7
```
