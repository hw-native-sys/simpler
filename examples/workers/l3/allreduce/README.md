# Allreduce — minimal L3 Worker demo

Simplest "how to use the L3 `Worker` interface for a collective" demo.
Walks through the full flow:

1. **`Worker(level=3)`** — create a host-level worker on the target platform
2. **`worker.register(chip_callable)`** — register a compiled `ChipCallable`
   (onephase allreduce kernel + orchestration shim)
3. **`worker.init()`** — fork chip children; lazy base-communication init
4. **`orch.allocate_domain(...)`** — allocate a communication domain with a
   `CommBufferSpec` scratch window
5. **`orch.submit_next_level(chip_handle, chip_args, cfg, worker=i)`** —
   submit the allreduce task for each rank
6. **`worker.run(orch_fn, ...)`** — execute the DAG and golden-check against
   the known expected sum

For the full algorithm corpus (twophase, ring, bidirectional_ring, ibing),
see the scene tests at
[`tests/st/worker/collectives/allreduce/`](../../../../tests/st/worker/collectives/allreduce/).

## Run

```bash
# Simulation (2 ranks)
python examples/workers/l3/allreduce/main.py -p a2a3sim -d 0-1

# Hardware (2 ranks)
python examples/workers/l3/allreduce/main.py -p a2a3 -d 0-1
```

## File Structure

```text
allreduce/
├── kernels/
│   ├── aiv/
│   │   └── allreduce_onephase_kernel.cpp
│   └── orchestration/
│       └── allreduce_onephase_orch.cpp
├── main.py
└── README.md
```
