# Allreduce — minimal onephase example

Simplest "how to do a collective" feature demo.  Each rank stages its private
vector into the HCCL window, waits for peers via a signal barrier, then reads
every peer's slot and accumulates locally.

For the full algorithm corpus (twophase, ring, bidirectional_ring, ibing),
see the scene tests at
[`tests/st/a2a3/tensormap_and_ringbuffer/collectives/allreduce/`](../../../../tests/st/a2a3/tensormap_and_ringbuffer/collectives/allreduce/).

## Run

```bash
# Simulation (2 ranks)
python examples/workers/l3/allreduce/main.py -p a2a3sim -d 0-1

# Hardware (2 ranks)
python examples/workers/l3/allreduce/main.py -p a2a3 -d 0-1
```
