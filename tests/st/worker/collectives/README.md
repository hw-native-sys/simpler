# Collective Scene Tests

Scene tests for distributed collective operations (allreduce, allgather,
reduce_scatter, broadcast, all_to_all) at L3 under the
`tensormap_and_ringbuffer` runtime.

These scene tests run on **a2a3, a2a3sim, and a5sim**. a5 onboard CI exposes
only 2 NPUs, so 2-rank allreduce cases additionally run on **a5** while 4-rank
cases run on a5sim only.

For a minimal "how to do a collective" demo, see
[`examples/workers/l3/allreduce/`](../../../../examples/workers/l3/allreduce/).

## Test Layout

```text
collectives/
├── _helpers.py           # Shared orch helpers, golden functions, arg builders
├── allreduce/            # 5-mode allreduce (onephase, twophase, ring, bidirectional_ring, ibing)
├── allgather/            # Mesh allgather
├── reduce_scatter/       # Mesh reduce-scatter
├── broadcast/            # Mesh broadcast
├── all_to_all/           # Mesh all-to-all
└── README.md
```
