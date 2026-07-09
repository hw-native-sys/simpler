# Collective Scene Tests

Scene tests for distributed collective operations (allreduce, allgather,
reduce_scatter, broadcast, all_to_all) at L3 under the
`tensormap_and_ringbuffer` runtime.

These scene tests target **a2a3 / a2a3sim only**. a5 coverage is not yet
exercised at this level.

For a minimal "how to do a collective" demo, see
[`examples/workers/l3/allreduce/`](../../../../../examples/workers/l3/allreduce/).

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
