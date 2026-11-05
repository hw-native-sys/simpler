# L3 Host DDR with asynchronous URMA reads

One ordinary L3 Host Buffer holds the complete table. `worker.map_view(tensor)`
prepares every A5 child against that same backing. A mapped Tensor retains its
shape and offset and is passed through normal TaskArgs. The buffer is not split
by rank, and no full table is copied to Device memory.

```python
table = worker.create_buffer(table_bytes)
native = table.tensor((rows, columns), DataType.BFLOAT16, byte_offset=512)
mapped = worker.map_view(native)
args.add_tensor(mapped)
# In the L3 callback:
orch.submit_next_level(entry, args, config, worker=rank)
```

The kernel obtains `get_dma_workspace(args, DMA_WORKSPACE_URMA)` and uses
`peer_mr_ptr(workspace, peer, byte_offset)` for a nonself peer. Each peer exposes
the same complete Host backing. For the same index, the local mapped Tensor
and the peer address refer to the same data but need not have equal numerical
addresses. `UrmaTget` registers real deferred completion before the dependent
consumer can read its Device output.

The local rank is passed as a scalar alongside the peer rank. Compute the view's
offset relative to the **local** registered base, since Tensor materialization
can fold its byte offset into `buffer.addr`:

```cpp
uint64_t offset = tensor->buffer.addr + tensor->start_offset * element_bytes
                - peer_mr_base_addr(workspace, local_rank);
auto *source = peer_mr_ptr<uint8_t>(workspace, peer, offset + row * row_bytes);
```

Subtracting the local base before adding the peer base preserves nonzero Tensor
offsets even when different ranks use different mapped virtual addresses.

`run.py` compares peer URMA with local mapped MTE, including repeated and boundary
indices, invalid indices/counts, two offset views, and CPU updates between drained
runs. It unmaps every view after completed work, remaps the same backing, rejects
the revoked reference, and reads through the newly published workspace. Its last
two rounds preserve the same view, indices and Host bytes; with at least three
ranks they read those same offsets through different nonself peers. A separate
ordinary H2D input feeds useful independent Device computation.
`run_mx.py` fetches Host E4M3FN rows into Device GM, then invokes the existing
`TMATMUL_MX` consumer with Device E8M0 scales. Neither example registers a public
Device communication domain.

`run_coexist.py` adds a separate Device communication domain. One producer issues
both Host-table and Device-window reads through their respective URMA workspaces,
and dependent consumers check both outputs. The two ranks reverse posting order.
This exercises completion tokens carrying two different workspace addresses in
one task.

Use two or more devices reserved by the repository's task queue:

```bash
python examples/a5/tensormap_and_ringbuffer/host_ddr_urma/run.py --devices 0,1
python examples/a5/tensormap_and_ringbuffer/host_ddr_urma/run.py --devices 0,1 --staged
python examples/a5/tensormap_and_ringbuffer/host_ddr_urma/run_mx.py --devices 0,1
python examples/a5/tensormap_and_ringbuffer/host_ddr_urma/run_coexist.py --devices 0,1
```

`--staged` selects depth two and submits two runs before waiting for either. Host
contents remain unchanged until both handles have drained.

`--prepare` compiles without initializing devices. `--output-prefix outputs/host_urma`
enables standard chip swimlane capture for each round and rank. The independent
branch permits overlap; numerical success alone is not evidence of overlapping
execution. The same option enables dependency capture to identify the URMA
producer and independent consumer in the device trace. Scheduler `finish_time`
marks AICore FIN observation, before deferred-completion polling; it is not the
URMA completion time. Inspect the `async_poll` phases and dependent-consumer
dispatch alongside the independent branch before claiming overlap.

Wait for every accepted L3 run before updating the shared table or unmapping.
Both views reuse the same registration until the last one is unmapped. Explicit
unmap revokes the view; ordinary Host CPU access to the buffer remains available.
These examples require A5 onboard `tensormap_and_ringbuffer`; HBG and simulation
do not implement this asynchronous Host workspace path.
