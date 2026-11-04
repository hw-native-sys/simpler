# WAR (Write-After-Read) Anti-Dependencies

## L2 automatic dependency tracking

In `tensormap_and_ringbuffer` and `host_build_graph`, an ordinary `add_input(X)`
registers a reader in an automatic scope. A later `add_inout(X)` or
`add_output(X)` of existing storage waits for all earlier overlapping readers.
Readers of the same storage remain independent of each other. No special input
tag or promotion of a read-only argument to INOUT is required.

Reader and writer entries use separate hash indexes. INPUT looks up writers;
a write also looks up readers. A fully covered access is removed after its
ordering edge is accepted; a partially covered access remains visible to later
writes of the remaining region. TMR also removes retired tasks' entries through
its existing per-task cleanup chain. Reader edges are WAIT-only: they do not
retain a reader's unrelated outputs. Creator lifetime dependencies remain intact.

Each registered INPUT uses an entry in the existing TensorMap pool. Lookup cost
on a write includes the matching reader chain, and distinct readers contribute
to the writer's fanin. TMR uses its existing dependency spill pool. HBG retains
its limit of 128 distinct predecessors per task; exceeding it fails graph
construction explicitly. Automatic tracking does not remove resource limits.

## Explicit dependency contracts

Manual scopes and tensors marked `manual_dep` do not publish automatic reader
entries. NO_DEP arguments do not publish accesses either. Their callers remain
responsible for ordering conflicting accesses through explicit dependencies.
Existing manual WAR edges remain valid in automatic scopes and are deduplicated
with inferred predecessors.

The L3 host orchestrator has a separate dependency map; its ordering contract
is described in [orchestrator semantics](orchestrator.md#semantics). L2 reader
tracking does not infer edges between independent L3 submissions. Use
`writer_args.add_dep_wait(reader)` for an ordering-only host task dependency.

## Scalar access

TMR `set_tensor_data(X)` waits for overlapping registered readers to complete,
in addition to its creator/writer and writer-consumer guards. It does not wait
for a reader's downstream consumers. `get_tensor_data(X)` only waits for
writers. Both operations are blocking, so later submissions follow the scalar
access in orchestration order. Manual access contracts still apply.

HBG scalar access only accepts explicit HOST tensors with no device owner;
DEVICE tensor access remains rejected. Automatic WAR does not introduce a
host-write task or change that address-space contract.

See [manual scopes](manual-scope.md) for explicit ordering and the runtime's
`SCALAR_DATA_ACCESS.md` for scalar access details.
