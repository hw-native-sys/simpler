# Graph Execution

Graph Execution is available only in the `host_build_graph` runtime. It caches a repeated task DAG and replaces each
later invocation with one Graph task in the host-built task window. The device Scheduler expands that task and submits
the cached nodes to AICore; the host Orchestrator does not replay the nodes one by one.

## API

Define a layer as a function that accepts the existing `L2TaskArgs` type:

```cpp
void layer(const L2TaskArgs &args) {
    const Tensor &input = args.tensor(0).ref();
    const Tensor &weight = args.tensor(1).ref();
    const Tensor &output = args.tensor(2).ref();

    L0TaskArgs task_args;
    task_args.add_input(input, weight);
    task_args.add_output(output);
    task_args.add_scalar(args.scalar(0));  // dynamic: read from this invocation
    task_args.add_scalar(uint32_t{16});    // static: stored in the definition
    rt_submit_aic_task(FUNC_MATMUL, task_args);
}

void submit_layer(const L2TaskArgs &args) {
    rt_submit_graph(&layer, args);
}
```

The upper layer constructs `L2TaskArgs` before calling `submit_layer`; the
wrapper does not repeat tensors or configuration values in its own signature.
The function pointer is the default Graph ID. An explicit stable ID can be
supplied when needed:

```cpp
rt_submit_graph(PTO2_GRAPH_KEY("decoder_layer_v1"), &layer, args);
```

There are no public `GraphBindings`, `Patch`, or `GraphArgs` types. Tensors, configuration values, and dynamic scalars
all enter through `L2TaskArgs`.

### Qwen decoder-layer integration

The upper layer packages every value that changes between decoder layers or
decode rounds into `L2TaskArgs`. The Graph function reads those values and
submits the layer's ordinary AIC/AIV tasks. Kernel constants remain literals in
the Graph function and are stored once in the Graph definition.

The following abbreviated Qwen3-14B layer illustrates the calling pattern. The
real layer may submit attention, normalization, matrix multiplication, and MLP
tasks in any DAG supported by the ordinary task API.

```cpp
enum QwenArg : int32_t {
    HIDDEN_STATES,
    ATTN_WEIGHT,
    MLP_WEIGHT,
    OUTPUT,
};

enum QwenScalar : int32_t {
    LAYER_ID,
    TOKEN_POSITION,
};

void qwen_decoder_layer(const L2TaskArgs &args) {
    const Tensor &hidden = args.tensor(HIDDEN_STATES).ref();
    const Tensor &attn_weight = args.tensor(ATTN_WEIGHT).ref();
    const Tensor &mlp_weight = args.tensor(MLP_WEIGHT).ref();
    const Tensor &output = args.tensor(OUTPUT).ref();

    uint32_t intermediate_shape[] = {hidden.shapes[0]};
    TensorCreateInfo intermediate(
        intermediate_shape, 1, hidden.dtype
    );

    L0TaskArgs attention_args;
    attention_args.add_input(hidden, attn_weight);
    attention_args.add_output(intermediate);
    attention_args.add_scalar(args.scalar(LAYER_ID));
    attention_args.add_scalar(args.scalar(TOKEN_POSITION));
    attention_args.add_scalar(uint32_t{16});  // fixed head-group size
    Tensor attention =
        rt_submit_aic_task(FUNC_ATTENTION, attention_args).get_ref(0);

    MixedKernels mlp_kernels;
    mlp_kernels.aic_kernel_id = FUNC_MLP_AIC;
    mlp_kernels.aiv0_kernel_id = FUNC_MLP_AIV;

    L0TaskArgs mlp_args;
    mlp_args.add_input(attention, mlp_weight);
    mlp_args.add_output(output);
    rt_submit_task(mlp_kernels, mlp_args);
}

void submit_qwen_decoder_layer(const L2TaskArgs &args) {
    rt_submit_graph(&qwen_decoder_layer, args);
}

void decode_three_layers(const L2TaskArgs layer_args[3]) {
    for (uint32_t layer_id = 0; layer_id < 3; ++layer_id) {
        submit_qwen_decoder_layer(layer_args[layer_id]);
    }
}
```

Each `layer_args[i]` contains that layer's hidden state, weights, output,
`layer_id`, and `token_position`. The common function pointer supplies the same
Graph ID for all structurally identical decoder layers. The first layer records
the task DAG. Later layers submit one Graph task because tensor addresses and
scalar values are dynamic. Tensor shape, dtype, stride, size, and direction
must remain structurally compatible; changing them selects a different cache
entry. If a model has decoder-layer variants with different task topology, use
the explicit-key overload with a distinct key for each variant.

## Record and execute

On the first call for a structural key, `rt_submit_graph` executes the function normally. The Orchestrator records each
C, V, MIX, or Dummy task and builds a compact definition containing:

- topological node order and kernel/launch metadata;
- internal fanin counts, fanout adjacency, and root nodes;
- tensor argument sources and per-node packed-output offsets;
- static scalar values and dynamic scalar source indices;
- the external tensor boundary and its input/output directions.

On a cache hit, the Orchestrator allocates the Graph's combined intermediate heap range, computes only its external
dependencies, and places one `GRAPH` descriptor in the task window. The uploaded submission contains only the compact
definition and current `L2TaskArgs`; it does not contain the expanded node array. The host uploads exactly the used bytes
and immediately returns its temporary block to a bounded pool. This avoids copying the former fixed 6.14 MiB execution
object for every invocation.

On first observation of the outer task, the Scheduler acquires an AICPU-local execution block and copies the compact
definition into it. It materializes four nodes per scheduling slice while the outer Graph waits for external fanin. Once
both preparation and external readiness are satisfied, it activates the saved roots. Node completion directly releases
the saved fanout list. The outer Graph completes only after all internal nodes complete, so downstream tasks observe it
as one normal dependency producer.

Host submission blocks and AICPU-local execution blocks use separate bounded pools (16 MiB and 64 blocks each). The
execution pool prefers a block previously used by the same Graph key. That graph-affine path keeps the local definition
and static node fields, then refreshes only task IDs, dynamic tensors/scalars, packed-buffer bases, and scheduling state.
Completed executions are reclaimed after their internal nodes retire; the final Scheduler thread performs a last
collection before runtime shutdown.

## DFX lanes

`--enable-l2-swimlane 4` produces five logical lanes in the converted Perfetto JSON:

- `Host Orchestrator`: one envelope for the host build/submit interval, with
  `task_submit(tN)` children for cache-miss nodes and one `graph_submit(tN)` child per cache-hit Graph invocation;
- `Graph Execution`: one envelope per outer Graph task, from its first prepare slice through its last visible node;
- `AICPU Scheduler`: includes `graph_prepare` separately from normal `dispatch`;
- `Scheduler View` and `Worker View`: the existing task scheduling and AICore execution lanes.

Host and device counters are separate clock domains. The converter logically places Host Orchestrator immediately before
device time zero, then rebases the complete trace so Host Orch starts at trace time zero and every device event follows
it. This preserves ordering without implying clock synchronization or host/device overlap, and avoids Perfetto clipping
the host slice as a negative-time event. Keeping `graph_prepare` out of `dispatch` makes Graph expansion cost visible
without inflating ordinary Scheduler dispatch time.

## Dynamic arguments and cache keys

Tensor addresses and scalar values may change between invocations. Tensor shape, dtype, stride, size, direction, and
other structural metadata participate in the cache key; addresses and scalar values do not. A scalar copied from
`args.scalar(i)` records source index `i` and is read from the current invocation during materialization. A literal or
locally computed scalar added directly to `L0TaskArgs` is stored in the definition.

The current implementation caches up to 16 definitions, 1024 nodes per definition, and 32 boundary tensors/scalars.
Explicit dependencies between nodes inside one Graph are saved in the topology. Nested Graph scopes, explicit
dependencies that cross the Graph boundary, and dispatch predicates are not cached yet; those calls still execute
their function body on the ordinary task-submission path.
