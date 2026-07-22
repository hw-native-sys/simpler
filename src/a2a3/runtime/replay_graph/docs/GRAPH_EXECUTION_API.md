# Graph Execution API Guide

Graph Execution records a repeated C/V/MIX/Dummy sub-DAG once. A later
execution occupies one outer `GRAPH` slot in the Task Window, and the
scheduler expands its saved topology.

The runtime and configuration value remain `replay_graph` for compatibility.
Graph Execution is currently an A2/A3 C++ orchestration API.

## 1. Enable Graph Execution

Select `replay_graph` and enable the cache for the run:

```python
@scene_test(level=2, runtime="replay_graph")
class TestAttentionGraph(SceneTestCase):
    CASES = [
        {
            "name": "AttentionGraph",
            "platforms": ["a2a3"],
            "config": {"enable_graph_cache": True},
        },
    ]
```

For a direct Worker call:

```python
config = CallConfig()
config.enable_graph_cache = True
worker.run(orchestration, args=task_args, config=config)
```

When the cache is disabled, `rt_submit_graph()` calls the Definition function
normally but does not record or submit a cached Graph Execution.

## 2. Public API

The normal entry point uses an existing `L2TaskArgs` and the Definition
function address as the Graph identity:

```cpp
PTO2GraphSubmitResult rt_submit_graph(
    void (*function)(const L2TaskArgs &),
    const L2TaskArgs &args
);
```

An overload accepts an explicit ID when one function implements multiple
structural variants:

```cpp
PTO2GraphSubmitResult rt_submit_graph(
    uint64_t graph_id,
    void (*function)(const L2TaskArgs &),
    const L2TaskArgs &args
);
```

`L2TaskArgs` is the only parameter container. No additional Graph-specific
argument object or handle is required.

## 3. Complete Example

The Definition function receives all dynamic values through `L2TaskArgs`.
Each internal task continues to use `L0TaskArgs`.

```cpp
static void attention(const L2TaskArgs &args) {
    const Tensor &input = args.tensor(0).ref();
    const Tensor &weight = args.tensor(1).ref();
    const Tensor &output = args.tensor(2).ref();

    uint32_t temporary_shape[] = {16, 5120};
    TensorCreateInfo temporary_info(
        temporary_shape, 2, DataType::FLOAT32
    );

    L0TaskArgs matmul_args;
    matmul_args.add_input(input);
    matmul_args.add_input(weight);
    matmul_args.add_output(temporary_info);
    matmul_args.add_scalar(args.scalar(0));
    matmul_args.add_scalar(uint32_t{16});
    TaskOutputTensors matmul =
        rt_submit_aic_task(FUNC_MATMUL, matmul_args);

    const Tensor &temporary = matmul.get_ref(0);
    L0TaskArgs activation_args;
    activation_args.add_input(temporary);
    activation_args.add_output(output);
    rt_submit_aiv_task(FUNC_ACTIVATION, activation_args);
}

void submit_attention(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &output,
    uint64_t layer_id
) {
    L2TaskArgs args;
    args.add_input(input);
    args.add_input(weight);
    args.add_output(output);
    args.add_scalar(layer_id);
    rt_submit_graph(&attention, args);
}
```

`L2TaskArgs` provides the Graph boundary contract:

- `add_input()` declares a read-only input;
- `add_output(Tensor)` declares a caller-owned output;
- `add_inout()` declares a read/write tensor;
- `add_no_dep()` declares a dependency-free tensor;
- `add_scalar()` adds a dynamic invocation value.

## 4. Dynamic And Static Arguments

Every value read directly from the Definition function's `args` is dynamic:

```cpp
task_args.add_scalar(args.scalar(0));
```

`args.scalar(0)` retains its source slot. During the first capture,
`L0TaskArgs::add_scalar()` records that source automatically. On a cache hit,
the scheduler reads the current value from the new invocation.

A value constructed inside the Definition function is static:

```cpp
task_args.add_scalar(uint32_t{16});
```

Static values are copied into the immutable Definition. Dynamic scalar values
must be passed directly from `args`; copying one into an ordinary integer and
then submitting that integer intentionally makes the resulting task argument
static.

Dynamic values must not change task count, kernel IDs, resource shapes, or
dependency topology. Use a different function or an explicit Graph ID for a
different topology:

```cpp
rt_submit_graph(&attention_fast, args);
rt_submit_graph(&attention_general, args);

// Alternatively, one Definition function with explicit identities.
rt_submit_graph(FAST_GRAPH_ID, &attention, args);
rt_submit_graph(GENERAL_GRAPH_ID, &attention, args);
```

Tensor addresses, tensor view offsets, and scalar values may change between
executions. Tensor rank, shape, strides, data type, dependency mode, and
input/output direction contribute to the structural key.

## 5. Supported Composition

A Definition function may call all existing task APIs:

```cpp
rt_submit_aic_task(aic_kernel_id, args);
rt_submit_aiv_task(aiv_kernel_id, args);
rt_submit_task(mixed_kernels, args);
rt_submit_dummy_task(args);
```

Internal outputs may feed later tasks. TensorMap dependencies and explicit
dependencies between tasks inside the same Definition are recorded. External
dependencies must be expressed through tensors in the outer `L2TaskArgs`, not
through a captured external `PTO2TaskId`.

## 6. Cache Identity

The complete Definition key contains:

- the Graph Execution schema version;
- the active callable-content hash;
- the explicit ID or current function address;
- boundary tensor structure and direction;
- boundary tensor and scalar counts.

The raw function address is process-local and is never persisted. The active
callable hash prevents definitions from different orchestration binaries from
sharing an identity accidentally.

## 7. Execution Lifecycle

Definition miss:

```text
rt_submit_graph(function, args)
  -> execute function(args)
  -> submit and record the original C/V/MIX/Dummy tasks
rt_graph_boundary()
  -> publish the original tasks first
  -> while Scheduler/AICore execute them, save ordered nodes, dependencies,
     static values, and Args sources
  -> compact the Definition to actual node and edge counts
```

Definition hit:

```text
rt_submit_graph(function, args)
  -> skip function(args)
  -> snapshot current dynamic Args into Graph Execution storage
  -> submit one outer GRAPH address to the Task Window
  -> scheduler prepares nodes before external dependencies are ready
  -> once the outer GRAPH is ready, activate and route the saved roots
```

The saved order is topological. Independent roots and branches remain free to
execute concurrently.

## 8. Definition And Execution Memory

The capture path supports up to 1024 nodes, but a cached Definition does not
copy or retain the fixed maximum-sized capture buffer. It allocates one
compact block containing only:

- `task_count + 1` fanout offsets;
- the actual internal edges;
- actual fanin counts and roots;
- actual node output offsets;
- compact node headers;
- only the tensor templates and tensor-source records used by those nodes;
- only the scalar values and scalar-source records used by those nodes.

Capture-only maximum arrays and recorded fanin lists are discarded after CSR
topology construction. Definition finalization therefore copies live fields,
not `task_count * sizeof(max-sized capture node)`.

Active executions pin the compact Definition slot. Each execution owns its
dynamic Args snapshot, node state, fanin counters, and output storage. Retired
execution blocks enter the bounded execution pool. Reuse prefers a block last
materialized from the same Definition and node count before applying best-fit
capacity selection. This Definition-affine path preserves immutable node data
and patches only invocation state and dynamic `TaskArgs`; a non-affine reuse
rewrites static fields. Scheduler preparation is split into bounded four-node
slices after completion and dispatch, avoiding both full payload initialization
and a monolithic Graph expansion pause on every hit.

## 9. Return Value

`PTO2GraphSubmitResult` reports the path taken:

| Path | `execute_block` | `recording` | `task_id` |
| ---- | --------------: | ----------: | --------- |
| Cache disabled | `true` | `false` | Invalid |
| Definition miss | `true` | `true` | Invalid |
| Definition hit | `false` | `false` | Outer Graph task |

Most wrappers can ignore the return value:

```cpp
void submit_layer(const L2TaskArgs &args) {
    rt_submit_graph(&attention, args);
}
```

Downstream dependencies should use output tensors rather than `task_id`, which
is invalid during the first define-and-execute call.

## 10. Limits

Current limits are:

- 32 boundary tensors;
- 32 boundary scalars;
- 1024 captured tasks;
- 128 fanins per internal task;
- 16 process-local Definition slots;
- 16 MiB or 64 blocks in the retired execution pool.

Nested Graph Definition functions are not supported. Unsupported capture
falls back to the already-submitted normal tasks.

## 11. Reference Implementation

- Public API: `orchestration/pto_orchestration_api.h`
- Args structural key: `runtime/pto_graph_cache.h`
- Definition capture and cache: `runtime/pto_orchestrator.cpp`
- Execution pool: `runtime/pto_graph_execution.cpp`
- Scheduler expansion: `runtime/scheduler/pto_scheduler.h`
- Minimal end-to-end example: `tests/st/a2a3/replay_graph/dummy_task/`

For ownership and scheduling invariants, see
[Graph Execution Runtime Logic](RUNTIME_LOGIC.md).
