# a2a3 — `tensormap_and_ringbuffer` examples

Kernels written against the `@scene_test` framework for the a2a3 architecture
and the `tensormap_and_ringbuffer` runtime.

**Read [`docs/INCORE_ORCHESTRATION_GUIDE.md`](docs/INCORE_ORCHESTRATION_GUIDE.md)
first.** It explains the one thing every example here has in common: the
orchestration function runs *on the AICPU* and builds the task graph on device,
dependencies are discovered by TensorMap from tensor overlap rather than
declared, and task memory comes from ring buffers. Nothing below makes much
sense without it.

For the `Worker` API underneath the framework, see
[`examples/workers/`](../../workers/README.md).

## Start here

| Example | What it teaches |
| ------- | --------------- |
| [`vector_example/`](vector_example/) | The smallest complete kernel: `f = (a+b+1)*(a+b+2) + (a+b)`. Runs on sim. |
| [`scalar_data/`](scalar_data/) | Orchestration-level data manipulation — `get_tensor_data` / `set_tensor_data` round-trips, runtime-created outputs with initial values, and automatic WAW / WAR waits. Also the reference for the one case where they **don't** fire: `add_input` on an external tensor registers no TensorMap entry, so a later `set_tensor_data` races the reader. |

## Compute

| Example | What it teaches |
| ------- | --------------- |
| [`benchmark_bgemm/`](benchmark_bgemm/) | Runtime-configurable tiled matmul `C = sum(k) A[k] @ B[k]`, laid out as single-axis moves over task count, tile size, work per task, and accumulation depth. On Sim, five of six cases are manual and Per-PR keeps the 500-task `Case0`; Onboard still runs `Case0` and `Bgemm64` by default. |
| [`paged_attention/`](paged_attention/) | Online softmax with AIC/AIV subgraph splitting, bfloat16. The baseline the three variants below are compared against. |
| [`paged_attention_manual_scope/`](paged_attention_manual_scope/) | The same computation with explicit scope control — kernels byte-identical to the baseline's, only the orchestration differs. See [`docs/manual-scope.md`](../../../docs/manual-scope.md). |
| [`paged_attention_unroll_manual_scope/`](paged_attention_unroll_manual_scope/) | A second implementation, not a patch on the baseline: KV blocks batched into groups of `N_UNROLL`, four tasks per group instead of per block, with the kernels rewritten to match. |
| [`paged_attention_ringbuffer/`](paged_attention_ringbuffer/) | Deliberately undersized rings, driven per case through `config.runtime_env` (`ring_task_window` / `ring_heap` / `ring_dep_pool`) rather than process-global env. A stress test for rotation and reclamation. |
| [`merge_pipeline_barrier/`](merge_pipeline_barrier/) | Three pipeline stages merged into **one** `block_num=8` SPMD task, ordered by an intra-task cross-core barrier instead of by three scheduled tasks. |
| [`qwen3_14b_decode/`](qwen3_14b_decode/README.md) | The whole Qwen3-14B 40-layer decode stack as a single fused dispatch, using CANN fused attention. The largest example in the repo. |
| [`deepseek_v4_flash_decode/`](deepseek_v4_flash_decode/README.md) | The whole DeepSeek-V4 FLASH 43-layer decode network on **2 dies** (EP2 expert-parallel MoE + TP2 LM head through a comm domain). The first pypto-harvested distributed network; Per-PR `skip_golden` completion smoke, like upstream. |

## Asynchronous completion and cross-card transfer

Each of these registers an async event and lets the consumer wait on deferred
completion rather than on task end. Both are **onboard-only**;
`sdma_async_completion_demo` needs two dies.

These mechanism-focused examples use the scene-test lifecycle. For a complete
direct `Worker` communication-domain walkthrough from construction through
`close()`, see [`examples/workers/l3/allreduce/`](../../workers/l3/allreduce/).

| Example | Mechanism | Devices |
| ------- | --------- | ------- |
| [`prefetch_async_demo/`](prefetch_async_demo/) | `TPREFETCH_ASYNC` over the runtime-injected SDMA workspace, provisioned once by `Worker(enable_sdma=True)` and injected into every kernel's `GlobalContext`. | 1 |
| [`sdma_async_completion_demo/`](sdma_async_completion_demo/) | `TGET_ASYNC` from a peer's window slot, completion registered via `defer_pto_async_event`. | 2 |

The cross-architecture notification-counter and deferred-notify watchdogs live
under [`tests/st/worker/comm_domain/`](../../../tests/st/worker/comm_domain/).

## Running

```bash
# Everything that runs on the simulator
pytest examples/a2a3/tensormap_and_ringbuffer --platform a2a3sim

# One example, on hardware
pytest examples/a2a3/tensormap_and_ringbuffer/paged_attention --platform a2a3 --device 0
```

Most examples here are marked `platforms=["a2a3"]` — onboard only — because
they exercise real AIC/AIV timing or cross-card transfer. `vector_example` and
`benchmark_bgemm` are marked for `a2a3sim` as well.

Wrap hardware runs in `task-submit` on a shared box; see
[`.claude/rules/running-onboard.md`](../../../.claude/rules/running-onboard.md).

## Relationship to `examples/a5/`

Five examples exist under both architectures with the same name:
`vector_example`, `paged_attention`, `paged_attention_manual_scope`,
`paged_attention_unroll_manual_scope`, and `sdma_async_completion_demo`. They
are ports of each other and differ mainly in tile shapes and platform strings
— `vector_example` differs by two lines. When you change one, check whether its
sibling needs the same change.

Only here: `benchmark_bgemm` (a5 has `bgemm` instead),
`deepseek_v4_flash_decode`, `merge_pipeline_barrier`,
`paged_attention_ringbuffer`, `prefetch_async_demo`, `qwen3_14b_decode`,
`scalar_data`.
