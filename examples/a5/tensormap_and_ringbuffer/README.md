# a5 — `tensormap_and_ringbuffer` examples

Kernels written against the `@scene_test` framework for the a5 architecture and
the `tensormap_and_ringbuffer` runtime.

The orchestration model is the same as on a2a3 — the orchestration function
runs on the AICPU, TensorMap discovers dependencies from tensor overlap, task
memory comes from ring buffers. That is written up once, under a2a3:
[`examples/a2a3/tensormap_and_ringbuffer/docs/INCORE_ORCHESTRATION_GUIDE.md`](../../a2a3/tensormap_and_ringbuffer/docs/INCORE_ORCHESTRATION_GUIDE.md).

For the `Worker` API underneath the framework, see
[`examples/workers/`](../../workers/README.md).

## Start here

| Example | What it teaches |
| ------- | --------------- |
| [`vector_example/`](vector_example/) | The smallest complete kernel: `f = (a+b+1)*(a+b+2) + (a+b)`. Runs on sim. |
| [`bgemm/`](bgemm/) | Batched tiled matmul `C = A @ B` over a fixed 4×4×4 grid of 64×64 tiles. **One source in `kernels/mix/`, compiled twice** — cube half `TPUSH`es the intermediate into `VEC_FIFO`, vector half `TPOP`s it, so the matmul product never round-trips through GM. Runs on sim. |

## Paged attention

| Example | What it teaches |
| ------- | --------------- |
| [`paged_attention/`](paged_attention/) | Online softmax with mixed AIC/AIV execution, bfloat16, at production scale. Onboard only. |
| [`paged_attention_manual_scope/`](paged_attention_manual_scope/) | The same computation with explicit scope control — all four kernels byte-identical to the baseline's, only the orchestration differs (52 lines of 288). See [`docs/manual-scope.md`](../../../docs/manual-scope.md). Also runs on sim. |
| [`paged_attention_unroll_manual_scope/`](paged_attention_unroll_manual_scope/) | A second implementation, not a patch on the baseline: KV blocks batched into groups of `N_UNROLL`, four tasks per group instead of per block, with the kernels rewritten to match. Onboard only. |

## Asynchronous completion and cross-card transfer

Each registers an async event and lets the consumer wait on deferred completion
rather than on task end. All need two dies.

These mechanism-focused examples use the scene-test lifecycle. For a complete
direct `Worker` communication-domain walkthrough from construction through
`close()`, see [`examples/workers/l3/allreduce/`](../../workers/l3/allreduce/).

| Example | Mechanism |
| ------- | --------- |
| [`sdma_async_completion_demo/`](sdma_async_completion_demo/) | `TGET_ASYNC` from a peer's window slot over SDMA, completion registered via `defer_pto_async_event`. Enabled by default on a5 onboard. |
| [`urma_deferred_completion_demo/`](urma_deferred_completion_demo/) | The same protocol over **URMA** — `kernel_consumer.cpp` is byte-identical to the SDMA demo's, so the transport is the only variable. The two overlays are **mutually exclusive in one build**, so comparing them means rebuilding. |

The cross-architecture notification-counter and deferred-notify watchdogs live
under [`tests/st/worker/comm_domain/`](../../../tests/st/worker/comm_domain/).

## Running

```bash
# Everything that runs on the simulator
pytest examples/a5/tensormap_and_ringbuffer --platform a5sim

# One example, on hardware
pytest examples/a5/tensormap_and_ringbuffer/bgemm --platform a5 --device 0
```

`vector_example`, `bgemm`, and `paged_attention_manual_scope` run under
`a5sim`; the rest are onboard-only.

Wrap hardware runs in `task-submit` on a shared box; see
[`.claude/rules/running-onboard.md`](../../../.claude/rules/running-onboard.md).
Check the box's silicon first — `.claude/skills/onboard-arch-precheck/check.sh a5`
refuses a wrong-arch invocation before any device lock is taken.

## Relationship to `examples/a2a3/`

Five examples exist under both architectures with the same name:
`vector_example`, `paged_attention`, `paged_attention_manual_scope`,
`paged_attention_unroll_manual_scope`, and `sdma_async_completion_demo`. They
are ports of each other and differ mainly in tile shapes and platform strings
— `vector_example` differs by two lines. When you change one, check whether its
sibling needs the same change.

Only here: `bgemm` (a2a3 has `benchmark_bgemm` instead) and
`urma_deferred_completion_demo`. a2a3 additionally carries
`merge_pipeline_barrier`, `paged_attention_ringbuffer`, `prefetch_async_demo`,
`qwen3_14b_decode`, and `scalar_data`, none of which have an a5 port —
`tests/st` tracks that gap separately (PR #1450 ports the `tests/st` side).
