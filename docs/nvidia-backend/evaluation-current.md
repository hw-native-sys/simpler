# CUDA Current Evaluation Capture

This page summarizes the latest full paired A100/H200 CUDA backend capture
from commit `61cf96cd`, plus compact current-head validation captures. The
latest compact gate is artifact label `f074746a`, which promotes the explicit
graph unary-square descriptor row into the selected baseline matrix. The raw
JSON, Markdown, and SVG reports are generated locally under
`tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `720`

## Artifact Paths

- `tmp/cuda-backend/a100-current-61cf96cd/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-61cf96cd/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-61cf96cd/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-61cf96cd/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/a100-current-2aedb40f/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-2aedb40f/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-2aedb40f/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-2aedb40f/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/a100-current-b2c5c8a4/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-b2c5c8a4/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-b2c5c8a4/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-b2c5c8a4/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/a100-current-06b8c0c6/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-06b8c0c6/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-06b8c0c6/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-06b8c0c6/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-06b8c0c6/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-06b8c0c6/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-06b8c0c6/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-06b8c0c6/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-06b8c0c6/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-06b8c0c6/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/a100-current-dbb01406/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-dbb01406/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-dbb01406/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-dbb01406/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/cublas-graph-compact-working/a100-current-5168f150/cuda-benchmark.json`
- `tmp/cuda-backend/cublas-graph-compact-working/h200-current-5168f150/cuda-benchmark.json`
- `tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/cuda-benchmark.json`
- `tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/cuda-benchmark.md`
- `tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/cuda-benchmark.svg`
- `tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/combined-current-945016c3/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-945016c3/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-945016c3/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-945016c3/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-945016c3/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-945016c3/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/tagged-inout-benchmark-working/a100-current-55a144de/cuda-benchmark.json`
- `tmp/cuda-backend/tagged-inout-benchmark-working/h200-current-55a144de/cuda-benchmark.json`
- `tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark.json`
- `tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark.md`
- `tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark.svg`
- `tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/a100-current-943620bf/cuda-benchmark.json`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/h200-current-943620bf/cuda-benchmark.json`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark.json`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark.md`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark.svg`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/a100-current-493ce832/cuda-benchmark.json`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/h200-current-493ce832/cuda-benchmark.json`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/cuda-benchmark.json`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/cuda-benchmark.md`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/cuda-benchmark.svg`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/a100-current-ca290b2a/cuda-benchmark.json`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/h200-current-ca290b2a/cuda-benchmark.json`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark.json`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark.md`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark.svg`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/a100-current-8c023f59/cuda-benchmark.json`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/h200-current-8c023f59/cuda-benchmark.json`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark.json`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark.md`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark.svg`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark-throughput.svg`
- `tmp/cuda-backend/a100-current-a46db551/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-a46db551/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-a46db551/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-a46db551/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/a100.json`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/h200.json`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/a100.json`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/h200.json`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/a100.json`
- `tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/h200.json`
- `tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/a100.json`
- `tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/h200.json`
- `tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/cuda-smoke-report.svg`
- `tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/a100.json`
- `tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/h200.json`
- `tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/cuda-smoke-report.md`
- `tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/a100.json`
- `tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/h200.json`
- `tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/a100.json`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/h200.json`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/a100.json`
- `tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/h200.json`
- `tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/a100.json`
- `tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/h200.json`
- `tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/a100.json`
- `tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/h200.json`
- `tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/cuda-smoke-report.svg`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/a100.json`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/h200.json`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/cuda-smoke-report.md`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/cuda-smoke-report.svg`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/a100.json`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/h200.json`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/cuda-smoke-report.md`
- `tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/cuda-smoke-report.svg`
- `tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/a100.json`
- `tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/h200.json`
- `tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/cuda-smoke-report.md`
- `tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/cuda-smoke-report.svg`
- `tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/a100.json`
- `tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/h200.json`
- `tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/cuda-smoke-report.md`
- `tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/cuda-smoke-report.svg`
- `tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-sweep.json`
- `tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-sweep.md`
- `tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-sweep.svg`
- `tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-throughput.svg`
- `tmp/cuda-backend/graph-unary-benchmark-working/a100-current-f074746a/cuda-benchmark.json`
- `tmp/cuda-backend/graph-unary-benchmark-working/h200-current-f074746a/cuda-benchmark.json`
- `tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark.json`
- `tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark.md`
- `tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark.svg`
- `tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark-throughput.svg`

## Latest Graph Unary-Square Benchmark Gate

The compact paired gate at artifact label `f074746a` adds
`pto_persistent_dag_graph_unary_square` to the selected benchmark matrix. It
uses the default `16x16x16` tensor descriptor, `N=1024`, one repeat,
`batch_tasks=2`, and `worker_blocks_per_task=4`. The paired runner synced the
local tree to `bizhaoh200`, captured local A100 and remote H200 reports,
merged them, and validated the combined JSON.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark.json \
    --preset compact-current
```

The combined JSON has `78` samples. The validator checked A100/H200 machine
names, size `1024`, one repeat, selected tensor baselines, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports,
expected generated-dispatch sequences, tensor descriptor metadata, graph
descriptor fan-in/dependent metadata, graph task-argument metadata, and zero
scheduler errors for PTO persistent DAG rows.

Selected unary rows:

| GPU | Host ns | Fixed DAG ns | Graph DAG ns | Dispatch | Fan-in | Dependents |
| --- | ------- | ------------ | ------------ | -------- | ------ | ---------- |
| A100 | 44032 | 41984 | 36864 | `7,1,1` | `0,1,1` | `1,2` |
| H200 | 38592 | 32416 | 31968 | `7,1,1` | `0,1,1` | `1,2` |

The graph unary-square row uses the same one-input square task body as the
fixed DAG row, but routes it through the explicit graph descriptor adapter.
This keeps the kernel body stable while validating that graph metadata can
select unary callable arguments, produce dispatch `[7,1,1]`, and record the
descriptor as `tasks=3`, `fanin=[0,1,1]`, and `dependents=[1,2]`.

The report directory contains `cuda-benchmark.json`, `cuda-benchmark.md`,
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

## Previous Tagged Scalar Graph Benchmark Gate

The compact paired gate at artifact label `8c023f59` adds
`pto_persistent_dag_graph_tagged` to the selected benchmark matrix. It uses
the default `16x16x16` tensor descriptor, `N=1024`, one repeat,
`batch_tasks=2`, and `worker_blocks_per_task=4`. The paired runner synced the
local tree to `bizhaoh200`, captured local A100 and remote H200 reports,
merged them, and validated the combined JSON.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/tagged-scalar-compact-current-working/combined-current-8c023f59/cuda-benchmark.json \
    --preset compact-current
```

The combined JSON has `76` samples. The validator checked A100/H200 machine
names, size `1024`, one repeat, selected tensor baselines, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports,
expected generated-dispatch sequences, tensor descriptor metadata, graph
descriptor fan-in/dependent metadata, graph task-argument metadata, and zero
scheduler errors for PTO persistent DAG rows.

Selected tagged graph rows:

| GPU | Tagged scalar ns | Tagged inout ns | Dispatch | Fan-in | Dependents |
| --- | ---------------- | --------------- | -------- | ------ | ---------- |
| A100 | 46080 | 52224 | `9,2,1` | `0,0,2` | `2,2` |
| H200 | 25632 | 28032 | `9,2,1` | `0,0,2` | `2,2` |

The tagged scalar graph row records scalar task metadata:
`task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]`,
`task1=input:a,input:b,output:tmp2`, and
`task2=input:tmp1,input:tmp2,output_existing:out`, with
`scalar_args[0]=1.5` and `scalar_args[1]=0.25`.

The report directory contains `cuda-benchmark.json`, `cuda-benchmark.md`,
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

## Previous Graph Tensor-Core Benchmark Gate

The compact paired gate at artifact label `ca290b2a` revalidates
`pto_persistent_dag_graph_tensor_core` in the selected benchmark matrix. It
uses the default `16x16x16` tensor descriptor, `N=1024`, one repeat,
`batch_tasks=2`, and `worker_blocks_per_task=4`. The paired runner synced the
local tree to `bizhaoh200`, captured local A100 and remote H200 reports,
merged them, and validated the combined JSON.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/cuda-benchmark.json \
    --preset compact-current
```

The combined JSON has `74` samples. The validator checked A100/H200 machine
names, size `1024`, one repeat, selected tensor baselines, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports,
expected generated-dispatch sequences, tensor descriptor metadata, graph
descriptor fan-in/dependent metadata, graph task-argument metadata, and zero
scheduler errors for PTO persistent DAG rows.

Selected tensor rows:

| GPU | Scalar tensor ns | Graph tensor ns | Tensor-core ns | Graph tensor-core ns | cuBLAS ns | cuBLAS Graph ns |
| --- | ---------------- | --------------- | -------------- | -------------------- | --------- | --------------- |
| A100 | 41984 | 36864 | 38912 | 37888 | 53247 | 11264 |
| H200 | 40640 | 40608 | 38880 | 39424 | 50528 | 11615 |

The graph tensor-core row validates dispatch `10,1,2,1`, graph fan-in
`0,1,1,2`, dependents `1,2,3,3`, tensor tile `16x16x16`,
`wmma:m16n16k8:tf32->f32`, and target-specific PTX (`compute_80` on A100,
`compute_90` on H200). The report directory contains `cuda-benchmark.json`,
`cuda-benchmark.md`, `cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

## Previous Tensor-Arity Graph Benchmark Gate

The compact paired gate at artifact label `943620bf` adds
`pto_persistent_dag_graph_triad` and `pto_persistent_dag_graph_quad` to the
selected benchmark matrix. It uses the default `16x16x16` tensor descriptor,
`N=1024`, one repeat, `batch_tasks=2`, and `worker_blocks_per_task=4`.
The paired runner synced the local tree to `bizhaoh200`, captured local A100
and remote H200 reports, merged them, and validated the combined JSON.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/graph-tensor-arity-benchmark-working/combined-current-943620bf/cuda-benchmark.json \
    --preset compact-current
```

The combined JSON has `72` samples. The validator checked A100/H200 machine
names, size `1024`, one repeat, selected tensor baselines, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports,
expected generated-dispatch sequences, tensor descriptor metadata, graph
descriptor fan-in/dependent metadata, graph task-argument metadata, and zero
scheduler errors for PTO persistent DAG rows.

Selected tensor-arity graph rows:

| GPU | Base DAG ns | Fixed triad ns | Graph triad ns | Fixed quad ns | Graph quad ns |
| --- | ----------- | -------------- | -------------- | ------------- | ------------- |
| A100 | 47104 | 33792 | 29696 | 39936 | 33792 |
| H200 | 41472 | 35296 | 30880 | 33376 | 28704 |

Both explicit graph rows validate graph fan-in `0,0,2` and dependents `2,2`.
The triad row validates dispatch `6,2,1` with `tensor_args={"c":"tmp0"}`;
the quad row validates dispatch `8,2,1` with
`tensor_args={"c":"tmp0","d":"tmp3"}`. These are small-`N` launch
microbenchmarks; use them as shape and metadata evidence, not tuned model
kernel throughput.

The report directory contains `cuda-benchmark.json`, `cuda-benchmark.md`,
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

## Previous Tagged-Inout Compact Gate

The compact paired gate at artifact label `55a144de` adds
`pto_persistent_dag_graph_tagged_inout` to the selected benchmark matrix. It
uses the default `16x16x16` tensor descriptor, `N=1024`, one repeat,
`batch_tasks=2`, and `worker_blocks_per_task=4`. The paired runner synced the
local tree to `bizhaoh200`, captured local A100 and remote H200 reports,
merged them, and validated the combined JSON.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/cuda-benchmark.json \
    --preset compact-current
```

The combined JSON has `68` samples. The validator checked A100/H200 machine
names, size `1024`, one repeat, selected tensor baselines, the tagged-inout
graph baseline, source-paper provenance, sanitized command examples, generated
Markdown/SVG reports, expected generated-dispatch sequences, tensor descriptor
metadata, graph descriptor fan-in/dependent metadata, graph task-argument
metadata, and zero scheduler errors for PTO persistent DAG rows.

Tagged-inout rows:

| GPU | Dispatch | Fan-in | Dependents | Tagged inout task | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ----------------- | --------- | ------- | ------ |
| A100 | `1,1,1` | `0,1,1` | `1,2` | `inout:tmp1,input:b` | 35840 | 48679 | pass |
| H200 | `1,1,1` | `0,1,1` | `1,2` | `inout:tmp1,input:b` | 30080 | 40316 | pass |

The report directory contains `cuda-benchmark.json`, `cuda-benchmark.md`,
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

## Latest Tensor-Arity Graph Descriptor Smokes

The paired smoke gate at artifact label `4cd73e6a` adds explicit runtime graph
descriptor coverage for the fixed generated-dispatch triad and quad task
descriptor fields. It uses `N=1024`, two repeat launches, queue capacity `2`,
`block_dim=256`, and one scheduler block plus three worker blocks on both
local A100 and remote H200.

Validation commands:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/a100.json \
    tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-scheduler-blocks 1 --expected-worker-blocks 3 \
    --expected-worker-blocks-per-task 1 --expected-stream-id 0 \
    --expected-block-dim 256 --expected-grid-dim 4 \
    --require-report-files --expected-dag-shape graph_descriptor_triad \
    --expected-dispatch 6,2,1 --expected-graph-fanin 0,0,2 \
    --expected-graph-dependents 2,2

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/a100.json \
    tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-scheduler-blocks 1 --expected-worker-blocks 3 \
    --expected-worker-blocks-per-task 1 --expected-stream-id 0 \
    --expected-block-dim 256 --expected-grid-dim 4 \
    --require-report-files --expected-dag-shape graph_descriptor_quad \
    --expected-dispatch 8,2,1 --expected-graph-fanin 0,0,2 \
    --expected-graph-dependents 2,2
```

Results:

| Shape | GPU | Dispatch | Tensor args | Device ns | Host ns | Status |
| ----- | --- | -------- | ----------- | --------- | ------- | ------ |
| `graph_descriptor_triad` | A100 | `6,2,1` | `c=tmp0` | 64512 | 95893 | pass |
| `graph_descriptor_triad` | H200 | `6,2,1` | `c=tmp0` | 63776 | 88670 | pass |
| `graph_descriptor_quad` | A100 | `8,2,1` | `c=tmp0,d=tmp3` | 68608 | 101095 | pass |
| `graph_descriptor_quad` | H200 | `8,2,1` | `c=tmp0,d=tmp3` | 62496 | 87014 | pass |

Both shapes validated `graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`, `launch_completed_counts=[3,3]`, and
zero device scheduler errors. The report directories contain paired JSON,
Markdown, and SVG files under
`tmp/cuda-backend/graph-tensor-arity-working/`.

## Latest Graph Unary Descriptor Smoke

The paired smoke gate at artifact label `02c99b5c` validates the one-input
generated-dispatch square task through explicit runtime graph metadata. It
uses `N=1024`, two repeat launches, queue capacity `2`, `block_dim=256`, and
one scheduler block plus three worker blocks on local A100 and remote H200.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/a100.json \
    tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-scheduler-blocks 1 --expected-worker-blocks 3 \
    --expected-worker-blocks-per-task 1 --expected-stream-id 0 \
    --expected-block-dim 256 --expected-grid-dim 4 \
    --require-report-files --expected-dag-shape graph_descriptor_unary_square \
    --expected-dispatch 7,1,1 --expected-graph-fanin 0,1,1 \
    --expected-graph-dependents 1,2
```

Results:

| Shape | GPU | Dispatch | Graph fan-in | Dependents | Device ns | Host ns |
| ----- | --- | -------- | ------------ | ---------- | --------- | ------- |
| `graph_descriptor_unary_square` | A100 | `7,1,1` | `0,1,1` | `1,2` | 75776 | 106800 |
| `graph_descriptor_unary_square` | H200 | `7,1,1` | `0,1,1` | `1,2` | 57056 | 74823 |

Both artifacts validated `launch_completed_counts=[3,3]`, zero device
scheduler errors, and generated Markdown/SVG report files under
`tmp/cuda-backend/graph-unary-square-working/`.

## Previous Graph-Generic Compact Gate

The compact paired gate at commit `b2c5c8a4` uses a WMMA-compatible
`16x16x16` tensor descriptor, `N=1024`, one repeat, `batch_tasks=2`, and
`worker_blocks_per_task=4`. The paired runner synced the local tree to
`bizhaoh200`, captured local A100 and remote H200 reports, merged them, and
validated the combined JSON with the then-current compact preset.

Original validation command at capture time:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark.json \
    --preset compact-current
```

The combined JSON has `60` samples: `30` selected benchmark rows per GPU,
including selected one-task baselines, same-work batch rows, and the worker
grid batch row. The validator checked A100/H200 machine names, size `1024`,
one repeat, selected tensor baselines, `pto_host_schedule_generic_args`,
`pto_persistent_dag_graph_generic_args4`, source-paper provenance, sanitized
command examples, generated Markdown/SVG reports, expected
generated-dispatch sequences, tensor descriptor metadata, and zero scheduler
errors for PTO persistent DAG rows.

Selected rows:

| GPU | Host ns | Host generic ns | Base DAG ns | Persistent generic ns | Graph generic4 ns | Tensor ns | Tensor-core ns | cuBLAS ns | Grid batch ns |
| --- | ------- | --------------- | ----------- | --------------------- | ----------------- | --------- | -------------- | --------- | ------------- |
| A100 | 22528 | 35840 | 44032 | 29696 | 27648 | 37888 | 36864 | 37888 | 37888 |
| H200 | 16992 | 31264 | 40320 | 30592 | 27520 | 48992 | 32480 | 34304 | 31872 |

Launch baseline comparison from the same raw JSON:

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 22528 | 20480 | 31743 | 22528 | 0.91x | 1.00x |
| H200 | 1024 | 16992 | 13376 | 21312 | 17247 | 0.79x | 1.02x |

Selected tensor throughput from the same raw JSON:

| GPU | N | Shape | Scalar ns | Graph ns | Tensor-core ns | cuBLAS ns | Scalar GF/s | Graph GF/s | Tensor-core GF/s | cuBLAS GF/s | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | --------- | -------- | -------------- | --------- | ----------- | ---------- | ---------------- | ----------- | ------------------ | ------------- |
| A100 | 1024 | 16x16x16 | 37888 | 41984 | 36864 | 37888 | 0.86 | 0.78 | 0.89 | 0.86 | 0.97x | 1.00x |
| H200 | 1024 | 16x16x16 | 48992 | 47264 | 32480 | 34304 | 0.67 | 0.69 | 1.01 | 0.96 | 0.66x | 0.70x |

Worker-grid result:

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 4 | 37888 | 1.32x |
| H200 | 1024 | 2 | 4 | 31872 | 1.60x |

The report directory contains `cuda-benchmark.json`, `cuda-benchmark.md`,
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`. This
capture is retained as the previous compact gate that added the
graph-generic-args4 benchmark promotion.

## Previous Generic-Args Compact Gate

The compact paired gate at commit `2aedb40f` uses a
WMMA-compatible `16x16x16` tensor descriptor, `N=1024`, one repeat,
`batch_tasks=2`, and `worker_blocks_per_task=4`. The paired runner synced the
local tree to `bizhaoh200`, captured A100 and H200 reports, merged them, and
validated the combined JSON with the then-current compact preset. That capture
checks all 29 selected baselines on A100 and H200, including
`pto_host_schedule_generic_args`, source-paper provenance, sanitized command
examples, generated Markdown/SVG report files, dispatch sequences, tensor tile
metadata, and zero scheduler errors.

This capture is intentionally retained as previous evidence before the
graph-generic-args4 benchmark promotion. The `b2c5c8a4` gate superseded it at
that point; the later `dbb01406` graph-scratch-reuse gate superseded it for
explicit graph scratch-reuse evidence.

Original validation command at capture time:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark.json \
    --preset compact-current
```

Selected rows:

| GPU | Host ns | Host generic ns | Base DAG ns | Persistent generic ns | Tensor ns | Tensor-core ns | cuBLAS ns | Grid batch ns |
| --- | ------- | --------------- | ----------- | --------------------- | --------- | -------------- | --------- | ------------- |
| A100 | 29696 | 43008 | 44032 | 29696 | 41984 | 41984 | 49152 | 40960 |
| H200 | 17920 | 36032 | 38112 | 31168 | 47008 | 32543 | 51520 | 32896 |

Additional graph and scalar rows:

| GPU | Scalar scale ns | Graph diamond ns | Graph tensor ns |
| --- | --------------- | ---------------- | --------------- |
| A100 | 38912 | 41984 | 39936 |
| H200 | 33952 | 35616 | 45536 |

Selected tensor throughput from the same raw JSON:

| GPU | N | Shape | Scalar ns | Graph ns | Tensor-core ns | cuBLAS ns | Scalar GF/s | Graph GF/s | Tensor-core GF/s | cuBLAS GF/s | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | --------- | -------- | -------------- | --------- | ----------- | ---------- | ---------------- | ----------- | ------------------ | ------------- |
| A100 | 1024 | 16x16x16 | 41984 | 39936 | 41984 | 49152 | 0.78 | 0.82 | 0.78 | 0.67 | 1.00x | 1.17x |
| H200 | 1024 | 16x16x16 | 47008 | 45536 | 32543 | 51520 | 0.70 | 0.72 | 1.01 | 0.64 | 0.69x | 1.10x |

This capture is a gate for command construction, validation coverage, and
real A100/H200 execution before the graph-generic-args4 benchmark promotion.
The validator also checked expected generated-dispatch IDs and tensor
descriptor shapes for tensor rows. The combined JSON has `58` samples. It is
intentionally smaller than the full `61cf96cd` capture and should not replace
the three-size,
three-repeat rows below for broad trend reading. The `2aedb40f` gate was
captured after adding the host-schedule generic-args benchmark row; all PTO
persistent DAG rows reported zero device scheduler errors.

## Current Compact Paired Gate

The compact paired gate at artifact label `dbb01406` promotes
`pto_persistent_dag_graph_scratch_reuse` into the selected benchmark path
after the previous `06b8c0c6` graph-chain gate. It uses the current compact
preset shape: `N=1024`, one repeat, `batch_tasks=2`,
`worker_blocks_per_task=4`, and the default `16x16x16` tensor descriptor.
The paired runner synced the local tree to `bizhaoh200`, captured local A100
and remote H200 reports, merged `64` rows, and validated the combined JSON
with source-paper provenance, sanitized command examples, report files, tensor
descriptor metadata, expected dispatch sequences, graph descriptor
fan-in/dependent metadata, and zero PTO scheduler errors.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark.json \
    --require-machine hina --require-machine dasys-h200x8 \
    --require-size 1024 --expected-repeats 1 --expected-result-count 64 \
    --require-baseline pto_persistent_dag_graph_scratch_reuse \
    --require-dispatch pto_persistent_dag_graph_scratch_reuse=1,2,1,2,1,1 \
    --require-scratch-reuse pto_persistent_dag_graph_scratch_reuse=reused_buffer=tmp0,reuse_task=4 \
    --require-tensor-tile pto_persistent_dag_tensor=16x16x16 \
    --require-tensor-tile pto_persistent_dag_graph_tensor=16x16x16 \
    --require-tensor-tile pto_persistent_dag_tensor_core=16x16x16 \
    --require-tensor-tile cublas_sgemm=16x16x16 \
    --require-report-files --require-command-examples \
    --require-zero-scheduler-errors --require-source-papers
```

Graph-chain and scratch-reuse rows from the capture:

- A100 fixed chain: dispatch `1,2,1,2,1`, device `50176 ns`,
  host `64959 ns`, status `pass`.
- A100 graph chain: dispatch `1,2,1,2,1`, fan-in `0,0,2,1,1`,
  dependents `2,2,3,4`, device `35840 ns`, host `46873 ns`, status `pass`.
- A100 fixed scratch reuse: dispatch `1,2,1,2,1,1`, scratch `tmp0@4`,
  device `56320 ns`, host `70489 ns`, status `pass`.
- A100 graph scratch reuse: dispatch `1,2,1,2,1,1`, fan-in
  `0,0,2,1,1,2`, dependents `2,2,3,4,5,5`, scratch `tmp0@4`,
  device `36864 ns`, host `47528 ns`, status `pass`.
- H200 fixed chain: dispatch `1,2,1,2,1`, device `45952 ns`,
  host `60262 ns`, status `pass`.
- H200 graph chain: dispatch `1,2,1,2,1`, fan-in `0,0,2,1,1`,
  dependents `2,2,3,4`, device `34528 ns`, host `44168 ns`, status `pass`.
- H200 fixed scratch reuse: dispatch `1,2,1,2,1,1`, scratch `tmp0@4`,
  device `48576 ns`, host `62052 ns`, status `pass`.
- H200 graph scratch reuse: dispatch `1,2,1,2,1,1`, fan-in
  `0,0,2,1,1,2`, dependents `2,2,3,4,5,5`, scratch `tmp0@4`,
  device `38240 ns`, host `48080 ns`, status `pass`.

The generated DAG-shapes summary for this compact capture now includes
`Graph Scratch Reuse/DAG`: A100 reports `0.86x` and H200 reports `1.05x`
relative to the matched base `pto_persistent_dag` row.

This capture makes the explicit graph-descriptor scratch-reuse shape visible
in the same benchmark/report/validator path as the selected graph,
graph-generic, chain, diamond, and tensor graph rows. It is still a compact
one-size gate, not a replacement for the full three-size paired capture.

## cuBLAS CUDA Graph Baseline Row

The compact paired run under
`tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/`
adds `cublas_sgemm_graph` to the selected one-task benchmark rows. It uses
`N=1024`, one repeat, no batch rows, and the default `16x16x16` tensor
descriptor. The paired runner synced the working tree to `bizhaoh200`,
captured local A100 and remote H200 reports, merged `58` rows, and validated
source-paper provenance, sanitized command examples, report files, tensor
descriptor metadata, expected PTO dispatch sequences, and zero scheduler
errors.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/cuda-benchmark.json \
    --require-size 1024 --expected-repeats 1 --expected-result-count 58 \
    --require-baseline cublas_sgemm --require-baseline cublas_sgemm_graph \
    --require-tensor-tile cublas_sgemm=16x16x16 \
    --require-tensor-tile cublas_sgemm_graph=16x16x16 \
    --require-report-files --require-command-examples \
    --require-zero-scheduler-errors --require-source-papers
```

Selected tensor rows from that capture:

| GPU | N | Shape | Scalar ns | Graph ns | Tensor-core ns | cuBLAS ns | cuBLAS graph ns | Scalar GF/s | Graph GF/s | Tensor-core GF/s | cuBLAS GF/s | cuBLAS graph GF/s | Tensor-core/scalar | cuBLAS/scalar | cuBLAS graph/scalar |
| --- | - | ----- | --------- | -------- | -------------- | --------- | --------------- | ----------- | ---------- | ---------------- | ----------- | ----------------- | ------------------ | ------------- | ------------------- |
| A100 | 1024 | 16x16x16 | 43008 | 41984 | 38912 | 48128 | 10239 | 0.76 | 0.78 | 0.84 | 0.68 | 3.20 | 0.90x | 1.12x | 0.24x |
| H200 | 1024 | 16x16x16 | 45920 | 47584 | 30944 | 9119 | 8543 | 0.71 | 0.69 | 1.06 | 3.59 | 3.84 | 0.67x | 0.20x | 0.19x |

The graph row uses the same cuBLAS SGEMM descriptor and maximum absolute
error as the plain cuBLAS row, but captures the warmed call sequence into a
CUDA Graph and excludes graph instantiation from the measured interval. At
this small descriptor size, the graph replay row mostly measures CUDA Graph
launch overhead around a library call; it is a launch-path comparison point,
not a tuned GEMM throughput claim.

## Supplemental Scalar-Scale Benchmark

The compact scalar-scale benchmark gate at artifact label `a46db551` adds
`pto_persistent_dag_scalar_scale` to the selected paired benchmark path. It
uses `N=4096`, one repeat, no batch rows, and the default `16x16x16` tensor
descriptor metadata. The paired runner synced the local tree to `bizhaoh200`,
captured A100 and H200 benchmark reports, merged `44` rows, and validated
required baselines, command examples, source-paper provenance, zero scheduler
errors, and generated Markdown/SVG report files.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.json \
    --require-size 4096 --expected-repeats 1 --expected-result-count 44 \
    --require-baseline pto_persistent_dag_scalar_scale \
    --require-report-files --require-command-examples \
    --require-zero-scheduler-errors --require-source-papers
```

| GPU | Baseline | Dispatch | Scalar args | Device ns | Host ns | Status |
| --- | -------- | -------- | ----------- | --------- | ------- | ------ |
| A100 | `pto_persistent_dag_scalar_scale` | `11,2,1` | `scalar0=2.0` | 37888 | 55626 | pass |
| H200 | `pto_persistent_dag_scalar_scale` | `11,2,1` | `scalar0=2.0` | 27744 | 2498273 | pass |

Both rows reported zero device scheduler errors and the report includes
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`, and
`cuda-benchmark-dag-deltas.svg`. The high H200 host time is launch-side noise
in this one-repeat compact gate; the device event time is the useful
scheduler-path signal.

## Supplemental Scalar-Scale Smoke

The scalar-scale persistent DAG smoke at artifact label `e9c9f5f2` validates a
single-input scalar task descriptor outside the scene-test framework. It uses
dispatch sequence `[11,2,1]`: scale `tmp0 = scalar0 * a`, multiply
`tmp1 = a * b`, then add `out = tmp0 + tmp1`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/a100.json \
    tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 1 --expected-completed-count 3 \
    --require-report-files --expected-dag-shape scalar_scale \
    --expected-dispatch 11,2,1
```

| GPU | PTX arch | Dispatch | Scalar args | Device ns | Host ns | Status |
| --- | -------- | -------- | ----------- | --------- | ------- | ------ |
| A100 | `compute_80` | `11,2,1` | `scalar0=2.0` | 40960 | 61301 | pass |
| H200 | `compute_90` | `11,2,1` | `scalar0=2.0` | 25856 | 34808 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. This capture is correctness evidence for the single-input
scalar descriptor and generated-dispatch registration, not a benchmark
replacement for the multi-baseline captures.

## Supplemental Graph Scalar-Scale Smoke

The graph scalar-scale persistent DAG smoke at artifact label `15e9038f`
validates the same scalar-scale task body as explicit runtime graph descriptor
metadata. It uses dispatch sequence `[11,2,1]`, graph fan-in `[0,0,2]`,
dependents `[2,2]`, and `scalar0=2.0`, then repeats the prepared callable
twice.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-scalar-scale-working/persistent-graph_descriptor_scalar_scale-repeat2-smoke-15e9038f/a100.json \
    tmp/cuda-backend/graph-scalar-scale-working/persistent-graph_descriptor_scalar_scale-repeat2-smoke-15e9038f/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-dag-shape graph_descriptor_scalar_scale \
    --expected-dispatch 11,2,1 \
    --expected-graph-fanin 0,0,2 --expected-graph-dependents 2,2 \
    --require-report-files
```

| GPU | PTX arch | Dispatch | Fan-in | Dependents | Device ns | Host ns | Status |
| --- | -------- | -------- | ------ | ---------- | --------- | ------- | ------ |
| A100 | `compute_80` | `11,2,1` | `0,0,2` | `2,2` | 53248 | 82872 | pass |
| H200 | `compute_90` | `11,2,1` | `0,0,2` | `2,2` | 59712 | 84805 | pass |

Both rows reported zero device scheduler errors, scalar args `scalar0=2.0`,
and launch completion counts `[3,3]`.

## Supplemental Graph Scalar Variant Smokes

The graph scalar AXPY and affine persistent DAG smokes at artifact label
`20af4fd9` validate the remaining fixed scalar task bodies as explicit runtime
graph descriptor metadata. Both use graph fan-in `[0,0,2]`, dependents
`[2,2]`, and repeat the prepared callable twice.

Validation commands:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-scalar-variants-working/persistent-graph_descriptor_scalar_axpy-repeat2-smoke-20af4fd9/a100.json \
    tmp/cuda-backend/graph-scalar-variants-working/persistent-graph_descriptor_scalar_axpy-repeat2-smoke-20af4fd9/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-dag-shape graph_descriptor_scalar_axpy \
    --expected-dispatch 4,2,1 \
    --expected-graph-fanin 0,0,2 --expected-graph-dependents 2,2 \
    --require-report-files

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-scalar-variants-working/persistent-graph_descriptor_scalar_affine-repeat2-smoke-20af4fd9/a100.json \
    tmp/cuda-backend/graph-scalar-variants-working/persistent-graph_descriptor_scalar_affine-repeat2-smoke-20af4fd9/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-dag-shape graph_descriptor_scalar_affine \
    --expected-dispatch 5,2,1 \
    --expected-graph-fanin 0,0,2 --expected-graph-dependents 2,2 \
    --require-report-files
```

| Shape | GPU | PTX arch | Dispatch | Scalar args | Device ns | Host ns | Status |
| ----- | --- | -------- | -------- | ----------- | --------- | ------- | ------ |
| AXPY | A100 | `compute_80` | `4,2,1` | `scalar0=1.5` | 62464 | 93331 | pass |
| AXPY | H200 | `compute_90` | `4,2,1` | `scalar0=1.5` | 59072 | 83500 | pass |
| Affine | A100 | `compute_80` | `5,2,1` | `scalar0=1.5,scalar1=0.5` | 67584 | 99491 | pass |
| Affine | H200 | `compute_90` | `5,2,1` | `scalar0=1.5,scalar1=0.5` | 59616 | 84826 | pass |

All rows reported zero device scheduler errors, graph metadata
`fanin=[0,0,2]`, `dependents=[2,2]`, and launch completion counts `[3,3]`.

## Supplemental Generic-Args Repeat-Run Smoke

The `generic_args` persistent DAG smoke at artifact label `6574c43b`
validates prepared-callable lifecycle reuse for the indexed tensor/scalar
descriptor path. It prepares one generated-dispatch callable, then runs it
twice after resetting fan-in, ready flags, counters, and scratch/output
buffers. The dispatch sequence is `[9,2,1]`; task `9` reads
`tensor_args[0]=tmp0`, `tensor_args[1]=tmp3`, `scalar_args[0]=1.5`, and
`scalar_args[1]=0.25`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/a100.json \
    tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape generic_args --expected-repeat-runs 2 \
    --expected-completed-count 3 --expected-dispatch 9,2,1 \
    --require-report-files
```

| GPU | Dispatch | Repeat runs | Launch completions | Launch device ns | Device ns | Host ns | Status |
| --- | -------- | ----------- | ------------------ | ---------------- | --------- | ------- | ------ |
| A100 | `9,2,1` | 2 | `3,3` | `44032,25600` | 69632 | 106096 | pass |
| H200 | `9,2,1` | 2 | `3,3` | `21088,19808` | 40896 | 4953113 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. The high H200 host time is launch-side noise in this single
capture; the per-launch CUDA event times are the useful lifecycle signal.

## Supplemental Four-Slot Graph-Descriptor Smoke

The four-slot graph-descriptor persistent DAG smoke at artifact label
`11db2c9d` validates that the explicit graph descriptor path carries the same
generic tensor/scalar argument packet as the direct `generic_args4` DAG shape.
It records graph fan-in `[0,0,2]`, dependents `[2,2]`, dispatch `9,2,1`,
and four generic slots:
`tensor_args[0]=tmp0`, `tensor_args[1]=tmp3`, `tensor_args[2]=a`,
`tensor_args[3]=b`, with scalar slots `[1.5,0.25,0.125,0.0625]`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_generic_args4 \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-dispatch 9,2,1 --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `9,2,1` | `0,0,2` | `2,2` | `3,3` | 52224 | 390302 | pass |
| H200 | `9,2,1` | `0,0,2` | `2,2` | `3,3` | 45280 | 62740 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. The matching L2 `SceneTestCase` selection also passed on H200
with `2 passed, 60 deselected` after the known PTO-ISA SSH refresh warning.

## Supplemental Four-Slot Graph-Descriptor Benchmark

The graph-descriptor four-slot path is now also a selected benchmark baseline
as `pto_persistent_dag_graph_generic_args4`. A quick single-baseline capture
uses `N=4096` and the same generated-dispatch sequence as the smoke artifact,
but goes through `cuda_benchmark.py` so paired-current validation can require
the row in future full captures.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/a100.json \
    tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_generic_args4 \
    --expected-completed-count 3 --expected-dispatch 9,2,1 \
    --expected-scheduler-blocks 1 --expected-worker-blocks 3 \
    --expected-worker-blocks-per-task 1 --expected-stream-id 0 \
    --expected-block-dim 256 --expected-grid-dim 4 --require-report-files
```

| GPU | Baseline | N | Dispatch | Device ns | Host ns | Status |
| --- | -------- | - | -------- | --------- | ------- | ------ |
| A100 | `pto_persistent_dag_graph_generic_args4` | 4096 | `9,2,1` | 43008 | 58143 | pass |
| H200 | `pto_persistent_dag_graph_generic_args4` | 4096 | `9,2,1` | 33664 | 43163 | pass |

Both rows reported zero device scheduler errors, graph fan-in `[0,0,2]`,
dependents `[2,2]`, and all four generic tensor/scalar slots.

## Supplemental Reordered Graph-Descriptor Smoke

The reordered graph-descriptor persistent DAG smoke at artifact label
`f877b7b3` validates order-independent tensor-flow dependency inference. The
runtime task list is `[final add, generic args, multiply]`, so the final
consumer has task id `0` but starts with fan-in `2`; both producer tasks point
back to it through dependents `[0,0]`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_reordered --expected-repeat-runs 2 \
    --expected-completed-count 3 --expected-dispatch 1,9,2 \
    --expected-graph-fanin 2,0,0 --expected-graph-dependents 0,0 \
    --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `1,9,2` | `2,0,0` | `0,0` | `3,3` | 63488 | 91185 | pass |
| H200 | `1,9,2` | `2,0,0` | `0,0` | `3,3` | 46240 | 63520 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. This is correctness evidence for graph lowering; the task body
and arithmetic are the same as the generic-args graph descriptor.

The tagged graph-descriptor paired smoke at artifact label `a618e624`
validates the first TaskArgs-like lowering slice in the paired report flow,
including scalar task-arg entries.
It lowers tagged task entries to the same three-task graph descriptor:
`graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`, dispatch `9,2,1`, and
`graph_task_args` metadata for
`input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]`,
`input:a,input:b,output:tmp2`, and
`input:tmp1,input:tmp2,output_existing:out`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/a100.json \
    tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_tagged \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-dispatch 9,2,1 \
    --expected-graph-fanin 0,0,2 \
    --expected-graph-dependents 2,2 \
    --expected-graph-task-args \
      'task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out' \
    --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `9,2,1` | `0,0,2` | `2,2` | `3,3` | 68608 | 98185 | pass |
| H200 | `9,2,1` | `0,0,2` | `2,2` | `3,3` | 43488 | 61711 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. The regenerated report and artifact index include a
`Graph task args` field, so the visible evidence shows both the lowered
fan-in/dependent topology and the original tagged task-argument roles. This is
still host-side descriptor construction, but it proves the tagged
task-argument representation can feed the same persistent-device scheduler
path on A100 and H200. The paired validator also checks
`--expected-graph-task-args`, so the artifact fails validation if those tagged
roles disappear from either GPU payload.

The role-keyed inout graph-descriptor paired smoke extends the same
three-task in-place shape to the preferred `role` task-argument spelling. The
artifact is under
`tmp/cuda-backend/role-keyed-inout-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-5075b400/`.
It records `graph_task_arg_key=role` beside the same graph task metadata:
`input:a,input:b,output:tmp1`, `inout:tmp1,input:b`, and
`input:tmp1,input:a,output_existing:out`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/role-keyed-inout-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-5075b400/a100.json \
    tmp/cuda-backend/role-keyed-inout-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-5075b400/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_role_keyed_inout \
    --expected-repeat-runs 2 --expected-completed-count 3 \
    --expected-dispatch 1,1,1 \
    --expected-graph-fanin 0,1,1 \
    --expected-graph-dependents 1,2 \
    --expected-graph-task-arg-key role \
    --expected-graph-task-args \
      'task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;task2=input:tmp1,input:a,output_existing:out' \
    --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `1,1,1` | `0,1,1` | `1,2` | `3,3` | 69632 | 103494 | pass |
| H200 | `1,1,1` | `0,1,1` | `1,2` | `3,3` | 67232 | 91251 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. The report includes a `Graph task arg key` column with `role`
for both GPUs, so this artifact can catch regressions where role-keyed
TaskArgs-like metadata is accidentally reported only as generic tagged
metadata.

The role-keyed graph descriptor is also in the compact selected-benchmark
path as `pto_persistent_dag_graph_role_keyed_inout`. The artifact under
`tmp/cuda-backend/role-keyed-benchmark-working/combined-current-a7787008/`
was produced with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/role-keyed-benchmark-working
```

The paired capture validator checked `72` rows, source-paper provenance,
sanitized command examples, report files, zero scheduler errors, dispatch
`1,1,1`, graph fan-in `0,1,1`, dependents `1,2`, the role-keyed graph task
args, and `graph_task_arg_key=role`.

| GPU | Baseline | Device ns | Host ns | Graph task arg key | Status |
| --- | -------- | --------- | ------- | ------------------ | ------ |
| A100 | `pto_persistent_dag_graph_role_keyed_inout` | 38912 | 52853 | `role` | pass |
| H200 | `pto_persistent_dag_graph_role_keyed_inout` | 20864 | 2446166 | `role` | pass |

The diamond graph-descriptor paired smoke at artifact label `072e396c`
validates a wider explicit descriptor shape than the three-task
graph-descriptor and reordered-graph smokes. It has two root producers, two
fan-out consumers, and one final join:
`graph_descriptor.fanin=[0,0,2,2,2]`,
`graph_descriptor.dependents=[2,3,2,3,4,4]`, and dispatch `9,2,1,2,1`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_diamond \
    --expected-repeat-runs 2 --expected-completed-count 5 \
    --expected-dispatch 9,2,1,2,1 \
    --expected-graph-fanin 0,0,2,2,2 \
    --expected-graph-dependents 2,3,2,3,4,4 \
    --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `9,2,1,2,1` | `0,0,2,2,2` | `2,3,2,3,4,4` | `5,5` | 80896 | 111293 | pass |
| H200 | `9,2,1,2,1` | `0,0,2,2,2` | `2,3,2,3,4,4` | `5,5` | 47616 | 4912047 | pass |

The first paired run exposed a repeat-run reset bug because `tmp0` and
`tmp3` are both input descriptor tensors and later scratch outputs in this
shape. The fixed smoke keeps immutable seed buffers for launch reset, then
the paired validator accepted both A100 and H200 rows with zero scheduler
errors.

The scratch-reuse graph-descriptor paired smoke at artifact label `d8f6d0bf`
validates that the explicit runtime graph descriptor can express the same
six-task scratch-reuse shape as the fixed `scratch_reuse` DAG. It records
`graph_descriptor.fanin=[0,0,2,1,1,2]`,
`graph_descriptor.dependents=[2,2,3,4,5,5]`, dispatch `1,2,1,2,1,1`, and
`scratch_reuse.reused_buffer=tmp0`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_scratch_reuse \
    --expected-repeat-runs 2 --expected-completed-count 6 \
    --expected-dispatch 1,2,1,2,1,1 \
    --expected-graph-fanin 0,0,2,1,1,2 \
    --expected-graph-dependents 2,2,3,4,5,5 \
    --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `1,2,1,2,1,1` | `0,0,2,1,1,2` | `2,2,3,4,5,5` | `6,6` | 89088 | 121166 | pass |
| H200 | `1,2,1,2,1,1` | `0,0,2,1,1,2` | `2,2,3,4,5,5` | `6,6` | 66144 | 84089 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. This is graph-lowering breadth evidence rather than a new
throughput baseline; the arithmetic and scratch lifetime match the fixed
`scratch_reuse` DAG shape.

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It can
reduce launch overhead in some rows, but it is still host-owned replay rather
than a device-side scheduler.

The `pto_host_schedule_compiler` row validates the task-body compiler path.
It uses the same host runtime path as `pto_host_schedule`, but the PTX comes
from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)` and
the shared task wrapper generator.

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 32768 | 30720 | 23552 | 15359 | 0.94x | 0.47x |
| A100 | 65536 | 21440 | 20096 | 26591 | 17823 | 0.94x | 0.83x |
| A100 | 1048576 | 301728 | 301696 | 233919 | 328319 | 1.00x | 1.09x |
| H200 | 1024 | 35968 | 40192 | 35392 | 30239 | 1.12x | 0.84x |
| H200 | 65536 | 23488 | 25408 | 39039 | 31456 | 1.08x | 1.34x |
| H200 | 1048576 | 31072 | 31264 | 38623 | 28480 | 1.01x | 0.92x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 large-vector launch rows are noisy in this capture, so use the
paired report as current smoke/evaluation evidence rather than a final
launch-overhead ranking.

## Host-Schedule Shape Rows

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The
`pto_host_schedule_quad` row validates the generated four-input
`(a, b, c, d, out, n)` ABI. Both rows compare against CUDA `float32`
goldens, which matters at larger sizes where exact Python integer arithmetic
does not match single-precision output.

| GPU | N | Unary square ns | Quad ns |
| --- | - | --------------- | ------- |
| A100 | 1024 | 33792 | 32768 |
| A100 | 65536 | 23616 | 20768 |
| A100 | 1048576 | 304128 | 545600 |
| H200 | 1024 | 39296 | 41216 |
| H200 | 65536 | 25056 | 22624 |
| H200 | 1048576 | 37312 | 45728 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 256 | 31744 | 1.19x |
| A100 | 1024 | 6 | 32 | 32768 | 0.36x |
| A100 | 1024 | 12 | 32 | 32768 | 0.21x |
| A100 | 65536 | 2 | 64 | 22528 | 0.85x |
| A100 | 65536 | 6 | 128 | 20480 | 0.39x |
| A100 | 65536 | 12 | 32 | 20480 | 0.24x |
| A100 | 1048576 | 2 | 256 | 22528 | 0.07x |
| A100 | 1048576 | 6 | 128 | 33792 | 0.09x |
| A100 | 1048576 | 12 | 256 | 58368 | 0.14x |
| H200 | 1024 | 2 | 64 | 35776 | 1.20x |
| H200 | 1024 | 6 | 32 | 35072 | 0.38x |
| H200 | 1024 | 12 | 128 | 35104 | 0.23x |
| H200 | 65536 | 2 | 256 | 25312 | 0.74x |
| H200 | 65536 | 6 | 256 | 24544 | 0.32x |
| H200 | 65536 | 12 | 32 | 25280 | 0.24x |
| H200 | 1048576 | 2 | 256 | 32864 | 0.78x |
| H200 | 1048576 | 6 | 256 | 39296 | 0.55x |
| H200 | 1048576 | 12 | 256 | 46816 | 0.39x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 worker
grid rows stay below matched host-schedule batch rows for most multi-task
rows in this capture, with the two-task small-vector points as the
current exception.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine, triad, quad, generic-args, graph-descriptor, and
unary-square rows use generated dispatch and different descriptor
fields/task-body arities without changing the persistent launch path.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Quad/DAG | Generic Args/DAG | Graph Descriptor/DAG | Unary Square/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | -------- | ---------------- | -------------------- | ---------------- | ---------- |
| A100 | 1024 | 2.04x | 1.81x | 1.19x | 1.38x | 1.15x | 1.19x | 1.19x | 1.08x | 1.15x | 1.46x |
| A100 | 65536 | 1.80x | 1.80x | 1.02x | 1.01x | 1.10x | 1.20x | 1.19x | 1.19x | 1.45x | 3.88x |
| A100 | 1048576 | 1.57x | 1.58x | 1.04x | 1.02x | 1.04x | 1.12x | 1.03x | 1.05x | 1.40x | 3.29x |
| H200 | 1024 | 1.32x | 1.38x | 1.02x | 1.03x | 0.98x | 0.96x | 0.96x | 0.95x | 1.04x | 1.24x |
| H200 | 65536 | 1.75x | 1.76x | 0.99x | 0.99x | 1.00x | 1.06x | 1.06x | 1.05x | 1.41x | 2.87x |
| H200 | 1048576 | 1.79x | 1.78x | 0.99x | 1.00x | 1.00x | 1.01x | 0.99x | 1.00x | 1.36x | 3.09x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, triad, quad, generic-args, graph-descriptor,
and unary-square rows prove mixed tensor/scalar fields, extra tensor pointers
through a fourth tensor task descriptor field, generic indexed argument
slots, explicit runtime graph lowering, and unary task-body lowering.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. Treat these DAG-shape rows as correctness and scheduler-shape
evidence first; throughput conclusions require a tuned tensor workload.

The compact paired benchmark at artifact label `945016c3` adds the wider
`pto_persistent_dag_graph_diamond` benchmark row. It uses `N=1024`, one
repeat, no batch rows, and validates `48` combined A100/H200 rows with source
paper provenance, command examples, report files, and zero scheduler errors.

| GPU | Base DAG ns | Graph Diamond ns | Graph Diamond/DAG | Dispatch | Tasks |
| --- | ----------- | ---------------- | ----------------- | -------- | ----- |
| A100 | 45056 | 36864 | 0.82x | `9,2,1,2,1` | 5 |
| H200 | 35488 | 31744 | 0.89x | `9,2,1,2,1` | 5 |

The row is a graph-lowering and scheduling-shape check, not an equal-work
throughput comparison: the diamond graph has two roots, two fan-out consumers,
and a final join, while the base DAG has three elementwise tasks.

## Supplemental Tensor Shape Sweep

The tensor DAG also has a small model-shaped descriptor sweep at commit
`c0ada3ad`. It runs only `pto_persistent_dag_tensor`, so it is shape and
scheduler evidence rather than a replacement for the paired baseline capture.
Each row uses `N=4096`, two repeats, and the generated-dispatch sequence
`[3,1,2,1]` on both A100 and H200. The raw JSON, Markdown, and SVG artifacts
are under `tmp/cuda-backend/tensor-shape-sweep-c0ada3ad/`.

| GPU | Shape | Tiles | Median device ns | Median host ns | Status |
| --- | ----- | ----- | ---------------- | -------------- | ------ |
| A100 | 8x4x12 | 128 | 68096 | 83972.5 | pass |
| A100 | 16x16x64 | 16 | 169472 | 185079 | pass |
| A100 | 32x16x64 | 8 | 132608 | 148213 | pass |
| H200 | 8x4x12 | 128 | 58336 | 68226 | pass |
| H200 | 16x16x64 | 16 | 109936 | 120146 | pass |
| H200 | 32x16x64 | 8 | 98064 | 107936.5 | pass |

This extends the earlier non-square descriptor smoke from a single tile shape
to three descriptor families that are closer to model-kernel tile shapes. The
kernel body is still scalar tiled GEMM followed by elementwise residual/gate
work, so the result should not be read as tensor-core throughput.

## Supplemental Tensor Baseline Sweep

The current multi-baseline tensor size sweep was captured at commit
`e79edba2` under `tmp/cuda-backend/tensor-shape-sweep-e79edba2/`. It runs
three repeats for a WMMA-compatible `16x16x16` descriptor at `N=256`, `4096`,
and `65536`, comparing the scalar tensor DAG,
`pto_persistent_dag_tensor_core`, and `cublas_sgemm` in one Markdown/SVG
report. The Markdown keeps raw repeat rows plus medians, records VDCores/MPK
source-paper provenance and per-baseline workload descriptions, and the SVG
plots cover median device time and median GFLOP/s with sample counts. The
table below reports median device time and normalized GFLOP/s across the
three samples.

| GPU | N | Shape | Scalar tensor ns | Tensor-core ns | cuBLAS ns | Scalar GF/s | Tensor-core GF/s | cuBLAS GF/s | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | ---------------- | -------------- | --------- | ----------- | ---------------- | ----------- | ------------------ | ------------- |
| A100 | 256 | 16x16x16 | 47104 | 47104 | 43007 | 0.17 | 0.17 | 0.19 | 1.00x | 0.91x |
| A100 | 4096 | 16x16x16 | 79872 | 71680 | 36864 | 1.64 | 1.83 | 3.56 | 0.90x | 0.46x |
| A100 | 65536 | 16x16x16 | 587616 | 470368 | 38911 | 3.57 | 4.46 | 53.90 | 0.80x | 0.07x |
| H200 | 256 | 16x16x16 | 30560 | 28160 | 50496 | 0.27 | 0.29 | 0.16 | 0.92x | 1.65x |
| H200 | 4096 | 16x16x16 | 88576 | 49888 | 37055 | 1.48 | 2.63 | 3.54 | 0.56x | 0.42x |
| H200 | 65536 | 16x16x16 | 1032896 | 390368 | 36127 | 2.03 | 5.37 | 58.05 | 0.38x | 0.03x |

The tensor-core rows use dispatch `10,1,2,1`, while the scalar tensor rows use
`3,1,2,1`. cuBLAS rows have no PTO dispatch sequence because they run through
CUDA Runtime API plus cuBLAS directly. The tensor-core PTO row improves over
the scalar tensor DAG as the number of tiles grows, especially on H200, but
the normalized throughput still stays in the single-digit GFLOP/s range
because the current PTO path schedules one small generated task per tile. The
cuBLAS path reaches about `54` GFLOP/s on A100 and `58` GFLOP/s on H200 at
`N=65536` because it uses a tuned library implementation. This remains a
compact descriptor/scheduler comparison rather than a tuned GEMM throughput
result.

A current-head one-repeat follow-up under
`tmp/cuda-backend/tensor-shape-sweep-0e84fd26/` adds
`pto_persistent_dag_graph_tensor` to the same tensor-baseline sweep family.
It uses the `16x16x16` descriptor at `N=256` and `4096`, validates sanitized
command examples, VDCores/MPK source-paper metadata, report files, and PTO
dispatch sequences, and keeps the explicit graph tensor row beside scalar
tensor, WMMA tensor-core, and cuBLAS rows.

| GPU | N | Shape | Scalar tensor ns | Graph tensor ns | Tensor-core ns | cuBLAS ns | Scalar GF/s | Graph tensor GF/s | Tensor-core GF/s | cuBLAS GF/s | Graph/scalar | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | ---------------- | --------------- | -------------- | --------- | ----------- | ----------------- | ---------------- | ----------- | ------------ | ------------------ | ------------- |
| A100 | 256 | 16x16x16 | 47104 | 47104 | 45056 | 48128 | 0.17 | 0.17 | 0.18 | 0.17 | 1.00x | 0.96x | 1.02x |
| A100 | 4096 | 16x16x16 | 80896 | 80896 | 82944 | 39935 | 1.62 | 1.62 | 1.58 | 3.28 | 1.00x | 1.03x | 0.49x |
| H200 | 256 | 16x16x16 | 29568 | 32800 | 27040 | 51711 | 0.28 | 0.25 | 0.30 | 0.16 | 1.11x | 0.91x | 1.75x |
| H200 | 4096 | 16x16x16 | 89472 | 89152 | 51872 | 35904 | 1.46 | 1.47 | 2.53 | 3.65 | 1.00x | 0.58x | 0.40x |

### Current-Head Reproducibility Check

A follow-up one-repeat compact tensor sweep at commit `a5fd4bfd` validates the
current report-generation and metadata gate after the tensor-sweep scripts
started recording sanitized command examples. The artifact is under
`tmp/cuda-backend/tensor-shape-sweep-a5fd4bfd/` and was validated with
`--require-command-examples`, `--require-source-papers`, required A100/H200
rows, required report files, and PTO dispatch sequences.

| GPU | N | Shape | Scalar tensor ns | Tensor-core ns | cuBLAS ns |
| --- | - | ----- | ---------------- | -------------- | --------- |
| A100 | 256 | 16x16x16 | 47104 | 46080 | 41983 |
| H200 | 256 | 16x16x16 | 31552 | 28544 | 51040 |

This is a current-HEAD smoke/evidence capture rather than a replacement for
the three-repeat size sweep above. It proves the exact command examples needed
to reconstruct the local A100 and remote H200 setup are now present in the
generated Markdown and JSON.

A later working-tree tensor sweep under
`tmp/cuda-backend/tensor-graph-library-baselines-working/`
`tensor-shape-sweep-848c4ee5/` adds `cublas_sgemm_graph` to the same compact
descriptor family. It was validated with required A100/H200 rows, report
files, command examples, VDCores/MPK source-paper metadata, and PTO dispatch
sequences. The table below is rendered by `cuda_current_summary.py --section
tensor-sweep` from that raw JSON.

| GPU | N | Shape | Scalar tensor ns | Graph tensor ns | Tensor-core ns | cuBLAS ns | cuBLAS Graph ns |
| --- | - | ----- | ---------------- | --------------- | -------------- | --------- | --------------- |
| A100 | 256 | 16x16x16 | 43008 | 41984 | 51200 | 89088 | 12288 |
| H200 | 256 | 16x16x16 | 37440 | 45120 | 37472 | 50271 | 10271 |

The CUDA Graph replay row is much faster than plain cuBLAS for this tiny
launch-dominated descriptor because the capture has already paid the library
setup cost. It is a launch/replay baseline, not a tuned large-GEMM throughput
claim.

## Tensor-Core Callable Smoke

The first tensor-core persistent DAG smoke was captured at commit `390eda4f`.
It adds a block-wide generated-dispatch task body using CUDA WMMA
`m16n16k8` with TF32 inputs and F32 accumulation. The task runs before the
same residual, gate, and fan-in elementwise tasks used by `tensor_tile`, so
this validates callable shape and scheduler integration rather than tuned GEMM
throughput.

Artifact:
`tmp/cuda-backend/persistent-tensor_core_tile-16x16x16-smoke-390eda4f/`

| GPU | Shape | Tensor core | Dispatch | Device ns | Host ns | Status |
| --- | ----- | ----------- | -------- | --------- | ------- | ------ |
| A100 | 16x16x16 | `wmma:m16n16k8:tf32->f32` | `10,1,2,1` | 46080 | 65963 | pass |
| H200 | 16x16x16 | `wmma:m16n16k8:tf32->f32` | `10,1,2,1` | 31808 | 41308 | pass |

Both rows report zero device scheduler errors, `completed_count=4`, and
target-specific `nvcc` PTX (`compute_80` on A100, `compute_90` on H200). The
generated Markdown/SVG report is in the artifact directory.

## Graph Tensor-Core Descriptor Smoke

The latest graph tensor-core persistent DAG smoke adds explicit runtime graph
descriptor coverage around the same WMMA first task as `tensor_core_tile`. It
uses a `16x16x16` descriptor, `N=256`, two repeat launches, queue capacity
`2`, and one scheduler block plus four worker blocks on both A100 and H200.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/a100.json \
    tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 2 --expected-completed-count 4 \
    --expected-scheduler-blocks 1 --expected-worker-blocks 4 \
    --expected-worker-blocks-per-task 1 --expected-stream-id 0 \
    --expected-block-dim 256 --expected-grid-dim 5 \
    --require-report-files --expected-dag-shape graph_tensor_core_tile \
    --expected-dispatch 10,1,2,1 --expected-tensor-tile 16x16x16 \
    --expected-graph-fanin 0,1,1,2 \
    --expected-graph-dependents 1,2,3,3
```

| GPU | Dispatch | Graph fan-in | Dependents | Tensor core | Device ns | Host ns |
| --- | -------- | ------------ | ---------- | ----------- | --------- | ------- |
| A100 | `10,1,2,1` | `0,1,1,2` | `1,2,3,3` | `wmma:m16n16k8:tf32->f32` | 76800 | 106982 |
| H200 | `10,1,2,1` | `0,1,1,2` | `1,2,3,3` | `wmma:m16n16k8:tf32->f32` | 57472 | 75314 |

Both artifacts report `launch_completed_counts=[4,4]`, zero scheduler errors,
tensor tile `16x16x16`, and target-specific PTX (`compute_80` on A100,
`compute_90` on H200). This is graph-lowering and callable metadata evidence;
throughput comparisons remain in the selected tensor baseline sweeps.

## Graph Tensor-Core Benchmark Row

The first explicit graph tensor-core benchmark row was captured from the
working tree under the artifact label `debe979d`. It uses the tensor-shape
sweep flow with one baseline, `N=256`, one repeat, and a `16x16x16`
descriptor. Because the working tree contained uncommitted benchmark support
when the sweep ran, treat this as a working-tree capture under that label, not
as evidence that commit `debe979d` alone contains the feature.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-sweep.json \
    --require-artifact a100 --require-artifact h200 \
    --require-baseline pto_persistent_dag_graph_tensor_core \
    --require-size 256 --require-shape 16x16x16 \
    --expected-repeats 1 --expected-result-count 2 \
    --require-dispatch pto_persistent_dag_graph_tensor_core=10,1,2,1 \
    --require-report-files --require-command-examples \
    --require-source-papers
```

| GPU | Shape | Dispatch | Graph fan-in | Device ns | Host ns | PTX |
| --- | ----- | -------- | ------------ | --------- | ------- | --- |
| A100 | 16x16x16 | `10,1,2,1` | `0,1,1,2` | 52224 | 73631 | `compute_80` |
| H200 | 16x16x16 | `10,1,2,1` | `0,1,1,2` | 50144 | 64644 | `compute_90` |

Both rows report graph dependents `[1,2,3,3]`, tensor tile `16x16x16`,
`wmma:m16n16k8:tf32->f32`, zero scheduler errors, and source-paper metadata
for VDCores and MPK.

## Tensor-Core Benchmark Row

The first tensor-core row in the selected benchmark report was captured at
commit `0879aa9e`. It uses the same compact A100/H200 benchmark report format
as the full paired capture, but with one size (`N=256`), one repeat, no batch
rows, and a `16x16x16` tensor descriptor. The raw JSON, Markdown, and SVG
artifacts are under
`tmp/cuda-backend/combined-tensor-core-current-0879aa9e/`.

Tensor-core row details:

- A100: `pto_persistent_dag_tensor_core`, `16x16x16`,
  `wmma:m16n16k8:tf32->f32`, `37888 ns` device, `52277 ns` host,
  `0.90x` versus `pto_persistent_dag`; signed DAG increment `-4096 ns`.
- H200: `pto_persistent_dag_tensor_core`, `16x16x16`,
  `wmma:m16n16k8:tf32->f32`, `38656 ns` device, `50211 ns` host,
  `0.97x` versus `pto_persistent_dag`; signed DAG increment `-1088 ns`.

For context, the scalar tensor DAG row in the same report measured `40960 ns`
on A100 and `43392 ns` on H200. The tensor-core row therefore validates that
WMMA callable bodies now participate in the normal selected-baseline report
and chart flow, but it still measures one small generated task shape rather
than a tuned tensor-core kernel.
The generated `cuda-benchmark-dag-deltas.svg` chart visualizes the signed
device-time increment over the matched `pto_persistent_dag` scheduler
baseline, which is the current report view for separating scheduler overhead
from additional generated-dispatch task work.
Regenerated benchmark reports also include `cuda-benchmark-throughput.svg`,
which normalizes tensor-DAG and cuBLAS rows by the recorded tensor tile
descriptor and tile count into median GF/s.

## cuBLAS Library Baseline Row

The first cuBLAS row in the selected benchmark report was captured in the
`a100-cublas-current-343924df` and `h200-cublas-current-343924df` artifacts,
then merged under `tmp/cuda-backend/combined-cublas-current-343924df/`. It
uses the same compact report shape as the tensor-core row: one size
(`N=256`), one repeat, no batch rows, and the `16x16x16` tensor descriptor.

cuBLAS row details:

- A100: `cublas_sgemm`, `16x16x16`,
  `cublasSgemmStridedBatched`, batch count `1`, `48128 ns` device,
  `64414 ns` host, `2.76x` versus `pto_host_schedule`.
- H200: `cublas_sgemm`, `16x16x16`,
  `cublasSgemmStridedBatched`, batch count `1`, `58623 ns` device,
  `71677 ns` host, `2.34x` versus `pto_host_schedule`.

For context, the same compact report measured `pto_persistent_dag_tensor_core`
at `33792 ns` on A100 and `32960 ns` on H200. The cuBLAS row is intentionally
a library-backed launch/compute baseline rather than PTO runtime work; at this
small descriptor size it is dominated by cuBLAS launch and dispatch overhead,
not GEMM throughput.

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 2,6,12 \
    --worker-blocks-per-task 32,64,128,256 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --label a100-current-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-current-$(git rev-parse --short HEAD)
```

Paired A100/H200:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Tensor shape sweep:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_tensor,pto_persistent_dag_graph_tensor,pto_persistent_dag_tensor_core,cublas_sgemm,cublas_sgemm_graph \
    --shapes 16x16x16,16x16x64 --n 256 --repeats 3 \
    --sync-remote-tree
```

Tensor-core smoke:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_core_tile --task-count 4 --queue-capacity 2 \
    --n 256 --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree
```

Tensor-core selected-baseline report:

```bash
COMMIT=$(git rev-parse --short HEAD)
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --sizes 256 --repeats 1 --arch compute_80 --include-persistent \
    --batch-tasks 0 --worker-blocks-per-task 1 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --label a100-tensor-core-current-$COMMIT \
    --output-dir tmp/cuda-backend/a100-tensor-core-current-$COMMIT
```

cuBLAS selected-baseline report:

```bash
COMMIT=$(git rev-parse --short HEAD)
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --sizes 256 --repeats 1 --arch compute_80 --include-persistent \
    --batch-tasks 0 --worker-blocks-per-task 1 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --label a100-cublas-current-$COMMIT \
    --output-dir tmp/cuda-backend/a100-cublas-current-$COMMIT
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-current-61cf96cd/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-61cf96cd/cuda-benchmark.json \
    --label combined-current-61cf96cd \
    --output-dir tmp/cuda-backend/combined-current-61cf96cd
```
