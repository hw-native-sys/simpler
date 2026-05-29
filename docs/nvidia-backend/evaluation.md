# CUDA Backend Evaluation Notes

This page is the landing index for CUDA backend evaluation evidence. The
measurements are early runtime microbenchmarks, not end-to-end LLM serving
results. They follow the VDCores and MPK papers only at the evaluation-shape
level: fixed GPU work, repeated problem sizes, selected launch baselines,
local A100 runs, and remote H200 runs.

## Current Evidence

The latest full paired A100/H200 benchmark capture was taken at commit
`61cf96cd`, and the latest compact current-head paired gate uses artifact
label `c0d327d2`. Supplemental tensor-shape and tensor-core captures were
taken at commits `c0ada3ad` and `0879aa9e`. The first cuBLAS library baseline
capture uses the `343924df` artifact label, and the first cuBLAS CUDA Graph
baseline capture is under `cublas-graph-compact-working`. The first
multi-baseline tensor shape sweep used the `6f9a0b78` artifact label, and the
latest multi-size tensor baseline sweep uses `e79edba2`:

- [Current capture](evaluation-current.md) summarizes the latest
  `8x4x12` tensor-descriptor sweep, selected baselines, host-schedule unary
  and quad rows, scalar AXPY, scalar affine, triad, quad, generic-args, and
  graph-descriptor and unary-square descriptor rows, and headline
  interpretation.
- [Current capture](evaluation-current.md) records the compact scalar-scale
  benchmark gate that adds `pto_persistent_dag_scalar_scale` to the selected
  paired benchmark path with A100/H200 report artifacts.
- [Current capture](evaluation-current.md) also records the supplemental
  `8x4x12`, `16x16x64`, and `32x16x64` tensor-shape sweep for
  `pto_persistent_dag_tensor`.
- [Current capture](evaluation-current.md) records a compact tensor-baseline
  size sweep comparing scalar tensor DAG, explicit-graph scalar tensor DAG,
  fixed WMMA tensor-core DAG, explicit-graph WMMA tensor-core DAG, and cuBLAS
  SGEMM rows for a `16x16x16` descriptor.
- [Current capture](evaluation-current.md) records the compact `d361006f`
  paired gate that validates the default `16x16x16` tensor descriptor with
  scalar tensor DAG, WMMA tensor-core DAG, and cuBLAS rows in one current-head
  A100/H200 report.
- [Current capture](evaluation-current.md) records the compact `dbb01406`
  paired gate that promotes the explicit graph-descriptor scratch-reuse DAG
  into the selected benchmark path.
- The supplemental graph tensor-tile sample under
  `tmp/cuda-backend/combined-graph-tensor-current-working/` validates
  `pto_persistent_dag_graph_tensor`, the explicit graph-descriptor variant of
  the scalar tiled-GEMM DAG, on A100 and H200.
- [Current capture](evaluation-current.md) records the first selected
  benchmark row for `pto_persistent_dag_tensor_core`, a WMMA
  `m16n16k8` TF32/F32 generated-dispatch task followed by the same residual,
  gate, and fan-in tasks as the scalar tensor DAG.
- [Current capture](evaluation-current.md) records the first
  `pto_persistent_dag_graph_tensor_core` benchmark row, the explicit
  graph-descriptor variant of the WMMA tensor-core DAG, on A100 and H200.
- [Current capture](evaluation-current.md) records the compact `ca290b2a`
  paired gate that includes scalar tensor, explicit graph tensor, fixed WMMA
  tensor-core, explicit graph WMMA tensor-core, cuBLAS, and cuBLAS Graph rows
  in one selected A100/H200 benchmark report.
- [Current capture](evaluation-current.md) records the first
  `cublas_sgemm` library-backed tensor baseline row in the same compact
  selected-baseline report shape.
- [Current capture](evaluation-current.md) records the first
  `cublas_sgemm_graph` row, which captures the same warmed cuBLAS descriptor
  into a CUDA Graph and times graph replay on A100 and H200.
- [Current capture](evaluation-current.md) records the compact `30a8974f`
  paired gate that compares tagged, role-keyed, and compact role-entry graph
  task-argument spellings in one selected A100/H200 benchmark report.
- [Current capture](evaluation-current.md) records the compact `01ddf564`
  paired gate that promotes incoming-edge `depends_on` graph descriptors into
  the selected A100/H200 benchmark path.
- [Current capture](evaluation-current.md) records the compact `7191db4e`
  current-head paired gate that validates 88 A100/H200 samples with
  report-visible graph topology, node attrs, scalar/tensor descriptor args,
  tensor throughput rows, command examples, and source-paper metadata.
- [Current capture](evaluation-current.md) records the compact `c0d327d2`
  paired gate that promotes graph node `inputs`/`outputs` metadata into the
  selected A100/H200 benchmark matrix with 98 validated samples.
- [Current capture](evaluation-current.md) records the compact `95be2b5b`
  paired gate that promotes graph named-callable task metadata into the
  selected A100/H200 benchmark matrix with 96 non-batch validated samples.
- [Current capture](evaluation-current.md) records the `1c683c1c` lifecycle
  matrix that adds graph tensor-core descriptors to the paired repeat-run
  prepared-callable validation path.
- [Current capture](evaluation-current.md) records the `35de3303` scheduler
  error matrix that captures all synthetic malformed-DAG diagnostics on A100
  and H200 with JSON, Markdown, and SVG artifacts validated by
  `cuda_validate_scheduler_error_matrix.py --preset default`.
- [Historical captures](evaluation-history.md) preserve the previous
  accumulated benchmark notes, including earlier graph, stream, task-count,
  worker-grid, DAG-chain, scratch-reuse, and tensor-tile captures.

The latest raw artifacts remain under `tmp/` and are intentionally not
committed:

- `tmp/cuda-backend/a100-current-61cf96cd/`
- `tmp/cuda-backend/h200-current-61cf96cd/`
- `tmp/cuda-backend/combined-current-61cf96cd/`
- `tmp/cuda-backend/a100-current-f0f43b2a/`
- `tmp/cuda-backend/h200-current-f0f43b2a/`
- `tmp/cuda-backend/combined-current-f0f43b2a/`
- `tmp/cuda-backend/a100-current-d361006f/`
- `tmp/cuda-backend/h200-current-d361006f/`
- `tmp/cuda-backend/combined-current-d361006f/`
- `tmp/cuda-backend/a100-current-a46db551/`
- `tmp/cuda-backend/h200-current-a46db551/`
- `tmp/cuda-backend/combined-current-a46db551/`
- `tmp/cuda-backend/a100-current-b2c5c8a4/`
- `tmp/cuda-backend/h200-current-b2c5c8a4/`
- `tmp/cuda-backend/combined-current-b2c5c8a4/`
- `tmp/cuda-backend/a100-current-06b8c0c6/`
- `tmp/cuda-backend/h200-current-06b8c0c6/`
- `tmp/cuda-backend/combined-current-06b8c0c6/`
- `tmp/cuda-backend/a100-current-dbb01406/`
- `tmp/cuda-backend/h200-current-dbb01406/`
- `tmp/cuda-backend/combined-current-dbb01406/`
- `tmp/cuda-backend/cublas-graph-compact-working/a100-current-5168f150/`
- `tmp/cuda-backend/cublas-graph-compact-working/h200-current-5168f150/`
- `tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/`
- `tmp/cuda-backend/persistent-scalar_affine-smoke-469f55cd/`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/`
- `tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/`
- `tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/`
- `tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/`
- `tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/`
- `tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/`
- `tmp/cuda-backend/persistent-node-io-smoke-working/persistent-graph_descriptor_node_io-repeat2-smoke-feddd21b/`
- `tmp/cuda-backend/persistent-node-port-dict-smoke-working/persistent-graph_descriptor_node_port_dict-repeat2-smoke-b336f9ff/`
- `tmp/cuda-backend/worker-square-smoke-4cdde399/`
- `tmp/cuda-backend/worker-quad-smoke-4327698e/`
- `tmp/cuda-backend/worker-mul-smoke-output-json/`
- `tmp/cuda-backend/tensor-descriptor-smoke-6c49c5cf/`
- `tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-5139ba23/`
- `tmp/cuda-backend/persistent-tensor_tile-8x4x12-repeat2-smoke-223425b6/`
- `tmp/cuda-backend/worker-generic_args4-smoke-03ed75da/`
- `tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/`
- `tmp/cuda-backend/combined-graph-tensor-current-working/`
- `tmp/cuda-backend/tensor-shape-sweep-c0ada3ad/`
- `tmp/cuda-backend/tensor-shape-sweep-0e84fd26/`
- `tmp/cuda-backend/persistent-tensor_core_tile-16x16x16-smoke-390eda4f/`
- `tmp/cuda-backend/a100-tensor-core-current-0879aa9e/`
- `tmp/cuda-backend/h200-tensor-core-current-0879aa9e/`
- `tmp/cuda-backend/combined-tensor-core-current-0879aa9e/`
- `tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/a100-current-493ce832/`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/h200-current-493ce832/`
- `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/a100-current-ca290b2a/`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/h200-current-ca290b2a/`
- `tmp/cuda-backend/current-head-compact-ca290b2a-working/combined-current-ca290b2a/`
- `tmp/cuda-backend/a100-cublas-current-343924df/`
- `tmp/cuda-backend/h200-cublas-current-343924df/`
- `tmp/cuda-backend/combined-cublas-current-343924df/`
- `tmp/cuda-backend/tensor-shape-sweep-6f9a0b78/`
- `tmp/cuda-backend/tensor-shape-sweep-c4ee08eb/`
- `tmp/cuda-backend/tensor-shape-sweep-47d857e1/`
- `tmp/cuda-backend/tensor-shape-sweep-e79edba2/`
- `tmp/cuda-backend/tensor-sweep-current-working/tensor-shape-sweep-219042f5/`
- `tmp/cuda-backend/compact-role-benchmark-working/a100-current-30a8974f/`
- `tmp/cuda-backend/compact-role-benchmark-working/h200-current-30a8974f/`
- `tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f/`
- `tmp/cuda-backend/graph-depends-benchmark-working/a100-current-01ddf564/`
- `tmp/cuda-backend/graph-depends-benchmark-working/h200-current-01ddf564/`
- `tmp/cuda-backend/graph-depends-benchmark-working/combined-current-01ddf564/`
- `tmp/cuda-backend/current-head-compact-args-summary-working/a100-current-7191db4e/`
- `tmp/cuda-backend/current-head-compact-args-summary-working/h200-current-7191db4e/`
- `tmp/cuda-backend/current-head-compact-args-summary-working/combined-current-7191db4e/`
- `tmp/cuda-backend/graph-node-io-benchmark-working/a100-current-c0d327d2/`
- `tmp/cuda-backend/graph-node-io-benchmark-working/h200-current-c0d327d2/`
- `tmp/cuda-backend/graph-node-io-benchmark-working/combined-current-c0d327d2/`
- `tmp/cuda-backend/tensor-throughput-gate-current-working/a100-current-a9d028de/`
- `tmp/cuda-backend/tensor-throughput-gate-current-working/h200-current-a9d028de/`
- `tmp/cuda-backend/tensor-throughput-gate-current-working/combined-current-a9d028de/`
- `tmp/cuda-backend/lifecycle-tensor-core-working/persistent-lifecycle-matrix-1c683c1c/`
- `tmp/cuda-backend/scheduler-error-matrix-working/scheduler-error-matrix-35de3303/`
- `tmp/cuda-backend/index.md`

`tmp/cuda-backend/index.md` is generated by
`.agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py`. It indexes
local benchmark, tensor-shape sweep, scheduler-error matrix, lifecycle
matrix, and smoke artifacts,
including tensor-tile descriptor shapes when the JSON payloads carry that
metadata. Persistent smoke and lifecycle matrix rows also include dispatch,
graph descriptor fan-in/dependent arrays, repeat-run counts, and per-launch
completion counts, which are the quick audit fields for graph-descriptor
lifecycle reuse captures. Lifecycle matrix reports now also carry the
VDCores/MPK source-paper IDs, collection mode, and sanitized reconstruction
commands, and can be refreshed from existing per-scenario smoke JSON with
`cuda_persistent_lifecycle_matrix.py --collect-existing-suffix <commit>`.
The lifecycle validator requires regenerated existing-JSON reports to retain
that flag in their local sample command. The artifact index reads the lifecycle
matrix metadata commit and collection mode so regenerated reports stay
distinguishable from paired smoke captures. Validators, smoke reports,
lifecycle matrix reports, benchmark diagnostics, and the artifact index render
known nonzero device scheduler codes with stable labels such as
`7(unreachable_task)`, while zero-error rows remain compact as `code=0`.
Benchmark report directories now include `cuda-benchmark-dag-deltas.svg`,
which visualizes the signed device-time increment of each
`pto_persistent_dag_*` row over the matched `pto_persistent_dag` scheduler
baseline. Tensor-DAG and cuBLAS rows also produce
`cuda-benchmark-throughput.svg`, which normalizes the recorded
`rows x cols x inner` descriptor and tile count into median GF/s.
Benchmark Markdown reports now include a graph descriptor metadata table, and
the primary SVG embeds the same graph topology and task-argument metadata for
explicit graph rows, so rendered report artifacts expose the lowered
`TaskArgs` shape instead of leaving it only in raw JSON.
The compact smoke Markdown and SVG reports generated by
`.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py` expose the
same graph descriptor fan-in/dependent arrays and lifecycle fields when
present. Tensor-core smoke rows also expose the WMMA shape and
input/accumulator types in the Markdown table, so the per-artifact visual
evidence and the local index agree.
The compact benchmark and tensor-sweep tables in
[Current capture](evaluation-current.md) can be regenerated from raw JSON with
`.agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py`, including
the selected benchmark tensor-throughput table via
`--section tensor-throughput`, graph tensor-core rows in both tensor table
sections, and graph scratch-reuse ratios via `--section dag-shapes` when the
capture includes `pto_persistent_dag_graph_scratch_reuse`. Use
`--section graph-metadata` to render the explicit graph descriptor dispatch
sequence, fan-in/dependent arrays, task-argument keying, tagged task
arguments, and tensor-tile shape beside the timing tables. Use
`--section graph-role-spelling` to compare tagged, role-keyed, and compact
role-entry graph rows in a focused table.
New benchmark and tensor-sweep Markdown reports embed source-paper provenance
for the VDCores and MPK notes kept under `tmp/sources/` and sanitized
local/remote command examples for reconstructing the run. Tensor sweeps also
include a workload description for each selected tensor baseline.
Validate a refreshed paired-current capture before updating committed docs
with `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py`.
The paired benchmark runner wires that validator with expected generated
`dispatch_func_ids` for known persistent DAG rows, so a numerically passing
capture is still rejected if it ran the wrong CUDA device task sequence. It
also checks tensor descriptor metadata for tensor and cuBLAS rows, plus
explicit `--require-report-graph-topology` and
`--require-report-graph-task-args` gates for visible graph topology and
task-argument metadata in `cuda-benchmark.md` and `cuda-benchmark.svg`.
Current paired presets also use `--require-report-graph-role-spelling` for
the focused tag/role/compact graph rows and
`--require-report-tensor-throughput` for the Markdown/SVG tensor throughput
view, so a report captured with the wrong `--tensor-rows`, `--tensor-cols`,
`--tensor-inner`, missing GF/s rows, or stale graph metadata does not get
copied into the current-evaluation tables.
Tensor-shape sweep validation uses the same report-quality contract through
`--require-report-throughput`, and the compact tensor-baseline preset enables
that gate by default.

## Baselines

- `direct_driver`: thin CUDA Driver API launch path for the same vector-add
  PTX kernel.
- `direct_driver_graph`: CUDA Graph replay of the same Driver API vector-add
  kernel, with graph instantiation outside the measured interval.
- `pto_host_schedule`: PTO CUDA host runtime C API and manifest dispatch.
- `pto_host_schedule_compiler`: same host runtime path using a
  `KernelCompiler(platform="cuda")` generated task-body wrapper and cached PTX.
- `pto_host_schedule_unary_square`: same generated host runtime path for the
  unary `(a, out, n)` ABI, using a square task body.
- `pto_host_schedule_quad`: same generated host runtime path for the
  four-input `(a, b, c, d, out, n)` ABI.
- `pto_persistent_device`: descriptor-array persistent executor.
- `pto_persistent_queue`: scheduler block plus bounded device ring queue.
- `pto_persistent_dag`: generated-dispatch-like task selection with fan-in
  counters.
- `pto_persistent_dag_chain`: five-task generated-dispatch DAG with a
  post-fan-in dependency chain.
- `pto_persistent_dag_reuse`: six-task generated-dispatch DAG that reuses a
  scratch buffer after the buffer's last dependent completes.
- `pto_persistent_dag_scalar_axpy`: generated-dispatch DAG that reads a
  `scalar0` task descriptor field for mixed tensor/scalar AXPY work before
  downstream fan-in.
- `pto_persistent_dag_scalar_scale`: generated-dispatch DAG that reads
  `scalar0` with one tensor input before downstream fan-in.
- `pto_persistent_dag_scalar_affine`: generated-dispatch DAG that reads
  `scalar0` and `scalar1` task descriptor fields for two-scalar affine work
  before downstream fan-in.
- `pto_persistent_dag_triad`: generated-dispatch DAG that reads a third tensor
  task descriptor field for three-input triad work before downstream fan-in.
- `pto_persistent_dag_quad`: generated-dispatch DAG that reads third and
  fourth tensor task descriptor fields for four-input work before downstream
  fan-in.
- `pto_persistent_dag_generic_args`: generated-dispatch DAG that reads generic
  tensor/scalar descriptor slots before downstream fan-in.
- `pto_persistent_dag_graph`: generated-dispatch DAG that reads an explicit
  runtime graph descriptor before downstream fan-in. This row validates the
  same graph-lowering shape used by `persistent_dag_graph_f32`.
- `pto_persistent_dag_graph_node_attrs`: explicit graph-descriptor row that
  records scalar/tensor node attributes in report-visible metadata.
- `pto_persistent_dag_graph_node_io`: explicit graph-descriptor row that
  records node `inputs` / `outputs` task-argument metadata.
- `pto_persistent_dag_graph_node_link`: explicit graph-descriptor row that
  accepts node-link style graph spelling and records graph-node ops.
- `pto_persistent_dag_graph_named_callable`: explicit graph-descriptor row
  that records callable names in task-argument metadata while dispatching the
  same add/mul/add device task functions.
- `pto_persistent_dag_graph_node_op`: explicit graph-descriptor row that
  records node `op` callable aliases in graph-node ops metadata.
- `pto_persistent_dag_graph_depends_on`: explicit graph-descriptor row that
  accepts incoming `depends_on` edge spelling.
- `pto_persistent_dag_graph_chain`: explicit graph-descriptor variant of the
  five-task chain DAG, validating that the benchmark path can time the chain
  shape from graph metadata rather than a fixed DAG adapter.
- `pto_persistent_dag_graph_scratch_reuse`: explicit graph-descriptor
  variant of the six-task scratch-reuse DAG, validating the benchmark path
  can time scratch-buffer reuse after the last consumer from graph metadata.
- `pto_persistent_dag_unary_square`: generated-dispatch DAG with a one-input
  square task body before downstream fan-in.
- `pto_persistent_dag_tensor`: four-task generated-dispatch DAG with a tiled
  GEMM task followed by residual, gate, and fan-in elementwise tasks.
- `pto_persistent_dag_graph_tensor`: explicit graph-descriptor variant of the
  tiled GEMM, residual, gate, and fan-in DAG.
- `pto_persistent_dag_tensor_core`: four-task generated-dispatch DAG with a
  block-wide WMMA `m16n16k8` TF32/F32 task followed by residual, gate, and
  fan-in elementwise tasks.
- `pto_persistent_dag_graph_tensor_core`: explicit graph-descriptor variant
  of the WMMA tensor-core DAG.
- `cublas_sgemm`: CUDA Runtime API plus cuBLAS
  `cublasSgemmStridedBatched` over the configured tensor descriptor. This is
  the first library-backed CUDA tensor baseline in the selected report.
- `cublas_sgemm_graph`: CUDA Runtime API plus cuBLAS captured into a CUDA
  Graph after handle warmup, with graph instantiation outside the measured
  interval and `cudaGraphLaunch` timed by CUDA events.
- `*_batch`: same-work rows comparing repeated host launches with one
  persistent launch over the same descriptor count.
- `pto_persistent_device_grid_batch`: direct persistent-device batch row with
  a swept number of CUDA worker blocks assigned to each task descriptor.

Ratios are relative to the matched reference for the same GPU, vector length,
and task count. For same-work batch rows, the reference is
`pto_host_schedule_batch`, not the one-task `pto_host_schedule` row. Stream
rows use `pto_stream_serial` as their reference.

The host-schedule stream-concurrency workflow has its own paired runner
because it is a focused launch-overlap benchmark rather than part of the
selected persistent-device baseline matrix. The latest paired stream capture
is under
`tmp/cuda-backend/stream-pair-working/combined-stream-pool6-a36d137b/` and
validates two repeats of `pto_stream_serial` and `pto_stream_parallel` on
A100 and H200 with `--stream-pool-size 6`.

## Reproduction

The committed workflow lives under
`.agents/skills/cuda-backend-eval/scripts/`. The most commonly used commands
are documented in `.agents/skills/cuda-backend-eval/SKILL.md`.

Use the paired runner to refresh the local A100 and remote H200 capture in one
step:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py
```

It runs the local benchmark, refreshes the remote `bizhaoh200` checkout,
copies the H200 artifact directory back, merges the JSON reports, validates
the combined artifact with the configured sizes/repeats/baselines, and rebuilds
the local artifact index. The remote checkout step uses bounded Git HTTPS
low-speed settings by default so a bad network fetch fails instead of
stranding the paired run, and the fetch is wrapped in a shell `timeout`.
When the remote checkout has already been updated by another path and Git
HTTPS is unhealthy on `bizhaoh200`, add `--skip-remote-refresh` to reuse that
checkout and still run the paired A100/H200 capture. In skip-refresh mode,
the runner reads the remote checkout commit and uses it in the H200 artifact
name; if local and remote commits differ, the combined artifact name includes
both commits. A skip-refresh dry run still performs the read-only remote
commit probe so the printed artifact paths are accurate.
When remote Git remains unhealthy and the local checkout is the source of
truth, add `--sync-remote-tree` to copy the current local tree to
`bizhaoh200` with `rsync` before running H200. This mode skips remote Git,
excludes `.venv`, `build`, and `tmp`, and labels both GPU artifacts with the
local commit. It syncs `.git` so the remote benchmark metadata reports the
same commit as the synced source tree.

Use the paired stream runner to refresh the host-schedule stream-concurrency
capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_stream_benchmark.py \
    --repeats 2 --stream-pool-size 6 --sync-remote-tree \
    --output-root tmp/cuda-backend/stream-pair-working
```

Use the paired smoke runner when the goal is a fast real-data A100/H200 check
without requiring `torch` on the H200 environment:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op scale --sync-remote-tree --build-runtime
```

It captures host-schedule Worker smoke JSON on both GPUs, renders
`cuda-smoke-report.md` and `cuda-smoke-report.svg`, and refreshes
`tmp/cuda-backend/index.md`. It supports the same remote refresh,
`--skip-remote-refresh`, `--sync-remote-tree`, and `--dry-run` controls as the
paired benchmark runner. Use `--build-runtime` after changing CUDA runtime C++
so the synced H200 checkout does not run stale shared objects.

Refresh the local artifact index after adding or merging captures:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py \
    --root tmp/cuda-backend
```

Validate the current paired capture before copying numbers into
[Current capture](evaluation-current.md):

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/<combined-capture>/cuda-benchmark.json \
    --preset compact-current
```

The compact-current preset now expects `pto_persistent_dag_graph_reordered`
in the selected matrix. For no-batch graph-row gates, the paired runner should
validate `86` samples across A100 and H200 and require dispatch `1,9,2`, graph
fan-in `2,0,0`, and dependents `0,0` for that baseline.

The default full paired benchmark shape uses:

- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- tensor descriptor: `16x16x16`
- local A100 PTX arch: `compute_80`
- remote H200 PTX arch: `compute_90`

The paired persistent smoke runner validates smoke artifacts before refreshing
the local index. Use `--skip-validation` only for intentional negative
scheduler-diagnostic captures. Standalone smoke validation uses:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-d3a86494/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-d3a86494/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor --expected-repeat-runs 2 \
    --expected-completed-count 3 --expected-dispatch 9,2,1 \
    --expected-graph-fanin 0,0,2 --expected-graph-dependents 2,2 \
    --require-report-files --require-report-graph-topology
```

For tensor-tile smokes, add `--expected-tensor-tile ROWSxCOLSxINNER` to
require the descriptor shape recorded in the A100/H200 JSON payloads.
For generated-dispatch DAG smokes, the paired persistent-smoke runner also
passes `--expected-dispatch` for the requested DAG shape so a numerically
passing artifact must still prove the expected device task sequence.
For explicit graph-descriptor smokes, it also passes
`--expected-graph-fanin` and `--expected-graph-dependents` so the JSON payloads
must prove the expected runtime graph topology. It also passes
`--require-report-graph-topology` so the Markdown and SVG smoke reports cannot
silently lag behind the JSON topology metadata.
For tagged graph smokes, it also passes `--expected-graph-task-args` and
`--require-report-graph-task-args`, so the visible reports must expose the
same lowered TaskArgs-like roles as the JSON payload.
For node-IO graph smokes, it additionally passes
`--expected-graph-task-arg-key node_io`, proving that graph node
`input`/`output` fields are preserved separately from role-keyed or tagged
task-argument spellings.
For node-port-map graph smokes, it passes
`--expected-graph-task-arg-key node_port_dict`,
`--expected-graph-task-args`, `--expected-graph-node-ops`, and report
requirements for both fields, so the A100/H200 JSON plus Markdown/SVG
artifacts must show the lowered port names and generated add/mul/add
dispatch.
