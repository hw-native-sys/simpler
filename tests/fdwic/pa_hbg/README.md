# A5 HBG resident scheduler validation

This directory tests A5 AICore-resident Scalar scheduling of host-built graphs.
Host orchestration still constructs the DAG, TensorMap and memory plan; AICore
workers discover ready tasks, admit cooperative work, execute blocks and release
dependencies. These tests do not use the earlier per-core replay runtime.

## Documentation

- [Architecture and supported shapes](a5-hbg-spmd-design.md)
- [Ownership, publication and lifecycle contracts](spmd/implementation-contract.md)
- [Validation matrix and commands](spmd/validation.md)
- [Final CSA performance and measurement boundaries](spmd/performance.md)
- [Frozen CSA fixture and user guide](../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md)

Runtime code and regression fixtures are retained independently of development
journals. Raw traces, performance samples, machine-local output paths and failed
optimization histories are not required to build or run these tests.

## Small paged-attention example

[test_pa_hbg.py](test_pa_hbg.py) owns its scene definition, argument adapter and
golden entry. Its [orchestration](kernels/orchestration/paged_attention_orch.cpp)
is local; QK/SF/PV/UP kernels reuse the existing
[HBG PA unroll implementation](../../st/a5/host_build_graph/paged_attention_unroll/kernels).
Input generation and reference math use `simpler_setup.goldens.paged_attention`.

| Parameter | Value |
| --------- | ----- |
| batch / heads / KV heads | 4 / 16 / 1 |
| head dimension / block size | 128 / 128 |
| context / maximum model length | 8192 / 32768 |
| dtype / tolerance | bfloat16 / rtol=atol=1e-3 |

Each batch allocates online-softmax state, then submits QK (AIC), SF (AIV),
PV (AIC) and UP (AIV). Four batches produce four Alloc services and sixteen
single-block compute tasks. Dependencies come from tensor IN/OUT/INOUT semantics;
there are no cross-batch data dependencies. This B4 functional example is not a
replacement for a B256 performance workload.

After installing this checkout and its runtimes, run from the repository root:

```bash
python tests/fdwic/pa_hbg/test_pa_hbg.py -p a5sim
python tests/fdwic/pa_hbg/test_pa_hbg.py -p a5 -d 0
python tests/fdwic/pa_hbg/test_pa_hbg.py -p a5 -d 0 --enable-chip-swimlane 3 --enable-dep-gen
```

Use the repository's hardware precheck and device reservation procedure before
hardware runs. Simulator timings are not A5 performance measurements. Generated
artifacts go to `outputs/`, not into this directory.

## Cooperative and business regressions

`spmd/` contains minimal native probes for ordinary waves, GM Mix, runtime order,
`sync_start`, publication and error handling. The CLA-shaped probe preserves a
multi-block graph and fan-in structure; it is not a numerical model benchmark.
`spmd/fixtures/cla_workload.json` is a required test fixture, not a trace to remove.

`csa/test_csa_business.py` uses the frozen CSA kernels, original block counts and
seven golden outputs. Its business group checks numerical behavior; its resident
group additionally requires a newly published AICore runtime mode on every run.
Strict L4 capture also checks producer, execution identities and native causality.
Successful numerical output alone does not establish the scheduling backend.
