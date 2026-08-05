# paged_attention — host_build_graph

Online-softmax paged attention in bfloat16, split across AIC and AIV, on the
`host_build_graph` runtime. The kernels and the orchestration source are the
same ones the `tensormap_and_ringbuffer` sibling compiles.

## Why the same orchestration compiles on both runtimes

`orchestration/pto_orchestration_api.h` is identical under
`src/a2a3/runtime/host_build_graph/` and
`src/a2a3/runtime/tensormap_and_ringbuffer/` — same entry points, same macros.
`KernelCompiler.compile_orchestration(runtime, source)` picks the include dirs
of whichever runtime `@scene_test(runtime=...)` names, so switching runtime is a
one-line change in the test class and nothing in the C++.

## The four-task loop

Per (batch, head) and per KV block the orchestration submits four tasks:

| Task | Core | Computation |
| ---- | ---- | ----------- |
| `QK` | AIC | `qi @ K^T` for the block |
| `SF` | AIV | softmax prepare — running `mi`, `li` |
| `PV` | AIC | `P @ V` |
| `UP` | AIV | online-softmax accumulation into the running output |

Ordering is not written down: the tasks sit in a plain `PTO2_SCOPE()` with a
`PTO2_SCOPE_GUARD()` and the dependency graph is derived from the tensors they
share. See `../paged_attention_manual_scope/` for the variant that wires the
same edges by hand.

## Ring sizing is the one place host_build_graph differs

`host_build_graph` builds the entire task graph on the host before the device
starts scheduling, so no ring slot can be reclaimed mid-orchestration — the ring
window and GM heap must hold every task of a case at once. Task count is

```text
tasks = batch * (4 * ceil(context_len / block_size) + 1)
```

so `Case1` needs 65 792 slots and `Case2` needs 32 832, both above the default
window of 16384. Each carries its own `runtime_env` in `CASES[*]["config"]`
(`ring_task_window` must be a power of two in `[4, INT32_MAX]`; `ring_heap` at
least 1024), so no `PTO2_RING_*` environment variable is needed.

## Cases

`Case1` (65 792 tasks) and `Case2` (32 832) at production scale, `CaseSmall1` /
`CaseSmall2`, and `CaseVarSeq2` / `CaseVarSeq4` for ragged sequence lengths.
All but `CaseSmall1` are `manual`, so the default run executes `CaseSmall1`
only; add `--manual include` to reach the rest.

The upstream `tensormap_and_ringbuffer` example also carries a `Case3`
(`head_dim: 256`). It is absent here because it does not produce correct
results on either runtime — see `KNOWN_ISSUES.md`.

## Run

```bash
python examples/a2a3/host_build_graph/paged_attention/test_paged_attention.py \
    -p a2a3 -d 0                                   # CaseSmall1, golden checked
python examples/a2a3/host_build_graph/paged_attention/test_paged_attention.py \
    -p a2a3 -d 0 --manual include --case Case2 --rounds 2
```
