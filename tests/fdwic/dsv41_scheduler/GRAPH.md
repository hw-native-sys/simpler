# Repeated native Graph benchmarks

Four DSL cases compare ordinary `rt_submit_task` replay with native `rt_submit_graph`.
Both modes use the same original kernel bytes, arguments, block counts and sync-start flags.
No runtime scheduling source is changed. The existing three-configuration, single-segment
baseline remains in [RESULTS.md](RESULTS.md); its numbers are not an equal-work comparator for attention repeated eight times.

## Current baseline

The retained matrix contains 20 configurations. These four Flat/Graph pairs are
included there; the MoE/layers Flat x1 configurations share the ordinary DSL rows.
The only current mean table and the paired L1 measurement/model contract live in
[RESULTS.md](RESULTS.md). See [baseline_matrix.json](baseline_matrix.json) for the selection.

## Reuse and correctness contract

- `swa` and `c1a_reuse` submit the same Graph definition eight times within
  one execution. Flat repeats the same segment eight times. Repetitions are configurable
  from 1 to 128; the default is 8.
- Instances are serial: a Graph depends on its predecessor's logical completion. Flat
  joins the preceding segment and gates the next segment's roots. Internal task dependencies
  and sync-start flags remain those of the source capture. Used Graph boundary buffers
  are conservatively INOUT; this test does not assess concurrency between instances.
- All repetitions reuse the fixed snapshot and backing buffers. KV positions and input
  values do not advance. These are eight repeated attention segments, not eight decoding
  steps or eight different model layers. Every run compares writable backing with the
  original prefix-plus-segment reference, which also verifies that repetition preserves
  this fixture's result.
- `moe` and `layers_19_21` each submit one Graph containing their original EP2 segment,
  including real communication and deferred waits. Repeating those communication protocols
  requires separate counter-generation and buffer-lifetime handling and is rejected here.
- The callback uses Graph boundary tensor provenance and inherited communication-context
  scalars. It does not capture a first instance's concrete GM addresses or context pointer.
  Intermediate allocations remain local to each Graph instance.
- Execution audits check task shapes, complete block/lane sets, unique identities,
  native child ownership and serial Graph dependencies. The audit checks execution
  structure; it is not a proof of all asynchronous timing orders.
- Inputs retain the zero-weight fixture: 72 valid tokens per rank, padded to 80 rows,
  64 local experts. This test does not establish arbitrary-checkpoint numerical accuracy,
  changing `recv_count`/`route` behavior, or a general data-visibility proof.

## Reproduce

Prepare original bundles and build native tools as described in [README.md](README.md).
From the repository root, create separate bundles for the two submission modes:

```bash
DSV41=tests/fdwic/dsv41_scheduler
for mode in flat graph; do
  python3 "$DSV41/production/graph_cases.py" \
    "$DSV41/build/prepared/cases/pypto_lib_dsl/swa" \
    "$DSV41/build/graph-cases/swa/$mode" \
    --runner "$DSV41/build/portable/host/native-runner" \
    --replay "$DSV41/build/portable/host/libreplay.so" --mode "$mode" --repetitions 8
done
```

Use the same command for `c1a_reuse`. For `moe` and `layers_19_21`,
omit `--repetitions` to select one. The generated immutable `graph.txt` records mode
and repetitions; generate a new bundle to change them. Each callable retains the
original orchestration slot size and kernel offsets.

Reserve one device for each attention pair or two for each EP2 pair; pass the assigned
`TASK_DEVICE` to both runs. Run the architecture precheck inside that allocation on
shared hosts, then invoke `production/run.sh BUNDLE RUNTIME OUTPUT 1 1 10` for each
mode. Both reported timings come from each same L1 execution, after one L1 warmup.

Audit each diagnostic capture, for example:

```bash
PYTHONPATH=.:python python3 "$DSV41/production/audit_graph.py" \
  "$DSV41/build/graph-cases/swa/graph" "$DSV41/outputs/graph/swa/graph/rank0/run1" \
  "$DSV41/workloads/pypto_lib_dsl/swa/tasks.tsv"
```

Repeat for every measured run and rank. Keep raw samples and capture files locally;
[RESULTS.md](RESULTS.md) retains the single current mean table.
