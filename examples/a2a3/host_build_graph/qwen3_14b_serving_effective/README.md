# Qwen3-14B Standalone HBG Decode

This example runs a post-prefill Qwen3-14B decode workload through Simpler's
`host_build_graph` runtime. A caller-provided fixture supplies the KV snapshot
and golden metadata; decode positions are overwritten on each run, which executes up to 127 decode
dispatches. Single- and dual-slot execution use separate Python entry points.

## Build the HBG artifact

The input is a generated Qwen decode artifact containing the distributed host
wrapper and one chip callable. The adapter preserves the source in-core programs
and converts the generated child runtime to `host_build_graph`. Generated artifacts
may use either a per-layer Definition or the flat 40-layer orchestration emitted by
the serving compiler; the latter is recorded with `graph_definition_count: 0` and
its emitted task count:

```bash
PYTHONPATH=/path/to/pypto/python:/path/to/simpler/python \
python build_hbg_artifact.py \
  --artifact-source /path/to/generated/artifact/qualification \
  --output-dir /path/to/hbg_decode_artifact \
  --platform a2a3
```

The source may be an artifact root with a `manifest.json`, or the distributed
decode directory itself. Both the 25-argument compatibility ABI and the current
26-argument ABI with `sampled_ids_host` are accepted. The generated manifest
records the source and output checksums, Definition shape, and whether all
in-core binaries remain byte-identical.
Both runners verify the copied metadata, orchestration source/shared library,
and in-core binary checksums before preparing the runtime.

Each decode frame executes the generated 40-layer orchestration, then final
RMSNorm, LM Head, and greedy sampling. The adapter requires the recognized
per-layer generated form and counts the emitted Definition tasks. Unrecognized
or incomplete generated orchestration is rejected.

## Run the benchmark

`run_standalone_0p1.sh` is environment-driven. Supply the runtime dependencies,
fixture, model, and HBG artifact explicitly:

```bash
PYTHON_BIN=/path/to/python \
PYPTO_ROOT=/path/to/pypto \
PYPTO_LIB_ROOT=/path/to/pypto-lib \
PTOAS_ROOT=/path/to/ptoas \
FIXTURE_ROOT=/path/to/fixture/qualification \
MODEL_DIR=/path/to/Qwen3-14B \
ARTIFACT_SOURCE=/path/to/hbg_decode_artifact \
bash run_standalone_0p1.sh DEVICE_ID OUTPUT_DIR [single|dual] [STEPS]
```

The default qualification uses zero warmups, one measured request, 127 decode
dispatches, and steady skip 5. `STEPS` may be reduced for a short correctness
run. Large model, fixture, artifact, and result files remain outside the source
repository.

Dual mode shares immutable weights, RoPE, and the autoregressive KV cache. It
owns separate metadata, logits, hidden-state, and sampled-token storage for
in-flight frames. The validator requires strict slot alternation, independent
slot generations, prepared dispatches after the first frame, serialized device
execution, and the full golden token digest.

## Metrics

- `rts_completion_interval_ms`: consecutive `chip.run` completion timestamps,
  used as the standalone decode TPOT proxy.
- `runner_run_ms`: host child span around the runtime call.
- `effective_ms`: `chip.run.runner_run.device_wall` for HBG.
- `graph_build_ms`: host `node.graph_build` span.
- `bind_ms`: HBG `chip.run.bind` span.

The steady set contains 122 rows after skipping the first five dispatches. HBG
exposes the complete device wall window rather than TMR's separate `.sched`
subspan, and records that source in `trace_summary.json`.
With zero steady skip, completion intervals start at the second dispatch.
A single-dispatch run has zero interval samples and null interval statistics.

The native STRACE log is exported to
`profile/simpler_strace_timeline.json`, which can be opened in Perfetto.

## Correctness contract

A successful full run requires:

- 127 decode dispatches on the selected slot configuration;
- monotonically increasing generation and completion order;
- every sampled token matching the fixture golden;
- fixture, model, and artifact validation before device execution;
- aligned native device STRACE spans;
- an HBG artifact whose Definition contains fewer than 1024 tasks.

The benchmark is a manual hardware entry point. Automated unit tests cover
slot metadata isolation, HBG STRACE aggregation, artifact ABI validation, and
source rewriting without requiring model weights or an NPU.

## Worker.submit adapter

`standalone_adapter.py` is the state boundary for a fixture-backed decode stream. It validates the sampled-token chain, advances `seq_lens`/`slot_mapping`/`block_table`, binds generated ABI buffers with explicit directions, and submits one `NEXT_LEVEL` callback per step. The caller waits for the returned `RunHandle`, reads the ABI-owned sampled output, and calls `complete_step` before requesting the next step. Synthetic hidden-state probes remain separate from this real-fixture adapter.

## Qualify a sealed logical KV reference

`reference_worker_submit.py` consumes a `qwen-reference-logical-kv-v1` bundle and
an appended-KV reference identifying the same initial bundle. It verifies model
hashes, materializes independent BSND pages for 16 requests and executes the real
25-argument callable through `Worker.submit` at launch depth one.

```bash
python reference_worker_submit.py \
  --fixture /path/to/reference-kv \
  --kv-reference /path/to/reference-decode-kv \
  --artifact /path/to/verified-hbg-artifact \
  --model /path/to/Qwen3-14B \
  --device DEVICE_ID --steps 127 --output /path/to/new-result-directory
```

On shared hardware, run this command through `task-submit` after the architecture
precheck and explicit CANN environment setup. The result directory must be new.
The consumer checks exact tokens and finite logits/KV with relative L2 at most
0.05 and cosine similarity at least 0.999. Initial prefixes must remain bitwise
unchanged; unused capacity must stay zero. The next input is the actual previous
sampled output after completion and readback. The report identifies the tested
sources and runtime binaries. A failed gate stops the chain.

The original prefill bundle remains sealed. The appended-KV bundle contains
`appended.safetensors` with `key_00`/`value_00` through `key_39`/`value_39`, each
`[8, 127, 128]`, and a manifest recording `reference_sums_sha256`,
`appended_sha256`, `steps`, `layers`, `layout`, and reference token/logit equality.

The host feedback edge is serial. This qualification does not establish DEVICE
early enqueue; see the [execution map](../../../../docs/qwen-step2-execution-map.md).
