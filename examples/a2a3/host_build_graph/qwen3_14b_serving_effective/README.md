# Qwen3-14B Standalone HBG Decode

This example runs a post-prefill Qwen3-14B decode workload through Simpler's
`host_build_graph` runtime. A caller-provided fixture supplies the KV snapshot
and golden metadata; decode positions are overwritten on each run, which executes up to 127 decode
dispatches. Single- and dual-slot execution use separate Python entry points.

## Build the HBG artifact

The input is a generated Qwen decode artifact containing the distributed host
wrapper and one chip callable. The adapter preserves the source in-core programs
and converts the generated child runtime to `host_build_graph`. The adapter accepts
the recognized per-layer Definition form and rejects incomplete or unrecognized
orchestration:

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

### Reference workload and validation

The reference workload uses Qwen3-14B's 40 layers, hidden size 5120, eight KV
heads and head dimension 128. Weights and KV use BF16; logits use FP32 and
sampling uses greedy argmax. The fixture contains one real 3338-token prompt,
replicated across 16 requests with independent physical pages. It does not
reproduce a vLLM scheduler's batch allocation history or qualify distinct prompts.

The initial KV snapshot covers prompt positions 0 through 3337. The first
generated token has been sampled but is not cached. The 127 decode dispatches
therefore produce 128 generated tokens including that prefill token. Every
frozen token is compared, including EOS; EOS does not stop qualification early.

Logical initial KV has shape `[1, 8, 3338, 128]`, with keys after Q/K normalization
and RoPE. Each physical layer cache uses BSND shape `[448, 128, 8, 128]`, with
28 pages per request and a 32-column block table. Both complete caches occupy
8.75 GiB of device storage. Decode step 118 crosses a page boundary.

The loader checks bundle checksums, model checkpoint hashes, supported geometry,
finite KV values and reference restoration. `reference_fixture.py` translates
the logical bundle into the standalone adapter contract; the
`serving-tmr-standalone-fixture-v1` schema remains separate. The qualified
25-argument artifact has 39 in-core kernels and 277 tasks per layer. The bridge
recompiles verified sources against the selected Simpler headers and tools;
historical binaries provide provenance. The bridge's 26-argument ABI with
`sampled_ids_host` is outside this reference consumer's hardware qualification.

The result records run IDs, positions, slots, artifact and consumer-source hashes,
and loaded-runtime binary hashes. Hardware measurements are recorded in
[PR #2447](https://github.com/hw-native-sys/simpler/pull/2447) and
[PR #2456](https://github.com/hw-native-sys/simpler/pull/2456).

### Execution dependencies and lifetime

The reference consumer uses `Worker(level=3)` with A3 `host_build_graph` and one
local chip child. Each step uploads the actual previous sampled token and step
metadata, calls `Worker.submit`, waits for `RunHandle.result(timeout=120)`,
explicitly copies sampled output and logits to the host, validates them, and
calls `complete_step`. Reference tokens are comparison data, never substituted
future inputs. `next_step()` rejects a second reservation while a step is in flight.

- Weights and RoPE are read-only and uploaded before the first run.
- KV is shared mutable device storage; the next step waits for the prior run's
  completion. Metadata values may be prepared early, but their device buffers
  cannot be reused before the previous consumer completes.
- Logits, sampled output and scratch storage are reused only after completion
  and required readback. Report-owned copies preserve consumed results.
- Worker-owned device allocations remain resident through `Worker.close`.
  Temporary host upload buffers close after synchronous copies complete. Run
  handles remain available through the chain. Device completion, host visibility
  and eligibility for release are separate events.

The host feedback edge is serial. Passing `--depth2-probe` records
`launch_depth_requested=2`, but keeps effective `launch_depth=1` and reports
`depth2_conclusion=safe_serial_fallback_host_sampled_token_feedback`. It does not
enable early enqueue. A dependent successor would require a device sampled-output
to next-input dependency, per-run metadata/output storage, last-consumer lifetime
guarantees, and runtime support for joined DEVICE-backed launches.

A numerical mismatch, run error or timeout stops further submission. Registration
and initialization failures also trigger Worker cleanup. The consumer closes the
Worker before writing the final report, and cleanup failure marks the report
failed. An execution error remains primary if cleanup or report writing also
fails; secondary errors remain in its exception context.

This reference qualification covers serial correctness on A3 HBG. It does not
establish full vLLM serving, dynamic request admission, A5/TMR coverage,
capture/replay, implicit host/device synchronization or a performance improvement.

## Independent requests

`clone_worker_submit.py` runs two logically independent requests with the same
Qwen fixture on one A3 HBG Worker at `launch_depth=2`. Immutable weights and RoPE
are shared; each live request owns its own KV, block table, metadata, sampled
token, logits, hidden output and run handle. Decode metadata is host staged at
bind time so a successor can prepare without a direct-control copy behind the
predecessor. Within either request, the next decode step uses that request's
actual sampled output after completion.

```bash
python clone_worker_submit.py \
  --fixture /path/to/reference-kv \
  --kv-reference /path/to/reference-decode-kv \
  --artifact /path/to/verified-hbg-artifact \
  --model /path/to/Qwen3-14B \
  --device DEVICE_ID --steps 127 --output /path/to/new-result-directory
```

The runner checks free device memory before allocating the second request and
records a capacity failure when both requests cannot remain resident. A pass
requires exact sampled tokens, logits/KV numerical gates, request-private
storage isolation, and an attributable joined-launch trace for every decode pair
showing the second enqueue before the first operator ends while device execution
remains serial.
