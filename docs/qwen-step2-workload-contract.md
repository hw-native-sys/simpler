# Qwen step-two workload contract

The real Qwen3-14B workload is qualified through `Worker.submit` at launch depth one.
The complete input and environment binding is recorded in
[`qwen-step2-workload-manifest.json`](qwen-step2-workload-manifest.json).
The qualification entry is
`examples/a2a3/host_build_graph/qwen3_14b_serving_effective/reference_worker_submit.py`.
It consumes an external, checksum-sealed `qwen-reference-logical-kv-v1` bundle.

## Frozen workload

- Model: Qwen3-14B, 40 layers, hidden size 5120, eight KV heads, head dimension 128.
- Weights and initial KV: BF16; logits: FP32; greedy sampling.
- Prompt: the exact 3338 token IDs in the reference metadata; compact token-ID SHA256
  `8db4a9a6897a5ae237fd3806ce39b4d49a549f8faf2881d5ae5740d5e6be6bb9`.
- Batch: 16 copies of that real prompt, each with independent physical pages.
- Initial snapshot: prompt positions 0 through 3337. The first generated token,
  32313, has been sampled but has not been appended to KV.
- Generation: 127 decode dispatches after the prefill token, giving 128 generated
  tokens. Qualification uses a fixed dispatch count and compares every frozen
  token, including any EOS token; it does not stop early on EOS.
- Logical KV: `[1, 8, 3338, 128]`, with keys after Q/K normalization and RoPE.
- Physical KV: BSND `[448, 128, 8, 128]` per layer, 28 pages per request.
  A 32-column block table describes each request. Both complete caches occupy
  8.75 GiB of device storage. Decode step 118 crosses a page boundary.

This fixture proves real model state and an independent-page batch conversion.
It does not reproduce a vLLM scheduler's original batch allocation history.
The single-request bundle remains the full 127-step numerical qualification
reference.

The original `serving-tmr-standalone-fixture-v1` schema remains separate;
`reference_fixture.py` explicitly translates the logical reference into the
standalone adapter contract.

## Input and artifact validation

The loader verifies the sealed bundle, the model checkpoint hashes, supported
geometry, finite KV values and the reference restoration result. The appended-KV
reference has its own checksum and identifies the same sealed initial bundle.

The qualified artifact has 25 tensor arguments and 39 in-core kernels. Its
40 decoder layers invoke a bounded HBG Definition with 277 tasks per layer.
The builder requires the recognized generated per-layer form and rejects a
failed rewrite. It records source/binary hashes and counts emitted tasks.
The bridge recompiles verified generated sources against the selected Simpler
headers and tools; frozen historical binaries establish provenance rather than
serving as the loaded current-runtime executable.

The generic bridge also recognizes a 26-argument ABI with `sampled_ids_host`.
That form is not covered by this reference consumer's hardware qualification.

## Acceptance

Every sampled token must match exactly. Every row of logits and appended KV must
be finite, with relative L2 at most 0.05 and cosine similarity at least 0.999.
These cross-implementation gates are fixed before the measured qualification.
Original KV prefixes must remain bitwise unchanged and unused tail capacity must
remain zero. Export/restoration equality within the reference implementation is
a separate, bitwise comparison.

The next input comes from the actual previous sampled output. Reference tokens
are comparison data, not substituted future inputs. Each result records run IDs,
positions, slots, artifact hashes, consumer source hashes and runtime binary hashes.

See [execution and lifetime map](qwen-step2-execution-map.md) for the serial host
feedback boundary and [real-input evidence](qwen-step2-real-input-evidence.md) for
hardware results and the remaining early-enqueue capability gap.
