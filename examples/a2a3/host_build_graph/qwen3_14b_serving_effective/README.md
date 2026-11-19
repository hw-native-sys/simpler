# Qwen3-14B HBG Decode

This directory contains a fixed-shape Torch-NPU eager adapter for the Qwen3-14B
HBG release. It uses BF16 weights and KV on A3, the stage-0 25-argument ABI,
one active request, the existing borrowed-context Worker API, caller-stream
ordering, RunHandle retirement, and explicit close.

The callable has no inactive-row mask. Row 0 carries the request. Rows 1–15
execute the same token and position with private KV pages, and only row 0 logits
are returned. Padding rows avoid every page reserved in the request's block
table, including pages needed by later decode steps. A 32-column table is
accepted only when columns beyond the 28-column ABI contain `-1`.

## Artifact

Obtain the sealed stage-0 `artifacts/qwen3-14b-hbg-release` bundle through
[#2429](https://github.com/hw-native-sys/simpler/issues/2429), together with its
producer source/environment manifests, source patches and artifact digest manifest. The producer
compiles the PyPTO-lib `models/qwen3_14b` decode contract with PyPTO's
`HOST_BUILD_GRAPH` pass context, then assembles the emitted `decode_fwd` child
with prebuilt binaries. Reproduction uses the source revisions and patches
recorded in those manifests, batch 16, maximum sequence length 3466, page size
128, and the pinned 40-layer model config.

Set `ARTIFACT` to the extracted release root and verify its identity before
running either smoke command:

```bash
export ARTIFACT=/path/to/qwen3-14b-hbg-release
test ! -e "$ARTIFACT/hbg_artifact_manifest.json"
printf '%s  %s\n' \
  ea179cb8eec8623970ae38b8b9965470e070c5ef3f0f308fcd1834727dce8d92 \
  "$ARTIFACT/distributed_meta.json" | sha256sum --check
```

Pass this directory as `--artifact "$ARTIFACT"`. Its layout selects the stage-0
reader, which verifies metadata, sources, registration order, tensor directions,
binary paths, sizes and SHA256 values before compiling against the current
Simpler runtime. The release has 25 external tensors, 32 registered in-core kernels,
40 Qwen layers, 128-token pages, BF16 KV in BSND storage, and FP32 logits with
the valid vocabulary range `[0, 151936)`.

`build_hbg_artifact.py` handles the older 39-41-kernel input and writes
`hbg_artifact_manifest.json`, selecting the manifest-backed reader. That output
uses a separate artifact contract and cannot substitute for this stage-0 bundle.

The stage-0 release directory and its producer must be trusted: its
`kernel_config.py` executes during inspection and its generated C++ sources
are compiled. Binary manifests verify the declared binaries. Load-time hashes
of the config and all compiled sources are recorded and rechecked before
compilation; these snapshots do not authenticate initially supplied source.

## Torch-NPU smoke

`run_torch_eager.py` loads the stage-0 checkpoint and release, allocates the
fixed batch-16 KV backing, and submits two decode steps with one active request.
It checks finite FP32 logits, every layer's active KV slot, completion, close, and caller
stream use after close. The zero-prefix KV input validates execution and
lifecycle wiring; it is not a prefill numerical reference.

Config, weight-index and consumed shard digests are checked against fixed pins
and recorded in the receipt. Success and failure receipts include completed
steps, the current step/position, and separate execution and cleanup errors.
Smoke checks remain active under `python -O`.

When a completed run reports a terminal execution error, `close()` preserves
that primary error while still attempting Worker and Buffer cleanup. Successful
cleanup makes a later close a no-op; an uncertain timeout retains resources for
a retry.

Run it through the shared device queue:

```bash
task-submit --device auto --device-num 1 --max-time 900 --run \
  'eval "$(pypto-setup --export)" && source "$ASCEND_HOME_PATH/set_env.sh" && \
   export TASK_QUEUE_ENABLE=1 && \
   python /path/to/qwen3_14b_serving_effective/run_torch_eager.py \
     --artifact /path/to/qwen3-14b-hbg-release \
     --model /path/to/Qwen3-14B \
     --prompt-ids /path/to/prompt-3338-token-ids.json \
     --device "$TASK_DEVICE" --output /path/to/result.json'
```

The caller owns tensor backing, context, and stream. The adapter retains tensor
backing through completion and closes its Buffer wrappers after retirement.
Returned logits use reusable storage; consume or copy them before the next
decode call.

RunHandle waits use a 120-second timeout, including the wait before close-time
stream synchronization. A timeout latches failure and retains backing and
Buffer wrappers until a later drain succeeds. `Worker.close()` uses its public
cleanup budget; Torch-NPU stream synchronization has no timeout argument, so
the RunHandle timeout is not an overall process deadline.

## Other qualification tools

The directory also contains longer fixture-driven qualification entry points:

- `run_standalone_0p1.sh` measures a sealed KV fixture through the standalone
  Worker adapter.
- `reference_worker_submit.py` compares a sealed logical-KV reference with
  appended decode KV, logits, and greedy tokens.
- `clone_worker_submit.py` exercises two request-private states.
- `device_feedback_worker_submit.py` exercises a bounded DEVICE feedback chain.
- `continuous_worker_submit.py` exercises bounded refill, retirement, and
  failure recovery.

Those tools use their own fixture manifests and acceptance contracts. Their
reports identify the source, artifact, model, and runtime identities used for
each run.

## vLLM Engine Smoke

`run_vllm_eager.py` supports vLLM 0.20.2 / vLLM-Ascend 0.20.2rc1 on a single
A3 with Torch/Torch-NPU 2.10 and NNAL/ATB 9.0. Environment and integration CI
are tracked by [#2715](https://github.com/hw-native-sys/simpler/issues/2715)
and [#2748](https://github.com/hw-native-sys/simpler/pull/2748).
Native vLLM owns prefill, scheduling, greedy
sampling and output; `vllm_eager.SimplerWorker` dispatches decode through the
stage0 bridge and the existing Torch backend. The release stays unchanged,
including its 32 kernels, 25 tensors and `[3466,128]` FP32 RoPE tables.

Requests use the qualified 3338-token prompt, BF16, 128-token pages, eager
execution and private KV. `--requests 16` (default) selects a complete clone
cohort; `--requests 1` selects one active request with isolated fixed-shape
padding. Chunked or mixed prefill, prefix sharing, non-greedy sampling, partial
cohorts and arbitrary request replacement are refused. Each new round retires
and closes the previous backend before starting the next request state.

Run both commands inside one exclusive `task-submit` allocation, with CANN and
ATB sourced. Use `--requests 1` on both commands for the single-active gate:

```bash
python run_vllm_eager.py --backend native --model /path/to/Qwen3-14B \
  --prompt-ids /path/to/prompt-3338-token-ids.json --device "$TASK_DEVICE" \
  --requests 16 --max-tokens 2 --output /path/to/native.json
python run_vllm_eager.py --backend simpler --model /path/to/Qwen3-14B \
  --prompt-ids /path/to/prompt-3338-token-ids.json --device "$TASK_DEVICE" \
  --requests 16 --max-tokens 2 --artifact /path/to/qwen3-14b-hbg-release \
  --reference /path/to/native.json --output /path/to/simpler.json
```

The runner validates fixed model/shard digests and prompt IDs. It checks the
complete output count, native token equality, dispatch statistics and Engine
close, and writes success or failure receipts with separate cleanup errors.
Two generated tokens exercise prefill plus one Simpler dispatch. The 16-request
and single-active results are separate smoke gates; they do not establish the
full 128-token tensor/numerical qualification tracked by #2429.

Projection banks are shared with native prefill through input-major linear
views to avoid retaining a second checkpoint on the device. Splitting packed
projections changes the floating-point computation path; matching short-smoke
tokens is not proof of exact native-prefill tensor equality.

`VLLM_ADAPTER_ROOT` can point at the checkout containing this directory, or
at this directory directly, for the integration harness in #2748. Pin the
checkout commit and release identity together when configuring the watch.
