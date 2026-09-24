# Qwen step-two workload contract

Status: model, tokenizer/prompt contract, and generated decode artifact restored; the prefill fixture and KV/golden payload are still required for device execution.

## Existing source of truth

The repository already contains a post-prefill Qwen3-14B HBG serving path under `examples/a2a3/host_build_graph/qwen3_14b_serving_effective/`. Its checked-in reference manifest records:

| Field | Value |
| --- | --- |
| Model | Qwen3-14B |
| Dtype | BF16 |
| Prompt tokens | 3338 |
| Batch size | 16 |
| Output tokens | 128 |
| Decode dispatches | 127 consumed dispatches |
| Steady skip | 5 dispatches |
| Runtime slots | 0 and 1 |
| KV backing pages | 691 |
| Runtime | A3 `host_build_graph` |
| Definition shape | One Definition invoked 40 times per frame |
| ABI | 25 arguments, or 26 with `sampled_ids_host` |

This reference is a workload shape contract. It does not contain the external model checkpoint, prefill snapshot, generated decode artifact, tokenizer files, or their checksums.

## Prefill snapshot contract

`fixture.py` defines the executable fixture schema as `serving-tmr-standalone-fixture-v1`. The fixture must prove all of the following before decode starts:

- all chunked prefill work is complete;
- the first token has been produced;
- no decode dispatch has run before the snapshot;
- prefill and sampling fences have retired;
- prompt token IDs have shape `[16, 3338]` and dtype `int32`;
- the first generated token, sequence lengths, block table, next slot mapping, and token counts are mutually consistent;
- every KV shard and metadata file matches its SHA256 manifest.

The first decode input is the prefill fixture's `first_generated_token_ids`. The first decode position is named by `next_slot_mapping`; subsequent positions advance the page/block metadata in the golden contract. The fixture provides golden sampled-token rows for the remaining decode dispatches.

## Runtime and artifact contract

The real serving driver validates a caller-provided Qwen checkpoint, fixture, and generated decode artifact. The artifact builder accepts only the 25-argument compatibility ABI or the 26-argument ABI that adds `sampled_ids_host`. The required semantic arguments include `out`, `embed_weight`, `sampled_ids_in`, `sampled_ids`, and `next_hidden`; the exact order comes from the artifact's `distributed_meta.json`.

The restored generated artifact has a 25-argument ABI, 40 in-core binaries, and a flat 40-layer orchestration. The HBG adapter rebuilds the orchestration shared library with the current runtime, restores the source in-core payload byte-for-byte, and records the seven assembly changes in its manifest. The flat source emits 279 tasks per frame; per-layer Definition artifacts remain supported when present. The verified HBG manifest and every copied metadata, source, shared-library, and in-core checksum are required before runtime use.

## Fields that remain unfrozen

The checkpoint and prompt contract are recorded in `docs/qwen-step2-real-input-evidence.md`. The following values must still be supplied and recorded before the real workload is declared frozen:

- CANN, torch-npu, vLLM/vLLM-Ascend, PyPTO, PyPTO-lib, and PTOAS versions;
- fixture manifest SHA256, metadata SHA256, KV shard checksums, and golden output-token SHA256;
- the device run evidence for the restored artifact and the selected 25-argument Worker adapter ABI.

The current `qwen3_14b_decode_worker_submit` manifest is therefore an implementation probe. It must be replaced or extended with these concrete values before step two closes.
