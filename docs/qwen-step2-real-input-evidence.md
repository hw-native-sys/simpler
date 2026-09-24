# Qwen step-two real-input evidence

The authorized data restore recovered the generated serving artifact and the external
model/prompt inputs. The prefill fixture and its KV/golden payload were not present in
the authorized Zhangtao workspace and remain an execution prerequisite.

## Restored inputs

| Input | Location or provenance | Evidence |
| --- | --- | --- |
| Qwen3-14B checkpoint | `/data/models/Qwen3-14B` | `config.json` SHA256 `e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a`; index SHA256 `62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc`; tokenizer SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; generation config SHA256 `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |
| Prompt | `/data/sunkaixuan/lcw_subdir/tmp/long_prompt_3.5k.txt` | text SHA256 `6b3e99d56ed58d58d98f149be7a68d2e454df2447414ac51f3f46da92adb9dfd`; 3338 tokenizer tokens; compact token-ID SHA256 `8db4a9a6897a5ae237fd3806ce39b4d49a549f8faf2881d5ae5740d5e6be6bb9` |
| Generated decode artifact | `/data/pyptouser/zhangtao/zt/simpler/build/pypto-serving/artifacts/tmr_compile_cache/serving_dp0_d0/decode_fwd` | copied to `/data/sunkaixuan/lcw_subdir/tmp/qwen-real-artifact-source`; `distributed_meta.json` SHA256 `03b1a7ea33b130525f5fff1937ea8f60b723079070ba08939f15d5c4132b4fc6`; 40 in-core binaries; 25-argument ABI |
| HBG artifact | generated under `/data/sunkaixuan/lcw_subdir/tmp/qwen-real-hbg-artifact` | manifest records runtime `host_build_graph`, flat graph (`graph_definition_count: 0`), 279 emitted tasks per frame, 40 in-core binaries, and restored source binary checksums |

The generated artifact contains the complete orchestration C++/SO and kernel sources.
The builder regenerates the orchestration shared library with the current PyPTO runtime,
then restores the source in-core payload byte-for-byte. Any content-addressed binary
name or digest changes observed during assembly are retained in the manifest as
`assembly_changed_incore_bins` provenance. The current `Worker.submit` bridge validates
those frozen binary checksums and recompiles the verified C++ sources through Simpler's
source-based callable compiler; it does not load the historical SO or binary payload
into the Python process.

## Validation

`validate_real_inputs.py` successfully validated the model, tokenizer/prompt contract,
and restored HBG artifact. The report is saved at
`/data/sunkaixuan/lcw_subdir/logs/vllm-step2-real-inputs-artifact.json`.

The focused CPU suite passed 34 tests covering the HBG adapter, artifact bridge, input
validator, slot lifetime, and Worker.submit driver. This evidence does not claim a
real token-level decode: the prefill fixture, KV shards, and golden sampled-token
sequence still need to be supplied before device qualification.
