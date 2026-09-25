# Qwen step-two real-input evidence

The qualified consumer uses the sealed `reference-kv-20260924` bundle and a
per-layer HBG artifact with the corrected next-layer RMS weight view. Results
below cover the real `Worker.submit` path on A3; earlier fixture shape checks do
not establish model correctness.

## Reference identity

- Bundle `SHA256SUMS` digest:
  `6659b433406cc99d9bd20df604e3faf10f07526e3d4b752af1bb4dc3035b73e3`.
- Qwen config digest:
  `e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a`.
- Model index digest:
  `62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc`.
- Reference implementation: Transformers 5.16.1 Qwen3 eager attention,
  torch 2.6.0 and torch-npu 2.6.0, BF16.
- Fresh prefill and page conversion/restoration matched reference KV and logits
  bitwise. An independent FP32 full-prefix/cache comparison had relative L2
  `3.21e-6`. BF16 full-prefix versus incremental execution is not bitwise equal.
- The separate appended-KV export reproduced all 127 frozen token/logit outputs
  bitwise before saving the newly written positions for all 40 layers.

## Consumer evidence

The tested tree is based on `badb5e4c`, with the review working changes applied.
The full-run report records exact consumer-source and loaded-runtime artifact
hashes; the working changes are not represented as a published commit.

- Baseline task `task_20260924_084843_18736539585`: all 16 first-step tokens
  matched, but logits relative L2 was about 0.9983, so numerical acceptance failed.
- Corrected-view task `task_20260924_085207_196497314981`: all first-step tokens
  matched and logits relative L2 was approximately 0.0287 to 0.0360.
- Eight-step task `task_20260924_085502_204337419442`: exit zero; all 128 decode
  tokens matched. All-layer KV checks passed at the first and final steps.
- Full task `task_20260924_085739_211464628886`: exit zero; 127 steps and all 2,032
  decode tokens matched. Maximum logits relative L2 was 0.034342 and minimum
  cosine similarity was 0.999498. The final appended-KV maximum relative L2 was
  0.019716. Every checked original prefix was bitwise unchanged and unused tail
  capacity remained zero. The chain crossed the page boundary at step 118.

The RMS correction restores a next-layer weight slice. The outlined kernel's
scalar index is zero, so passing the complete weight table selected row zero
instead of the next layer. This controlled comparison supports that correction;
it does not establish the cause of NaNs in the earlier candidate prefill.

The original candidate contained NaNs in 37 of 40 layers. It is excluded from
correctness qualification. The qualified reference bundle remains unchanged.

## Scope and handoff

Real-data depth-one correctness and the host/device dependency classification are
established for this workload. The closeout probe
`task_20260925_043921_34195018347` requested `launch_depth=2`, applied the
explicit safe serial fallback required by host sampled-token feedback, and
completed all 127 dispatches with KV readback at steps 0, 117, 118, and 126.
DEVICE joined early enqueue is not demonstrated; the capability handoff is
detailed in the [execution map](qwen-step2-execution-map.md) and the depth-2
decision record.

The distinct-request identity bundle is also generated and independently
validated: 16 prompt-token rows differ, the shared prefix is checked bitwise,
and each request owns a real KV tail. Its HBG consumer qualification is a
follow-up boundary. The current HBG attention path reads the uploaded distinct
KV bitwise correctly, but its first-step logits differ from the eager reference
by about 0.42 relative L2; the same artifact remains within about 0.03 for the
original single-request KV. This distinct-request HBG consumer issue is handled
by a separate workstream and is outside the closeout PR described here.

These results do not establish native vLLM batching, dynamic request admission,
A5/TMR coverage, capture/replay, or a performance improvement. The existing
step-two PR #2447 and tracking issue #2429 should be updated with the closeout
PR only after review of the prepared working changes and evidence.
