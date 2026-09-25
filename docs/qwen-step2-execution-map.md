# Qwen step-two execution and lifetime map

The qualified path uses one real `Worker(level=3)` with A3 `host_build_graph`,
one local chip child and launch depth one.

## Execution path

```text
sealed logical KV + checkpoint
  -> validate hashes / geometry / reference qualification
  -> register compiled chip callable; Worker.init
  -> alloc_child_tensor; stream weights and BSND KV through host staging
  -> StandaloneDecodeAdapter.next_step
  -> upload actual previous token and immutable step metadata
  -> Worker.submit -> submit_next_level(DEVICE TaskArgs)
  -> RunHandle.result(timeout=120)
  -> explicit sampled-token and logits D2H
  -> validate; copy results into report-owned storage; complete_step
  -> next decode step
```

The consumer retains device allocations through `Worker.close`. Temporary host
upload buffers close after synchronous copy completion. The Worker owns child
allocations; no foreign pointer ownership or process reset authority is acquired.

## Dependencies and reuse

- **Weights and RoPE:** read-only across the chain; upload before the first run.
  Preparation may happen early once their identity and storage are fixed.
- **KV:** shared mutable device storage. Every later read follows the prior run's
  device completion on this driver. Original prompt pages and newly written slots
  are verified separately. Any future early enqueue must preserve whole-op FIFO
  and retain the allocation through its last device consumer.
- **Sampled token:** the first value comes from prefill. Subsequent values are
  read from the actual sampler output after the run fence and D2H. Host preparation
  currently needs that value, so it must wait. `next_step()` rejects a second
  reservation while the prior step is in flight.
- **Sequence length, slot mapping and block table:** values are known from the
  fixed workload. Their host snapshots can be prepared early, but reusing the
  current device metadata allocation waits for the previous consumer.
- **Logits, sampled output and hidden scratch:** resident allocations are reused
  after completion and required readback. Report-owned copies preserve consumed
  results; these allocations are not independent storage for concurrent runs.
- **Run handles:** retained through the chain and read again after the last step.
  Device completion, host visibility and release eligibility remain distinct
  events; the driver waits and copies explicitly.

## Early-enqueue conclusion

The selected runtime's `ChipRunLane::joinable_shape` rejects DEVICE-backed tensor
arguments for joined native launch. The original bounded P4 capability covers
HOST tensors. The separate DEVICE-chain PR #2446 was still open at the review
checkpoint and is not part of this tested baseline.

Even after integrating that capability, this driver needs an explicit device
sampled-output-to-next-input edge, per-run metadata/output storage and a proven
last-consumer lifetime before it can enqueue a real dependent successor early.
A host-substituted reference token would not satisfy the autoregressive contract.

The demonstrated execution mode is safe depth-one serialization. Real Qwen
depth-two early enqueue remains a capability gap for the next integration stage.
No change to runtime admission, implicit host synchronization, group/SUB,
cross-endpoint ordering, A5/TMR or capture/replay is included.

## Failure boundaries

A numerical mismatch fails the chain and records the observed values; no next
step is submitted. A run error or timeout also stops submission, writes the
failure report and enters Worker teardown. Existing runtime contract tests cover
retryable wait timeouts, graph errors, subsequent-submit behavior and close
admission/draining. Those CPU tests are distinct from the successful real-model
hardware runs; hardware fault injection is not claimed.
