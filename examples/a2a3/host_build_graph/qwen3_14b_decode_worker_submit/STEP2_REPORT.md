# Qwen fixture execution evidence

The recorded tasks used runtime baseline `ee27950582c1a4c8a92dc9756264cb8ed6ec2f58`,
A3 device 1, and CANN 9.0.0. They used deterministic random tensors with replicated
layer weights. The callable produces hidden output and KV updates; it has no
embedding, LM head, or sampled-token output.

- `task_20260922_214625_200553426315`: one submission, golden skipped, exit 0.
- `task_20260922_214915_21565905113`: one submission with hidden/KV golden, exit 0.
- `task_20260922_215600_246694023753`: two submissions, golden skipped, exit 1.
  The second bind requested 40,860,165,120 bytes of retained staging and failed
  with allocation error 207001. The first run completed successfully.

The driver used for these tasks did not pass `launch_depth` to Worker. All three
therefore used the default launch depth of 1. The two-submission failure is
prepared-successor staging evidence, not a depth-2 early-launch test. Its allocation
size is approximately 38.05 GiB (40.86 GB), not a measured execution-scratch peak.
The log's whole-operator timestamp was unavailable (`wo_rc=-1000`); it does not
provide a valid whole-operator ordering proof.

The driver now forwards its depth setting to Worker, with a CPU configuration
regression. Hardware validation of that change is pending. The Qwen checkpoint,
prompt contract, generated decode artifact, and host-wrapper ABI are recorded in
`docs/qwen-step2-real-input-evidence.md`. A real prefill fixture with KV shards and
sampled-token goldens, followed by device validation through the restored artifact
bridge, is still required for token-level autoregressive qualification. These tasks
cannot establish those properties or a general Qwen capacity limit.
