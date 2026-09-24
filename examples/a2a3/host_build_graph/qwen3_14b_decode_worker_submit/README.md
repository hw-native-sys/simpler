# Qwen hidden-state fixture through Worker.submit

This driver sends the existing deterministic Qwen decoder fixture through a local
level-3 Worker and one HBG NEXT_LEVEL task per request. It uses random replicated
layer weights and synthetic KV. It checks hidden output and KV against the torch
reference. It does not produce sampled tokens.

`--depth` sets both Worker launch depth and the number of independent fixture
requests submitted before waiting. Immutable host inputs are shared; mutable KV
and output buffers are separate. This is an independent-request runtime probe;
it does not implement an autoregressive token feedback chain.

Run through the shared device queue after loading the CANN environment:

```bash
task-submit --device auto --device-num 1 --max-time 1800 --run \
  '.venv/bin/python examples/a2a3/host_build_graph/qwen3_14b_decode_worker_submit/main.py \
   -p a2a3 -d "$TASK_DEVICE" --depth 1'
```

The fixture requires approximately 38 GiB of staging per prepared request.
Read [execution evidence](STEP2_REPORT.md) for the tested code and its limits.
The full token-producing artifact path lives in
[Qwen serving effective](../qwen3_14b_serving_effective/README.md).
