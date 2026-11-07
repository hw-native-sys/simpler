# Qwen3-14B Program Serving and Standalone TMR Decode

`run_program_serving.py` runs complete requests through PyPTO Serving's
`AsyncLLMEngine`: its scheduler, worker, real checkpoint loader, resident KV,
prefill, decode and greedy sampler. It checks the prefill-produced first token
and every subsequent token against the existing sealed serving fixture.
This is a Program eager TMR request gate. It does not qualify HBG or any
Kernel eager/capture combination.

## Program serving request gate

Install PyPTO Serving, PyPTO, PyPTO-Lib and this checkout's Simpler in a local
environment using their normal build instructions. The API used here was checked
against serving `61b350e01a57afe9aea4a15af648f3a09f170049` and PyPTO
`15cee98ec9d065283f1310174a4bc5a47b925e0e`. Serving pins Lib
`ec58b8ad7454b321f379260e95a0ae4431a5ae57`; this differs from Lib main
`a948d60581e0ec95b339859cb4554b6f51e456cb`. Those source observations are
not a qualified composite build or a reason to bypass a dependency pin.

Supply the existing complete `serving-tmr-standalone-fixture-v1` fixture,
including its referenced golden/artifact, and the original prompt text. The
fixture authenticates the checkpoint files and golden. The loaded serving
tokenizer must produce exactly the fixture's prompt IDs for all 16 reference
rows, and the full 127-step reference feedback chain must agree. No KV snapshot
is uploaded by this entry: the Engine performs prefill and retains its own KV
through the request. Weight loading remains the Engine's per-layer implementation.

After the required architecture precheck, run in an exclusive device allocation:

```bash
task-submit --device auto --device-num 1 --timeout 2400 --max-time 2400 --run \
  'python examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_serving_effective/run_program_serving.py \
    --model /path/to/Qwen3-14B --prompt /path/to/prompt.txt \
    --fixture /path/to/fixture/qualification --device "$TASK_DEVICE" \
    --requests 16 --max-tokens 128 --rounds 2 --output /path/to/new-result'
```

The existing serving source path must resolve its Lib checkout (for a wheel,
use its documented `PYPTO_LIB_ROOT`). Source CANN/toolchain settings before
launch, and retain the normal task-submit log. This command is a full model
integration workload, never a default CPU test. `--requests 1` and fewer
tokens are explicit diagnostic subsets; the full gate is 16 requests and
128 tokens per request. The model worker starts once and serves fresh requests
in each round. Output directories must be new.

The gate fixes greedy sampling, ignores EOS to consume the full reference,
uses BF16 weights/KV, 128-token pages, a 4096-token context limit, no prefix
cache, synchronous scheduling, and serving's chunked prefill with a 4096-token
scheduling budget. The compile batch stays 16, including when only one request
is active. Ring heap is 4 GiB and task/dependency capacities are 262144. Model
startup includes serving's own allocation/profile warmup; no capture is performed.

`result.json` retains observed and expected tokens, input digests, loaded parent
module/binding identities, package origins, Lib sources, generated artifacts,
startup and request timings, and separate execution/cleanup errors. The existing
`PYPTO_PROG_BUILD_DIR` diagnostic output is scoped to this invocation. A failed
request cancels and drains its siblings before shutdown. `Engine.stop() == False`
is a failure, with `closed: false`; a cleanup failure never replaces the original
request error. The entry does not reset a device or retry a failed workload.

Success is named `token_gate_passed`, with `six_cell_acceptance: not_qualified`.
The Engine's public output API does not expose logits or per-layer KV, so this
entry cannot check their numerical error or untouched pages. Parent-process
binding identities do not establish every child process's loaded libraries.
TTFT, TPOT and end-to-end values are host-observed serving samples, include
scheduler overhead, and are not device or host-submit latency. Preserve the
existing tensor/KV qualification tools and separate device measurements for
full acceptance. HBG additionally requires serving runtime/pipeline configuration
and legal HOST prefill control lowering in its compiler/model dependencies;
changing a label on this TMR entry does not provide that baseline.

## Standalone decode

The standalone benchmark entry points run the post-prefill Qwen3-14B decode workload directly through
Simpler's `tensormap_and_ringbuffer` runtime. They never import or start
Serving. A fixture contains the KV snapshot and golden metadata produced by
one prefill; each run restores that snapshot and executes the consumed 127
decode dispatches. The `single` and `dual` modes are separate Python entry
points so the dual-slot pipeline cannot change the single-slot execution
layout.

## Running

`run_standalone_0p1.sh` is environment-driven so it can be used with a local
checkout, a CI worker, or a `task-submit` command. Supply the device and
output directory as positional arguments, and select `single` or `dual` as the
optional third argument:

```bash
PYTHON_BIN=/path/to/python \
PYPTO_ROOT=/path/to/pypto \
PYPTO_LIB_ROOT=/path/to/pypto-lib \
PTOAS_ROOT=/path/to/ptoas \
FIXTURE_ROOT=/path/to/fixture/qualification \
MODEL_DIR=/path/to/Qwen3-14B \
ARTIFACT_SOURCE=/path/to/artifact/qualification \
bash run_standalone_0p1.sh DEVICE_ID OUTPUT_DIR [single|dual] [STEPS]
```

The default is a zero-warmup, one-measured-run qualification. `STEPS` defaults
to 127 and may be reduced for a fast validation run. The caller owns output
and input locations; no private absolute paths are embedded in this case.

Required environment variables are `PYTHON_BIN`, `PYPTO_ROOT`,
`PYPTO_LIB_ROOT`, `PTOAS_ROOT`, `FIXTURE_ROOT`, `MODEL_DIR`, and
`ARTIFACT_SOURCE`. No `pypto-serving` path is required at runtime, and the
large KV/artifact payloads are deliberately kept outside the source repository.

## Reproducibility

`stack_manifest.json` records the exact software and input contract used to
produce the reference fixture and artifact. The reference stack is Simpler
`56511f80`, PyPTO `f1a2f35f`, pypto-lib `7f8d7ea8`, Python 3.10, torch/torch-npu
2.6, CANN 9.0 and PTOAS 0.57. Reproduce those versions when comparing against
the frozen numbers; regenerate the artifact when using a different PyPTO or
pypto-lib ABI.

The case fixes batch size 16, 127 consumed decode dispatches, steady skip 5,
ring task/dependency capacity 131072, and ring heaps 1/1/1/8 GiB. Every output
token is checked against the fixture golden before the run succeeds.

Dual mode keeps immutable weights, RoPE and the autoregressive KV cache shared.
It uses separate metadata, logits and next-hidden buffers for the two in-flight
frames. The dual validator requires strict native slot 0/1 alternation,
independent per-slot generations, prepared dispatches after the first frame,
serialized device execution and an exact full-output golden SHA.

## Metrics

- `effective_ms`: `chip.run.runner_run.device_wall.sched` for each decode
  dispatch. The steady set contains 122 rows after skipping the first five.
- `rts_completion_interval_ms`: consecutive `chip.run` completion timestamps
  within the decode sequence. This is the standalone decode TPOT proxy, not a
  full Serving request TPOT.
- `runner_run_ms`: host child span around the runtime call.

`trace_summary.json` contains aggregate statistics. The native Simpler STRACE
log is exported to `profile/simpler_strace_timeline.json`, which can be opened
by Perfetto.

## Correctness Contract

A successful run requires:

- 127 decode dispatches on the selected slot(s);
- monotonically increasing generation and completion order;
- every sampled token matching the 127-step golden;
- the first decode input matching the fixture's prefill token;
- fixture, model and artifact checksum validation before device execution;
- native device STRACE phases present and alignable to their host invocation.

The scripts are a hardware benchmark entry point rather than a default CI
scene test. Preserve the generated `benchmark.json`, `trace_summary.json`,
timeline and `SHA256SUMS` files with each result.
