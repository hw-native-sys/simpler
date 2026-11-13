# URMA deferred completion validation

These A5 `tensormap_and_ringbuffer` transfer regressions run two ranks with a registered
communication window. Every transfer is followed by an add-one consumer that
depends on the transfer output. Both the copied bytes and consumer result are
checked against a host golden.

| Test | Transfer | Submissions per producer |
| ---- | -------- | ------------------------ |
| `TestUrmaGet` | TGET from peer input | 1 |
| `TestUrmaPut` | TPUT to peer receive window, then dependent TGET readback | 1 |
| `TestUrmaRepeatedFetch` | TGET from peer input | 16 |
| `TestUrmaRepeatedPut` | TPUT plus dependent readback | 16 |
| `TestUrmaChunkedFetch` | Static-shape TGET of 256 MiB + 32 KiB | 1 logical request, split by the adapter |
| `TestUrmaChunkedPut` | Dynamic-shape TPUT of 256 MiB + 32 KiB plus dependent readback | 1 logical request, split by the adapter |

The lifecycle test requires three devices for two overlapping two-rank domains:
AB uses members `[0, 1]`, and BC uses `[1, 2]` or `[2, 1]`. Both domains stay live
while TGET and TPUT run in alternation. The test checks payloads, dependent
consumers, receive windows, and independent release and recreation in both
directions. It does not exercise simultaneous transfers across domains.

The repeated cases check the number of registered completion entries against
the first request and the requested repeat count. Identical final tensor values
therefore cannot hide a loop that issues fewer requests.

The PUT readback is issued by the sending rank after its deferred PUT task
completes. It reads the destination region on the peer, so it does not depend
on an unsynchronized peer-side consumer. Producers are serialized per actual
queue; these cases do not claim multi-AIV shared-queue concurrency coverage.
The [concurrent sender suite](../urma_concurrent_senders/README.md) covers
independent AIV queue groups, MIX lanes, multiple peers, and physical ring reuse.
The consumer processes full 8192-element tiles; the case builder rejects nonpositive
or unaligned extents. Large cases use approximately 512 MiB of registered window per rank plus local
output and consumer buffers. CPU regressions separately exercise counter wrap,
CQ owner phases, errors and capacity boundaries; the hardware cases do not
inject those conditions.

Chunked GET uses a fully static original tensor shape and strides; chunked PUT
uses runtime shape and strides. Both require the adapter's slices to describe
the smaller transfer with packed strides rather than retain the original extent.

## Build and run

Use a project-local venv and an A5 machine. SDMA and URMA providers are compiled by default.
Domains request URMA resources explicitly and validate their context before submission.
Reserve three devices to run the full directory, including both lifecycle cases.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
source /usr/local/Ascend/cann/set_env.sh
pip install --no-build-isolation -e '.[test]' --config-settings=build.targets=build_package_a5
python -m simpler_setup.tools.scene_test_compile \
  tests/st/a5/tensormap_and_ringbuffer/urma_deferred_completion \
  --platform a5 --require-pto-isa --compile-workers 4
mkdir -p outputs/urma/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/urma/ascend"
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 3 --timeout 1800 --max-time 1800 \
  --run ".venv/bin/python -m pytest tests/st/a5/tensormap_and_ringbuffer/urma_deferred_completion --platform a5 --device \$TASK_DEVICE --require-pto-isa --pto-session-timeout 1200 -v"
```

The run must report all six transfer tests and both lifecycle cases passed with
no skips. Check the build's ISA revision against `pto_isa.pin` and retain the
build log and pytest output, including each resource child result. Device logs
remain under `outputs/urma/`.

## Validation scope

The main PR CI workflow does not include an A5 onboard job. Hardware validation
of these transfers and lifecycle cases requires a separate A5 run using the
three-device command above. RDMA remains OFF by default.

These cases request `workspace_engines=("urma",)` and obtain a checked context
with `handle.device_context(rank, workspace_engines=("urma",))`. Kernels read
`comm_workspace<CommWorkspaceKind::URMA>(*ctx)`; the SDMA demo uses its independent
descriptor. RDMA defaults to OFF when its build environment variable is unset.
