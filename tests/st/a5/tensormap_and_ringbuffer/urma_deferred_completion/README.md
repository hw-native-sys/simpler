# URMA deferred completion validation

These A5 `tensormap_and_ringbuffer` regression tests run two ranks with a registered
communication window. Every transfer is followed by an add-one consumer that
depends on the transfer output. Both the copied bytes and consumer result are
checked against a host golden.

| Required test | Transfer | Submissions per producer |
| ------------- | -------- | ------------------------ |
| `TestUrmaGet` | TGET from peer input | 1 |
| `TestUrmaPut` | TPUT to peer receive window, then dependent TGET readback | 1 |
| `TestUrmaRepeatedFetch` | TGET from peer input | 16 |
| `TestUrmaRepeatedPut` | TPUT plus dependent readback | 16 |
| `TestUrmaChunkedFetch` | Static-shape TGET of 256 MiB + 32 KiB | 1 logical request, split by the adapter |
| `TestUrmaChunkedPut` | Dynamic-shape TPUT of 256 MiB + 32 KiB plus dependent readback | 1 logical request, split by the adapter |

The repeated cases check the number of registered completion entries against
the first request and the requested repeat count. Identical final tensor values
therefore cannot hide a loop that issues fewer requests.

The PUT readback is issued by the sending rank after its deferred PUT task
completes. It reads the destination region on the peer, so it does not depend
on an unsynchronized peer-side consumer. Producers are serialized per actual
queue; these cases do not claim multi-AIV shared-queue concurrency coverage.
The consumer processes full 8192-element tiles; the case builder rejects nonpositive
or unaligned extents. Large cases use approximately 512 MiB of registered window per rank plus local
output and consumer buffers. CPU regressions separately exercise counter wrap,
CQ owner phases, errors and capacity boundaries; the hardware cases do not
inject those conditions.

Chunked GET uses a fully static original tensor shape and strides; chunked PUT
uses runtime shape and strides. Both require the adapter's slices to describe
the smaller transfer with packed strides rather than retain the original extent.

## Build and run

Use a project-local venv and an A5 machine. The existing workspace overlay
options select the compiled backend; setting the test environment alone cannot
change an already-built runtime.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
source /usr/local/Ascend/cann/set_env.sh
export SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON
pip install --no-build-isolation -e '.[test]' --config-settings=build.targets=build_package_a5
python -m simpler_setup.tools.scene_test_compile \
  tests/st/a5/tensormap_and_ringbuffer/urma_deferred_completion \
  --platform a5 --require-pto-isa --compile-workers 4
mkdir -p outputs/urma/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/urma/ascend"
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --timeout 1800 --max-time 1800 \
  --run ".venv/bin/python -m pytest tests/st/a5/tensormap_and_ringbuffer/urma_deferred_completion --platform a5 --device \$TASK_DEVICE --require-pto-isa --pto-session-timeout 1200 -v"
```

The run must report all six tests passed with no skips. Check the build's ISA
revision against `pto_isa.pin` and retain the build log and pytest output, including each resource child
result. Device logs remain under `outputs/urma/`.

## CI integration dependency

These cases are validated locally on A5 hardware with URMA enabled. CI retains
its default SDMA build, where the URMA cases are skipped. A validation job can
enable `SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON` and run these cases alongside the
SDMA example and its mixed-workspace regression without a second build.

The host build can include SDMA and URMA together. These cases explicitly
request `workspace_engines=("urma",)` and read `comm_workspace<CommWorkspaceKind::URMA>(*ctx)`;
the SDMA demo uses its independent descriptor. The URMA build option remains
opt-in, and RDMA defaults to OFF when its build environment variable is unset.
