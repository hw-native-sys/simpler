# A5 TMR Device URMA

The URMA overlay is opt-in. The default A5 build uses SDMA. Build and test
with the same explicit overlay settings; setting only the test environment
cannot change an already-built runtime.

## Supported contract

The adapter targets the revision in `pto_isa.pin` (currently
`b1d460ee8c001585ede36b68ba3678dc5a226bbf`) and the existing Device window
path in `tensormap_and_ringbuffer`:

- `PER_PEER`, one row and one jetty per peer, jetty base zero. There must be
  one serialized AIV producer per peer queue and one scheduler CQ consumer.
- Contiguous one-dimensional, equal-sized, equal-dtype TGET/TPUT requests.
  Requests larger than PTO's 256 MiB WQE limit are split into WQEs.
- Dense-prefix domain membership with at least two ranks. The mandatory
  hardware suite validates two ranks and TGET followed by a dependent consumer.
  Derived base-window contexts require the full dense rank set and zero offset.
  Allocation IDs cannot be reused while their resources remain retained.
- 64-byte CQEs/WQEs and power-of-two rings. BB and CQE progress are distinct
  32-bit sequence numbers; wraparound and owner phases are handled separately.

Shared-pool, multi-row, multi-jetty and nonzero jetty-base configurations are
rejected. Notify operations are not exposed by this adapter. A full ring or
completion slab is an error, not a synchronous fallback. The caller must
serialize producers for each peer; the adapter does not arbitrate concurrent
AIV producers. No Host DDR registration or HBG capability is added.

`UrmaInfo`, WQ and CQ layout checks compare size, alignment and every field
offset with the original PTO declarations on Host and AICore. The pinned
`urma_types.hpp` is directly usable by the Host compiler; no generated header
view or manager copy is needed.

The submitted event preserves both BB and CQE targets. A zero handle can be a
valid wrapped event. The scheduler acquires the complete successful CQE prefix,
publishes independent tails and the CQ doorbell, then releases the producer's
output dependency. Pending/error prefixes do not advance either tail. AIV
submission preflights ring space so PTO's internal full-ring blocking drain is
not reached. There is no `AsyncEvent.Wait()` fallback.

## Resource ownership

Each domain owns its existing PTO manager and a public HCCL subcommunicator.
PTO uses fixed memory-registration tags; isolating domains in child
communicators avoids replacing another domain's registrations. Domain release
is logical: manager, child communicator, context and backing windows move to
the base communicator's retained collection. Other live domains remain usable.
The collection is reclaimed at successful base communicator destruction.

Destruction closes admission, performs the peer handshake, synchronizes the
device, verifies all published workspaces' SQ head/tail and CQ submitted/tail
counters, destroys child communicators and the base HCCL communicator, then
finalizes managers and frees contexts, imports and local windows. A failed
manager Init remains owned even if no usable workspace was returned: its
registrations may already belong to HCCL. It follows the same destroy order.

If stop, drain, handshake or communication-resource destruction cannot be
proved successful, destruction returns an error and quarantines the entire
owner. It must not be reused. This failure path deliberately retains C++ owners
for the process lifetime; device reset/process termination is the device-resource
reclamation boundary. It is not an in-process retry/recovery mechanism. Normal
domain releases retain memory until base communicator destruction, so long-lived
communicators must budget for all domains created during their lifetime.

## Build and verify

Use an A5 machine and a project-local venv. The SDK must provide the HCCL
channel APIs used by the pinned PTO manager and `HcclCreateSubCommConfig`.
Hardware validation before the mainline rebase used CANN 9.3.0 and PTO
`c0d7148e95ef73bd12a73165fdce4b723a3b7e72`. It must be repeated for the
current pin; the local CANN 9.0.0 SDK lacks
`HcclChannelConfig*` declarations needed to compile this pin.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
source /usr/local/Ascend/cann/set_env.sh
export SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON
export SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=OFF
pip install scikit-build-core 'nanobind<3' 'cmake>=3.20'
pip install --no-build-isolation -e '.[test]' --config-settings=build.targets=build_package_a5
python -m simpler_setup.tools.scene_test_compile \
  examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo \
  --platform a5 --require-pto-isa --compile-workers 4 -v
.claude/skills/onboard-arch-precheck/check.sh a5
mkdir -p outputs/urma/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/urma/ascend"
task-submit --list
task-submit --timeout 1800 --max-time 1800 --device auto --device-num 2 \
  --run ".venv/bin/python -m simpler_setup.tools.urma_verify --device \$TASK_DEVICE --report outputs/urma/verification.json"
```

`urma_verify` checks the loaded runtime's backend and build pin, records binary
hashes, and requires each named pytest item to execute and pass. Missing
capability, a skipped target, zero collection/execution, xfail or teardown
failure fails the job. The four items cover peer read plus consumer, repeated
posts, multiple domains with release/recreation across completed runs, and
explicit native communication destruction before device reset.

PR CI and Daily share `_st-urma-a5.yml`. The PR job follows the canonical
`a5_changed && st_affected` gate. `_st-npu-a5.yml` explicitly uses URMA=OFF and
RDMA=OFF for default-path regression. CPU contract/fault tests are separate
from hardware acceptance; simulation skips are not URMA hardware evidence.
