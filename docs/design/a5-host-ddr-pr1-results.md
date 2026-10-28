# A5 Host DDR PR1: local synchronous reads

PR1 implements one in-process L2 Worker holding a table in its own Host DDR,
read on demand by one A5 on the same machine into Device GM. The gather kernel
finishes its mapped MTE transfers before returning; a dependent consumer then
uses the result. This is ACL mapping plus synchronous MTE, not URMA asynchronous
completion. `submit()` may still return before the run finishes on the host.

The accepted scope replaces the original single-process URMA requirement.
Cross-rank Host URMA, L3-to-L2/L4-to-L3 resource protocols and remote-machine
access belong to later PRs. A PTO receive-region extension, distributed lifetime
protocol, and evidence of detached issue/transfer overlap are not PR1 prerequisites.
The local path supports both URMA=OFF and URMA=ON builds without initializing a
communicator. Existing Device URMA remains a regression baseline.

## Interface and lifecycle

```python
ordinary = l2.create_buffer(nbytes)                        # existing staging
ordinary = l2.create_buffer(nbytes, device_readable=False) # unchanged behavior
table = l2.create_buffer(nbytes, device_readable=True)     # local mapped Host
```

True is a keyword-only bool. Creation checks the actual loaded runtime's Host
map/unmap exports before allocating backing. An environment variable cannot
supply missing binary support. Direct L2 on A5 onboard is supported; simulators,
older runtimes without these exports, and L3+ creation reject this option.
There is no user attach/register API or external Host-allocation import.

The backing stays HOST/POSIX_SHM with READ permission. BufferDescriptor uses
former padding byte 5 for a fixed policy, retaining its 88-byte size. Tensor
views inherit the policy; range, offset, ownership and process/context checks
apply. TaskArgs accepts INPUT/NONE only for these tables. The callable direction
must independently be explicit IN. Ordinary arguments keep their existing rules;
Device buffers still use `l2.malloc`.

The owner calls `aclrtHostRegisterV2(ACL_HOST_REG_MAPPED)` and
`aclrtHostGetDevicePointer` once per Buffer. Materialization resolves a checked
view offset against that returned Device address. It neither sends a raw CPU
pointer to a kernel nor creates a table-sized Device copy. The kernel selects
rows; a kernel that ignores the table triggers no automatic read or communication
wait. Both TMR and HBG reuse this path and have separate hardware acceptance.

Submission reserves the backing before native entry. A single run-id entry owns
the complete set of Host tables; interruption during selection or after publication
can be rolled back using the same run id. Successful wait releases that reservation;
preparation refusal rolls it back. `release_buffer` and direct
`Buffer.close` reject outstanding uses, including a completed run whose cleanup
has not yet been awaited. Direct `shm.close`/`unlink` also refuse registered
backing, and reject a foreign PID before touching an inherited lock.
The examples always call `Worker.close`, including on initialization failure;
if close also fails, the original exception remains primary with the cleanup
error chained as its cause. `Worker.close` drains successful unawaited runs before unmapping; native
finalize refuses remaining mappings. Mapping or unregister failures keep partial
ownership for cleanup retry. Unproven execution failures retain backing, mapping
and context through a process-lifetime source registry and block further use;
timeout alone does not establish safety. Automatic device recovery is excluded.

Finish CPU writes before submitting readers, and wait for all readers before
updating or releasing the table. Arbitrary writes through exported Python/numpy
views cannot be intercepted; release those views before closing the backing.

## Examples and acceptance

The [user example](../../examples/a5/tensormap_and_ringbuffer/host_ddr_gather/README.md)
provides BF16 gather followed by a dependent FP32-plus-one consumer. It exercises
TMR and HBG separately. `run_mx.py` combines a Host E4M3FN payload with existing
Device E8M0 scales and the existing TMATMUL_MX kernel; it adds no model or
quantization operator.

The BF16 case checks first/last/repeated IDs, invalid IDs, zero requests, capacity
overflow, offset views, CPU update after wait, and release rejection before wait.
It creates a second Host table after the first run, increasing Host backing from
about 2 MiB to 8 MiB while keeping Device payload allocations fixed. Repeated runs
reuse each mapping. Invalid IDs leave their destination rows unchanged with
status 2; excess requests transfer nothing with status 1; success/zero requests
return status 0. These statuses and row alignment limits belong to the example.

The acceptance below uses the local-only source on the publication rebase.
The runtime and bindings were rebuilt before repeating these checks. Earlier
mixed-source and implementation-start runs remain historical evidence only.
Host cleanup uses the mainline accepted RunHandle drain; regression tests cover
unawaited successful readers and an already-reported failure with unused backing.

| Local-only check | Result |
| ---------------- | ------ |
| Python policy, capability, Buffer, ChipWorker and Worker suites | 1,621 passed, 15 skipped |
| C++ Buffer/descriptor/wire cases | 24 passed |
| a5sim full scene corpus below L4 | 105 passed, 3 skipped |
| A5 URMA=OFF full scene corpus below L4 | 164 passed, 17 skipped, 2 HBG shutdown-timeout failures; see below |
| A5 URMA=OFF local TMR BF16, HBG BF16 and MXFP8 consumer | 3 passed; one device per case |
| A5 URMA=ON existing Device URMA and local Host cases | 1 Device case + 3 Host cases passed, zero skips |
| Actual loaded-runtime mapping capability under OFF/ON and forged environment values | Both runtimes passed under both builds (4 hardware cases) |

The publication-base full sweep is not a clean pass: `benchmark_bgemm` and
`batch_paged_attention` reported scheduler error 100 with device
`site=102` (`AICPU_SHUTDOWN_BARRIER_TIMEOUT`). The PR does not change their
kernels or the HBG resident scheduler. Both cases passed separately in fresh processes on device 2 (one pass each).
These reruns do not erase the initial failures. No claim of root cause is made.

Hardware task records:

- OFF local Host, capability checks and full regression: `task_20261026_101914_343861612274` (exit 1 for the two corpus failures).
- Isolated HBG reruns: `task_20261026_103757_27286112397` (both passed).
- ON Device URMA, local Host and capability checks: `task_20261026_103913_49663718723` (exit 0).

The remote SHA256 check covers all 22 changed code files; all match the local
source. Review and execution evidence is under `outputs/host-ddr-pr1/`:
`changes.patch`, `source-SHA256SUMS`, `pr-code-SHA256SUMS`,
`pr-publication-evidence.tar.gz` and `delivery.json`. No pending cross-rank work is
counted against this PR1 acceptance.

### Review repair validation

On review base `daf2f8418907592b5b881792bd9a2a0a5ea8f19d`, the three reported
lifetime defects first reproduced as 8 failing CPU/fork cases. After repair:

- Related Python suites: 1,635 passed, 15 skipped. A final focused run, including
  an additional overlapping-reader case, passed all 31 Host mapping/cleanup cases.
- `mkdocs build --strict` and all pre-commit hooks passed; the page is in navigation.
- Rebuilt A5 URMA=OFF: TMR BF16, HBG BF16 and MXFP8 all passed in
  `task_20261026_163704_281110125120` (device 0).
- Rebuilt A5 URMA=ON: existing Device URMA plus all three Host cases passed in
  `task_20261026_163742_28434368970` (devices 1,2; each Host case uses one device).

Both hardware tasks exited zero and released their locks. Source fingerprints and
logs are in `outputs/host-ddr-pr1/review-evidence.tar.gz`; CPU failure/success and
strict docs logs share that output directory. The full-corpus results above belong
to the original publication base, not this focused repair validation.

### Forwarding and concurrent release validation

On upstream base `8f592545b969a7433f0a7cd27f2a7c6a4728c4d8`, `re_export()`
rejects device-readable Host descriptors instead of silently reverting their
policy to ordinary Host staging. Direct and wire-decoded descriptors both
reproduced the defect before the guard. The owning L2 can still use the table;
another L2 cannot materialize it. Ordinary forwarding remains supported.

Two threaded tests force acquire and close to contend for the actual mapping
lock in both orders. A published reader prevents unmapping; a completed close
prevents a waiting reader from reserving the released mapping. No sleep-based
ordering is used.

Rebuilt local bindings: 1,640 related Python tests passed, 15 skipped;
the focused Host mapping suite passed all 29 cases. Strict MkDocs and pre-commit
checks passed. On A5DT, a fresh URMA=OFF build passed all 5 hardware cases:
TMR/HBG BF16, MXFP8 and both runtimes' mapping capabilities. Queue task
`task_20261027_154137_4701703419` used device 0 and exited zero. All 25 source/pin
fingerprints matched the local tested tree. URMA=ON and the full hardware corpus
were not rerun for this forwarding repair; their earlier results remain historical.

The queued command, after architecture precheck, was:

```bash
task-submit --device auto --device-num 1 --timeout 600 --max-time 600 --run \
  "$PWD/.venv/bin/python -m pytest tests/ut/py/test_host_ddr_onboard.py tests/ut/py/test_host_ddr_capability.py -m requires_hardware --platform a5 --device \$TASK_DEVICE -q -s"
```

## Versions and reproduction

- Implementation-start upstream/main: `2522f02bfead6071a70603d0ccb35431ca03139b`.
- Publication-rebase upstream/main: `a2e5ed00a7c34a48461299ac2f95c1c535700cd7`.
- Forwarding-repair upstream/main: `8f592545b969a7433f0a7cd27f2a7c6a4728c4d8`.
- Current PTO pin: `dcebd22bec597256e264b1bff93226823d782962`, inherited from main.
  Original publication/review validation used `b1d460ee8c001585ede36b68ba3678dc5a226bbf`.
- Independent implementation branch: `feat/a5-host-ddr-pr1`.
- A5DT: Ascend950DT_9581, aarch64, one device per local Host case.
- CANN toolkit 9.3.0 / V100R001C12B126; driver 25.6.rc2.b023 /
  V100R001C10SPC100B202. OPP/compiler manifests report 9.2.0.
- Remote Python 3.11.6, pytest 9.1.1; all Python operations use a project-local venv.
- The local A3 machine is used only for CPU and a5sim checks.

Source the installed A5 CANN environment before remote builds. Pre-create a
per-run device log directory, run architecture precheck, then acquire devices
through the queue. The four-device corpus validates existing multi-card cases;
each Host DDR example itself uses exactly one device.

```bash
CCACHE_DIR="$PWD/build/ccache" .venv/bin/python -m pip install --no-build-isolation -e . \
  --config-settings=build.targets=build_package_sim
.venv/bin/python -m pytest tests/ut/py/test_chip_worker.py \
  tests/ut/py/test_host_ddr_mapping.py tests/ut/py/test_host_ddr_capability.py \
  tests/ut/py/test_buffer.py tests/ut/py/test_worker -q
.venv/bin/python -m pytest examples tests/st --platform a5sim --exclude-level 4 \
  --device 0-7 --pto-session-timeout 1200 -q
SIMPLER_ENABLE_PTO_URMA_WORKSPACE=OFF .venv/bin/python -m pip install \
  --no-build-isolation -e . --config-settings=build.targets=build_package_a5
mkdir -p outputs/local-only-off-ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/local-only-off-ascend"
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device 4 --timeout 600 --max-time 600 --run \
  "$PWD/.venv/bin/python -m pytest tests/ut/py/test_host_ddr_onboard.py --platform a5 --device 4 -q -s"
task-submit --device 4,5,6,7 --timeout 1800 --max-time 1800 --run \
  "$PWD/.venv/bin/python -m pytest examples tests/st --platform a5 --exclude-level 4 --device 4-7 --pto-session-timeout 1200 -q"
SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON .venv/bin/python -m pip install \
  --no-build-isolation -e . --config-settings=build.targets=build_package_a5
mkdir -p outputs/local-only-on-ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/local-only-on-ascend"
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device 4,5 --timeout 600 --max-time 600 \
  --env SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON --run \
  "$PWD/.venv/bin/python -m pytest examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo tests/ut/py/test_host_ddr_onboard.py --platform a5 --device 4-5 --manual include -q -s"
```

Each runtime's local example runs in its own process. An earlier mixed-runtime
serial sweep crashed in later native L3 initialization; the same case passed
alone and the CI-shaped process-isolated sweep passed. The native cause remains
unproven. Use the corpus command above without `--runtime`, which disables that
isolation in this repository.

## Deferred work

Cross-rank Host access, PTO independent receive registrations, TMR URMA completion
adapter changes and communication-domain teardown repairs are preserved separately
on `wip/a5-host-ddr-async-followup-20261025`. They are not included in PR1. The
[local-path investigation](../investigations/2026-10-a5-host-ddr-single-rank.md)
records why mapped MTE is the qualified local implementation; unsuccessful
loopback probes do not prove that all possible local URMA configurations fail.

Future cross-rank work must reuse existing communication and upper-level reader
completion dependencies, without adding automatic distributed reference counting.
Its registration, actual asynchronous completion, failure/lifetime and DFX overlap
acceptance remain separate from this local PR.
