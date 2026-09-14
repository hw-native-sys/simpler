# A5 AICPU Affinity Preflight — Design

The Ascend950 (a5) affinity preflight measures the user AICPU pool and writes
an authoritative five-thread placement, ordered as `[S0,S1,S2,S3,O]`. The
production runtime reads the small line-oriented companion file; it does not
run RTT measurement or re-rank the result.

Related hardware/OCCUPY background: [hardware.md](hardware.md).
Tool entry points: `python -m simpler_setup.tools.rtt_die_preflight` and
`simpler_setup/tools/aicpu_device_query/`.

## Runtime contract

`aicpu_thread_num` is the total active AICPU count: one orchestrator plus the
remaining scheduler threads. It is separate from the physical launch count,
which is always `popcount(OCCUPY)` so the affinity gate sees the complete user
pool.

- Total active count 5 means four schedulers plus one orchestrator. Both auto
  mode and an explicit value of 5 may consume a valid RTT plan.
- Explicit total counts 2–4 are honored exactly. The runtime warns that the
  four-scheduler RTT layout does not apply and selects that many CPUs in
  ascending OCCUPY-bit order.
- Auto mode normally resolves to 5, but shrinks to the available OCCUPY count
  when the pool is smaller. The shrink path also warns and uses contiguous
  OCCUPY-bit selection.
- An explicit active count greater than the OCCUPY population fails prepare.
- The supported physical launch population is 2–14. It is never clamped.

Schedulers own balanced contiguous AICore-cluster ranges using
`[t*N/A, (t+1)*N/A)`. For the measured four-scheduler shape, logical S0/S1 own
the first half (die0) and S2/S3 own the second half (die1).

## Layering

```text
ChipWorker.init (Python, a5 onboard)
  resolve exact per-device JSON + .cpus paths
  missing/invalid artifacts -> rtt_die_preflight --probe --out <JSON>
                         |
                         v
Preflight toolchain (selection authority)
  topology/serial enum -> handshake orch -> COND die scores -> pack
  success: publish one-device JSON and matching .cpus
  failure: publish nothing
                         |
                         v
DeviceRunner (thin production consumer)
  OCCUPY -> full launch count
  active==5 and valid side plan -> use [S0,S1,S2,S3,O]
  otherwise -> warn as applicable and use contiguous [S...,O]
                         |
                         v
Device affinity gate + balanced contiguous AICore ownership
```

`DeviceRunner` never invokes Python. `ChipWorker.init` reports probe
start/skip/done/fail, but a failed automatic preflight does not abort init:
the runtime subsequently uses contiguous OCCUPY allocation without writing a
fallback plan.

The automatic preflight gives the device RTT/COND probe a fixed 30-second
wall-clock budget after helper compilation finishes. Helper/dispatcher
builds use a separate longer budget and do not consume the probe timer. A
device-probe timeout writes a per-device `.timeout` record beside the plan;
later automatic initialization reads that record and immediately uses the
OCCUPY fallback without probing again. An explicit manual `--probe` clears
the record before retrying. A helper-build timeout fails the probe without
writing `.timeout`, so a later init may retry once artifacts can be built.

## Measurement and packing

Public CLI:

```bash
python -m simpler_setup.tools.rtt_die_preflight --device N --probe
```

The backend is `simpler_setup/tools/aicpu_device_query` in `--rtt-json` mode.
Wheel installs pass the installed simpler source root explicitly to both CMake
configure steps, so the backend does not depend on a repository-relative
directory depth.

The probe first reads topology and OCCUPY. When driver or verified JSON
topology exactly covers the OCCUPY CPU set, it enumerates that complete list
directly. Otherwise it runs the existing serialized one-thread discovery, but
accepts the result only if it exactly equals the OCCUPY set. Partial sampling
is a probe failure, not a smaller real pool. Pools larger than 14 are rejected.

| Phase | Work | Output |
| ----- | ---- | ------ |
| 1 | Complete user-pool enumeration | `user_pool_cpus` |
| 2 | Atomic-flag pairwise handshake, 1000 iterations | orchestrator with minimum average peer latency |
| 3 | Serialized non-orchestrator COND scoring, 100 samples/core | `die0_sum_ticks`, `die1_sum_ticks` |
| 4 | Physical picks in die order `{0,1,1,0}`, packed as `[P0,P3,P1,P2,O]` | `[S0,S1,S2,S3,O]` |

The full measured path requires at least five and at most 14 unique user CPUs.
Missing/incomplete enumeration, a pool smaller than five, failed RTT/COND,
malformed output, or an invalid plan all return nonzero and leave the artifact
paths untouched.

Offline operators may deliberately write either a fully specified manual plan
or an explicit contiguous five-thread plan:

```bash
python -m simpler_setup.tools.rtt_die_preflight --device 0 \
  --soc Ascend950PR_9599 \
  --allowed-cpus 3,4,5,6,7 \
  --occupy-cpus 3,4,5,6,7,8

python -m simpler_setup.tools.rtt_die_preflight --device 0 \
  --soc Ascend950PR_9599 --fallback-occupy 3,4,5,6,7,8
```

`--fallback-occupy` is an explicit administrative write. It is never invoked
automatically after a failed probe.

## Per-device artifacts and paths

The default files for device `N` are:

```text
build/config/aicpu_affinity_plan.N.json
build/config/aicpu_affinity_plan.N.cpus
```

Each JSON document contains exactly one device under the schema-v3
`socs.<soc>.devices.<device_id>` shape. Devices never read-modify-write a
shared document, so parallel multi-card initialization cannot lose another
device's update.

`SIMPLER_AICPU_AFFINITY_PLAN` may name a base path such as `plan.json` or a
template such as `plan.{device}.json`. A base becomes `plan.N.json`; a template
substitutes `N`. `--out` is already the exact per-device JSON path and is not
renamed. In every case the companion replaces the JSON filename extension
with `.cpus`. Python's producer, init hook, and C++ consumer use the same
derivation.

The JSON and companion are written through temporary files and atomic rename.
If either publication fails, the pair is rolled back to its previous state.
Probe/validation failure occurs before publication and does not create either
file.

The runtime companion has exactly seven keys:

```text
schema_version=3
device_id=0
soc=Ascend950PR_9599
source=atomic-flag-orch+cond-die-v1
occupy_mask=0x1f8
active_count=5
cpus=3,4,5,6,7
```

Before applying the file, C++ requires the exact schema version, device id,
SoC, supported source, current OCCUPY mask, active count 5, and exactly five
unique CPU ids in `[0,63]`, all present in OCCUPY. Missing, duplicate, or
unknown keys are rejected. The cache key includes path, device, SoC, OCCUPY,
size, and modification time, and only successful validations are cached.

## Runtime sequence

On `ChipWorker.init` for a5 onboard:

1. Resolve the per-device JSON and companion paths.
2. Skip only when both JSON and the hardware-independent companion structure
   are present and valid.
3. If a `.timeout` record is present, skip probing and use OCCUPY-contiguous
   placement.
4. Otherwise run `rtt_die_preflight --device N --probe --out <exact JSON>`.
5. Report non-timeout failure without synthesizing or writing a fallback.

On `DeviceRunner::prepare_execution`:

1. Query OCCUPY and reject populations outside `[2,14]`.
2. Resolve the exact requested/automatic active count; reject an explicit
   value larger than the pool.
3. For active count 5, validate and apply the per-device companion. On any
   miss or mismatch, warn and select the first five OCCUPY bits.
4. For active counts 2–4, warn and select exactly that many OCCUPY bits.
5. Launch `popcount(OCCUPY)` physical threads and let the device affinity gate
   retain the ordered active set.

Recovery, reset, and finalize clear both OCCUPY and affinity caches.

## Non-goals

- Do not move RTT/COND measurement into the production runtime.
- Do not let C++ spawn Python.
- Do not change a2a3 behavior.
- Do not remove `aicpu_topology_probe`; tooling and unit tests still use it.
- Do not use diagnostic FG/PG selection for production `ALLOWED_CPUS`.

## File map

| Role | Path |
| ---- | ---- |
| Ranking, validation, persistence | `simpler_setup/tools/rtt_die_preflight.py` |
| Device enum, handshake, COND | `simpler_setup/tools/aicpu_device_query/` |
| Python init hook | `python/simpler/task_interface.py` |
| Side-file reader and contiguous helper | `src/a5/platform/onboard/host/affinity_allowed_file.*` |
| Production consumer | `src/a5/platform/onboard/host/device_runner.cpp` |
| Affinity gate | `src/common/platform/onboard/aicpu/platform_aicpu_affinity.cpp` |
| Balanced cluster partition | `src/a5/platform/include/common/scheduler_cluster_partition.h` |
| Topology diagnostics/probe support | `src/a5/platform/onboard/host/aicpu_topology_probe.*` |
