# PMU Profiling (a5)

This document describes how to use a5 PMU profiling, what output it produces,
and the current usage limitations.

## Overview

PMU profiling collects per-task AICore hardware counter data and exports a
LuoPan-compatible CSV file on the host side. It targets DAV_3510 hardware
(10 counters + dual PMU CTRL register).

Use `a5` hardware runs for meaningful PMU data. The simulation path can
exercise the PMU export flow, but does not provide real hardware counters.

## Design

### Layered Responsibilities

- **Host** owns user entry, event-type selection, PMU session setup, and
  final CSV export
- **AICPU** owns PMU init/finalize (event selectors, `PMU_CTRL_0/1`
  start, CTRL restore), publishes per-core `PmuBuffer` and PMU MMIO base
  into each `Handshake`, and on each task FIN copies AICore's
  `dual_issue_slots[]` snapshot into `PmuBuffer::records[]` while
  filling `func_id` / `core_type`. AICPU also stamps each core's
  `PmuBufferState::owning_thread_id` (the AICPU scheduler thread that
  drives that core) so host can emit a per-record `thread_id` column.
- **AICore** gates the counting window around the kernel body via CTRL
  SPR bit 0, reads the 10 PMU counters + `PMU_CNT_TOTAL` via the
  `ld_dev` MMIO load intrinsic after each task, and writes the snapshot
  into `PmuBuffer::dual_issue_slots[reg_task_id & 1]` (the 32-bit
  `DATA_MAIN_BASE` register token, not the 64-bit logical task id)

This AICore/AICPU split mirrors the a2a3 perf collector's producer/
consumer protocol (counter and timing writes on the core that ran the
task; metadata and commit on the controller). The two slots exist
because AICore's dual-issue dispatch can have up to two tasks in flight
per core — parity `reg_task_id & 1` keeps adjacent dispatches from
colliding (the runtime's `dispatch_seq++` guarantees neighboring
register tokens differ by 1, so they always land on different slots).

### Slot Match Key vs Logical task_id

`pmu_aicpu_complete_record` takes both a 32-bit `reg_task_id` (used to
select and validate the dual-issue slot — must equal the value AICore
read from `DATA_MAIN_BASE` and stored in `slot->task_id`) and a 64-bit
logical `task_id` written into the record itself. In runtimes whose
logical id encodes more than 32 bits (e.g. PTO2's
`(ring_id<<32)|local_id` in `tensormap_and_ringbuffer`), the two
values differ — slot match must use the register token, otherwise the
slot will never validate and every commit silently drops.

### Device Memory Layout

```text
[ PmuSetupHeader ]              ← num_cores, event_type, buffer_ptrs[N]
[ PmuBufferState[num_cores] ]   ← owning_thread_id, dropped_record_count,
                                   total_record_count (per core)
```

This single shared region (`calc_pmu_setup_size`) is allocated once at
init and pulled back via one `rtMemcpy` at finalize. The per-core
`PmuBuffer`s themselves are separate device allocations (one per core),
copied back individually on demand because their `count`-sized payloads
vary.

### Device → Host Transfer

halHostRegister is not supported on DAV_3510, so the PMU collector uses
the same two-step rtMemcpy pattern already used by the performance and
tensor-dump collectors:

1. At init: host allocates the `[PmuSetupHeader][PmuBufferState[]]`
   region plus one `PmuBuffer` per core. The setup region's device
   address is published into `kernel_args.pmu_data_base`. AICPU then
   publishes each core's `PmuBuffer` address and PMU MMIO base into
   the matching `Handshake` (`pmu_buffer_addr`, `pmu_reg_base`) so
   AICore can do its own MMIO read — parallels `perf_records_addr`.
2. During execution: AICore, after each kernel completes, reads the
   10 PMU counters via `ld_dev(base, offset)` and writes them into
   `PmuBuffer::dual_issue_slots[reg_task_id & 1]`. AICPU, on observing
   COND FIN, validates that slot's `task_id` against the register
   token (`pmu_aicpu_complete_record`), copies register state into
   `PmuBuffer::records[count]`, fills `func_id` / `core_type`, stamps
   `PmuBufferState::owning_thread_id`, and advances `count`.
3. After stream sync: host pulls back the entire setup region (all
   `PmuBufferState`s in one shot) plus each core's `PmuBuffer` payload
   (header first to learn `count`, then `count * sizeof(PmuRecord)`
   records). The host writes a CSV under `outputs/` and emits a
   cross-check log line:

   ```text
   PMU collector: record counts match (collected=N, dropped=K, device_total=N+K)
   ```

   If `collected + dropped != device_total`, the difference is silent
   slot-mismatch loss (AICore had not yet published the slot when
   AICPU tried to commit) and the line becomes a `record count
   mismatch (... diff=M silent slot-mismatch losses)` warning.

This design fits naturally into the existing a5 `PerformanceCollector` /
`TensorDumpCollector` lifecycle — `initialize / collect_all / export /
finalize` is called from `DeviceRunner::run`.

## Usage

### SceneTest CLI

Enable PMU with the default event group:

```bash
python tests/st/<case>/test_<name>.py -p a5 -d 0 --enable-pmu
```

or with pytest:

```bash
pytest tests/st/<case> --platform a5 -d 0 --enable-pmu
```

The bare flag is equivalent to:

```bash
--enable-pmu 2
```

which selects `PIPE_UTILIZATION`.

Pass an explicit event type to collect a different counter group:

```bash
python tests/st/<case>/test_<name>.py -p a5 -d 0 --enable-pmu 4
```

`--rounds > 1` disables PMU collection in the test harness.

### Event Types

| Value | Event Type | Example Counters |
| ----- | ---------- | ---------------- |
| `1` | `ARITHMETIC_UTILIZATION` | cube/vector execution counters |
| `2` | `PIPE_UTILIZATION` | vector, cube, scalar, MTE busy cycles |
| `4` | `MEMORY` | UB/L1/L2/main memory requests |
| `5` | `MEMORY_L0` | L0A/L0B/L0C requests |
| `6` | `RESOURCE_CONFLICT` | bank and vector resource stalls |
| `7` | `MEMORY_UB` | UB and memory bandwidth counters |
| `8` | `L2_CACHE` | L2 cache hit/miss allocation counters |

Invalid nonzero values fall back to `PIPE_UTILIZATION`. The
`SIMPLER_PMU_EVENT_TYPE` environment variable overrides the CLI value.

DAV_3510 has its own per-counter event-code space (see pypto
`pmu_common.cpp::SetPmuEventTypeDAV3510`); the CSV counter column set
depends on which event group is selected. a5 has 10 hardware counter
slots; unused slots in a given event group are zero.

## Output

The PMU artifact is a CSV file under `outputs/`:

```text
outputs/pmu_<YYYYMMDD>_<HHMMSS>_<mmm>.csv
```

Columns (in order) — matches a2a3 host PMU CSV for tooling parity:

| Column | Meaning |
| ------ | ------- |
| `thread_id` | AICPU scheduler thread that drives this core (read from `PmuBufferState::owning_thread_id`) |
| `core_id` | Logical AICore id in the runtime |
| `task_id` | Runtime task id, printed as hex |
| `func_id` | Kernel function id |
| `core_type` | `0` = AIC, `1` = AIV |
| `pmu_total_cycles` | 64-bit PMU total cycle counter snapshot |
| event-specific counters | Counter columns selected by the event type |
| `event_type` | Numeric event type used for the run |

For the default `PIPE_UTILIZATION` event type (`2`), the counter columns
on a5 are (from pypto `tilefwk_pmu_to_csv.py` table_pmu_header_3510):

```text
pmu_idc_aic_vec_busy_o,cube_instr_busy,scalar_instr_busy,mte1_instr_busy,
mte2_instr_busy,mte3_instr_busy,icache_req,icache_miss,pmu_fix_instr_busy
```

The number of counter columns varies by event type — each DAV_3510 event
group populates a different subset of the 10 hardware counter slots and
the CSV lists only the slots that have a defined name in pypto's
`table_pmu_header_3510`. Consumers should use the `event_type` column to
discover which columns are present.

## Limitations

- `a5sim` can validate the PMU export path, but the simulated counter
  register block does not model AICore execution, so counter values are
  always 0. The AICPU still programs the PMU event selectors and the CSV
  still carries one row per task with a zero counter tuple — useful for
  verifying the end-to-end data flow but not for performance analysis.
- The per-core on-device `PmuBuffer` is a fixed size
  (`PLATFORM_PMU_RECORDS_PER_BUFFER = 4096`). Tasks past that count are
  dropped and accounted in `PmuBufferState::dropped_record_count`; the
  host surfaces the total in the finalize log line. Increase the
  constant in [platform_config.h](../platform/include/common/platform_config.h)
  if your workload executes more tasks per core.
- A non-zero `diff` in the host's `record count mismatch` warning means
  AICPU attempted to commit `diff` records whose dual-issue slot still
  carried an older `task_id`. Under the current design on DAV_3510
  hardware **this should always be zero** — AICore's slot-write order
  is `counters → pmu_total_cycles → store barrier → task_id → dcci →
  dsb → write FIN to COND`, so by the time AICPU observes FIN and
  invalidates the slot, the slot write is guaranteed visible.

  A persistent non-zero `diff` therefore indicates a regression in
  one of four places, not a tolerable tail loss:

  1. The `reg_task_id` producer/consumer drifted out of sync (AICPU
     uses a different task-id encoding than AICore wrote into the
     slot — this was the bug behind `collected=4 / diff=32` during
     the AICore-side PMU rewrite).
  2. AICPU calls `pmu_aicpu_complete_record` for a task AICore never
     executed (e.g. a new code path that commits PMU for an
     AICPU-only task; AICore never wrote that slot, so `task_id`
     stays stale).
  3. AICore's `dcci` / `dsb` ordering around the slot write was
     rearranged, or a barrier was weakened from the required full
     `dsb` to a store-only flavor.
  4. The hardware target's `dcci(..., CACHELINE_OUT)` semantics differ
     (e.g. non-DAV_3510 ports) and no longer guarantee HBM
     writeback before the following `dsb`.

  Treat `diff != 0` as a sharp diagnostic — find the regression, do
  not tune `PLATFORM_PMU_RECORDS_PER_BUFFER` around it.

Keep PMU comparisons consistent across runs by using the same runtime
build flags.
