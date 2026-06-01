# st-onboard-a2a3 mass 507899/507018 — AICPU shared-SO device fault

## Symptom

`st-onboard-a2a3` fails intermittently (observed ~19% of runs after #870, vs
~10% before) with a whole-suite collapse: the `L2 tensormap_and_ringbuffer`
phase reports ~10 failed + ~23 errors at once. The surfaced Python errors are

```text
RuntimeError: simpler_init failed with code 507899
RuntimeError: prepare_callable failed with code -1
RuntimeError: run_prepared failed with code 507018 / 507046 / 507901
[ERROR] rtMalloc failed: 507899 (size=...)
[ERROR] Failed to allocate device GM for ChipCallable buffer
```

It looks like an out-of-memory. **It is not.** `npu-smi info` taken right after
the failure shows every chip `Health=OK` with HBM ~3 GB / 64 GB used.

## Root cause

`507899` decodes (in the CANN runtime slog) as `[driver error:internal error]`
and `507901` as `[hdc disconnect]`. They are a **cascade after a device fault**,
not the cause. The earliest real error is an **AICPU exception**:

```text
ProcLogicCqReport: Task run failed, sqe_type=1(aicpu), errType=0x1(task exception)
ProcessStarsAicpuErrorInfo: error from device(chipId:N, dieId:0/1),
                            an exception occurred during AICPU execution
PrintAicpuErrorInfo: Aicpu kernel execute failed,
                     soName=simpler_inner_<fp>.so, funcName=simpler_aicpu_exec,
                     errorCode=0x2a                       # 507018 ACL_ERROR_RT_AICPU_EXCEPTION
```

`simpler_aicpu_exec` faults the **whole chip**; afterwards every
`rtStreamCreate` / `rtMalloc` on that chip returns 507899/507901, so the next
test's `simpler_init` (which eagerly creates streams since #870) fails and the
rest of the suite collapses.

Two facts pin the mechanism:

1. On a2a3 the `npu-smi` Phy-IDs pair as **die0/die1 of one Ascend910**
   (devices 8/9, 4/5, …). The exception fires on **both dies of one chip at the
   same instant** — a chip-shared resource was corrupted.
2. The runtime stages SOs under the **shared preinstall directory**
   `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/`. The dispatcher (#870)
   wrote the AICPU runtime SO there under a **content-fingerprint-only** name
   `simpler_inner_<fp>.so` — identical across both dies. Paired dies share that
   filesystem, so both wrote/renamed/executed the same file; concurrent
   bootstrap corrupted the mmap'd image and trapped `simpler_aicpu_exec`.

A single-die 50× solo loop of the unregister/re-prepare/dedup tests **never**
reproduced; only the parallel multi-die suite did — consistent with a cross-die
shared-file race, not an intra-process use-after-free.

## How this was diagnosed (repeatable recipe)

The host-side simpler log shows only the cascade. To see the originating AICPU
exception you must surface the CANN device slog, which is otherwise hidden:

1. Run the job with the Ascend slog redirected to stdout (the test harness's
   `parallel_scheduler` captures each case's stdout into a per-case `::group::`):

   ```bash
   ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1 \
   python -m pytest examples tests/st --platform a2a3 --device <range> -v \
     --pto-session-timeout 600
   ```

   (`pto_runtime_c_api.cpp` skips its own `dlog_setlevel` when
   `ASCEND_GLOBAL_LOG_LEVEL` is set, so the env value wins.)

2. In the captured log, find the **first** `PrintAicpuErrorInfo` /
   `ProcessStarsAicpuErrorInfo` — note `soName`, `funcName`, `errorCode`, and the
   `chipId/dieId`. Everything after it that says `507899 [driver error...]` is
   noise.

3. `npu-smi info` confirms whether it is really OOM (it is not here).

4. Device-side AICPU dumps under `/var/log/npu/...` and `msnpureport` are
   **not readable by the CI runner user** (`Not have permission to export
   files`), so the stdout slog above is the only channel from CI. The exception
   *type* (0x2a) plus the faulting `soName/funcName` is enough to localize.

## Fix

Make the staged SO names **per-device** so paired dies never touch the same
file:

- Dispatcher inner SO: `simpler_inner_<fp>_<device_id>.so`
  (`src/common/aicpu_dispatcher/aicpu_dispatcher.cpp` `MakeInnerSoPath`,
  `src/common/host/load_aicpu_op.cpp` `MakeInnerSoBasename`). The real
  `device_id` (was hardcoded `0`) is threaded from `DeviceRunner` through
  `BootstrapDispatcher` into both the host JSON reader name and the device-side
  writer (`DeviceArgs.device_id`). Bootstrap cache keyed by `(fp, device_id)`.
- AICPU executor orchestration SO: `libdevice_orch_<pid>_<cid>_<device_id>.so`
  (`create_orch_so_file`). `device_id` reaches the executor via a new
  `KernelArgs.device_id` field → `set_orch_device_id()` in `kernel.cpp` →
  `get_orch_device_id()` in `aicpu_executor.cpp`. This one was already pid-named
  (mostly safe); the suffix is defense-in-depth for the case where two dies'
  device-side pids collide.

Applied symmetrically to a2a3 and a5.

## Confirming the fix

Rerun `st-onboard-a2a3` several times (it is flaky, so one green run is weak
evidence). The fix landed with 5/5 consecutive passes plus the root-cause
match. If the AICPU 0x2a fault ever returns, re-run the slog recipe above and
check whether any *other* SO staged under the shared `aicpu_kernels/0/`
directory still uses a non-per-device name.
