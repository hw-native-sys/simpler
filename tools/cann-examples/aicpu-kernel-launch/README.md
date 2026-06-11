# aicpu-kernel-launch

The smallest possible end-to-end demonstration of launching a custom AICPU
kernel from a host process using the production dispatcher bootstrap path —
**no sudo, no tar.gz pre-deployment, no scene-test infrastructure**. Two
files of device-side C++, ~370 lines of host-side C++.

If you want to add new AICPU work to this repo and aren't sure how the
plumbing fits together, read this tool first.

This tool implements **Method 2 (Path A — dispatcher Mode A bootstrap)** as
catalogued in
[`docs/aicpu-kernel-launch-mechanisms.md`](../../../docs/aicpu-kernel-launch-mechanisms.md).
Read that doc first if you want the full landscape: the older tar.gz
pre-deployment method, this Path A, and the broken Path B
(`KERNEL_TYPE_AICPU_CUSTOM`) along with the four user-space workarounds
that all fail on its cust-subprocess L1 cache coherency bug (issue #822).
The short version of why this tool sticks to Path A is in "Scope and
limits" below.

## What it demonstrates

The full Mode A "Path A" pipeline:

1. Host process maps the dispatcher SO and your inner SO into GM, hands
   them to CANN's `libaicpu_extend_kernels.so` via
   `rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC)`.
2. `libaicpu_extend_kernels` dlopens the dispatcher inside the AICPU OS
   process; the dispatcher writes your inner SO bytes to
   `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<dev>.so`
   (writable from the AICPU OS process, no host-side filesystem access
   needed).
3. Host fingerprints the inner SO by ELF Build-ID, generates a JSON
   descriptor pointing at the preinstall basename, registers via
   `rtsBinaryLoadFromFile`.
4. Host resolves `simpler_aicpu_init_<fp>` and `simpler_aicpu_run_<fp>` via
   `rtsFuncGetByName`.
5. Host launches `simpler_aicpu_run` via `rtsLaunchCpuKernel`; the kernel
   reads `DeviceArgs.input_token`, makes a `halGetDeviceInfo` call, writes a
   `HelloResult` to GM.
6. Host D2H copies `HelloResult`, verifies the magic, the echoed token, and
   prints the HAL value.

This is the same flow the production runtime
(`src/{a2a3,a5}/runtime/.../host/runtime_maker.cpp`) uses for every AICPU
runtime SO; everything specific to this repo's runtime (ringbuffer setup,
tensormap encoding, ChipWorker fork, etc.) is stripped out.

## When to use this vs the sibling tools

| Tool | Purpose |
| ---- | ------- |
| `tools/cann-examples/query` | Host-side ACL / HAL device-info CLI. No AICPU work. |
| `tools/cann-examples/aicpu-device-query` | Device-side HAL probe — uses the dispatcher path to run **specific** HAL queries inside an AICPU OS process. |
| **`tools/cann-examples/aicpu-kernel-launch`** | **The bootstrap path itself, with the smallest possible inner kernel. Use when you want to learn the launch pattern, not to answer a specific HAL question.** |

If your goal is to write a new device-side AICPU SO for this repo,
`aicpu-kernel-launch` is the template — copy the device side, replace the
kernel body with your work, and the host launcher will keep working with
zero changes as long as you preserve the two-export contract.

## Pipeline diagram

```text
+---------------------+        rtAicpuKernelLaunchExWithArgs (KFC, libaicpu_extend_kernels)
|   host launcher     | ---->  bootstrap: libaicpu_extend_kernels dlopens dispatcher SO,
|   launch_hello      |        which writes our inner SO bytes to
+---------------------+        /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/
          |
          | rtsBinaryLoadFromFile(JSON), rtsFuncGetByName, rtsLaunchCpuKernel
          v
+---------------------+
|  libhello_aicpu.so  |   inside AICPU OS process:
|  (inner SO)         |     DlogRecord("hello kernel running")
+---------------------+     halGetDeviceInfo(AICPU, OS_SCHED) -> 0x1
          |                 write HelloResult { magic, echoed_token, hal_rc, hal_value }
          | D2H aclrtMemcpy
          v
+---------------------+
|   host launcher     |  verify magic == 0xDEADBEEFC0FFEE01
+---------------------+  verify echoed_token == input_token (proves end-to-end)
```

The dispatcher SO is **reused unchanged** from the standard runtime build:
`build/lib/{a2a3,a5}/dispatcher/libsimpler_aicpu_dispatcher.so`. The
dispatcher exports three symbols
(`StaticTileFwkBackendKernelServer` +
`DynTileFwkBackendKernelServerInit` +
`DynTileFwkBackendKernelServer`) — see
`src/common/aicpu_dispatcher/README.md`.

## I/O contract

```c
struct DeviceArgs {                  // 160 B GM buffer
    uint64_t reserved_pre[12];       // 0..95 — used during bootstrap only
    uint64_t result_addr;            // 96    — &HelloResult in GM
    uint64_t input_token;            // 104   — host-supplied nonce
    // (112+ unused)
};

#pragma pack(push, 4)
struct HelloResult {                 // 32 B GM buffer
    uint64_t magic;                  // = 0xDEADBEEFC0FFEE01 on success
    uint64_t echoed_token;           // = DeviceArgs.input_token
    int32_t  hal_rc;                 // halGetDeviceInfo(AICPU, CORE_NUM) rc
    int32_t  _pad;
    int64_t  hal_value;              // hal value or 0 on error
};
#pragma pack(pop)
```

The first 96 B of `DeviceArgs` are dispatcher-owned during bootstrap (the
dispatcher reads `aicpu_so_bin`/`aicpu_so_len`/`device_id`/`inner_so_bin`/
`inner_so_len` at offsets 96/104/112/120/128). The host reuses the same
160 B buffer post-bootstrap by overwriting offsets 96 (`result_addr`) and
104 (`input_token`) before launching `simpler_aicpu_run`.

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 1. Cross-compile the device SO (aarch64 AICPU target).
cd device
mkdir -p build && cd build
cmake .. \
    -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++
cmake --build .
# -> libhello_aicpu.so

# 2. Native host launcher.
cd ../../host
mkdir -p build && cd build
cmake ..
cmake --build .
# -> launch_hello
```

## Run

The dispatcher SO comes from a normal runtime build:

```bash
pip install --no-build-isolation .   # builds build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so
```

```bash
# Required: tell launch_hello where the dispatcher and inner SO are.
export SIMPLER_DISPATCHER_SO=$REPO/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
export SIMPLER_HELLO_AICPU_SO=$REPO/tools/cann-examples/aicpu-kernel-launch/device/build/libhello_aicpu.so

# This dev box is shared. Hardware work goes through task-submit (see
# .claude/rules/running-onboard.md). Gate by arch first to avoid
# wasting a device lock on a wrong-arch attempt
# (.claude/skills/onboard-arch-precheck/).
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1

task-submit --device auto --device-num 1 \
    --run "$REPO/tools/cann-examples/aicpu-kernel-launch/host/build/launch_hello \$TASK_DEVICE"
```

Successful output:

```text
[bootstrap] inner SO landed at /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<dev>.so
[bootstrap] fp=<16 hex>  (token=<16 hex>)

=== device=<N>  hello_aicpu HelloResult ===
  magic         = 0xdeadbeefc0ffee01  OK
  echoed_token  = 0x????????????????  OK (expected 0x????????????????)
  hal AICPU+OS_SCHED  rc=0 val=0x1  OK
```

The HAL `AICPU + OS_SCHED` result is `0x1` on both a3 and a5: the AICPU
OS scheduler owns cpu_id 0 in the per-die AICPU layout (see
[`src/a2a3/docs/hardware.md`](../../../src/a2a3/docs/hardware.md#device-side-probe-resolves-the-aicpu-question)
and `src/a5/docs/hardware.md` for the device-side probe writeup that
established this).

Exit code: 0 on full success, 1 on bootstrap / load / launch failure, 2 on
mismatched magic or token (kernel ran but the I/O contract is broken).

## Adapting it

To launch your own AICPU work:

1. Copy `device/hello_aicpu.cpp` to a new file under your tool.
2. Keep `simpler_aicpu_init` as a no-op.
3. Rewrite `simpler_aicpu_run` — read `KernelArgs.device_args`, do your
   work, write your result(s).
4. Define your own `DeviceArgs` payload layout starting at offset 96.
5. Keep the device CMakeLists.txt identical except for `OUTPUT_NAME`.
6. Adapt `host/launch_hello.cpp`'s "Rewrite DeviceArgs for the run() phase"
   block to populate your payload, and the "D2H + verify" block to read it
   back.

The bootstrap, JSON descriptor, `rtsBinaryLoadFromFile`,
`rtsFuncGetByName`, and `rtsLaunchCpuKernel` calls do not change.

## Scope and limits

- **Single block_dim**: the example launches with `blockDim=1`. AICPU
  multi-thread fanout is a separate axis that this tool deliberately
  ignores — it's part of the inner kernel's design, not the launch path.
- **One stream, one device**: no multi-stream, no multi-device. Both can
  be added by allocating GM / streams per device and looping; the
  bootstrap is per-(device, dispatcher-fp) and would land at the same
  preinstall basename, so per-device launches are mutually compatible.
- **Path A only**. The two alternatives — tar.gz pre-deployment and
  Path B (`KERNEL_TYPE_AICPU_CUSTOM`) — both have hard downsides
  (sudo + fleet provisioning for tar.gz; silent 507018 deadlock for
  Path B due to a cust-subprocess L1 cache coherency bug). The full
  comparison + Path B's failure modes + the four user-space workarounds
  that all fail live in
  [`docs/aicpu-kernel-launch-mechanisms.md`](../../../docs/aicpu-kernel-launch-mechanisms.md)
  and [issue #822](https://github.com/hw-native-sys/simpler/issues/822).
- **Path A latch — one inner SO per host process**. CANN's
  `libaicpu_processer.so::BackendServerHandleManager` has a process-wide
  `firstCreatSo_` one-shot latch; the second runtime's inner SO load
  silently no-ops. This forces this repo's
  [`ChipWorker`](../../../src/common/worker/) model (one process binds
  one (arch, runtime) pair). Lifting the latch was the entire reason PR
  #537 attempted Path B — see the mechanisms doc above.
- **Inner SO must export `simpler_aicpu_init` + `simpler_aicpu_run`**.
  The dispatcher requires the init symbol to resolve even though it's a
  no-op; the run symbol's name is reused as the fingerprint-suffixed
  opType. Renaming either breaks the JSON descriptor.
- **Build-ID gives stable fingerprints across builds with identical
  inner code**; FNV-1a fallback (when `--build-id` is missing) does too,
  so the preinstall basename collides for repeat runs. The dispatcher's
  upload logic skips writing when the basename already exists, which is
  the desired behavior — repeat launches reuse the prior upload.
