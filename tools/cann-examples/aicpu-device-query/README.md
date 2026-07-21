# aicpu-device-query

Runs `halGetDeviceInfo` queries from **inside an AICPU OS process** on the
device, using the same dispatcher bootstrap path as the production
`simpler` runtime. Resolves the queries that CANN's header flags as
"used in device" — `AICPU + OS_SCHED`, `AICPU + PF_*`, `CCPU/DCPU/TSCPU +
OCCUPY`, etc. — which always return failure from host code.

## What it answered

On a3 (`Ascend910_9392`), running `query_device_hal 4` returned:

```text
AICPU + OS_SCHED      rc=0  val=0x1     ← AICPU OS owns cpu_id 0
AICPU + OCCUPY        rc=0  val=0xfc    ← cpu_id 2..7 are the 6 user cores
AICPU + PF_OCCUPY     rc=0  val=0xfc    ← matches OCCUPY → no vNPU slicing
AICPU + PF_CORE_NUM   rc=0  val=0x6     ← PF view = 6, confirms no virtualization
CCPU  + OCCUPY        rc=0  val=0x1     ← CCPU has 1 core, occupied
DCPU/TSCPU queries fail (rc=3) — module-level access restricted device-side
```

Combined with the absence of vNPU mode, this closes the long-standing "is
the AICPU 8 → 6 gap OS-reservation or PG fab-disable?" question on a3:

- **cpu_id 0** = AICPU OS scheduler (`OS_SCHED` bit 0)
- **cpu_id 1** = PG fab-disabled (not in any module's OCCUPY mask, no
  virtualization can hide it)
- **cpu_id 2..7** = 6 user-schedulable AICPU cores

The full writeup is in
[`src/a2a3/docs/hardware.md`](../../../src/a2a3/docs/hardware.md#device-side-probe-resolves-the-aicpu-question).

On a5 (`Ascend950PR_9599`), running `query_device_hal 0` returned:

```text
AICPU + OS_SCHED      rc=0  val=0x1     ← AICPU OS owns cpu_id 0
AICPU + OCCUPY        rc=0  val=0x1f8   ← cpu_id 3..8 are the 6 user cores
                                          (DIFFERS from host-side 0x1fe)
AICPU + PF_OCCUPY     rc=0  val=0x1f8   ← matches OCCUPY → no vNPU slicing
AICPU + PF_CORE_NUM   rc=0  val=0x6     ← PF view = 6, confirms no virtualization
AICPU + CORE_NUM      rc=3              ← restricted device-side on a5 (worked on a3)
CCPU  + OCCUPY        rc=0  val=0x1     ← CCPU has 1 core, occupied
DCPU/TSCPU queries fail (rc=3) — module-level access restricted device-side
```

The 2-bit host-vs-device OCCUPY divergence (host `0x1fe`, device `0x1f8`)
exactly matches DSMI CPU_TOPO's single SMT pair at cpu_id 1, 2 — the
AICPU OS keeps the SMT pair for itself rather than exposing it to user
kernel dispatch. So the a5 9 → 6 gap is:

- **cpu_id 0** = AICPU OS scheduler (`OS_SCHED` bit 0)
- **cpu_id 1, 2** = SMT pair on phy_cpu_id 1, AICPU-OS-reserved (present
  in host-side OCCUPY so **not** PG fab-disabled — a5 does not have the
  a3-style fab-disable pattern)
- **cpu_id 3..8** = 6 user-schedulable AICPU cores

Full writeup: [`src/a5/docs/hardware.md`](../../../src/a5/docs/hardware.md#device-side-probe-resolves-the-aicpu-question).

## Architecture

Three pieces, exact same wiring as the production runtime's AICPU upload
chain — see [`src/common/aicpu_dispatcher/README.md`](../../../src/common/aicpu_dispatcher/README.md)
for the dispatcher's role and
[`docs/aicpu-kernel-launch-mechanisms.md`](../../../docs/aicpu-kernel-launch-mechanisms.md)
for why this method (Path A) and not tar.gz pre-deployment or the
broken Path B (`KERNEL_TYPE_AICPU_CUSTOM`, issue #822).

```text
+---------------------+        rtAicpuKernelLaunchExWithArgs (KFC, libaicpu_extend_kernels)
|   host launcher     | ---->  bootstrap: libaicpu_extend_kernels dlopens dispatcher SO,
|  query_device_hal   |        which writes our inner SO bytes to /usr/lib64/aicpu_kernels/0/...
+---------------------+
          |
          | rtsBinaryLoadFromFile (JSON), rtsFuncGetByName, rtsLaunchCpuKernel
          v
+---------------------+
|  libaicpu_query.so  |   inside AICPU OS process:
|  (inner SO)         |   for each (module, infoType) request -> halGetDeviceInfo
+---------------------+   -> writes QueryResult[] to GM
          |
          | D2H aclrtMemcpy
          v
+---------------------+
|   host launcher     | pretty-prints the results
+---------------------+
```

Reused unchanged: `build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so`
from the standard runtime build.

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 1. cross-compile device SO
cd device
mkdir -p build && cd build
cmake .. \
    -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++
cmake --build .

# 2. host launcher
cd ../../host
mkdir -p build && cd build
cmake ..
cmake --build .
```

## Run

The dispatcher SO comes from a normal `pip install .` runtime build —
build the runtime first if you have not.

```bash
export SIMPLER_DISPATCHER_SO=$REPO/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
export SIMPLER_AICPU_QUERY_SO=$REPO/tools/cann-examples/aicpu-device-query/device/build/libaicpu_query.so

# Always run hardware work via task-submit on this dev box (see
# .claude/rules/running-onboard.md).
task-submit --device auto --device-num 1 \
    --run "$REPO/tools/cann-examples/aicpu-device-query/host/build/query_device_hal \$TASK_DEVICE"
```

## Running on each arch

The same CMakeLists builds for a3 and a5 — only the dispatcher SO you
point at via `SIMPLER_DISPATCHER_SO` is per-arch:

- a3: `build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so`
- a5: `build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so`

`libascend_hal.so` lives at the same relative path
(`${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/...`) on
both arches. Both `Ascend910_9392` (a3) and `Ascend950PR_9599` (a5)
have been validated with this tool — see "What it answered" above.

## Scope and limits

- This is **not** a generic device-side HAL inspector. The hardcoded
  query list in `host/query_device_hal.cpp` reflects exactly what was
  needed to close the a3 AICPU question. Extending the list to other
  modules / infoTypes is a small edit — `requests` vector + the
  `kModuleName`/`kInfoName` switches.
- The inner SO uses local device id 0 (`self_did = 0`) — validated as
  the correct convention from inside an AICPU OS process. Passing the
  host's logical device id from inside the kernel returns rc=1 for all
  AICPU queries.
- "Used in device" PF queries that work device-side (`PF_OCCUPY`,
  `PF_CORE_NUM`) are useful for confirming SR-IOV/vNPU state — they
  must equal `OCCUPY` / `CORE_NUM` when not in a virtualized split.
