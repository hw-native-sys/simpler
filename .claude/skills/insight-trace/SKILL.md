---
name: insight-trace
description: Generate a MindStudio Insight trace for any `kernel_entry(args)` style kernel in this repo — SPMD mix, AIC-only single-task (e.g. `aic_pv_matmul`), or AIV-only single-task (e.g. `aiv_softmax_prepare`). Use when the user asks to "produce/generate/run an Insight trace", "trace this kernel under msprof op simulator", or troubleshoot Insight trace collection. AICore-only replay path — bypasses AICPU orchestration. For PTOAS-style kernels, use [PTOAS msprof_op_simulator_usage_zh.md](https://github.com/hw-native-sys/PTOAS/blob/main/.claude/skill/msprof_op_sim_insight_skill.md) instead.
---

# Insight Trace for `kernel_entry(args)` Kernels

Builds a single mix-arch `.so` wrapper around a target `kernel_entry(args)`,
launches it under `msprof op simulator`, and exports the per-core
trace.json + instr_exe CSVs that MindStudio Insight consumes. The
recipe is self-contained and covers all three kernel shapes; it does
not assume any pre-existing workspace in `outputs/`.

## When to use

Required entry: `extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args)`.

Three supported kernel shapes (Step 0 picks the right one):

| Shape | Example | Wrapper branches | Args layout |
| ----- | ------- | ---------------- | ----------- |
| **SPMD mix** | `tests/.../spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp` | both `__DAV_CUBE__` *and* `__DAV_VEC__` include the kernel | slot 0..15 tensor/scalar, **slot 48/49 = `LocalContext*`/`GlobalContext*`** (per `intrinsic.h`) |
| **AIC-only task** | `tests/st/a2a3/.../paged_attention_unroll/kernels/aic/aic_pv_matmul.cpp`, `aic_qk_matmul.cpp` | only `__DAV_CUBE__` includes the kernel; `__DAV_VEC__` body is empty | positional Tensor pointers + scalars only; **no slot 48/49** |
| **AIV-only task** | `tests/st/a2a3/.../paged_attention_unroll/kernels/aiv/aiv_softmax_prepare.cpp`, `aiv_online_update.cpp` | only `__DAV_VEC__` includes the kernel; `__DAV_CUBE__` body is empty | positional Tensor pointers + scalars only; **no slot 48/49** |

In all three shapes the kernel is launched via `<<<HW_BLOCK_NUM, ...>>>`.
For SPMD mix, HW_BLOCK_NUM is the kernel's true hw block dim (typically
24). For per-task AIC-only / AIV-only, **HW_BLOCK_NUM=1** is the default
— each hw core would just redo identical work, which doesn't help the
trace and only inflates simulated runtime.

**Skip this skill** if:

- The kernel is PTOAS-style — the dedicated msprof flow already works.
- The user wants AICPU scheduler / tensormap / ringbuffer state machine
  in the trace. Those need real hardware `msprof --application`; the
  simulator returns 207000 (`ACL_ERROR_RT_FEATURE_NOT_SUPPORT`) on the
  AICPU KFC launch. The replay produced here covers AIC/AIV pipeline
  behavior only.

## Step 0 — Classify the kernel shape

Read the kernel source and decide:

1. **Mix or single-arch?** If the file path is under `kernels/mix/` and
   uses `__DAV_CUBE__`/`__DAV_VEC__` macros internally, it's **SPMD mix**.
   If it lives under `kernels/aic/`, it's **AIC-only**. Under
   `kernels/aiv/`, **AIV-only**.
2. **Does `kernel_entry` read slot 48 or 49?** Grep for
   `SPMD_LOCAL_CONTEXT_INDEX`, `get_block_idx(args)`, `get_sub_block_id(args)`,
   or direct `args[48]` / `args[49]` reads. Present → SPMD-context layout.
   Absent → positional layout (typical for AIC-only / AIV-only tasks; the
   orchestration submits one logical task per dispatch).
3. **Does the kernel reference `__DAV_VEC__`-only or `__DAV_CUBE__`-only
   intrinsics?** AIC-only kernels use `TMATMUL`, `TLOAD`/`TSTORE` on
   `TileMat*`, `PIPE_M`, `PIPE_FIX` — these don't compile under
   `__DAV_VEC__`. So an AIC-only kernel **must not** be `#include`d in
   the `__DAV_VEC__` branch of the wrapper, and vice versa.

## Step 1 — Gather inputs

Before writing any code, confirm with the user (or read from the kernel
source):

| Item | Where to look |
| ---- | ------------- |
| Kernel source path | User input |
| Kernel shape (mix / aic-only / aiv-only) | Step 0 above |
| Tensor args (count, dtypes, shapes) | `args[0..N]` reads in `kernel_entry` |
| Scalar args (count, values) | `args[N..]` reads, treat float scalars as `f32_bits()` |
| `hw_block_dim` | SPMD mix: `<<<HW, ...>>>` in the original launcher (typically 24); AIC-only / AIV-only: **default 1** |
| FIFO slot sizes (SPMD mix only) | `PAConfig<...>::*_SLOT_SIZE * FIFO_DEPTH * hw_blocks` — **compute from kernel constants, not from memory** |
| Per-task work knobs | Anything that scales per-block work (`n_blocks`, `context_len`, batch) |

Decide a **shrunk shape**. The simulator runs camodel in serial mode
(`[TmSim]: Run in serial mode.`), so total runtime ≈ Σ (per-block work).
Practical targets:

- Aim for a few-minute simulated run. As a rule of thumb, total work
  units (e.g. `batch × n_blocks` for paged-attention) in the low
  hundreds finishes in a few minutes; tens of thousands will exceed
  any reasonable `task-submit --max-time`. Single-task AIC-only /
  AIV-only kernels with small `n_blocks` finish in well under a minute.
- For SPMD mix: keep `total_logical_blocks ≈ hw_block_dim` so every hw
  block has work and the trace shows all cores active.
- Shrink the *inner* dimension that scales per-block work (e.g.
  `context_len` / `n_blocks` for paged-attention) when the kernel
  allows.

If the kernel hardcodes the shape via `static constexpr` (e.g.
`CASE1_BATCH = 256`), copy the source into the workspace and `sed -i` it
smaller. The wrapper `#include`s the local copy. Sync the host-side
`kBatch` to the same value — both must move together or the kernel walks
off-end. For per-task kernels (`aic_pv_matmul` etc.) the shape comes
from the `Tensor` shapes you build in `replay_host.cpp`, so just shrink
those directly.

## Step 2 — Create the workspace

```bash
TS=$(date +%Y%m%d_%H%M%S)
WS="$REPO_ROOT/outputs/insight_trace_<case>_${TS}"
mkdir -p "$WS"
```

Five files go in `$WS`. Each has a small fixed shape; the only kernel-
specific bits are the `#include` path, `HW_BLOCK_NUM`, and the host-side
shape constants and tensor `init_tensor(...)` calls. The full templates
for each file are inlined in this skill — you do not need to copy from
any pre-existing workspace.

The build must be a **single mix-arch `.so` with a single `launch_replay`**.
Splitting into separate `--cce-aicore-arch=dav-c220-cube` and
`dav-c220-vec` libraries does not produce an OPPROF dump under
`msprof op simulator` — see "Common pitfalls" below.

### File 1 — `replay_kernel.cpp` (mix-arch wrapper)

Inline the **absolute** include path to the target `.cpp`. The wrapper
file is compiled twice by bisheng under `--cce-aicore-arch=dav-c220`
(once per arch); both variants land in the same `.so`. **Only `#include`
the kernel under the arch macro it actually compiles for** — including
an AIC-only kernel under `__DAV_VEC__` (or vice versa) makes bisheng try
to compile pipe/tile intrinsics for the wrong arch and fails.

All three variants below share the same top-of-file boilerplate:

```cpp
#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif
```

Where the per-variant template says `<prologue block>`, paste these
eight lines verbatim before the kernel `#include`:

```cpp
#ifndef __CCE_AICORE__
#define __CCE_AICORE__ 220
#endif
#include <cce_aicore_intrinsics.h>
#ifndef PTO_NPU_ARCH_A2A3
#define PTO_NPU_ARCH_A2A3
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 ((event_t)7)
#endif
#ifndef PIPE_FIX
#define PIPE_FIX ((pipe_t)10)
#endif
```

#### 1a. SPMD mix kernel (e.g. `paged_attention_parallel.cpp`)

```cpp
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
// <prologue block>
#include "<ABSOLUTE PATH TO KERNEL .cpp>"
#endif

extern "C" __global__ AICORE void replay_entry(
    __gm__ int64_t *aic_args, __gm__ int64_t *aiv_args
) {
#if defined(__DAV_CUBE__)
    int32_t hw_idx = get_block_idx();
    kernel_entry(aic_args + static_cast<uint64_t>(hw_idx) * 50);
#endif
#if defined(__DAV_VEC__)
    int32_t lane_idx = static_cast<int32_t>(
        get_block_idx() * get_subblockdim() + get_subblockid());
    kernel_entry(aiv_args + static_cast<uint64_t>(lane_idx) * 50);
#endif
}
```

`50` = `MAX_TENSOR_ARGS(16) + MAX_SCALAR_ARGS(32) + 2`. Do **not** write
16/17 here — that was the old broken pattern.

#### 1b. AIC-only single-task kernel (e.g. `aic_pv_matmul.cpp`, `aic_qk_matmul.cpp`)

The kernel only includes under `__DAV_CUBE__`. The `__DAV_VEC__` variant
of `replay_entry` still gets emitted (so the `replay_entry_mix_aiv`
symbol exists for msprof to find) but its body is empty.

```cpp
#if defined(__DAV_CUBE__)
// <prologue block>
#include "<ABSOLUTE PATH TO AIC KERNEL .cpp>"
#endif

extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args) {
#if defined(__DAV_CUBE__)
    kernel_entry(args);
#endif
    // __DAV_VEC__ variant: intentionally empty — AIC-only kernel.
}
```

For `HW_BLOCK_NUM > 1` (each AIC core reading its own row), use
`kernel_entry(args + static_cast<uint64_t>(get_block_idx()) * 50)` and
allocate per-row args host-side. Default `HW_BLOCK_NUM=1` shares one
args row across all cores.

#### 1c. AIV-only single-task kernel (e.g. `aiv_softmax_prepare.cpp`, `aiv_online_update.cpp`)

Mirror of 1b with arch macros swapped:

```cpp
#if defined(__DAV_VEC__)
// <prologue block>
#include "<ABSOLUTE PATH TO AIV KERNEL .cpp>"
#endif

extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args) {
#if defined(__DAV_VEC__)
    kernel_entry(args);
#endif
    // __DAV_CUBE__ variant: intentionally empty — AIV-only kernel.
}
```

For `HW_BLOCK_NUM > 1`, both AIV lanes (sub_block_id 0/1) of each hw
core are dispatched — use
`kernel_entry(args + (get_block_idx() * get_subblockdim() + get_subblockid()) * 50)`
to give each lane its own row.

### File 2 — `replay_launch.cpp`

The launcher signature must match the wrapper (`replay_entry`) above.
For SPMD mix the wrapper takes two args pointers; for AIC-only / AIV-only
it takes one.

#### 2a. SPMD mix

```cpp
#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void replay_entry(
    __gm__ int64_t *aic_args, __gm__ int64_t *aiv_args);

extern "C" void launch_replay(void *aic_args, void *aiv_args, void *stream) {
    replay_entry<<<HW_BLOCK_NUM, nullptr, stream>>>(
        (__gm__ int64_t *)aic_args, (__gm__ int64_t *)aiv_args);
}
```

#### 2b. AIC-only or AIV-only

```cpp
#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args);

extern "C" void launch_replay(void *args, void *stream) {
    replay_entry<<<HW_BLOCK_NUM, nullptr, stream>>>(
        (__gm__ int64_t *)args);
}
```

Replace `HW_BLOCK_NUM` with the kernel's hw block dim literal — typically
24 for SPMD mix, **1** for per-task AIC-only / AIV-only kernels (see
Step 1).

### File 3 — `replay_host.cpp`

ACL host runner. Common responsibilities (all shapes):

1. `aclInit` → `aclrtSetDevice(getenv("ACL_DEVICE_ID"))` → one stream.
2. `aclrtMalloc` for every input/output tensor and every FIFO/scratch
   buffer the kernel reads or writes.
3. Build the kernel's `Tensor` structs on host, `aclrtMemcpy` to a
   device-side `d_tensors` array.
4. Build per-row `args[]` arrays. Per row: slot 0..N-1 = device pointer
   to `d_tensors[i]`, scalar slots per kernel.
5. `launch_replay(...)` → `aclrtSynchronizeStream` → free.

The shape-specific differences are in the args layout and whether
`LocalContext`/`GlobalContext` arrays are built.

#### 3a. SPMD mix args layout

- FIFO sizes: `hw_blocks * FIFO_DEPTH * SLOT_SIZE` using the kernel's
  `MAX_Q_TILE`-equivalent constants.
- `LocalContext[hw_blocks]` + `GlobalContext[hw_blocks]` for AIC, and
  `LocalContext[2*hw_blocks]` + `GlobalContext[2*hw_blocks]` for AIV.
  `block_idx` = hw block index for both; `sub_block_id` = lane index
  (0/1) for AIV, 0 for AIC.
- `aic_args[hw_blocks * 50]` and `aiv_args[2*hw_blocks * 50]`. Per row:
  slot 0..N-1 = tensor pointers, scalar slots per kernel,
  **`row[48]` = device-side `LocalContext` for that row,
  `row[49]` = device-side `GlobalContext`**.
- `launch_replay(d_aic_args, d_aiv_args, stream)`.

Sketch (initialize ACL, allocate device buffers for inputs/outputs and
the FIFO rings, build per-hw-block `LocalContext`/`GlobalContext`,
populate the args rows, then launch and synchronize):

```cpp
constexpr int kArgsSlots = 50;
constexpr int kHwBlocks = /* kernel's hw block dim, e.g. 24 */;
constexpr int kAivLanes = kHwBlocks * 2;

// d_tensors, d_*_local, d_*_global already aclrtMalloc'd and populated.
auto fill_args = [&](std::vector<int64_t> &args, int rows,
                     uintptr_t local_base, uintptr_t global_base) {
    for (int r = 0; r < rows; ++r) {
        int64_t *row = args.data() + uint64_t(r) * kArgsSlots;
        for (int i = 0; i < kNumTensors; ++i) {
            row[i] = (int64_t)((uintptr_t)d_tensors + i * sizeof(Tensor));
        }
        // row[kNumTensors..] = scalar values (f32_bits() for floats)
        row[48] = (int64_t)(local_base  + r * sizeof(LocalContext));
        row[49] = (int64_t)(global_base + r * sizeof(GlobalContext));
    }
};
std::vector<int64_t> aic_args(kHwBlocks  * kArgsSlots, 0);
std::vector<int64_t> aiv_args(kAivLanes * kArgsSlots, 0);
fill_args(aic_args, kHwBlocks,  (uintptr_t)d_aic_local,  (uintptr_t)d_aic_global);
fill_args(aiv_args, kAivLanes,  (uintptr_t)d_aiv_local,  (uintptr_t)d_aiv_global);
// aclrtMemcpy both into d_aic_args / d_aiv_args, then:
launch_replay(d_aic_args, d_aiv_args, stream);
```

`LocalContext.block_idx` = hw block index (`r` for AIC, `r/2` for AIV);
`GlobalContext.sub_block_id` = lane index (0/1 for AIV, 0 for AIC).

#### 3b. AIC-only / AIV-only positional args layout

- No FIFO rings (those are runtime-managed; here you trace one task in
  isolation).
- **No `LocalContext` / `GlobalContext` arrays** — the kernel doesn't
  read slot 48/49.
- `args[HW_BLOCK_NUM * 50]` (50 slots is uniform; using the kernel's
  exact slot count works too). Per row: slot 0..N-1 = tensor pointers
  for the kernel's positional `args[0..N-1]` reads, slot N..M = scalar
  values (treat float scalars as `f32_bits()`). Slots 48/49 are zero.
- `launch_replay(d_args, stream)`.
- Default `HW_BLOCK_NUM=1` → only fill row 0 and `aclrtMemcpy` 50
  `int64_t`s.

Sketch for `aic_pv_matmul.cpp` (4 tensors + 2 scalars, HW_BLOCK_NUM=1):

```cpp
constexpr int kArgsSlots = 50;
// pij_buf, value_cache, block_table, oi_new + (n_blocks, bt_offset)
init_tensor(&tensors[0], d_pij,        kPijBytes,  DataType::BFLOAT16, {n_blocks, M, K});
init_tensor(&tensors[1], d_value_cache,kVcBytes,   DataType::BFLOAT16, {kTotalKvBlocks, kBlockSize, kKvHeads, kHeadDim});
init_tensor(&tensors[2], d_block_table,kBtBytes,   DataType::INT32,    {kBatch, kMaxBlocksPerReq});
init_tensor(&tensors[3], d_oi_new,     kOiBytes,   DataType::FLOAT32,  {kQTile, kHeadDim});

std::array<int64_t, kArgsSlots> args{};
args[0] = (int64_t)((uintptr_t)d_tensors + 0 * sizeof(Tensor));
args[1] = (int64_t)((uintptr_t)d_tensors + 1 * sizeof(Tensor));
args[2] = (int64_t)((uintptr_t)d_tensors + 2 * sizeof(Tensor));
args[3] = (int64_t)((uintptr_t)d_tensors + 3 * sizeof(Tensor));
args[4] = (int64_t)kNBlocks;    // n_blocks
args[5] = (int64_t)kBtOffset;   // bt_offset
ACL_CHECK(aclrtMemcpy(d_args, sizeof(args), args.data(), sizeof(args),
                      ACL_MEMCPY_HOST_TO_DEVICE));
launch_replay(d_args, stream);
```

For `aiv_softmax_prepare.cpp` etc., follow the same pattern — count the
positional `args[i]` reads in `kernel_entry`, allocate tensors of
matching shape, and pack scalars (remember `f32_bits()` for floats).

Filling args[] with zeros (`aclrtMemset(d_query, ..., 0)`) is fine — the
trace covers pipeline structure, not numerical correctness. If the
kernel has data-dependent early-exit / scale-by-max paths and you want
those branches in the trace, fill with random instead. (Note: if a
kernel reads `block_table` to compute address offsets, leaving it zero
makes every block point at `value_cache[0]` — fine for pipeline tracing
but identical addresses across blocks.)

### File 4 — `CMakeLists.txt`

Single mix-arch shared library: bisheng compiles the wrapper twice under
`--cce-aicore-arch=dav-c220` (no `-cube`/`-vec` suffix) and ships both
variants in one `.so`. Do **not** split into two
`--cce-aicore-arch=dav-c220-cube` + `dav-c220-vec` libs — that layout
produces an empty OPPROF dump under `msprof op simulator`.

Generic template (set `REPO_ROOT` / `PTO_ISA_ROOT` from the environment;
adjust `project()` name only):

```cmake
cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project(insight_trace_replay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set (source CANN set_env.sh first)")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(SOC_VERSION dav_2201 CACHE STRING "Simulator SoC version")
set(PTO_ISA_ROOT $ENV{PTO_ISA_ROOT} CACHE PATH "PTO ISA root")
set(REPO_ROOT $ENV{REPO_ROOT} CACHE PATH "simpler repo root")

add_compile_options(
    -D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong -fPIC
)
add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce -fenable-matrix --cce-aicore-enable-tl -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)
set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

set(COMMON_INCLUDES
    ${PTO_ISA_ROOT}/include
    ${PTO_ISA_ROOT}/include/pto
    ${REPO_ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer/runtime
    ${REPO_ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer/common
    ${REPO_ROOT}/src/common/task_interface
    ${REPO_ROOT}/src/a2a3/platform/include
    ${REPO_ROOT}/simpler_setup/incore
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_HOME_PATH}/include
)

add_library(replay_kernel SHARED replay_kernel.cpp replay_launch.cpp)
target_compile_options(replay_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DREGISTER_BASE -std=c++17)
target_include_directories(replay_kernel PRIVATE ${COMMON_INCLUDES})
target_link_options(replay_kernel PRIVATE --cce-fatobj-link)

add_executable(replay_host replay_host.cpp)
target_compile_options(replay_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(replay_host PRIVATE ${COMMON_INCLUDES})
target_link_directories(replay_host PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib
)
target_link_libraries(replay_host PRIVATE
    replay_kernel
    runtime_camodel
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)
```

The `COMMON_INCLUDES` paths assume the standard simpler-repo layout
(`src/a2a3/runtime/...`, `src/common/...`). If your kernel pulls in
headers from another arch directory, append to `COMMON_INCLUDES`.

### File 5 — `run_collect.sh`

Build + collect + export wrapped in one bash script so `task-submit` can
run it under a single NPU lock. Generic template — set
`CANN_HOME` / `PTO_ISA_ROOT` / `REPO_ROOT` to whatever this machine
provides; do not hard-code absolute paths into the workspace:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Required env (caller sets these — typical values are machine-specific):
#   CANN_HOME       e.g. .../cann-x.y.z-betaN  (must contain set_env.sh)
#   PTO_ISA_ROOT    path to the pto-isa repo checkout
#   REPO_ROOT       path to the simpler repo checkout
: "${CANN_HOME:?CANN_HOME must be set}"
: "${PTO_ISA_ROOT:?PTO_ISA_ROOT must be set}"
: "${REPO_ROOT:?REPO_ROOT must be set}"

WS="${WS:-$(dirname "$(readlink -f "$0")")}"
SOC_VERSION="${SOC_VERSION:-dav_2201}"
DEVICE_ID="${TARGET_DEVICE_ID:-${NPU_LOCKED_DEVICE:-0}}"
BUILD_DIR="$WS/build"
COLLECT_DIR="$WS/msprof_collect"
EXPORT_ROOT="$WS/insight_export"

source "$CANN_HOME/../cann/set_env.sh" 2>/dev/null \
  || source "$CANN_HOME/set_env.sh"
export ASCEND_HOME_PATH="$CANN_HOME"
SIM_LIB_DIR="$CANN_HOME/aarch64-linux/simulator/$SOC_VERSION/lib"
export LD_LIBRARY_PATH="$BUILD_DIR:$SIM_LIB_DIR:$CANN_HOME/lib64:$CANN_HOME/aarch64-linux/devlib:$CANN_HOME/devlib:${LD_LIBRARY_PATH:-}"
export ACL_DEVICE_ID="$DEVICE_ID"
mkdir -p "$BUILD_DIR" "$COLLECT_DIR" "$EXPORT_ROOT"

cmake -G Ninja -S "$WS" -B "$BUILD_DIR" \
    -DSOC_VERSION="$SOC_VERSION" \
    -DPTO_ISA_ROOT="$PTO_ISA_ROOT" \
    -DREPO_ROOT="$REPO_ROOT"
cmake --build "$BUILD_DIR" --target replay_host

msprof op simulator \
  --application="$BUILD_DIR/replay_host" \
  --kernel-name="replay_entry" \
  --launch-count=1 \
  --soc-version="$SOC_VERSION" \
  --timeout=120 \
  --output="$COLLECT_DIR/out" \
  2>&1 | tee "$COLLECT_DIR/msprof_collect.log"

OPPROF_DIR="$(find "$COLLECT_DIR/out" -maxdepth 1 -mindepth 1 -type d -name 'OPPROF_*' | sort | tail -n 1)"
test -n "$OPPROF_DIR"
if [[ -d "$OPPROF_DIR/device0/tmp_dump" ]]; then
  EXPORT_SRC="$OPPROF_DIR/device0/tmp_dump"
else
  EXPORT_SRC="$OPPROF_DIR/dump"
fi

msprof op simulator --export="$EXPORT_SRC" --output="$EXPORT_ROOT" \
  2>&1 | tee "$EXPORT_ROOT/msprof_export.log"
```

`--kernel-name=replay_entry` is **mandatory** under recent CANN releases
— without it, the simulator reports success but writes no `tmp_dump`.
msprof auto-matches `replay_entry_mix_aic` / `replay_entry_mix_aiv` from
this unmangled name. This is the single most common reason for an empty
OPPROF dump.

## Step 3 — Smoke-build before submitting to the queue

If your dev environment uses an NPU-lock queue (`task-submit` or
similar), compile errors should fail locally first. Source the CANN
environment and run cmake:

```bash
source "$CANN_HOME/set_env.sh"   # or the project's set_env wrapper
export PTO_ISA_ROOT="..."
export REPO_ROOT="..."
cmake -G Ninja -S "$WS" -B "$WS/build" \
    -DSOC_VERSION=dav_2201 \
    -DPTO_ISA_ROOT="$PTO_ISA_ROOT" \
    -DREPO_ROOT="$REPO_ROOT"
cmake --build "$WS/build" --target replay_host
nm -D "$WS/build/libreplay_kernel.so" | grep -E ' T (replay_entry|launch_replay)$'
```

Both `replay_entry` and `launch_replay` must show as `T` symbols. Mix-arch
device variants live in fatbinary sections and do not appear in `nm -D` —
that is expected.

## Step 4 — Run the collect script

If your environment has a queue gate (e.g. `task-submit`), wrap the
collect run with the appropriate `--max-time`:

```bash
task-submit --device auto --max-time 1800 --run "bash $WS/run_collect.sh"
```

If you have direct simulator access, just run `bash $WS/run_collect.sh`
after exporting the env vars from Step 3.

Choose `--max-time` based on expected work units (Σ per-block work).
A few hundred units finishes in well under 30 minutes; tens of thousands
of units may need the maximum allowed time and is usually a sign the
shape needs to be shrunk further (Step 1).

The collect log should show:

```text
[INFO] <ProfInit> Start profiling on kernel: replay_entry_mix_aic
... Profiling running finished. All task success.
```

## Step 5 — Verify Insight artifacts

The new (CANN 9.0.0-beta.2) collector already writes the export-shaped
layout under `OPPROF_*/simulator/`; the export step in the script just
regularizes paths. Required output:

```text
$WS/insight_export/OPPROF_*/simulator/
├── trace.json                          # top-level timeline
├── visualize_data.bin                  # Insight visualizer payload
├── core{N}.cubecore0/                  # one per AIC hw block
│   ├── trace.json
│   └── core{N}.cubecore0_instr_exe_*.csv
└── core{N}.veccore{0,1}/               # one per AIV lane
    ├── trace.json
    └── core{N}.veccore{0,1}_instr_exe_*.csv
```

For an SPMD mix launch you should see one `cubecore0` + two `veccore`
subdirectories per hw block (so `HW_BLOCK_NUM × 3` instr_exe CSVs). For
an AIC-only launch with `HW_BLOCK_NUM=1`, you see one `cubecore0/`
(no `veccore/`); for AIV-only, two `veccore{0,1}/` (no `cubecore0/`).
Drag the `simulator/` directory into MindStudio Insight.

## Common pitfalls (in order of how often they bite)

1. **Empty OPPROF dump after a "successful" simulator run.** Almost
   always: `--kernel-name` was missing. Re-add it and re-run.
2. **Compile error: AIC pipe/tile intrinsic in AIV variant (or vice
   versa).** The wrapper `#include`d an AIC-only kernel under both
   `__DAV_CUBE__` and `__DAV_VEC__`. Use the 1b/1c templates: include
   the kernel only under the matching arch, leave the other branch's
   `replay_entry` body empty.
3. **`<<<>>>` launch fails to dispatch AIV (SPMD mix).** Wrapper compiled
   as cube-only — confirm `--cce-aicore-arch=dav-c220` (no suffix) and
   that the wrapper's `__DAV_VEC__` branch is present and includes the
   kernel.
4. **Simulator runs full duration then SIGKILL.** Shape too large for
   `--max-time`. Shrink the inner dimension first (`context_len`,
   `n_blocks`), only then batch. Partial dumps may still export but
   trace.json bloats to GB.
5. **Compile error: `intrinsic.h` not found.** The kernel includes
   `intrinsic.h` and `tensor.h` by name. Confirm `COMMON_INCLUDES` has
   `src/a2a3/runtime/tensormap_and_ringbuffer/{runtime,common}` —
   `tensor.h` lives in `runtime/`, `intrinsic.h` in `common/`.
6. **Off-by-one in SPMD args layout.** Slot 48/49 are positional; if you
   write 16/17 (right after the kernel's last scalar) the kernel reads
   garbage and silently misbehaves. Always 48/49. Does not apply to
   AIC-only / AIV-only positional args (slot 48/49 unused there).
7. **AIC-only/AIV-only kernel reads zeros from a tensor index.**
   Forgot to populate `block_table` etc. — zeros are valid pointers but
   collapse all per-block lookups onto entry 0. Acceptable for pipeline
   tracing; fix only if the trace needs to exercise distinct addresses.
8. **`Kernel missed debug_line information` warning.** Cosmetic. Insight
   loses source-level call stack but trace.json/CSVs are unaffected.
   Add `-g` to the wrapper's compile options if needed.
9. **`x86_64-linux/simulator/...` not found.** The CANN install on this
   host is aarch64. CMakeLists and `LD_LIBRARY_PATH` must point at
   `aarch64-linux/simulator/$SOC_VERSION/lib`.

## Final checklist

- [ ] Step 0 classification done: shape is SPMD mix / AIC-only / AIV-only
- [ ] Workspace under `outputs/insight_trace_<case>_<ts>/`
- [ ] `replay_kernel.cpp` includes the *absolute* path to the target
      kernel **only under the matching arch macro**; the unused-arch
      `replay_entry` body is empty (for AIC-only / AIV-only) or uses
      `args + idx * 50` (for SPMD mix)
- [ ] `replay_launch.cpp` `<<<HW_BLOCK_NUM, ...>>>` matches the kernel
      (typically 24 for SPMD mix, 1 for per-task AIC-only / AIV-only);
      launcher signature matches the wrapper (one or two args pointers)
- [ ] `replay_host.cpp` shape constants match the (possibly shrunk)
      shape. SPMD mix: FIFO sizes computed from kernel constants and
      `LocalContext`/`GlobalContext` arrays built. AIC-only / AIV-only:
      no contexts, only positional args[0..N-1] populated.
- [ ] CMake target uses `--cce-aicore-arch=dav-c220` (single mix-arch)
- [ ] `run_collect.sh` includes `--kernel-name=replay_entry`
- [ ] Local smoke build succeeds; `nm -D` shows `replay_entry` and
      `launch_replay`
- [ ] `task-submit ... bash run_collect.sh` exits 0
- [ ] `msprof_collect.log` shows `Start profiling on kernel:
      replay_entry_mix_aic` (or `_mix_aiv` for AIV-only) and
      `All task success`
- [ ] `simulator/` directory contains top-level `trace.json` +
      `visualize_data.bin`, plus one subdir per cubecore0/veccore lane
      that the kernel actually used
