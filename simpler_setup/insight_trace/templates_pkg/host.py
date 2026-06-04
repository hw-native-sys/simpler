# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from functools import reduce
from operator import mul

from ..models import KernelShape, TraceConfig, TraceScalarArg, TraceTensorArg
from .common import _DTYPE_CPP, _DTYPE_SIZE, _camel, _require_kernel, _require_spmd_meta, _validate_args


def render_host(config: TraceConfig) -> str:
    _validate_args(config.args)
    kernel = _require_kernel(config)
    tensors = [arg for arg in config.args if isinstance(arg, TraceTensorArg)]
    scalars = [arg for arg in config.args if isinstance(arg, TraceScalarArg)]
    constants = []
    allocs = []
    tensor_inits = []
    frees = []
    for i, arg in enumerate(tensors):
        elements = reduce(mul, arg.shape, 1)
        size = elements * _DTYPE_SIZE[arg.dtype]
        shape_name = f"shape_{arg.name}"
        constants.append(f"constexpr int k{_camel(arg.name)}Bytes = {size};")
        constants.append(f"uint32_t {shape_name}[{len(arg.shape)}] = {{{', '.join(str(dim) for dim in arg.shape)}}};")
        allocs.append(f"    void *d_{arg.name} = nullptr;")
        allocs.append(
            f"    ACL_CHECK(aclrtMalloc(&d_{arg.name}, k{_camel(arg.name)}Bytes, ACL_MEM_MALLOC_HUGE_FIRST));"
        )
        allocs.append(
            f"    ACL_CHECK(aclrtMemset(d_{arg.name}, k{_camel(arg.name)}Bytes, 0, k{_camel(arg.name)}Bytes));"
        )
        tensor_inits.append(
            f"    tensors[{i}] = make_tensor_external("
            f"d_{arg.name}, {shape_name}, {len(arg.shape)}, {_DTYPE_CPP[arg.dtype]});"
        )
        frees.append(f"    ACL_CHECK(aclrtFree(d_{arg.name}));")

    if kernel.shape == KernelShape.SPMD_MIX:
        if config.platform_arch.family.value == "a5" and config.spmd_meta is not None:
            return _render_a5_spmd_host(config, tensors, scalars, constants, allocs, tensor_inits, frees)
        if config.platform_arch.family.value == "a5":
            return _render_a5_mixed_host(config, tensors, scalars, constants, allocs, tensor_inits, frees)
        return _render_spmd_host(config, tensors, scalars, constants, allocs, tensor_inits, frees)
    return _render_single_task_host(tensors, scalars, constants, allocs, tensor_inits, frees)


def _render_single_task_host(tensors, scalars, constants, allocs, tensor_inits, frees) -> str:
    arg_assigns = []
    for i, arg in enumerate(tensors):
        arg_assigns.append(
            f"    args[{arg.index}] = static_cast<int64_t>("
            f"reinterpret_cast<uintptr_t>(d_tensors) + {i} * sizeof(Tensor));"
        )
    for arg in scalars:
        arg_assigns.append(f"    args[{arg.index}] = static_cast<int64_t>({int(arg.value)});")

    return f"""#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "acl/acl.h"
#include "pto_orchestration_api.h"

constexpr int kArgsSlots = 50;
constexpr int kNumTensors = {len(tensors)};
{chr(10).join(constants)}

#define ACL_CHECK(expr) \\
    do {{ \\
        aclError _err = (expr); \\
        if (_err != ACL_SUCCESS) {{ \\
            fprintf(stderr, "ACL error %d at %s:%d\\n", _err, __FILE__, __LINE__); \\
            exit(1); \\
        }} \\
    }} while (0)

extern "C" void launch_replay(void *args, void *stream);

int main() {{
    ACL_CHECK(aclInit(nullptr));
    int device_id = 0;
    if (const char *env = getenv("ACL_DEVICE_ID")) {{
        device_id = atoi(env);
    }}
    ACL_CHECK(aclrtSetDevice(device_id));

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

{chr(10).join(allocs)}

    void *tensors_mem = malloc(kNumTensors * sizeof(Tensor));
    Tensor *tensors = static_cast<Tensor *>(tensors_mem);
{chr(10).join(tensor_inits)}

    void *d_tensors = nullptr;
    ACL_CHECK(aclrtMalloc(&d_tensors, kNumTensors * sizeof(Tensor), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_tensors, kNumTensors * sizeof(Tensor), tensors,
                           kNumTensors * sizeof(Tensor), ACL_MEMCPY_HOST_TO_DEVICE));

    std::array<int64_t, kArgsSlots> args{{}};
{chr(10).join(arg_assigns)}

    void *d_args = nullptr;
    ACL_CHECK(aclrtMalloc(&d_args, sizeof(args), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_args, sizeof(args), args.data(), sizeof(args),
                           ACL_MEMCPY_HOST_TO_DEVICE));

    launch_replay(d_args, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(d_args));
    ACL_CHECK(aclrtFree(d_tensors));
    free(tensors_mem);
{chr(10).join(reversed(frees))}
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());

    printf("Replay completed successfully.\\n");
    return 0;
}}
"""


def _render_a5_mixed_host(config, tensors, scalars, constants, allocs, tensor_inits, frees) -> str:
    if config.spmd_meta is not None:
        raise ValueError("A5 mixed host replay does not support SPMD context synthesis in the first pass")

    aic_assigns = []
    aiv_assigns = []
    for i, arg in enumerate(tensors):
        expr = f"static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_tensors) + {i} * sizeof(Tensor))"
        aic_assigns.append(f"        row[{arg.index}] = {expr};")
        aiv_assigns.append(f"        row[{arg.index}] = {expr};")
    for arg in scalars:
        expr = f"static_cast<int64_t>({int(arg.value)})"
        aic_assigns.append(f"        row[{arg.index}] = {expr};")
        aiv_assigns.append(f"        row[{arg.index}] = {expr};")

    return f"""#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "pto_orchestration_api.h"

constexpr int kArgsSlots = 50;
constexpr int kNumTensors = {len(tensors)};
constexpr int kHwBlocks = {config.hw_block_num};
constexpr int kAivLanesPerCore = 2;
constexpr int kAivRows = kHwBlocks * kAivLanesPerCore;
{chr(10).join(constants)}

#define ACL_CHECK(expr) \\
    do {{ \\
        aclError _err = (expr); \\
        if (_err != ACL_SUCCESS) {{ \\
            fprintf(stderr, "ACL error %d at %s:%d\\n", _err, __FILE__, __LINE__); \\
            exit(1); \\
        }} \\
    }} while (0)

extern "C" void launch_replay(void *aic_args, void *aiv_args, void *stream);

int main() {{
    ACL_CHECK(aclInit(nullptr));
    int device_id = 0;
    if (const char *env = getenv("ACL_DEVICE_ID")) {{
        device_id = atoi(env);
    }}
    ACL_CHECK(aclrtSetDevice(device_id));

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

{chr(10).join(allocs)}

    void *tensors_mem = malloc(kNumTensors * sizeof(Tensor));
    Tensor *tensors = static_cast<Tensor *>(tensors_mem);
{chr(10).join(tensor_inits)}

    void *d_tensors = nullptr;
    ACL_CHECK(aclrtMalloc(&d_tensors, kNumTensors * sizeof(Tensor), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_tensors, kNumTensors * sizeof(Tensor), tensors,
                           kNumTensors * sizeof(Tensor), ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> aic_args(kHwBlocks * kArgsSlots, 0);
    std::vector<int64_t> aiv_args(kAivRows * kArgsSlots, 0);
    for (int r = 0; r < kHwBlocks; ++r) {{
        int64_t *row = aic_args.data() + static_cast<uint64_t>(r) * kArgsSlots;
{chr(10).join(aic_assigns)}
    }}
    for (int r = 0; r < kAivRows; ++r) {{
        int64_t *row = aiv_args.data() + static_cast<uint64_t>(r) * kArgsSlots;
{chr(10).join(aiv_assigns)}
    }}

    void *d_aic_args = nullptr;
    void *d_aiv_args = nullptr;
    ACL_CHECK(aclrtMalloc(&d_aic_args, aic_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_args, aiv_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_aic_args, aic_args.size() * sizeof(int64_t), aic_args.data(),
                           aic_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_args, aiv_args.size() * sizeof(int64_t), aiv_args.data(),
                           aiv_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    launch_replay(d_aic_args, d_aiv_args, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(d_aiv_args));
    ACL_CHECK(aclrtFree(d_aic_args));
    ACL_CHECK(aclrtFree(d_tensors));
    free(tensors_mem);
{chr(10).join(reversed(frees))}
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());

    printf("Replay completed successfully.\\n");
    return 0;
}}
"""


def _render_a5_spmd_host(config, tensors, scalars, constants, allocs, tensor_inits, frees) -> str:
    meta = _require_spmd_meta(config)

    aic_assigns = []
    aiv_assigns = []
    for i, arg in enumerate(tensors):
        expr = f"static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_tensors) + {i} * sizeof(Tensor))"
        aic_assigns.append(f"        row[{arg.index}] = {expr};")
        aiv_assigns.append(f"        row[{arg.index}] = {expr};")
    for arg in scalars:
        expr = f"static_cast<int64_t>({int(arg.value)})"
        aic_assigns.append(f"        row[{arg.index}] = {expr};")
        aiv_assigns.append(f"        row[{arg.index}] = {expr};")

    dispatch_inits = []
    for dispatch in meta.dispatches:
        scalar_pairs = ", ".join(f"{{{index}, {value}}}" for index, value in dispatch.scalar_overrides)
        dispatch_inits.append(f"        {{{dispatch.logical_block_num}, {{{scalar_pairs}}}}}")

    aiv_lane_assigns = chr(10).join(line.replace("row[", "lane_row[") for line in aiv_assigns)

    return f"""#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "pto_orchestration_api.h"
#include "intrinsic.h"

constexpr int kArgsSlots = 50;
constexpr int kNumTensors = {len(tensors)};
constexpr int kAivLanesPerCore = {meta.aiv_lanes_per_core};
constexpr int kHwBlocks = {config.hw_block_num};
{chr(10).join(constants)}

#define ACL_CHECK(expr) \\
    do {{ \\
        aclError _err = (expr); \\
        if (_err != ACL_SUCCESS) {{ \\
            fprintf(stderr, "ACL error %d at %s:%d\\n", _err, __FILE__, __LINE__); \\
            exit(1); \\
        }} \\
    }} while (0)

struct ReplayDispatch {{
    int logical_block_num;
    std::vector<std::pair<int, int64_t>> scalar_overrides;
}};

extern "C" void launch_replay(void *aic_args, void *aiv_args, void *stream);

int main() {{
    ACL_CHECK(aclInit(nullptr));
    int device_id = 0;
    if (const char *env = getenv("ACL_DEVICE_ID")) {{
        device_id = atoi(env);
    }}
    ACL_CHECK(aclrtSetDevice(device_id));

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

{chr(10).join(allocs)}

    void *tensors_mem = malloc(kNumTensors * sizeof(Tensor));
    Tensor *tensors = static_cast<Tensor *>(tensors_mem);
{chr(10).join(tensor_inits)}

    void *d_tensors = nullptr;
    ACL_CHECK(aclrtMalloc(&d_tensors, kNumTensors * sizeof(Tensor), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_tensors, kNumTensors * sizeof(Tensor), tensors,
                           kNumTensors * sizeof(Tensor), ACL_MEMCPY_HOST_TO_DEVICE));

    const std::vector<ReplayDispatch> dispatches = {{
{chr(10).join(dispatch_inits)}
    }};
    int total_rows = 0;
    for (const auto &dispatch : dispatches) {{
        total_rows += dispatch.logical_block_num;
    }}

    std::vector<LocalContext> aic_local(total_rows);
    std::vector<GlobalContext> aic_global(total_rows);
    std::vector<LocalContext> aiv_local(total_rows * kAivLanesPerCore);
    std::vector<GlobalContext> aiv_global(total_rows * kAivLanesPerCore);

    int row_base = 0;
    for (const auto &dispatch : dispatches) {{
        for (int block_idx = 0; block_idx < dispatch.logical_block_num; ++block_idx) {{
            int row_index = row_base + block_idx;
            aic_local[row_index].s_block_idx = block_idx;
            aic_local[row_index].s_block_num = dispatch.logical_block_num;
            aic_global[row_index].sub_block_id = 0;
            for (int lane = 0; lane < kAivLanesPerCore; ++lane) {{
                int lane_row_index = row_index * kAivLanesPerCore + lane;
                aiv_local[lane_row_index].s_block_idx = block_idx;
                aiv_local[lane_row_index].s_block_num = dispatch.logical_block_num;
                aiv_global[lane_row_index].sub_block_id = lane;
            }}
        }}
        row_base += dispatch.logical_block_num;
    }}

    void *d_aic_local = nullptr;
    void *d_aic_global = nullptr;
    void *d_aiv_local = nullptr;
    void *d_aiv_global = nullptr;
    ACL_CHECK(aclrtMalloc(&d_aic_local, aic_local.size() * sizeof(LocalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aic_global, aic_global.size() * sizeof(GlobalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_local, aiv_local.size() * sizeof(LocalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_global, aiv_global.size() * sizeof(GlobalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_aic_local, aic_local.size() * sizeof(LocalContext),
                           aic_local.data(), aic_local.size() * sizeof(LocalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aic_global, aic_global.size() * sizeof(GlobalContext),
                           aic_global.data(), aic_global.size() * sizeof(GlobalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_local, aiv_local.size() * sizeof(LocalContext),
                           aiv_local.data(), aiv_local.size() * sizeof(LocalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_global, aiv_global.size() * sizeof(GlobalContext),
                           aiv_global.data(), aiv_global.size() * sizeof(GlobalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> aic_args(total_rows * kArgsSlots, 0);
    std::vector<int64_t> aiv_args(total_rows * kAivLanesPerCore * kArgsSlots, 0);
    row_base = 0;
    for (const auto &dispatch : dispatches) {{
        for (int block_idx = 0; block_idx < dispatch.logical_block_num; ++block_idx) {{
            int row_index = row_base + block_idx;
            int64_t *row = aic_args.data() + static_cast<uint64_t>(row_index) * kArgsSlots;
{chr(10).join(aic_assigns)}
            for (const auto &[arg_index, value] : dispatch.scalar_overrides) {{
                row[arg_index] = value;
            }}
            row[48] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aic_local)
                                   + row_index * sizeof(LocalContext));
            row[49] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aic_global)
                                   + row_index * sizeof(GlobalContext));

            for (int lane = 0; lane < kAivLanesPerCore; ++lane) {{
                int lane_row_index = row_index * kAivLanesPerCore + lane;
                int64_t *lane_row = aiv_args.data() + static_cast<uint64_t>(lane_row_index) * kArgsSlots;
{aiv_lane_assigns}
                for (const auto &[arg_index, value] : dispatch.scalar_overrides) {{
                    lane_row[arg_index] = value;
                }}
                lane_row[48] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aiv_local)
                                       + lane_row_index * sizeof(LocalContext));
                lane_row[49] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aiv_global)
                                       + lane_row_index * sizeof(GlobalContext));
            }}
        }}
        row_base += dispatch.logical_block_num;
    }}

    void *d_aic_args = nullptr;
    void *d_aiv_args = nullptr;
    ACL_CHECK(aclrtMalloc(&d_aic_args, aic_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_args, aiv_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_aic_args, aic_args.size() * sizeof(int64_t), aic_args.data(),
                           aic_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_args, aiv_args.size() * sizeof(int64_t), aiv_args.data(),
                           aiv_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    launch_replay(d_aic_args, d_aiv_args, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(d_aiv_args));
    ACL_CHECK(aclrtFree(d_aic_args));
    ACL_CHECK(aclrtFree(d_aiv_global));
    ACL_CHECK(aclrtFree(d_aiv_local));
    ACL_CHECK(aclrtFree(d_aic_global));
    ACL_CHECK(aclrtFree(d_aic_local));
    ACL_CHECK(aclrtFree(d_tensors));
    free(tensors_mem);
{chr(10).join(reversed(frees))}
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());

    printf("Replay completed successfully.\\n");
    return 0;
}}
"""


def _render_spmd_host(config, tensors, scalars, constants, allocs, tensor_inits, frees) -> str:
    if config.platform_arch.family.value == "a5":
        raise ValueError(
            "A5 SPMD mix host replay not supported; use a single-task A5 kernel "
            "or an A5 mixed kernel without context synthesis"
        )
    meta = _require_spmd_meta(config)
    spmd_fifo_names = {"sij_fifo", "pij_fifo", "oi_fifo"}

    filtered_constants = []
    filtered_allocs = []
    filtered_tensor_inits = []
    filtered_frees = []
    spmd_tensor_lines = []
    filtered_tensors: list[TraceTensorArg] = []
    fifo_tensor_map: dict[str, TraceTensorArg] = {}

    for i, arg in enumerate(tensors):
        if arg.name in spmd_fifo_names:
            fifo_tensor_map[arg.name] = arg
            continue
        filtered_tensors.append(arg)
        filtered_constants.append(constants[i * 2])
        filtered_constants.append(constants[i * 2 + 1])
        filtered_allocs.extend(allocs[i * 3 : i * 3 + 3])
        filtered_tensor_inits.append(tensor_inits[len(filtered_tensors) - 1])
        filtered_frees.append(frees[i])

    fifo_specs = (
        ("sij_fifo", "kSpmdSijFifoBytes", "d_sij_fifo", "shape_sij_fifo", "DataType::FLOAT32", meta.fifo_sizes[0]),
        ("pij_fifo", "kSpmdPijFifoBytes", "d_pij_fifo", "shape_pij_fifo", "DataType::BFLOAT16", meta.fifo_sizes[1]),
        ("oi_fifo", "kSpmdOiFifoBytes", "d_oi_fifo", "shape_oi_fifo", "DataType::FLOAT32", meta.fifo_sizes[2]),
    )
    for name, const_name, dev_name, shape_name, dtype_cpp, fifo_bytes in fifo_specs:
        arg = fifo_tensor_map.get(name)
        if arg is None:
            continue
        filtered_constants.append(f"constexpr int {const_name} = {fifo_bytes};")
        filtered_constants.append(f"uint32_t {shape_name}[1] = {{{fifo_bytes}}};")
        spmd_tensor_lines.append(
            f"    tensors[{len(filtered_tensors)}] = make_tensor_external({dev_name}, {shape_name}, 1, {dtype_cpp});"
        )
        filtered_tensors.append(arg)

    aic_assigns = []
    aiv_assigns = []
    for i, arg in enumerate(filtered_tensors):
        expr = f"static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_tensors) + {i} * sizeof(Tensor))"
        aic_assigns.append(f"        row[{arg.index}] = {expr};")
        aiv_assigns.append(f"        row[{arg.index}] = {expr};")
    for arg in scalars:
        aic_assigns.append(f"        row[{arg.index}] = static_cast<int64_t>({int(arg.value)});")
        aiv_assigns.append(f"        row[{arg.index}] = static_cast<int64_t>({int(arg.value)});")

    return f"""#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "pto_orchestration_api.h"
#include "intrinsic.h"

constexpr int kArgsSlots = 50;
constexpr int kNumTensors = {len(filtered_tensors)};
constexpr int kHwBlocks = {meta.hw_block_dim};
constexpr int kAivLanesPerCore = {meta.aiv_lanes_per_core};
constexpr int kAivRows = kHwBlocks * kAivLanesPerCore;
{chr(10).join(filtered_constants)}

#define ACL_CHECK(expr) \\
    do {{ \\
        aclError _err = (expr); \\
        if (_err != ACL_SUCCESS) {{ \\
            fprintf(stderr, "ACL error %d at %s:%d\\n", _err, __FILE__, __LINE__); \\
            exit(1); \\
        }} \\
    }} while (0)

extern "C" void launch_replay(void *aic_args, void *aiv_args, void *stream);

int main() {{
    ACL_CHECK(aclInit(nullptr));
    int device_id = 0;
    if (const char *env = getenv("ACL_DEVICE_ID")) {{
        device_id = atoi(env);
    }}
    ACL_CHECK(aclrtSetDevice(device_id));

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

{chr(10).join(filtered_allocs)}
    void *d_sij_fifo = nullptr;
    void *d_pij_fifo = nullptr;
    void *d_oi_fifo = nullptr;
    ACL_CHECK(aclrtMalloc(&d_sij_fifo, kSpmdSijFifoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_sij_fifo, kSpmdSijFifoBytes, 0, kSpmdSijFifoBytes));
    ACL_CHECK(aclrtMalloc(&d_pij_fifo, kSpmdPijFifoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_pij_fifo, kSpmdPijFifoBytes, 0, kSpmdPijFifoBytes));
    ACL_CHECK(aclrtMalloc(&d_oi_fifo, kSpmdOiFifoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_oi_fifo, kSpmdOiFifoBytes, 0, kSpmdOiFifoBytes));

    void *tensors_mem = malloc(kNumTensors * sizeof(Tensor));
    Tensor *tensors = static_cast<Tensor *>(tensors_mem);
{chr(10).join(filtered_tensor_inits)}
{chr(10).join(spmd_tensor_lines)}

    void *d_tensors = nullptr;
    ACL_CHECK(aclrtMalloc(&d_tensors, kNumTensors * sizeof(Tensor), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_tensors, kNumTensors * sizeof(Tensor), tensors,
                           kNumTensors * sizeof(Tensor), ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<LocalContext> aic_local(kHwBlocks);
    std::vector<GlobalContext> aic_global(kHwBlocks);
    std::vector<LocalContext> aiv_local(kAivRows);
    std::vector<GlobalContext> aiv_global(kAivRows);
    for (int r = 0; r < kHwBlocks; ++r) {{
        aic_local[r].block_idx = r;
        aic_local[r].block_num = kHwBlocks;
        aic_global[r].sub_block_id = 0;
    }}
    for (int r = 0; r < kAivRows; ++r) {{
        aiv_local[r].block_idx = r / kAivLanesPerCore;
        aiv_local[r].block_num = kHwBlocks;
        aiv_global[r].sub_block_id = r % kAivLanesPerCore;
    }}

    void *d_aic_local = nullptr;
    void *d_aic_global = nullptr;
    void *d_aiv_local = nullptr;
    void *d_aiv_global = nullptr;
    ACL_CHECK(aclrtMalloc(&d_aic_local, aic_local.size() * sizeof(LocalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aic_global, aic_global.size() * sizeof(GlobalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_local, aiv_local.size() * sizeof(LocalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_global, aiv_global.size() * sizeof(GlobalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_aic_local, aic_local.size() * sizeof(LocalContext),
                           aic_local.data(), aic_local.size() * sizeof(LocalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aic_global, aic_global.size() * sizeof(GlobalContext),
                           aic_global.data(), aic_global.size() * sizeof(GlobalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_local, aiv_local.size() * sizeof(LocalContext),
                           aiv_local.data(), aiv_local.size() * sizeof(LocalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_global, aiv_global.size() * sizeof(GlobalContext),
                           aiv_global.data(), aiv_global.size() * sizeof(GlobalContext),
                           ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> aic_args(kHwBlocks * kArgsSlots, 0);
    std::vector<int64_t> aiv_args(kAivRows * kArgsSlots, 0);
    for (int r = 0; r < kHwBlocks; ++r) {{
        int64_t *row = aic_args.data() + static_cast<uint64_t>(r) * kArgsSlots;
{chr(10).join(aic_assigns)}
        row[48] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aic_local) + r * sizeof(LocalContext));
        row[49] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aic_global) + r * sizeof(GlobalContext));
    }}
    for (int r = 0; r < kAivRows; ++r) {{
        int64_t *row = aiv_args.data() + static_cast<uint64_t>(r) * kArgsSlots;
{chr(10).join(aiv_assigns)}
        row[48] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aiv_local) + r * sizeof(LocalContext));
        row[49] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(d_aiv_global) + r * sizeof(GlobalContext));
    }}

    void *d_aic_args = nullptr;
    void *d_aiv_args = nullptr;
    ACL_CHECK(aclrtMalloc(&d_aic_args, aic_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_aiv_args, aiv_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(d_aic_args, aic_args.size() * sizeof(int64_t), aic_args.data(),
                           aic_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_aiv_args, aiv_args.size() * sizeof(int64_t), aiv_args.data(),
                           aiv_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    launch_replay(d_aic_args, d_aiv_args, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(d_aiv_args));
    ACL_CHECK(aclrtFree(d_aic_args));
    ACL_CHECK(aclrtFree(d_aiv_global));
    ACL_CHECK(aclrtFree(d_aiv_local));
    ACL_CHECK(aclrtFree(d_aic_global));
    ACL_CHECK(aclrtFree(d_aic_local));
    ACL_CHECK(aclrtFree(d_tensors));
    free(tensors_mem);
    ACL_CHECK(aclrtFree(d_oi_fifo));
    ACL_CHECK(aclrtFree(d_pij_fifo));
    ACL_CHECK(aclrtFree(d_sij_fifo));
{chr(10).join(reversed(filtered_frees))}
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());

    printf("Replay completed successfully.\\n");
    return 0;
}}
"""
