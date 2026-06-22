/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <type_traits>

// Type headers needed by orchestration
#include "pto_runtime2_types.h"  // PTO2_ERROR_*
#include "pto_submit_types.h"    // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "pto_types.h"           // Arg, TaskOutputTensors, TensorArgType
#include "task_args.h"           // ChipStorageTaskArgs, ContinuousTensor
#include "tensor.h"              // Tensor, TensorCreateInfo

inline Tensor make_tensor_external(void *addr, const uint32_t shapes[], uint32_t ndims, DataType dtype = DataType::FLOAT32, bool manual_dep = false, int32_t version = 0)
{
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) total *= shapes[i];
    return {addr, total * get_element_size(dtype), shapes, ndims, dtype, version, manual_dep};
}

// Convert ContinuousTensor to Tensor
static_assert(CONTINUOUS_TENSOR_MAX_DIMS == RUNTIME_MAX_TENSOR_DIMS, "ContinuousTensor and runtime max dims must match");
inline Tensor from_tensor_arg(const ContinuousTensor &t, bool manual_dep = false, int32_t version = 0)
{
    return make_tensor_external(reinterpret_cast<void *>(static_cast<uintptr_t>(t.data)), t.shapes, t.ndims, t.dtype, manual_dep, version);
}

typedef struct PTO2Runtime PTO2Runtime;

#ifdef __cplusplus
extern "C" {
#endif

PTO2Runtime *framework_current_runtime(void);
void framework_bind_runtime(PTO2Runtime *rt);

#ifdef __cplusplus
}
#endif

typedef struct PTO2RuntimeOps
{
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    // INFO with explicit verbosity tier (v ∈ [0, 9]; gating done inside).
    void (*log_info_v)(const char *func, int v, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value);
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const Arg &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const Arg &args);

    void (*scope_set_site)(const char *file, int line);
} PTO2RuntimeOps;

struct PTO2Runtime
{
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode;
};

static inline PTO2Runtime *current_runtime()
{
    return framework_current_runtime();
}

static inline TaskOutputTensors alloc_tensors(const Arg &args)
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return TaskOutputTensors{};
    return rt->ops->alloc_tensors(rt, args);
}

static inline TaskOutputTensors alloc_tensors(const TensorCreateInfo create_infos[], uint32_t count)
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return TaskOutputTensors{};
    Arg args;
    for (uint32_t i = 0; i < count; i++) args.add_output(create_infos[i]);
    if (args.has_error)
    {
        rt->ops->report_fatal(rt, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s", args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg");
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

template <typename... CIs>
static inline TaskOutputTensors alloc_tensors(const CIs &...cis)
{
    static_assert(sizeof...(cis) > 0, "alloc_tensors requires at least one TensorCreateInfo");
    static_assert((std::is_same_v<std::decay_t<CIs>, TensorCreateInfo> && ...), "alloc_tensors only accepts TensorCreateInfo arguments");
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return TaskOutputTensors{};
    Arg args;
    (args.add_output(cis), ...);
    if (args.has_error)
    {
        rt->ops->report_fatal(rt, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s", args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg");
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const Arg &args)
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return TaskOutputTensors{};
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const Arg &args)
{
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const Arg &args)
{
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

static inline TaskOutputTensors rt_submit_dummy_task(const Arg &args)
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return TaskOutputTensors{};
    return rt->ops->submit_dummy_task(rt, args);
}

static inline void rt_scope_begin(PTO2ScopeMode mode = PTO2ScopeMode::AUTO)
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return;
    rt->pending_scope_mode = mode;
    rt->ops->scope_begin(rt);
}

static inline void rt_scope_end()
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return;
    rt->ops->scope_end(rt);
}

static inline void rt_orchestration_done()
{
    PTO2Runtime *rt = current_runtime();
    rt->ops->orchestration_done(rt);
}

static inline bool rt_is_fatal()
{
    PTO2Runtime *rt = current_runtime();
    return rt->ops->is_fatal(rt);
}

#define rt_report_fatal(code, fmt, ...)                                          \
    do {                                                                         \
        PTO2Runtime *_rt = current_runtime();                                    \
        _rt->ops->report_fatal(_rt, (code), __FUNCTION__, (fmt), ##__VA_ARGS__); \
    } while (0)

// INFO verbosity tiers. v=0 most verbose, v=9 must-see, v=5 default.

template <typename T = uint64_t>
static inline T get_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[])
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return from_u64<T>(0);
    return from_u64<T>(rt->ops->get_tensor_data(rt, tensor, ndims, indices));
}

template <typename T = uint64_t>
static inline void set_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value)
{
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return;
    rt->ops->set_tensor_data(rt, tensor, ndims, indices, to_u64(value));
}

class PTO2ScopeGuard
{
public:
    explicit PTO2ScopeGuard(PTO2ScopeMode mode = PTO2ScopeMode::AUTO, const char *file = __builtin_FILE(), int line = __builtin_LINE()) :
        rt_(current_runtime())
    {
        if (!rt_->ops->is_fatal(rt_))
        {
            rt_->pending_scope_mode = mode;
            if (rt_->ops->scope_set_site) rt_->ops->scope_set_site(file, line);
            rt_->ops->scope_begin(rt_);
        }
    }
    ~PTO2ScopeGuard()
    {
        if (!rt_->ops->is_fatal(rt_)) rt_->ops->scope_end(rt_);
    }

private:
    PTO2Runtime *rt_;
};

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD() [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__)

#define PTO2_SCOPE(...) if (PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

// User-orchestration logging macros. Route through the runtime's ops table so
// the verbosity gating (V0..V9) and the actual logging sink stay owned by the
// runtime. The orchestration .so just calls — gating is done inside.
#define LOG_INFO_V0(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 0, fmt, ##__VA_ARGS__)
#define LOG_INFO_V1(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 1, fmt, ##__VA_ARGS__)
#define LOG_INFO_V2(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 2, fmt, ##__VA_ARGS__)
#define LOG_INFO_V3(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 3, fmt, ##__VA_ARGS__)
#define LOG_INFO_V4(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 4, fmt, ##__VA_ARGS__)
#define LOG_INFO_V5(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 5, fmt, ##__VA_ARGS__)
#define LOG_INFO_V6(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 6, fmt, ##__VA_ARGS__)
#define LOG_INFO_V7(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 7, fmt, ##__VA_ARGS__)
#define LOG_INFO_V8(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 8, fmt, ##__VA_ARGS__)
#define LOG_INFO_V9(fmt, ...) current_runtime()->ops->log_info_v(__FUNCTION__, 9, fmt, ##__VA_ARGS__)

#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig
{
    int expected_arg_count;
};
#endif

#include "pto_arg_with_deps.h"  // NOLINT(build/include_subdir)
