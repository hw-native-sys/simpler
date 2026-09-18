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

#include "worker/runtime_c_api.h"
#include "worker/pipeline_contract.h"

// Internal host-runtime hook, not a dlsym lifecycle API. Input is borrowed and
// immutable during the call; output is caller-exclusive and unchanged on error.
// No device resources are acquired or retained. Separate calls may run concurrently.
// TMR rejects invalid config with INVALID_ARGUMENT and a null output or invalid
// generated contract with INTERNAL. Unsupported implementations return UNSUPPORTED.
extern "C" int build_kernel_pipeline_contract_impl(const CallConfig *config, PipelineContract *out);

class Runtime;
// Host-only preparation of runtime-specific static fields; no device work.
int configure_kernel_runtime_impl(Runtime &runtime, bool serial_orch_sched);

struct HostApi;
// Commit this runtime's context-static device regions on a kernel context and
// wire `runtime` to them. Runs once per context, before the runtime image is
// uploaded, and allocates nothing afterwards: a kernel context's regions keep
// their addresses for its whole life because a captured graph replays the
// addresses of the run it captured.
int prepare_kernel_runtime_impl(Runtime &runtime, const HostApi *api, const CallConfig *config);

// Nonzero when this runtime implements the kernel-mode launch path: a device
// entry the AICPU loader can resolve, and a payload consumer behind it. The
// capability entry reports this, so a caller that gates on it never reaches a
// launch the runtime cannot service.
extern "C" int runtime_supports_kernel_launch_impl(void);

// Runtime selector for the shared platform owner. HBG uses a graph packet and
// execution slot; TMR keeps the fixed dispatch packet path.
extern "C" int runtime_uses_hbg_kernel_impl(void);

// Runtime-specific architecture tag used by the common HBG context-resource
// owner. The value is one of hbg::RuntimeArchitecture.
extern "C" uint32_t runtime_hbg_kernel_architecture_impl(void);

// Borrowed, call-local role bindings; no stream is created or retained here.
// Caller, dedicated AICPU and hidden AICore are three distinct streams.
struct KernelStreamBinding {
    void *caller_stream{nullptr};
    void *aicpu_stream{nullptr};
    void *aicore_stream{nullptr};
};

inline int bind_kernel_stream_roles(
    const PipelineContract *contract, void *caller_stream, void *aicpu_stream, void *hidden_aicore_stream,
    KernelStreamBinding &out
) {
    if (!is_valid_pipeline_contract(contract, SIMPLER_MODE_KERNEL) || !has_serviceable_arena_topology(*contract) ||
        !has_serviceable_stream_topology(*contract) || caller_stream == nullptr || hidden_aicore_stream == nullptr ||
        aicpu_stream == nullptr || caller_stream == aicpu_stream || caller_stream == hidden_aicore_stream ||
        aicpu_stream == hidden_aicore_stream) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    out = {caller_stream, aicpu_stream, hidden_aicore_stream};
    return 0;
}
