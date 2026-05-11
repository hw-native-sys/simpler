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
/**
 * Chevron-launch wrapper for the a5 AICore kernel.
 *
 * Compiled with `bisheng -xcce -Xhost-start -Xhost-end --cce-aicore-arch=dav-c310`:
 * a single translation unit holds both the AICore kernel definition and the
 * host-side <<<>>> launch site. bisheng's classic CCE frontend emits a fatbin
 * (the AICore code lives inside an embedded section of this object) and
 * expands <<<>>> into the corresponding rtKernelLaunch* call at link time —
 * --cce-fatobj-link on host_runtime.so wires the two halves together. This
 * mirrors the path that the pto-isa mscatter testcase uses successfully.
 *
 * <<<>>> form used here:
 *   <<<numBlocks, l2Ctrl, stream>>>
 *
 * - numBlocks : execution logic core count (== block_dim_)
 * - l2Ctrl    : rtL2Ctrl_t* — nullptr means "use the driver default L2
 *               policy". The runtime ignores this field for non-SIMT kernels.
 * - stream    : aclrtStream the launch is queued on
 *
 * The kernel body mirrors KERNEL_ENTRY(aicore_kernel) from kernel.cpp — it
 * just hands control to the runtime's existing aicore_execute() loop, so
 * AICPU keeps dispatching tasks via the register handshake.
 */
#include "acl/acl.h"
#include "aicore/aicore.h"
#include "aicore_executor.h"
#include "common/core_type.h"

// Local variable names are prefixed `s_` to avoid clashing with the
// `block_idx` / `core_type` macros used by kernel.cpp's per-subcore rename
// block; the bisheng aicore builtin headers also expose builtin names that
// don't allow re-use as declarators inside an __aicore__ function. Same
// workaround as runtime/tensormap_and_ringbuffer/aicore/aicore_executor.h.
// `__aicore__` marks the function as device-side; under
// `--cce-aicore-arch=dav-c310` (mix), bisheng emits both AIC and AIV
// variants of the same source, branching at compile time on __DAV_VEC__.
extern "C" __global__ __aicore__ void aicore_chevron_entry(__gm__ Runtime *runtime) {
    int s_block_idx;
    CoreType s_core_type;
#ifdef __DAV_VEC__
    s_block_idx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    s_core_type = CoreType::AIV;
#else
    s_block_idx = get_block_idx();
    s_core_type = CoreType::AIC;
#endif
    aicore_execute(runtime, s_block_idx, s_core_type);
}

extern "C" int launch_aicore_chevron(uint32_t blockDim, void *stream, void *runtime_dev) {
    // C-style cast bypasses the bisheng address-space strictness that
    // refuses reinterpret_cast<__gm__ T*>(void*). Host callers pass a raw
    // device pointer; the kernel side runs in __aicore__ context where
    // GM is the natural address space for runtime args.
    aicore_chevron_entry<<<blockDim, nullptr, static_cast<aclrtStream>(stream)>>>((__gm__ Runtime *)runtime_dev);
    return 0;
}
