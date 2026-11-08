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
#include "platform_comm/ccu/ccu_device.h"
#include "platform_comm/ccu/ccu_device_access.h"
#include "task_interface/tensor.h"

#if defined(__CCE_AICORE__)
#define CCU_COMPILE_KERNEL extern "C" __global__[aicore]
#define CCU_COMPILE_GM __gm__
#else
#define CCU_COMPILE_KERNEL extern "C"
#define CCU_COMPILE_GM
#endif
CCU_COMPILE_KERNEL void ccu_device_compile(
    uint64_t address, uint64_t size, CCU_COMPILE_GM ChipTensor *input, CCU_COMPILE_GM ChipTensor *output, uint64_t count
) {
    CCU_COMPILE_GM CcuDeviceContext *ctx = nullptr;
    if (ccu_get_device_context(address, size, ctx) != CcuStatus::OK) return;
    CcuProgramKey key{};
    CcuProgramRef program{};
    if (ccu_resolve_program(ctx, key, program) != CcuStatus::OK) return;
    CcuRequestRef request{};
    if (ccu_reserve_request(ctx, program, request, *input, *output, count) != CcuStatus::OK) return;
    if (ccu_submit_request(request) != CcuStatus::OK) return;
    const auto result = ccu_poll_request(request);
    if (result.state == CcuRequestState::SUCCEEDED) ccu_retire_request(request);
    if (ccu_resolve_program(ctx, uint32_t{0}, program) != CcuStatus::OK) return;
    CcuPositionalArguments positional{};
    positional.tensor_count = 2;
    positional.scalar_count = 1;
    const auto src = ccu_detail::Value(*input), dst = ccu_detail::Value(*output);
    positional.tensors[0] = {src.address, 0, src.bytes};
    positional.tensors[1] = {dst.address, 0, dst.bytes};
    positional.scalars[0] = {count, CcuScalarKind::UNSIGNED, 64};
    if (ccu_reserve_request(ctx, program, request, positional) == CcuStatus::OK) ccu_cancel_reserved_request(request);
}
CCU_COMPILE_KERNEL void ccu_facade_compile(uint64_t address, uint64_t size, int32_t callable_id) {
    using namespace simpler::ccu::device;
    auto *workspace = reinterpret_cast<CCU_COMPILE_GM Workspace *>(address);
    Reference reference{};
    Arguments arguments{};
    if (!Protocol<>::resolve(workspace, size, callable_id, 0, reference)) return;
    if (!Protocol<>::reserve(reference, arguments)) return;
    if (!Protocol<>::submit(reference)) {
        Protocol<>::cancel(reference);
        return;
    }
    if (Protocol<>::poll(reference) == Outcome::SUCCEEDED) {
        Protocol<>::synchronize_arguments(reference);
        Protocol<>::retire(reference);
    }
}
#undef CCU_COMPILE_KERNEL
#undef CCU_COMPILE_GM
