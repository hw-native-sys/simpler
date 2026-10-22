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

#include "aicpu/platform_entry_args.h"
#include "kernel_invocation.h"
#include "runtime.h"
#include "task_interface/kernel_capture.h"

namespace simpler::tmr {

inline bool admit_kernel_launch_entry(
    const Runtime &runtime, const PlatformEntryArgs &view, const KernelCallableIdentity &registered, uint64_t run_epoch,
    EntryArgsStorage *output
) {
    if (!valid_kernel_identity(registered) || view.args_base == nullptr || run_epoch == 0 ||
        (view.source != static_cast<uint32_t>(EntryArgsSource::KernelInvocation) &&
         (view.source != static_cast<uint32_t>(EntryArgsSource::KernelCaptureInvocation) || view.binding_epoch == 0)) ||
        runtime.get_entry_args_source() != EntryArgsSource::KernelInvocation ||
        view.offset != LAUNCH_ENVELOPE_HEADER_BYTES || view.tensor_count != registered.tensor_count ||
        view.scalar_count != registered.scalar_count || view.tensor_count != runtime.get_entry_tensor_count() ||
        view.scalar_count != runtime.get_entry_scalar_count() ||
        registered.callable_id != runtime.get_active_callable_id())
        return false;
    const size_t offset = view.source == static_cast<uint32_t>(EntryArgsSource::KernelCaptureInvocation) ?
                              KERNEL_CAPTURE_INVOCATION_OFFSET :
                              LAUNCH_ENVELOPE_HEADER_BYTES;
    const auto *packet = static_cast<const unsigned char *>(view.args_base) + offset;
    return consume_kernel_invocation(
        packet, kernel_invocation_bytes(registered.tensor_count, registered.scalar_count), registered,
        {reinterpret_cast<uint64_t>(&runtime), run_epoch}, output
    );
}

}  // namespace simpler::tmr
