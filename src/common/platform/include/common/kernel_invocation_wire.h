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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/launch_entry_args.h"
#include "task_interface/kernel_invocation_header.h"
#include "task_interface/arg_direction.h"
#include "task_interface/tensor.h"

namespace simpler::tmr {

struct EagerKernelBinding {
    uint32_t runtime_descriptor_offset{0};
    uint32_t runtime_descriptor_bytes{0};
    uint64_t run_epoch{0};
};
static_assert(std::is_trivially_copyable_v<EagerKernelBinding> && std::is_standard_layout_v<EagerKernelBinding>);
static_assert(sizeof(EagerKernelBinding) == 16);

inline size_t kernel_invocation_bytes(uint32_t tensors, uint32_t scalars) noexcept {
    return sizeof(SimplerKernelInvocationHeader) + sizeof(EagerKernelBinding) +
           static_cast<size_t>(tensors) * sizeof(ChipTensor) + static_cast<size_t>(scalars) * sizeof(uint64_t);
}

// The descriptor follows the complete invocation (header, binding, tensors and scalars).
// Both the writer and the reader derive the only permitted offset from bounded counts.
inline size_t kernel_runtime_descriptor_offset(uint32_t tensors, uint32_t scalars, size_t alignment) noexcept {
    const size_t end = LAUNCH_ENVELOPE_HEADER_BYTES + kernel_invocation_bytes(tensors, scalars);
    return (end + alignment - 1) & ~(alignment - 1);
}

}  // namespace simpler::tmr
