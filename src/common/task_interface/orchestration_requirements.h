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

#include <cstdint>

namespace simpler::orchestration {

// Optional metadata exported by an orchestration SO. Program mode keeps
// accepting legacy SOs; HBG kernel mode requires this symbol because absence
// cannot prove that Host graph construction is independent of device bytes.
inline constexpr const char *REQUIREMENTS_V1_SYMBOL = "pypto_orchestration_requirements_v1";

inline constexpr uint64_t REQUIREMENT_TENSOR_DATA_READ = UINT64_C(1) << 0;
inline constexpr uint64_t REQUIREMENT_TENSOR_DATA_WRITE = UINT64_C(1) << 1;
inline constexpr uint64_t REQUIREMENTS_V1_KNOWN_MASK = REQUIREMENT_TENSOR_DATA_READ | REQUIREMENT_TENSOR_DATA_WRITE;

using RequirementsV1Function = uint64_t (*)();

enum class HbgKernelRequirementsStatus : uint8_t {
    Ok = 0,
    MetadataUnavailable,
    UnknownRequirement,
    HostCopyRequired,
    TensorDataWriteUnsupported,
};

// A Host read is legal only when the invocation explicitly carries at least
// one trailing host-only duplicate. Runtime access control still decides which
// tensor is readable, so declaring a copy cannot make a DEVICE tensor readable.
// Host writes remain unsupported: a host-only duplicate is input metadata, not
// a second output channel whose mutations would be replayed.
inline constexpr HbgKernelRequirementsStatus validate_hbg_kernel_requirements(
    bool metadata_available, uint64_t requirements, int32_t host_copy_tensor_count
) noexcept {
    if (!metadata_available) return HbgKernelRequirementsStatus::MetadataUnavailable;
    if ((requirements & ~REQUIREMENTS_V1_KNOWN_MASK) != 0) return HbgKernelRequirementsStatus::UnknownRequirement;
    if ((requirements & REQUIREMENT_TENSOR_DATA_WRITE) != 0)
        return HbgKernelRequirementsStatus::TensorDataWriteUnsupported;
    if ((requirements & REQUIREMENT_TENSOR_DATA_READ) != 0 && host_copy_tensor_count <= 0)
        return HbgKernelRequirementsStatus::HostCopyRequired;
    return HbgKernelRequirementsStatus::Ok;
}

}  // namespace simpler::orchestration
