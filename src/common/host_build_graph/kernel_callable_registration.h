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
#include <type_traits>

#include "callable.h"
#include "task_interface/kernel_invocation_validation.h"

namespace hbg {

inline constexpr uint32_t HBG_CALLABLE_REGISTRATION_VERSION = 1;

// Prepare-owned metadata. The device callable image remains owned by the
// context callable cache; this record only lends it to one context generation.
struct HbgCallableRegistration {
    uint32_t version{HBG_CALLABLE_REGISTRATION_VERSION};
    uint32_t bytes{sizeof(HbgCallableRegistration)};
    uint64_t context_generation{0};
    uint64_t runtime_address{0};
    uint64_t register_table_address{0};
    uint64_t callable_address{0};
    uint64_t callable_bytes{0};
    uint64_t callable_hash{0};
    uint64_t function_hash{0};
    int32_t callable_id{-1};
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t device_id{-1};
    uint64_t runtime_binary_id{0};
};

static_assert(std::is_standard_layout_v<HbgCallableRegistration>);
static_assert(std::is_trivially_copyable_v<HbgCallableRegistration>);
static_assert(sizeof(HbgCallableRegistration) == 88);

inline bool valid_hbg_callable_registration(const HbgCallableRegistration &r) noexcept {
    return r.version == HBG_CALLABLE_REGISTRATION_VERSION && r.bytes == sizeof(r) && r.context_generation != 0 &&
           r.runtime_address != 0 && r.register_table_address != 0 &&
           r.register_table_address % alignof(uint64_t) == 0 && r.callable_address != 0 &&
           r.callable_bytes >= sizeof(ChipCallable) && r.callable_hash != 0 && r.function_hash != 0 &&
           r.device_id >= 0 && r.runtime_binary_id != 0 &&
           simpler::kernel::valid_prepared_invocation({r.callable_id, r.tensor_count, r.scalar_count});
}

}  // namespace hbg

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_callable(void *arg);
