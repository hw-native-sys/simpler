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
#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "host_build_graph/kernel_callable_registration.h"
#include "runtime.h"

namespace hbg {

// The serialized kernel leader is the persistent Runtime table's sole writer.
// Registered callable images remain immutable for their context generation.
// Registration identity and packet restoration succeed before binding.
enum class CallableBindingResult { Rejected, Rebound, Reused };

class KernelCallableBinding {
public:
    CallableBindingResult bind(Runtime *runtime, const HbgCallableRegistration &registration) noexcept {
        if (runtime != nullptr && valid_ && registration.runtime_address == reinterpret_cast<uintptr_t>(runtime) &&
            runtime->get_active_callable_id() == registration.callable_id &&
            std::memcmp(&registration_, &registration, sizeof(registration)) == 0)
            return CallableBindingResult::Reused;
        valid_ = false;
        if (runtime == nullptr) return CallableBindingResult::Rejected;
        runtime->set_active_callable_id(-1);
        if (registration.runtime_address != reinterpret_cast<uintptr_t>(runtime) ||
            registration.callable_bytes < sizeof(ChipCallable))
            return CallableBindingResult::Rejected;
        auto *callable = reinterpret_cast<const ChipCallable *>(registration.callable_address);
        // Binding reads metadata headers; uploaded orchestration and kernel code stay immutable.
        cache_invalidate_range(callable, sizeof(*callable));
        const int32_t children = callable->child_count_;
        if (children < 0 || children > KERNEL_MAX_FUNC_ID) return CallableBindingResult::Rejected;
        runtime->clear_function_bin_addrs();
        for (int32_t i = 0; i < children; ++i) {
            const int32_t func_id = callable->child_func_ids_[i];
            const uint32_t offset = callable->child_offsets_[i];
            constexpr size_t header = offsetof(ChipCallable, storage_);
            if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID || offset > registration.callable_bytes - header ||
                sizeof(CoreCallable) > registration.callable_bytes - header - offset)
                return CallableBindingResult::Rejected;
            const auto *child = reinterpret_cast<const CoreCallable *>(
                reinterpret_cast<const uint8_t *>(callable) + offsetof(ChipCallable, storage_) + offset
            );
            cache_invalidate_range(child, sizeof(*child));
            if (child->resolved_addr_ == 0) return CallableBindingResult::Rejected;
            // HBG stores the device-side CoreCallable header in func_id_to_addr_.
            // SchedulerContext::build_payload() dereferences this header to obtain
            // resolved_addr(); storing the code address here would make it interpret
            // AICore instructions as a CoreCallable object.
            runtime->replay_function_bin_addr(func_id, reinterpret_cast<uint64_t>(child));
        }
        runtime->set_active_callable_id(registration.callable_id);
        registration_ = registration;
        valid_ = true;
        return CallableBindingResult::Rebound;
    }

private:
    HbgCallableRegistration registration_{};
    bool valid_{false};
};

}  // namespace hbg
