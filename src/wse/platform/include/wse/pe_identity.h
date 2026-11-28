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

#include "common/hardware_spec.h"
#include <cstdint>
#include <dlfcn.h>

namespace wse {
// The process-resident simulator owns this pthread key. Its value on a WSE PE
// thread is the physical worker index plus one. Other simulators bind object
// addresses, and a host thread normally has no binding.
static inline int32_t current_worker_id() noexcept {
    using ReadContext = void *(*)();
    static const auto get = reinterpret_cast<ReadContext>(dlsym(RTLD_DEFAULT, "sim_context_get_invocation_context"));
    const auto encoded = get ? reinterpret_cast<uintptr_t>(get()) : 0;
    return encoded > 0 && encoded <= MAX_CORE_COUNT * PE_PER_CORE ? static_cast<int32_t>(encoded - 1) : -1;
}
}  // namespace wse

// Both coordinates are physical and remain fixed for the lifetime of a simulated PE thread.
static inline int32_t get_core_id() noexcept {
    const auto worker = wse::current_worker_id();
    return worker < 0 ? -1 : worker / static_cast<int32_t>(wse::PE_PER_CORE);
}
static inline int32_t get_pe_id() noexcept {
    const auto worker = wse::current_worker_id();
    return worker < 0 ? -1 : worker % static_cast<int32_t>(wse::PE_PER_CORE);
}
