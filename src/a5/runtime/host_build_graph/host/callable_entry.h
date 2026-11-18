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

#include "callable.h"

enum class SchedulerCallableAddressSpace { DEVICE, HOST };

inline uint64_t scheduler_resolve_callable_entry(uint64_t object, SchedulerCallableAddressSpace address_space) {
    if (object == 0 || object % alignof(CoreCallable) != 0) return 0;
    if (address_space == SchedulerCallableAddressSpace::HOST)
        return reinterpret_cast<const CoreCallable *>(object)->resolved_addr();
    // Onboard registration patches resolved_addr to this binary address;
    // the device object itself is not readable through a Host pointer.
    constexpr uint64_t offset = CoreCallable::binary_data_offset();
    return object > UINT64_MAX - offset ? 0 : object + offset;
}
