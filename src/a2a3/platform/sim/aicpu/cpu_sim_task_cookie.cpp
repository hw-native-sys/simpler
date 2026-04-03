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

#include <cstdint>
#include <dlfcn.h>

#include "aicpu/cpu_sim_task_cookie.h"

void platform_set_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id, uint64_t task_cookie) {
    using Fn = void (*)(uint32_t, uint32_t, uint64_t);
    static auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "platform_set_cpu_sim_task_cookie"));
    if (fn != nullptr) {
        fn(core_id, reg_task_id, task_cookie);
    }
}
