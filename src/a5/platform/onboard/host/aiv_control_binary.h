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

#include <dlfcn.h>
#include <filesystem>

namespace simpler::a5 {

inline std::filesystem::path aiv_control_binary(const void *host_symbol) {
    Dl_info info{};
    if (::dladdr(host_symbol, &info) == 0 || !info.dli_fname) return {};
    return std::filesystem::path(info.dli_fname).parent_path() / "ccu_event_trigger.o";
}

}  // namespace simpler::a5
