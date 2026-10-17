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

#include <cstdlib>

namespace {
int (*release_bindings)(void *, const void *) = nullptr;
int injected_result = 0;
}  // namespace

extern "C" void set_binding_release_result(void *function, int result) {
    release_bindings = reinterpret_cast<int (*)(void *, const void *)>(function);
    injected_result = result;
}

extern "C" int release_run_bindings_impl(void *runtime, const void *api) {
    if (release_bindings == nullptr) std::abort();
    const int rc = release_bindings(runtime, api);
    return rc != 0 ? rc : injected_result;
}
