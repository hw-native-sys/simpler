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

#include <dlfcn.h>
#include <runtime/context.h>

#include <cstdlib>

#include "acl/acl.h"

namespace {

constexpr int kSwitchFailure = -771;
constexpr int kRegisterFailure = -772;
thread_local int pending_failure = 0;

template <typename Function>
Function resolve(const char *library, const char *symbol) {
    void *module = dlopen(library, RTLD_LAZY | RTLD_NOLOAD);
    if (module == nullptr) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(module, symbol));
    dlclose(module);
    if (function == nullptr) std::abort();
    return function;
}

}  // namespace

extern "C" void simpler_test_a3_host_map_fail_next(int failure) { pending_failure = failure; }

extern "C" rtError_t rtCtxSetCurrent(rtContext_t context) {
    if (pending_failure == kSwitchFailure) {
        pending_failure = 0;
        return kSwitchFailure;
    }
    static auto real = resolve<decltype(&rtCtxSetCurrent)>("libruntime.so", "rtCtxSetCurrent");
    return real(context);
}

extern "C" aclError aclrtHostRegister(void *host, uint64_t bytes, aclrtHostRegisterType type, void **mapped) {
    if (pending_failure == kRegisterFailure) {
        pending_failure = 0;
        return kRegisterFailure;
    }
    static auto real = resolve<decltype(&aclrtHostRegister)>("libascendcl.so", "aclrtHostRegister");
    return real(host, bytes, type, mapped);
}
