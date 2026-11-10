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

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>

extern "C" int pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
    static auto real =
        reinterpret_cast<int (*)(pthread_key_t *, void (*)(void *))>(dlsym(RTLD_NEXT, "pthread_key_create"));
    static std::atomic<bool> injected{false};
    Dl_info caller{};
    if (dladdr(__builtin_return_address(0), &caller) && caller.dli_fname &&
        std::strstr(caller.dli_fname, "aicore_sim_") && !injected.exchange(true)) {
        std::fprintf(stderr, "SIM_CORE_KEY_CREATE_EAGAIN\n");
        return EAGAIN;
    }
    return real(key, destructor);
}
