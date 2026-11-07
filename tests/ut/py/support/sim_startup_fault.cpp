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
#include <cstring>
#include <new>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

namespace {
// Substituted by the test before compilation; no production configuration.
constexpr const char *trigger = "@THREAD@";
constexpr bool fail_creation = false;
constexpr int skip_bindings = 0;
std::atomic<int> matching_bindings{0};
std::atomic<bool> triggered{false};
std::atomic<bool> reject_next_create{false};
pthread_t target_creator{};
thread_local pthread_t creator{};
struct Start {
    void *(*entry)(void *);
    void *arg;
    pthread_t parent;
};
void *start(void *arg) {
    auto *payload = static_cast<Start *>(arg);
    const Start saved = *payload;
    delete payload;
    creator = saved.parent;
    return saved.entry(saved.arg);
}
}  // namespace

extern "C" int pthread_setspecific(pthread_key_t key, const void *value) noexcept {
    using Set = int (*)(pthread_key_t, const void *);
    static auto real = reinterpret_cast<Set>(dlsym(RTLD_NEXT, "pthread_setspecific"));
    char name[16]{};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    if (std::strcmp(name, trigger) == 0 && matching_bindings.fetch_add(1) >= skip_bindings &&
        !triggered.exchange(true)) {
        if (!fail_creation) {
            const char message[] = "SIM_STARTUP_BIND_ENOMEM\n";
            (void)write(2, message, sizeof(message) - 1);
            return ENOMEM;
        }
        // Only the launch parent may consume the failure; unrelated background
        // threads can create workers while the parent waits for this binding.
        target_creator = creator;
        reject_next_create.store(true);
    }
    return real(key, value);
}

extern "C" int
pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*entry)(void *), void *arg) noexcept {
    using Create = int (*)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    static auto real = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
    if (reject_next_create.load() && pthread_equal(pthread_self(), target_creator) &&
        reject_next_create.exchange(false)) {
        const char message[] = "SIM_STARTUP_CREATE_EAGAIN\n";
        (void)write(2, message, sizeof(message) - 1);
        return EAGAIN;
    }
    auto *payload = new (std::nothrow) Start{entry, arg, pthread_self()};
    if (payload == nullptr) return ENOMEM;
    const int rc = real(thread, attr, start, payload);
    if (rc != 0) delete payload;
    return rc;
}
