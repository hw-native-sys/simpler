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

#include <cassert>
#include <cstddef>
#include <pthread.h>

// Keys borrow cells owned by the wrapper's stack. Binding writes happen before
// the startup gate; profiling/handshake setters only update bound cells.
// Exit-time unbinding still checks and returns pthread errors.
template <std::size_t N>
class SimAicoreTlsBinding {
public:
    explicit SimAicoreTlsBinding(pthread_key_t *const (&keys)[N]) :
        keys_(keys) {}
    ~SimAicoreTlsBinding() { clear(); }
    SimAicoreTlsBinding(const SimAicoreTlsBinding &) = delete;
    SimAicoreTlsBinding &operator=(const SimAicoreTlsBinding &) = delete;
    int bind() {
        for (; bound_ < N; ++bound_) {
            const int rc = pthread_setspecific(*keys_[bound_], &values_[bound_]);
            if (rc != 0) return rc;
        }
        return 0;
    }
    int clear() noexcept {
        int first_error = 0;
        while (bound_ != 0) {
            const int rc = pthread_setspecific(*keys_[--bound_], nullptr);
            if (first_error == 0) first_error = rc;
        }
        return first_error;
    }

private:
    pthread_key_t *const (&keys_)[N];
    void *values_[N]{};
    std::size_t bound_{0};
};

inline void *sim_tls_value(pthread_key_t key) {
    auto *cell = static_cast<void **>(pthread_getspecific(key));
    return cell != nullptr ? *cell : nullptr;
}
inline void sim_set_tls_value(pthread_key_t key, void *value) {
    auto *cell = static_cast<void **>(pthread_getspecific(key));
    // A bound wrapper cell is required in every build; this assertion diagnoses
    // violations in debug builds only.
    assert(cell != nullptr && "sim TLS setter requires a bound wrapper cell");
    if (cell != nullptr) *cell = value;
}
