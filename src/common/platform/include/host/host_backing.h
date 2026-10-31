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
#include <cstdio>
#include <exception>
#include <stdexcept>

// Private same-build host API. A failed operation retains its opaque owner for
// release retry; none of these handles or function pointers enter device PODs.
extern "C" {
int host_backing_allocate(
    void **owner, int device, uint32_t numa, uint64_t bytes, uint64_t *mapping_size, uint8_t *export_data,
    size_t export_capacity, size_t *export_size, char *error, size_t error_capacity
);
int host_backing_authorize(void *owner, const int32_t *identities, size_t count, char *error, size_t error_capacity);
int host_backing_get_base(void *owner, uint64_t *base, char *error, size_t error_capacity);
int host_backing_release(void **owner, char *error, size_t error_capacity);
int host_backing_process_identity(int32_t *identity, char *error, size_t error_capacity);
}

namespace simpler::host {
inline void check_host_backing_result(int status, const char *error) {
    if (status == 1) throw std::invalid_argument(error);
    if (status) throw std::runtime_error(error);
}

template <typename Operation>
int call_host_backing(char *error, size_t capacity, Operation operation) noexcept {
    if (error && capacity) error[0] = '\0';
    try {
        operation();
        return 0;
    } catch (const std::invalid_argument &failure) {
        if (error && capacity) std::snprintf(error, capacity, "%s", failure.what());
        return 1;
    } catch (const std::exception &failure) {
        if (error && capacity) std::snprintf(error, capacity, "%s", failure.what());
    } catch (...) {
        if (error && capacity) std::snprintf(error, capacity, "%s", "Unknown Host backing failure");
    }
    return 2;
}
}  // namespace simpler::host
