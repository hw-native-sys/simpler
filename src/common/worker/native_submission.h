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
#include <stdexcept>
#include <utility>

// The named Python capsule is inspected only on the caller thread, with GIL.
// Retain a NativeSubmissionRef before handing it to a native queue callback.
inline constexpr const char *SIMPLER_NATIVE_SUBMISSION_CAPSULE = "simpler.NativeSubmission.v1";

struct SimplerNativeSubmissionV1 {
    uint32_t version;
    void *context;
    void (*retain)(void *) noexcept;
    void (*release)(void *) noexcept;
    int (*submit)(void *, char *, size_t) noexcept;
};

// This value, including its final destructor, never accesses Python.
class NativeSubmissionRef {
public:
    explicit NativeSubmissionRef(const SimplerNativeSubmissionV1 &api) :
        api_(api) {
        if (api_.version != 1 || !api_.context || !api_.retain || !api_.release || !api_.submit)
            throw std::invalid_argument("invalid Simpler native submission capsule");
        api_.retain(api_.context);
    }
    NativeSubmissionRef(const NativeSubmissionRef &other) :
        api_(other.api_) {
        if (api_.context) api_.retain(api_.context);
    }
    NativeSubmissionRef(NativeSubmissionRef &&other) noexcept :
        api_(other.api_) {
        other.api_.context = nullptr;
    }
    NativeSubmissionRef &operator=(NativeSubmissionRef other) noexcept {
        std::swap(api_, other.api_);
        return *this;
    }
    ~NativeSubmissionRef() {
        if (api_.context) api_.release(api_.context);
    }
    void submit() const {
        if (!api_.context) throw std::logic_error("empty Simpler native submission reference");
        char message[1024]{};
        if (api_.submit(api_.context, message, sizeof(message)) != 0) throw std::runtime_error(message);
    }

private:
    SimplerNativeSubmissionV1 api_;
};
