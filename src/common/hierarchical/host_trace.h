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

#include <cstdint>
#include <string>

#include "profiling_config.h"

namespace simpler::host_trace {

#if SIMPLER_HOST_STRACE

bool bind_process_sink() noexcept;
int64_t now_ns() noexcept;
void emit(
    const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, int64_t timestamp_ns,
    int64_t duration_ns, const char *attributes
) noexcept;

class SpanScope {
public:
    SpanScope(const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, std::string attributes);
    ~SpanScope();

    SpanScope(const SpanScope &) = delete;
    SpanScope &operator=(const SpanScope &) = delete;
    SpanScope(SpanScope &&) = delete;
    SpanScope &operator=(SpanScope &&) = delete;

private:
    const char *name_;
    uint64_t invocation_id_;
    uint64_t callable_hash_;
    int32_t depth_;
    int64_t timestamp_ns_;
    std::string attributes_;
};

#else

inline bool bind_process_sink() noexcept { return false; }
inline int64_t now_ns() noexcept { return 0; }
inline void emit(const char *, uint64_t, uint64_t, int32_t, int64_t, int64_t, const char *) noexcept {}

#endif

}  // namespace simpler::host_trace
