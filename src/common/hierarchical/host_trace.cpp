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

#include "host_trace.h"

#if SIMPLER_HOST_STRACE

#include <atomic>
#include <chrono>
#include <utility>

#include <dlfcn.h>

#include "common/host_span.h"

namespace simpler::host_trace {
namespace {

std::atomic<SimplerHostSpanSink> g_sink{nullptr};

}  // namespace

bool bind_process_sink() noexcept {
    auto sink = reinterpret_cast<SimplerHostSpanSink>(dlsym(RTLD_DEFAULT, "simpler_log_emit_host_span"));
    if (sink == nullptr) return false;
    g_sink.store(sink, std::memory_order_release);
    return true;
}

int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void emit(
    const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, int64_t timestamp_ns,
    int64_t duration_ns, const char *attributes
) noexcept {
    auto sink = g_sink.load(std::memory_order_acquire);
    if (sink == nullptr) return;
    const SimplerHostSpan span{
        SIMPLER_HOST_SPAN_ABI_VERSION,
        sizeof(SimplerHostSpan),
        invocation_id,
        callable_hash,
        depth,
        0,
        timestamp_ns,
        duration_ns,
        name,
        attributes
    };
    sink(&span);
}

SpanScope::SpanScope(
    const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, std::string attributes
) :
    name_(name),
    invocation_id_(invocation_id),
    callable_hash_(callable_hash),
    depth_(depth),
    timestamp_ns_(now_ns()),
    attributes_(std::move(attributes)) {}

SpanScope::~SpanScope() {
    const int64_t end_ns = now_ns();
    emit(name_, invocation_id_, callable_hash_, depth_, timestamp_ns_, end_ns - timestamp_ns_, attributes_.c_str());
}

}  // namespace simpler::host_trace

#endif
