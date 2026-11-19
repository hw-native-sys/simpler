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
// The active binding and completion state live in the AICPU DSO. Sim invokes
// these same C symbols through dlsym instead of creating a second Host copy.
#include "aicpu/device_run_result_base_aicpu.h"

#include <atomic>
#include "aicpu/native_invocation_completion.h"
#include "aicpu/device_run_result_aicpu.h"

namespace {
std::atomic<uint64_t> g_platform_run_result_base{0};
std::atomic<uint64_t> g_platform_run_result_epoch{0};
}  // namespace

extern "C" void set_platform_run_result(uint64_t region_base, uint64_t run_epoch) {
    g_platform_run_result_base.store(region_base, std::memory_order_relaxed);
    g_platform_run_result_epoch.store(run_epoch, std::memory_order_relaxed);
}

extern "C" uint64_t get_platform_run_result_base() {
    return g_platform_run_result_base.load(std::memory_order_relaxed);
}

extern "C" uint64_t get_platform_run_result_epoch() {
    return g_platform_run_result_epoch.load(std::memory_order_relaxed);
}

namespace {
NativeInvocationCompletion g_native_completion;
NativeInvocationResultStorage *g_native_storage{nullptr};
}  // namespace

extern "C" int32_t aicpu_begin_native_invocation(int32_t callbacks, uint64_t region_base, uint64_t run_epoch) {
    // Called by the entry gate's sole initializer after prior physical exit.
    if (callbacks <= 0 || callbacks > MAX_GATE_THREADS) return NativeInvocationCompletion::kInternalError;
    g_native_storage = nullptr;
    set_platform_run_result(region_base, run_epoch);
    if (region_base != 0 && run_epoch == 0) {
        g_native_completion.begin(callbacks);
        return NativeInvocationCompletion::kInternalError;
    }
    auto *storage = reinterpret_cast<NativeInvocationResultStorage *>(region_base);
    if (storage != nullptr) cache_invalidate_range(&storage->evidence, sizeof(storage->evidence));
    if (!g_native_completion.begin(callbacks, storage != nullptr ? &storage->evidence : nullptr))
        return NativeInvocationCompletion::kInternalError;
    g_native_storage = storage;
    if (storage != nullptr) cache_flush_range(&storage->evidence, sizeof(storage->evidence));
    return aicpu_invalidate_run_terminal(region_base, run_epoch) ? 0 : NativeInvocationCompletion::kInternalError;
}

extern "C" bool aicpu_accept_runtime_terminal(RunTerminalSelection selection, const void *payload, size_t bytes) {
    return g_native_completion.accept_runtime_terminal(selection, payload, bytes);
}

extern "C" int32_t aicpu_finish_native_invocation(int32_t callback_index, int32_t status) {
    return g_native_completion.finish(
        callback_index, status,
        [](RunTerminalSelection result, const void *payload, size_t bytes) {
            const uint64_t base = get_platform_run_result_base();
            const uint64_t epoch = get_platform_run_result_epoch();
            if (base == 0) return NativeTerminalPublication::NotRequested;
            // This publisher performs synchronous stores/cache maintenance,
            // not a fallible RTS transfer. False means invalid producer inputs.
            return aicpu_publish_run_terminal(base, epoch, result.verdict, result.code, result.source, payload, bytes) ?
                       NativeTerminalPublication::Published :
                       NativeTerminalPublication::InvalidState;
        },
        [] {
            if (g_native_storage != nullptr)
                cache_flush_range(&g_native_storage->evidence, sizeof(g_native_storage->evidence));
        }
    );
}
