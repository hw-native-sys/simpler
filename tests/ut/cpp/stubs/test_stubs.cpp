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
/**
 * Link-time stubs for platform APIs used by runtime headers.
 *
 * Provides x86-compatible implementations of functions declared in
 * platform headers (unified_log.h, device_time.h, common.h) so that
 * runtime data structures can be unit-tested on CI runners without
 * Ascend hardware or SDK.
 */

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "aicpu/platform_regs.h"  // get_reg_ptr / RegId (arch header picked up via rt_objs include path)

// =============================================================================
// unified_log.h stubs (5 log-level functions)
// =============================================================================

extern "C" {

void unified_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[WARN]  %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_debug(const char * /* func */, const char * /* fmt */, ...) {
    // Suppress debug in tests
}

void unified_log_info_v(const char *func, int v, const char *fmt, ...) {
    // Only emit V9 (must-see) in tests; quieter tiers are suppressed.
    if (v < 9) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO_V%d] %s: ", v, func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

}  // extern "C"

// =============================================================================
// device_time.h stub
// =============================================================================

uint64_t get_sys_cnt_aicpu() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

void cache_invalidate_range(const void * /* addr */, size_t /* size */) {}

void cache_flush_range(const void * /* addr */, size_t /* size */) {}

// =============================================================================
// platform_regs.h stub (get_reg_ptr)
// =============================================================================

// PTO2SchedulerState::ring_one_doorbell (pto_scheduler.h, speculative
// early-dispatch) is an inline that resolves a register id to its MMIO pointer
// via get_reg_ptr and writes a 64-bit token through it. There is no MMIO on the
// host UT runner; hand back writable static storage (8 bytes — the doorbell is a
// 64-bit store) so the inline links and any write is harmless.
volatile uint32_t *get_reg_ptr(uint64_t /* reg_base_addr */, RegId /* reg */) {
    static volatile uint64_t dummy_reg = 0;
    return reinterpret_cast<volatile uint32_t *>(&dummy_reg);
}

// =============================================================================
// runtime_maker.cpp stub (bind_callable_to_runtime_impl)
// =============================================================================

// DeviceRunnerBase::bind_callable_to_runtime (the merged bind facade) calls the
// runtime's bind_callable_to_runtime_impl, which is defined in runtime_maker.cpp
// and only present in the production host_runtime.so. These runner-only unit
// tests link device_runner_base.cpp without any runtime_maker, and their mock
// runners never bind (TestSimRunner::run returns 0), so the impl is never
// invoked — it only has to resolve at link time. C linkage matches by symbol
// name, so this opaque-pointer signature satisfies the reference.
extern "C" int bind_callable_to_runtime_impl(
    void * /* runtime */, const void * /* api */, const void * /* orch_args */, void * /* host_orch_func_ptr */,
    const void * /* signature */, int /* sig_count */, const uint64_t * /* ring_task_window */,
    const uint64_t * /* ring_heap */, const uint64_t * /* ring_dep_pool */
) {
    return -1;
}

// =============================================================================
// common.h stubs (assert_impl, get_stacktrace, AssertionError)
// =============================================================================

std::string get_stacktrace(int /* skip_frames */) { return "<stacktrace not available in test stubs>"; }

class AssertionError : public std::runtime_error {
public:
    AssertionError(const char *condition, const char *file, int line) :
        std::runtime_error(std::string("Assertion failed: ") + condition + " at " + file + ":" + std::to_string(line)),
        condition_(condition),
        file_(file),
        line_(line) {}

    const char *condition() const { return condition_; }
    const char *file() const { return file_; }
    int line() const { return line_; }

private:
    const char *condition_;
    const char *file_;
    int line_;
};

[[noreturn]] void assert_impl(const char *condition, const char *file, int line) {
    throw AssertionError(condition, file, line);
}
