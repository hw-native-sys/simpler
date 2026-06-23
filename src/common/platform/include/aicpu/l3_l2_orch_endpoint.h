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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_ORCH_ENDPOINT_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_ORCH_ENDPOINT_H_

#if !defined(__aarch64__)
#include <chrono>
#endif
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common/l3_l2_orch_comm.h"

inline void l3_l2_orch_endpoint_cache_invalidate_range(const void *addr, size_t size) {
#if defined(L3_L2_ORCH_ENDPOINT_ENABLE_CACHE_MAINTENANCE) && defined(__aarch64__)
    if (size == 0) {
        return;
    }
    const size_t kCacheLineSize = 64;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(kCacheLineSize - 1);
    uintptr_t end = (reinterpret_cast<uintptr_t>(addr) + size + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    for (uintptr_t p = start; p < end; p += kCacheLineSize) {
        __asm__ __volatile__("dc civac, %0" ::"r"(p) : "memory");
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
#else
    (void)addr;
    (void)size;
#endif
}

inline void l3_l2_orch_endpoint_cache_flush_range(const void *addr, size_t size) {
#if defined(L3_L2_ORCH_ENDPOINT_ENABLE_CACHE_MAINTENANCE) && defined(__aarch64__)
    if (size == 0) {
        return;
    }
    const size_t kCacheLineSize = 64;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(kCacheLineSize - 1);
    uintptr_t end = (reinterpret_cast<uintptr_t>(addr) + size + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    for (uintptr_t p = start; p < end; p += kCacheLineSize) {
        __asm__ __volatile__("dc cvac, %0" ::"r"(p) : "memory");
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
#else
    (void)addr;
    (void)size;
#endif
}

struct L3L2OrchPayloadView {
    uint64_t gm_addr;
    uint64_t nbytes;
};

enum class L3L2EndpointErrorKind : uint32_t {
    NONE = 0,
    BAD_DESCRIPTOR = 1,
    OUT_OF_BOUNDS = 2,
    SIGNAL_TIMEOUT = 3,
    SIGNAL_PROTOCOL = 4,
};

struct L3L2EndpointError {
    L3L2EndpointErrorKind kind;
    const char *op;
    uint64_t region_id;
    uint64_t counter_addr;
    int32_t counter_operand;
    int32_t observed_counter;
    const char *message;
};

inline uint64_t l3_l2_orch_endpoint_now() {
#if defined(__aarch64__)
    uint64_t value = 0;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
#elif defined(__x86_64__)
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#else
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#endif
}

inline uint64_t l3_l2_orch_endpoint_timer_frequency_hz() {
#if defined(__aarch64__)
    uint64_t value = 0;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
#else
    return 1'000'000'000ULL;
#endif
}

inline uint64_t l3_l2_orch_endpoint_ticks_to_ns(uint64_t elapsed_ticks, uint64_t frequency_hz) {
    if (frequency_hz == 0) {
        return UINT64_MAX;
    }
    __uint128_t elapsed_ns = static_cast<__uint128_t>(elapsed_ticks) * 1'000'000'000ULL / frequency_hz;
    if (elapsed_ns > UINT64_MAX) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(elapsed_ns);
}

inline uint64_t l3_l2_orch_endpoint_elapsed_ns(uint64_t start_tick, uint64_t now_tick, uint64_t frequency_hz) {
#if defined(__aarch64__)
    return l3_l2_orch_endpoint_ticks_to_ns(now_tick - start_tick, frequency_hz);
#else
    (void)frequency_hz;
    return now_tick - start_tick;
#endif
}

class L3L2OrchEndpoint {
public:
    explicit L3L2OrchEndpoint(const L3L2OrchRegionDesc &desc) :
        desc_(desc) {
        if (l3_l2_orch_comm_validate_desc(desc_) != L3L2OrchCommValidationError::OK) {
            set_error(L3L2EndpointErrorKind::BAD_DESCRIPTOR, "init", desc_.region_id, 0, 0, "invalid descriptor");
        }
    }

    L3L2OrchEndpoint(const uint64_t *scalars, size_t scalar_count) {
        L3L2OrchCommValidationError error = L3L2OrchCommValidationError::OK;
        if (!l3_l2_orch_comm_decode_desc(scalars, scalar_count, &desc_, &error)) {
            uint64_t region_id = scalar_count > 1 && scalars != nullptr ? scalars[1] : 0;
            set_error(L3L2EndpointErrorKind::BAD_DESCRIPTOR, "init", region_id, 0, 0, "invalid descriptor scalars");
        }
    }

    const L3L2EndpointError &error() const { return error_; }

    const L3L2OrchRegionDesc &descriptor() const { return desc_; }

    bool counter_addr(uint64_t offset, uint64_t *out_addr) {
        if (out_addr != nullptr) {
            *out_addr = 0;
        }
        if (has_error()) {
            return false;
        }
        if (out_addr == nullptr) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, "counter_addr", desc_.region_id, 0, 0,
                "null counter address output"
            );
            return false;
        }
        if (l3_l2_orch_comm_add_overflows(desc_.counter_base, offset)) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, "counter_addr", desc_.region_id, 0, 0,
                "counter offset is out of bounds"
            );
            return false;
        }
        uint64_t addr = desc_.counter_base + offset;
        if (!validate_counter_addr_for_op("counter_addr", addr, 0, 0, "counter offset is out of bounds")) {
            return false;
        }
        *out_addr = addr;
        return true;
    }

    bool validate_counter_addr(uint64_t counter_addr) const {
        return l3_l2_orch_comm_validate_counter_addr(desc_, counter_addr) == L3L2OrchCommValidationError::OK;
    }

    bool payload_read(uint64_t offset, uint64_t nbytes, L3L2OrchPayloadView *out) {
        if (out != nullptr) {
            *out = L3L2OrchPayloadView{0, 0};
        }
        if (has_error()) {
            return false;
        }
        if (out == nullptr) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, "payload_read", desc_.region_id, 0, 0, "null payload view output"
            );
            return false;
        }
        if (!validate_payload_range("payload_read", offset, nbytes)) {
            return false;
        }
        uint64_t gm_addr = desc_.payload_base + offset;
        l3_l2_orch_endpoint_cache_invalidate_range(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(gm_addr)), static_cast<size_t>(nbytes)
        );
        *out = L3L2OrchPayloadView{gm_addr, nbytes};
        return true;
    }

    bool payload_write(uint64_t offset, const void *src, uint64_t nbytes) {
        if (has_error()) {
            return false;
        }
        if (src == nullptr) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, "payload_write", desc_.region_id, 0, 0, "null payload source"
            );
            return false;
        }
        if (!validate_payload_range("payload_write", offset, nbytes)) {
            return false;
        }
        void *dst = reinterpret_cast<void *>(static_cast<uintptr_t>(desc_.payload_base + offset));
        memcpy(dst, src, static_cast<size_t>(nbytes));
        l3_l2_orch_endpoint_cache_flush_range(dst, static_cast<size_t>(nbytes));
        return true;
    }

    bool signal_notify(uint64_t counter_addr, int32_t value, L3L2OrchNotifyOp op) {
        if (has_error()) {
            return false;
        }
        if (!validate_counter_addr_for_op("signal_notify", counter_addr, value, 0, "invalid counter address")) {
            return false;
        }
        if (!l3_l2_orch_comm_valid_notify_op(op)) {
            set_error(
                L3L2EndpointErrorKind::SIGNAL_PROTOCOL, "signal_notify", desc_.region_id, counter_addr, value,
                "invalid notify operation"
            );
            return false;
        }

        volatile int32_t *counter = counter_ptr(counter_addr);
        if (op == L3L2OrchNotifyOp::Set) {
            *counter = value;
        } else {
            l3_l2_orch_endpoint_cache_invalidate_range(
                reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter)
            );
            *counter = static_cast<int32_t>(*counter + value);
        }
        l3_l2_orch_endpoint_cache_flush_range(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter)
        );
        return true;
    }

    bool signal_test(uint64_t counter_addr, int32_t cmp_value, L3L2OrchWaitCmp cmp, L3L2OrchSignalTestResult *out) {
        if (out != nullptr) {
            *out = L3L2OrchSignalTestResult{false, 0};
        }
        if (has_error()) {
            return false;
        }
        if (out == nullptr) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, "signal_test", desc_.region_id, counter_addr, cmp_value,
                "null signal test output"
            );
            return false;
        }
        if (!validate_counter_addr_for_op("signal_test", counter_addr, cmp_value, 0, "invalid counter address")) {
            return false;
        }
        if (!l3_l2_orch_comm_valid_wait_cmp(cmp)) {
            set_error(
                L3L2EndpointErrorKind::SIGNAL_PROTOCOL, "signal_test", desc_.region_id, counter_addr, cmp_value,
                "invalid wait comparison"
            );
            return false;
        }
        int32_t observed = load_counter(counter_addr);
        *out = L3L2OrchSignalTestResult{l3_l2_orch_comm_compare_counter(observed, cmp_value, cmp), observed};
        return true;
    }

    bool
    signal_wait(uint64_t counter_addr, int32_t cmp_value, L3L2OrchWaitCmp cmp, uint64_t timeout, int32_t *observed) {
        if (observed != nullptr) {
            *observed = 0;
        }
        if (has_error()) {
            return false;
        }
        if (observed == nullptr) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, "signal_wait", desc_.region_id, counter_addr, cmp_value,
                "null signal wait output"
            );
            return false;
        }
        if (!validate_counter_addr_for_op("signal_wait", counter_addr, cmp_value, 0, "invalid counter address")) {
            return false;
        }
        if (!l3_l2_orch_comm_valid_wait_cmp(cmp)) {
            set_error(
                L3L2EndpointErrorKind::SIGNAL_PROTOCOL, "signal_wait", desc_.region_id, counter_addr, cmp_value,
                "invalid wait comparison"
            );
            return false;
        }

        uint64_t start = l3_l2_orch_endpoint_now();
        uint64_t frequency_hz = l3_l2_orch_endpoint_timer_frequency_hz();
        while (true) {
            int32_t current = load_counter(counter_addr);
            *observed = current;
            if (l3_l2_orch_comm_compare_counter(current, cmp_value, cmp)) {
                return true;
            }
            uint64_t now = l3_l2_orch_endpoint_now();
            if (timeout == 0 || l3_l2_orch_endpoint_elapsed_ns(start, now, frequency_hz) >= timeout) {
                set_error(
                    L3L2EndpointErrorKind::SIGNAL_TIMEOUT, "signal_wait", desc_.region_id, counter_addr, cmp_value,
                    current, "wait timed out"
                );
                return false;
            }
        }
    }

private:
    bool has_error() const { return error_.kind != L3L2EndpointErrorKind::NONE; }

    bool validate_payload_range(const char *op, uint64_t offset, uint64_t nbytes) {
        L3L2OrchCommValidationError error =
            l3_l2_orch_comm_validate_payload_bounds(offset, nbytes, desc_.payload_bytes);
        if (error == L3L2OrchCommValidationError::OK) {
            return true;
        }
        set_error(L3L2EndpointErrorKind::OUT_OF_BOUNDS, op, desc_.region_id, 0, 0, "payload range is out of bounds");
        return false;
    }

    bool validate_counter_addr_for_op(
        const char *op, uint64_t counter_addr, int32_t counter_operand, int32_t observed_counter, const char *message
    ) {
        if (l3_l2_orch_comm_validate_counter_addr(desc_, counter_addr) == L3L2OrchCommValidationError::OK) {
            return true;
        }
        set_error(
            L3L2EndpointErrorKind::OUT_OF_BOUNDS, op, desc_.region_id, counter_addr, counter_operand, observed_counter,
            message
        );
        return false;
    }

    volatile int32_t *counter_ptr(uint64_t counter_addr) {
        return reinterpret_cast<volatile int32_t *>(static_cast<uintptr_t>(counter_addr));
    }

    int32_t load_counter(uint64_t counter_addr) {
        volatile int32_t *counter = counter_ptr(counter_addr);
        l3_l2_orch_endpoint_cache_invalidate_range(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter)
        );
        return *counter;
    }

    void set_error(
        L3L2EndpointErrorKind kind, const char *op, uint64_t region_id, uint64_t counter_addr, int32_t counter_operand,
        const char *message
    ) {
        set_error(kind, op, region_id, counter_addr, counter_operand, 0, message);
    }

    void set_error(
        L3L2EndpointErrorKind kind, const char *op, uint64_t region_id, uint64_t counter_addr, int32_t counter_operand,
        int32_t observed_counter, const char *message
    ) {
        if (has_error()) {
            return;
        }
        error_ = L3L2EndpointError{kind, op, region_id, counter_addr, counter_operand, observed_counter, message};
    }

    L3L2OrchRegionDesc desc_{};
    L3L2EndpointError error_{L3L2EndpointErrorKind::NONE, "", 0, 0, 0, 0, ""};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_ORCH_ENDPOINT_H_
