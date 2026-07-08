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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "common/l3_l2_orch_comm.h"

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

enum class L3L2EndpointOp : uint32_t {
    INIT = 1,
    COUNTER_ADDR = 2,
    PAYLOAD_READ = 3,
    PAYLOAD_WRITE = 4,
    SIGNAL_NOTIFY = 5,
    SIGNAL_TEST = 6,
    SIGNAL_WAIT = 7,
};

inline const char *l3_l2_endpoint_op_to_string(L3L2EndpointOp op) {
    switch (op) {
    case L3L2EndpointOp::INIT:
        return "init";
    case L3L2EndpointOp::COUNTER_ADDR:
        return "counter_addr";
    case L3L2EndpointOp::PAYLOAD_READ:
        return "payload_read";
    case L3L2EndpointOp::PAYLOAD_WRITE:
        return "payload_write";
    case L3L2EndpointOp::SIGNAL_NOTIFY:
        return "signal_notify";
    case L3L2EndpointOp::SIGNAL_TEST:
        return "signal_test";
    case L3L2EndpointOp::SIGNAL_WAIT:
        return "signal_wait";
    default:
        return "unknown";
    }
}

struct L3L2EndpointError {
    L3L2EndpointErrorKind kind;
    L3L2EndpointOp op;
    uint64_t region_id;
    uint64_t counter_addr;
    int32_t counter_operand;
    int32_t observed_counter;
    char message[256];
};

class L3L2OrchEndpoint {
public:
    explicit L3L2OrchEndpoint(const L3L2OrchRegionDesc &desc) :
        desc_(desc) {
        if (l3_l2_orch_comm::validate_desc(desc_) != L3L2OrchCommValidationError::OK) {
            set_error(
                L3L2EndpointErrorKind::BAD_DESCRIPTOR, L3L2EndpointOp::INIT, desc_.region_id, 0, 0, "invalid descriptor"
            );
        }
    }

    L3L2OrchEndpoint(const uint64_t *scalars, size_t scalar_count) {
        L3L2OrchCommValidationError error = L3L2OrchCommValidationError::OK;
        if (!l3_l2_orch_comm::decode_desc(scalars, scalar_count, &desc_, &error)) {
            uint64_t region_id = scalar_count > 1 && scalars != nullptr ? scalars[1] : 0;
            set_error(
                L3L2EndpointErrorKind::BAD_DESCRIPTOR, L3L2EndpointOp::INIT, region_id, 0, 0,
                "invalid descriptor scalars"
            );
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
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, L3L2EndpointOp::COUNTER_ADDR, desc_.region_id, 0, 0,
                "null counter address output"
            );
            return false;
        }
        if (l3_l2_orch_comm_add_overflows(desc_.counter_base, offset)) {
            set_error(
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, L3L2EndpointOp::COUNTER_ADDR, desc_.region_id, 0, 0,
                "counter offset is out of bounds"
            );
            return false;
        }
        uint64_t addr = desc_.counter_base + offset;
        if (!validate_counter_addr_for_op(
                L3L2EndpointOp::COUNTER_ADDR, addr, 0, 0, "counter offset is out of bounds"
            )) {
            return false;
        }
        *out_addr = addr;
        return true;
    }

    bool validate_counter_addr(uint64_t counter_addr) const {
        return l3_l2_orch_comm::validate_counter_addr(desc_, counter_addr) == L3L2OrchCommValidationError::OK;
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
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, L3L2EndpointOp::PAYLOAD_READ, desc_.region_id, 0, 0,
                "null payload view output"
            );
            return false;
        }
        if (!validate_payload_range(L3L2EndpointOp::PAYLOAD_READ, offset, nbytes)) {
            return false;
        }
        uint64_t gm_addr = desc_.payload_base + offset;
        cache_invalidate_range(
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
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, L3L2EndpointOp::PAYLOAD_WRITE, desc_.region_id, 0, 0,
                "null payload source"
            );
            return false;
        }
        if (!validate_payload_range(L3L2EndpointOp::PAYLOAD_WRITE, offset, nbytes)) {
            return false;
        }
        void *dst = reinterpret_cast<void *>(static_cast<uintptr_t>(desc_.payload_base + offset));
        memcpy(dst, src, static_cast<size_t>(nbytes));
        cache_flush_range(dst, static_cast<size_t>(nbytes));
        return true;
    }

    bool signal_notify(uint64_t counter_addr, int32_t value, L3L2OrchNotifyOp op) {
        if (has_error()) {
            return false;
        }
        if (!validate_counter_addr_for_op(
                L3L2EndpointOp::SIGNAL_NOTIFY, counter_addr, value, 0, "invalid counter address"
            )) {
            return false;
        }
        if (!l3_l2_orch_comm::valid_notify_op(op)) {
            set_error(
                L3L2EndpointErrorKind::SIGNAL_PROTOCOL, L3L2EndpointOp::SIGNAL_NOTIFY, desc_.region_id, counter_addr,
                value, "invalid notify operation"
            );
            return false;
        }

        volatile int32_t *counter = counter_ptr(counter_addr);
        if (op == L3L2OrchNotifyOp::Set) {
            *counter = value;
        } else {
            cache_invalidate_range(
                reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter)
            );
            *counter = static_cast<int32_t>(*counter + value);
        }
        cache_flush_range(reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter));
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
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, L3L2EndpointOp::SIGNAL_TEST, desc_.region_id, counter_addr,
                cmp_value, "null signal test output"
            );
            return false;
        }
        if (!validate_counter_addr_for_op(
                L3L2EndpointOp::SIGNAL_TEST, counter_addr, cmp_value, 0, "invalid counter address"
            )) {
            return false;
        }
        if (!l3_l2_orch_comm::valid_wait_cmp(cmp)) {
            set_error(
                L3L2EndpointErrorKind::SIGNAL_PROTOCOL, L3L2EndpointOp::SIGNAL_TEST, desc_.region_id, counter_addr,
                cmp_value, "invalid wait comparison"
            );
            return false;
        }
        int32_t observed = load_counter(counter_addr);
        *out = L3L2OrchSignalTestResult{l3_l2_orch_comm::compare_counter(observed, cmp_value, cmp), observed};
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
                L3L2EndpointErrorKind::OUT_OF_BOUNDS, L3L2EndpointOp::SIGNAL_WAIT, desc_.region_id, counter_addr,
                cmp_value, "null signal wait output"
            );
            return false;
        }
        if (!validate_counter_addr_for_op(
                L3L2EndpointOp::SIGNAL_WAIT, counter_addr, cmp_value, 0, "invalid counter address"
            )) {
            return false;
        }
        if (!l3_l2_orch_comm::valid_wait_cmp(cmp)) {
            set_error(
                L3L2EndpointErrorKind::SIGNAL_PROTOCOL, L3L2EndpointOp::SIGNAL_WAIT, desc_.region_id, counter_addr,
                cmp_value, "invalid wait comparison"
            );
            return false;
        }

        uint64_t start = device_time_now_ticks();
        uint64_t frequency_hz = device_time_frequency_hz();
        while (true) {
            int32_t current = load_counter(counter_addr);
            *observed = current;
            if (l3_l2_orch_comm::compare_counter(current, cmp_value, cmp)) {
                return true;
            }
            uint64_t now = device_time_now_ticks();
            if (timeout == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout) {
                set_error(
                    L3L2EndpointErrorKind::SIGNAL_TIMEOUT, L3L2EndpointOp::SIGNAL_WAIT, desc_.region_id, counter_addr,
                    cmp_value, current, "wait timed out"
                );
                return false;
            }
        }
    }

private:
    bool has_error() const { return error_.kind != L3L2EndpointErrorKind::NONE; }

    bool validate_payload_range(L3L2EndpointOp op, uint64_t offset, uint64_t nbytes) {
        L3L2OrchCommValidationError error =
            l3_l2_orch_comm::validate_payload_bounds(offset, nbytes, desc_.payload_bytes);
        if (error == L3L2OrchCommValidationError::OK) {
            return true;
        }
        set_error(L3L2EndpointErrorKind::OUT_OF_BOUNDS, op, desc_.region_id, 0, 0, "payload range is out of bounds");
        return false;
    }

    bool validate_counter_addr_for_op(
        L3L2EndpointOp op, uint64_t counter_addr, int32_t counter_operand, int32_t observed_counter, const char *message
    ) {
        if (l3_l2_orch_comm::validate_counter_addr(desc_, counter_addr) == L3L2OrchCommValidationError::OK) {
            return true;
        }
        set_error(
            L3L2EndpointErrorKind::OUT_OF_BOUNDS, op, desc_.region_id, counter_addr, counter_operand, observed_counter,
            message
        );
        return false;
    }

    static volatile int32_t *counter_ptr(uint64_t counter_addr) {
        return reinterpret_cast<volatile int32_t *>(static_cast<uintptr_t>(counter_addr));
    }

    static int32_t load_counter(uint64_t counter_addr) {
        volatile int32_t *counter = counter_ptr(counter_addr);
        cache_invalidate_range(reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter));
        return *counter;
    }

    void set_error(
        L3L2EndpointErrorKind kind, L3L2EndpointOp op, uint64_t region_id, uint64_t counter_addr,
        int32_t counter_operand, const char *message
    ) {
        set_error(kind, op, region_id, counter_addr, counter_operand, 0, message);
    }

    void set_error(
        L3L2EndpointErrorKind kind, L3L2EndpointOp op, uint64_t region_id, uint64_t counter_addr,
        int32_t counter_operand, int32_t observed_counter, const char *message
    ) {
        if (has_error()) {
            return;
        }
        error_ = L3L2EndpointError{kind, op, region_id, counter_addr, counter_operand, observed_counter, ""};
        l3_l2_orch_comm::copy_error_message(error_.message, sizeof(error_.message), message);
    }

    L3L2OrchRegionDesc desc_{};
    L3L2EndpointError error_{L3L2EndpointErrorKind::NONE, L3L2EndpointOp::INIT, 0, 0, 0, 0, ""};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_ORCH_ENDPOINT_H_
