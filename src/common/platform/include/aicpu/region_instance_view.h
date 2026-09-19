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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "common/region_instance_semantics.h"

struct RegionSignalTestResult {
    bool matched;
    int32_t observed;
};

struct RegionPayloadView {
    uint64_t local_addr;
    uint64_t nbytes;
};

enum class RegionViewErrorKind : uint32_t {
    NONE = 0,
    INVALID_VIEW = 1,
    OUT_OF_BOUNDS = 2,
    INVALID_ENUM = 3,
    TIMEOUT = 4,
    ISSUED_FAILURE = 5,
};

enum class RegionViewOp : uint32_t {
    CONSTRUCT = 1,
    PAYLOAD_READ = 2,
    PAYLOAD_WRITE = 3,
    NOTIFY = 4,
    TEST = 5,
    WAIT = 6,
    COUNTER_ADDR = 7,
};

struct RegionViewError {
    RegionViewErrorKind kind;
    RegionViewOp op;
    int32_t observed;
    char message[256];
};

struct PayloadLocalOps {
    bool invalidate(const void *addr, size_t nbytes) const {
        cache_invalidate_range(addr, nbytes);
        return true;
    }

    void flush(const void *addr, size_t nbytes) const { cache_flush_range(addr, nbytes); }

    bool copy_bytes(void *dst, const void *src, size_t nbytes) const {
        if (dst == nullptr || src == nullptr) {
            return false;
        }
        memcpy(dst, src, nbytes);
        return true;
    }
};

struct CounterLocalOps {
    bool load_i32(uint64_t addr, int32_t &out) const {
        volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(static_cast<uintptr_t>(addr));
        cache_invalidate_range(reinterpret_cast<const void *>(static_cast<uintptr_t>(addr)), sizeof(int32_t));
        out = *ptr;
        return true;
    }

    bool store_i32(uint64_t addr, int32_t value) const {
        volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(static_cast<uintptr_t>(addr));
        *ptr = value;
        cache_flush_range(reinterpret_cast<const void *>(static_cast<uintptr_t>(addr)), sizeof(int32_t));
        return true;
    }
};

struct LocalMemoryOps : PayloadLocalOps,
                        CounterLocalOps {};

struct RegionViewState {
    bool sticky_failed;
    RegionViewError last_error;
};

inline void region_view_copy_message(char *dst, size_t dst_size, const char *message) {
    if (dst == nullptr || dst_size == 0) {
        return;
    }
    const char *src = message == nullptr ? "" : message;
    size_t n = strnlen(src, dst_size - 1);
    memcpy(dst, src, n);
    dst[n] = '\0';
}

template <typename Ops = LocalMemoryOps>
class RegionInstanceViewImpl {
public:
    class PayloadPart {
    public:
        template <typename View>
        bool read(uint64_t offset, uint64_t nbytes, View &out) {
            RegionPayloadView view{};
            bool ok = view_->payload_read(offset, nbytes, view);
            out.local_addr = view.local_addr;
            out.nbytes = view.nbytes;
            return ok;
        }

        bool write(uint64_t offset, const void *src, uint64_t nbytes) {
            return view_->payload_write(offset, src, nbytes);
        }

        RegionPartLocalSpan span() const { return view_->payload_span(); }

    private:
        friend class RegionInstanceViewImpl<Ops>;
        explicit PayloadPart(RegionInstanceViewImpl *view) :
            view_(view) {}
        RegionInstanceViewImpl *view_;
    };

    class CounterPart {
    public:
        bool addr(uint64_t offset, uint64_t &out) { return view_->counter_addr(offset, out); }

        bool notify(uint64_t offset, int32_t value, RegionNotifyOp op) { return view_->notify(offset, value, op); }

        template <typename Sample>
        bool test(uint64_t offset, int32_t cmp_value, RegionWaitCmp cmp, Sample &out) {
            RegionSignalTestResult sample{};
            bool ok = view_->test(offset, cmp_value, cmp, sample);
            out.matched = sample.matched;
            out.observed = sample.observed;
            return ok;
        }

        bool wait(uint64_t offset, int32_t cmp_value, RegionWaitCmp cmp, uint64_t timeout_ns, int32_t &observed) {
            return view_->wait(offset, cmp_value, cmp, timeout_ns, observed);
        }

        bool contains_addr(uint64_t counter_addr) const { return view_->counter_contains(counter_addr); }

        RegionPartLocalSpan span() const { return view_->counter_span(); }

    private:
        friend class RegionInstanceViewImpl<Ops>;
        explicit CounterPart(RegionInstanceViewImpl *view) :
            view_(view) {}
        RegionInstanceViewImpl *view_;
    };

    RegionInstanceViewImpl() {
        mark_sticky(RegionViewErrorKind::INVALID_VIEW, RegionViewOp::CONSTRUCT, 0, "uninitialized view");
    }

    RegionInstanceViewImpl(RegionPartLocalSpan payload, RegionPartLocalSpan counter) {
        if (!region_validate_independent_spans(payload, counter)) {
            mark_sticky(
                RegionViewErrorKind::INVALID_VIEW, RegionViewOp::CONSTRUCT, 0, "invalid independent local spans"
            );
            return;
        }
        payload_span_ = payload;
        counter_span_ = counter;
    }

    bool failed() const { return state_.sticky_failed; }

    const RegionViewError &error() const { return state_.last_error; }

    RegionPartLocalSpan payload_span() const { return payload_span_; }

    RegionPartLocalSpan counter_span() const { return counter_span_; }

    PayloadPart payload() { return PayloadPart(this); }

    CounterPart counter() { return CounterPart(this); }

    bool payload_read(uint64_t offset, uint64_t nbytes, RegionPayloadView &out) {
        out = RegionPayloadView{0, 0};
        if (state_.sticky_failed) {
            return false;
        }
        if (!validate_payload_range(RegionViewOp::PAYLOAD_READ, offset, nbytes)) {
            return false;
        }
        uint64_t addr = payload_span_.base + offset;
        if (!ops_.invalidate(
                reinterpret_cast<const void *>(static_cast<uintptr_t>(addr)), static_cast<size_t>(nbytes)
            )) {
            mark_sticky(
                RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::PAYLOAD_READ, 0, "payload invalidate failed"
            );
            return false;
        }
        out = RegionPayloadView{addr, nbytes};
        return true;
    }

    bool payload_write(uint64_t offset, const void *src, uint64_t nbytes) {
        if (state_.sticky_failed) {
            return false;
        }
        if (src == nullptr) {
            record(RegionViewErrorKind::OUT_OF_BOUNDS, RegionViewOp::PAYLOAD_WRITE, 0, false, "null payload source");
            return false;
        }
        if (!validate_payload_range(RegionViewOp::PAYLOAD_WRITE, offset, nbytes)) {
            return false;
        }
        void *dst = reinterpret_cast<void *>(static_cast<uintptr_t>(payload_span_.base + offset));
        if (!ops_.copy_bytes(dst, src, static_cast<size_t>(nbytes))) {
            mark_sticky(RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::PAYLOAD_WRITE, 0, "payload copy failed");
            return false;
        }
        ops_.flush(dst, static_cast<size_t>(nbytes));
        return true;
    }

    bool counter_addr(uint64_t offset, uint64_t &out) {
        out = 0;
        if (state_.sticky_failed) {
            return false;
        }
        uint64_t addr = 0;
        if (!resolve_counter_addr(RegionViewOp::COUNTER_ADDR, offset, addr, "counter offset is out of bounds")) {
            return false;
        }
        out = addr;
        return true;
    }

    bool counter_contains(uint64_t counter_addr) const {
        if (state_.sticky_failed) {
            return false;
        }
        return counter_addr_in_span(counter_addr);
    }

    bool notify(uint64_t offset, int32_t value, RegionNotifyOp op) {
        if (state_.sticky_failed) {
            return false;
        }
        uint64_t addr = 0;
        if (!resolve_counter_addr(RegionViewOp::NOTIFY, offset, addr, "invalid counter address")) {
            return false;
        }
        if (!region_valid_notify_op(op)) {
            record(RegionViewErrorKind::INVALID_ENUM, RegionViewOp::NOTIFY, 0, false, "invalid notify operation");
            return false;
        }
        if (op == RegionNotifyOp::Set) {
            if (!ops_.store_i32(addr, value)) {
                mark_sticky(RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::NOTIFY, 0, "counter store failed");
                return false;
            }
            return true;
        }
        int32_t observed = 0;
        if (!ops_.load_i32(addr, observed)) {
            mark_sticky(RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::NOTIFY, 0, "counter load failed");
            return false;
        }
        uint32_t wrapped = static_cast<uint32_t>(observed) + static_cast<uint32_t>(value);
        int32_t next = 0;
        memcpy(&next, &wrapped, sizeof(next));
        if (!ops_.store_i32(addr, next)) {
            mark_sticky(RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::NOTIFY, 0, "counter store failed");
            return false;
        }
        return true;
    }

    bool test(uint64_t offset, int32_t cmp_value, RegionWaitCmp cmp, RegionSignalTestResult &out) {
        out = RegionSignalTestResult{false, 0};
        if (state_.sticky_failed) {
            return false;
        }
        uint64_t addr = 0;
        if (!resolve_counter_addr(RegionViewOp::TEST, offset, addr, "invalid counter address")) {
            return false;
        }
        if (!region_valid_wait_cmp(cmp)) {
            record(RegionViewErrorKind::INVALID_ENUM, RegionViewOp::TEST, 0, false, "invalid wait comparison");
            return false;
        }
        int32_t observed = 0;
        if (!ops_.load_i32(addr, observed)) {
            mark_sticky(RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::TEST, 0, "counter load failed");
            return false;
        }
        out = RegionSignalTestResult{region_compare_counter(observed, cmp_value, cmp), observed};
        return true;
    }

    bool wait(uint64_t offset, int32_t cmp_value, RegionWaitCmp cmp, uint64_t timeout_ns, int32_t &observed) {
        observed = 0;
        if (state_.sticky_failed) {
            return false;
        }
        uint64_t addr = 0;
        if (!resolve_counter_addr(RegionViewOp::WAIT, offset, addr, "invalid counter address")) {
            return false;
        }
        if (!region_valid_wait_cmp(cmp)) {
            record(RegionViewErrorKind::INVALID_ENUM, RegionViewOp::WAIT, 0, false, "invalid wait comparison");
            return false;
        }
        uint64_t start = device_time_now_ticks();
        uint64_t frequency_hz = device_time_frequency_hz();
        while (true) {
            int32_t current = 0;
            if (!ops_.load_i32(addr, current)) {
                mark_sticky(RegionViewErrorKind::ISSUED_FAILURE, RegionViewOp::WAIT, 0, "counter load failed");
                return false;
            }
            observed = current;
            if (region_compare_counter(current, cmp_value, cmp)) {
                return true;
            }
            uint64_t now = device_time_now_ticks();
            if (timeout_ns == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout_ns) {
                record(RegionViewErrorKind::TIMEOUT, RegionViewOp::WAIT, current, false, "wait timed out");
                return false;
            }
        }
    }

private:
    void record(RegionViewErrorKind kind, RegionViewOp op, int32_t observed, bool sticky, const char *message) {
        if (state_.sticky_failed) {
            return;
        }
        state_.last_error = RegionViewError{kind, op, observed, ""};
        region_view_copy_message(state_.last_error.message, sizeof(state_.last_error.message), message);
        state_.sticky_failed = sticky;
    }

    void mark_sticky(RegionViewErrorKind kind, RegionViewOp op, int32_t observed, const char *message) {
        record(kind, op, observed, true, message);
    }

    bool validate_payload_range(RegionViewOp op, uint64_t offset, uint64_t nbytes) {
        if (!region_validate_payload_range(offset, nbytes, payload_span_.logical_bytes)) {
            record(RegionViewErrorKind::OUT_OF_BOUNDS, op, 0, false, "payload range is out of bounds");
            return false;
        }
        return true;
    }

    bool counter_addr_in_span(uint64_t counter_addr) const {
        return region_counter_addr_in_span(counter_span_, counter_addr);
    }

    bool resolve_counter_addr(RegionViewOp op, uint64_t offset, uint64_t &addr, const char *message) {
        addr = 0;
        if ((offset % REGION_COUNTER_LOGICAL_ALIGNMENT) != 0 || region_add_overflows(counter_span_.base, offset)) {
            record(RegionViewErrorKind::OUT_OF_BOUNDS, op, 0, false, message);
            return false;
        }
        uint64_t resolved = counter_span_.base + offset;
        if (!counter_addr_in_span(resolved)) {
            record(RegionViewErrorKind::OUT_OF_BOUNDS, op, 0, false, message);
            return false;
        }
        addr = resolved;
        return true;
    }

    RegionPartLocalSpan payload_span_{};
    RegionPartLocalSpan counter_span_{};
    Ops ops_{};
    RegionViewState state_{};
};

using RegionInstanceView = RegionInstanceViewImpl<LocalMemoryOps>;
