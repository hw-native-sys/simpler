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

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../platform/include/common/host_api.h"
#include "../log/include/common/unified_log.h"
#include "../task_interface/task_args.h"

// Per-run bump over the DeviceRunner's retained per-pipeline-slot staging
// buffer. IN and INOUT tensors may reuse device bytes only when their producer
// supplies a matching nonzero host-content generation. Unknown inputs are
// always copied. The opaque population key is runner-owned, so it is
// invalidated with the buffer and released at runner finalization.
class HbgRetainedArgStaging {
public:
    static constexpr size_t kAlignment = 1024;

    bool begin(const HostApi *api, const ChipStorageTaskArgs *orch_args, const ArgDirection *signature, int sig_count) {
        if (api == nullptr || orch_args == nullptr) return false;
        api_ = api;
        orch_args_ = orch_args;
        signature_ = signature;
        sig_count_ = sig_count;
        offset_ = 0;

        size_t required = 0;
        for (int i = 0; i < orch_args->tensor_count(); ++i) {
            const ChipTensor t = orch_args->tensor(i);
            if (t.is_device_memory() || t.nbytes() == 0) continue;
            const uint64_t nbytes = t.nbytes();
            if (nbytes > std::numeric_limits<size_t>::max()) {
                LOG_ERROR("HBG arg staging tensor %d exceeds host size_t: bytes=%" PRIu64, i, nbytes);
                return false;
            }
            size_t aligned_bytes = 0;
            if (!checked_align_up(static_cast<size_t>(nbytes), &aligned_bytes) ||
                !checked_add(required, aligned_bytes, &required)) {
                LOG_ERROR("HBG arg staging size overflow at tensor %d: bytes=%" PRIu64, i, nbytes);
                return false;
            }
        }

        void *addr = nullptr;
        size_t size = 0;
        api->get_retained_temp_buffer(&addr, &size);
        if (required > size) {
            // Detach first so both the pointer and its validity key are gone
            // before the old allocation can be freed or a replacement fails.
            if (addr != nullptr) {
                api->set_retained_temp_buffer(nullptr, 0);
                api->device_free(addr);
            }
            addr = required == 0 ? nullptr : api->device_malloc(required);
            if (required != 0 && addr == nullptr) {
                LOG_ERROR("HBG arg staging buffer grow failed: required bytes %zu", required);
                base_ = nullptr;
                capacity_ = 0;
                return false;
            }
            api->set_retained_temp_buffer(addr, required);
            size = required;
        }
        base_ = addr;
        capacity_ = size;
        if (required != 0 && base_ == nullptr) {
            LOG_ERROR("HBG arg staging buffer is null for %zu required bytes", required);
            return false;
        }

        build_population_key();
        reuse_inputs_ =
            cacheable_tensors_ && api->retained_temp_metadata_matches(key_.data(), key_.size() * sizeof(key_[0]));

        // Consume the prior population before this run can copy or mutate any
        // retained slice. validate_runtime_impl republishes the key only after
        // execution and every required INOUT copy-back succeed. Thus any bind,
        // execution, or validation failure leaves the slot unpopulated.
        api->set_retained_temp_metadata(nullptr, 0);
        return true;
    }

    bool reuses_inputs() const { return reuse_inputs_; }

    bool copy_to_device_required(int tensor_index) const {
        const ChipTensor t = orch_args_->tensor(tensor_index);
        if (t.is_device_memory() || t.nbytes() == 0) return false;
        const ArgDirection direction = direction_at(tensor_index);
        if (direction == ArgDirection::OUT) return false;
        if (direction == ArgDirection::IN || direction == ArgDirection::INOUT) return !reuse_inputs_;
        return true;
    }

    void *acquire(size_t bytes) {
        size_t aligned = 0;
        if (!checked_align_up(offset_, &aligned) || base_ == nullptr || aligned > capacity_ ||
            bytes > capacity_ - aligned) {
            LOG_ERROR("HBG retained staging slice miss: bytes=%zu offset=%zu capacity=%zu", bytes, aligned, capacity_);
            return nullptr;
        }
        void *ptr = static_cast<char *>(base_) + aligned;
        offset_ = aligned + bytes;
        return ptr;
    }

    void finish_population(std::vector<uint64_t> *pending_key, bool *needs_copy_back) const {
        if (pending_key == nullptr || needs_copy_back == nullptr) return;
        if (cacheable_tensors_) {
            *pending_key = key_;
            *needs_copy_back = cacheable_inout_;
        } else {
            pending_key->clear();
            *needs_copy_back = false;
        }
    }

private:
    static bool checked_add(size_t lhs, size_t rhs, size_t *out) {
        if (lhs > std::numeric_limits<size_t>::max() - rhs) return false;
        *out = lhs + rhs;
        return true;
    }

    static bool checked_align_up(size_t value, size_t *out) {
        constexpr size_t kMask = kAlignment - 1;
        static_assert((kAlignment & kMask) == 0, "staging alignment must be a power of two");
        if (value > std::numeric_limits<size_t>::max() - kMask) return false;
        *out = (value + kMask) & ~kMask;
        return true;
    }

    ArgDirection direction_at(int tensor_index) const {
        if (signature_ == nullptr || tensor_index < 0 || tensor_index >= sig_count_) {
            return ArgDirection::SCALAR;
        }
        return signature_[tensor_index];
    }

    void build_population_key() {
        static constexpr uint64_t kKeySchema = 1;
        key_.clear();
        key_.reserve(2 + static_cast<size_t>(orch_args_->tensor_count()) * 5);
        key_.push_back(kKeySchema);
        key_.push_back(static_cast<uint64_t>(orch_args_->tensor_count()));
        cacheable_tensors_ = false;
        cacheable_inout_ = false;
        bool saw_host_read = false;
        bool all_host_reads_versioned = true;
        for (int i = 0; i < orch_args_->tensor_count(); ++i) {
            const ChipTensor t = orch_args_->tensor(i);
            const ArgDirection direction = direction_at(i);
            const bool host_read = !t.is_device_memory() && t.nbytes() != 0 &&
                                   (direction == ArgDirection::IN || direction == ArgDirection::INOUT);
            saw_host_read |= host_read;
            cacheable_inout_ |= host_read && direction == ArgDirection::INOUT;
            if (host_read && t.host_content_generation == 0) all_host_reads_versioned = false;
            key_.push_back(t.buffer.addr);
            key_.push_back(t.nbytes());
            key_.push_back(static_cast<uint64_t>(static_cast<int32_t>(direction)));
            key_.push_back(static_cast<uint64_t>(t.address_space));
            key_.push_back(host_read ? t.host_content_generation : 0);
        }
        cacheable_tensors_ = saw_host_read && all_host_reads_versioned;
    }

    const HostApi *api_{nullptr};
    const ChipStorageTaskArgs *orch_args_{nullptr};
    const ArgDirection *signature_{nullptr};
    int sig_count_{0};
    void *base_{nullptr};
    size_t capacity_{0};
    size_t offset_{0};
    bool cacheable_tensors_{false};
    bool cacheable_inout_{false};
    bool reuse_inputs_{false};
    std::vector<uint64_t> key_;
};
