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

#include <type_traits>

#if defined(__CCE_AICORE__)
#include <pto/pto-inst.hpp>
#endif

#include "async_completion_api.h"
#include "platform_comm/ccu/ccu_device_access.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

class CcuCallArgs {
public:
    __aicore__ void add_tensor(const Tensor &tensor) {
        using namespace simpler::ccu::device;
        if (!valid_) return;
        if (values_.tensor_count == MAX_TENSORS) return reject_tensor(1, tensor);
        if (tensor.dtype != DataType::FLOAT32) return reject_tensor(2, tensor);
        if (tensor.address_space != AddressSpace::DEVICE) return reject_tensor(3, tensor);
        if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS) return reject_tensor(4, tensor);
        uint64_t elements = 1;
        for (uint32_t i = tensor.ndims; i > 0; --i) {
            const auto dim = i - 1;
            if (tensor.shapes[dim] == 0) return reject_tensor(5, tensor);
            if (tensor.shapes[dim] != 1 && tensor.strides[dim] != elements) return reject_tensor(6, tensor);
            if (elements > (UINT64_MAX / 4) / tensor.shapes[dim]) return reject_tensor(7, tensor);
            elements *= tensor.shapes[dim];
        }
        if (elements > UINT64_MAX / 4) return reject_tensor(7, tensor);
        if (tensor.start_offset > UINT64_MAX / 4) return reject_tensor(8, tensor);
        if (tensor.buffer.size > UINT64_MAX - tensor.buffer.addr) return reject_tensor(9, tensor);
        if (tensor.buffer.addr % 4) return reject_tensor(10, tensor);
        if (!fits(tensor.start_offset * 4, elements * 4, tensor.buffer.size)) return reject_tensor(11, tensor);
        values_.tensors[values_.tensor_count++] = {tensor.buffer.addr + tensor.start_offset * 4, elements * 4, 0};
    }

    template <typename T>
    __aicore__ void add_scalar(T value) {
        using namespace simpler::ccu::device;
        static_assert(
            (std::is_integral_v<T> || std::is_floating_point_v<T>) && (sizeof(T) == 4 || sizeof(T) == 8),
            "CCU scalar must be 32-bit or 64-bit arithmetic; pass memory as a Tensor"
        );
        if (!valid_) return;
        if (values_.scalar_count == MAX_SCALARS) {
            failure_reason_ = 12U | (values_.scalar_count << 4);
            valid_ = false;
            return;
        }
        auto &slot = values_.scalars[values_.scalar_count++];
        slot.width = sizeof(T) * 8;
        if constexpr (std::is_integral_v<T>) {
            slot.kind = std::is_signed_v<T> ? ScalarKind::SIGNED : ScalarKind::UNSIGNED;
            slot.bits = static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(value));
        } else {
            slot.kind = ScalarKind::FLOAT;
            slot.bits = 0;
            const auto *src = reinterpret_cast<const uint8_t *>(&value);
            auto *dst = reinterpret_cast<uint8_t *>(&slot.bits);
            for (uint32_t i = 0; i < sizeof(T); ++i)
                dst[i] = src[i];
        }
    }

    __aicore__ bool valid() const { return valid_; }
    __aicore__ uint32_t failure_reason() const { return failure_reason_; }
    __aicore__ const simpler::ccu::device::Arguments &values() const { return values_; }

private:
    __aicore__ void reject_tensor(uint32_t reason, const Tensor &tensor) {
        // First failure: reason[3:0], parameter index[8:4], dtype[16:9],
        // address space[24:17], ndims[31:25] (saturated at 127).
        failure_reason_ = reason | (values_.tensor_count << 4) | (static_cast<uint32_t>(tensor.dtype) << 9) |
                          (static_cast<uint32_t>(tensor.address_space) << 17) |
                          ((tensor.ndims < 127 ? tensor.ndims : 127U) << 25);
        valid_ = false;
    }

    simpler::ccu::device::Arguments values_{};
    bool valid_{true};
    uint32_t failure_reason_{};
};
static_assert(sizeof(CcuCallArgs) == 656, "CcuCallArgs diagnostic must fit its existing padding");

struct CcuCallDescriptor {
    const __gm__ CommContext *domain_ctx;
    uint32_t ccu_id;
    CcuCallArgs parameters;
};

inline __aicore__ CcuCallDescriptor
CcuCall(const __gm__ CommContext *domain_ctx, uint32_t ccu_id, const CcuCallArgs &parameters) {
    return {domain_ctx, ccu_id, parameters};
}

namespace simpler::ccu::detail {

inline __aicore__ bool fail(AsyncCtx &task, int32_t code) {
    if (task.completion_error_code && *task.completion_error_code == SIMPLER_ERROR_NONE)
        pto2::detail::defer_error(task, code);
    pto2::detail::defer_flush(task);
    return false;
}

inline __aicore__ bool reject_accept(
    AsyncCtx &task, const CcuCallDescriptor &call, uint32_t stage, CcuStatus status = CcuStatus::INVALID_ARGUMENT,
    uint32_t details = 0
) {
    if (task.task_token.is_valid() && task.completion_error_code && *task.completion_error_code == SIMPLER_ERROR_NONE &&
        task.completion_count && task.completion_entries) {
        const uint32_t index = *task.completion_count;
        if (index < task.completion_capacity && index < MAX_COMPLETIONS_PER_TASK) {
            auto &entry = task.completion_entries[index];
            // Flags: parameters valid, task token valid, slab present, domain present.
            const uint32_t flags =
                static_cast<uint32_t>(call.parameters.valid()) | 2U | 4U | (call.domain_ctx ? 8U : 0U);
            const uint32_t ranks = call.domain_ctx ? call.domain_ctx->rankNum : 0;
            const uint32_t rank = call.domain_ctx ? call.domain_ctx->rankId : UINT32_MAX;
            entry.addr = static_cast<uint32_t>(status) | (uint64_t{stage & 0xff} << 32) | (uint64_t{flags} << 40) |
                         (uint64_t{ranks < UINT16_MAX ? ranks : UINT16_MAX} << 48);
            entry.backend_cookie = static_cast<uint32_t>(task.callable_id) | (uint64_t{rank} << 32);
            entry.expected_value = details;
            entry.engine = COMPLETION_ENGINE_CCU;
            entry.completion_type = COMPLETION_TYPE_CCU_REQUEST;
            entry._pad = CCU_ACCEPT_DIAGNOSTIC_MAGIC;
            pto2::detail::defer_flush_range(&entry, sizeof(entry));
        }
    }
    return fail(task, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

template <typename Io>
inline __aicore__ uint32_t diagnose_instance(
    __gm__ device::DomainDirectory *directory, uint64_t available, int32_t callable_id, uint32_t rank, uint32_t ranks
) {
    using namespace device;
    // Failure bits: directory pointer/version/header/generation/admission/rank/ranks/bounds (0..7),
    // missing/duplicate callable (8..9), workspace pointer/valid/owner/domain/generation/rank/ranks (10..16),
    // platform context (17). Pointer-dependent checks stop at their containing image bounds.
    if (!directory || reinterpret_cast<uint64_t>(directory) % alignof(DomainDirectory) ||
        available < sizeof(DomainDirectory))
        return 1U;
    Io::sync(directory, sizeof(*directory));
    uint32_t failures = (uint32_t{directory->version != PROTOCOL_VERSION} << 1) |
                        (uint32_t{directory->header_bytes != sizeof(DomainDirectory)} << 2) |
                        (uint32_t{directory->generation == 0} << 3) |
                        (uint32_t{directory->admission != Admission::OPEN} << 4) |
                        (uint32_t{directory->rank != rank} << 5) | (uint32_t{directory->rank_count != ranks} << 6);
    if (directory->image_bytes > available || directory->entries_offset < sizeof(DomainDirectory) ||
        directory->entries_offset % alignof(InstanceEntry) ||
        !fits(
            directory->entries_offset, uint64_t{directory->entry_count} * sizeof(InstanceEntry), directory->image_bytes
        ))
        return failures | (1U << 7);
    auto *entries = reinterpret_cast<__gm__ InstanceEntry *>(
        reinterpret_cast<__gm__ uint8_t *>(directory) + directory->entries_offset
    );
    bool found = false;
    for (uint32_t i = 0; i < directory->entry_count; ++i) {
        if (entries[i].callable_id != callable_id) continue;
        if (found) failures |= 1U << 9;
        found = true;
        auto *w = reinterpret_cast<__gm__ Workspace *>(entries[i].image_addr);
        if (!w || entries[i].image_addr % alignof(Workspace) || entries[i].image_bytes < sizeof(Workspace))
            return failures | (1U << 10);
        Io::sync(w, sizeof(*w));
        failures |= (uint32_t{!Protocol<Io>::valid(w, entries[i].image_bytes)} << 11) |
                    (uint32_t{w->domain_owner != directory->domain_owner} << 12) |
                    (uint32_t{w->domain_id != directory->domain_id} << 13) |
                    (uint32_t{w->generation != directory->generation} << 14) | (uint32_t{w->rank != rank} << 15) |
                    (uint32_t{w->rank_count != ranks} << 16);
        if (w->platform_context && w->image_bytes >= sizeof(Workspace) && w->image_bytes <= entries[i].image_bytes &&
            !Protocol<Io>::platform({w, 0, 0}))
            failures |= 1U << 17;
    }
    return failures | (found ? 0U : 1U << 8);
}

template <typename Io = device::Transport>
inline __aicore__ bool accept(AsyncCtx &task, const CcuCallDescriptor &call, device::Reference &ref) {
    if (!task.task_token.is_valid() || !task.completion_error_code || !task.completion_count ||
        !task.completion_entries || *task.completion_error_code != SIMPLER_ERROR_NONE || !call.domain_ctx ||
        !call.parameters.valid())
        return reject_accept(task, call, 1, CcuStatus::INVALID_ARGUMENT, call.parameters.failure_reason());
    const auto &workspace = comm_workspace<CommWorkspaceKind::CCU>(*call.domain_ctx);
    auto *directory = reinterpret_cast<__gm__ device::DomainDirectory *>(workspace.addr);
    uint64_t image_bytes = 0;
    auto *w = device::Protocol<Io>::instance(
        directory, workspace.size, task.callable_id, call.domain_ctx->rankId, call.domain_ctx->rankNum, image_bytes
    );
    if (!w) {
        const uint32_t details = diagnose_instance<Io>(
            directory, workspace.size, task.callable_id, call.domain_ctx->rankId, call.domain_ctx->rankNum
        );
        return reject_accept(task, call, 2, CcuStatus::INVALID_ARGUMENT, details);
    }
    if (!device::Protocol<Io>::resolve(w, image_bytes, task.callable_id, call.ccu_id, ref))
        return reject_accept(task, call, 3);
    CcuStatus status{};
    if (!device::Protocol<Io>::reserve(ref, call.parameters.values(), &status))
        return reject_accept(task, call, 4, status);
    return true;
}

template <typename Io = device::Transport>
inline __aicore__ bool send(AsyncCtx &task, const CcuCallDescriptor &call) {
    using Protocol = device::Protocol<Io>;
    // Never trigger a request that cannot be handed to the completion owner.
    if (!task.completion_count || *task.completion_count >= task.completion_capacity ||
        *task.completion_count >= MAX_COMPLETIONS_PER_TASK)
        return fail(task, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    device::Reference ref;
    if (!accept<Io>(task, call, ref)) return false;
    const CompletionToken token{
        reinterpret_cast<uint64_t>(ref.workspace), ref.binding_index, COMPLETION_ENGINE_CCU,
        COMPLETION_TYPE_CCU_REQUEST, ref.sequence
    };
    if (!register_completion_condition(task, token)) {
        Protocol::cancel(ref);
        return fail(task, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    }
    pto2::detail::defer_flush(task);
    if (!Protocol::submit(ref)) {
        // submit returning false guarantees the doorbell was not written.
        Protocol::cancel(ref);
        *task.completion_count = *task.completion_count - 1;
        return fail(task, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    }
    return true;
}

template <typename Io = device::Transport>
inline __aicore__ bool wait(AsyncCtx &task, const CcuCallDescriptor &call) {
    using Protocol = device::Protocol<Io>;
    device::Reference ref;
    if (!accept<Io>(task, call, ref)) return false;
    if (!Protocol::submit(ref)) {
        Protocol::cancel(ref);
        return fail(task, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    }
    device::Outcome outcome;
    do {
        outcome = Protocol::poll(ref);
    } while (outcome == device::Outcome::PENDING);
    if (outcome == device::Outcome::UNCONFIRMED) return fail(task, SIMPLER_ERROR_CCU_UNCONFIRMED);
    // Completion is the platform's data-visibility fence. Drop any AIV cached
    // Tensor data before returning to user computation, then reclaim the slot.
    if (!Protocol::synchronize_arguments(ref)) return fail(task, SIMPLER_ERROR_CCU_UNCONFIRMED);
    if (!Protocol::retire(ref)) return fail(task, SIMPLER_ERROR_CCU_UNCONFIRMED);
    return outcome == device::Outcome::SUCCEEDED || fail(task, SIMPLER_ERROR_CCU_FAILED);
}

}  // namespace simpler::ccu::detail

inline __aicore__ bool send_request_entry(AsyncCtx &task, const CcuCallDescriptor &call) {
    return simpler::ccu::detail::send(task, call);
}

inline __aicore__ bool wait_request_entry(AsyncCtx &task, const CcuCallDescriptor &call) {
    return simpler::ccu::detail::wait(task, call);
}
