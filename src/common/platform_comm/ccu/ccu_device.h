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

#include "platform_comm/ccu/ccu_types.h"
#if !defined(__CCE_AICORE__) && !defined(__aarch64__)
#include <chrono>
#endif

#if defined(__CCE_AICORE__)
#define SIMPLER_CCU_DEVICE [aicore]
#define SIMPLER_CCU_GM __gm__
#else
#define SIMPLER_CCU_DEVICE
#define SIMPLER_CCU_GM
#endif

namespace ccu_detail {
SIMPLER_CCU_DEVICE inline uint64_t NowNs() {
#if defined(__CCE_AICORE__)
    return get_sys_cnt();  // A5 system counter is 1 GHz.
#elif defined(__aarch64__)
    uint64_t ticks, frequency;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(ticks));
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(frequency));
    return frequency ? ticks / frequency * 1000000000 + ticks % frequency * 1000000000 / frequency : 0;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}
SIMPLER_CCU_DEVICE inline void Sync(SIMPLER_CCU_GM void *address, uint64_t bytes) {
#if defined(__CCE_AICORE__)
    (void)bytes;
    __asm__ __volatile__("" ::: "memory");
    dcci(address, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
#elif defined(__aarch64__)
    const auto first = reinterpret_cast<uint64_t>(address) & ~uint64_t{63};
    const auto end = reinterpret_cast<uint64_t>(address) + bytes;
    for (auto line = first; line < end; line += 64)
        __asm__ __volatile__("dc civac, %0" ::"r"(line) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    (void)address;
    (void)bytes;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

SIMPLER_CCU_DEVICE inline bool Fits(uint64_t offset, uint64_t bytes, uint64_t size) {
    return offset <= size && bytes <= size - offset;
}
struct ImageRegion {
    uint64_t offset, bytes, alignment;
};
SIMPLER_CCU_DEVICE inline bool RegionValid(const ImageRegion &region, uint64_t image_size) {
    return region.offset % region.alignment == 0 && Fits(region.offset, region.bytes, image_size) &&
           (region.bytes == 0 || region.offset >= sizeof(CcuDeviceContext));
}
SIMPLER_CCU_DEVICE inline bool RegionsOverlap(const ImageRegion &a, const ImageRegion &b) {
    return a.bytes != 0 && b.bytes != 0 &&
           (a.offset <= b.offset ? b.offset - a.offset < a.bytes : a.offset - b.offset < b.bytes);
}
template <typename T>
SIMPLER_CCU_DEVICE inline SIMPLER_CCU_GM T *At(SIMPLER_CCU_GM CcuDeviceContext *ctx, uint64_t offset) {
    return reinterpret_cast<SIMPLER_CCU_GM T *>(reinterpret_cast<SIMPLER_CCU_GM uint8_t *>(ctx) + offset);
}
SIMPLER_CCU_DEVICE inline bool BindingValid(SIMPLER_CCU_GM CcuDeviceContext *ctx, CcuProgramRef ref) {
    return ref.context_addr == reinterpret_cast<uint64_t>(ctx) && ref.session_epoch == ctx->session_epoch &&
           ref.binding_offset >= ctx->bindings_offset &&
           (ref.binding_offset - ctx->bindings_offset) % sizeof(CcuProgramBinding) == 0 &&
           (ref.binding_offset - ctx->bindings_offset) / sizeof(CcuProgramBinding) < ctx->binding_count;
}
SIMPLER_CCU_DEVICE inline bool Accepting(SIMPLER_CCU_GM CcuDeviceContext *ctx) {
    if (ctx->state != CcuSessionState::READY) return false;
    if (!ctx->admission_offset) return true;
    auto *admission = At<CcuAdmissionRecord>(ctx, ctx->admission_offset);
    Sync(admission, sizeof(*admission));
    return admission->closed == 0;
}

SIMPLER_CCU_DEVICE inline SIMPLER_CCU_GM CcuRequestRecord *Record(CcuRequestRef request) {
    if (!request.record_addr || !request.sequence || request.record_addr % alignof(CcuRequestRecord)) return nullptr;
    auto *record = reinterpret_cast<SIMPLER_CCU_GM CcuRequestRecord *>(request.record_addr);
    Sync(record, sizeof(*record));
    if (record->sequence != request.sequence || record->record_offset > request.record_addr) return nullptr;
    auto *ctx = reinterpret_cast<SIMPLER_CCU_GM CcuDeviceContext *>(request.record_addr - record->record_offset);
    Sync(ctx, sizeof(*ctx));
    if (ctx->abi_version != CCU_DEVICE_ABI_VERSION || ctx->session_epoch != record->session_epoch ||
        ctx->record_offset != record->record_offset)
        return nullptr;
    return record;
}
}  // namespace ccu_detail

SIMPLER_CCU_DEVICE inline CcuStatus
ccu_get_device_context(uint64_t addr, uint64_t size, SIMPLER_CCU_GM CcuDeviceContext *&out) {
    out = nullptr;
    if (!addr || addr % alignof(CcuDeviceContext) || size < sizeof(CcuDeviceContext))
        return CcuStatus::INVALID_ARGUMENT;
    auto *ctx = reinterpret_cast<SIMPLER_CCU_GM CcuDeviceContext *>(addr);
    ccu_detail::Sync(ctx, sizeof(*ctx));
    if (ctx->abi_version != CCU_DEVICE_ABI_VERSION || ctx->struct_size != sizeof(*ctx)) return CcuStatus::ABI_MISMATCH;
    if (!ctx->session_epoch || ctx->state != CcuSessionState::READY) return CcuStatus::STALE_HANDLE;
    if (ctx->image_size < sizeof(*ctx) || ctx->image_size > size || ctx->image_size > UINT64_MAX - addr)
        return CcuStatus::ABI_MISMATCH;
    const ccu_detail::ImageRegion regions[]{
        {ctx->bindings_offset, uint64_t{ctx->binding_count} * sizeof(CcuProgramBinding), alignof(CcuProgramBinding)},
        {ctx->ranges_offset, uint64_t{ctx->range_count} * sizeof(CcuMemoryRange), alignof(CcuMemoryRange)},
        {ctx->record_offset, sizeof(CcuRequestRecord), alignof(CcuRequestRecord)},
        {ctx->arguments_offset, ctx->arguments_size, alignof(CcuPositionalArguments)},
        {ctx->admission_offset, ctx->admission_offset ? sizeof(CcuAdmissionRecord) : 0, alignof(CcuAdmissionRecord)}
    };
    for (uint32_t i = 0; i < sizeof(regions) / sizeof(regions[0]); ++i) {
        if (!ccu_detail::RegionValid(regions[i], ctx->image_size)) return CcuStatus::ABI_MISMATCH;
        for (uint32_t j = 0; j < i; ++j)
            if (ccu_detail::RegionsOverlap(regions[i], regions[j])) return CcuStatus::ABI_MISMATCH;
    }
    if (!ccu_detail::Accepting(ctx)) return CcuStatus::STALE_HANDLE;
    out = ctx;
    return CcuStatus::OK;
}
SIMPLER_CCU_DEVICE inline CcuStatus
ccu_resolve_program(SIMPLER_CCU_GM CcuDeviceContext *ctx, uint32_t ccu_id, CcuProgramRef &out) {
    out = {};
    if (!ctx) return CcuStatus::INVALID_ARGUMENT;
    SIMPLER_CCU_GM CcuDeviceContext *validated = nullptr;
    const auto status = ccu_get_device_context(reinterpret_cast<uint64_t>(ctx), ctx->image_size, validated);
    if (status != CcuStatus::OK) return status;
    if (ccu_id >= ctx->binding_count) return CcuStatus::INVALID_ARGUMENT;
    const auto offset = ctx->bindings_offset + uint64_t{ccu_id} * sizeof(CcuProgramBinding);
    const auto *binding = ccu_detail::At<CcuProgramBinding>(ctx, offset);
    if (binding->program_index != ccu_id || binding->session_epoch != ctx->session_epoch)
        return CcuStatus::ABI_MISMATCH;
    out = {reinterpret_cast<uint64_t>(ctx), ctx->session_epoch, offset};
    return CcuStatus::OK;
}

SIMPLER_CCU_DEVICE inline CcuStatus ccu_reserve_request(
    SIMPLER_CCU_GM CcuDeviceContext *ctx, CcuProgramRef program, CcuRequestRef &out, const CcuPositionalArguments &args
) {
    out = {};
    if (!ctx || !ccu_detail::BindingValid(ctx, program) || !ccu_detail::Accepting(ctx)) return CcuStatus::STALE_HANDLE;
    const auto *binding = ccu_detail::At<CcuProgramBinding>(ctx, program.binding_offset);
    if (binding->argument_bytes != sizeof(CcuPositionalArguments) ||
        ctx->arguments_size < sizeof(CcuPositionalArguments) || args.tensor_count > CCU_MAX_TENSORS ||
        args.scalar_count > CCU_MAX_SCALARS)
        return CcuStatus::INVALID_ARGUMENT;
    auto *record = ccu_detail::At<CcuRequestRecord>(ctx, ctx->record_offset);
    ccu_detail::Sync(record, sizeof(*record));
    if (record->slot_state != CcuSlotState::FREE) return CcuStatus::BUSY;
    CcuPositionalArguments snapshot{};
    snapshot.tensor_count = args.tensor_count;
    snapshot.scalar_count = args.scalar_count;
    const auto *ranges = ccu_detail::At<CcuMemoryRange>(ctx, ctx->ranges_offset);
    for (uint32_t i = 0; i < args.tensor_count; ++i) {
        const auto &tensor = args.tensors[i];
        if (!tensor.address || tensor.address % 4 || !tensor.byte_size || tensor.byte_size % 4 ||
            tensor.byte_size > UINT64_MAX - tensor.address)
            return CcuStatus::INVALID_ARGUMENT;
        bool covered = false;
        for (uint32_t j = 0; j < ctx->range_count; ++j) {
            if (ranges[j].byte_size <= UINT64_MAX - ranges[j].addr && tensor.address >= ranges[j].addr &&
                ccu_detail::Fits(tensor.address - ranges[j].addr, tensor.byte_size, ranges[j].byte_size)) {
                snapshot.tensors[i] = {tensor.address, ranges[j].token, tensor.byte_size};
                covered = true;
                break;
            }
        }
        if (!covered) return CcuStatus::INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < args.scalar_count; ++i) {
        const auto &scalar = args.scalars[i];
        if ((scalar.width != 32 && scalar.width != 64) ||
            (scalar.kind != CcuScalarKind::SIGNED && scalar.kind != CcuScalarKind::UNSIGNED &&
             scalar.kind != CcuScalarKind::FLOAT) ||
            (scalar.width == 32 && scalar.bits > UINT32_MAX))
            return CcuStatus::UNSUPPORTED;
        snapshot.scalars[i] = scalar;
    }
    if (!ctx->next_sequence || ctx->next_sequence == UINT64_MAX) {
        ctx->state = CcuSessionState::UNCONFIRMED;
        ccu_detail::Sync(ctx, sizeof(*ctx));
        return CcuStatus::RECLAMATION_UNCONFIRMED;
    }
    auto *destination = ccu_detail::At<uint8_t>(ctx, ctx->arguments_offset);
    const auto *source = reinterpret_cast<const uint8_t *>(&snapshot);
    for (uint32_t i = 0; i < sizeof(snapshot); ++i)
        destination[i] = source[i];
    record->session_epoch = ctx->session_epoch;
    record->sequence = ctx->next_sequence++;
    record->binding_offset = program.binding_offset;
    record->record_offset = ctx->record_offset;
    record->program_index = binding->program_index;
    record->argument_bytes = sizeof(snapshot);
    record->local_state = CcuRequestState::PENDING;
    record->local_status = CcuStatus::OK;
    record->slot_state = CcuSlotState::RESERVED;
    ccu_detail::Sync(destination, sizeof(snapshot));
    ccu_detail::Sync(record, 128);
    ccu_detail::Sync(ctx, sizeof(*ctx));
    out = {reinterpret_cast<uint64_t>(record), record->sequence};
    return CcuStatus::OK;
}
SIMPLER_CCU_DEVICE inline CcuStatus ccu_submit_request(const CcuRequestRef &request) {
    auto *record = ccu_detail::Record(request);
    if (!record) return CcuStatus::STALE_HANDLE;
    if (record->slot_state != CcuSlotState::RESERVED) return CcuStatus::INVALID_ARGUMENT;
    auto *ctx = reinterpret_cast<SIMPLER_CCU_GM CcuDeviceContext *>(request.record_addr - record->record_offset);
    auto *binding = ccu_detail::At<CcuProgramBinding>(ctx, record->binding_offset);
    record->slot_state = CcuSlotState::SUBMITTED;
    if (!ccu_detail::Accepting(ctx) || !binding->gate_addr) {
        record->local_state = CcuRequestState::FAILED;
        record->local_status = CcuStatus::STALE_HANDLE;
        ccu_detail::Sync(record, 128);
        return CcuStatus::STALE_HANDLE;
    }
    record->submitted_at_ns = ccu_detail::NowNs();
    ccu_detail::Sync(record, 128);
    auto *gate = reinterpret_cast<volatile SIMPLER_CCU_GM uint64_t *>(binding->gate_addr);
    ccu_detail::Sync(const_cast<SIMPLER_CCU_GM uint64_t *>(gate), 8);
    *gate = (uint64_t{1} << 63) | 1;
    ccu_detail::Sync(const_cast<SIMPLER_CCU_GM uint64_t *>(gate), 8);
#if defined(__CCE_AICORE__)
    pipe_barrier(PIPE_ALL);
#endif
    return CcuStatus::OK;
}
SIMPLER_CCU_DEVICE inline void ccu_cancel_reserved_request(const CcuRequestRef &request) {
    auto *record = ccu_detail::Record(request);
    if (record && record->slot_state == CcuSlotState::RESERVED) {
        record->slot_state = CcuSlotState::FREE;
        ccu_detail::Sync(record, 128);
    }
}
SIMPLER_CCU_DEVICE inline CcuPollResult ccu_poll_request(const CcuRequestRef &request) {
    auto *record = ccu_detail::Record(request);
    if (!record) return {CcuRequestState::UNCONFIRMED, CcuStatus::STALE_HANDLE, 0};
    auto *ctx = reinterpret_cast<SIMPLER_CCU_GM CcuDeviceContext *>(request.record_addr - record->record_offset);
    if (record->local_state != CcuRequestState::PENDING) return {record->local_state, record->local_status, 0};
    if (ctx->state == CcuSessionState::UNCONFIRMED)
        return {CcuRequestState::UNCONFIRMED, CcuStatus::RECLAMATION_UNCONFIRMED, 0};
    if (record->slot_state != CcuSlotState::SUBMITTED) return {CcuRequestState::PENDING, CcuStatus::OK, 0};
    if (record->completed_sequence < request.sequence) {
        if (ctx->timeout_ns && ccu_detail::NowNs() - record->submitted_at_ns >= ctx->timeout_ns) {
            record->local_state = CcuRequestState::UNCONFIRMED;
            record->local_status = CcuStatus::TIMEOUT;
            ctx->state = CcuSessionState::UNCONFIRMED;
            ccu_detail::Sync(record, 128);
            ccu_detail::Sync(ctx, sizeof(*ctx));
            return {CcuRequestState::UNCONFIRMED, CcuStatus::TIMEOUT, 0};
        }
        return {CcuRequestState::PENDING, CcuStatus::OK, 0};
    }
    if (record->completed_sequence != request.sequence || record->completed_epoch != record->session_epoch ||
        record->completed_program != record->program_index) {
        ctx->state = CcuSessionState::UNCONFIRMED;
        ccu_detail::Sync(ctx, sizeof(*ctx));
        return {CcuRequestState::UNCONFIRMED, CcuStatus::STALE_HANDLE, 0};
    }
    if (record->completed_state == CcuRequestState::UNCONFIRMED) {
        ctx->state = CcuSessionState::UNCONFIRMED;
        ccu_detail::Sync(ctx, sizeof(*ctx));
    }
    if ((record->completed_state != CcuRequestState::SUCCEEDED && record->completed_state != CcuRequestState::FAILED &&
         record->completed_state != CcuRequestState::UNCONFIRMED) ||
        (record->completed_state == CcuRequestState::SUCCEEDED && record->status != CcuStatus::OK)) {
        ctx->state = CcuSessionState::UNCONFIRMED;
        ccu_detail::Sync(ctx, sizeof(*ctx));
        return {CcuRequestState::UNCONFIRMED, CcuStatus::ABI_MISMATCH, 0};
    }
    return {record->completed_state, record->status, record->native_error};
}
SIMPLER_CCU_DEVICE inline void ccu_retire_request(const CcuRequestRef &request) {
    auto *record = ccu_detail::Record(request);
    if (!record || record->slot_state != CcuSlotState::SUBMITTED) return;
    const auto result = ccu_poll_request(request);
    if (result.state != CcuRequestState::SUCCEEDED && result.state != CcuRequestState::FAILED) return;
    record->slot_state = CcuSlotState::FREE;
    ccu_detail::Sync(record, 128);
}

#undef SIMPLER_CCU_DEVICE
#undef SIMPLER_CCU_GM
