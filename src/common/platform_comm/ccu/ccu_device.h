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
template <typename T>
SIMPLER_CCU_DEVICE inline SIMPLER_CCU_GM T *At(SIMPLER_CCU_GM CcuDeviceContext *ctx, uint64_t offset) {
    return reinterpret_cast<SIMPLER_CCU_GM T *>(reinterpret_cast<SIMPLER_CCU_GM uint8_t *>(ctx) + offset);
}
SIMPLER_CCU_DEVICE inline bool Equal(const CcuProgramKey &a, const SIMPLER_CCU_GM CcuProgramKey &b) {
    for (uint32_t i = 0; i < 32; ++i)
        if (a.bytes[i] != b.bytes[i]) return false;
    return true;
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

template <typename T, typename = void>
struct IsTensor : std::false_type {};
template <typename T>
struct IsTensor<
    T, std::void_t<
           decltype(T{}.buffer.addr), decltype(T{}.start_offset), decltype(T{}.ndims), decltype(T{}.shapes[0]),
           decltype(T{}.strides[0]), decltype(T{}.dtype)>> : std::true_type {};

struct ArgumentValue {
    CcuArgumentKind kind;
    CcuArgumentType type;
    uint64_t address;
    uint64_t bytes;
    bool valid;
};
template <typename T>
SIMPLER_CCU_DEVICE inline ArgumentValue Value(const T &arg) {
    if constexpr (IsTensor<T>::value) {
        // DataType::FLOAT32 is zero in the existing kernel Tensor ABI.
        if (static_cast<uint32_t>(arg.dtype) != 0 || arg.ndims == 0 ||
            arg.ndims > sizeof(arg.shapes) / sizeof(arg.shapes[0]))
            return {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 0, 0, false};
        uint64_t elements = 1;
        for (uint32_t d = arg.ndims; d > 0; --d) {
            if (arg.shapes[d - 1] == 0 || (arg.shapes[d - 1] != 1 && arg.strides[d - 1] != elements) ||
                elements > UINT64_MAX / arg.shapes[d - 1])
                return {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 0, 0, false};
            elements *= arg.shapes[d - 1];
        }
        if (elements > UINT64_MAX / 4 || arg.start_offset > UINT64_MAX / 4 ||
            !Fits(arg.start_offset * 4, elements * 4, arg.buffer.size) ||
            arg.buffer.size > UINT64_MAX - arg.buffer.addr)
            return {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 0, 0, false};
        return {
            CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, arg.buffer.addr + arg.start_offset * 4, elements * 4,
            true
        };
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return {CcuArgumentKind::SCALAR, CcuArgumentType::UINT64, arg, 0, true};
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return {CcuArgumentKind::SCALAR, CcuArgumentType::INT64, static_cast<uint64_t>(arg), 0, true};
    } else {
        return {CcuArgumentKind::SCALAR, CcuArgumentType::UINT64, 0, 0, false};
    }
}

SIMPLER_CCU_DEVICE inline CcuStatus WriteArgument(
    SIMPLER_CCU_GM CcuDeviceContext *ctx, SIMPLER_CCU_GM CcuProgramBinding *binding, uint32_t index,
    const ArgumentValue &arg
) {
    auto *signature = At<CcuParameterLayout>(ctx, binding->signature_offset);
    const auto &parameter = signature[index];
    const uint64_t words = arg.kind == CcuArgumentKind::TENSOR ? 3 : 1;
    if (!arg.valid || parameter.kind != arg.kind || parameter.type != arg.type ||
        !Fits(uint64_t{parameter.word_offset} * 8, words * 8, binding->argument_bytes))
        return CcuStatus::INVALID_ARGUMENT;
    auto *arguments = At<uint64_t>(ctx, ctx->arguments_offset);
    if (arg.kind == CcuArgumentKind::SCALAR) {
        arguments[parameter.word_offset] = arg.address;
        return CcuStatus::OK;
    }
    bool covered = false;
    uint64_t token = 0;
    const auto *ranges = At<CcuMemoryRange>(ctx, ctx->ranges_offset);
    for (uint32_t i = 0; i < ctx->range_count; ++i) {
        if (arg.address >= ranges[i].addr && Fits(arg.address - ranges[i].addr, arg.bytes, ranges[i].byte_size)) {
            covered = true;
            token = ranges[i].token;
            break;
        }
    }
    if (!covered || arg.address % 4) return CcuStatus::INVALID_ARGUMENT;
    for (uint32_t i = 0; i < index; ++i) {
        if (signature[i].kind != CcuArgumentKind::TENSOR) continue;
        const auto offset = signature[i].word_offset;
        const auto previous_addr = arguments[offset], previous_bytes = arguments[offset + 2];
        if (arg.address < previous_addr + previous_bytes && previous_addr < arg.address + arg.bytes) {
            if (arg.address != previous_addr || arg.bytes != previous_bytes ||
                (parameter.allowed_aliases & (uint64_t{1} << i)) == 0)
                return CcuStatus::INVALID_ARGUMENT;
        }
    }
    arguments[parameter.word_offset] = arg.address;
    arguments[parameter.word_offset + 1] = token;
    arguments[parameter.word_offset + 2] = arg.bytes;
    return CcuStatus::OK;
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
    if (ctx->image_size > size ||
        !ccu_detail::Fits(
            ctx->bindings_offset, uint64_t{ctx->binding_count} * sizeof(CcuProgramBinding), ctx->image_size
        ) ||
        !ccu_detail::Fits(ctx->ranges_offset, uint64_t{ctx->range_count} * sizeof(CcuMemoryRange), ctx->image_size) ||
        !ccu_detail::Fits(ctx->record_offset, sizeof(CcuRequestRecord), ctx->image_size) ||
        ctx->record_offset % alignof(CcuRequestRecord) ||
        !ccu_detail::Fits(ctx->arguments_offset, ctx->arguments_size, ctx->image_size) ||
        (ctx->admission_offset &&
         (ctx->admission_offset % 128 ||
          !ccu_detail::Fits(ctx->admission_offset, sizeof(CcuAdmissionRecord), ctx->image_size))))
        return CcuStatus::ABI_MISMATCH;
    if (!ccu_detail::Accepting(ctx)) return CcuStatus::STALE_HANDLE;
    out = ctx;
    return CcuStatus::OK;
}
SIMPLER_CCU_DEVICE inline CcuStatus
ccu_resolve_program(SIMPLER_CCU_GM CcuDeviceContext *ctx, const CcuProgramKey &key, CcuProgramRef &out) {
    out = {};
    if (!ctx) return CcuStatus::INVALID_ARGUMENT;
    SIMPLER_CCU_GM CcuDeviceContext *validated = nullptr;
    const auto status = ccu_get_device_context(reinterpret_cast<uint64_t>(ctx), ctx->image_size, validated);
    if (status != CcuStatus::OK) return status;
    auto *bindings = ccu_detail::At<CcuProgramBinding>(ctx, ctx->bindings_offset);
    for (uint32_t i = 0; i < ctx->binding_count; ++i) {
        if (ccu_detail::Equal(key, bindings[i].key)) {
            if (bindings[i].session_epoch != ctx->session_epoch) return CcuStatus::STALE_HANDLE;
            out = {
                reinterpret_cast<uint64_t>(ctx), ctx->session_epoch,
                ctx->bindings_offset + i * sizeof(CcuProgramBinding)
            };
            return CcuStatus::OK;
        }
    }
    return CcuStatus::INVALID_ARGUMENT;
}
template <typename Binding>
SIMPLER_CCU_DEVICE inline CcuStatus ccu_resolve_program(SIMPLER_CCU_GM CcuDeviceContext *ctx, CcuProgramRef &out) {
    return ccu_resolve_program(ctx, Binding::key, out);
}
template <typename... Args>
SIMPLER_CCU_DEVICE inline CcuStatus ccu_reserve_request(
    SIMPLER_CCU_GM CcuDeviceContext *ctx, CcuProgramRef program, CcuRequestRef &out, const Args &...args
) {
    out = {};
    if (!ctx || !ccu_detail::BindingValid(ctx, program) || !ccu_detail::Accepting(ctx)) return CcuStatus::STALE_HANDLE;
    auto *binding = ccu_detail::At<CcuProgramBinding>(ctx, program.binding_offset);
    if (binding->parameter_count != sizeof...(Args) || binding->parameter_count > 64 ||
        binding->argument_bytes > ctx->arguments_size ||
        !ccu_detail::Fits(
            binding->signature_offset, binding->parameter_count * sizeof(CcuParameterLayout), ctx->image_size
        ))
        return CcuStatus::INVALID_ARGUMENT;
    auto *record = ccu_detail::At<CcuRequestRecord>(ctx, ctx->record_offset);
    ccu_detail::Sync(record, sizeof(*record));
    if (record->slot_state != CcuSlotState::FREE) return CcuStatus::BUSY;
    if (ctx->next_sequence == 0 || ctx->next_sequence == UINT64_MAX) {
        ctx->state = CcuSessionState::UNCONFIRMED;
        ccu_detail::Sync(ctx, sizeof(*ctx));
        return CcuStatus::RECLAMATION_UNCONFIRMED;
    }
    record->slot_state = CcuSlotState::RESERVED;
    const auto sequence = ctx->next_sequence++;
    CcuStatus status = CcuStatus::OK;
    uint32_t index = 0;
    ((status =
          status == CcuStatus::OK ? ccu_detail::WriteArgument(ctx, binding, index++, ccu_detail::Value(args)) : status),
     ...);
    if (status != CcuStatus::OK) {
        record->slot_state = CcuSlotState::FREE;
        ccu_detail::Sync(record, 128);
        return status;
    }
    record->session_epoch = ctx->session_epoch;
    record->sequence = sequence;
    record->binding_offset = program.binding_offset;
    record->record_offset = ctx->record_offset;
    record->program_index = binding->program_index;
    record->argument_bytes = binding->argument_bytes;
    record->local_state = CcuRequestState::PENDING;
    record->local_status = CcuStatus::OK;
    ccu_detail::Sync(ctx, sizeof(*ctx));
    ccu_detail::Sync(record, 128);
    ccu_detail::Sync(ccu_detail::At<uint8_t>(ctx, ctx->arguments_offset), binding->argument_bytes);
    out = {reinterpret_cast<uint64_t>(record), sequence};
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
