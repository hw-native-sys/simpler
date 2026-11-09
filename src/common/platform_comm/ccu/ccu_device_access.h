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

#include "ccu_device_protocol.h"
#include "platform_comm/ccu/ccu_device.h"
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

namespace simpler::ccu::device {

static_assert(MAX_TENSORS == CCU_MAX_TENSORS && MAX_SCALARS == CCU_MAX_SCALARS);
static_assert(static_cast<uint32_t>(ScalarKind::UNSIGNED) == static_cast<uint32_t>(CcuScalarKind::UNSIGNED));
static_assert(static_cast<uint32_t>(ScalarKind::SIGNED) == static_cast<uint32_t>(CcuScalarKind::SIGNED));
static_assert(static_cast<uint32_t>(ScalarKind::FLOAT) == static_cast<uint32_t>(CcuScalarKind::FLOAT));

struct Transport {
    static SIMPLER_CCU_DEVICE uint64_t now_ns() {
#if defined(__aarch64__)
        uint64_t ticks, frequency;
        __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(ticks));
        __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(frequency));
        return ticks / frequency * 1000000000 + ticks % frequency * 1000000000 / frequency;
#elif defined(__CCE_AICORE__)
        return get_sys_cnt();  // A5 counter frequency: 1 GHz.
#else
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
#endif
    }

    static SIMPLER_CCU_DEVICE void sync(SIMPLER_CCU_GM void *address, uint64_t bytes) {
#if defined(__aarch64__)
        const auto first = reinterpret_cast<uint64_t>(address) & ~uint64_t{63};
        const auto end = reinterpret_cast<uint64_t>(address) + bytes;
        for (auto line = first; line < end; line += 64)
            __asm__ __volatile__("dc civac, %0" ::"r"(line) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__CCE_AICORE__)
        (void)bytes;
        dcci(address, cache_line_t::ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
#else
        (void)address;
        (void)bytes;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }

    static SIMPLER_CCU_DEVICE void trigger(const SIMPLER_CCU_GM ProgramBinding &binding) {
        auto *gate = reinterpret_cast<volatile SIMPLER_CCU_GM uint64_t *>(binding.doorbell_addr);
        *gate = binding.doorbell_value;
        sync(const_cast<SIMPLER_CCU_GM uint64_t *>(gate), sizeof(uint64_t));
    }
};

SIMPLER_CCU_DEVICE inline bool fits(uint64_t offset, uint64_t bytes, uint64_t size) {
    return offset <= size && bytes <= size - offset;
}

template <typename T>
SIMPLER_CCU_DEVICE inline SIMPLER_CCU_GM T *at(SIMPLER_CCU_GM Workspace *workspace, uint64_t offset) {
    return reinterpret_cast<SIMPLER_CCU_GM T *>(reinterpret_cast<SIMPLER_CCU_GM uint8_t *>(workspace) + offset);
}

// An accepted reference identifies a single generation. It is internal and
// never exposed as a user-managed ticket.
struct Reference {
    SIMPLER_CCU_GM Workspace *workspace{};
    uint32_t binding_index{};
    uint64_t sequence{};
};

template <typename Io = Transport>
struct Protocol {
    static SIMPLER_CCU_DEVICE SIMPLER_CCU_GM CcuDeviceContext *platform(Reference ref) {
        auto *w = ref.workspace;
        if (!w || !w->platform_context || w->platform_context % alignof(CcuDeviceContext) ||
            w->platform_context_bytes < sizeof(CcuDeviceContext) || ref.binding_index >= w->binding_count ||
            w->bindings_offset < sizeof(Workspace) || w->bindings_offset % alignof(ProgramBinding) ||
            !fits(w->bindings_offset, uint64_t(w->binding_count) * sizeof(ProgramBinding), w->image_bytes))
            return nullptr;
        const auto &binding = at<ProgramBinding>(w, w->bindings_offset)[ref.binding_index];
        auto *ctx = reinterpret_cast<SIMPLER_CCU_GM CcuDeviceContext *>(w->platform_context);
        Io::sync(ctx, sizeof(*ctx));
        if (ctx->abi_version != CCU_DEVICE_ABI_VERSION || ctx->struct_size != sizeof(*ctx) ||
            ctx->image_size < sizeof(*ctx) || ctx->image_size > w->platform_context_bytes ||
            ctx->allocation_id != w->domain_id || ctx->domain_owner != w->domain_owner ||
            ctx->callable_id != binding.callable_id || ctx->session_epoch != w->generation || ctx->rank != w->rank ||
            ctx->rank_count != w->rank_count || ctx->binding_count != w->binding_count ||
            ctx->record_offset < sizeof(*ctx) || ctx->record_offset % alignof(CcuRequestRecord) ||
            !fits(ctx->record_offset, sizeof(CcuRequestRecord), ctx->image_size) ||
            !fits(ctx->arguments_offset, sizeof(CcuPositionalArguments), ctx->image_size))
            return nullptr;
        return ctx;
    }

    static SIMPLER_CCU_DEVICE CcuRequestRef platform_request(Reference ref) {
        auto *ctx = platform(ref);
        if (!ctx || ref.binding_index >= ref.workspace->binding_count || !ref.sequence) return {};
        const auto &binding = at<ProgramBinding>(ref.workspace, ref.workspace->bindings_offset)[ref.binding_index];
        auto *record = ccu_detail::At<CcuRequestRecord>(ctx, ctx->record_offset);
        Io::sync(record, sizeof(*record));
        if (record->program_index != binding.ccu_id || record->sequence != ref.sequence ||
            record->session_epoch != ctx->session_epoch)
            return {};
        return {reinterpret_cast<uint64_t>(record), ref.sequence};
    }

    static SIMPLER_CCU_DEVICE SIMPLER_CCU_GM Workspace *instance(
        SIMPLER_CCU_GM DomainDirectory *directory, uint64_t available, int32_t callable_id, uint32_t rank,
        uint32_t rank_count, uint64_t &image_bytes
    ) {
        if (!directory || reinterpret_cast<uint64_t>(directory) % alignof(DomainDirectory) ||
            available < sizeof(DomainDirectory) || callable_id < 0)
            return nullptr;
        Io::sync(directory, sizeof(*directory));
        if (directory->version != PROTOCOL_VERSION || directory->header_bytes != sizeof(DomainDirectory) ||
            directory->image_bytes > available || directory->generation == 0 ||
            directory->admission != Admission::OPEN || directory->rank != rank || directory->rank_count != rank_count ||
            directory->entries_offset < sizeof(DomainDirectory) || directory->entries_offset % alignof(InstanceEntry) ||
            !fits(
                directory->entries_offset, uint64_t(directory->entry_count) * sizeof(InstanceEntry),
                directory->image_bytes
            ))
            return nullptr;
        auto *entries = reinterpret_cast<SIMPLER_CCU_GM InstanceEntry *>(
            reinterpret_cast<SIMPLER_CCU_GM uint8_t *>(directory) + directory->entries_offset
        );
        SIMPLER_CCU_GM Workspace *found = nullptr;
        for (uint32_t i = 0; i < directory->entry_count; ++i) {
            if (entries[i].callable_id != callable_id) continue;
            if (found) return nullptr;
            auto *w = reinterpret_cast<SIMPLER_CCU_GM Workspace *>(entries[i].image_addr);
            if (!valid(w, entries[i].image_bytes) || w->domain_owner != directory->domain_owner ||
                w->domain_id != directory->domain_id || w->generation != directory->generation || w->rank != rank ||
                w->rank_count != rank_count)
                return nullptr;
            found = w;
            image_bytes = entries[i].image_bytes;
        }
        return found;
    }

    static SIMPLER_CCU_DEVICE bool valid(SIMPLER_CCU_GM Workspace *w, uint64_t available) {
        if (w == nullptr || reinterpret_cast<uint64_t>(w) % alignof(Workspace) || available < sizeof(Workspace))
            return false;
        Io::sync(w, sizeof(*w));
        if (w->version != PROTOCOL_VERSION || w->header_bytes != sizeof(Workspace) || w->image_bytes > available ||
            w->image_bytes < sizeof(Workspace) || w->generation == 0 || w->timeout_ns == 0 || w->rank_count != 2 ||
            w->rank >= w->rank_count || w->range_count == 0 || w->binding_count == 0)
            return false;
        if ((w->platform_context == 0) != (w->platform_context_bytes == 0) ||
            (w->platform_context && !platform({w, 0, 0})))
            return false;
        return w->ranges_offset >= sizeof(Workspace) && w->ranges_offset % alignof(MemoryRange) == 0 &&
               fits(w->ranges_offset, uint64_t(w->range_count) * sizeof(MemoryRange), w->image_bytes) &&
               w->bindings_offset >= sizeof(Workspace) && w->bindings_offset % alignof(ProgramBinding) == 0 &&
               fits(w->bindings_offset, uint64_t(w->binding_count) * sizeof(ProgramBinding), w->image_bytes) &&
               (w->ranges_offset + uint64_t(w->range_count) * sizeof(MemoryRange) <= w->bindings_offset ||
                w->bindings_offset + uint64_t(w->binding_count) * sizeof(ProgramBinding) <= w->ranges_offset);
    }

    static SIMPLER_CCU_DEVICE SIMPLER_CCU_GM Request *request(Reference ref) {
        auto *w = ref.workspace;
        if (w && w->platform_context) return nullptr;
        if (!w || ref.binding_index >= w->binding_count) return nullptr;
        const auto &b = at<ProgramBinding>(w, w->bindings_offset)[ref.binding_index];
        if (b.request_offset % alignof(Request) || b.request_offset < sizeof(Workspace) ||
            !fits(b.request_offset, sizeof(Request), w->image_bytes))
            return nullptr;
        const uint64_t end = b.request_offset + sizeof(Request);
        if ((b.request_offset < w->ranges_offset + uint64_t(w->range_count) * sizeof(MemoryRange) &&
             end > w->ranges_offset) ||
            (b.request_offset < w->bindings_offset + uint64_t(w->binding_count) * sizeof(ProgramBinding) &&
             end > w->bindings_offset))
            return nullptr;
        return at<Request>(w, b.request_offset);
    }

    static SIMPLER_CCU_DEVICE bool
    resolve(SIMPLER_CCU_GM Workspace *w, uint64_t available, int32_t callable_id, uint32_t ccu_id, Reference &ref) {
        if (callable_id < 0 || !valid(w, available) || w->admission != Admission::OPEN) return false;
        bool found = false;
        auto *bindings = at<ProgramBinding>(w, w->bindings_offset);
        for (uint32_t i = 0; i < w->binding_count; ++i) {
            if (bindings[i].callable_id != callable_id || bindings[i].ccu_id != ccu_id) continue;
            if (found) return false;
            ref = {w, i, 0};
            if (w->platform_context) {
                CcuProgramRef program{};
                if (ccu_resolve_program(platform(ref), ccu_id, program) != CcuStatus::OK) return false;
            } else if (!bindings[i].doorbell_addr || bindings[i].doorbell_addr % sizeof(uint64_t) || !request(ref)) {
                return false;
            }
            found = true;
        }
        return found;
    }

    static SIMPLER_CCU_DEVICE bool reserve(Reference &ref, const Arguments &args, CcuStatus *status_out = nullptr) {
        if (status_out) *status_out = CcuStatus::INVALID_ARGUMENT;
        if (args.tensor_count > MAX_TENSORS || args.scalar_count > MAX_SCALARS) return false;
        for (uint32_t i = 0; i < args.scalar_count; ++i) {
            const auto &scalar = args.scalars[i];
            if ((scalar.width != 32 && scalar.width != 64) ||
                (scalar.kind != ScalarKind::UNSIGNED && scalar.kind != ScalarKind::SIGNED &&
                 scalar.kind != ScalarKind::FLOAT) ||
                (scalar.width == 32 && scalar.bits > UINT32_MAX))
                return false;
        }
        auto *w = ref.workspace;
        if (w && w->platform_context) {
            if (args.tensor_count > MAX_TENSORS || args.scalar_count > MAX_SCALARS ||
                ref.binding_index >= w->binding_count || w->admission != Admission::OPEN)
                return false;
            auto *ctx = platform(ref);
            if (!ctx) return false;
            CcuProgramRef program{};
            const auto &binding = at<ProgramBinding>(w, w->bindings_offset)[ref.binding_index];
            const auto resolved = ccu_resolve_program(ctx, binding.ccu_id, program);
            if (status_out) *status_out = resolved;
            if (resolved != CcuStatus::OK) return false;
            CcuPositionalArguments snapshot{};
            snapshot.tensor_count = args.tensor_count;
            snapshot.scalar_count = args.scalar_count;
            for (uint32_t i = 0; i < args.tensor_count; ++i) {
                snapshot.tensors[i] = {args.tensors[i].addr, 0, args.tensors[i].bytes};
            }
            for (uint32_t i = 0; i < args.scalar_count; ++i) {
                snapshot.scalars[i] = {
                    args.scalars[i].bits, static_cast<CcuScalarKind>(args.scalars[i].kind), args.scalars[i].width
                };
            }
            CcuRequestRef accepted{};
            const auto reserved = ccu_reserve_request(ctx, program, accepted, snapshot);
            if (status_out) *status_out = reserved;
            if (reserved != CcuStatus::OK) return false;
            ref.sequence = accepted.sequence;
            return true;
        }
        auto *r = request(ref);
        if (!r || args.tensor_count > MAX_TENSORS || args.scalar_count > MAX_SCALARS) return false;
        Io::sync(w, sizeof(*w));
        Io::sync(r, sizeof(*r));
        if (w->admission != Admission::OPEN || r->slot != Slot::FREE || r->sequence == UINT64_MAX) return false;
        Arguments snapshot = args;
        auto *ranges = at<MemoryRange>(w, w->ranges_offset);
        for (uint32_t i = 0; i < args.tensor_count; ++i) {
            const auto &tensor = args.tensors[i];
            if (!tensor.addr || tensor.addr % 4 || !tensor.bytes || tensor.bytes % 4 ||
                tensor.bytes > UINT64_MAX - tensor.addr)
                return false;
            bool found = false;
            for (uint32_t j = 0; j < w->range_count; ++j) {
                if (tensor.addr >= ranges[j].addr && ranges[j].bytes <= UINT64_MAX - ranges[j].addr &&
                    fits(tensor.addr - ranges[j].addr, tensor.bytes, ranges[j].bytes)) {
                    snapshot.tensors[i].credential = ranges[j].credential;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        // Exactly one producer is permitted for this instance. Cross-task
        // serialization is declared by the caller's ordering Tensor.
        ref.sequence = r->sequence + 1;
        r->sequence = ref.sequence;
        r->generation = w->generation;
        r->binding_index = ref.binding_index;
        r->submitted_ns = 0;
        auto *dst = reinterpret_cast<SIMPLER_CCU_GM uint8_t *>(&r->arguments);
        const auto *src = reinterpret_cast<const uint8_t *>(&snapshot);
        for (uint32_t i = 0; i < sizeof(snapshot); ++i)
            dst[i] = src[i];
        r->slot = Slot::RESERVED;
        Io::sync(r, offsetof(Request, completion));
        if (status_out) *status_out = CcuStatus::OK;
        return true;
    }

    static SIMPLER_CCU_DEVICE bool matches(Reference ref, SIMPLER_CCU_GM Request *r) {
        return r && ref.sequence != 0 && r->sequence == ref.sequence && r->generation == ref.workspace->generation &&
               r->binding_index == ref.binding_index;
    }

    static SIMPLER_CCU_DEVICE bool cancel(Reference ref) {
        if (ref.workspace && ref.workspace->platform_context) {
            const auto accepted = platform_request(ref);
            auto *record = ccu_detail::Record(accepted);
            if (!record) return false;
            if (record->slot_state == CcuSlotState::RESERVED) {
                ccu_cancel_reserved_request(accepted);
                return record->slot_state == CcuSlotState::FREE;
            }
            // Failed publication has not written the hardware doorbell.
            if (record->slot_state == CcuSlotState::SUBMITTED && record->local_state == CcuRequestState::FAILED) {
                ccu_retire_request(accepted);
                return record->slot_state == CcuSlotState::FREE;
            }
            return false;
        }
        auto *r = request(ref);
        if (!matches(ref, r) || r->slot != Slot::RESERVED) return false;
        r->slot = Slot::FREE;
        Io::sync(r, offsetof(Request, arguments));
        return true;
    }

    static SIMPLER_CCU_DEVICE bool submit(Reference ref) {
        if (ref.workspace && ref.workspace->platform_context) {
            Io::sync(ref.workspace, sizeof(Workspace));
            if (ref.workspace->admission != Admission::OPEN) return false;
            const auto accepted = platform_request(ref);
            return accepted.record_addr && ccu_submit_request(accepted) == CcuStatus::OK;
        }
        auto *r = request(ref);
        if (!matches(ref, r) || r->slot != Slot::RESERVED) return false;
        Io::sync(ref.workspace, sizeof(Workspace));
        if (ref.workspace->admission != Admission::OPEN) return false;
        r->submitted_ns = Io::now_ns();
        r->slot = Slot::SUBMITTED;
        Io::sync(r, offsetof(Request, completion));
        Io::trigger(at<ProgramBinding>(ref.workspace, ref.workspace->bindings_offset)[ref.binding_index]);
        return true;
    }

    static SIMPLER_CCU_DEVICE Outcome poll(Reference ref) {
        if (ref.workspace && ref.workspace->platform_context) {
            const auto accepted = platform_request(ref);
            auto *record = ccu_detail::Record(accepted);
            if (!record || record->slot_state != CcuSlotState::SUBMITTED) return Outcome::UNCONFIRMED;
            const auto result = ccu_poll_request(accepted);
            if (result.state == CcuRequestState::PENDING) return Outcome::PENDING;
            if (result.state == CcuRequestState::SUCCEEDED) return Outcome::SUCCEEDED;
            if (result.state == CcuRequestState::FAILED) return Outcome::FAILED;
            ref.workspace->admission = Admission::UNCONFIRMED;
            Io::sync(ref.workspace, sizeof(Workspace));
            return Outcome::UNCONFIRMED;
        }
        auto *r = request(ref);
        if (!r) return Outcome::UNCONFIRMED;
        Io::sync(r, sizeof(*r));
        if (!matches(ref, r) || r->slot != Slot::SUBMITTED) return Outcome::UNCONFIRMED;
        auto &c = r->completion;
        if (c.sequence == ref.sequence && c.generation == r->generation && c.binding_index == ref.binding_index) {
            // Only these terminal values prove all remote readers/writers stopped.
            if (c.outcome == Outcome::SUCCEEDED || c.outcome == Outcome::FAILED) return c.outcome;
            if (c.outcome == Outcome::UNCONFIRMED) {
                ref.workspace->admission = Admission::UNCONFIRMED;
                Io::sync(ref.workspace, sizeof(Workspace));
                return Outcome::UNCONFIRMED;
            }
        }
        if (Io::now_ns() - r->submitted_ns >= ref.workspace->timeout_ns) {
            ref.workspace->admission = Admission::UNCONFIRMED;
            Io::sync(ref.workspace, sizeof(Workspace));
            return Outcome::UNCONFIRMED;
        }
        return Outcome::PENDING;
    }

    static SIMPLER_CCU_DEVICE bool retire(Reference ref) {
        const Outcome result = poll(ref);
        if (result != Outcome::SUCCEEDED && result != Outcome::FAILED) return false;
        if (ref.workspace->platform_context) {
            const auto accepted = platform_request(ref);
            ccu_retire_request(accepted);
            const auto *record = ccu_detail::Record(accepted);
            return record && record->slot_state == CcuSlotState::FREE;
        }
        auto *r = request(ref);
        r->slot = Slot::FREE;
        Io::sync(r, offsetof(Request, arguments));
        return true;
    }

    static SIMPLER_CCU_DEVICE bool synchronize_arguments(Reference ref) {
        if (ref.workspace && ref.workspace->platform_context) {
            auto *ctx = platform(ref);
            if (!ctx || !platform_request(ref).record_addr) return false;
            const auto *args = ccu_detail::At<CcuPositionalArguments>(ctx, ctx->arguments_offset);
            if (args->tensor_count > CCU_MAX_TENSORS) return false;
            for (uint32_t i = 0; i < args->tensor_count; ++i) {
                Io::sync(reinterpret_cast<SIMPLER_CCU_GM void *>(args->tensors[i].address), args->tensors[i].byte_size);
            }
            return true;
        }
        const auto *r = request(ref);
        if (!r || r->arguments.tensor_count > MAX_TENSORS) return false;
        for (uint32_t i = 0; i < r->arguments.tensor_count; ++i) {
            const auto &tensor = r->arguments.tensors[i];
            Io::sync(reinterpret_cast<SIMPLER_CCU_GM void *>(tensor.addr), tensor.bytes);
        }
        return true;
    }
};

}  // namespace simpler::ccu::device

#undef SIMPLER_CCU_DEVICE
#undef SIMPLER_CCU_GM
