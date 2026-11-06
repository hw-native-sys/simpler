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

#include "async_completion_writer.h"
#include "scheduler/scheduler_local_storage.h"
#include "async_observer.h"
#include "scheduler/scheduler_async_check.h"
#include "scheduler/scheduler_submission_provenance.h"
#include "host_build_graph/async_retirement.h"

inline constexpr uint64_t ASYNC_RESOURCE_VERSION = 5;
inline constexpr uint64_t ASYNC_EXECUTION_UNQUALIFIED = HBG_ASYNC_CONTRACT_UNQUALIFIED;
inline constexpr uint64_t ASYNC_EXECUTION_CORE_PIPELINES_ONLY = HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY;

inline __aicore__ bool async_execution_contract_valid(uint64_t contract) {
    return contract == ASYNC_EXECUTION_UNQUALIFIED || contract == ASYNC_EXECUTION_CORE_PIPELINES_ONLY;
}
inline constexpr uint64_t ASYNC_RESOURCE_NONE = UINT64_MAX;
inline constexpr uint32_t ASYNC_CORE_AIC = 1;
inline constexpr uint32_t ASYNC_CORE_AIV = 2;

enum AsyncRangeAccess : uint64_t {
    ASYNC_RANGE_READ = 1,
    ASYNC_RANGE_WRITE = 2,
    ASYNC_RANGE_CONDITION_CLEAN = 4,
};

// Host exports these ranges from owners already retained by this run/domain.
// These rows neither retain allocations nor prove lifetime by their presence.
// CLEAN grants include the entire touched cache lines, not only logical bytes.
struct AsyncRetainedSpan {
    uint64_t address{0};
    uint64_t bytes{0};
};

struct AsyncResourceRange {
    uint64_t span_index{ASYNC_RESOURCE_NONE};
    uint64_t byte_offset{0};
    uint64_t bytes{0};
    uint64_t access{0};
};

// One allowed use of an actual physical execution, independent of the order in
// which its kernel registers conditions. Host sorts disjoint record spans by
// address and coalesces aliases, preserving every associated payload range.
// A record span contains exactly record_count scalar slots at record_stride.
// SDMA's committed-post metadata following its public records is not a slot.
struct AsyncConditionContract {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint32_t use_index{0};
    uint32_t observer_cores{0};
    int32_t completion_type{-1};
    uint32_t payload_none{0};
    uint64_t record_range{ASYNC_RESOURCE_NONE};
    uint64_t record_count{0};
    uint64_t record_stride{0};
    uint64_t source_range{ASYNC_RESOURCE_NONE};
    uint64_t payload_range{ASYNC_RESOURCE_NONE};
    uint64_t payload_count{1};
    // SDMA use-to-owner association, never a directory of allowed records.
    // COUNTER uses keep this zero and retain their exact static record slots.
    uint64_t sdma_workspace_address{0};
};

// Private immutable resources and condition contracts share the existing scheduler owner.
struct AsyncResourceHeader {
    uint64_t version{ASYNC_RESOURCE_VERSION};
    uint64_t run_epoch{0};
    uint64_t span_count{0};
    uint64_t range_count{0};
    uint64_t condition_count{0};
    uint64_t execution_contract{ASYNC_EXECUTION_UNQUALIFIED};
};

static_assert(sizeof(AsyncResourceHeader) == 48 && sizeof(AsyncRetainedSpan) == 16);
static_assert(sizeof(AsyncResourceRange) == 32 && sizeof(AsyncConditionContract) == 88);

namespace async_resource_detail {
template <typename T>
inline __aicore__ void load_into(__gm__ const T *source, T *destination) {
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % 8 == 0 && alignof(T) >= 8);
    for (uint32_t offset = 0; offset < sizeof(T); offset += 8) {
        uint64_t word = 0;
#if defined(__CCE_AICORE__)
        auto *address = reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ uint8_t *>(reinterpret_cast<__gm__ const uint8_t *>(source) + offset)
        );
        word = static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
#else
        __builtin_memcpy(&word, reinterpret_cast<const uint8_t *>(source) + offset, 8);
#endif
        __builtin_memcpy(reinterpret_cast<uint8_t *>(destination) + offset, &word, 8);
    }
}

template <typename T>
inline __aicore__ T load(__gm__ const T *source) {
    static_assert(alignof(T) <= 8, "Over-aligned snapshots require qualified local storage and load_into");
    T result{};
    load_into(source, &result);
    return result;
}

inline __aicore__ bool add_rows(uint64_t count, uint64_t stride, uint64_t &bytes) {
    return count <= (UINT64_MAX - bytes) / stride && (bytes += count * stride, true);
}

inline __aicore__ bool contains(uint64_t begin, uint64_t size, uint64_t address, uint64_t bytes) {
    return size <= UINT64_MAX - begin && address >= begin && address - begin <= size &&
           bytes <= size - (address - begin);
}

inline __aicore__ bool core(uint32_t allowed, uint32_t current) {
    return (current == ASYNC_CORE_AIC || current == ASYNC_CORE_AIV) && allowed != 0 && (allowed & ~3u) == 0 &&
           (allowed & current) != 0;
}

inline __aicore__ bool section(uint64_t total_bytes, uint64_t offset, uint64_t bytes) {
    return offset >= sizeof(SchedulerAsyncLayout) && offset % 128 == 0 && contains(0, total_bytes, offset, bytes);
}
}  // namespace async_resource_detail

struct AsyncResourceTables {
    AsyncResourceHeader header{};
    uint64_t task_count{0};
    uint64_t uses_offset{0};
    __gm__ const AsyncRetainedSpan *spans{nullptr};
    __gm__ const AsyncResourceRange *ranges{nullptr};
    __gm__ const AsyncConditionContract *conditions{nullptr};

    // Mount checks format, not owner qualification. Only the retained-owner Host
    // exporter may populate this image. An absent table is never a permissive backend.
    static inline __aicore__ bool
    mount(__gm__ const uint8_t *base, const SchedulerAsyncLayout &layout, AsyncResourceTables *out) {
        using namespace async_resource_detail;
        if (base == nullptr || out == nullptr || reinterpret_cast<uint64_t>(base) % 128 != 0 ||
            layout.version != SCHEDULER_ASYNC_LAYOUT_VERSION || layout.run_epoch == 0 || layout.group_count == 0 ||
            layout.total_bytes > UINT64_MAX - reinterpret_cast<uint64_t>(base))
            return false;
        if (layout.resources_bytes < sizeof(AsyncResourceHeader) ||
            !section(layout.total_bytes, layout.resources_offset, layout.resources_bytes) ||
            (layout.uses_offset != 0 && !section(layout.total_bytes, layout.uses_offset, layout.uses_bytes)) ||
            (layout.uses_bytes != 0 && layout.uses_offset == 0) ||
            (layout.scratch_offset != 0 && !section(layout.total_bytes, layout.scratch_offset, layout.scratch_bytes)) ||
            (layout.scratch_bytes != 0 && layout.scratch_offset == 0))
            return false;
        const uint64_t resource_end = layout.resources_offset + layout.resources_bytes;
        const uint64_t use_end = layout.uses_offset == 0 ? resource_end : layout.uses_offset + layout.uses_bytes;
        if ((layout.uses_offset != 0 && resource_end > layout.uses_offset) ||
            (layout.scratch_offset != 0 && use_end > layout.scratch_offset))
            return false;
        const auto h = load(reinterpret_cast<__gm__ const AsyncResourceHeader *>(base + layout.resources_offset));
        if (h.version != ASYNC_RESOURCE_VERSION || h.run_epoch != layout.run_epoch ||
            !async_execution_contract_valid(h.execution_contract) || layout.task_count == 0 ||
            layout.task_count > INT32_MAX)
            return false;
        uint64_t resource_bytes = sizeof(h), use_bytes = 0;
        if (!add_rows(h.span_count, sizeof(AsyncRetainedSpan), resource_bytes) ||
            !add_rows(h.range_count, sizeof(AsyncResourceRange), resource_bytes) ||
            !add_rows(h.condition_count, sizeof(AsyncConditionContract), use_bytes) ||
            resource_bytes != layout.resources_bytes || use_bytes != layout.uses_bytes)
            return false;
        AsyncResourceTables next{};
        next.header = h;
        next.task_count = layout.task_count;
        next.uses_offset = layout.uses_offset;
        next.spans = reinterpret_cast<__gm__ const AsyncRetainedSpan *>(base + layout.resources_offset + sizeof(h));
        next.ranges = reinterpret_cast<__gm__ const AsyncResourceRange *>(next.spans + h.span_count);
        next.conditions = reinterpret_cast<__gm__ const AsyncConditionContract *>(base + layout.uses_offset);
        *out = next;
        return true;
    }
};

struct AsyncObservationBarrier {
    inline __aicore__ void finish() { scheduler_cache_barrier(); }
};

template <typename Reader = AsyncCompletionScalarReader, typename Cache = AsyncObservationBarrier>
class AsyncResourceBindings {
public:
    inline __aicore__ AsyncResourceBindings(
        const AsyncResourceTables &tables, uint32_t core_kind,
        const SchedulerSubmissionProvenanceView *provenance = nullptr
    ) :
        tables_(tables),
        core_(core_kind),
        provenance_(provenance) {}
    inline __aicore__ AsyncResourceBindings(
        const AsyncResourceTables &tables, uint32_t core_kind, Reader reader, Cache cache,
        const SchedulerSubmissionProvenanceView *provenance = nullptr
    ) :
        tables_(tables),
        core_(core_kind),
        reader_(reader),
        cache_(cache),
        provenance_(provenance) {}

    inline __aicore__ bool bind_condition(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        SchedulerAsyncConditionBinding *out
    ) {
        uint64_t begin = 0, count = 0;
        AsyncConditionContract use{};
        if (out == nullptr || entry.completion_type != COMPLETION_TYPE_COUNTER || condition >= identity.capacity ||
            !use_bounds(identity, &begin, &count) || count == 0)
            return false;
        // A uint64-sized sorted set requires at most 64 comparisons. No scan
        // depends on the number of declared resources or on registration order.
        uint64_t lower = 0, upper = count;
        while (lower < upper) {
            const uint64_t middle = lower + (upper - lower) / 2;
            const auto candidate = async_resource_detail::load(tables_.conditions + begin + middle);
            if (candidate.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) {
                upper = middle;
                continue;
            }
            if (!condition_contract(identity, begin + middle, &use)) return false;
            uint64_t address = 0, bytes = 0;
            AsyncRetainedSpan retained{};
            if (!range(use.record_range, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN, &address, &bytes, &retained))
                return false;
            if (address <= entry.addr) lower = middle + 1;
            else upper = middle;
        }
        if (lower == 0) return false;
        const uint64_t index = begin + lower - 1;
        if (!condition_contract(identity, index, &use) || !entry_valid(use, entry)) return false;
        *out = {index, 0};
        return true;
    }

    // Only the original executor supplies this identity. CHECK never uses this
    // method and cannot accept an unsealed context as a completed publication.
    inline __aicore__ bool bind_condition_at_seal(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        const SchedulerSubmissionSealIdentity &sealing, SchedulerAsyncConditionBinding *out
    ) {
        if (entry.completion_type == COMPLETION_TYPE_COUNTER) return bind_condition(identity, condition, entry, out);
        if (out == nullptr || provenance_ == nullptr || condition >= identity.capacity ||
            sealing.run_epoch != identity.run_epoch || sealing.task_ref != identity.task_ref ||
            sealing.block_idx != identity.block_idx || sealing.subtask_slot != identity.subtask_slot)
            return false;
        SchedulerSubmissionProvenance record{};
        SchedulerSubmissionWorkspace workspace{};
        AsyncConditionContract use{};
        if (!scheduler_submission_read_seal_provenance(*provenance_, sealing, condition, entry, &record, &workspace) ||
            !condition_contract(identity, record.use_index, &use) || !entry_valid(use, entry) ||
            use.sdma_workspace_address != workspace.address)
            return false;
        *out = {record.use_index, 0};
        return true;
    }

    inline __aicore__ bool validate(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &binding
    ) {
        AsyncConditionContract use{};
        if (condition >= identity.capacity || binding.owner_generation != 0 ||
            !condition_contract(identity, binding.binding_index, &use) || !entry_valid(use, entry))
            return false;
        if (entry.completion_type == COMPLETION_TYPE_COUNTER) return true;
        SchedulerSubmissionProvenance record{};
        SchedulerSubmissionWorkspace workspace{};
        return provenance_ != nullptr &&
               scheduler_submission_read_sealed_provenance(
                   *provenance_, identity, condition, entry, binding.binding_index, &record, &workspace
               ) &&
               use.sdma_workspace_address == workspace.address;
    }

    inline __aicore__ SchedulerAsyncConditionOutcome observe_once(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &binding
    ) {
        if (!validate(identity, condition, entry, binding)) return failed();
        const auto result = async_observe_once(entry, reader_);
        if (result.state == CompletionPollState::FAILED)
            return {SchedulerAsyncConditionState::FAILED, result.error_code};
        return {
            result.state == CompletionPollState::READY ? SchedulerAsyncConditionState::READY :
                                                         SchedulerAsyncConditionState::PENDING,
            0
        };
    }

    inline __aicore__ SchedulerAsyncConditionOutcome acquire_completed(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &binding, uint64_t &stage, uint64_t &offset
    ) {
        if (stage != 0 || offset != 0 || !validate(identity, condition, entry, binding)) return failed();
        // This completes observation-side ordering only. Payload acquisition
        // belongs to the common kernel entry on the core that will actually read it.
        cache_.finish();
        return {SchedulerAsyncConditionState::READY, 0};
    }

    // Before the backend submit, validate every possible returned condition slot
    // and both data ranges, then reserve the existing single-writer capacity.
    // The public event's actual records must still pass bind_condition at seal.
    inline __aicore__ bool prepare_sdma_submit(
        const SchedulerAsyncGroupIdentity &identity, AsyncCtx &ctx, uint64_t source, uint64_t destination,
        uint64_t bytes, uint32_t record_upper_bound, AsyncCompletionAllowance *allowance,
        uint64_t allowed_use_index = ASYNC_RESOURCE_NONE, uint64_t workspace_address = 0
    ) {
        if (allowance != nullptr) *allowance = {};
        // A5's SDMA submit implementation is AIV-only. Observer qualification
        // independently permits an AIC to check its resulting completion record.
        if (core_ != ASYNC_CORE_AIV || !async_completion_writer_valid(ctx) ||
            ctx.completion_capacity != identity.capacity || record_upper_bound == 0 ||
            record_upper_bound > CORE_ASYNC_MAX_CONDITIONS || allowance == nullptr)
            return reject_submit(ctx);
        const uint32_t begin = *ctx.completion_count;
        if (begin > identity.capacity || record_upper_bound > identity.capacity - begin) return reject_submit(ctx);
        uint64_t use_begin = 0, use_count = 0;
        if (!use_bounds(identity, &use_begin, &use_count)) return reject_submit(ctx);
        if (allowed_use_index == ASYNC_RESOURCE_NONE) {
            if (use_count != 1) return reject_submit(ctx);
            allowed_use_index = use_begin;
        }
        AsyncConditionContract use{};
        AsyncRetainedSpan ignored{};
        uint64_t address = 0, length = 0;
        if (!condition_contract(identity, allowed_use_index, &use) ||
            use.completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD || use.payload_none != 0 ||
            workspace_address == 0 || use.sdma_workspace_address != workspace_address ||
            !range(use.source_range, ASYNC_RANGE_READ, &address, &length, &ignored) ||
            !async_resource_detail::contains(address, length, source, bytes) ||
            !range(use.payload_range, ASYNC_RANGE_WRITE, &address, &length, &ignored) ||
            !async_resource_detail::contains(address, length, destination, bytes))
            return reject_submit(ctx);
        return async_completion_reserve(ctx, record_upper_bound, allowance);
    }

private:
    AsyncResourceTables tables_;
    uint32_t core_;
    Reader reader_{};
    Cache cache_{};
    const SchedulerSubmissionProvenanceView *provenance_{nullptr};

    inline __aicore__ bool valid_tables() const {
        return tables_.header.version == ASYNC_RESOURCE_VERSION && tables_.header.run_epoch != 0 &&
               async_execution_contract_valid(tables_.header.execution_contract) && tables_.task_count != 0 &&
               tables_.spans != nullptr && tables_.ranges != nullptr && tables_.conditions != nullptr &&
               async_resource_detail::core(3, core_);
    }

    static inline __aicore__ SchedulerAsyncConditionOutcome failed() {
        return {SchedulerAsyncConditionState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    static inline __aicore__ bool reject_submit(AsyncCtx &ctx) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    inline __aicore__ bool
    range(uint64_t index, uint64_t access, uint64_t *address, uint64_t *bytes, AsyncRetainedSpan *retained) const {
        using namespace async_resource_detail;
        if (index >= tables_.header.range_count) return false;
        const auto row = load(tables_.ranges + index);
        if (row.span_index >= tables_.header.span_count || row.bytes == 0 || (row.access & ~UINT64_C(7)) != 0 ||
            (row.access & access) != access)
            return false;
        *retained = load(tables_.spans + row.span_index);
        if (retained->address == 0 || retained->bytes > UINT64_MAX - retained->address ||
            row.byte_offset > retained->bytes || row.bytes > retained->bytes - row.byte_offset)
            return false;
        *address = retained->address + row.byte_offset;
        *bytes = row.bytes;
        return true;
    }

    static inline __aicore__ bool cache_lines(
        const AsyncRetainedSpan &retained, uint64_t address, uint64_t bytes, uint64_t *first, uint64_t *length
    ) {
        if (bytes == 0 || bytes > UINT64_MAX - address || address + bytes > UINT64_MAX - 63) return false;
        *first = address & ~UINT64_C(63);
        const uint64_t end = (address + bytes + 63) & ~UINT64_C(63);
        *length = end - *first;
        return async_resource_detail::contains(retained.address, retained.bytes, *first, *length);
    }

    inline __aicore__ bool
    use_bounds(const SchedulerAsyncGroupIdentity &identity, uint64_t *begin, uint64_t *count) const {
        using namespace async_resource_detail;
        if (!valid_tables() || identity.run_epoch != tables_.header.run_epoch || identity.task_ref < 0 ||
            static_cast<uint64_t>(identity.task_ref) >= tables_.task_count || identity.subtask_slot >= 3 ||
            identity.capacity == 0 || identity.capacity > CORE_ASYNC_MAX_CONDITIONS ||
            identity.binding_uses_offset < tables_.uses_offset)
            return false;
        const uint64_t delta = identity.binding_uses_offset - tables_.uses_offset;
        if (delta % sizeof(AsyncConditionContract) != 0) return false;
        *begin = delta / sizeof(AsyncConditionContract);
        *count = identity.binding_use_count;
        return *begin <= tables_.header.condition_count && *count <= tables_.header.condition_count - *begin;
    }

    inline __aicore__ bool
    condition_contract(const SchedulerAsyncGroupIdentity &identity, uint64_t index, AsyncConditionContract *use) const {
        using namespace async_resource_detail;
        uint64_t begin = 0, count = 0;
        if (!use_bounds(identity, &begin, &count) || index < begin || index - begin >= count) return false;
        *use = load(tables_.conditions + index);
        if (use->task_ref != identity.task_ref || use->block_idx != identity.block_idx ||
            use->subtask_slot != identity.subtask_slot || use->use_index != index - begin ||
            !core(use->observer_cores, core_) || use->payload_none > 1)
            return false;
        AsyncRetainedSpan retained{};
        uint64_t address = 0, bytes = 0;
        const bool counter = use->completion_type == COMPLETION_TYPE_COUNTER;
        const uint32_t condition_type = counter ? CORE_ASYNC_CONDITION_COUNTER : CORE_ASYNC_CONDITION_SDMA_POST_DONE;
        if ((!counter && use->completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD) ||
            (identity.condition_types & condition_type) == 0)
            return false;
        if (counter) {
            constexpr uint64_t width = sizeof(int32_t);
            if (use->sdma_workspace_address != 0 ||
                !range(
                    use->record_range, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN, &address, &bytes, &retained
                ) ||
                address % width != 0 || use->record_count == 0 || use->record_stride < width ||
                use->record_stride % width != 0 || use->record_count - 1 > (UINT64_MAX - width) / use->record_stride ||
                (use->record_count - 1) * use->record_stride + width != bytes)
                return false;
        } else if (tables_.header.execution_contract != ASYNC_EXECUTION_UNQUALIFIED ||
                   use->record_range != ASYNC_RESOURCE_NONE || use->record_count != 0 || use->record_stride != 0 ||
                   use->sdma_workspace_address == 0 || use->payload_none != 0 || use->payload_count != 1 ||
                   !range(use->source_range, ASYNC_RANGE_READ, &address, &bytes, &retained))
            return false;
        if (use->payload_none != 0) return use->payload_range == ASYNC_RESOURCE_NONE && use->payload_count == 0;
        if (use->payload_count == 0 || use->payload_range >= tables_.header.range_count ||
            use->payload_count > tables_.header.range_count - use->payload_range)
            return false;
        return range(
            use->payload_range, use->completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD ? ASYNC_RANGE_WRITE : 0,
            &address, &bytes, &retained
        );
    }

    inline __aicore__ bool entry_valid(const AsyncConditionContract &use, const DeferredCompletionEntry &entry) const {
        if (entry.engine != COMPLETION_ENGINE_SDMA || entry.completion_type != use.completion_type || entry._pad != 0 ||
            (entry.completion_type == COMPLETION_TYPE_COUNTER ?
                 (entry.backend_cookie != 0 || entry.expected_value > INT32_MAX) :
                 (entry.backend_cookie == 0 || entry.expected_value != 0)))
            return false;
        if (entry.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) return true;
        AsyncRetainedSpan retained{};
        uint64_t address = 0, bytes = 0, first = 0, length = 0;
        const uint64_t width = entry.completion_type == COMPLETION_TYPE_COUNTER ? 4 : 8;
        return range(use.record_range, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN, &address, &bytes, &retained) &&
               entry.addr >= address && (entry.addr - address) % use.record_stride == 0 &&
               (entry.addr - address) / use.record_stride < use.record_count &&
               cache_lines(retained, entry.addr, width, &first, &length);
    }
};
