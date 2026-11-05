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
#include "async_observer.h"
#include "scheduler/scheduler_async_check.h"
#include "scheduler/scheduler_async_cohort.h"

inline constexpr uint64_t ASYNC_RESOURCE_VERSION = 1;
inline constexpr uint64_t ASYNC_RESOURCE_NONE = UINT64_MAX;
inline constexpr uint32_t ASYNC_CORE_AIC = 1;
inline constexpr uint32_t ASYNC_CORE_AIV = 2;

enum AsyncRangeAccess : uint64_t {
    ASYNC_RANGE_READ = 1,
    ASYNC_RANGE_WRITE = 2,
    ASYNC_RANGE_CONDITION_CLEAN = 4,
    ASYNC_RANGE_CONSUMER_CLEAN = 8,
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

// One declared condition capacity slot of an actual physical execution. The
// payload relationship is explicit; an address in a workspace is insufficient.
// A record span contains exactly record_count scalar slots at record_stride.
// SDMA's committed-post metadata following its public records is not a slot.
struct AsyncConditionContract {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint32_t condition_index{0};
    uint32_t observer_cores{0};
    int32_t completion_type{-1};
    uint32_t payload_none{0};
    uint64_t record_range{ASYNC_RESOURCE_NONE};
    uint64_t record_count{0};
    uint64_t record_stride{0};
    uint64_t source_range{ASYNC_RESOURCE_NONE};
    uint64_t payload_range{ASYNC_RESOURCE_NONE};
};

enum class AsyncConsumerMapKind : uint32_t { UNKNOWN, NONE, REQUIRED };

struct AsyncConsumerTaskMap {
    uint64_t contract_base{ASYNC_RESOURCE_NONE};
    uint32_t block_count{0};
    uint32_t lane_mask{0};
    AsyncConsumerMapKind kind{AsyncConsumerMapKind::UNKNOWN};
    uint32_t reserved{0};
};

struct AsyncConsumerContract {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint64_t range_begin{0};
    uint64_t range_count{0};
    uint32_t consumer_cores{0};
    uint32_t reserved{0};
};

// Private sections: resources = header, spans, ranges; uses = conditions;
// consumers = task maps, execution contracts, uint64 range indices. All are
// immutable after Host publication and share the existing scheduler owner.
struct AsyncResourceHeader {
    uint64_t version{ASYNC_RESOURCE_VERSION};
    uint64_t run_epoch{0};
    uint64_t span_count{0};
    uint64_t range_count{0};
    uint64_t condition_count{0};
    uint64_t consumer_count{0};
    uint64_t consumer_range_count{0};
    uint64_t reserved{0};
};

static_assert(sizeof(AsyncResourceHeader) == 64 && sizeof(AsyncRetainedSpan) == 16);
static_assert(sizeof(AsyncResourceRange) == 32 && sizeof(AsyncConditionContract) == 72);
static_assert(sizeof(AsyncConsumerTaskMap) == 24 && sizeof(AsyncConsumerContract) == 40);

namespace async_resource_detail {
template <typename T>
inline __aicore__ T load(__gm__ const T *source) {
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % 8 == 0 && alignof(T) >= 8);
    T result{};
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
        __builtin_memcpy(reinterpret_cast<uint8_t *>(&result) + offset, &word, 8);
    }
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
    SchedulerAsyncCohortView cohorts{};
    AsyncResourceHeader header{};
    uint64_t task_count{0};
    uint64_t uses_offset{0};
    __gm__ const AsyncRetainedSpan *spans{nullptr};
    __gm__ const AsyncResourceRange *ranges{nullptr};
    __gm__ const AsyncConditionContract *conditions{nullptr};
    __gm__ const AsyncConsumerTaskMap *consumer_maps{nullptr};
    __gm__ const AsyncConsumerContract *consumers{nullptr};
    __gm__ const uint64_t *consumer_ranges{nullptr};

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
            !section(layout.total_bytes, layout.uses_offset, layout.uses_bytes) ||
            !section(layout.total_bytes, layout.consumers_offset, layout.consumers_bytes))
            return false;
        if (layout.resources_offset + layout.resources_bytes > layout.uses_offset ||
            layout.uses_offset + layout.uses_bytes > layout.consumers_offset ||
            layout.consumers_offset + layout.consumers_bytes > layout.scratch_offset)
            return false;
        const auto h = load(reinterpret_cast<__gm__ const AsyncResourceHeader *>(base + layout.resources_offset));
        if (h.version != ASYNC_RESOURCE_VERSION || h.run_epoch != layout.run_epoch || h.reserved != 0 ||
            h.span_count == 0 || h.range_count == 0 || h.condition_count == 0 || layout.task_count == 0 ||
            layout.task_count > INT32_MAX)
            return false;
        uint64_t resource_bytes = sizeof(h), use_bytes = 0, consumer_bytes = 0;
        if (!add_rows(h.span_count, sizeof(AsyncRetainedSpan), resource_bytes) ||
            !add_rows(h.range_count, sizeof(AsyncResourceRange), resource_bytes) ||
            !add_rows(h.condition_count, sizeof(AsyncConditionContract), use_bytes) ||
            !add_rows(layout.task_count, sizeof(AsyncConsumerTaskMap), consumer_bytes) ||
            !add_rows(h.consumer_count, sizeof(AsyncConsumerContract), consumer_bytes) ||
            !add_rows(h.consumer_range_count, sizeof(uint64_t), consumer_bytes) ||
            resource_bytes != layout.resources_bytes || use_bytes != layout.uses_bytes ||
            consumer_bytes != layout.consumers_bytes)
            return false;
        AsyncResourceTables next{};
        if (!SchedulerAsyncCohortView::mount(base, layout, &next.cohorts)) return false;
        next.header = h;
        next.task_count = layout.task_count;
        next.uses_offset = layout.uses_offset;
        next.spans = reinterpret_cast<__gm__ const AsyncRetainedSpan *>(base + layout.resources_offset + sizeof(h));
        next.ranges = reinterpret_cast<__gm__ const AsyncResourceRange *>(next.spans + h.span_count);
        next.conditions = reinterpret_cast<__gm__ const AsyncConditionContract *>(base + layout.uses_offset);
        next.consumer_maps = reinterpret_cast<__gm__ const AsyncConsumerTaskMap *>(base + layout.consumers_offset);
        next.consumers = reinterpret_cast<__gm__ const AsyncConsumerContract *>(next.consumer_maps + layout.task_count);
        next.consumer_ranges = reinterpret_cast<__gm__ const uint64_t *>(next.consumers + h.consumer_count);
        *out = next;
        return true;
    }
};

struct AsyncConsumerIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint64_t dispatch_generation{0};
    uint64_t core_write_serial{0};
};

struct AsyncConsumerCursor {
    AsyncConsumerIdentity identity{};
    SchedulerAsyncCohortCursor cohort{};
    uint64_t range_index{0};
    uint64_t line_offset{0};
    bool initialized{false};
};

enum class AsyncConsumerResult : uint8_t { PENDING, READY, ERROR };

struct AsyncConsumerScalarCache {
    inline __aicore__ void invalidate_line(uint64_t address) {
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ void *>(address));
    }
    inline __aicore__ void finish() { scheduler_cache_barrier(); }
};

template <typename Reader = AsyncCompletionScalarReader, typename Cache = AsyncConsumerScalarCache>
class AsyncResourceBindings {
public:
    using ConsumerIdentity = AsyncConsumerIdentity;
    using ConsumerCursor = AsyncConsumerCursor;
    using ConsumerResult = AsyncConsumerResult;
    inline __aicore__ AsyncResourceBindings(const AsyncResourceTables &tables, uint32_t core_kind) :
        tables_(tables),
        core_(core_kind) {}
    inline __aicore__
    AsyncResourceBindings(const AsyncResourceTables &tables, uint32_t core_kind, Reader reader, Cache cache) :
        tables_(tables),
        core_(core_kind),
        reader_(reader),
        cache_(cache) {}

    inline __aicore__ bool bind_condition(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        SchedulerAsyncConditionBinding *out
    ) {
        uint64_t index = 0;
        AsyncConditionContract use{};
        if (out == nullptr || !condition_contract(identity, condition, &index, &use) || !entry_valid(use, entry))
            return false;
        *out = {index, 0};
        return true;
    }

    inline __aicore__ bool validate(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &binding
    ) {
        uint64_t index = 0;
        AsyncConditionContract use{};
        return binding.owner_generation == 0 && condition_contract(identity, condition, &index, &use) &&
               binding.binding_index == index && entry_valid(use, entry);
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
        // belongs to prepare_consumer on the core that will actually read it.
        cache_.finish();
        return {SchedulerAsyncConditionState::READY, 0};
    }

    // Before the backend submit, validate every possible returned condition slot
    // and both data ranges, then reserve the existing single-writer capacity.
    // The public event's actual records must still pass bind_condition at seal.
    inline __aicore__ bool prepare_sdma_submit(
        const SchedulerAsyncGroupIdentity &identity, AsyncCtx &ctx, uint64_t source, uint64_t destination,
        uint64_t bytes, uint32_t record_upper_bound, AsyncCompletionAllowance *allowance
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
        for (uint32_t n = 0; n < record_upper_bound; ++n) {
            uint64_t index = 0;
            AsyncConditionContract use{};
            AsyncRetainedSpan ignored{};
            uint64_t address = 0, length = 0;
            if (!condition_contract(identity, begin + n, &index, &use) ||
                use.completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD || use.payload_none != 0 ||
                !range(use.source_range, ASYNC_RANGE_READ, &address, &length, &ignored) ||
                !async_resource_detail::contains(address, length, source, bytes) ||
                !range(use.payload_range, ASYNC_RANGE_WRITE, &address, &length, &ignored) ||
                !async_resource_detail::contains(address, length, destination, bytes))
                return reject_submit(ctx);
        }
        return async_completion_reserve(ctx, record_upper_bound, allowance);
    }

    // The caller supplies a monotonically changing local write serial whenever
    // a business kernel runs. Any such change discards partial/finished acquire
    // work. READY must be followed by admission/kernel entry without another
    // business kernel on this core. No cohort permit is held while PENDING.
    inline __aicore__ AsyncConsumerResult
    prepare_consumer(const AsyncConsumerIdentity &identity, AsyncConsumerCursor &cursor, uint32_t &line_budget) {
        using namespace async_resource_detail;
        if (!valid_tables() || identity.run_epoch != tables_.header.run_epoch || identity.task_ref < 0 ||
            static_cast<uint64_t>(identity.task_ref) >= tables_.task_count || identity.dispatch_generation == 0 ||
            identity.core_write_serial == UINT64_MAX || identity.subtask_slot >= 3)
            return AsyncConsumerResult::ERROR;
        const auto map = load(tables_.consumer_maps + identity.task_ref);
        if (map.reserved != 0 || map.block_count == 0 || map.block_count > INT16_MAX ||
            identity.block_idx >= map.block_count || map.lane_mask == 0 || (map.lane_mask & ~7u) != 0 ||
            (map.lane_mask & (1u << identity.subtask_slot)) == 0)
            return AsyncConsumerResult::ERROR;
        if (cursor.cohort.arrived && (!cursor.initialized || !same_identity(cursor.identity, identity)))
            return AsyncConsumerResult::ERROR;
        if (!cursor.initialized || !same_identity(cursor.identity, identity)) {
            cursor = {};
            cursor.identity = identity;
            cursor.initialized = true;
        }
        if (map.kind == AsyncConsumerMapKind::NONE) {
            if (map.contract_base != ASYNC_RESOURCE_NONE) return AsyncConsumerResult::ERROR;
            return prepare_cohort(identity, cursor, line_budget);
        }
        const uint64_t lanes = __builtin_popcount(map.lane_mask);
        const uint64_t count = lanes * map.block_count;
        if (map.kind != AsyncConsumerMapKind::REQUIRED || map.contract_base >= tables_.header.consumer_count ||
            count > tables_.header.consumer_count - map.contract_base)
            return AsyncConsumerResult::ERROR;
        const uint64_t index = map.contract_base + identity.block_idx * lanes +
                               __builtin_popcount(map.lane_mask & ((1u << identity.subtask_slot) - 1));
        const auto use = load(tables_.consumers + index);
        if (use.task_ref != identity.task_ref || use.block_idx != identity.block_idx ||
            use.subtask_slot != identity.subtask_slot || use.reserved != 0 || !core(use.consumer_cores, core_) ||
            use.range_count == 0 || use.range_begin > tables_.header.consumer_range_count ||
            use.range_count > tables_.header.consumer_range_count - use.range_begin)
            return AsyncConsumerResult::ERROR;
        if (cursor.range_index > use.range_count || (cursor.line_offset & 63u) != 0 ||
            (cursor.range_index == use.range_count && cursor.line_offset != 0))
            return AsyncConsumerResult::ERROR;
        uint32_t available = line_budget < 8 ? line_budget : 8;
        while (cursor.range_index < use.range_count && available != 0) {
            const uint64_t range_index = load(tables_.consumer_ranges + use.range_begin + cursor.range_index);
            AsyncRetainedSpan retained{};
            uint64_t address = 0, bytes = 0, first = 0, length = 0;
            if (!range(range_index, ASYNC_RANGE_READ | ASYNC_RANGE_CONSUMER_CLEAN, &address, &bytes, &retained) ||
                !cache_lines(retained, address, bytes, &first, &length) || cursor.line_offset >= length) {
                cache_.finish();
                return AsyncConsumerResult::ERROR;
            }
            cache_.invalidate_line(first + cursor.line_offset);
            --available;
            --line_budget;
            cursor.line_offset += 64;
            if (cursor.line_offset == length) {
                cursor.line_offset = 0;
                ++cursor.range_index;
            }
        }
        cache_.finish();
        if (cursor.range_index != use.range_count) return AsyncConsumerResult::PENDING;
        const uint32_t before_cohort = available;
        const auto result = prepare_cohort(identity, cursor, available);
        line_budget -= before_cohort - available;
        return result;
    }

    inline __aicore__ bool consumer_pinned(const ConsumerCursor &cursor) const { return cursor.cohort.arrived; }

private:
    inline __aicore__ AsyncConsumerResult
    prepare_cohort(const AsyncConsumerIdentity &identity, AsyncConsumerCursor &cursor, uint32_t &budget) {
        const auto result = scheduler_async_cohort_prepare(tables_.cohorts, identity, cursor.cohort, budget);
        return result == SchedulerAsyncCohortResult::READY   ? AsyncConsumerResult::READY :
               result == SchedulerAsyncCohortResult::PENDING ? AsyncConsumerResult::PENDING :
                                                               AsyncConsumerResult::ERROR;
    }

    AsyncResourceTables tables_;
    uint32_t core_;
    Reader reader_{};
    Cache cache_{};

    inline __aicore__ bool valid_tables() const {
        return tables_.header.version == ASYNC_RESOURCE_VERSION && tables_.header.run_epoch != 0 &&
               tables_.header.reserved == 0 && tables_.header.condition_count != 0 && tables_.task_count != 0 &&
               tables_.cohorts.base != nullptr && tables_.cohorts.header.run_epoch == tables_.header.run_epoch &&
               tables_.spans != nullptr && tables_.ranges != nullptr && tables_.conditions != nullptr &&
               tables_.consumer_maps != nullptr && tables_.consumers != nullptr && tables_.consumer_ranges != nullptr &&
               tables_.header.span_count != 0 && tables_.header.range_count != 0 &&
               async_resource_detail::core(3, core_);
    }

    static inline __aicore__ SchedulerAsyncConditionOutcome failed() {
        return {SchedulerAsyncConditionState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    static inline __aicore__ bool reject_submit(AsyncCtx &ctx) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    static inline __aicore__ bool same_identity(const AsyncConsumerIdentity &a, const AsyncConsumerIdentity &b) {
        return a.run_epoch == b.run_epoch && a.task_ref == b.task_ref && a.block_idx == b.block_idx &&
               a.subtask_slot == b.subtask_slot && a.dispatch_generation == b.dispatch_generation &&
               a.core_write_serial == b.core_write_serial;
    }

    inline __aicore__ bool
    range(uint64_t index, uint64_t access, uint64_t *address, uint64_t *bytes, AsyncRetainedSpan *retained) const {
        using namespace async_resource_detail;
        if (index >= tables_.header.range_count) return false;
        const auto row = load(tables_.ranges + index);
        if (row.span_index >= tables_.header.span_count || row.bytes == 0 || (row.access & ~UINT64_C(15)) != 0 ||
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

    inline __aicore__ bool condition_contract(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, uint64_t *index, AsyncConditionContract *use
    ) const {
        using namespace async_resource_detail;
        if (!valid_tables() || identity.run_epoch != tables_.header.run_epoch || identity.task_ref < 0 ||
            static_cast<uint64_t>(identity.task_ref) >= tables_.task_count || identity.subtask_slot >= 3 ||
            identity.capacity == 0 || identity.capacity > CORE_ASYNC_MAX_CONDITIONS || condition >= identity.capacity ||
            identity.binding_use_count != identity.capacity || identity.binding_uses_offset < tables_.uses_offset)
            return false;
        const uint64_t delta = identity.binding_uses_offset - tables_.uses_offset;
        if (delta % sizeof(*use) != 0) return false;
        const uint64_t begin = delta / sizeof(*use);
        if (begin > tables_.header.condition_count || identity.capacity > tables_.header.condition_count - begin)
            return false;
        *index = begin + condition;
        *use = load(tables_.conditions + *index);
        if (use->task_ref != identity.task_ref || use->block_idx != identity.block_idx ||
            use->subtask_slot != identity.subtask_slot || use->condition_index != condition ||
            !core(use->observer_cores, core_) || use->payload_none > 1)
            return false;
        AsyncRetainedSpan retained{};
        uint64_t address = 0, bytes = 0;
        const uint64_t width = use->completion_type == COMPLETION_TYPE_COUNTER ? 4 : 8;
        const uint32_t condition_type = use->completion_type == COMPLETION_TYPE_COUNTER ?
                                            CORE_ASYNC_CONDITION_COUNTER :
                                            CORE_ASYNC_CONDITION_SDMA_POST_DONE;
        if ((use->completion_type != COMPLETION_TYPE_COUNTER &&
             use->completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD) ||
            (identity.condition_types & condition_type) == 0 ||
            !range(use->record_range, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN, &address, &bytes, &retained) ||
            address % width != 0 || use->record_count == 0 || use->record_stride < width ||
            use->record_stride % width != 0 || use->record_count - 1 > (UINT64_MAX - width) / use->record_stride ||
            (use->record_count - 1) * use->record_stride + width != bytes)
            return false;
        if (use->completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD &&
            (use->payload_none != 0 || !range(use->source_range, ASYNC_RANGE_READ, &address, &bytes, &retained)))
            return false;
        if (use->payload_none != 0) return use->payload_range == ASYNC_RESOURCE_NONE;
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
        AsyncRetainedSpan retained{};
        uint64_t address = 0, bytes = 0, first = 0, length = 0;
        const uint64_t width = entry.completion_type == COMPLETION_TYPE_COUNTER ? 4 : 8;
        return range(use.record_range, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN, &address, &bytes, &retained) &&
               entry.addr >= address && (entry.addr - address) % use.record_stride == 0 &&
               (entry.addr - address) / use.record_stride < use.record_count &&
               cache_lines(retained, entry.addr, width, &first, &length);
    }
};
