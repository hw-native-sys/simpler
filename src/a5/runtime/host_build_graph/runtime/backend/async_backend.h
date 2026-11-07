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

#include "async_observer.h"
#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_async_check.h"

inline constexpr uint64_t ASYNC_BACKEND_VERSION = 1;

// Optional declarations are shared by a logical task's physical executions.
// They constrain registration/submission, never CHECK or dependency resolution.
struct AsyncBackendTask {
    uint64_t task_ref{0};
    uint64_t counter_begin{0};
    uint64_t counter_count{0};
    uint64_t transfer_begin{0};
    uint64_t transfer_count{0};
};

struct AsyncCounterRange {
    uint64_t address{0};
    uint64_t count{0};
    uint64_t stride{0};
};

struct AsyncSdmaTransfer {
    uint64_t workspace{0};
    uint64_t workspace_bytes{0};
    uint64_t source{0};
    uint64_t source_bytes{0};
    uint64_t destination{0};
    uint64_t destination_bytes{0};
    uint64_t queue_count{0};
};

struct AsyncBackendLayout {
    uint64_t version{ASYNC_BACKEND_VERSION};
    uint64_t total_bytes{0};
    uint64_t task_count{0};
    uint64_t counter_count{0};
    uint64_t transfer_count{0};
    uint64_t tasks_offset{0};
    uint64_t counters_offset{0};
    uint64_t transfers_offset{0};
};

static_assert(sizeof(AsyncBackendTask) == 40 && sizeof(AsyncCounterRange) == 24);
static_assert(sizeof(AsyncSdmaTransfer) == 56 && sizeof(AsyncBackendLayout) == 64);
static_assert(std::is_standard_layout_v<AsyncBackendLayout> && std::is_trivially_copyable_v<AsyncBackendLayout>);
static_assert(std::is_standard_layout_v<AsyncBackendTask> && std::is_trivially_copyable_v<AsyncBackendTask>);
static_assert(std::is_standard_layout_v<AsyncCounterRange> && std::is_trivially_copyable_v<AsyncCounterRange>);
static_assert(std::is_standard_layout_v<AsyncSdmaTransfer> && std::is_trivially_copyable_v<AsyncSdmaTransfer>);

namespace async_backend_detail {
inline __aicore__ bool contains(uint64_t base, uint64_t bytes, uint64_t address, uint64_t length) {
    return bytes <= UINT64_MAX - base && address >= base && address - base <= bytes &&
           length <= bytes - (address - base);
}
inline __aicore__ bool section(uint64_t total, uint64_t offset, uint64_t count, uint64_t stride) {
    return count == 0 ? offset == 0 :
                        offset >= sizeof(AsyncBackendLayout) && offset % 8 == 0 && offset <= total &&
                            count <= (total - offset) / stride;
}
}  // namespace async_backend_detail

// The run already owns this immutable allocation. Mount validates its image,
// and selected declarations are checked before any condition is observed or DMA
// is submitted. An absent task row means the existing kernel-owned contract.
struct AsyncBackendView {
    __gm__ const uint8_t *base{nullptr};
    AsyncBackendLayout layout{};

    inline __aicore__ bool mount(__gm__ const uint8_t *region, const SchedulerAsyncLayout &async) {
        if (async.version != SCHEDULER_ASYNC_LAYOUT_VERSION || async.backend_offset > async.total_bytes ||
            async.backend_bytes > async.total_bytes - async.backend_offset)
            return false;
        return mount(async.backend_bytes == 0 ? nullptr : region + async.backend_offset, async.backend_bytes);
    }

    inline __aicore__ bool mount(__gm__ const uint8_t *address, uint64_t bytes) {
        base = nullptr;
        layout = {};
        if (bytes == 0) return address == nullptr;
        if (address == nullptr || reinterpret_cast<uint64_t>(address) % 8 != 0 || bytes < sizeof(AsyncBackendLayout) ||
            bytes > UINT64_MAX - reinterpret_cast<uint64_t>(address))
            return false;
        const auto *header = reinterpret_cast<__gm__ const AsyncBackendLayout *>(address);
        layout.version = header->version;
        layout.total_bytes = header->total_bytes;
        layout.task_count = header->task_count;
        layout.counter_count = header->counter_count;
        layout.transfer_count = header->transfer_count;
        layout.tasks_offset = header->tasks_offset;
        layout.counters_offset = header->counters_offset;
        layout.transfers_offset = header->transfers_offset;
        using async_backend_detail::section;
        if (layout.version != ASYNC_BACKEND_VERSION || layout.total_bytes != bytes || layout.task_count == 0 ||
            !section(bytes, layout.tasks_offset, layout.task_count, sizeof(AsyncBackendTask)) ||
            !section(bytes, layout.counters_offset, layout.counter_count, sizeof(AsyncCounterRange)) ||
            !section(bytes, layout.transfers_offset, layout.transfer_count, sizeof(AsyncSdmaTransfer)))
            return false;
        const uint64_t tasks_end = layout.tasks_offset + layout.task_count * sizeof(AsyncBackendTask);
        const uint64_t counters_end = layout.counter_count == 0 ?
                                          tasks_end :
                                          layout.counters_offset + layout.counter_count * sizeof(AsyncCounterRange);
        if ((layout.counter_count != 0 && layout.counters_offset < tasks_end) ||
            (layout.transfer_count != 0 && layout.transfers_offset < counters_end))
            return false;
        base = address;
        return true;
    }

    inline __aicore__ bool task(uint64_t task_ref, AsyncBackendTask *out, bool *found) const {
        if (out == nullptr || found == nullptr) return false;
        *found = false;
        *out = {};
        if (base == nullptr) return true;
        const auto *tasks = reinterpret_cast<__gm__ const AsyncBackendTask *>(base + layout.tasks_offset);
        uint64_t first = 0, last = layout.task_count;
        while (first < last) {
            const uint64_t middle = first + (last - first) / 2;
            if (tasks[middle].task_ref < task_ref) first = middle + 1;
            else last = middle;
        }
        if (first == layout.task_count || tasks[first].task_ref != task_ref) return true;
        const auto &row = tasks[first];
        out->task_ref = row.task_ref;
        out->counter_begin = row.counter_begin;
        out->counter_count = row.counter_count;
        out->transfer_begin = row.transfer_begin;
        out->transfer_count = row.transfer_count;
        if (out->counter_begin > layout.counter_count ||
            out->counter_count > layout.counter_count - out->counter_begin ||
            out->transfer_begin > layout.transfer_count ||
            out->transfer_count > layout.transfer_count - out->transfer_begin)
            return false;
        *found = true;
        return true;
    }

    inline __aicore__ AsyncSdmaTransfer transfer(uint64_t index) const {
        const auto &row = reinterpret_cast<__gm__ const AsyncSdmaTransfer *>(base + layout.transfers_offset)[index];
        return {row.workspace,   row.workspace_bytes,   row.source,     row.source_bytes,
                row.destination, row.destination_bytes, row.queue_count};
    }

    inline __aicore__ bool counter(const AsyncBackendTask &task, uint64_t address) const {
        const auto *rows = reinterpret_cast<__gm__ const AsyncCounterRange *>(base + layout.counters_offset);
        uint64_t first = task.counter_begin, last = first + task.counter_count;
        while (first < last) {
            const uint64_t middle = first + (last - first) / 2;
            if (rows[middle].address <= address) first = middle + 1;
            else last = middle;
        }
        if (first == task.counter_begin) return false;
        const auto &row = rows[first - 1];
        return row.address != 0 && row.address % 4 == 0 && row.count != 0 && row.stride >= 4 && row.stride % 4 == 0 &&
               row.address <= UINT64_MAX - sizeof(int32_t) &&
               row.count - 1 <= (UINT64_MAX - sizeof(int32_t) - row.address) / row.stride && address >= row.address &&
               (address - row.address) % row.stride == 0 && (address - row.address) / row.stride < row.count;
    }

    inline __aicore__ bool select_transfer(
        uint64_t task_ref, uint64_t workspace, uint64_t source, uint64_t destination, uint64_t bytes,
        AsyncSdmaTransfer *out, bool *declared
    ) const {
        AsyncBackendTask row{};
        if (out == nullptr || !task(task_ref, &row, declared)) return false;
        *out = {};
        if (!*declared) return true;
        bool found = false;
        for (uint64_t i = 0; i < row.transfer_count; ++i) {
            const auto candidate = transfer(row.transfer_begin + i);
            if (candidate.workspace != workspace ||
                !async_backend_detail::contains(candidate.source, candidate.source_bytes, source, bytes) ||
                !async_backend_detail::contains(candidate.destination, candidate.destination_bytes, destination, bytes))
                continue;
            if (found || candidate.queue_count == 0 || candidate.queue_count > ASYNC_SDMA_CHANNELS ||
                candidate.workspace == 0 || candidate.workspace % 64 != 0 ||
                candidate.workspace_bytes > UINT64_MAX - candidate.workspace)
                return false;
            *out = candidate;
            found = true;
        }
        return found;
    }
};

struct AsyncObservationBarrier {
    inline __aicore__ void finish() { scheduler_cache_barrier(); }
};

template <typename Reader = AsyncCompletionScalarReader, typename Cache = AsyncObservationBarrier>
class AsyncCompletionBackend {
public:
    inline __aicore__ AsyncCompletionBackend(
        const AsyncBackendView &declarations, uint64_t execution_contract, Reader reader = {}, Cache cache = {}
    ) :
        declarations_(declarations),
        execution_contract_(execution_contract),
        reader_(reader),
        cache_(cache) {}

    inline __aicore__ bool validate_registration(
        const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &entry
    ) const {
        if (identity.task_ref < 0 || condition >= identity.capacity || !async_completion_entry_valid(entry) ||
            execution_contract_ > HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY)
            return false;
        const bool counter = entry.completion_type == COMPLETION_TYPE_COUNTER;
        if (!counter && execution_contract_ == HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY) return false;
        AsyncBackendTask row{};
        bool declared = false;
        if (!declarations_.task(identity.task_ref, &row, &declared)) return false;
        if (!declared) return true;
        if (counter) return declarations_.counter(row, entry.addr);
        for (uint64_t i = 0; i < row.transfer_count; ++i) {
            const auto transfer = declarations_.transfer(row.transfer_begin + i);
            if (async_backend_detail::contains(
                    transfer.workspace, transfer.workspace_bytes, entry.addr & ~UINT64_C(63), 64
                ))
                return true;
        }
        return false;
    }

    inline __aicore__ SchedulerAsyncConditionOutcome
    observe_once(const SchedulerAsyncGroupIdentity &, uint32_t, const DeferredCompletionEntry &entry) {
        const auto result = async_observe_once(entry, reader_);
        if (result.state == CompletionPollState::FAILED)
            return {SchedulerAsyncConditionState::FAILED, result.error_code};
        if (result.state == CompletionPollState::READY) cache_.finish();
        return {
            result.state == CompletionPollState::READY ? SchedulerAsyncConditionState::READY :
                                                         SchedulerAsyncConditionState::PENDING,
            0
        };
    }

private:
    AsyncBackendView declarations_;
    uint64_t execution_contract_;
    Reader reader_;
    Cache cache_;
};

// Only backend entry points interpret the execution's optional declarations and
// physical queue guards. The generic writer needs neither table.
inline __aicore__ bool async_backend_context(
    const AsyncCtx &ctx, AsyncBackendView *declarations, uint64_t *guard_address, uint64_t *contract
) {
    if (declarations == nullptr || guard_address == nullptr || contract == nullptr) return false;
    *declarations = {};
    *guard_address = 0;
    *contract = HBG_ASYNC_CONTRACT_UNQUALIFIED;
    if (ctx.execution_context == nullptr) return true;
    const auto &execution = *ctx.execution_context;
    if (execution.magic != ASYNC_EXECUTION_CONTEXT_MAGIC || execution.region_address == 0 ||
        execution.region_address % 128 != 0 || execution.subtask_slot >= 3)
        return false;
    const auto *layout = reinterpret_cast<__gm__ const SchedulerAsyncLayout *>(execution.region_address);
    if (layout->version != SCHEDULER_ASYNC_LAYOUT_VERSION || layout->run_epoch == 0 ||
        layout->total_bytes < sizeof(*layout) || layout->total_bytes > UINT64_MAX - execution.region_address ||
        layout->execution_contract > HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY ||
        execution.task_ref >= layout->task_count || layout->backend_offset > layout->total_bytes ||
        layout->backend_bytes > layout->total_bytes - layout->backend_offset ||
        !declarations->mount(
            layout->backend_bytes == 0 ?
                nullptr :
                reinterpret_cast<__gm__ const uint8_t *>(execution.region_address + layout->backend_offset),
            layout->backend_bytes
        ) ||
        !async_sdma_guard_region(
            execution.region_address, layout->total_bytes, layout->run_epoch, layout->sdma_guard_context_offset,
            layout->sdma_guard_count, guard_address
        ))
        return false;
    *contract = layout->execution_contract;
    return true;
}
