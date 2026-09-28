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

/**
 * @file scope_stats_runs.h
 * @brief What one retained ScopeStats run owns, and how its host storage is
 *        bounded.
 *
 * The policy constants, the verdict vocabulary, the sticky error record and the
 * byte accountant are the shared ones in `simpler::dfx::runs`
 * (host/chip_swimlane_runs.h). Only the pieces whose shape is ScopeStats' own
 * live here: the device-sourced values an artifact needs, the record store that
 * is charged by allocated capacity rather than by size, and the export a writer
 * thread owns outright.
 *
 * Nothing here is constructed when retention is off, and with it off every
 * ScopeStats path behaves exactly as it does today.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/scope_stats.h"
#include "host/chip_swimlane_runs.h"
#include "host/collected_record.h"

namespace simpler::dfx::scope_stats_runs {

/**
 * One record block's byte size.
 *
 * Storage grows by whole blocks and a block is never reallocated, so the
 * charge that precedes an allocation is the allocation — there is no
 * grow-and-move transient for the budget to have to cover.
 */
inline constexpr size_t kRecordBlockBytes = 64 * 1024;

using Record = CollectedRecord<ScopeStatsRecord>;

inline constexpr size_t kRecordsPerBlock = kRecordBlockBytes / sizeof(Record);
static_assert(kRecordsPerBlock > 0, "a record block must hold at least one record");

/**
 * The device-written values one artifact needs, copied out while the run still
 * holds its execution claim.
 *
 * Copied rather than referenced because the next run rewrites every one of
 * them: the orchestrator's capacity setters republish the caps at its own init,
 * and `begin_run()` zeroes both counters. `valid` is false until a *checked*
 * device read has filled the rest, so a failed copy can never leave a stale
 * field to be published as this run's.
 */
struct DeviceSnapshot {
    bool valid{false};
    uint32_t fatal_latched{0};
    uint32_t dropped_records{0};
    uint32_t total_records{0};
    int32_t task_window_cap[SCOPE_STATS_MAX_RING_DEPTH]{};
    int32_t dep_pool_cap[SCOPE_STATS_MAX_RING_DEPTH]{};
    uint64_t heap_cap[SCOPE_STATS_MAX_RING_DEPTH]{};
    int32_t tensormap_cap{0};

    /** True while a counter is at the saturation sentinel its producer clamps to. */
    bool counters_saturated() const { return total_records == UINT32_MAX || dropped_records == UINT32_MAX; }
};

/**
 * Host-owned records in fixed blocks, charged as they are allocated.
 *
 * `append` reports a refusal rather than throwing or dropping silently: the
 * caller keeps counting what it received, so the difference between received
 * and retained is the loss the artifact reports.
 */
class RecordBlocks {
public:
    RecordBlocks() = default;
    RecordBlocks(const RecordBlocks &) = delete;
    RecordBlocks &operator=(const RecordBlocks &) = delete;
    RecordBlocks(RecordBlocks &&) noexcept = default;
    RecordBlocks &operator=(RecordBlocks &&) noexcept = default;

    /**
     * Append one record, charging a new block first when this one starts it.
     *
     * `noexcept` is the contract, not a hope: this runs on a collector thread
     * whose entry point has no exception boundary of its own, so an allocation
     * failure here has to become a refusal rather than terminate the process.
     * A charge whose allocation then failed bought storage that does not
     * exist, so it is given back — the budget never holds bytes nothing
     * occupies.
     *
     * @param charge called only when a block is about to be allocated; a false
     *               return refuses the record and allocates nothing
     * @param credit called with the same figure when the allocation fails
     */
    template <typename Charge, typename Credit>
    bool append(const Record &record, Charge &&charge, Credit &&credit) noexcept {
        if (size_ == blocks_.size() * kRecordsPerBlock) {
            if (!charge(kRecordBlockBytes)) return false;
            try {
                blocks_.push_back(std::make_unique<Record[]>(kRecordsPerBlock));
            } catch (...) {
                credit(kRecordBlockBytes);
                return false;
            }
        }
        blocks_[size_ / kRecordsPerBlock][size_ % kRecordsPerBlock] = record;
        size_++;
        return true;
    }

    size_t size() const { return size_; }
    const Record &operator[](size_t i) const { return blocks_[i / kRecordsPerBlock][i % kRecordsPerBlock]; }

    /** Bytes this store has charged, which is its whole allocation. */
    size_t charged_bytes() const { return blocks_.size() * kRecordBlockBytes; }

    /**
     * Drop every block and report what the caller must credit back.
     *
     * Swaps an empty vector in rather than clearing and shrinking, so the
     * release allocates nothing and cannot throw on a teardown path.
     */
    size_t release() noexcept {
        const size_t charged = charged_bytes();
        std::vector<std::unique_ptr<Record[]>>().swap(blocks_);
        size_ = 0;
        return charged;
    }

private:
    std::vector<std::unique_ptr<Record[]>> blocks_;
    size_t size_{0};
};

/** How a run's collection ended, before any statement about its file. */
struct Collection {
    simpler::dfx::runs::Verdict verdict{simpler::dfx::runs::Verdict::Published};
    bool counts_unknown{false};
    uint64_t received{0};
    uint64_t retained{0};
};

/**
 * One run's whole artifact, owned by the writer thread once it is handed over.
 *
 * It names no collector state, so the writer reads nothing a successor can
 * mutate and nothing the device can still write.
 */
struct RunExport {
    uint64_t run_epoch{0};
    /** The export slot this run holds until its artifact is published. */
    size_t slot{0};
    std::string output_dir;
    DeviceSnapshot device{};
    Collection collection{};
    RecordBlocks records{};
};

/**
 * Classify a run whose device completion and terminal read were both proved.
 *
 * Order matters and is the contract: counters that cannot be trusted outrank
 * every statement built on them, so saturation and a failed identity settle
 * first. A latched device fatal does **not** make the counts unknown — the
 * accounting still balances, and the fatal travels in the artifact's own
 * `fatal` field — so it settles a partial result that is still count-known.
 */
inline Collection classify(const DeviceSnapshot &device, uint64_t received, uint64_t retained) {
    Collection out;
    out.received = received;
    out.retained = retained;
    if (device.counters_saturated() || received + device.dropped_records != device.total_records) {
        out.verdict = simpler::dfx::runs::Verdict::CutUnknown;
        out.counts_unknown = true;
        return out;
    }
    if (device.fatal_latched != 0 || device.dropped_records != 0 || retained < received) {
        out.verdict = simpler::dfx::runs::Verdict::PartialSafe;
        return out;
    }
    out.verdict = simpler::dfx::runs::Verdict::Published;
    return out;
}

}  // namespace simpler::dfx::scope_stats_runs
