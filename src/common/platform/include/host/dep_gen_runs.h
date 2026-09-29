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
 * @file dep_gen_runs.h
 * @brief DepGen's cross-run types: the per-run export, the charged record
 *        store, the transport report the boundary produces, and the two-state
 *        publication classifier.
 *
 * DepGen publishes one whole graph or no file: `deps.json` carries no metadata
 * line and no completeness field, so there is nowhere to mark a partial graph.
 * Everything here therefore reduces to one question — is this run's graph
 * trustworthy end to end — and the answer is a `Verdict` that either publishes
 * or does not.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "common/dep_gen.h"
#include "host/chip_swimlane_runs.h"

namespace simpler::dfx::dep_gen_runs {

/**
 * Bytes per record block.
 *
 * `sizeof(DepGenRecord)` is 4736 B, so a 64 KiB block holds 13 records and
 * wastes 3948 B (6.0%). Larger blocks waste less and allocate less often; this
 * is the same 64 KiB the other retained collectors use, kept for uniformity
 * rather than tuned.
 */
inline constexpr size_t kRecordBlockBytes = 64 * 1024;
inline constexpr size_t kRecordsPerBlock = kRecordBlockBytes / sizeof(DepGenRecord);
static_assert(kRecordsPerBlock > 0, "a record block must hold at least one record");

/**
 * Checked size arithmetic.
 *
 * Every product and sum that reaches an allocation size or an arena reserve
 * goes through these: the inputs are device-written counts, so a corrupted
 * record must produce a refusal rather than a wrapped length.
 */
inline bool checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

inline bool checked_add(size_t a, size_t b, size_t *out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

/**
 * The largest `local_id` a task window may be sized from.
 *
 * `ceil_pow2` in the replay is `int32_t`: for an input above 2^30 the bit
 * smear produces `0x80000000`, which is negative as `int32_t` and becomes
 * ~1.8e19 when cast to `size_t` for an arena reserve. Records are device
 * written, so the domain is checked rather than assumed.
 */
inline constexpr int32_t kMaxTaskLocalId = (1 << 30) - 1;

/**
 * Everything the boundary established about one run's transport, separately
 * from whether the counts happened to balance.
 *
 * `reconcile_counters()` historically answered one `bool` that conflated "the
 * comparison balanced" with "the comparison was made". Each field below is a
 * distinct reason a graph is not trustworthy, and the default path reports the
 * same set through its own channel.
 */
struct ReconcileReport {
    bool clean{false};               // every check below passed
    bool region_read_failed{false};  // the shared region D2H reported failure
    bool buffer_read_failed{false};  // the in-flight buffer's D2H reported failure
    bool buffer_unresolved{false};   // current_buf_ptr had no host mapping, so nothing was checked
    bool unflushed_records{false};   // a buffer the device still held carried records
    bool counters_unknown{false};    // a counter is at the saturation sentinel
    bool identity_broken{false};     // collected + dropped != total + overflow
    bool device_dropped{false};      // the device dropped at least one submit
    bool host_clamped{false};        // a buffer's count exceeded the slot capacity and was clamped
    bool identity_foreign{false};    // a record arrived stamped with a non-admitted epoch
    bool reset_unpublished{false};   // this run's counter reset did not reach the device
    uint64_t total_device{0};
    uint64_t dropped_device{0};
    uint64_t overflow_device{0};
    uint64_t collected_host{0};
};

/** True iff this run's graph may be published. */
inline bool publishable(const ReconcileReport &r) { return r.clean; }

/** The first reason a report is not clean, for the error record. */
inline const char *first_refusal(const ReconcileReport &r) {
    if (r.reset_unpublished) return "this run's device counter reset was not published";
    if (r.region_read_failed) return "the shared region read failed";
    if (r.buffer_read_failed) return "the in-flight buffer read failed";
    if (r.buffer_unresolved) return "the in-flight buffer had no host mapping";
    if (r.unflushed_records) return "the device still held records it never handed over";
    if (r.counters_unknown) return "a device counter reached its counting limit";
    if (r.host_clamped) return "a buffer carried more records than its slot holds";
    if (r.identity_foreign) return "a record arrived stamped with a non-admitted run";
    if (r.device_dropped) return "the device dropped submits";
    if (r.identity_broken) return "the record count identity did not balance";
    return "the graph could not be established";
}

/**
 * Host-owned records in fixed blocks, charged as they are allocated.
 *
 * `append` reports a refusal rather than throwing or dropping silently: it runs
 * on a collector thread whose entry point has no exception boundary of its own,
 * so an allocation failure has to become a refusal. A charge whose allocation
 * then failed bought storage that does not exist, so it is given back.
 *
 * Blocks are never reallocated, so the charge that precedes an allocation *is*
 * the allocation — there is no grow-and-move transient for the budget to cover.
 */
class RecordBlocks {
public:
    RecordBlocks() = default;
    RecordBlocks(const RecordBlocks &) = delete;
    RecordBlocks &operator=(const RecordBlocks &) = delete;
    RecordBlocks(RecordBlocks &&) noexcept = default;
    RecordBlocks &operator=(RecordBlocks &&) noexcept = default;

    template <typename Charge, typename Credit>
    bool append(const DepGenRecord &record, Charge &&charge, Credit &&credit) noexcept {
        if (size_ == blocks_.size() * kRecordsPerBlock) {
            if (!charge(kRecordBlockBytes)) return false;
            try {
                blocks_.push_back(std::make_unique<DepGenRecord[]>(kRecordsPerBlock));
            } catch (...) {
                credit(kRecordBlockBytes);
                return false;
            }
            charged_ += kRecordBlockBytes;
        }
        blocks_[size_ / kRecordsPerBlock][size_ % kRecordsPerBlock] = record;
        size_++;
        return true;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const DepGenRecord &operator[](size_t i) const { return blocks_[i / kRecordsPerBlock][i % kRecordsPerBlock]; }

    /**
     * Drop every block and report the bytes that were charged for them.
     *
     * A move-out, not a shrink: the caller credits exactly the returned figure,
     * and a second call returns 0 — so a credit cannot happen twice however
     * many exits a path has. Swaps rather than clearing, so it cannot throw on
     * a teardown path.
     */
    size_t release() noexcept {
        std::vector<std::unique_ptr<DepGenRecord[]>>{}.swap(blocks_);
        const size_t charged = charged_;
        charged_ = 0;
        size_ = 0;
        return charged;
    }

    size_t charged_bytes() const { return charged_; }

private:
    std::vector<std::unique_ptr<DepGenRecord[]>> blocks_;
    size_t size_{0};
    size_t charged_{0};
};

/**
 * One run's graph, owned independently of the collector that produced it.
 *
 * The writer reads nothing mutable that a later run touches: the records were
 * moved out of the collector's map, and the identity and the destination are
 * the ones the run was admitted with rather than whatever the runner's config
 * says by the time the writer gets here.
 */
struct RunExport {
    uint64_t run_epoch{0};
    std::string output_dir;
    ReconcileReport report;
    RecordBlocks records;
};

}  // namespace simpler::dfx::dep_gen_runs
