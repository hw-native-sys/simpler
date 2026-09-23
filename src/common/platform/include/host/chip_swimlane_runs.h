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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#include "common/unified_log.h"

/**
 * Cross-run collection support types: the memory accountant, the
 * per-run verdict carried into the artifact, and the collector's permanent
 * error summary.
 *
 * These are host-only and are reached exclusively through the swimlane
 * collector's retained-run path. With retention off none of them is
 * constructed against a budget and every collector behaves as it does today.
 */
namespace simpler::dfx::runs {

// Policy constants. Each is a minimum viable default, not a tuned value; every
// one of them is a bound the contract promises rather than a performance knob.
inline constexpr size_t kMaxOpenEpochs = 2;   // unpublished epochs, sealing and in-hand write included
inline constexpr size_t kMaxTombstones = 16;  // late-buffer classification ring only, never the error record
inline constexpr size_t kDefaultBudgetBytes = 256ull * 1024 * 1024;
inline constexpr size_t kMinWorkingSetBytes = 16ull * 1024 * 1024;
inline constexpr size_t kWriterScratchBytes = 1ull * 1024 * 1024;
// Reserved for each path string a run and its writer retain. A prefix longer
// than this is refused at admission rather than charged, so the allowance is a
// bound on the reservation and not a guess about the caller.
inline constexpr size_t kPathAllowanceBytes = 4096;
inline constexpr int kDrainQuantum = 32;                // entries per queue per sweep, then rotate
inline constexpr uint64_t kCounterMargin = 1ull << 20;  // monotonic counters fail this far from the end
inline constexpr int kCutAckBudgetMs = 2000;
inline constexpr int kControlAckBudgetMs = 2000;
inline constexpr size_t kErrorMsgBytes = 256;

/**
 * Why a run's artifact says what it says.
 *
 * `Published` and `PartialSafe` both produce a file; `CutUnknown` produces one
 * too, with `processing_complete = false`, because the reference handshake
 * proved the records were safe to move even though the transport count did
 * not settle. The remaining three produce no file and must wake every waiter.
 */
enum class Verdict {
    Published,    // stages settled, nothing lost on the host side
    PartialSafe,  // stages settled, content incomplete (retired / evicted / untrusted)
    CutUnknown,   // count unreachable, references proved released
    WriteFailed,  // sealed, but no file exists
    Quarantined,  // references not proved; nothing may be moved or freed
    CounterExhausted,
};

inline const char *verdict_name(Verdict v) {
    switch (v) {
    case Verdict::Published:
        return "published";
    case Verdict::PartialSafe:
        return "partial_safe";
    case Verdict::CutUnknown:
        return "partial_cut_unknown";
    case Verdict::WriteFailed:
        return "write_failed";
    case Verdict::Quarantined:
        return "quarantined";
    case Verdict::CounterExhausted:
        return "counter_exhausted";
    }
    return "unknown";
}

/** True when this verdict leaves a readable artifact behind. */
inline bool verdict_publishes(Verdict v) {
    return v == Verdict::Published || v == Verdict::PartialSafe || v == Verdict::CutUnknown;
}

/** The `metadata.collection` object a published run carries. */
struct CollectionVerdict {
    bool present{false};
    uint64_t run_epoch{0};
    // Which collector wrote this file, for a reader that finds two artifact
    // directories under one output root. The field keeps the name it is
    // published under; the collector member behind it is
    // `artifact_dir_index_`.
    uint64_t session_id{0};
    bool processing_complete{false};
    // False when the budget could not admit this run's caller-sized metadata —
    // JSON extensions, host phase records — and the artifact was written
    // without it. The omission is reported rather than left to be inferred
    // from a missing section.
    bool metadata_complete{true};
    uint64_t not_received_buffers{0};
    uint64_t unpublished_loss{0};
    uint64_t transport_retired{0};
    uint64_t cut_failed_queues{0};
    Verdict verdict{Verdict::Published};
};

/**
 * The collector's permanent progress and error record.
 *
 * Deliberately not the tombstone ring: that ring exists to classify late
 * buffers and is 16 entries deep, so a flush spanning more epochs than that
 * must not depend on it to remember a failure. Every counter here is
 * cumulative for the collector's whole life and the first error is retained
 * verbatim, so neither a later flush nor `close()` can forget an earlier
 * failed publication.
 *
 * A collector-level fatal is recorded here too, and is an error in its own
 * right: a background writer that died before sealing anything leaves no
 * epoch-scoped verdict behind, so an emptiness test over the per-epoch rows
 * would report a clean flush over a collector that published nothing.
 */
class ErrorSummary {
public:
    void record(uint64_t epoch, Verdict v, const char *detail) {
        std::lock_guard<std::mutex> lk(mu_);
        switch (v) {
        case Verdict::Published:
            published_++;
            break;
        case Verdict::PartialSafe:
            partial_++;
            break;
        case Verdict::CutUnknown:
            unknown_++;
            break;
        case Verdict::WriteFailed:
            write_failed_++;
            break;
        case Verdict::Quarantined:
            quarantined_++;
            break;
        case Verdict::CounterExhausted:
            counter_exhausted_++;
            break;
        }
        if (verdict_publishes(v)) {
            if (epoch > highest_published_) highest_published_ = epoch;
            return;
        }
        if (!error_recorded_) {
            error_recorded_ = true;
            first_error_epoch_ = epoch;
            first_error_verdict_ = v;
            if (detail != nullptr) {
                std::snprintf(first_error_msg_, sizeof(first_error_msg_), "%s", detail);
            }
        }
    }

    /**
     * A failure that belongs to the collector rather than to one run.
     *
     * Epoch zero is the "no snapshot" value everywhere else in this subsystem,
     * so a fatal cannot be represented as `record(0, …)`: the flag below is
     * what makes it visible, not the epoch.
     */
    void record_fatal(const char *detail) {
        std::lock_guard<std::mutex> lk(mu_);
        fatal_++;
        if (fatal_msg_[0] == '\0' && detail != nullptr) {
            std::snprintf(fatal_msg_, sizeof(fatal_msg_), "%s", detail);
        }
        error_recorded_ = true;
    }

    bool has_error() const {
        std::lock_guard<std::mutex> lk(mu_);
        return error_recorded_;
    }

    /** One bounded line for a flush or close failure. Never empty on error. */
    std::string report() const {
        std::lock_guard<std::mutex> lk(mu_);
        char buf[kErrorMsgBytes * 3];
        int n = std::snprintf(
            buf, sizeof(buf),
            "published=%llu partial=%llu cut_unknown=%llu write_failed=%llu quarantined=%llu counter_exhausted=%llu",
            static_cast<unsigned long long>(published_), static_cast<unsigned long long>(partial_),
            static_cast<unsigned long long>(unknown_), static_cast<unsigned long long>(write_failed_),
            static_cast<unsigned long long>(quarantined_), static_cast<unsigned long long>(counter_exhausted_)
        );
        if (n < 0) return {};
        size_t used = static_cast<size_t>(n) < sizeof(buf) ? static_cast<size_t>(n) : sizeof(buf) - 1;
        if (fatal_ != 0) {
            used += static_cast<size_t>(std::snprintf(
                buf + used, sizeof(buf) - used, "; collector fatal (%llu): %s", static_cast<unsigned long long>(fatal_),
                fatal_msg_
            ));
            if (used >= sizeof(buf)) return {buf};
        }
        if (first_error_epoch_ != 0 || first_error_verdict_ != Verdict::Published) {
            std::snprintf(
                buf + used, sizeof(buf) - used, "; first failed epoch %llu (%s): %s",
                static_cast<unsigned long long>(first_error_epoch_), verdict_name(first_error_verdict_),
                first_error_msg_
            );
        }
        return {buf};
    }

    uint64_t published_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return published_ + partial_ + unknown_;
    }
    uint64_t highest_published() const {
        std::lock_guard<std::mutex> lk(mu_);
        return highest_published_;
    }

    /** Every per-verdict count, read as one consistent set under the lock. */
    struct Counts {
        uint64_t published{0};
        uint64_t partial{0};
        uint64_t cut_unknown{0};
        uint64_t write_failed{0};
        uint64_t quarantined{0};
        uint64_t counter_exhausted{0};
        uint64_t fatal{0};
    };
    Counts counts() const {
        std::lock_guard<std::mutex> lk(mu_);
        return {published_, partial_, unknown_, write_failed_, quarantined_, counter_exhausted_, fatal_};
    }

private:
    mutable std::mutex mu_;
    uint64_t published_{0};
    uint64_t partial_{0};
    uint64_t unknown_{0};
    uint64_t write_failed_{0};
    uint64_t quarantined_{0};
    uint64_t counter_exhausted_{0};
    uint64_t fatal_{0};
    uint64_t highest_published_{0};
    bool error_recorded_{false};
    uint64_t first_error_epoch_{0};
    Verdict first_error_verdict_{Verdict::Published};
    char first_error_msg_[kErrorMsgBytes]{};
    char fatal_msg_[kErrorMsgBytes]{};
};

/**
 * The retained runs' host-memory accountant.
 *
 * Every charge is taken **before** the allocation it pays for, so a refusal
 * never leaves an allocation unaccounted, and a failed allocation releases its
 * charge in the same scope. The fixed part is reserved once at open, which is
 * why `open()` refuses a budget that cannot also hold a minimum working set:
 * a collector that could not grow a single record vector would report emptiness
 * rather than pressure.
 *
 * Device buffers are not here — they are capped in paired units by the buffer
 * pool manager, because a device buffer and its non-SVM host shadow are
 * allocated and freed together.
 */
class HostBudget {
public:
    bool open(size_t budget_bytes, size_t fixed_overhead) {
        if (budget_bytes == 0) {
            LOG_ERROR("ChipSwimlane: budget must be a positive byte count");
            return false;
        }
        if (budget_bytes < fixed_overhead + kMinWorkingSetBytes) {
            LOG_ERROR(
                "ChipSwimlane: budget %zu B cannot hold the fixed overhead %zu B plus a %zu B working set",
                budget_bytes, fixed_overhead, kMinWorkingSetBytes
            );
            return false;
        }
        limit_.store(budget_bytes, std::memory_order_relaxed);
        charged_.store(fixed_overhead, std::memory_order_relaxed);
        fixed_.store(fixed_overhead, std::memory_order_relaxed);
        // Collector-scoped, like every other figure it publishes.
        refusals_.store(0, std::memory_order_relaxed);
        return true;
    }

    void close() {
        limit_.store(0, std::memory_order_relaxed);
        charged_.store(0, std::memory_order_relaxed);
        fixed_.store(0, std::memory_order_relaxed);
    }

    /** Charge before allocating. Returns false when the budget cannot pay. */
    bool charge(size_t bytes) {
        const size_t limit = limit_.load(std::memory_order_relaxed);
        if (limit == 0) {
            refusals_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        size_t current = charged_.load(std::memory_order_relaxed);
        while (true) {
            if (bytes > limit - current) {  // no overflow: bytes and current are both <= limit
                refusals_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (charged_.compare_exchange_weak(current, current + bytes, std::memory_order_acq_rel)) return true;
        }
    }

    void credit(size_t bytes) {
        size_t current = charged_.load(std::memory_order_relaxed);
        while (true) {
            const size_t next = bytes > current ? 0 : current - bytes;
            if (charged_.compare_exchange_weak(current, next, std::memory_order_acq_rel)) return;
        }
    }

    size_t charged() const { return charged_.load(std::memory_order_relaxed); }
    size_t limit() const { return limit_.load(std::memory_order_relaxed); }
    size_t fixed() const { return fixed_.load(std::memory_order_relaxed); }
    uint64_t refusals() const { return refusals_.load(std::memory_order_relaxed); }

private:
    std::atomic<size_t> limit_{0};
    std::atomic<size_t> charged_{0};
    std::atomic<size_t> fixed_{0};
    std::atomic<uint64_t> refusals_{0};
};

/** True while a counter at `value` can still absorb `by` and stay trustworthy. */
inline bool counter_headroom(uint64_t value, uint64_t by = 1) {
    return by <= kCounterMargin && value <= UINT64_MAX - kCounterMargin - by;
}

/** Checked monotonic increment: false means the counter may no longer be trusted. */
inline bool checked_increment(uint64_t &counter, uint64_t by = 1) {
    if (!counter_headroom(counter, by)) return false;
    counter += by;
    return true;
}

/**
 * Checked monotonic increment of a counter a single writer owns but other
 * threads read.
 *
 * The check is on the value the increment produced rather than on the one
 * before it, because the release store must not be split into a load and a
 * store that a reader could observe between. `kCounterMargin` is what makes
 * that sound: the counter still has a million increments of headroom when this
 * first reports false, so acting on the report is not a race against the wrap.
 */
inline bool checked_increment(std::atomic<uint64_t> &counter, uint64_t by = 1) {
    const uint64_t produced = counter.fetch_add(by, std::memory_order_release) + by;
    return counter_headroom(produced, 0);
}

/**
 * Checked `count * unit` in bytes. False means the product is not
 * representable, which is a refusal rather than a wrapped small charge.
 */
inline bool checked_bytes(size_t count, size_t unit, size_t *out) {
    if (out == nullptr) return false;
    if (unit != 0 && count > SIZE_MAX / unit) return false;
    *out = count * unit;
    return true;
}

}  // namespace simpler::dfx::runs
