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

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "common/unified_log.h"
#include "host/chip_swimlane_runs.h"

/**
 * Cross-run support types for the PMU collector: the per-run verdict, the
 * collector's sticky error record, the epoch table, and the shard-file merge.
 *
 * Both single-arch PMU collectors instantiate these, so the epoch state machine
 * and the failure classification exist once. Everything here is host-only and
 * is reached exclusively through a PMU collector that retains runs; with
 * retention off no epoch is ever claimed and every PMU path behaves as it does
 * today.
 *
 * The policy constants are shared with the swimlane collector
 * (`simpler::dfx::runs`) rather than re-declared: two open epochs, a 4096 byte
 * path allowance, a 2000 ms acknowledgement budget and a 256 byte error detail
 * are bounds this contract promises at the same values.
 */
namespace simpler::dfx::pmu {

using simpler::dfx::runs::kControlAckBudgetMs;
using simpler::dfx::runs::kCutAckBudgetMs;
using simpler::dfx::runs::kErrorMsgBytes;
using simpler::dfx::runs::kMaxOpenEpochs;
using simpler::dfx::runs::kPathAllowanceBytes;

/**
 * Why a retained PMU run's flush says what it says.
 *
 * Only the first two are successes. Every other verdict must fail the public
 * flush even when a readable CSV was published, because a PMU CSV has no field
 * in which to record that it is partial: the swimlane artifact carries
 * `metadata.collection`, and the flush conclusion is all PMU has. So this
 * enumeration deliberately does *not* reuse `simpler::dfx::runs::Verdict`,
 * whose `verdict_publishes` treats a published partial as a non-error.
 */
enum class Verdict {
    Published,         // cut proved, records balanced, file published
    PublishedEmpty,    // cut proved, provably zero records, no file — as today
    PublishedShort,    // file published, content provably incomplete
    CountsUnknown,     // rows may be published, record accounting unreadable
    CutUnproved,       // cut unknown/failed, processing unsettled, or run unfinished
    CounterExhausted,  // a transport counter has no headroom; no cut is trustworthy
    WriteFailed,       // shard write, merge or rename failed; temps preserved
    Quarantined,       // references or the cut slot not proved released
    Abandoned,         // the collector was torn down before this run was published
};

inline const char *verdict_name(Verdict v) {
    switch (v) {
    case Verdict::Published:
        return "published";
    case Verdict::PublishedEmpty:
        return "published_empty";
    case Verdict::PublishedShort:
        return "published_short";
    case Verdict::CountsUnknown:
        return "counts_unknown";
    case Verdict::CutUnproved:
        return "cut_unproved";
    case Verdict::CounterExhausted:
        return "counter_exhausted";
    case Verdict::WriteFailed:
        return "write_failed";
    case Verdict::Quarantined:
        return "quarantined";
    case Verdict::Abandoned:
        return "abandoned";
    }
    return "unknown";
}

/** True only for the two verdicts a flush may report as success. */
inline bool verdict_succeeds(Verdict v) { return v == Verdict::Published || v == Verdict::PublishedEmpty; }

/**
 * The cut proof, evaluated by the writer when an epoch reaches a terminal.
 *
 * Stage 2 and the capture acknowledgement can only settle after the close, so
 * these are not close-time values; the close captures the slot and request the
 * questions are asked against.
 */
struct CutProofs {
    bool counters_exhausted{false};
    // False means *unknown*, never zero: `cut_failed_queues` returns false
    // until every drain owner has acknowledged the capture request.
    bool cut_known{false};
    int failed_queues{0};
    bool stage2_done{false};
};

/**
 * The record proof, captured at the close under this run's execution claim.
 *
 * `device_execution_complete` is the host-side flag the run boundary already
 * carries. PMU has no device-published terminal report, so it is the whole of
 * the "this run finished" proof.
 */
struct RecordProofs {
    bool snapshot_readable{false};
    // A core left a non-zero `current_buf_ptr` that could not be mapped, so
    // whether it held records is unknowable.
    bool live_buffer_unreadable{false};
    bool device_execution_complete{false};
    uint64_t total_device{0};
    uint64_t dropped_device{0};
    uint64_t mismatch_device{0};
    // Records a core still held after the cut settled. Any is a loss.
    uint64_t unflushed_records{0};
};

/**
 * Settle one epoch's verdict.
 *
 * The order is the precedence the contract states: least provable first, and
 * the file-level failure ahead of the content ones because it leaves no file at
 * all. `Quarantined` is not produced here — it is a control outcome, set by the
 * path that could not prove a release, and it never reaches a merge.
 */
inline Verdict classify(const CutProofs &cut, const RecordProofs &rec, uint64_t collected, bool io_failed) {
    if (io_failed) return Verdict::WriteFailed;
    if (cut.counters_exhausted) return Verdict::CounterExhausted;
    if (!cut.cut_known || cut.failed_queues != 0 || !cut.stage2_done || !rec.device_execution_complete) {
        return Verdict::CutUnproved;
    }
    if (!rec.snapshot_readable || rec.live_buffer_unreadable) return Verdict::CountsUnknown;
    const uint64_t accounted = collected + rec.dropped_device + rec.mismatch_device;
    if (rec.dropped_device != 0 || rec.mismatch_device != 0 || rec.unflushed_records != 0 ||
        accounted != rec.total_device) {
        return Verdict::PublishedShort;
    }
    if (collected == 0) return Verdict::PublishedEmpty;
    return Verdict::Published;
}

/**
 * The collector's sticky, bounded error record.
 *
 * Every counter is cumulative for the collector's whole life and the first
 * error is retained verbatim, so neither a later flush, a collector rebuild,
 * nor turning PMU off can forget an earlier failure. There is deliberately no
 * reset or clear: no acknowledgement API exists, so a failure recorded here is
 * reported by every later flush until the device runner is destroyed.
 *
 * It also owns the two collector-scoped signals that cannot be attributed to
 * one run — the drain path's retirement count and late buffers naming an epoch
 * this collector does not know. Both are errors and neither is a fatal: they
 * fail the flush without refusing further admission.
 */
class ErrorSummary {
public:
    void record(uint64_t epoch, Verdict v, const char *detail) {
        std::lock_guard<std::mutex> lk(mu_);
        counts_[static_cast<size_t>(v)]++;
        if (verdict_succeeds(v)) {
            if (epoch > highest_published_) highest_published_ = epoch;
            return;
        }
        note_error_locked(epoch, v, detail);
    }

    /**
     * A failure of the collector rather than of one run.
     *
     * Epoch zero is the "no snapshot" value everywhere else in this subsystem,
     * so a fatal cannot be represented as `record(0, …)`: the counter below is
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

    /**
     * Observe the drain path's cumulative retirement count.
     *
     * The count belongs to the collector, not to an epoch: it is incremented by
     * a drain owner after it has processed a ready entry, which can be later
     * than the run's device completion and later than its close, and the
     * unreadable buffer is exactly the one whose stamped epoch may be
     * unrecoverable. So an advance is reported conservatively, in buffers, as a
     * collector-scoped error — never as a per-run count, and never against any
     * epoch's verdict counter.
     *
     * Compare and advance happen in one critical section, so several observers
     * report one advance once.
     */
    void observe_transport(uint64_t seen) {
        std::lock_guard<std::mutex> lk(mu_);
        if (seen <= transport_mark_) return;
        const uint64_t advance = seen - transport_mark_;
        transport_mark_ = seen;
        transport_retired_ += advance;
        char detail[kErrorMsgBytes];
        std::snprintf(
            detail, sizeof(detail), "the drain path retired %llu ready buffer(s) while runs were retained",
            static_cast<unsigned long long>(advance)
        );
        note_error_locked(0, Verdict::PublishedShort, detail);
    }

    /**
     * Re-baseline the observation mark to a counter that has been reset.
     *
     * The base counter is consumed by `report_drain_drops()`, which zeroes it.
     * A mark left above a reset counter would hide every later retirement, so
     * the two move together. This clears no error: an advance already observed
     * stays recorded, and one that was consumed without being observed is
     * exactly what the caller observes first.
     */
    void rebaseline_transport(uint64_t seen) {
        std::lock_guard<std::mutex> lk(mu_);
        transport_mark_ = seen;
    }

    /**
     * A buffer named an epoch this collector cannot place.
     *
     * Bounded by construction: the count grows, no per-run object is retained
     * to carry it, and a published epoch's verdict is never rewritten. Like the
     * transport count it is collector-scoped, so it touches no epoch's verdict
     * counter.
     */
    void record_unknown_epoch(uint64_t buffer_epoch, uint64_t records) {
        std::lock_guard<std::mutex> lk(mu_);
        unknown_epoch_buffers_++;
        unknown_epoch_records_ += records;
        char detail[kErrorMsgBytes];
        std::snprintf(
            detail, sizeof(detail), "%llu record(s) arrived for run %llu, which this collector cannot place",
            static_cast<unsigned long long>(records), static_cast<unsigned long long>(buffer_epoch)
        );
        note_error_locked(0, Verdict::CountsUnknown, detail);
    }

    bool has_error() const {
        std::lock_guard<std::mutex> lk(mu_);
        return error_recorded_;
    }

    uint64_t fatal_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return fatal_;
    }

    /** One bounded line for a flush or close failure. Never empty on error. */
    std::string report() const {
        std::lock_guard<std::mutex> lk(mu_);
        char buf[kErrorMsgBytes * 3];
        int n = std::snprintf(
            buf, sizeof(buf),
            "published=%llu empty=%llu short=%llu counts_unknown=%llu cut_unproved=%llu counter_exhausted=%llu "
            "write_failed=%llu quarantined=%llu abandoned=%llu transport_retired=%llu unknown_epoch_records=%llu",
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::Published)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::PublishedEmpty)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::PublishedShort)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::CountsUnknown)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::CutUnproved)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::CounterExhausted)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::WriteFailed)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::Quarantined)]),
            static_cast<unsigned long long>(counts_[static_cast<size_t>(Verdict::Abandoned)]),
            static_cast<unsigned long long>(transport_retired_), static_cast<unsigned long long>(unknown_epoch_records_)
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
        if (error_recorded_ && first_error_msg_[0] != '\0') {
            // Epoch zero is the "no snapshot" value everywhere in this
            // subsystem, so it names a collector-scoped fault rather than a run.
            if (first_error_epoch_ == 0) {
                std::snprintf(
                    buf + used, sizeof(buf) - used, "; first failure (collector, %s): %s",
                    verdict_name(first_error_verdict_), first_error_msg_
                );
            } else {
                std::snprintf(
                    buf + used, sizeof(buf) - used, "; first failure (run %llu, %s): %s",
                    static_cast<unsigned long long>(first_error_epoch_), verdict_name(first_error_verdict_),
                    first_error_msg_
                );
            }
        }
        return {buf};
    }

    /** Every count, read as one consistent set under the lock. */
    struct Counts {
        uint64_t published{0};
        uint64_t published_empty{0};
        uint64_t published_short{0};
        uint64_t counts_unknown{0};
        uint64_t cut_unproved{0};
        uint64_t counter_exhausted{0};
        uint64_t write_failed{0};
        uint64_t quarantined{0};
        uint64_t abandoned{0};
        uint64_t fatal{0};
        uint64_t transport_retired{0};
        uint64_t unknown_epoch_buffers{0};
        uint64_t unknown_epoch_records{0};
        bool has_error{false};
    };
    Counts counts() const {
        std::lock_guard<std::mutex> lk(mu_);
        Counts c;
        c.published = counts_[static_cast<size_t>(Verdict::Published)];
        c.published_empty = counts_[static_cast<size_t>(Verdict::PublishedEmpty)];
        c.published_short = counts_[static_cast<size_t>(Verdict::PublishedShort)];
        c.counts_unknown = counts_[static_cast<size_t>(Verdict::CountsUnknown)];
        c.cut_unproved = counts_[static_cast<size_t>(Verdict::CutUnproved)];
        c.counter_exhausted = counts_[static_cast<size_t>(Verdict::CounterExhausted)];
        c.write_failed = counts_[static_cast<size_t>(Verdict::WriteFailed)];
        c.quarantined = counts_[static_cast<size_t>(Verdict::Quarantined)];
        c.abandoned = counts_[static_cast<size_t>(Verdict::Abandoned)];
        c.fatal = fatal_;
        c.transport_retired = transport_retired_;
        c.unknown_epoch_buffers = unknown_epoch_buffers_;
        c.unknown_epoch_records = unknown_epoch_records_;
        c.has_error = error_recorded_;
        return c;
    }

private:
    void note_error_locked(uint64_t epoch, Verdict v, const char *detail) {
        if (!error_recorded_) {
            error_recorded_ = true;
            first_error_epoch_ = epoch;
            first_error_verdict_ = v;
            if (detail != nullptr) {
                std::snprintf(first_error_msg_, sizeof(first_error_msg_), "%s", detail);
            }
        }
    }

    static constexpr size_t kVerdictCount = static_cast<size_t>(Verdict::Abandoned) + 1;

    mutable std::mutex mu_;
    std::array<uint64_t, kVerdictCount> counts_{};
    uint64_t fatal_{0};
    uint64_t highest_published_{0};
    uint64_t transport_mark_{0};
    uint64_t transport_retired_{0};
    uint64_t unknown_epoch_buffers_{0};
    uint64_t unknown_epoch_records_{0};
    bool error_recorded_{false};
    uint64_t first_error_epoch_{0};
    Verdict first_error_verdict_{Verdict::Published};
    char first_error_msg_[kErrorMsgBytes]{};
    char fatal_msg_[kErrorMsgBytes]{};
};

// ---------------------------------------------------------------------------
// Per-epoch file names and publication
// ---------------------------------------------------------------------------

/**
 * This epoch's shard temp path. The epoch is in the name, so two open epochs
 * writing under one destination can never collide and nothing is deleted
 * implicitly.
 */
inline std::string shard_temp_path(const std::string &csv_path, uint64_t epoch, size_t shard) {
    return csv_path + ".e" + std::to_string(epoch) + ".shard" + std::to_string(shard) + ".tmp";
}

/** The merge's output, renamed onto `csv_path` as the single publication point. */
inline std::string publication_temp_path(const std::string &csv_path, uint64_t epoch) {
    return csv_path + ".e" + std::to_string(epoch) + ".part.tmp";
}

/**
 * True when a previous epoch's preserved failure artifacts are still present
 * for this destination.
 *
 * A failed epoch's temps are the evidence, so a later run under the same prefix
 * is refused rather than allowed to write beside them — the contract never
 * deletes them, and the operator decides when they have been read. A directory
 * that cannot be listed is reported as clear: refusing every run because a
 * scan failed would be worse than admitting one, and the admission's own file
 * opens are what then fail.
 */
inline bool failure_evidence_present(const std::string &csv_path) {
    std::filesystem::path path(csv_path);
    const std::string prefix = path.filename().string() + ".e";
    std::error_code ec;
    std::filesystem::directory_iterator it(path.parent_path(), ec);
    if (ec) return false;
    for (const auto &entry : it) {
        const std::string name = entry.path().filename().string();
        if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) return true;
    }
    return false;
}

/**
 * Merge this epoch's shard files into its publication temp and rename that onto
 * the final path.
 *
 * Rename is the single publication point: no reader sees a half-merged file,
 * and an epoch that fails leaves the previous successful `pmu.csv` exactly
 * where it was. On any failure the shard files and whatever was merged are
 * **preserved** as evidence and the final path is not touched — which is why
 * this never removes the destination, unlike the in-place single-run merge.
 *
 * `shard_rows` selects which shards have content; a shard with no rows is not
 * opened. Returns false on the first I/O failure.
 */
inline bool merge_shards_and_publish(
    const std::string &csv_path, uint64_t epoch, const std::string &csv_header,
    const std::vector<std::string> &shard_paths, const std::vector<uint64_t> &shard_rows
) {
    const std::string part = publication_temp_path(csv_path, epoch);
    std::ofstream out(part, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        LOG_ERROR("PmuCollector: run %llu could not open %s", static_cast<unsigned long long>(epoch), part.c_str());
        return false;
    }
    out << csv_header;
    for (size_t shard = 0; shard < shard_paths.size(); shard++) {
        if (shard >= shard_rows.size() || shard_rows[shard] == 0) continue;
        std::ifstream shard_file(shard_paths[shard], std::ios::binary);
        if (!shard_file.is_open()) {
            LOG_ERROR(
                "PmuCollector: run %llu could not read shard %s", static_cast<unsigned long long>(epoch),
                shard_paths[shard].c_str()
            );
            return false;
        }
        out << shard_file.rdbuf();
        if (shard_file.bad() || !out.good()) {
            LOG_ERROR(
                "PmuCollector: run %llu could not merge shard %s", static_cast<unsigned long long>(epoch),
                shard_paths[shard].c_str()
            );
            return false;
        }
    }
    out.flush();
    if (!out.good()) {
        LOG_ERROR("PmuCollector: run %llu could not flush %s", static_cast<unsigned long long>(epoch), part.c_str());
        return false;
    }
    // The close is part of the write, not bookkeeping after it: a filesystem
    // may only report a short write when the stream's own buffer is handed
    // over, and `close()` reports that through the failbit. Renaming without
    // checking it would publish a file the write never finished.
    out.close();
    if (out.fail()) {
        LOG_ERROR(
            "PmuCollector: run %llu could not close %s; it is not published", static_cast<unsigned long long>(epoch),
            part.c_str()
        );
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(part, csv_path, ec);
    if (ec) {
        LOG_ERROR(
            "PmuCollector: run %llu could not publish %s: %s", static_cast<unsigned long long>(epoch), csv_path.c_str(),
            ec.message().c_str()
        );
        return false;
    }
    return true;
}

/** Remove one epoch's shard temps. Only ever called after a publication. */
inline void remove_shard_temps(const std::vector<std::string> &shard_paths) {
    for (const auto &p : shard_paths) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
}

// ---------------------------------------------------------------------------
// The epoch table
// ---------------------------------------------------------------------------

/**
 * A retained epoch's lifecycle.
 *
 * `Admitting` is the only state a collector shard may route a buffer into.
 * `Closing` is the window in which the reference release is being proved, and
 * `Quarantined` is terminal: its files and its slot are held until the reader
 * join, so the slot never returns to `Free` by any other path.
 */
enum class EpochState : int { Free = 0, Admitting, Closing, Quarantined };

/**
 * One PMU epoch's frozen state.
 *
 * `Frozen` is the arch's own per-run configuration — the event type and the
 * resolved column set — captured at admission so a successor's configuration
 * can never reach a predecessor's rows. Everything else here is
 * arch-independent.
 */
template <typename Frozen>
struct Epoch {
    std::atomic<int> state{static_cast<int>(EpochState::Free)};
    std::atomic<uint64_t> epoch{0};

    // Frozen at admission, before any callback can receive this run's data.
    Frozen frozen{};
    std::string csv_path;
    std::string csv_header;
    std::vector<std::string> shard_paths;

    // Written by the collector shard that owns each index, read by the writer
    // after the references are proved released.
    std::vector<uint64_t> shard_rows;
    std::atomic<bool> io_failed{false};
    // A buffer arrived with a count past its capacity, so records beyond it are
    // unreachable. Set by a collector shard, read once by the writer.
    std::atomic<bool> clamped{false};

    // Captured at close, under this run's execution claim.
    RecordProofs records{};
    CutProofs cut{};
    int cut_slot{-1};
    uint64_t cut_request{0};
    bool target_installed{false};
    std::chrono::steady_clock::time_point closed_at{};

    void reset_run_state(size_t shard_count) {
        shard_rows.assign(shard_count, 0);
        io_failed.store(false, std::memory_order_relaxed);
        clamped.store(false, std::memory_order_relaxed);
        records = RecordProofs{};
        cut = CutProofs{};
        cut_slot = -1;
        cut_request = 0;
        target_installed = false;
    }
};

/**
 * Which epochs a collector shard may route into, refreshed on the
 * reference-release handshake.
 *
 * A shard reads its own copy with no lock: the handshake is what publishes a
 * change, and a shard acknowledges only after it has adopted the new view and
 * while it holds no epoch reference.
 */
struct ShardEpochView {
    struct Entry {
        uint64_t epoch{0};
        int slot{-1};
    };
    std::array<Entry, kMaxOpenEpochs> entries{};
    size_t count{0};

    int slot_for(uint64_t epoch) const {
        for (size_t i = 0; i < count; i++) {
            if (entries[i].epoch == epoch && entries[i].slot >= 0) return entries[i].slot;
        }
        return -1;
    }
};

/**
 * Counts a test can assert on without reaching into collector internals.
 */
struct RetainedRunStats {
    bool ready{false};
    bool fatal{false};
    size_t open_epochs{0};
    size_t quarantined_epochs{0};
    ErrorSummary::Counts errors{};
};

/**
 * The epoch table, the background writer and the failure accounting — once, for
 * both architectures.
 *
 * `Collector` is the arch's PMU collector, which must befriend this template
 * and supply these hooks:
 *
 *   size_t      retained_shard_count() const
 *   Frozen      freeze_run_config(EventType) const
 *   std::string build_csv_header(const Frozen &) const
 *   bool        publish_run_config(EventType)          // device header + counter reset;
 *                                                       // false = the device kept a previous value
 *   RecordProofs snapshot_run_records(bool complete) const
 *   uint64_t    buffer_run_epoch(const void *) const
 *   uint64_t    buffer_record_count(const void *) const
 *   uint64_t    write_buffer_rows(std::ofstream &, const Frozen &, int core, int thread,
 *                                 const void *buf, uint64_t epoch, bool *clamped)
 *
 * and, through that friendship, `ProfilerBase`'s cut and reference-release
 * primitives. No new mechanism is introduced here: the finality proofs are
 * exactly the base's, in the order the swimlane writer already evaluates them.
 */
template <typename Collector, typename Frozen, typename EventType, size_t MaxShards>
class RetainedRuns {
public:
    explicit RetainedRuns(Collector &owner) :
        owner_(owner) {}
    RetainedRuns(const RetainedRuns &) = delete;
    RetainedRuns &operator=(const RetainedRuns &) = delete;
    ~RetainedRuns() { stop_writer(); }

    void configure(bool enabled) { retain_across_runs_ = enabled; }
    bool retains_runs() const { return retain_across_runs_; }
    bool ready() const { return ready_.load(std::memory_order_acquire); }
    bool fatal() const { return fatal_.load(std::memory_order_acquire); }
    const ErrorSummary &errors() const { return errors_; }

    // -----------------------------------------------------------------------
    // Admission
    // -----------------------------------------------------------------------

    bool run_begin(uint64_t run_epoch, const std::string &csv_path, EventType event_type) {
        // A guard, not the configuration question's answer: a caller that
        // reaches here on a collector that retains nothing has already taken
        // the wrong path.
        if (!retain_across_runs_) return false;
        if (!ensure_ready()) {
            LOG_ERROR(
                "PmuCollector: run %llu refused, retention could not be prepared",
                static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
        // Refused before a slot is claimed: every path this epoch retains is
        // reserved against the allowance rather than charged, so one that
        // cannot fit is turned away here. The 64 bytes cover the epoch-bearing
        // suffixes derived from it.
        if (csv_path.size() + 64 > kPathAllowanceBytes) {
            LOG_ERROR(
                "PmuCollector: run %llu refused, its CSV path of %zu bytes exceeds the %zu byte allowance",
                static_cast<unsigned long long>(run_epoch), csv_path.size(), kPathAllowanceBytes
            );
            return false;
        }
        if (fatal_.load(std::memory_order_acquire)) {
            LOG_ERROR(
                "PmuCollector: run %llu refused, the collector is fatal", static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
        // A failed epoch's temps are its evidence and this contract never
        // deletes them, so a later run under the same destination is refused
        // rather than allowed to write beside them.
        if (failure_evidence_present(csv_path)) {
            LOG_ERROR(
                "PmuCollector: run %llu refused, %s still holds a failed run's temporary files; "
                "read them and remove them to reuse this destination",
                static_cast<unsigned long long>(run_epoch), csv_path.c_str()
            );
            return false;
        }

        size_t slot = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            bool found = false;
            for (size_t i = 0; i < epochs_.size(); i++) {
                if (epochs_[i].state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Free)) {
                    // A destination an open epoch owns would have two writers
                    // and one publication point.
                    if (epochs_[i].csv_path == csv_path) {
                        LOG_ERROR(
                            "PmuCollector: run %llu refused, %s is still owned by run %llu",
                            static_cast<unsigned long long>(run_epoch), csv_path.c_str(),
                            static_cast<unsigned long long>(epochs_[i].epoch.load(std::memory_order_acquire))
                        );
                        return false;
                    }
                    continue;
                }
                if (!found) {
                    slot = i;
                    found = true;
                }
            }
            if (!found) {
                // Refused, not queued: the device is about to be handed this
                // run, and a wait here would hold the launch behind a slow disk.
                LOG_ERROR(
                    "PmuCollector: run %llu refused, both retained PMU runs are still unpublished",
                    static_cast<unsigned long long>(run_epoch)
                );
                return false;
            }

            Epoch<Frozen> &epoch = epochs_[slot];
            epoch.epoch.store(run_epoch, std::memory_order_relaxed);
            epoch.frozen = owner_.freeze_run_config(event_type);
            epoch.csv_path = csv_path;
            epoch.csv_header = owner_.build_csv_header(epoch.frozen);
            const size_t shard_count = owner_.retained_shard_count();
            epoch.shard_paths.clear();
            epoch.shard_paths.reserve(shard_count);
            for (size_t shard = 0; shard < shard_count; shard++) {
                epoch.shard_paths.push_back(shard_temp_path(csv_path, run_epoch, shard));
            }
            epoch.reset_run_state(shard_count);
            epoch.state.store(static_cast<int>(EpochState::Admitting), std::memory_order_release);
            progress_++;
        }

        // The device header and the per-core counter reset. The predecessor's
        // close has already snapshotted those counters, which is what makes
        // resetting them here safe.
        //
        // A field the device did not take refuses the run **before it is
        // launched**, because this epoch's whole promise is that its frozen
        // host configuration and the device's agree: an event type the device
        // kept at the previous run's value would have it measure one event
        // group while these columns name another, and the record counts could
        // still balance, so nothing later in the pipeline could catch it.
        if (!owner_.publish_run_config(event_type)) {
            LOG_ERROR(
                "PmuCollector: run %llu refused, the device did not take this run's PMU configuration",
                static_cast<unsigned long long>(run_epoch)
            );
            withdraw_unpublished_slot(slot);
            return false;
        }
        // Every shard must see this epoch before the device can publish into
        // it, or its first buffers would belong to no epoch.
        if (!owner_.request_run_reference_release(kControlAckBudgetMs)) {
            set_fatal("a collector shard did not acknowledge the epoch table in time");
        }
        return !fatal_.load(std::memory_order_acquire);
    }

    /**
     * Snapshot this run's device state while it still holds the execution
     * claim, then arm its cut and hand the epoch to the writer.
     */
    void run_close(uint64_t run_epoch, bool device_execution_complete) {
        if (!ready_.load(std::memory_order_acquire)) return;
        const int found = find_slot(run_epoch);
        if (found < 0) return;
        Epoch<Frozen> &epoch = epochs_[static_cast<size_t>(found)];

        // The successor has not launched, so the per-core counters are still
        // this run's and the ready-queue tails the cut reads are stable.
        epoch.records = owner_.snapshot_run_records(device_execution_complete);

        // Every buffer this run will ever publish is already in a device ready
        // queue at this instant, so each queue's target is finite and a
        // successor's later traffic cannot discharge it.
        uint64_t request = 0;
        epoch.cut_slot = owner_.cut_arm(&request);
        epoch.cut_request = request;
        if (epoch.cut_slot < 0) {
            LOG_WARN("PmuCollector: no cut slot for run %llu", static_cast<unsigned long long>(run_epoch));
        } else if (!owner_.cut_wait_for_ack(request, kCutAckBudgetMs)) {
            LOG_WARN(
                "PmuCollector: run %llu cut capture did not complete in %d ms",
                static_cast<unsigned long long>(run_epoch), kCutAckBudgetMs
            );
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            epoch.target_installed = true;
            epoch.closed_at = std::chrono::steady_clock::now();
            const uint64_t watermark = close_watermark_.load(std::memory_order_acquire);
            if (watermark != UINT64_MAX && run_epoch > watermark) {
                close_watermark_.store(run_epoch, std::memory_order_release);
            }
            progress_++;
        }
        cv_.notify_all();
    }

    /** Give back the slot of a run that was admitted and never launched. */
    bool abandon_run(uint64_t run_epoch) {
        if (!ready_.load(std::memory_order_acquire)) return true;
        size_t slot = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const int found = find_slot(run_epoch);
            // No slot under this identity: nothing of this run's to withdraw,
            // and nothing of anyone else's that this may touch.
            if (found < 0) return true;
            slot = static_cast<size_t>(found);
            if (epochs_[slot].state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) {
                return true;
            }
            // A target makes the epoch the writer's; only an unclosed one is
            // still this thread's to withdraw.
            if (epochs_[slot].target_installed) return false;
        }
        if (fatal_.load(std::memory_order_acquire)) {
            // A fatal collector cannot prove anything released, and its storage
            // is already the reader-join teardown's to free.
            release_deferred_.store(true, std::memory_order_release);
            return false;
        }
        Epoch<Frozen> &epoch = epochs_[slot];
        epoch.state.store(static_cast<int>(EpochState::Closing), std::memory_order_release);
        bool released = false;
        try {
            released = owner_.request_run_reference_release(kControlAckBudgetMs);
        } catch (...) {
            // An acknowledgement that could not even be asked for is not a
            // proof of release, so it settles the way a timeout does.
            released = false;
        }
        if (!released) {
            epoch.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
            finish_epoch(slot, Verdict::Quarantined, "a collector shard still holds an unlaunched run's reference");
            return false;
        }
        // This run submitted nothing, so it promised no file and owes no
        // verdict: recording one would make a rolled-back launch read as a lost
        // artifact and fail every later flush.
        (void)close_shards(slot);
        remove_shard_temps(epoch.shard_paths);
        release_slot(slot);
        try {
            LOG_WARN(
                "PmuCollector: run %llu was admitted and never launched; its slot is released without a file",
                static_cast<unsigned long long>(run_epoch)
            );
        } catch (...) {}
        return true;
    }

    // -----------------------------------------------------------------------
    // Hot path
    // -----------------------------------------------------------------------

    /**
     * Route one collected buffer into its own epoch's shard file.
     *
     * Called on a collector shard thread. The epoch is resolved from the
     * buffer's own `run_epoch` through this shard's view, so a predecessor's
     * late buffer lands in that run's file with that run's columns.
     */
    void route_buffer(const void *buf_host_ptr, int core_id, int thread_idx, int shard_index) {
        if (buf_host_ptr == nullptr) return;
        // The base guarantees a shard index inside its own count, so anything
        // else is a contract break to drop rather than to fold onto shard 0 —
        // which would put two collector threads on one stream.
        if (shard_index < 0 || static_cast<size_t>(shard_index) >= views_.size()) {
            LOG_ERROR("PmuCollector: collected buffer carried shard index %d", shard_index);
            return;
        }
        const size_t shard = static_cast<size_t>(shard_index);
        const uint64_t buffer_epoch = owner_.buffer_run_epoch(buf_host_ptr);
        const int slot = views_[shard].slot_for(buffer_epoch);
        if (slot < 0) {
            // Sealed or never admitted. These records cannot be attributed to
            // an epoch whose verdict is already published, and rewriting one is
            // what a bounded collector-scoped error replaces.
            errors_.record_unknown_epoch(buffer_epoch, owner_.buffer_record_count(buf_host_ptr));
            return;
        }
        Epoch<Frozen> &epoch = epochs_[static_cast<size_t>(slot)];
        if (!ensure_shard_open(static_cast<size_t>(slot), shard)) return;
        bool clamped = false;
        const uint64_t rows = owner_.write_buffer_rows(
            shard_files_[static_cast<size_t>(slot)][shard], epoch.frozen, core_id, thread_idx, buf_host_ptr,
            buffer_epoch, &clamped
        );
        if (clamped) {
            // A count past the buffer's capacity is a producer-side fault and
            // the records past it are unreachable, so the clamp is a loss.
            epoch.clamped.store(true, std::memory_order_relaxed);
        }
        if (!shard_files_[static_cast<size_t>(slot)][shard].good()) {
            LOG_ERROR("PmuCollector: failed to write shard file %s", epoch.shard_paths[shard].c_str());
            epoch.io_failed.store(true, std::memory_order_relaxed);
            return;
        }
        epoch.shard_rows[shard] += rows;
    }

    /** ProfilerBase hook: adopt the epoch table, then acknowledge it. */
    void refresh_view(int collector_shard) {
        if (collector_shard < 0 || static_cast<size_t>(collector_shard) >= views_.size()) return;
        ShardEpochView view;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t slot = 0; slot < epochs_.size(); slot++) {
                if (epochs_[slot].state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) {
                    continue;
                }
                view.entries[view.count].epoch = epochs_[slot].epoch.load(std::memory_order_acquire);
                view.entries[view.count].slot = static_cast<int>(slot);
                view.count++;
            }
        }
        views_[static_cast<size_t>(collector_shard)] = view;
    }

    /**
     * ProfilerBase hook: the transport reached a transition the writer waits
     * on — a queue's stage 1, or a shard catching up to its watermark.
     *
     * Without it the writer would learn stage 2 only from its own bounded
     * expiry, so every retained run would pay that wait before its file
     * appeared. Cheap and a no-op while no run is retained.
     */
    void note_transport_progress() {
        if (!ready_.load(std::memory_order_acquire)) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            progress_++;
        }
        cv_.notify_all();
    }

    // -----------------------------------------------------------------------
    // Reporting and teardown
    // -----------------------------------------------------------------------

    /**
     * Wait for every run closed up to now, then report.
     *
     * Always both halves: the wait is what a caller reading this run's file
     * needs, and the sticky report is what a caller needs after a rebuild
     * turned retention off, where the wait has nothing to do.
     */
    bool flush(int timeout_ms, std::string *error) {
        if (ready_.load(std::memory_order_acquire)) {
            const uint64_t watermark = close_watermark_.load(std::memory_order_acquire);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            while (true) {
                bool pending = false;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    for (const auto &epoch : epochs_) {
                        const int state = epoch.state.load(std::memory_order_acquire);
                        if (state == static_cast<int>(EpochState::Free)) continue;
                        // Terminal, and reported below rather than waited for.
                        if (state == static_cast<int>(EpochState::Quarantined)) continue;
                        if (epoch.epoch.load(std::memory_order_acquire) <= watermark && epoch.target_installed) {
                            pending = true;
                        }
                    }
                    if (!pending) break;
                    if (fatal_.load(std::memory_order_acquire)) break;
                    if (cv_.wait_until(lk, deadline) == std::cv_status::timeout &&
                        std::chrono::steady_clock::now() >= deadline) {
                        observe_transport();
                        if (error != nullptr) {
                            *error = "PMU flush timed out with runs still unpublished; " + errors_.report();
                        }
                        return false;
                    }
                }
            }
            // After the waited-for epochs are published, so a retirement that
            // happened past the last close still reaches this call.
            observe_transport();
        }
        if (fatal_.load(std::memory_order_acquire)) {
            if (error != nullptr) {
                std::lock_guard<std::mutex> lk(mu_);
                *error = "PMU collector is fatal: " + fatal_reason_ + "; " + errors_.report();
            }
            return false;
        }
        // Sticky: reported by every flush for as long as the runner lives,
        // because there is no acknowledgement that could clear it.
        if (errors_.has_error()) {
            if (error != nullptr) *error = "PMU collector reported failures: " + errors_.report();
            return false;
        }
        return true;
    }

    /** Stop admitting and publish what is still retained. Reports nothing. */
    void finish() {
        if (!ready_.load(std::memory_order_acquire)) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            close_watermark_.store(UINT64_MAX, std::memory_order_release);
            progress_++;
        }
        cv_.notify_all();
        // Void by contract: this publishes, and the caller's own flush is what
        // reports. A failure recorded here stays in the sticky summary.
        std::string ignored;
        (void)flush(kCutAckBudgetMs * 8, &ignored);
    }

    /** Join the writer. Called before the collector's own threads are joined. */
    void stop_writer() {
        if (!writer_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            // A held writer must not be joined while it is holding: releasing
            // here is what keeps a test that unwound mid-case from hanging
            // teardown.
            writer_held_.store(false, std::memory_order_release);
            writer_running_.store(false, std::memory_order_release);
            progress_++;
        }
        cv_.notify_all();
        writer_.join();
    }

    /**
     * Hand back the transport counters and the pool cap. The sticky summary is
     * logged here and **not** cleared: it belongs to the collector's whole
     * life, and a flush is what delivers it.
     */
    void release_resources() {
        if (!ready_.exchange(false, std::memory_order_acq_rel)) return;
        owner_.set_drain_quantum(0);
        owner_.set_run_counters(false);
        owner_.release_paired_caps();
        // Last observation, then the consumption that emits the base's
        // aggregate line. The mark moves with the counter it watches: left
        // above a counter the report has zeroed it would hide every later
        // retirement.
        observe_transport();
        owner_.report_drain_drops();
        errors_.rebaseline_transport(0);
        for (size_t slot = 0; slot < epochs_.size(); slot++) {
            const int state = epochs_[slot].state.load(std::memory_order_acquire);
            if (state == static_cast<int>(EpochState::Free)) continue;
            release_deferred_.store(true, std::memory_order_release);
            const uint64_t run_epoch = epochs_[slot].epoch.load(std::memory_order_acquire);
            // A slot still held here owes an artifact that will now never
            // appear, and the sticky record is the only thing that can say so:
            // its shard temps stay on disk and refuse the next admission to
            // that destination, so a flush that reported success would leave
            // the operator with evidence of a run the API called fine.
            //
            // Reachable without any failure of the epoch's own: `finish()`
            // discards its flush result, and `flush()` gives up on its deadline
            // or breaks out early on another epoch's fatal, after which
            // `stop_writer()` joins the writer whether or not it sealed this
            // one. That is the gap this records.
            //
            // A `Quarantined` epoch already has its verdict from the seal that
            // could not prove a release, and a run that was **withdrawn** — the
            // never-launched rollback — has already returned its slot as `Free`
            // and is deliberately not here: it promised no file, so recording
            // one for it would make a rolled-back launch read as lost output.
            if (state != static_cast<int>(EpochState::Quarantined)) {
                errors_.record(
                    run_epoch, Verdict::Abandoned, "the collector was torn down before this run was published"
                );
            }
            LOG_WARN(
                "PmuCollector: run %llu storage is held until the collector threads are joined",
                static_cast<unsigned long long>(run_epoch)
            );
        }
        LOG_INFO("PmuCollector: retained runs released: %s", errors_.report().c_str());
    }

    /**
     * Release what an unpublished epoch held.
     *
     * Only from finalize, after every drain owner and collector shard is
     * joined: storage a seal could not prove safe to touch is unreachable by
     * any reader only then. The files stay on disk as evidence, and the verdict
     * for each of these slots was already recorded by `release_resources`,
     * which is the one walk that can still reach them under a live summary.
     */
    void release_quarantined() {
        if (!release_deferred_.exchange(false, std::memory_order_acq_rel)) return;
        for (size_t slot = 0; slot < epochs_.size(); slot++) {
            if (epochs_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) continue;
            (void)close_shards(slot);
            release_slot(slot);
        }
    }

    /**
     * Test seam: hold the writer just before it would seal anything.
     *
     * A case that has to prove *when* a row was written needs one epoch to be
     * provably still open at a known instant, and the writer is what ends that
     * window. Deliberately the writer and not a collector shard: a shard held
     * here would never acknowledge the epoch table, so the next admission's
     * reference handshake would time out and the collector would go fatal
     * instead of admitting.
     *
     * Bounded and self-releasing: `stop_writer` clears it before joining, so a
     * case that fails an assertion and unwinds still tears down. Production
     * never sets it — nothing outside a test calls this — so the writer pays
     * one relaxed load per service pass.
     */
    void hold_writer_for_test(bool held) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            writer_held_.store(held, std::memory_order_release);
            progress_++;
        }
        cv_.notify_all();
    }

    RetainedRunStats stats() const {
        RetainedRunStats s;
        s.ready = ready_.load(std::memory_order_acquire);
        s.fatal = fatal_.load(std::memory_order_acquire);
        for (const auto &epoch : epochs_) {
            const int state = epoch.state.load(std::memory_order_acquire);
            if (state == static_cast<int>(EpochState::Free)) continue;
            if (state == static_cast<int>(EpochState::Quarantined)) {
                s.quarantined_epochs++;
                continue;
            }
            s.open_epochs++;
        }
        s.errors = errors_.counts();
        return s;
    }

private:
    /**
     * Give back a slot whose configuration the device would not take.
     *
     * Safe because **no buffer can ever carry this run's identity**, not
     * because no reader can see the slot. A shard *may* already have this
     * epoch in its view: the slot is `Admitting` before the publication is
     * attempted, and a predecessor's writer can issue its own reference-release
     * request at any moment, which makes every shard refresh and pick the
     * successor up. What cannot happen is a buffer stamped with this epoch —
     * the stamp is written on the device by `pmu_aicpu_init`, and this run is
     * refused before the runner submits anything, so no producer ever runs
     * under it. `route_buffer` therefore has nothing to resolve to this slot,
     * no shard file for it is ever opened, and a shard left holding a stale
     * view of it can still never write through it.
     *
     * `abandon_run` cannot make that argument — its run *did* launch — which is
     * why it pays for the handshake and quarantines when the acknowledgement
     * does not land.
     *
     * No verdict is recorded. This run never launched, so it promised no file,
     * and reporting one would make a refused admission read as a lost artifact
     * and fail every later flush. The refusal reaches the caller as the launch
     * rc instead. A predecessor's slot, its files and the sticky error record
     * are all untouched.
     */
    void withdraw_unpublished_slot(size_t slot) {
        Epoch<Frozen> &epoch = epochs_[slot];
        remove_shard_temps(epoch.shard_paths);
        release_slot(slot);
    }

    bool ensure_ready() {
        if (ready_.load(std::memory_order_acquire)) return true;
        if (!retain_across_runs_) return false;
        const size_t shard_count = owner_.retained_shard_count();
        if (shard_count == 0) return false;
        for (size_t slot = 0; slot < epochs_.size(); slot++) {
            epochs_[slot].state.store(static_cast<int>(EpochState::Free), std::memory_order_relaxed);
            epochs_[slot].epoch.store(0, std::memory_order_relaxed);
            shard_files_[slot].clear();
            shard_files_[slot].resize(shard_count);
        }
        for (auto &view : views_)
            view = ShardEpochView{};
        // The *gate* is per retention window, so a re-prepared collector can
        // admit runs again. The record of the fatal is in the summary, which is
        // never cleared.
        fatal_.store(false, std::memory_order_release);
        fatal_reason_.clear();
        release_deferred_.store(false, std::memory_order_release);
        close_watermark_.store(0, std::memory_order_release);
        // The drain path's retirement count is cumulative for the collector and
        // is consumed only by the aggregate report, so a new window starts from
        // wherever it stands. Re-baselining clears no recorded failure.
        errors_.rebaseline_transport(owner_.drain_dropped_buffers());
        // A finite quantum so a quiet queue's cut is not starved by a busy
        // sibling, and the per-entry counters a cut's targets compare against.
        owner_.set_drain_quantum(simpler::dfx::runs::kDrainQuantum);
        owner_.set_run_counters(true);
        owner_.install_paired_caps();
        ready_.store(true, std::memory_order_release);
        try {
            start_writer();
        } catch (...) {
            release_resources();
            return false;
        }
        LOG_INFO("PmuCollector: retaining runs, up to %zu unpublished", kMaxOpenEpochs);
        return true;
    }

    void start_writer() {
        if (!ready_.load(std::memory_order_acquire)) return;
        // Joinability, not the flag: a flag left set by a throwing construction
        // would claim a writer that does not exist and can never be replaced.
        if (writer_.joinable()) return;
        writer_running_.store(true, std::memory_order_release);
        try {
            writer_ = std::thread(&RetainedRuns::writer_main, this);
        } catch (...) {
            writer_running_.store(false, std::memory_order_release);
            throw;
        }
    }

    void writer_main() {
        while (writer_running_.load(std::memory_order_acquire)) {
            uint64_t seen = 0;
            {
                std::lock_guard<std::mutex> lk(mu_);
                seen = progress_;
            }
            try {
                service();
            } catch (const std::exception &e) {
                set_fatal(e.what());
            } catch (...) {
                set_fatal("the retained-run writer failed");
            }
            std::unique_lock<std::mutex> lk(mu_);
            if (!writer_running_.load(std::memory_order_acquire)) break;
            if (progress_ != seen) continue;
            const auto wakeup = next_wakeup();
            if (wakeup.has_value()) {
                cv_.wait_until(lk, wakeup.value());
            } else {
                cv_.wait(lk);
            }
        }
    }

    /**
     * The expiry of a stage-2 wait is the only thing the writer needs a clock
     * for; everything else bumps `progress_` and wakes it on the spot.
     */
    std::optional<std::chrono::steady_clock::time_point> next_wakeup() const {
        std::optional<std::chrono::steady_clock::time_point> earliest;
        for (const auto &epoch : epochs_) {
            if (epoch.state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) continue;
            if (!epoch.target_installed) continue;
            const auto expiry = epoch.closed_at + std::chrono::milliseconds(kCutAckBudgetMs * 4);
            if (!earliest.has_value() || expiry < earliest.value()) earliest = expiry;
        }
        return earliest;
    }

    void service() {
        // The test seam, read once per pass: held means "seal nothing yet",
        // never "stop receiving" — the shards keep routing and keep
        // acknowledging the epoch table while this is set.
        if (writer_held_.load(std::memory_order_acquire)) return;
        for (size_t slot = 0; slot < epochs_.size(); slot++) {
            Epoch<Frozen> &epoch = epochs_[slot];
            if (epoch.state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) continue;
            bool closed = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                closed = epoch.target_installed;
            }
            if (!closed) continue;

            // The cut proof, in the order the swimlane writer already
            // evaluates it. A false `cut_failed_queues` is *unknown*, not zero.
            int failed = 0;
            const bool cut_known =
                epoch.cut_slot >= 0 && owner_.cut_failed_queues(epoch.cut_slot, epoch.cut_request, &failed);
            const bool exhausted = owner_.cut_counters_exhausted();
            const bool stage2 = epoch.cut_slot >= 0 && owner_.cut_stage2_done(epoch.cut_slot);
            if (!exhausted && cut_known && failed == 0 && !stage2) {
                // Still settling. Give up only after the bounded wait, and
                // treat the expiry as a failure of proof, not as a pass.
                const auto waited = std::chrono::steady_clock::now() - epoch.closed_at;
                if (waited < std::chrono::milliseconds(kCutAckBudgetMs * 4)) continue;
            }
            epoch.cut.counters_exhausted = exhausted;
            epoch.cut.cut_known = cut_known;
            epoch.cut.failed_queues = failed;
            epoch.cut.stage2_done = stage2;
            seal(slot);
        }
    }

    void seal(size_t slot) {
        Epoch<Frozen> &epoch = epochs_[slot];
        const uint64_t run_epoch = epoch.epoch.load(std::memory_order_acquire);

        // Withdraw admission, then prove no collector shard still holds a
        // reference. Until that is proved nothing may be closed, merged or
        // deleted: a shard could still be appending.
        epoch.state.store(static_cast<int>(EpochState::Closing), std::memory_order_release);
        bool released = false;
        try {
            released = owner_.request_run_reference_release(kControlAckBudgetMs);
        } catch (...) {
            released = false;
        }
        if (!released) {
            epoch.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
            finish_epoch(slot, Verdict::Quarantined, "a collector shard did not release this run's reference");
            return;
        }
        // Retiring the cut slot is the second release and its failure is the
        // same kind: the slot is left un-reused and these files stay untouched.
        if (epoch.cut_slot >= 0 && !owner_.cut_release(epoch.cut_slot, kCutAckBudgetMs)) {
            epoch.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
            finish_epoch(slot, Verdict::Quarantined, "this run's cut slot could not be retired");
            return;
        }
        epoch.cut_slot = -1;

        // Proved released: the files are the writer's alone from here.
        const bool shards_closed = close_shards(slot);
        uint64_t collected = 0;
        for (const uint64_t rows : epoch.shard_rows)
            collected += rows;
        const bool io_failed = !shards_closed || epoch.io_failed.load(std::memory_order_relaxed);
        RecordProofs records = epoch.records;
        if (epoch.clamped.load(std::memory_order_relaxed)) {
            // A clamped buffer is a loss the equation cannot see, so it is
            // folded in as one rather than left to look balanced.
            records.unflushed_records++;
        }

        Verdict verdict = classify(epoch.cut, records, collected, io_failed);
        char detail[kErrorMsgBytes];
        std::snprintf(
            detail, sizeof(detail),
            "collected=%llu dropped=%llu mismatch=%llu unflushed=%llu device_total=%llu cut_known=%d failed=%d "
            "stage2=%d device_complete=%d",
            static_cast<unsigned long long>(collected), static_cast<unsigned long long>(records.dropped_device),
            static_cast<unsigned long long>(records.mismatch_device),
            static_cast<unsigned long long>(records.unflushed_records),
            static_cast<unsigned long long>(records.total_device), epoch.cut.cut_known ? 1 : 0, epoch.cut.failed_queues,
            epoch.cut.stage2_done ? 1 : 0, records.device_execution_complete ? 1 : 0
        );

        if (verdict != Verdict::WriteFailed && collected > 0) {
            if (merge_shards_and_publish(
                    epoch.csv_path, run_epoch, epoch.csv_header, epoch.shard_paths, epoch.shard_rows
                )) {
                remove_shard_temps(epoch.shard_paths);
            } else {
                // A write failure keeps every temp it has: they are the
                // evidence, and the next run under this destination is refused
                // until they are gone.
                verdict = Verdict::WriteFailed;
            }
        } else if (verdict != Verdict::WriteFailed) {
            // No rows, so no file — exactly as the single-run path leaves none.
            // Whether that is a success is the verdict's question.
            remove_shard_temps(epoch.shard_paths);
        }
        finish_epoch(slot, verdict, detail);
    }

    void finish_epoch(size_t slot, Verdict verdict, const char *detail) {
        Epoch<Frozen> &epoch = epochs_[slot];
        const uint64_t run_epoch = epoch.epoch.load(std::memory_order_acquire);
        errors_.record(run_epoch, verdict, detail);
        observe_transport();

        if (verdict == Verdict::Quarantined) {
            // Nothing is freed and the slot is not returned. The reader-join
            // teardown is the only release, and every waiter is woken now so a
            // capacity check or a flush learns instead of blocking.
            release_deferred_.store(true, std::memory_order_release);
            set_fatal(detail);
            try {
                LOG_ERROR(
                    "PmuCollector: run %llu ended %s (%s)", static_cast<unsigned long long>(run_epoch),
                    verdict_name(verdict), detail != nullptr ? detail : ""
                );
            } catch (...) {}
            return;
        }
        try {
            if (verdict_succeeds(verdict)) {
                LOG_INFO(
                    "PmuCollector: run %llu %s", static_cast<unsigned long long>(run_epoch), verdict_name(verdict)
                );
            } else {
                LOG_ERROR(
                    "PmuCollector: run %llu ended %s (%s)", static_cast<unsigned long long>(run_epoch),
                    verdict_name(verdict), detail != nullptr ? detail : ""
                );
            }
        } catch (...) {}
        release_slot(slot);
    }

    /**
     * Hand one slot back.
     *
     * Every field here is read by an admission or a flush **under `mu_`** while
     * the slot is still non-Free — `csv_path` by the destination-collision
     * scan, `target_installed` by the flush's pending test — so the reset and
     * the `Free` publication happen in one critical section with them. Clearing
     * them ahead of the lock would be a data race on the strings and could also
     * make a flush miss a pending transition. Nothing that blocks or does I/O
     * runs under this lock: the reference handshake and the merge are both
     * finished by the time a slot is released.
     */
    void release_slot(size_t slot) {
        Epoch<Frozen> &epoch = epochs_[slot];
        {
            std::lock_guard<std::mutex> lk(mu_);
            epoch.csv_path.clear();
            epoch.csv_header.clear();
            epoch.shard_paths.clear();
            epoch.shard_rows.assign(epoch.shard_rows.size(), 0);
            epoch.target_installed = false;
            epoch.clamped.store(false, std::memory_order_relaxed);
            epoch.io_failed.store(false, std::memory_order_relaxed);
            epoch.state.store(static_cast<int>(EpochState::Free), std::memory_order_release);
            progress_++;
        }
        cv_.notify_all();
    }

    bool ensure_shard_open(size_t slot, size_t shard) {
        auto &files = shard_files_[slot];
        if (shard >= files.size() || shard >= epochs_[slot].shard_paths.size()) {
            epochs_[slot].io_failed.store(true, std::memory_order_relaxed);
            return false;
        }
        auto &file = files[shard];
        if (file.is_open()) return true;
        file.open(epochs_[slot].shard_paths[shard], std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            LOG_ERROR("PmuCollector: failed to open shard file %s", epochs_[slot].shard_paths[shard].c_str());
            epochs_[slot].io_failed.store(true, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    bool close_shards(size_t slot) {
        bool ok = true;
        for (auto &file : shard_files_[slot]) {
            if (!file.is_open()) continue;
            file.flush();
            if (!file.good()) ok = false;
            file.close();
            if (file.fail()) ok = false;
            file.clear();
        }
        if (!ok) epochs_[slot].io_failed.store(true, std::memory_order_relaxed);
        return ok;
    }

    int find_slot(uint64_t run_epoch) const {
        for (size_t slot = 0; slot < epochs_.size(); slot++) {
            if (epochs_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) continue;
            if (epochs_[slot].epoch.load(std::memory_order_acquire) == run_epoch) return static_cast<int>(slot);
        }
        return -1;
    }

    void observe_transport() { errors_.observe_transport(owner_.drain_dropped_buffers()); }

    void set_fatal(const char *reason) {
        bool first = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!fatal_.load(std::memory_order_acquire)) {
                fatal_.store(true, std::memory_order_release);
                try {
                    fatal_reason_ = reason != nullptr ? reason : "unknown";
                } catch (...) {}
                first = true;
            }
            progress_++;
        }
        // Every waiter is woken: a refusal is what a capacity check and a flush
        // are waiting to read, so neither may block past it.
        cv_.notify_all();
        if (first) errors_.record_fatal(reason);
    }

    Collector &owner_;

    bool retain_across_runs_{false};
    std::atomic<bool> ready_{false};
    std::array<Epoch<Frozen>, kMaxOpenEpochs> epochs_{};
    // One output stream per (epoch slot, collector shard). Written only by the
    // collector shard that owns that index while its epoch admits, and read by
    // the writer only after every shard's reference is proved released, so the
    // streams themselves need no lock.
    std::array<std::vector<std::ofstream>, kMaxOpenEpochs> shard_files_{};
    // Per collector shard, not shared: a shard reads its own copy with no lock
    // and only ever adopts a new one inside the reference-release handshake.
    std::array<ShardEpochView, MaxShards> views_{};

    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint64_t progress_{0};
    // The highest run epoch whose close a flush must wait for. UINT64_MAX once
    // `finish` has stopped admission.
    std::atomic<uint64_t> close_watermark_{0};
    std::atomic<bool> fatal_{false};
    std::string fatal_reason_;
    std::thread writer_;
    std::atomic<bool> writer_running_{false};
    // Test seam only; see `hold_writer_for_test`. Production never sets it.
    std::atomic<bool> writer_held_{false};
    // Set when a quarantined epoch's storage may only be released after the
    // collector threads are joined.
    std::atomic<bool> release_deferred_{false};
    // Sticky for the collector's whole life: no reset, no clear, no ack API.
    ErrorSummary errors_;
};

}  // namespace simpler::dfx::pmu
