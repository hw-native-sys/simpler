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
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "common/unified_log.h"
#include "host/chip_swimlane_runs.h"

/**
 * ArgsDump's cross-run support types: its verdicts, its sticky failure record,
 * the per-lane receipt ledger that decides a leftover buffer's identity, and
 * the exclusive output token a retained run publishes under.
 *
 * Only the collector-independent halves live here. Unlike PMU — whose epoch
 * table is a template because each architecture has its own collector — there
 * is one `ArgsDumpCollector` for both architectures, so its epoch table and
 * background writer live in the collector itself rather than in a template
 * parameterized over it.
 *
 * The accountant, the policy constants and the checked-counter helpers are
 * shared with the other retaining collectors and are used from
 * `simpler::dfx::runs` unchanged.
 */
namespace simpler::dfx::args_dump {

using simpler::dfx::runs::kErrorMsgBytes;

// How many `args.<token>.bin` / `args_dump.json.<token>.tmp` name pairs one
// epoch will try before refusing the run. A run epoch is process-unique, so an
// ordinary repeated run takes the first candidate; the sweep exists only for a
// destination two processes reach with the same epoch number.
inline constexpr int kMaxOutputTokenCandidates = 16;

/** Why a retained ArgsDump run's output says what it says. */
enum class Verdict {
    Published,       // every record and payload byte this run produced is in its pair
    PublishedEmpty,  // proved to have produced no record; no payload file content
    PublishedShort,  // published, with host-discarded or device-dropped records
    CountsUnknown,   // completeness not provable: device stop, identity, cut or a read
    WriteFailed,     // sealed, but the pair was not published
    Quarantined,     // references not proved released; nothing may be sorted or freed
    Abandoned,       // torn down before this run was published
    CounterExhausted,
    NameUnavailable,
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
    case Verdict::WriteFailed:
        return "write_failed";
    case Verdict::Quarantined:
        return "quarantined";
    case Verdict::Abandoned:
        return "abandoned";
    case Verdict::CounterExhausted:
        return "counter_exhausted";
    case Verdict::NameUnavailable:
        return "name_unavailable";
    }
    return "unknown";
}

/** True when this verdict lets `flush_diagnostics` report success. */
inline bool verdict_succeeds(Verdict v) { return v == Verdict::Published || v == Verdict::PublishedEmpty; }

/**
 * True when this verdict still publishes a manifest.
 *
 * `PublishedShort` and `CountsUnknown` do: the manifest carries what the run
 * did collect and says on its face that it is incomplete, which is more use to
 * a reader than an orphan payload file. Neither lets the flush succeed.
 */
inline bool verdict_publishes(Verdict v) {
    return v == Verdict::Published || v == Verdict::PublishedEmpty || v == Verdict::PublishedShort ||
           v == Verdict::CountsUnknown;
}

/**
 * Per lane, per run: what this run's buffers proved about their own delivery.
 *
 * Built by the drain shard as it receives, never reconstructed afterwards. The
 * ready queue's head and tail are ring indices — `(tail + 1) % kReadyQueueSize`
 * at both the producer's gate and its publish — so differencing two snapshots
 * of them cannot count publications, and a device buffer address cannot
 * identify an incarnation on its own.
 */
struct LaneReceipt {
    uint64_t received_buffers{0};
    // The `local_seq` this lane's next publication must carry. The device
    // stamps 0 at `dump_args_init` and advances one per mid-run switch, so this
    // is a uint64 accumulator against the device's uint32 field: a run that
    // wrapped it is detected below rather than aliased.
    uint64_t next_expected_seq{0};
    // Set when a delivery did not carry the successor sequence, when the ready
    // entry and the buffer body disagreed about it, or when the accumulator
    // passed the device field's range. Any of the three makes this lane's
    // leftover buffer undecidable.
    bool identity_uncertain{false};

    /**
     * Fold in one delivered buffer of this run on this lane.
     *
     * `entry_seq` comes from the ready-queue entry and `buffer_seq` from the
     * buffer body; the device writes both from `current_buf_seq`, so they are
     * two independent witnesses of one value.
     */
    void observe(uint32_t entry_seq, uint32_t buffer_seq) {
        if (entry_seq != buffer_seq) {
            identity_uncertain = true;
            return;
        }
        if (static_cast<uint64_t>(buffer_seq) != next_expected_seq) {
            identity_uncertain = true;
            return;
        }
        received_buffers++;
        next_expected_seq++;
        if (next_expected_seq > UINT32_MAX) identity_uncertain = true;
    }
};

/** What the close decided about one lane's leftover buffer. */
enum class LeftoverOutcome {
    None,              // no buffer was named, or the one named held nothing
    AlreadyDelivered,  // handed over already; its contents were not read
    Recovered,         // proved unpublished: metadata and payload copied under the claim
    Unknown,           // identity, mapping or copy unproved; nothing was read
};

/** This run's terminal device-side state, read narrowly under its claim. */
struct RecordProofs {
    bool device_execution_complete{false};
    // False when a narrow read of the shared region failed. No partial sum from
    // a failed read is ever folded in.
    bool terminal_readable{false};
    // False when the close could not prove, inside the execution claim, that
    // every buffer this run published had been processed into host-owned
    // storage. A capture acknowledgement is not that proof; stage 2 of the cut
    // is. Without it no leftover is read and no unprocessed payload is
    // acknowledged, so the run is incomplete rather than wrong.
    bool processing_proved{true};
    uint64_t published_payloads{0};
    uint64_t dropped_records{0};
    uint64_t recovered_records{0};
    uint64_t unknown_lanes{0};
    uint64_t already_delivered_lanes{0};
};

/** The finite per-queue transport cut's outcome, in the order it is proved. */
struct CutProofs {
    bool counters_exhausted{false};
    bool cut_known{false};
    int failed_queues{0};
    bool stage2_done{false};
};

/** Host-side losses this run recorded while receiving. */
struct HostLoss {
    uint64_t discarded_args{0};
    uint64_t discarded_metadata_records{0};
    bool io_failed{false};
};

/**
 * One run's verdict from its proofs.
 *
 * Ordered by severity of what cannot be said: a write failure leaves no pair,
 * an exhausted counter voids every count, an unproved completeness question
 * outranks a known loss, and only a run with nothing unproved and nothing lost
 * may succeed.
 */
inline Verdict
classify(const CutProofs &cut, const RecordProofs &rec, const HostLoss &loss, uint64_t collected_records) {
    if (loss.io_failed) return Verdict::WriteFailed;
    if (cut.counters_exhausted) return Verdict::CounterExhausted;
    if (!rec.device_execution_complete || !rec.terminal_readable || !rec.processing_proved || rec.unknown_lanes > 0) {
        return Verdict::CountsUnknown;
    }
    if (!cut.cut_known || cut.failed_queues > 0 || !cut.stage2_done) return Verdict::CountsUnknown;
    if (loss.discarded_args > 0 || loss.discarded_metadata_records > 0 || rec.dropped_records > 0) {
        return Verdict::PublishedShort;
    }
    if (collected_records == 0) return Verdict::PublishedEmpty;
    return Verdict::Published;
}

/**
 * The collector's permanent progress and failure record.
 *
 * Sticky to the runner's destruction: there is no reset, no clear and no
 * acknowledgement API, so a failure recorded before a collector rebuild — or
 * before the operator deleted the evidence files on disk — is still reported by
 * every later flush. A collector-level fatal is an error in its own right,
 * because a writer that died before sealing anything leaves no per-epoch
 * verdict for an emptiness test to find.
 */
class ErrorSummary {
public:
    void record(uint64_t epoch, Verdict v, const char *detail) {
        std::lock_guard<std::mutex> lk(mu_);
        counts_[static_cast<size_t>(v)]++;
        if (verdict_publishes(v) && epoch > highest_published_) highest_published_ = epoch;
        if (verdict_succeeds(v)) return;
        note_error_locked(epoch, v, detail);
    }

    void record_fatal(const char *detail) {
        std::lock_guard<std::mutex> lk(mu_);
        fatal_++;
        if (fatal_msg_[0] == '\0' && detail != nullptr) {
            std::snprintf(fatal_msg_, sizeof(fatal_msg_), "%s", detail);
        }
        error_recorded_ = true;
    }

    /** A buffer arrived for an epoch that is sealed or was never admitted. */
    void record_unknown_epoch(uint64_t buffer_epoch, uint64_t records) {
        std::lock_guard<std::mutex> lk(mu_);
        unknown_epoch_buffers_++;
        unknown_epoch_records_ += records;
        error_recorded_ = true;
        if (first_unknown_epoch_ == 0) first_unknown_epoch_ = buffer_epoch;
    }

    bool has_error() const {
        std::lock_guard<std::mutex> lk(mu_);
        return error_recorded_;
    }

    uint64_t published_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return counts_[static_cast<size_t>(Verdict::Published)] +
               counts_[static_cast<size_t>(Verdict::PublishedEmpty)] +
               counts_[static_cast<size_t>(Verdict::PublishedShort)] +
               counts_[static_cast<size_t>(Verdict::CountsUnknown)];
    }

    /** One bounded line for a flush or close failure. Never empty on error. */
    std::string report() const {
        std::lock_guard<std::mutex> lk(mu_);
        char buf[kErrorMsgBytes * 3];
        int n = std::snprintf(
            buf, sizeof(buf),
            "published=%llu empty=%llu short=%llu counts_unknown=%llu write_failed=%llu quarantined=%llu "
            "abandoned=%llu counter_exhausted=%llu name_unavailable=%llu",
            count_of(Verdict::Published), count_of(Verdict::PublishedEmpty), count_of(Verdict::PublishedShort),
            count_of(Verdict::CountsUnknown), count_of(Verdict::WriteFailed), count_of(Verdict::Quarantined),
            count_of(Verdict::Abandoned), count_of(Verdict::CounterExhausted), count_of(Verdict::NameUnavailable)
        );
        if (n < 0) return {};
        size_t used = static_cast<size_t>(n) < sizeof(buf) ? static_cast<size_t>(n) : sizeof(buf) - 1;
        if (unknown_epoch_buffers_ != 0 && used < sizeof(buf)) {
            const int k = std::snprintf(
                buf + used, sizeof(buf) - used, "; %llu buffer(s) for no open epoch (first %llu, %llu records)",
                static_cast<unsigned long long>(unknown_epoch_buffers_),
                static_cast<unsigned long long>(first_unknown_epoch_),
                static_cast<unsigned long long>(unknown_epoch_records_)
            );
            if (k > 0) used += static_cast<size_t>(k);
        }
        if (fatal_ != 0 && used < sizeof(buf)) {
            const int k = std::snprintf(
                buf + used, sizeof(buf) - used, "; collector fatal (%llu): %s", static_cast<unsigned long long>(fatal_),
                fatal_msg_
            );
            if (k > 0) used += static_cast<size_t>(k);
        }
        if (error_recorded_ && first_error_epoch_ != 0 && used < sizeof(buf)) {
            std::snprintf(
                buf + used, sizeof(buf) - used, "; first failed run %llu (%s): %s",
                static_cast<unsigned long long>(first_error_epoch_), verdict_name(first_error_verdict_),
                first_error_msg_
            );
        }
        return {buf};
    }

    /** Every per-verdict count, read as one consistent set under the lock. */
    struct Counts {
        uint64_t published{0};
        uint64_t published_empty{0};
        uint64_t published_short{0};
        uint64_t counts_unknown{0};
        uint64_t write_failed{0};
        uint64_t quarantined{0};
        uint64_t abandoned{0};
        uint64_t counter_exhausted{0};
        uint64_t name_unavailable{0};
        uint64_t fatal{0};
        uint64_t unknown_epoch_buffers{0};
    };
    Counts counts() const {
        std::lock_guard<std::mutex> lk(mu_);
        Counts c;
        c.published = count_of(Verdict::Published);
        c.published_empty = count_of(Verdict::PublishedEmpty);
        c.published_short = count_of(Verdict::PublishedShort);
        c.counts_unknown = count_of(Verdict::CountsUnknown);
        c.write_failed = count_of(Verdict::WriteFailed);
        c.quarantined = count_of(Verdict::Quarantined);
        c.abandoned = count_of(Verdict::Abandoned);
        c.counter_exhausted = count_of(Verdict::CounterExhausted);
        c.name_unavailable = count_of(Verdict::NameUnavailable);
        c.fatal = fatal_;
        c.unknown_epoch_buffers = unknown_epoch_buffers_;
        return c;
    }

private:
    static constexpr size_t kVerdictCount = static_cast<size_t>(Verdict::NameUnavailable) + 1;

    unsigned long long count_of(Verdict v) const {
        return static_cast<unsigned long long>(counts_[static_cast<size_t>(v)]);
    }

    void note_error_locked(uint64_t epoch, Verdict v, const char *detail) {
        error_recorded_ = true;
        if (first_error_epoch_ != 0) return;
        first_error_epoch_ = epoch;
        first_error_verdict_ = v;
        if (detail != nullptr) std::snprintf(first_error_msg_, sizeof(first_error_msg_), "%s", detail);
    }

    mutable std::mutex mu_;
    std::array<uint64_t, kVerdictCount> counts_{};
    uint64_t fatal_{0};
    uint64_t highest_published_{0};
    uint64_t unknown_epoch_buffers_{0};
    uint64_t unknown_epoch_records_{0};
    uint64_t first_unknown_epoch_{0};
    bool error_recorded_{false};
    uint64_t first_error_epoch_{0};
    Verdict first_error_verdict_{Verdict::Published};
    char first_error_msg_[kErrorMsgBytes]{};
    char fatal_msg_[kErrorMsgBytes]{};
};

// ---------------------------------------------------------------------------
// Output identity
// ---------------------------------------------------------------------------

/**
 * One epoch's exclusively owned output names.
 *
 * A run epoch is minted by a process-local counter, so it is unique within a
 * process and not across processes: reserving only the payload name would
 * still leave two writers able to truncate or rename each other's temporary
 * manifest. So the reservation owns the *pair*, and the payload is afterwards
 * opened for append only — it is never truncated, which is what keeps an
 * in-progress run from touching a published pair.
 */
struct OutputToken {
    std::string payload_name;   // what the manifest's `bin_file` names
    std::string payload_path;   // <run_dir>/args.<token>.bin
    std::string manifest_tmp;   // <run_dir>/args_dump.json.<token>.tmp
    std::string manifest_path;  // <run_dir>/args_dump.json
    bool valid{false};
};

/** Create `path` and fail if it already exists. Closes its descriptor. */
inline bool create_exclusive(const std::string &path) {
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) return false;
    ::close(fd);
    return true;
}

/**
 * Reserve an exclusive payload + temporary-manifest pair for one epoch.
 *
 * Both names must be free for a candidate to be taken; a candidate that
 * created one of the two removes **the file it just created** before moving
 * on, which is the only removal this collector ever performs — a file it made
 * microseconds earlier, that was never published and that nothing names.
 */
inline bool reserve_output_token(const std::filesystem::path &run_dir, uint64_t run_epoch, OutputToken *out) {
    if (out == nullptr) return false;
    char suffix[64];
    for (int n = 0; n < kMaxOutputTokenCandidates; n++) {
        if (n == 0) {
            std::snprintf(suffix, sizeof(suffix), "e%llu", static_cast<unsigned long long>(run_epoch));
        } else {
            std::snprintf(suffix, sizeof(suffix), "e%llu.%d", static_cast<unsigned long long>(run_epoch), n);
        }
        OutputToken token;
        token.payload_name = std::string("args.") + suffix + ".bin";
        token.payload_path = (run_dir / token.payload_name).string();
        token.manifest_tmp = (run_dir / (std::string("args_dump.json.") + suffix + ".tmp")).string();
        token.manifest_path = (run_dir / "args_dump.json").string();

        if (!create_exclusive(token.payload_path)) continue;
        if (!create_exclusive(token.manifest_tmp)) {
            // Half a reservation is never kept: this candidate's own payload
            // file goes, and nothing that predates this call is touched.
            std::error_code ec;
            std::filesystem::remove(token.payload_path, ec);
            continue;
        }
        token.valid = true;
        *out = std::move(token);
        return true;
    }
    return false;
}

/**
 * Whether a destination still holds a failed run's preserved evidence.
 *
 * A failed epoch keeps its payload file and its temporary manifest, and this
 * collector never deletes either, so the next run under the same destination is
 * refused rather than allowed to publish beside evidence nobody has read.
 * A *published* run's payload file is not evidence and does not refuse
 * anything — it is named by a manifest and is kept for readers holding it.
 */
inline bool failure_evidence_present(const std::filesystem::path &run_dir) {
    std::error_code ec;
    std::filesystem::directory_iterator it(run_dir, ec);
    if (ec) return false;
    for (const auto &entry : it) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("args_dump.json.", 0) == 0 && name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace simpler::dfx::args_dump
