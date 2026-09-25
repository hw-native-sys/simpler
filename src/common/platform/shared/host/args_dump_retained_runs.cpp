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
 * @file args_dump_retained_runs.cpp
 * @brief The retained half of ArgsDumpCollector: one run's payload file and
 *        manifest are finished on a background writer while the next run
 *        executes on the device.
 *
 * Everything a run's output depends on is frozen at its admission, everything
 * it retains is charged against a per-collector budget before it is allocated,
 * and the completeness of what it publishes is decided from proofs taken while
 * that run still holds the execution claim. The receive path, the manifest
 * shape and the arena acknowledgement are shared with the single-run path in
 * `args_dump_collector.cpp`; only the ownership and the sealing live here.
 */

#include "host/args_dump_collector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <thread>
#include <system_error>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/args_dump_manifest.h"
#include "../../../worker/runtime_c_api.h"

namespace {

using simpler::dfx::args_dump::LeftoverOutcome;
using simpler::dfx::args_dump::Verdict;
using simpler::dfx::runs::kControlAckBudgetMs;
using simpler::dfx::runs::kCutAckBudgetMs;
using simpler::dfx::runs::kErrorMsgBytes;
using simpler::dfx::runs::kMaxOpenEpochs;
using simpler::dfx::runs::kPathAllowanceBytes;

// Payload requests one writer pass takes before it re-evaluates seals, so a
// continuously producing successor cannot starve a closed run's publication and
// a closed run's seal cannot stall payload acknowledgement.
constexpr int kWriterPayloadBatch = 64;

// How long the writer waits for a cut to settle before treating the expiry as
// a failure of proof rather than as a pass.
constexpr int kStage2WaitMs = kCutAckBudgetMs * 4;

// A metadata bucket's first charged block, in records.
constexpr size_t kBucketInitialRecords = 256;

bool arg_order_before(const DumpedArg &a, const DumpedArg &b) {
    if (a.task_id != b.task_id) return a.task_id < b.task_id;
    if (a.stage != b.stage) return static_cast<uint8_t>(a.stage) < static_cast<uint8_t>(b.stage);
    if (a.arg_index != b.arg_index) return a.arg_index < b.arg_index;
    return static_cast<uint8_t>(a.role) < static_cast<uint8_t>(b.role);
}

}  // namespace

// ---------------------------------------------------------------------------
// Epoch state
// ---------------------------------------------------------------------------

void ArgsDumpCollector::RetainedEpoch::reset_run_state(size_t bucket_count, int lane_count) {
    buckets.assign(bucket_count, {});
    bucket_charged.assign(bucket_count, 0);
    receipts.assign(static_cast<size_t>(lane_count), {});
    admit_published.assign(static_cast<size_t>(lane_count), 0);
    admit_dropped.assign(static_cast<size_t>(lane_count), 0);
    next_bin_offset = 0;
    queued_payloads.store(0, std::memory_order_relaxed);
    payload_opened = false;
    payload_bytes_written.store(0, std::memory_order_relaxed);
    collected_records.store(0, std::memory_order_relaxed);
    truncated_records.store(0, std::memory_order_relaxed);
    discarded_args.store(0, std::memory_order_relaxed);
    discarded_metadata_records.store(0, std::memory_order_relaxed);
    io_failed.store(false, std::memory_order_relaxed);
    records = simpler::dfx::args_dump::RecordProofs{};
    cut = simpler::dfx::args_dump::CutProofs{};
    cut_slot = -1;
    cut_request = 0;
    target_installed = false;
    cut_settled_at_close = false;
    fixed_charge = 0;
}

void ArgsDumpCollector::configure_retained_runs(bool enabled, size_t budget_bytes) {
    retain_across_runs_ = enabled;
    retained_budget_bytes_ = budget_bytes;
}

void ArgsDumpCollector::retained_bump_progress() {
    {
        std::scoped_lock<std::mutex> lock(write_mutex_);
        retained_progress_++;
    }
    write_cv_.notify_all();
}

void ArgsDumpCollector::note_transport_progress() {
    if (!retained_ready_.load(std::memory_order_acquire)) return;
    retained_bump_progress();
}

void ArgsDumpCollector::refresh_retained_run_view(int collector_shard) {
    if (collector_shard < 0 || static_cast<size_t>(collector_shard) >= shard_views_.size()) return;
    ShardEpochView view;
    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        for (size_t slot = 0; slot < retained_epochs_.size(); slot++) {
            if (retained_epochs_[slot].state.load(std::memory_order_acquire) !=
                static_cast<int>(EpochState::Admitting)) {
                continue;
            }
            view.entries[view.count].epoch = retained_epochs_[slot].epoch.load(std::memory_order_acquire);
            view.entries[view.count].slot = static_cast<int>(slot);
            view.count++;
        }
    }
    shard_views_[static_cast<size_t>(collector_shard)] = view;
}

void ArgsDumpCollector::retained_set_fatal(const char *reason) {
    bool first = false;
    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        if (!retained_fatal_.load(std::memory_order_acquire)) {
            retained_fatal_.store(true, std::memory_order_release);
            try {
                retained_fatal_reason_ = reason != nullptr ? reason : "unknown";
            } catch (...) {}
            first = true;
        }
    }
    // Every waiter is woken: a refusal is what a capacity check and a flush are
    // waiting to read, so neither may block past it.
    retained_cv_.notify_all();
    retained_bump_progress();
    if (first) retained_errors_.record_fatal(reason);
}

int ArgsDumpCollector::retained_find_slot(uint64_t run_epoch) const {
    for (size_t slot = 0; slot < retained_epochs_.size(); slot++) {
        if (retained_epochs_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) {
            continue;
        }
        if (retained_epochs_[slot].epoch.load(std::memory_order_acquire) == run_epoch) return static_cast<int>(slot);
    }
    return -1;
}

bool ArgsDumpCollector::retained_ensure_ready() {
    if (retained_ready_.load(std::memory_order_acquire)) return true;
    if (!retain_across_runs_) return false;
    if (shm_host_ == nullptr || dump_shared_mem_dev_ == nullptr) return false;
    if (num_dump_threads_ <= 0) return false;

    const size_t bucket_count = static_cast<size_t>(manager_.shard_count()) + 1;
    // The fixed reservation covers each epoch's frozen path strings, its
    // per-lane ledger and admission snapshots, and its bucket headers —
    // everything an admitted run holds before it receives a single record.
    size_t per_epoch_fixed = 0;
    if (!simpler::dfx::runs::checked_bytes(4, kPathAllowanceBytes, &per_epoch_fixed)) return false;
    per_epoch_fixed += static_cast<size_t>(num_dump_threads_) *
                       (sizeof(simpler::dfx::args_dump::LaneReceipt) + sizeof(uint64_t) + sizeof(uint32_t));
    per_epoch_fixed += bucket_count * (sizeof(std::vector<DumpedArg>) + sizeof(size_t));
    if (!retained_budget_.open(retained_budget_bytes_, per_epoch_fixed * kMaxOpenEpochs)) return false;

    for (auto &epoch : retained_epochs_) {
        epoch.state.store(static_cast<int>(EpochState::Free), std::memory_order_relaxed);
        epoch.epoch.store(0, std::memory_order_relaxed);
        epoch.reset_run_state(bucket_count, num_dump_threads_);
    }
    for (auto &view : shard_views_)
        view = ShardEpochView{};
    // The *gate* is per retention window, so a re-prepared collector can admit
    // runs again. The record of the fatal is in the summary, which is never
    // cleared.
    retained_fatal_.store(false, std::memory_order_release);
    retained_fatal_reason_.clear();
    retained_release_deferred_.store(false, std::memory_order_release);
    retained_close_watermark_.store(0, std::memory_order_release);
    // A finite quantum so a quiet queue's cut is not starved by a busy sibling,
    // and the per-entry counters a cut's targets compare against.
    set_drain_quantum(simpler::dfx::runs::kDrainQuantum);
    set_run_counters(true);
    retained_ready_.store(true, std::memory_order_release);
    try {
        retained_start_writer();
    } catch (...) {
        retained_ready_.store(false, std::memory_order_release);
        set_run_counters(false);
        set_drain_quantum(0);
        retained_budget_.close();
        LOG_ERROR("Args dump: the retained-run writer could not be started");
        return false;
    }
    LOG_INFO(
        "Args dump: retaining runs, up to %zu unpublished, %zu MiB host budget", kMaxOpenEpochs,
        retained_budget_bytes_ / (1024 * 1024)
    );
    return true;
}

void ArgsDumpCollector::retained_start_writer() {
    if (!retained_ready_.load(std::memory_order_acquire)) return;
    // Joinability, not a flag: a flag left set by a throwing construction would
    // claim a writer that does not exist and can never be replaced.
    if (retained_writer_thread_.joinable()) return;
    retained_writer_running_.store(true, std::memory_order_release);
    try {
        retained_writer_thread_ = std::thread(&ArgsDumpCollector::retained_writer_loop, this);
    } catch (...) {
        retained_writer_running_.store(false, std::memory_order_release);
        throw;
    }
}

void ArgsDumpCollector::retained_stop_writer() {
    if (!retained_writer_thread_.joinable()) return;
    {
        std::scoped_lock<std::mutex> lock(write_mutex_);
        // A held writer must not be joined while it is holding: releasing here
        // is what keeps a case that unwound mid-test from hanging teardown.
        retained_writer_held_.store(false, std::memory_order_release);
        retained_writer_running_.store(false, std::memory_order_release);
        retained_progress_++;
    }
    write_cv_.notify_all();
    retained_writer_thread_.join();
}

bool ArgsDumpCollector::retained_publish_level(DumpArgsLevel level) {
    if (shm_host_ == nullptr || dump_shared_mem_dev_ == nullptr) return false;
    DumpDataHeader *header = get_dump_header(shm_host_);
    header->dump_args_level = static_cast<uint32_t>(level);
    wmb();
    publish_field(&header->dump_args_level, sizeof(header->dump_args_level), "dump_args_level");

    // Read back what the device will actually latch. Which read that is depends
    // on the platform, and the difference is not cosmetic: on an SVM platform
    // the host shadow and the device region are one allocation and the copy
    // hooks are deliberate no-ops, so a copy into a local would observe
    // nothing and report every run's level as refused.
    DumpDataHeader *device_header = get_dump_header(dump_shared_mem_dev_);
    uint32_t observed = 0;
    if (shm_host_ == dump_shared_mem_dev_) {
        observed = device_header->dump_args_level;
    } else if (profiling_copy_from_device(&observed, &device_header->dump_args_level, sizeof(observed)) != 0) {
        return false;
    }
    // A level the device did not take would have it select one set of arguments
    // while this run's manifest names another, and the record counts could
    // still balance — so nothing later in the pipeline could catch it.
    return observed == static_cast<uint32_t>(level);
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

bool ArgsDumpCollector::run_begin(uint64_t run_epoch, const std::string &output_prefix, DumpArgsLevel dump_args_level) {
    // A guard, not the configuration question's answer: a caller that reaches
    // here on a collector that retains nothing has already taken the wrong path.
    if (!retain_across_runs_) return false;
    if (!retained_ensure_ready()) {
        LOG_ERROR(
            "Args dump: run %llu refused, retention could not be prepared", static_cast<unsigned long long>(run_epoch)
        );
        return false;
    }
    if (retained_fatal_.load(std::memory_order_acquire)) {
        LOG_ERROR("Args dump: run %llu refused, the collector is fatal", static_cast<unsigned long long>(run_epoch));
        return false;
    }
    // Refused before a slot is claimed: every path this run retains is reserved
    // against the allowance rather than charged, so one that cannot fit is
    // turned away here. The margin covers the token-bearing suffixes.
    if (output_prefix.size() + 128 > kPathAllowanceBytes) {
        LOG_ERROR(
            "Args dump: run %llu refused, its output prefix of %zu bytes exceeds the %zu byte allowance",
            static_cast<unsigned long long>(run_epoch), output_prefix.size(), kPathAllowanceBytes
        );
        return false;
    }

    std::filesystem::path run_dir;
    try {
        run_dir = std::filesystem::path(output_prefix) / "args_dump";
        std::filesystem::create_directories(run_dir);
    } catch (const std::exception &e) {
        LOG_ERROR("Args dump: run %llu refused, %s", static_cast<unsigned long long>(run_epoch), e.what());
        return false;
    }
    // A failed run's temporary manifest is its evidence and this contract never
    // deletes it, so a later run under the same destination is refused rather
    // than allowed to publish beside evidence nobody has read.
    if (simpler::dfx::args_dump::failure_evidence_present(run_dir)) {
        LOG_ERROR(
            "Args dump: run %llu refused, %s still holds a failed run's temporary manifest; "
            "read it and remove it to reuse this destination",
            static_cast<unsigned long long>(run_epoch), run_dir.c_str()
        );
        return false;
    }

    size_t slot = 0;
    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        bool found = false;
        for (size_t i = 0; i < retained_epochs_.size(); i++) {
            if (retained_epochs_[i].state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Free)) {
                // A destination an open run owns would have two writers and one
                // publication point.
                if (retained_epochs_[i].run_dir == run_dir) {
                    LOG_ERROR(
                        "Args dump: run %llu refused, %s is still owned by run %llu",
                        static_cast<unsigned long long>(run_epoch), run_dir.c_str(),
                        static_cast<unsigned long long>(retained_epochs_[i].epoch.load(std::memory_order_acquire))
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
            // Refused, not queued: the device is about to be handed this run,
            // and a wait here would hold the launch behind a slow disk.
            LOG_ERROR(
                "Args dump: run %llu refused, both retained runs are still unpublished",
                static_cast<unsigned long long>(run_epoch)
            );
            return false;
        }
    }

    RetainedEpoch &epoch = retained_epochs_[slot];
    const size_t bucket_count = static_cast<size_t>(manager_.shard_count()) + 1;
    epoch.reset_run_state(bucket_count, num_dump_threads_);
    epoch.epoch.store(run_epoch, std::memory_order_relaxed);
    epoch.run_dir = run_dir;
    epoch.level = dump_args_level;

    // One exclusive token owns both this run's payload file and its temporary
    // manifest, reserved before either is opened.
    if (!simpler::dfx::args_dump::reserve_output_token(run_dir, run_epoch, &epoch.token)) {
        LOG_ERROR(
            "Args dump: run %llu refused, no exclusive output name pair was available in %s",
            static_cast<unsigned long long>(run_epoch), run_dir.c_str()
        );
        retained_errors_.record(run_epoch, Verdict::NameUnavailable, "no exclusive output name pair was available");
        return false;
    }
    // Append only. The exclusive reservation is what makes that safe, and it is
    // also what keeps an in-progress run from ever truncating a published pair.
    epoch.payload_file.open(epoch.token.payload_path, std::ios::binary | std::ios::app);
    if (!epoch.payload_file.is_open()) {
        LOG_ERROR(
            "Args dump: run %llu refused, could not open %s", static_cast<unsigned long long>(run_epoch),
            epoch.token.payload_path.c_str()
        );
        epoch.payload_file.clear();
        return false;
    }
    epoch.payload_opened = true;

    // Per-lane admission snapshots. The device's published and dropped counters
    // are never reset on this path — a predecessor's writer is still
    // acknowledging against them — so a run's own figures are its close reading
    // minus this instant.
    for (int t = 0; t < num_dump_threads_ && static_cast<size_t>(t) < epoch.admit_published.size(); t++) {
        DumpBufferState *host_state = get_dump_buffer_state(shm_host_, t);
        DumpBufferState *device_state = get_dump_buffer_state(dump_shared_mem_dev_, t);
        static_assert(
            offsetof(DumpBufferState, dropped_record_count) ==
                offsetof(DumpBufferState, published_payload_count) + 2 * sizeof(uint64_t),
            "the payload counters must stay contiguous for this single narrow read to cover them"
        );
        constexpr size_t kCounterSpan = 2 * sizeof(uint64_t) + sizeof(uint32_t);
        if (profiling_copy_from_device(
                &host_state->published_payload_count, &device_state->published_payload_count, kCounterSpan
            ) != 0) {
            LOG_ERROR(
                "Args dump: run %llu refused, lane %d counters were not readable",
                static_cast<unsigned long long>(run_epoch), t
            );
            epoch.payload_file.close();
            epoch.payload_file.clear();
            epoch.payload_opened = false;
            return false;
        }
        if (!simpler::dfx::runs::counter_headroom(host_state->published_payload_count)) {
            // The counters are monotonic for this collector's life, so their
            // headroom is checked where a run can still be refused. A rebuild
            // is what resets both sides consistently; this check lowers the
            // risk of reaching the bound and does not prove that an
            // arbitrarily long single run stays inside it.
            retained_errors_.record(
                run_epoch, Verdict::CounterExhausted, "a lane's payload counter is out of headroom"
            );
            LOG_ERROR(
                "Args dump: run %llu refused, lane %d payload counter is out of headroom",
                static_cast<unsigned long long>(run_epoch), t
            );
            epoch.payload_file.close();
            epoch.payload_file.clear();
            epoch.payload_opened = false;
            return false;
        }
        epoch.admit_published[static_cast<size_t>(t)] = host_state->published_payload_count;
        epoch.admit_dropped[static_cast<size_t>(t)] = host_state->dropped_record_count;
    }

    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        epoch.state.store(static_cast<int>(EpochState::Admitting), std::memory_order_release);
        retained_progress_++;
    }

    if (!retained_publish_level(dump_args_level)) {
        LOG_ERROR(
            "Args dump: run %llu refused, the device did not take this run's dump level",
            static_cast<unsigned long long>(run_epoch)
        );
        retained_withdraw_unpublished_slot(slot);
        return false;
    }
    // Every shard must see this run before the device can publish into it, or
    // its first buffers would belong to no run.
    if (!request_run_reference_release(kControlAckBudgetMs)) {
        retained_set_fatal("a collector shard did not acknowledge the run table in time");
    }
    return !retained_fatal_.load(std::memory_order_acquire);
}

void ArgsDumpCollector::retained_withdraw_unpublished_slot(size_t slot) {
    // Safe because **no buffer can ever carry this run's identity**: the level
    // publication is what a producer reads, this run is refused before the
    // runner submits anything, and so nothing ever runs under this epoch. No
    // verdict is recorded — a run that never launched promised no file, and
    // recording one would make a refused admission read as a lost artifact and
    // fail every later flush. The reserved names stay on disk as the names this
    // run owned: they are empty and no manifest points at them.
    RetainedEpoch &epoch = retained_epochs_[slot];
    if (epoch.payload_opened) {
        epoch.payload_file.close();
        epoch.payload_file.clear();
        epoch.payload_opened = false;
    }
    retained_release_slot(slot);
}

bool ArgsDumpCollector::abandon_run(uint64_t run_epoch) {
    if (!retained_ready_.load(std::memory_order_acquire)) return true;
    size_t slot = 0;
    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        const int found = retained_find_slot(run_epoch);
        // No slot under this identity: nothing of this run's to withdraw, and
        // nothing of anyone else's that this may touch.
        if (found < 0) return true;
        slot = static_cast<size_t>(found);
        if (retained_epochs_[slot].state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) {
            return true;
        }
        // A target makes the epoch the writer's; only an unclosed one is still
        // this thread's to withdraw.
        if (retained_epochs_[slot].target_installed) return false;
    }
    if (retained_fatal_.load(std::memory_order_acquire)) {
        // A fatal collector cannot prove anything released, and its storage is
        // already the reader-join teardown's to free.
        retained_release_deferred_.store(true, std::memory_order_release);
        return false;
    }
    RetainedEpoch &epoch = retained_epochs_[slot];
    epoch.state.store(static_cast<int>(EpochState::Closing), std::memory_order_release);
    bool released = false;
    try {
        released = request_run_reference_release(kControlAckBudgetMs);
    } catch (...) {
        // An acknowledgement that could not even be asked for is not a proof of
        // release, so it settles the way a timeout does.
        released = false;
    }
    if (!released) {
        epoch.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
        retained_finish_epoch(
            slot, Verdict::Quarantined, "a collector shard still holds an unlaunched run's reference"
        );
        return false;
    }
    // This run submitted nothing, so it promised no file and owes no verdict.
    if (epoch.payload_opened) {
        epoch.payload_file.close();
        epoch.payload_file.clear();
        epoch.payload_opened = false;
    }
    retained_credit_epoch(slot);
    retained_release_slot(slot);
    try {
        LOG_WARN(
            "Args dump: run %llu was admitted and never launched; its slot is released without a manifest",
            static_cast<unsigned long long>(run_epoch)
        );
    } catch (...) {}
    return true;
}

// ---------------------------------------------------------------------------
// Receive path
// ---------------------------------------------------------------------------

bool ArgsDumpCollector::retained_bucket_reserve_one(RetainedEpoch &epoch, size_t bucket) {
    if (bucket >= epoch.buckets.size()) return false;
    std::vector<DumpedArg> &records = epoch.buckets[bucket];
    if (records.size() < records.capacity()) return true;

    const size_t old_cap = records.capacity();
    const size_t new_cap = old_cap == 0 ? kBucketInitialRecords : old_cap * 2;
    size_t old_bytes = 0;
    size_t new_bytes = 0;
    if (!simpler::dfx::runs::checked_bytes(old_cap, sizeof(DumpedArg), &old_bytes)) return false;
    if (!simpler::dfx::runs::checked_bytes(new_cap, sizeof(DumpedArg), &new_bytes)) return false;
    // The old and the new block are both alive while the elements are moved, so
    // the transient peak is what is charged: the invariant is that
    // `bucket_charged[bucket]` equals this bucket's capacity in bytes, and
    // charging the new block on top of it is exactly that peak.
    if (!retained_budget_.charge(new_bytes)) return false;
    try {
        records.reserve(new_cap);
    } catch (const std::bad_alloc &) {
        retained_budget_.credit(new_bytes);
        return false;
    }
    retained_budget_.credit(old_bytes);
    epoch.bucket_charged[bucket] = new_bytes;
    return true;
}

bool ArgsDumpCollector::retained_append_record(
    RetainedEpoch &epoch, size_t bucket, DumpedArg &&arg, uint32_t lane, bool has_payload, bool transport_published
) {
    if (bucket >= epoch.buckets.size()) return false;
    DumpedArg record = std::move(arg);
    if (!has_payload) {
        record.bin_offset = 0;
        record.bytes.clear();
        // The slot was reserved before anything else this record needed, so
        // this append cannot allocate and cannot fail.
        epoch.buckets[bucket].push_back(std::move(record));
        return true;
    }

    const uint64_t payload_size = record.payload_size;
    const int slot = static_cast<int>(&epoch - retained_epochs_.data());
    PayloadWriteRequest item{lane, slot, epoch.epoch.load(std::memory_order_acquire), std::move(record.bytes)};
    bool queued = false;
    {
        // One commit: the node is taken first and the offset is allocated only
        // once it is, and both happen inside the one critical section that also
        // orders the writer's appends. So a failed enqueue can never leave a
        // published record naming bytes the file does not hold, and a
        // successful one is in the same order as the offsets it was given.
        std::scoped_lock<std::mutex> lock(write_mutex_);
        try {
            write_queue_.push(std::move(item));
            queued = true;
        } catch (const std::bad_alloc &) {
            queued = false;
        }
        if (queued) {
            record.bin_offset = epoch.next_bin_offset;
            epoch.next_bin_offset += payload_size;
            epoch.queued_payloads.fetch_add(1, std::memory_order_relaxed);
            // Host ownership, committed with the offset: from here the arena
            // bytes this payload came from are free whatever the disk does,
            // which is what `publish_arena_acks` releases the producer on.
            //
            // Only for a payload the device actually published. A recovered
            // one was never counted in `published_payload_count`, so crediting
            // it would advance the lifetime equation past the device's own
            // count and let a later run's unread payload be acknowledged.
            if (transport_published && lane < received_payload_counts_.size()) {
                received_payload_counts_[lane].fetch_add(1, std::memory_order_release);
            }
        }
    }
    if (!queued) {
        // Reinitialize rather than inspect: whether the failed `push` left the
        // bytes in `item` — which is what it does when the node allocation is
        // what threw — or took them into an element it then destroyed, this
        // releases whatever is still held and allocates nothing.
        item = PayloadWriteRequest{};
        retained_budget_.credit(static_cast<size_t>(payload_size) + kRetainedQueueNodeBytes);
        record.bin_offset = 0;
        record.payload_size = 0;
        record.host_discarded = true;
        record.bytes.clear();
        epoch.discarded_args.fetch_add(1, std::memory_order_relaxed);
        // Same rule as the receipt above: a recovered payload's loss is this
        // run's output accounting, never a transport acknowledgement.
        if (transport_published && lane < discarded_payload_counts_.size()) {
            discarded_payload_counts_[lane].fetch_add(1, std::memory_order_release);
        }
        epoch.buckets[bucket].push_back(std::move(record));
        return false;
    }
    record.bytes.clear();
    epoch.buckets[bucket].push_back(std::move(record));
    write_cv_.notify_one();
    return true;
}

void ArgsDumpCollector::deliver_buffer_for_test(const DumpReadyBufferInfo &info, int collector_shard) {
    process_dump_buffer(info, collector_shard);
}

// ---------------------------------------------------------------------------
// Close: terminal state, the cut, and the leftover buffers
// ---------------------------------------------------------------------------

bool ArgsDumpCollector::retained_read_terminal_state(RetainedEpoch &epoch) {
    if (shm_host_ == nullptr || dump_shared_mem_dev_ == nullptr) return false;
    bool ok = true;
    uint64_t published_delta = 0;
    uint64_t dropped_delta = 0;
    for (int t = 0; t < num_dump_threads_ && static_cast<size_t>(t) < epoch.admit_published.size(); t++) {
        DumpBufferState *host_state = get_dump_buffer_state(shm_host_, t);
        DumpBufferState *device_state = get_dump_buffer_state(dump_shared_mem_dev_, t);
        constexpr size_t kCounterSpan = 2 * sizeof(uint64_t) + sizeof(uint32_t);
        // Narrow reads into the host shadow's own fields, never the whole
        // region: a bulk copy would overwrite the free-queue cursors this
        // collector's drain and replenish threads are writing.
        if (profiling_copy_from_device(
                &host_state->published_payload_count, &device_state->published_payload_count, kCounterSpan
            ) != 0) {
            ok = false;
            continue;
        }
        static_assert(
            offsetof(DumpBufferState, current_buf_seq) == offsetof(DumpBufferState, current_buf_ptr) + sizeof(uint64_t),
            "the leftover pointer and its sequence must stay contiguous for this single narrow read"
        );
        constexpr size_t kLeftoverSpan = sizeof(uint64_t) + sizeof(uint32_t);
        if (profiling_copy_from_device(&host_state->current_buf_ptr, &device_state->current_buf_ptr, kLeftoverSpan) !=
            0) {
            ok = false;
            continue;
        }
        published_delta += host_state->published_payload_count - epoch.admit_published[static_cast<size_t>(t)];
        // One wrap of the device's 32-bit counter is absorbed; more than that is
        // pre-existing behaviour of a per-run counter this path does not change.
        dropped_delta +=
            static_cast<uint32_t>(host_state->dropped_record_count - epoch.admit_dropped[static_cast<size_t>(t)]);
    }
    epoch.records.published_payloads = published_delta;
    epoch.records.dropped_records = dropped_delta;
    epoch.records.terminal_readable = ok;
    return ok;
}

int ArgsDumpCollector::retained_recover_leftover_records(
    RetainedEpoch &epoch, int lane, uint64_t dev_ptr, uint32_t expected_seq
) {
    void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(dev_ptr));
    if (host_ptr == nullptr) {
        // An address this collector does not map is not an identity it can
        // check, so nothing is read.
        return -1;
    }
    if (profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(dev_ptr), sizeof(DumpMetaBuffer)) != 0) {
        return -1;
    }
    rmb();
    DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(host_ptr);
    // The buffer's own stamp must agree with the run being closed and with the
    // sequence the terminal state named. A disagreement means the address has
    // been recycled into another incarnation, or the mapping is not what it
    // was, and either way the contents are not this run's to read.
    if (buf->run_epoch != epoch.epoch.load(std::memory_order_acquire) || buf->local_seq != expected_seq) {
        return -1;
    }
    const uint32_t count = buf->count;
    if (count == 0) return 0;
    if (count > PLATFORM_DUMP_RECORDS_PER_BUFFER) return -1;

    // The whole receive path, on this thread, into the bucket the close
    // boundary owns: the metadata **and** the payload bytes those records name
    // become host-owned here, before the execution claim is released and
    // therefore before the successor can reuse either the buffer or the arena.
    DumpReadyBufferInfo info;
    info.thread_index = static_cast<uint32_t>(lane);
    info.dev_buffer_ptr = reinterpret_cast<void *>(dev_ptr);
    info.host_buffer_ptr = host_ptr;
    info.buffer_seq = expected_seq;
    const size_t recovery_bucket = epoch.buckets.empty() ? 0 : epoch.buckets.size() - 1;
    const uint64_t before = epoch.collected_records.load(std::memory_order_relaxed);
    process_dump_buffer(info, -1, &epoch, recovery_bucket);
    const uint64_t appended = epoch.collected_records.load(std::memory_order_relaxed) - before;
    return static_cast<int>(appended);
}

void ArgsDumpCollector::retained_decide_leftovers(RetainedEpoch &epoch) {
    for (int t = 0; t < num_dump_threads_ && static_cast<size_t>(t) < epoch.receipts.size(); t++) {
        DumpBufferState *host_state = get_dump_buffer_state(shm_host_, t);
        const uint64_t dev_ptr = host_state->current_buf_ptr;
        if (dev_ptr == 0) continue;

        const simpler::dfx::args_dump::LaneReceipt &receipt = epoch.receipts[static_cast<size_t>(t)];
        LeftoverOutcome outcome = LeftoverOutcome::Unknown;
        if (receipt.identity_uncertain) {
            // A delivery on this lane did not carry the successor sequence, its
            // two witnesses disagreed, or the accumulator left the device
            // field's range. Nothing is read.
            outcome = LeftoverOutcome::Unknown;
        } else {
            const uint64_t terminal_seq = static_cast<uint64_t>(host_state->current_buf_seq);
            if (terminal_seq < receipt.next_expected_seq) {
                // The sequence the device still names was already handed over:
                // its records reached the host through the ordinary path, and
                // the buffer behind this pointer may already have been
                // recycled. This is the publish-then-clear window, and the
                // answer is to read nothing and lose nothing.
                outcome = LeftoverOutcome::AlreadyDelivered;
            } else if (terminal_seq == receipt.next_expected_seq) {
                const int recovered =
                    retained_recover_leftover_records(epoch, t, dev_ptr, static_cast<uint32_t>(terminal_seq));
                if (recovered < 0) {
                    outcome = LeftoverOutcome::Unknown;
                } else if (recovered == 0) {
                    outcome = LeftoverOutcome::None;
                } else {
                    epoch.records.recovered_records += static_cast<uint64_t>(recovered);
                    outcome = LeftoverOutcome::Recovered;
                }
            } else {
                // Ahead of the ledger: a publication this run made is missing
                // from it, so completeness cannot be asserted and nothing is
                // read.
                outcome = LeftoverOutcome::Unknown;
            }
        }
        switch (outcome) {
        case LeftoverOutcome::Unknown:
            epoch.records.unknown_lanes++;
            LOG_WARN(
                "Args dump: run %llu lane %d leftover buffer identity is unknown; its contents were not read",
                static_cast<unsigned long long>(epoch.epoch.load(std::memory_order_acquire)), t
            );
            break;
        case LeftoverOutcome::AlreadyDelivered:
            epoch.records.already_delivered_lanes++;
            break;
        case LeftoverOutcome::Recovered:
            LOG_WARN(
                "Args dump: run %llu lane %d had an unpublished buffer; its records and payload were recovered "
                "under this run's execution claim",
                static_cast<unsigned long long>(epoch.epoch.load(std::memory_order_acquire)), t
            );
            break;
        case LeftoverOutcome::None:
            break;
        }
    }
}

bool ArgsDumpCollector::retained_wait_for_processing(RetainedEpoch &epoch, uint64_t run_epoch) {
    // A capture acknowledgement says only that every drain owner passed a
    // boundary after the request. It does **not** say the buffers it captured
    // have been processed: `ring_processed_` is advanced after
    // `on_buffer_collected` returns, and stage 2 is the comparison against it.
    //
    // Both halves of this run's ownership hang on that distinction. A buffer
    // published but not yet processed has not updated this lane's receipt
    // ledger, so the close would read a stale `next_expected_seq` and could
    // recover a buffer that was in fact handed over — while a shard is inside
    // that same mapping. Its payload is also still in the arena, so releasing
    // the execution claim would let the successor overwrite bytes no one has
    // copied. So the wait belongs here, inside the claim, and its failure is
    // not something a later file annotation can repair.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kStage2WaitMs);
    bool acked = false;
    while (true) {
        if (cut_counters_exhausted()) break;
        if (!acked) acked = cut_acked(epoch.cut_request);
        if (acked) {
            int failed = 0;
            if (cut_failed_queues(epoch.cut_slot, epoch.cut_request, &failed) && failed == 0 &&
                cut_stage2_done(epoch.cut_slot)) {
                return true;
            }
            // A queue whose capture failed can never reach its target, so this
            // run's proof is unobtainable rather than late.
            if (failed != 0) break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            LOG_ERROR(
                "Args dump: run %llu could not prove its published buffers were processed within %d ms",
                static_cast<unsigned long long>(run_epoch), kStage2WaitMs
            );
            break;
        }
        // The drain owners and collector shards are the ones making progress
        // here; this thread only waits for them, so a short sleep on a teardown
        // path is the right primitive (codestyle.md rule 5 exempts it).
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

int ArgsDumpCollector::run_close(uint64_t run_epoch, bool device_execution_complete) {
    if (!retained_ready_.load(std::memory_order_acquire)) return 0;
    const int found = retained_find_slot(run_epoch);
    if (found < 0) return 0;
    const size_t slot = static_cast<size_t>(found);
    RetainedEpoch &epoch = retained_epochs_[slot];
    epoch.records.device_execution_complete = device_execution_complete;
    int rc = 0;

    if (device_execution_complete) {
        // The successor has not launched, so this run's terminal lane state is
        // still its own and nothing can recycle a buffer under the decision
        // below.
        (void)retained_read_terminal_state(epoch);

        // Every buffer this run will ever publish is already in a device ready
        // queue at this instant, so each queue's target is finite and a
        // successor's later traffic cannot discharge it.
        uint64_t request = 0;
        epoch.cut_slot = cut_arm(&request);
        epoch.cut_request = request;
        bool processed = false;
        if (epoch.cut_slot < 0) {
            LOG_ERROR(
                "Args dump: run %llu got no cut slot, so its transport cannot be proved",
                static_cast<unsigned long long>(run_epoch)
            );
        } else if (!cut_wait_for_ack(request, kCutAckBudgetMs)) {
            LOG_WARN(
                "Args dump: run %llu cut capture did not complete in %d ms", static_cast<unsigned long long>(run_epoch),
                kCutAckBudgetMs
            );
            // Not a verdict yet: the wait below re-reads the acknowledgement,
            // so a capture that lands a moment later still counts.
            processed = retained_wait_for_processing(epoch, run_epoch);
        } else {
            processed = retained_wait_for_processing(epoch, run_epoch);
        }

        int failed = 0;
        const bool cut_known = epoch.cut_slot >= 0 && cut_failed_queues(epoch.cut_slot, epoch.cut_request, &failed);
        epoch.cut.counters_exhausted = cut_counters_exhausted();
        epoch.cut.cut_known = cut_known;
        epoch.cut.failed_queues = failed;
        epoch.cut.stage2_done = processed;
        epoch.cut_settled_at_close = true;

        // Only with this run's buffers proved processed is the receipt ledger
        // stable and the arena proved free of anything unread; only then may a
        // leftover be decided at all.
        if (processed && epoch.records.terminal_readable) {
            retained_decide_leftovers(epoch);
        } else {
            // Nothing unproved is read. Every lane still naming a buffer is
            // undecidable, and the payloads that were not processed were never
            // counted as received or discarded, so `publish_arena_acks` will
            // not acknowledge them and the producer's own barrier keeps the
            // arena from being reused under them.
            for (int t = 0; t < num_dump_threads_ && static_cast<size_t>(t) < epoch.receipts.size(); t++) {
                DumpBufferState *host_state = get_dump_buffer_state(shm_host_, t);
                if (host_state->current_buf_ptr != 0) epoch.records.unknown_lanes++;
            }
            if (!processed) epoch.records.processing_proved = false;
            LOG_ERROR(
                "Args dump: run %llu closed without proof that its published buffers were processed; "
                "no leftover buffer is read and no unprocessed payload is acknowledged",
                static_cast<unsigned long long>(run_epoch)
            );
            rc = PTO_RUNTIME_ERR_INTERNAL;
        }
    } else {
        // No fence was observed, so no device-side producer is proved stopped:
        // nothing here reads the arena or a metadata buffer, and no recovery is
        // attempted. What this run already owns on the host is published as an
        // incomplete result and its flush fails. `quiesce()` is deliberately
        // not called — its precondition is the stop it would be used to prove.
        LOG_WARN(
            "Args dump: run %llu closed without an observed device fence; its dump is incomplete and no "
            "leftover buffer is read",
            static_cast<unsigned long long>(run_epoch)
        );
        // The caller already owns the failure that cleared the fence, and this
        // path must not displace it, so the incompleteness travels in the
        // sticky record and the manifest rather than in this return.
        epoch.cut_settled_at_close = true;
    }

    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        epoch.target_installed = true;
        epoch.closed_at = std::chrono::steady_clock::now();
        const uint64_t watermark = retained_close_watermark_.load(std::memory_order_acquire);
        if (watermark != UINT64_MAX && run_epoch > watermark) {
            retained_close_watermark_.store(run_epoch, std::memory_order_release);
        }
    }
    retained_bump_progress();
    retained_cv_.notify_all();
    return rc;
}

// ---------------------------------------------------------------------------
// The background writer
// ---------------------------------------------------------------------------

bool ArgsDumpCollector::retained_drain_payload_batch(int max_requests) {
    bool drained_any = false;
    for (int taken = 0; taken < max_requests; taken++) {
        PayloadWriteRequest request;
        {
            std::scoped_lock<std::mutex> lock(write_mutex_);
            if (write_queue_.empty()) break;
            request = std::move(write_queue_.front());
            write_queue_.pop();
        }
        drained_any = true;
        const size_t payload_bytes = request.bytes.size();
        if (request.epoch_slot >= 0 && static_cast<size_t>(request.epoch_slot) < retained_epochs_.size()) {
            RetainedEpoch &epoch = retained_epochs_[static_cast<size_t>(request.epoch_slot)];
            if (epoch.epoch.load(std::memory_order_acquire) != request.epoch) {
                // The slot has moved on, so this request's own run was sealed
                // without draining it. Recorded rather than appended: these
                // bytes must never land in a successor's file, and the count
                // it was holding went back with that run's slot.
                retained_errors_.record_unknown_epoch(request.epoch, 1);
            } else if (epoch.queued_payloads.fetch_sub(1, std::memory_order_acq_rel) == 0) {
                // Unreachable by construction; the guard keeps a miscount from
                // wrapping into an unbounded seal wait.
                epoch.queued_payloads.store(0, std::memory_order_relaxed);
                retained_errors_.record_unknown_epoch(request.epoch, 1);
            } else if (payload_bytes > 0) {
                epoch.payload_file.write(
                    reinterpret_cast<const char *>(request.bytes.data()), static_cast<std::streamsize>(payload_bytes)
                );
                if (!epoch.payload_file.good()) {
                    LOG_ERROR("Args dump: failed to append to %s", epoch.token.payload_path.c_str());
                    epoch.io_failed.store(true, std::memory_order_relaxed);
                } else {
                    epoch.payload_bytes_written.fetch_add(payload_bytes, std::memory_order_relaxed);
                    // Only a write the stream accepted credits a lane: an
                    // unwritten payload must never be acknowledged.
                    written_payload_counts_[request.thread_index].fetch_add(1, std::memory_order_release);
                }
            }
        }
        // The bytes and their node were charged together and are released
        // together, whatever the write reported.
        std::vector<uint8_t>().swap(request.bytes);
        retained_budget_.credit(payload_bytes + kRetainedQueueNodeBytes);
        bytes_written_ += payload_bytes;
    }
    return drained_any;
}

void ArgsDumpCollector::retained_writer_loop() {
    while (retained_writer_running_.load(std::memory_order_acquire)) {
        uint64_t seen = 0;
        {
            std::scoped_lock<std::mutex> lock(write_mutex_);
            seen = retained_progress_;
        }
        bool drained = false;
        // The test seam, read once per pass: held stops this thread doing
        // anything — no payload reaches a file and nothing is sealed — while the
        // shards keep routing, keep acknowledging the run table, and keep taking
        // payloads into host-owned storage. That is what lets a case stand at
        // the instant where the bytes are owned and the disk has not moved.
        const bool held = retained_writer_held_.load(std::memory_order_acquire);
        try {
            if (!held) {
                // Payload first and bounded, then the seals: a continuously
                // producing successor cannot starve a closed run's publication,
                // and a closed run's seal cannot stall payload acknowledgement.
                drained = retained_drain_payload_batch(kWriterPayloadBatch);
                retained_service();
            }
        } catch (const std::exception &e) {
            retained_set_fatal(e.what());
        } catch (...) {
            retained_set_fatal("the retained-run writer failed");
        }
        std::unique_lock<std::mutex> lock(write_mutex_);
        if (!retained_writer_running_.load(std::memory_order_acquire)) break;
        if (!held && (drained || !write_queue_.empty())) continue;
        if (retained_progress_ != seen) continue;
        // Nothing here needs a clock: a closed run carries the cut proof its
        // own close took under the execution claim, so every wakeup this thread
        // waits for is an event — an enqueue, a close, a release of the seam or
        // a stop — and each of them bumps `retained_progress_`.
        write_cv_.wait(lock);
    }
}

void ArgsDumpCollector::retained_service() {
    for (size_t slot = 0; slot < retained_epochs_.size(); slot++) {
        RetainedEpoch &epoch = retained_epochs_[slot];
        if (epoch.state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) continue;
        bool closed = false;
        {
            std::scoped_lock<std::mutex> lock(retained_mu_);
            closed = epoch.target_installed;
        }
        if (!closed) continue;
        // The proof is the one the close took under the execution claim, and it
        // is not re-read here. Re-evaluating the same cut could observe a
        // stage 2 that landed after the claim was released and seal a run whose
        // leftover was deliberately left unread as though it were complete.
        if (!epoch.cut_settled_at_close) continue;
        retained_seal(slot);
    }
}

void ArgsDumpCollector::retained_seal(size_t slot) {
    RetainedEpoch &epoch = retained_epochs_[slot];

    // Withdraw admission, then prove no collector shard still holds a
    // reference. Until that is proved nothing may be sorted, merged or freed: a
    // shard could still be appending to a bucket.
    epoch.state.store(static_cast<int>(EpochState::Closing), std::memory_order_release);
    bool released = false;
    try {
        released = request_run_reference_release(kControlAckBudgetMs);
    } catch (...) {
        released = false;
    }
    if (!released) {
        epoch.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
        retained_finish_epoch(slot, Verdict::Quarantined, "a collector shard did not release this run's reference");
        return;
    }
    // Retiring the cut slot is the second release and its failure is the same
    // kind: the slot is left un-reused and these records stay untouched.
    if (epoch.cut_slot >= 0 && !cut_release(epoch.cut_slot, kCutAckBudgetMs)) {
        epoch.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
        retained_finish_epoch(slot, Verdict::Quarantined, "this run's cut slot could not be retired");
        return;
    }
    epoch.cut_slot = -1;

    // Proved released: the buckets and the files are the writer's alone now.
    // Anything **this run** still has queued is drained before its payload file
    // is closed, or the manifest would name bytes the file does not hold. The
    // count is this run's own, so a successor producing into the shared queue
    // cannot extend this wait: no new request can carry a released run's
    // identity, so the count only falls.
    while (epoch.queued_payloads.load(std::memory_order_acquire) != 0) {
        if (!retained_drain_payload_batch(kWriterPayloadBatch)) {
            // The queue is empty while this run still counts a request: its
            // bytes are unaccounted for, so the run is short rather than
            // silently complete.
            LOG_ERROR(
                "Args dump: run %llu still counted %llu queued payload(s) with an empty queue",
                static_cast<unsigned long long>(epoch.epoch.load(std::memory_order_acquire)),
                static_cast<unsigned long long>(epoch.queued_payloads.load(std::memory_order_acquire))
            );
            epoch.discarded_args.fetch_add(
                epoch.queued_payloads.exchange(0, std::memory_order_acq_rel), std::memory_order_relaxed
            );
            break;
        }
    }
    if (epoch.payload_opened) {
        epoch.payload_file.flush();
        if (!epoch.payload_file.good()) epoch.io_failed.store(true, std::memory_order_relaxed);
        epoch.payload_file.close();
        // A short write may only surface at close, so the state is checked
        // after it and not only after each append.
        if (epoch.payload_file.fail()) epoch.io_failed.store(true, std::memory_order_relaxed);
        epoch.payload_file.clear();
        epoch.payload_opened = false;
    }

    simpler::dfx::args_dump::HostLoss loss;
    loss.discarded_args = epoch.discarded_args.load(std::memory_order_relaxed);
    loss.discarded_metadata_records = epoch.discarded_metadata_records.load(std::memory_order_relaxed);
    loss.io_failed = epoch.io_failed.load(std::memory_order_relaxed);
    const uint64_t collected = epoch.collected_records.load(std::memory_order_relaxed);
    Verdict verdict = classify(epoch.cut, epoch.records, loss, collected);

    char detail[kErrorMsgBytes];
    std::snprintf(
        detail, sizeof(detail),
        "collected=%llu published_payloads=%llu dropped=%llu recovered=%llu discarded=%llu meta_discarded=%llu "
        "unknown_lanes=%llu cut_known=%d failed=%d stage2=%d processed=%d device_complete=%d terminal=%d",
        static_cast<unsigned long long>(collected), static_cast<unsigned long long>(epoch.records.published_payloads),
        static_cast<unsigned long long>(epoch.records.dropped_records),
        static_cast<unsigned long long>(epoch.records.recovered_records),
        static_cast<unsigned long long>(loss.discarded_args),
        static_cast<unsigned long long>(loss.discarded_metadata_records),
        static_cast<unsigned long long>(epoch.records.unknown_lanes), epoch.cut.cut_known ? 1 : 0,
        epoch.cut.failed_queues, epoch.cut.stage2_done ? 1 : 0, epoch.records.processing_proved ? 1 : 0,
        epoch.records.device_execution_complete ? 1 : 0, epoch.records.terminal_readable ? 1 : 0
    );

    if (simpler::dfx::args_dump::verdict_publishes(verdict)) {
        if (!retained_publish_manifest(epoch, verdict)) {
            // The temporary manifest and the payload file stay as the evidence,
            // and the next run under this destination is refused until they are
            // read and removed.
            verdict = Verdict::WriteFailed;
        }
    }
    retained_finish_epoch(slot, verdict, detail);
}

bool ArgsDumpCollector::retained_publish_manifest(RetainedEpoch &epoch, Verdict verdict) {
    // Each bucket is sorted in place and the manifest is streamed from a k-way
    // merge over them, so no second copy of this run's metadata is ever built
    // and the seal has nothing to reserve.
    for (auto &bucket : epoch.buckets) {
        std::sort(bucket.begin(), bucket.end(), arg_order_before);
    }

    simpler::dfx::args_dump::ManifestMeta meta;
    meta.run_dir_name = epoch.run_dir.filename().string();
    meta.dump_args_level = static_cast<uint32_t>(epoch.level);
    meta.truncated_args = epoch.truncated_records.load(std::memory_order_relaxed);
    meta.dropped_records = epoch.records.dropped_records;
    meta.retained = true;
    meta.host_discarded_args = epoch.discarded_args.load(std::memory_order_relaxed);
    meta.metadata_discarded_records = epoch.discarded_metadata_records.load(std::memory_order_relaxed);
    meta.counts_unknown = !simpler::dfx::args_dump::verdict_succeeds(verdict);
    meta.verdict = simpler::dfx::args_dump::verdict_name(verdict);
    for (const auto &bucket : epoch.buckets) {
        for (const DumpedArg &dt : bucket) {
            meta.total_args++;
            if (dt.stage == ArgsDumpStage::BEFORE_DISPATCH) {
                meta.before_dispatch++;
            } else {
                meta.after_completion++;
            }
            switch (dt.role) {
            case ArgsDumpRole::INPUT:
                meta.input_args++;
                break;
            case ArgsDumpRole::OUTPUT:
                meta.output_args++;
                break;
            case ArgsDumpRole::INOUT:
                meta.inout_args++;
                break;
            }
        }
    }
    // The manifest names the payload file this run exclusively owns, through
    // the same `bin_file` indirection the readers already take the name from. A
    // run that wrote no payload byte names none, exactly as the single-run path
    // leaves none.
    if (epoch.payload_bytes_written.load(std::memory_order_relaxed) > 0) {
        meta.bin_file = epoch.token.payload_name;
    }

    {
        std::ofstream json(epoch.token.manifest_tmp, std::ios::out | std::ios::trunc);
        if (!json.is_open()) {
            LOG_ERROR("Args dump: could not open %s", epoch.token.manifest_tmp.c_str());
            return false;
        }
        simpler::dfx::args_dump::write_manifest_prologue(json, meta);
        // A fixed cursor array over the buckets: at most one per collector
        // shard plus the close boundary's own, and nothing is allocated per
        // record.
        std::vector<size_t> cursors(epoch.buckets.size(), 0);
        bool first_entry = true;
        while (true) {
            size_t pick = epoch.buckets.size();
            for (size_t b = 0; b < epoch.buckets.size(); b++) {
                if (cursors[b] >= epoch.buckets[b].size()) continue;
                if (pick == epoch.buckets.size() ||
                    arg_order_before(epoch.buckets[b][cursors[b]], epoch.buckets[pick][cursors[pick]])) {
                    pick = b;
                }
            }
            if (pick == epoch.buckets.size()) break;
            if (!first_entry) json << ",\n";
            first_entry = false;
            simpler::dfx::args_dump::write_arg_json(json, epoch.buckets[pick][cursors[pick]]);
            cursors[pick]++;
        }
        simpler::dfx::args_dump::write_manifest_epilogue(json);
        json.flush();
        if (!json.good()) {
            LOG_ERROR("Args dump: failed to write %s", epoch.token.manifest_tmp.c_str());
            return false;
        }
        json.close();
        if (json.fail()) {
            LOG_ERROR("Args dump: failed to close %s", epoch.token.manifest_tmp.c_str());
            return false;
        }
    }

    // One atomic directory operation switches the manifest and, because the
    // manifest carries `bin_file`, the payload file it points at. Before it the
    // published pair is untouched; after it the pair is this run's.
    std::error_code ec;
    std::filesystem::rename(epoch.token.manifest_tmp, epoch.token.manifest_path, ec);
    if (ec) {
        LOG_ERROR("Args dump: could not publish %s: %s", epoch.token.manifest_path.c_str(), ec.message().c_str());
        return false;
    }
    LOG_INFO(
        "Args dump: run %llu published %s (%llu args, %s)",
        static_cast<unsigned long long>(epoch.epoch.load(std::memory_order_acquire)), epoch.token.manifest_path.c_str(),
        static_cast<unsigned long long>(meta.total_args), meta.bin_file.empty() ? "no payload" : meta.bin_file.c_str()
    );
    return true;
}

void ArgsDumpCollector::retained_credit_epoch(size_t slot) {
    RetainedEpoch &epoch = retained_epochs_[slot];
    for (size_t b = 0; b < epoch.buckets.size(); b++) {
        // The vector's storage goes back with its charge, in that order: the
        // charge covers this bucket's capacity, so crediting before the release
        // would make the budget claim space that is still held.
        std::vector<DumpedArg>().swap(epoch.buckets[b]);
        if (b < epoch.bucket_charged.size()) {
            retained_budget_.credit(epoch.bucket_charged[b]);
            epoch.bucket_charged[b] = 0;
        }
    }
}

void ArgsDumpCollector::retained_finish_epoch(size_t slot, Verdict verdict, const char *detail) {
    RetainedEpoch &epoch = retained_epochs_[slot];
    const uint64_t run_epoch = epoch.epoch.load(std::memory_order_acquire);
    retained_errors_.record(run_epoch, verdict, detail);

    if (verdict == Verdict::Quarantined) {
        // Nothing is freed and the slot is not returned. The collector-thread
        // join is the only release, and every waiter is woken now so a capacity
        // check or a flush learns instead of blocking.
        retained_release_deferred_.store(true, std::memory_order_release);
        retained_set_fatal(detail);
        try {
            LOG_ERROR(
                "Args dump: run %llu ended %s (%s)", static_cast<unsigned long long>(run_epoch),
                simpler::dfx::args_dump::verdict_name(verdict), detail != nullptr ? detail : ""
            );
        } catch (...) {}
        return;
    }
    try {
        if (simpler::dfx::args_dump::verdict_succeeds(verdict)) {
            LOG_INFO(
                "Args dump: run %llu %s", static_cast<unsigned long long>(run_epoch),
                simpler::dfx::args_dump::verdict_name(verdict)
            );
        } else {
            LOG_ERROR(
                "Args dump: run %llu ended %s (%s)", static_cast<unsigned long long>(run_epoch),
                simpler::dfx::args_dump::verdict_name(verdict), detail != nullptr ? detail : ""
            );
        }
    } catch (...) {}
    retained_credit_epoch(slot);
    retained_release_slot(slot);
}

void ArgsDumpCollector::retained_release_slot(size_t slot) {
    RetainedEpoch &epoch = retained_epochs_[slot];
    {
        // Every field cleared here is read by an admission or a flush under
        // this lock while the slot is still non-Free — `run_dir` by the
        // destination scan, `target_installed` by the flush's pending test — so
        // the reset and the `Free` publication happen in one critical section
        // with them.
        std::scoped_lock<std::mutex> lock(retained_mu_);
        epoch.run_dir.clear();
        epoch.token = simpler::dfx::args_dump::OutputToken{};
        epoch.target_installed = false;
        epoch.next_bin_offset = 0;
        epoch.io_failed.store(false, std::memory_order_relaxed);
        epoch.state.store(static_cast<int>(EpochState::Free), std::memory_order_release);
        retained_progress_++;
    }
    retained_cv_.notify_all();
    write_cv_.notify_all();
}

// ---------------------------------------------------------------------------
// Reporting and teardown
// ---------------------------------------------------------------------------

bool ArgsDumpCollector::flush_retained_runs(int timeout_ms, std::string *error) {
    if (retained_ready_.load(std::memory_order_acquire)) {
        const uint64_t watermark = retained_close_watermark_.load(std::memory_order_acquire);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            bool pending = false;
            std::unique_lock<std::mutex> lock(retained_mu_);
            for (const auto &epoch : retained_epochs_) {
                const int state = epoch.state.load(std::memory_order_acquire);
                if (state == static_cast<int>(EpochState::Free)) continue;
                // Terminal, and reported below rather than waited for.
                if (state == static_cast<int>(EpochState::Quarantined)) continue;
                if (epoch.epoch.load(std::memory_order_acquire) <= watermark && epoch.target_installed) {
                    pending = true;
                }
            }
            if (!pending) break;
            if (retained_fatal_.load(std::memory_order_acquire)) break;
            if (retained_cv_.wait_until(lock, deadline) == std::cv_status::timeout &&
                std::chrono::steady_clock::now() >= deadline) {
                // A disk that is not draining is exactly this case: the wait is
                // bounded by the caller's budget, never by a promise that the
                // writer finishes.
                if (error != nullptr) {
                    *error = "args dump flush timed out with runs still unpublished; " + retained_errors_.report();
                }
                return false;
            }
        }
    }
    if (retained_fatal_.load(std::memory_order_acquire)) {
        if (error != nullptr) {
            std::scoped_lock<std::mutex> lock(retained_mu_);
            *error = "args dump collector is fatal: " + retained_fatal_reason_ + "; " + retained_errors_.report();
        }
        return false;
    }
    // Sticky: reported by every flush for as long as this collector lives,
    // because there is no acknowledgement that could clear it — and removing
    // the evidence files on disk does not clear it either.
    if (retained_errors_.has_error()) {
        if (error != nullptr) *error = "args dump reported failures: " + retained_errors_.report();
        return false;
    }
    return true;
}

void ArgsDumpCollector::retained_finish() {
    if (!retained_ready_.load(std::memory_order_acquire)) return;
    {
        std::scoped_lock<std::mutex> lock(retained_mu_);
        retained_close_watermark_.store(UINT64_MAX, std::memory_order_release);
    }
    retained_bump_progress();
    // Void by contract: this publishes, and the caller's own flush is what
    // reports. A failure recorded here stays in the sticky summary and is
    // returned by `finalize`.
    std::string ignored;
    (void)flush_retained_runs(kCutAckBudgetMs * 8, &ignored);
}

void ArgsDumpCollector::finish_retained_runs() { retained_finish(); }

void ArgsDumpCollector::retained_release_resources() {
    if (!retained_ready_.exchange(false, std::memory_order_acq_rel)) return;
    set_drain_quantum(0);
    set_run_counters(false);
    report_drain_drops();
    for (size_t slot = 0; slot < retained_epochs_.size(); slot++) {
        const int state = retained_epochs_[slot].state.load(std::memory_order_acquire);
        if (state == static_cast<int>(EpochState::Free)) continue;
        retained_release_deferred_.store(true, std::memory_order_release);
        const uint64_t run_epoch = retained_epochs_[slot].epoch.load(std::memory_order_acquire);
        // A slot still held here owes a manifest that will now never appear,
        // and the sticky record is the only thing that can say so. A
        // `Quarantined` epoch already has its verdict from the seal that could
        // not prove a release, and a withdrawn admission has already returned
        // its slot as `Free`: it promised no file, so recording one for it
        // would make a rolled-back launch read as lost output.
        if (state != static_cast<int>(EpochState::Quarantined)) {
            retained_errors_.record(
                run_epoch, Verdict::Abandoned, "the collector was torn down before this run was published"
            );
        }
        LOG_WARN(
            "Args dump: run %llu storage is held until the collector threads are joined",
            static_cast<unsigned long long>(run_epoch)
        );
    }
    LOG_INFO("Args dump: retained runs released: %s", retained_errors_.report().c_str());
}

void ArgsDumpCollector::retained_release_quarantined() {
    if (!retained_release_deferred_.exchange(false, std::memory_order_acq_rel)) return;
    // Only from finalize, after every drain owner and collector shard is
    // joined: storage a seal could not prove safe to touch is unreachable by
    // any reader only then. The files stay on disk as evidence.
    for (size_t slot = 0; slot < retained_epochs_.size(); slot++) {
        if (retained_epochs_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) {
            continue;
        }
        RetainedEpoch &epoch = retained_epochs_[slot];
        if (epoch.payload_opened) {
            epoch.payload_file.close();
            epoch.payload_file.clear();
            epoch.payload_opened = false;
        }
        retained_credit_epoch(slot);
        retained_release_slot(slot);
    }
    retained_budget_.close();
}

void ArgsDumpCollector::hold_retained_writer_for_test(bool held) {
    {
        std::scoped_lock<std::mutex> lock(write_mutex_);
        retained_writer_held_.store(held, std::memory_order_release);
        retained_progress_++;
    }
    write_cv_.notify_all();
}

ArgsDumpCollector::RetainedRunStats ArgsDumpCollector::retained_run_stats_for_test() const {
    RetainedRunStats s;
    s.retaining = retain_across_runs_;
    s.ready = retained_ready_.load(std::memory_order_acquire);
    s.fatal = retained_fatal_.load(std::memory_order_acquire);
    for (const auto &epoch : retained_epochs_) {
        const int state = epoch.state.load(std::memory_order_acquire);
        if (state == static_cast<int>(EpochState::Free)) continue;
        s.open_collected_records += epoch.collected_records.load(std::memory_order_relaxed);
        s.open_discarded_args += epoch.discarded_args.load(std::memory_order_relaxed);
        if (state == static_cast<int>(EpochState::Quarantined)) {
            s.quarantined_epochs++;
            continue;
        }
        s.open_epochs++;
    }
    s.charged_bytes = retained_budget_.charged();
    s.budget_refusals = retained_budget_.refusals();
    s.errors = retained_errors_.counts();
    return s;
}
